"""
This code allows you to train clustering modulation with pool mo technique.

Author: Weiduo Liao
Date: 2023.06.21
"""

import os
import sys
import pickle
import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss, DistillKL, distillation_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR, WeightAnnealing)
from models.model_helpers import get_model, get_model_moe, get_optimizer
from utils import Accumulator, device, set_determ, check_dir
from config import args, BATCHSIZES

from pmo_utils import (Pool, Mixer,
                       cal_hv_loss, cal_hv, draw_objs, draw_heatmap, available_setting, check_available, task_to_device)
from debug import Debugger

import warnings
warnings.filterwarnings('ignore')


def train():
    # Set seed
    set_determ(seed=1234)

    debugger = Debugger(activate=True)

    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        '''--------------------'''
        '''Initialization Phase'''
        '''--------------------'''
        # initialize datasets and loaders
        trainsets = TRAIN_METADATASET_NAMES
        valsets = TRAIN_METADATASET_NAMES
        testsets = TRAIN_METADATASET_NAMES
        trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
        print(f'Train on: {trainsets}.')    # "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower"
        print(f'Val on: {valsets}.')
        # print(f'Test on: {testsets}.')

        print(f'devices: {device}.')
        assert (device != 'cpu'), f'device is cpu'

        train_loaders = []
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders.append(MetaDatasetBatchReader(
                'train', [trainset], valsets, testsets,
                batch_size=BATCHSIZES[trainset]))
            num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}
        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['test.type'])

        '''initialize models and optimizer'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, -1

        # pmo model, fe load from url
        args['model.num_clusters'] = 8
        pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe

        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_film_parameters())    # for films
        optimizer_selector = torch.optim.Adam(pmo.get_trainable_selector_parameters(True),
                                              lr=args['train.selector_learning_rate'],
                                              weight_decay=args['train.selector_learning_rate'] / 50
                                              )
        checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                checkpointer.restore_model(ckpt='last', strict=False)       # since only store film and selector
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
            lr_manager_selector = UniformStepLR(optimizer_selector, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
            lr_manager_selector = ExpDecayLR(optimizer_selector, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, 0)       # start_iter
            lr_manager_selector = CosineAnnealRestartLR(optimizer_selector, args, 0)       # start_iter

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']

        def init_train_log():
            epoch_loss = {}
            epoch_loss[f'task/gumbel_sim'] = []
            epoch_loss[f'task/softmax_sim'] = []
            # epoch_loss[f'task/selection_ce_loss'] = []
            epoch_loss[f'pool/selection_ce_loss'] = []
            epoch_loss[f'pure/selection_ce_loss'] = []
            epoch_loss.update({f'task/{name}': [] for name in trainsets})
            epoch_loss['kd'] = []
            # epoch_loss['task/rec'] = []
            # if 'hv' in args['train.loss_type']:
            epoch_loss['hv/loss'], epoch_loss['hv'] = [], []
            epoch_loss.update({
                f'hv/obj{obj_idx}': {
                    f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_loss.update({
                f'pure/C{cluster_idx}': [] for cluster_idx in range(args['model.num_clusters'])})

            epoch_acc = {}
            epoch_acc['task/avg'] = []     # average over all trainsets
            epoch_acc.update({f'task/{name}': [] for name in trainsets})
            # if 'hv' in args['train.loss_type']:
            epoch_acc['hv'] = []
            epoch_acc.update({
                f'hv/obj{obj_idx}': {
                    f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_acc.update({
                f'pure/C{cluster_idx}': [] for cluster_idx in range(args['model.num_clusters'])})

            return epoch_loss, epoch_acc

        def model_train():
            # train mode
            pmo.train()
            if pmo.feature_extractor is not None:
                pmo.feature_extractor.eval()        # to extract task features

        def model_eval():
            # eval mode
            pmo.eval()

        def zero_grad():
            optimizer.zero_grad()
            optimizer_selector.zero_grad()

        def lr_manager_step(idx):
            lr_manager.step(idx)
            lr_manager_selector.step(idx)

        epoch_loss, epoch_acc = init_train_log()
        epoch_val_loss = {}
        epoch_val_acc = {}

        print(f'\n>>>> Train start from {start_iter}.')
        verbose = True
        for i in tqdm(range(start_iter, max_iter), ncols=100):

            zero_grad()
            model_train()

            image_features = []
            cluster_labels = []
            # loading images and labels
            for t_indx, (name, train_loader) in enumerate(zip(trainsets, train_loaders)):
                sample = train_loader.get_train_batch(session)
                images = sample['images'].to(device)
                # labels = sample['labels'].long().to(device)

                features = pmo.embed(images)        # embed using url

                image_features.append(features)
                cluster_labels.append([t_indx] * images.size(0))

            image_features = torch.cat(image_features)
            cluster_labels = torch.from_numpy(np.concatenate(cluster_labels)).long().to(device)

            '''----------------'''
            ''' Train Phase'''
            '''----------------'''
            _, selection_info = pmo.selector(image_features, gumbel=False, hard=False, average=False)

            '''log img sim (softmax and gumbel)'''
            epoch_loss[f'task/gumbel_sim'].append(selection_info['y_soft'].detach().cpu().numpy())    # [bs,8]
            epoch_loss[f'task/softmax_sim'].append(selection_info['normal_soft'].detach().cpu().numpy())

            '''selection ce loss on image_features'''
            print(f"\n>> Iter: {i}, clustering loss calculation: ")

            # fn = torch.nn.CrossEntropyLoss()
            # dist = selection_info['dist']  # [img_size, 8]
            # selection_ce_loss = fn(dist, cluster_labels)
            '''allow gumbel loss'''
            soft = selection_info['y_soft']     # can be gumbel soft
            log_p_y = torch.log(soft)
            preds = log_p_y.argmax(1)
            labels = cluster_labels.type(torch.long)
            loss = F.nll_loss(log_p_y, labels, reduction='mean')
            # acc = torch.eq(preds, labels).float().mean()
            # stats_dict = {'loss': loss.item(), 'acc': acc.item()}
            # pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}

            selection_ce_loss = loss
            '''log ce loss'''
            epoch_loss[f'pool/selection_ce_loss'].append(selection_ce_loss.item())

            zero_grad()
            '''ce loss coefficient'''
            selection_ce_loss = selection_ce_loss * args['train.ce_coefficient']
            selection_ce_loss.backward()

            '''debug'''
            debugger.print_grad(pmo, key='selector', prefix=f'iter{i} after selection_ce_loss backward:\n')

            optimizer_selector.step()

            if args['train.cluster_center_mode'] == 'mov_avg':
                print(f"\n>> Iter: {i}, update prototypes: ")
                centers = []
                for c_indx in range(args['model.num_clusters']):
                    center = torch.mean(selection_info['embeddings'][cluster_labels == c_indx], dim=0)     # [64]
                    centers.append(center)
                centers = torch.stack(centers).detach()

                pmo.selector.update_prototypes(centers)

                '''debug'''
                debugger.print_prototype_change(pmo, i=i, writer=writer)

            lr_manager_step(i)

            '''log iter-wise params change'''
            writer.add_scalar('params/learning_rate', optimizer.param_groups[0]['lr'], i+1)
            writer.add_scalar('params/gumbel_tau', pmo.selector.tau.item(), i+1)
            writer.add_scalar('params/sim_logit_scale', pmo.selector.logit_scale.item(), i+1)

            if (i + 1) % args['train.summary_freq'] == 0:
                print(f">> Iter: {i + 1}, train summary:")
                '''save epoch_loss and epoch_acc'''
                epoch_train_history = dict()
                if os.path.exists(os.path.join(args['out.dir'], 'summary', 'train_log.pickle')):
                    epoch_train_history = pickle.load(
                        open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'rb'))
                epoch_train_history[i + 1] = {'loss': epoch_loss.copy(), 'acc': epoch_acc.copy()}
                with open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'wb') as f:
                    pickle.dump(epoch_train_history, f)

                '''write task similarities'''
                if len(epoch_loss[f'task/gumbel_sim']) > 0:
                    similarities = np.concatenate(epoch_loss[f'task/gumbel_sim'][-10:])      # [num_tasks, 8]
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/task-gumbel-sim", figure, i+1)
                    similarities = np.concatenate(epoch_loss[f'task/softmax_sim'][-10:])      # [num_tasks, 8]
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/task-softmax-sim", figure, i+1)

                '''log pool ce loss'''
                if len(epoch_loss[f'pool/selection_ce_loss']) > 0:      # did selection loss on pool samples
                    writer.add_scalar('train_ce_loss',
                                      np.mean(epoch_loss[f'pool/selection_ce_loss']), i+1)

                '''write cluster centers'''
                centers = pmo.selector.prototypes
                centers = centers.view(*centers.shape[:2]).detach().cpu().numpy()
                figure = draw_heatmap(centers, verbose=False)
                writer.add_figure(f"train_image/cluster-centers", figure, i+1)

                epoch_loss, epoch_acc = init_train_log()

            '''----------'''
            '''Eval Phase'''
            '''----------'''
            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:   #  or i == 0:          # eval at init
                print(f"\n>> Iter: {i + 1}, evaluation:")
                # eval mode
                model_eval()

                '''nvidia-smi'''
                print(os.system('nvidia-smi'))

                '''collect val_losses/accs for all sources and cluster_losses/accs for all FiLMs'''
                val_accs, val_losses = {f'{name}': [] for name in valsets}, {f'{name}': [] for name in valsets}

                with torch.no_grad():
                    for j in tqdm(range(args['train.eval_size']), ncols=100):
                        '''obtain 1 task from all val_loader'''
                        for v_indx, valset in enumerate(valsets):
                            samples = val_loader.get_validation_task(session, valset, d=device)
                            context_images, target_images = samples['context_images'], samples['target_images']
                            context_labels, target_labels = samples['context_labels'], samples['target_labels']
                            context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                            domain = v_indx

                            features = pmo.embed(torch.cat([context_images, target_images]))
                            labels = [domain] * features.size(0).long().to(device)

                            _, selection_info = pmo.selector(features, gumbel=False, hard=False, average=False)
                            _, stats_dict, _ = cross_entropy_loss(selection_info['dist'], labels)

                            val_losses[valset].append(stats_dict['loss'])
                            val_accs[valset].append(stats_dict['acc'])

                '''write and print val on source'''
                for v_indx, valset in enumerate(valsets):
                    print(f"==>> evaluate results on {valset}.")
                    epoch_val_loss[valset] = np.mean(val_losses[valset])
                    epoch_val_acc[valset] = np.mean(val_accs[valset])
                    writer.add_scalar(f"val-domain-loss/{valset}", epoch_val_loss[valset], i+1)
                    writer.add_scalar(f"val-domain-accuracy/{valset}", epoch_val_acc[valset], i+1)
                    print(f"==>> val: loss {np.mean(val_losses[valset]):.3f}, "
                          f"accuracy {np.mean(val_accs[valset]):.3f}.")

                '''write summaries averaged over sources'''
                avg_val_source_loss = np.mean(np.concatenate([val_loss for val_loss in val_losses.values()]))
                avg_val_source_acc = np.mean(np.concatenate([val_acc for val_acc in val_accs.values()]))
                writer.add_scalar(f"val-domain-loss/avg_val_source_loss", avg_val_source_loss, i+1)
                writer.add_scalar(f"val-domain-accuracy/avg_val_source_acc", avg_val_source_acc, i+1)
                print(f"==>> val: avg_loss {avg_val_source_loss:.3f}, "
                      f"avg_accuracy {avg_val_source_acc:.3f}.")

                '''evaluation acc based on source domain acc'''
                avg_val_loss, avg_val_acc = avg_val_source_loss, avg_val_source_acc

                # saving checkpoints
                if avg_val_acc > best_val_acc and i > 0:    # do not consider init
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc,
                              'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                checkpointer.save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=optimizer,
                        state_dict=pmo.get_state_dict(whole=False), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': i + 1, 'loss': epoch_val_loss, 'acc': epoch_val_acc}, f)

                print(f"====>> Trained and evaluated at {i + 1}.\n")

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.3f}""")
    else:
        print(f"""Training not completed. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()       # mute for only do testing

    # # run testing
    # from test_extractor_pa import main as test
    # test(test_model='best')
    # print("↑↑ best model")
    # test(test_model='last')
    # print("↑↑ last model")

    '''nvidia-smi'''
    print(os.system('nvidia-smi'))

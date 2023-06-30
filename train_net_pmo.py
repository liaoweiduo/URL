"""
This code allows you to train clustering modulation with pool mo technique.

Author: Weiduo Liao
Date: 2023.06.21
"""

import os
import sys
import pickle
import copy
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, device, set_determ, check_dir
from config import args

from pmo_utils import Pool, Mixer, prototype_similarity, cal_hv_loss, cal_hv, draw_objs, draw_heatmap, pmo_embed

import warnings
warnings.filterwarnings('ignore')


def train():
    # Set seed
    set_determ(seed=1234)

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

        train_loaders = dict()
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                'train', [trainset], valsets, testsets, test_type=args['train.type'])
            num_train_classes[trainset] = train_loaders[trainset].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        # train_loader = MetaDatasetEpisodeReader('train', trainsets, valsets, testsets, test_type='5shot')
        # num_train_classes = train_loader.num_classes('train')
        # print(f'num_train_classes: {num_train_classes}')

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['test.type'])

        '''initialize models and optimizer'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0

        # pmo model load from url
        pmo = get_model(None, args, base_network_name='url')    # resnet18_moe

        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_parameters())
        checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
        if start_iter > 0:
            pool.restore(start_iter)      # restore pool cluster centers.   no use
        pool = pool.to(device)

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'],
                      num_sources=args['train.n_mix_source'], num_mixes=args['train.n_mix'])

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']

        def init_train_log():
            epoch_loss = {}
            if 'task' in args['train.loss_type']:
                epoch_loss.update({f'task/{name}': [] for name in trainsets})
            # epoch_loss['task/rec'] = []
            # if 'hv' in args['train.loss_type']:
            epoch_loss['hv/loss'], epoch_loss['hv'] = [], []
            epoch_loss.update({
                f'hv/obj{obj_idx}': {
                    f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})

            epoch_acc = {}
            if 'task' in args['train.loss_type']:
                epoch_acc['task/avg'] = []     # average over all trainsets
                epoch_acc.update({f'task/{name}': [] for name in trainsets})
            # if 'hv' in args['train.loss_type']:
            epoch_acc['hv'] = []
            epoch_acc.update({
                f'hv/obj{obj_idx}': {
                    f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            return epoch_loss, epoch_acc

        def model_train():
            # train mode
            pmo.train()
            pool.train()

        def model_eval():
            # eval mode
            pmo.eval()
            pool.eval()

        def zero_grad():
            optimizer.zero_grad()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.zero_grad()

        def update_step(idx):
            optimizer.step()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.step()

            lr_manager.step(idx)
            if args['train.cluster_center_mode'] == 'trainable':
                pool.lr_manager.step(idx)

        epoch_loss, epoch_acc = init_train_log()
        epoch_val_loss = {}
        epoch_val_acc = {}

        print(f'\n>>>> Train start from {start_iter}.')
        for i in tqdm(range(start_iter, max_iter), ncols=100):

            zero_grad()
            model_train()

            '''----------------'''
            '''Task Train Phase'''
            '''----------------'''
            if 'task' in args['train.loss_type']:
                '''obtain tasks from train_loaders'''
                p = np.ones(len(trainsets))
                if 'ilsvrc_2012' in trainsets:
                    p[trainsets.index('ilsvrc_2012')] = 2.0
                p = p / sum(p)
                t_indx = np.random.choice(len(trainsets), p=p)
                trainset = trainsets[t_indx]

                samples = train_loaders[trainset].get_train_task(session, d=device)
                context_images, target_images = samples['context_images'], samples['target_images']
                context_labels, target_labels = samples['context_labels'], samples['target_labels']
                # context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                # domain = t_indx

                enriched_context_features = pmo(context_images, gumbel=True)
                enriched_target_features = pmo(target_images, gumbel=True)

                task_loss, stats_dict, _ = prototype_loss(
                    enriched_context_features, context_labels,
                    enriched_target_features, target_labels,
                    distance=args['test.distance'])
                task_loss.backward()

                '''log task loss and acc'''
                epoch_loss[f'task/{trainset}'].append(stats_dict['loss'])
                epoch_acc[f'task/{trainset}'].append(stats_dict['acc'])
                # ilsvrc_2012 has 2 times larger len than others.

                # del samples, context_images, target_images, task_loss

            '''----------------'''
            '''MO Train Phase  '''
            '''----------------'''
            if (i + 1) % args['train.mo_freq'] == 0:
                print(f"\n>> Iter: {i + 1}, MO phase: "
                      f"({'train' if 'hv' in args['train.loss_type'] else 'eval'})")

                pool.clear_clusters()
                pool.clear_buffer()

                '''fill pool from train_loaders'''
                verbose = True
                for t in tqdm(range(args['train.max_sampling_iter_for_pool']), ncols=100):
                # while True:     # apply sampling multiple times to make sure enough samples
                    for t_indx, trainset in enumerate(trainsets):
                        num_task_per_batch = 1 if trainset != 'ilsvrc_2012' else 2
                        for _ in range(num_task_per_batch):
                            samples = train_loaders[trainset].get_train_task(session, d='cpu')
                            images = torch.cat([samples['context_images'], samples['target_images']])
                            re_labels = torch.cat([samples['context_labels'], samples['target_labels']]).numpy()
                            gt_labels = torch.cat([samples['context_gt'], samples['target_gt']]).numpy()
                            domain = np.array([t_indx] * len(gt_labels))

                            # put in sequence
                            # '''obtain selection vec for images'''
                            # with torch.no_grad():
                            #     _, selection_info = pmo.selector(
                            #         pmo.embed(images.to(device)), gumbel=True)  # [bs, n_clusters]
                            #     similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]
                            #     cluster_idxs = np.argmax(similarities, axis=1)  # [bs]
                            #     similarities = selection_info['normal_soft'].detach().cpu().numpy()
                            #     # using gumbel to determine which cluster to put, but similarity use normal softmax
                            #
                            # pool.put_batch(
                            #     images, cluster_idxs, {
                            #         'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})

                            # put to buffer then put to cluster
                            '''obtain selection vec for images'''
                            with torch.no_grad():
                                _, selection_info = pmo.selector(
                                    pmo.embed(images.to(device)), gumbel=False)  # [bs, n_clusters]
                                similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                            pool.put_buffer(images, {
                                'domain': domain, 'gt_labels': gt_labels,
                                're_labels': re_labels, 'similarities': similarities})

                    # '''check pool has enough samples'''
                    # available_cluster_idxs = []
                    # for idx, classes in enumerate(pool.current_classes()):
                    #     # if len(classes) >= args['train.n_way']:
                    #     num_imgs = np.array([cls[1] for cls in classes])
                    #     if len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']]
                    #            ) >= args['train.n_way']:
                    #         available_cluster_idxs.append(idx)
                    #
                    # if len(available_cluster_idxs) >= args['train.n_obj'] and verbose:
                    #     print(f"==>> pool has enough samples after "
                    #           f"{t+1}/{args['train.max_sampling_iter_for_pool']} iters of sampling.")
                    #     verbose = False
                    #     # break
                    #
                    # if t == args['train.max_sampling_iter_for_pool'] - 1 and verbose:
                    #     print(f"==>> pool has not enough samples. skip MO training")

                '''buffer -> clusters'''
                pool.buffer2cluster()

                '''check pool has enough samples'''
                available_cluster_idxs = []
                for idx, classes in enumerate(pool.current_classes()):
                    # if len(classes) >= args['train.n_way']:
                    num_imgs = np.array([cls[1] for cls in classes])
                    if len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']]
                           ) >= args['train.n_way']:
                        available_cluster_idxs.append(idx)
                if verbose:
                    if len(available_cluster_idxs) >= args['train.n_obj']:
                        print(f"==>> pool has enough samples.")
                    else:
                        print(f"==>> pool has not enough samples. skip MO training")

                '''repeat collecting MO loss'''
                if len(available_cluster_idxs) >= args['train.n_obj']:
                    for mo_train_idx in range(args['train.n_mo']):
                        selected_cluster_idxs = sorted(np.random.choice(
                            available_cluster_idxs, args['train.n_obj'], replace=False))
                        # which is also devices idx
                        # device_list = list(set([devices[idx] for idx in selected_cluster_idxs]))    # unique devices

                        '''sample pure tasks from clusters in selected_cluster_idxs'''
                        numpy_tasks = []
                        for cluster_idx in selected_cluster_idxs:
                            numpy_pure_task = pool.episodic_sample(cluster_idx)
                            numpy_tasks.append(numpy_pure_task)

                        '''sample mix tasks by mixer'''
                        for mix_id in range(args['train.n_mix']):
                            numpy_mix_task, _ = mixer.mix(
                                task_list=[pool.episodic_sample(idx) for idx in selected_cluster_idxs],
                                mix_id=mix_id
                            )
                            numpy_tasks.append(numpy_mix_task)

                        '''obtain ncc loss multi-obj matrix and put to last device'''
                        ncc_losses_multi_obj = []    # [4, 2]
                        for task_idx, task in enumerate(numpy_tasks):

                            '''to device'''
                            context_images = torch.from_numpy(task['context_images']).to(device)
                            context_labels = torch.from_numpy(task['context_labels']).long().to(device)
                            target_images = torch.from_numpy(task['target_images']).to(device)
                            target_labels = torch.from_numpy(task['target_labels']).long().to(device)

                            if 'hv' in args['train.loss_type'] or 'pure' in args['train.loss_type']:
                                enriched_context_features_list = [
                                    pmo(context_images, gumbel=True, selected_idx=selected_idx)
                                    for selected_idx in selected_cluster_idxs]
                                enriched_target_features_list = [
                                    pmo(target_images, gumbel=True, selected_idx=selected_idx)
                                    for selected_idx in selected_cluster_idxs]
                            else:
                                with torch.no_grad():
                                    enriched_context_features_list = [
                                        pmo(context_images, gumbel=True, selected_idx=selected_idx)
                                        for selected_idx in selected_cluster_idxs]
                                    enriched_target_features_list = [
                                        pmo(target_images, gumbel=True, selected_idx=selected_idx)
                                        for selected_idx in selected_cluster_idxs]

                            losses = []     # [2,]
                            for obj_idx in range(len(selected_cluster_idxs)):       # obj_idx is the selected model
                                loss, stats_dict, _ = prototype_loss(
                                    enriched_context_features_list[obj_idx], context_labels,
                                    enriched_target_features_list[obj_idx], target_labels,
                                    distance=args['test.distance'])

                                # loss for all tasks on 1 model on 1 device. to last device
                                losses.append(loss)

                                epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['loss'])  # [2, 4]
                                epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['acc'])
                                if task_idx < len(selected_cluster_idxs):       # [2, 2]
                                    if task_idx == obj_idx and 'pure' in args['train.loss_type']:
                                        # backward pure loss on the corresponding model and cluster.
                                        retain_graph = 'hv' in args['train.loss_type']
                                        loss.backward(retain_graph=retain_graph)

                            ncc_losses_multi_obj.append(torch.stack(losses))
                        ncc_losses_multi_obj = torch.stack(ncc_losses_multi_obj)   # shape [num_tasks, num_objs], [4, 2]

                        '''calculate HV loss'''
                        ref = args['train.ref']
                        ncc_losses_multi_obj = ncc_losses_multi_obj.T       # [2, 4]
                        hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
                        epoch_loss['hv/loss'].append(hv_loss.item())

                        '''calculate HV value for mutli-obj loss and acc'''
                        obj = np.array([[
                            epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                            for task_idx in range(len(numpy_tasks))
                        ] for obj_idx in range(len(selected_cluster_idxs))])
                        hv = cal_hv(obj, ref, target='loss')
                        epoch_loss['hv'].append(hv)
                        obj = np.array([[
                            epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                            for task_idx in range(len(numpy_tasks))
                        ] for obj_idx in range(len(selected_cluster_idxs))])
                        hv = cal_hv(obj, 0, target='acc')
                        epoch_acc['hv'].append(hv)

                        if 'hv' in args['train.loss_type']:
                            hv_loss = hv_loss * args['train.hv_coefficient']
                            '''since no torch is saved in the pool, do not need to retain_graph'''
                            # retain_graph = True if mo_train_idx < args['train.n_mo'] - 1 else False
                            # hv_loss.backward(retain_graph=retain_graph)
                            hv_loss.backward()

            update_step(i)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i+1)

            # # saving pool
            # pool.store(i, train_loaders, trainsets, False,
            #            class_filename=f'pool-{i+1}.json', center_filename=f'pool-{i+1}.npy')

            if (i + 1) % args['train.summary_freq'] == 0:        # 5; 2 for DEBUG
                print(f">> Iter: {i + 1}, train summary:")
                '''save epoch_loss and epoch_acc'''
                epoch_train_history = dict()
                if os.path.exists(os.path.join(args['out.dir'], 'summary', 'train_log.pickle')):
                    epoch_train_history = pickle.load(
                        open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'rb'))
                epoch_train_history[i + 1] = {'loss': epoch_loss.copy(), 'acc': epoch_acc.copy()}
                with open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'wb') as f:
                    pickle.dump(epoch_train_history, f)

                '''log task loss and accuracy'''
                average_loss, average_accuracy = [], []
                for dataset_name in trainsets:
                    if f'task/{dataset_name}' in epoch_loss.keys() and len(epoch_loss[f'task/{dataset_name}']) > 0:
                        writer.add_scalar(f"train_loss/task/{dataset_name}",
                                          np.mean(epoch_loss[f'task/{dataset_name}']), i+1)
                        writer.add_scalar(f"train_accuracy/task/{dataset_name}",
                                          np.mean(epoch_acc[f'task/{dataset_name}']), i+1)
                        average_loss.append(epoch_loss[f'task/{dataset_name}'])
                        average_accuracy.append(epoch_acc[f'task/{dataset_name}'])
                average_loss = np.mean(np.concatenate(average_loss))
                average_accuracy = np.mean(np.concatenate(average_accuracy))
                writer.add_scalar(f"train_loss/task/average", average_loss, i+1)
                writer.add_scalar(f"train_accuracy/task/average", average_accuracy, i+1)
                print(f"==>> task: loss {average_loss:.3f}, "
                      f"accuracy {average_accuracy:.3f}.")

                # '''log task_rec'''
                # writer.add_scalar(f"loss/train/task_rec",
                #                   np.mean(epoch_loss['task_rec']), i+1)

                if len(epoch_loss['hv/loss']) > 0:      # did mo process
                    '''log multi-objective loss and accuracy'''
                    objs_loss, objs_acc = [], []        # for average figure visualization
                    for obj_idx in range(args['train.n_obj']):
                        obj_loss, obj_acc = [], []
                        for pop_idx in range(args['train.n_mix'] + args['train.n_obj']):
                            loss_values = epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"train_loss/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(loss_values), i+1)
                            obj_loss.append(np.mean(loss_values))
                            acc_values = epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"train_accuracy/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(acc_values), i+1)
                            obj_acc.append(np.mean(acc_values))
                        objs_loss.append(obj_loss)
                        objs_acc.append(obj_acc)

                    '''log objs figure'''
                    pop_labels = [
                        f"p{idx}" if idx < args['train.n_obj'] else f"m{idx-args['train.n_obj']}"
                        for idx in range(args['train.n_mix'] + args['train.n_obj'])
                    ]       # ['p0', 'p1', 'm0', 'm1']
                    objs = np.array(objs_loss)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"train_image/objs_loss", figure, i+1)
                    objs = np.array(objs_acc)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"train_image/objs_acc", figure, i+1)

                    '''log hv'''
                    writer.add_scalar('train_loss/hv_loss', np.mean(epoch_loss['hv/loss']), i+1)
                    writer.add_scalar('train_loss/hv', np.mean(epoch_loss['hv']), i+1)
                    writer.add_scalar('train_accuracy/hv', np.mean(epoch_acc['hv']), i+1)
                    print(f"==>> hv: hv_loss {np.mean(epoch_loss['hv/loss']):.3f}, "
                          f"loss {np.mean(epoch_loss['hv']):.3f}, "
                          f"accuracy {np.mean(epoch_acc['hv']):.3f}.")

                '''write pool images'''
                images = pool.current_images(single_image=True)
                for cluster_id, cluster in enumerate(images):
                    if len(cluster) > 0:
                        writer.add_image(f"train_image/pool-{cluster_id}", cluster, i+1)
                    #     img_in_cluster = np.concatenate(cluster)
                    #     writer.add_images(f"train_image/pool-{cluster_id}", img_in_cluster, i+1)

                '''write pool similarities'''
                similarities = pool.current_similarities()
                for cluster_id, cluster in enumerate(similarities):
                    if len(cluster) > 0:
                        sim_in_cluster = np.stack(cluster)  # [num_cls, 8]
                        figure = draw_heatmap(sim_in_cluster, verbose=False)
                        writer.add_figure(f"train_image/pool-{cluster_id}-sim", figure, i+1)

                '''write cluster centers'''
                centers = pmo.selector.prototypes
                centers = centers.view(*centers.shape[:2]).detach().cpu().numpy()
                figure = draw_heatmap(centers, verbose=False)
                writer.add_figure(f"train_image/cluster-centers", figure, i+1)

                '''write pool assigns & gates'''
                if args['train.cluster_center_mode'] == 'hierarchical':
                    assigns, gates = pool.current_assigns_gates()
                    for cluster_id, (assign, gate) in enumerate(zip(assigns, gates)):
                        # assign num_cls*[8,]   gate num_cls*[8,4,]
                        if len(assign) > 0:
                            asn_in_cluster = np.stack(assign)  # [num_cls, 8]
                            argmx_asn = np.argmax(asn_in_cluster, axis=1)   # [num_cls, ]
                            gat_in_cluster = np.stack([
                                gat[asn_idx] for asn_idx, gat in zip(argmx_asn, gate)])  # [num_cls, 4]
                            figure = draw_heatmap(asn_in_cluster)
                            writer.add_figure(f"train_image/pool-{cluster_id}-assigns", figure, i+1)
                            figure = draw_heatmap(gat_in_cluster)
                            writer.add_figure(f"train_image/pool-{cluster_id}-gates", figure, i+1)

                if len(epoch_loss['hv/loss']) > 0:      # did mo process
                    '''write pure and mixed tasks'''
                    for task_id, task in enumerate(numpy_tasks):
                        imgs = np.concatenate([task['context_images'], task['target_images']])
                        writer.add_images(f"train_image/task-{pop_labels[task_id]}", imgs, i+1)

                epoch_loss, epoch_acc = init_train_log()

            '''----------'''
            '''Eval Phase'''
            '''----------'''
            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:      # args['train.eval_freq']; 10 for DEBUG
                print(f"\n>> Iter: {i + 1}, evaluation:")
                # eval mode
                model_eval()

                val_pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
                val_pool.centers = pool.centers     # same centers and device as train_pool
                val_pool.eval()

                '''collect val_losses/accs for all sources and cluster_losses/accs for all FiLMs'''
                val_accs, val_losses = {f'{name}': [] for name in valsets}, {f'{name}': [] for name in valsets}
                cluster_accs, cluster_losses = [[] for _ in range(args['model.num_clusters'])], \
                                               [[] for _ in range(args['model.num_clusters'])]
                with torch.no_grad():
                    for v_indx, valset in enumerate(valsets):
                        print(f"==>> evaluate on {valset}.")
                        for j in tqdm(range(args['train.eval_size']), ncols=100):
                            '''obtain 1 task from val_loader'''
                            samples = val_loader.get_validation_task(session, valset, d=device)
                            context_images, target_images = samples['context_images'], samples['target_images']
                            context_labels, target_labels = samples['context_labels'], samples['target_labels']
                            context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                            domain = v_indx

                            enriched_context_features = pmo(context_images, gumbel=False)
                            enriched_target_features = pmo(target_images, gumbel=False)

                            _, stats_dict, _ = prototype_loss(
                                enriched_context_features, context_labels,
                                enriched_target_features, target_labels,
                                distance=args['test.distance'])

                            val_losses[valset].append(stats_dict['loss'])
                            val_accs[valset].append(stats_dict['acc'])

                            '''put to val_pool'''
                            images = torch.cat([context_images, target_images])
                            gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
                            domain = np.array([domain] * len(gt_labels))

                            '''obtain selection vec for images'''
                            _, selection_info = pmo.selector(
                                pmo.embed(images), gumbel=False)  # [bs, n_clusters]
                            similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]
                            cluster_idxs = np.argmax(similarities, axis=1)  # [bs]

                            val_pool.put_batch(
                                images.cpu(), cluster_idxs, {
                                    'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})

                            '''check if any cluster have sufficient class to construct 1 task'''
                            for idx, classes in enumerate(val_pool.current_classes()):
                                num_imgs = np.array([cls[1] for cls in classes])
                                available_way = len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']])
                                if available_way >= args['train.n_way']:
                                    # enough classes to construct 1 task
                                    # then use all available classes to construct 1 task
                                    task = val_pool.episodic_sample(
                                        idx, n_way=available_way, remove_sampled_classes=True,
                                        d=device
                                    )

                                    enriched_context_features = pmo(task['context_images'], gumbel=False)
                                    enriched_target_features = pmo(task['target_images'], gumbel=False)

                                    _, stats_dict, _ = prototype_loss(
                                        enriched_context_features, task['context_labels'],
                                        enriched_target_features, task['target_labels'],
                                        distance=args['test.distance'])

                                    cluster_losses[idx].append(stats_dict['loss'])
                                    cluster_accs[idx].append(stats_dict['acc'])

                        '''write and print val on source'''
                        epoch_val_loss[valset] = np.mean(val_losses[valset])
                        epoch_val_acc[valset] = np.mean(val_accs[valset])
                        writer.add_scalar(f"val_loss/{valset}", epoch_val_loss[valset], i+1)
                        writer.add_scalar(f"val_accuracy/{valset}", epoch_val_acc[valset], i+1)
                        print(f"==>> val: loss {np.mean(val_losses[valset]):.3f}, "
                              f"accuracy {np.mean(val_accs[valset]):.3f}.")

                '''write summaries averaged over sources'''
                avg_val_source_loss = np.mean(np.concatenate([val_loss for val_loss in val_losses.values()]))
                avg_val_source_acc = np.mean(np.concatenate([val_acc for val_acc in val_accs.values()]))
                writer.add_scalar(f"val_loss/avg_val_source_loss", avg_val_source_loss, i+1)
                writer.add_scalar(f"val_accuracy/avg_val_source_acc", avg_val_source_acc, i+1)
                print(f"==>> val: avg_loss {avg_val_source_loss:.3f}, "
                      f"avg_accuracy {avg_val_source_acc:.3f}.")

                '''write and print val on cluster'''
                for cluster_idx, (loss_list, acc_list) in enumerate(zip(cluster_losses, cluster_accs)):
                    if len(loss_list) > 0:
                        cluster_acc, cluster_loss = np.mean(acc_list).item(), np.mean(loss_list).item()

                        epoch_val_loss[f"C{cluster_idx}"] = cluster_loss
                        epoch_val_acc[f"C{cluster_idx}"] = cluster_acc
                        writer.add_scalar(f"val_loss/C{cluster_idx}", cluster_loss, i+1)
                        writer.add_scalar(f"val_accuracy/C{cluster_idx}", cluster_acc, i+1)
                        print(f"==>> val C{cluster_idx}: "
                              f"val_loss {cluster_loss:.3f}, accuracy {cluster_acc:.3f}")

                    else:   # no class(task) is assign to this cluster
                        print(f"==>> val C{cluster_idx}: "
                              f"val_loss No value, val_acc No value")

                # write summaries averaged over sources/clusters
                avg_val_cluster_loss = np.mean(np.concatenate(cluster_losses))
                avg_val_cluster_acc = np.mean(np.concatenate(cluster_accs))
                writer.add_scalar(f"val_loss/avg_val_cluster_loss", avg_val_cluster_loss, i+1)
                writer.add_scalar(f"val_accuracy/avg_val_cluster_acc", avg_val_cluster_acc, i+1)
                print(f"==>> val: avg_loss {avg_val_cluster_loss:.3f}, "
                      f"avg_accuracy {avg_val_cluster_acc:.3f}.")

                # evaluation acc based on cluster acc
                avg_val_loss, avg_val_acc = avg_val_cluster_loss, avg_val_cluster_acc

                # saving checkpoints
                if avg_val_acc > best_val_acc:
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
                        state_dict=pmo.get_state_dict(), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': i + 1, 'loss': epoch_val_loss, 'acc': epoch_val_acc}, f)

                print(f"====>> Trained and evaluated at {i + 1}.\n")

                # # saving pool
                # pool.store(i, train_loaders, trainsets, is_best)

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.3f}""")
    else:
        print(f"""Training not completed. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

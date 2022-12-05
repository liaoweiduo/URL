"""
This code allows you to train multi learned domain learning networks with pool mo technique.

Author: Weiduo Liao
Date: 2022.11.12
"""

import os
import sys
import pickle
import copy
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, device, devices, set_determ, check_dir
from config import args

from pmo_utils import Pool, Mixer, prototype_similarity, cal_hv_loss


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

        print(f'Devices: {devices}')

        # train_loaders = []
        # num_train_classes = dict()
        # for t_indx, trainset in enumerate(trainsets):
        #     train_loaders.append(MetaDatasetEpisodeReader('train', [trainset], valsets, testsets, test_type='1shot'))
        #     num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        train_loader = MetaDatasetEpisodeReader('train', trainsets, valsets, testsets, test_type='5shot')
        num_train_classes = train_loader.num_classes('train')
        print(f'num_train_classes: {num_train_classes}')

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type='5shot')

        '''initialize model and optimizer'''
        models = []
        optimizers = []
        checkpointers = []
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        lr_managers = []
        # init all starting issues for M(8) models.
        model_names = [args['model.name'].format(m_indx) for m_indx in range(args['model.num_clusters'])]
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]
        # M0-net - M7-net
        for m_indx in range(args['model.num_clusters']):        # 8
            model_args_with_name = copy.deepcopy(args)
            model_args_with_name['model.name'] = model_names[m_indx]
            _model = get_model(None, model_args_with_name, multi_device_id=m_indx)  # distribute model to multi-devices
            models.append(_model)
            _optimizer = get_optimizer(_model, model_args_with_name, params=_model.get_parameters())
            optimizers.append(_optimizer)

            # restoring the last checkpoint
            _checkpointer = CheckPointer(model_args_with_name, _model, optimizer=_optimizer)
            checkpointers.append(_checkpointer)

            if os.path.isfile(_checkpointer.last_ckpt) and args['train.resume']:
                start_iter, best_val_loss, best_val_acc =\
                    _checkpointer.restore_model(ckpt='last')    # all 3 things are the same for all models
            else:
                print('No checkpoint restoration')

            # define learning rate policy
            if args['train.lr_policy'] == "step":
                _lr_manager = UniformStepLR(_optimizer, args, start_iter)
            elif "exp_decay" in args['train.lr_policy']:
                _lr_manager = ExpDecayLR(_optimizer, args, start_iter)
            elif "cosine" in args['train.lr_policy']:
                _lr_manager = CosineAnnealRestartLR(_optimizer, args, start_iter)
            lr_managers.append(_lr_manager)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'])
        if start_iter > 0:
            pool.restore()      # restore pool cluster centers.

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'])

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']
        epoch_train_history = dict()
        epoch_loss = {model_name: {name: [] for name in cluster_names} for model_name in model_names}
        epoch_loss.update({
            obj_idx: {
                pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
            } for obj_idx in range(args['train.n_obj'])})
        epoch_loss['hv'] = []
        epoch_acc = {model_name: {name: [] for name in cluster_names} for model_name in model_names}
        epoch_acc.update({
            obj_idx: {
                pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
            } for obj_idx in range(args['train.n_obj'])})
        # epoch_val_loss = {model_name: {name: [] for name in valsets} for model_name in model_names}
        # epoch_val_acc = {model_name: {name: [] for name in valsets} for model_name in model_names}
        epoch_val_loss = {model_name: [] for model_name in model_names}
        epoch_val_acc = {model_name: [] for model_name in model_names}

        print(f'>>>> Train start from {start_iter}.')
        for i in tqdm(range(max_iter), ncols=50):
            if i < start_iter:
                continue

            '''----------------'''
            '''Clustering Phase'''
            '''----------------'''
            # eval mode
            for _model in models:
                _model.eval()

            with torch.no_grad():
                '''obtain 1 task from train_loader'''
                sample_numpy = train_loader.get_train_task(session, d='numpy')
                images_numpy = np.concatenate([sample_numpy['context_images'], sample_numpy['target_images']])
                re_labels_numpy = np.concatenate([sample_numpy['context_labels'], sample_numpy['target_labels']])
                gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                domain = sample_numpy['domain'].item()

                pool.clustering(images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                train_loader, devices, models, i, update_cluster_centers=True)

            # back to train mode
            for _model in models:
                _model.train()

            '''--------------'''
            '''Training Phase'''
            '''--------------'''
            for optimizer in optimizers:
                optimizer.zero_grad()

            '''select args['train.n_obj']=2 clusters'''
            assert args['train.n_obj'] == 2
            available_cluster_idxs = []
            for idx, classes in enumerate(pool.current_classes()):
                # if len(classes) >= args['train.n_way']:
                num_imgs = np.array([cls[1] for cls in classes])
                if len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']]) >= args['train.n_way']:
                    available_cluster_idxs.append(idx)
            if len(available_cluster_idxs) < args['train.n_obj']:      # not enough available clusters
                print(f'Iter: {i + 1}, not enough available clusters.')
            else:
                selected_cluster_idxs = sorted(np.random.choice(
                    available_cluster_idxs, args['train.n_obj'], replace=False))
                # which is also devices idx
                device_list = list(set([devices[idx] for idx in selected_cluster_idxs]))    # unique devices

                '''sample pure tasks from clusters in selected_cluster_idxs'''
                numpy_tasks = []
                pure_tasks = []
                for idx in selected_cluster_idxs:
                    numpy_pure_task = pool.episodic_sample(idx)
                    numpy_tasks.append(numpy_pure_task)
                    pure_tasks.append(pool.to_torch(numpy_pure_task, device_list=device_list))

                '''sample mix tasks by mixer'''
                mix_tasks = []
                for mix_id in range(args['train.n_mix']):
                    numpy_mix_task = mixer.mix(
                        task_list=[pool.episodic_sample(idx) for idx in selected_cluster_idxs],
                        mix_id=mix_id
                    )[0]
                    numpy_tasks.append(numpy_mix_task)
                    mix_tasks.append(pool.to_torch(numpy_mix_task, device_list=device_list))

                '''obtain ncc loss multi-obj matrix and put to last device'''
                ncc_losses_multi_obj = []   # shape [num_objs, num_tasks], [2, 4]
                for obj_idx, cluster_idx in enumerate(selected_cluster_idxs):
                    _model, d = models[cluster_idx], devices[cluster_idx]
                    tasks = [pure_tasks[idx][d] for idx in range(len(pure_tasks))]
                    tasks.extend([mix_tasks[idx][d] for idx in range(len(mix_tasks))])

                    losses = []
                    for task_idx, task in enumerate(tasks):
                        context_features = _model.embed(task['context_images'])
                        target_features = _model.embed(task['target_images'])
                        context_labels = task['context_labels']
                        target_labels = task['target_labels']
                        loss, stats_dict, _ = prototype_loss(
                            context_features, context_labels,
                            target_features, target_labels, distance=args['test.distance'])
                        # loss for all tasks on 1 model on 1 device. to last device
                        losses.append(loss.to(device))
                        epoch_loss[obj_idx][task_idx].append(stats_dict['loss'])    # [2, 4]
                        epoch_acc[obj_idx][task_idx].append(stats_dict['acc'])
                        if task_idx < len(selected_cluster_idxs):       # [2, 2]
                            epoch_loss[model_names[cluster_idx]][
                                cluster_names[selected_cluster_idxs[task_idx]]
                            ].append(stats_dict['loss'])
                            epoch_acc[model_names[cluster_idx]][
                                cluster_names[selected_cluster_idxs[task_idx]]
                            ].append(stats_dict['acc'])
                    ncc_losses_multi_obj.append(torch.stack(losses))
                ncc_losses_multi_obj = torch.stack(ncc_losses_multi_obj)   # shape [num_objs, num_tasks], [2, 4]

                '''calculate HV loss'''
                ref = args['train.ref']
                hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
                epoch_loss['hv'].append(hv_loss.item())

                hv_loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                for lr_manager in lr_managers:
                    lr_manager.step(i)

            if (i + 1) % 200 == 0:        # 200; 5 for DEBUG
                print(f">> Iter: {i + 1}, train summary:")
                '''save epoch_loss and epoch_acc'''
                epoch_train_history[i + 1] = {'loss': epoch_loss, 'acc': epoch_acc}
                np.save(os.path.join(args['out.dir'], 'summary', 'train_log.npy'), epoch_train_history)

                objs = []
                '''log multi-objective loss and accuracy'''
                for obj_idx in range(args['train.n_obj']):
                    obj = []
                    for pop_idx in range(args['train.n_mix'] + args['train.n_obj']):
                        loss_values = epoch_loss[obj_idx][pop_idx]
                        writer.add_scalar(f"loss/{obj_idx}/{pop_idx}/train_loss",
                                          np.mean(loss_values), i+1)
                        obj.append(np.mean(loss_values))
                        acc_values = epoch_acc[obj_idx][pop_idx]
                        writer.add_scalar(f"accuracy/{obj_idx}/{pop_idx}/train_acc",
                                          np.mean(acc_values), i+1)
                    objs.append(obj)

                '''log objs'''
                objs = np.array(objs)     # [2, 4]
                figure = plt.figure()
                plt.scatter(objs[0], objs[1])
                writer.add_figure(f"image/train_objs", figure, i+1)

                # writer.add_embedding(objs, metadata=[f'p{p_idx}' for p_idx in range(objs.shape[0])])
                '''log loss and accuracy on all models on all clusters'''
                for model_name in model_names:
                    for cluster_name in cluster_names:
                        loss_values = epoch_loss[model_name][cluster_name]
                        acc_values = epoch_acc[model_name][cluster_name]
                        if len(loss_values) > 0:
                            writer.add_scalar(f"loss/{model_name}/{cluster_name}-train_loss",
                                              np.mean(loss_values), i+1)
                            writer.add_scalar(f"accuracy/{model_name}/{cluster_name}-train_acc",
                                              np.mean(acc_values), i+1)
                        epoch_loss[model_name][cluster_name], epoch_acc[model_name][cluster_name] = [], []
                writer.add_scalar('loss/train_hv', np.mean(epoch_loss['hv']), i+1)
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], i+1)
                print(f"==>> loss/train_hv {np.mean(epoch_loss['hv']):.3f}.")

            '''----------'''
            '''Eval Phase'''
            '''----------'''
            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:      # args['train.eval_freq']; 10 for DEBUG
                print(f"\n>> Iter: {i + 1}, evaluation:")
                # eval mode
                for _model in models:
                    _model.eval()

                val_pool = Pool(capacity=args['model.num_clusters'])
                val_pool.centers = pool.centers     # same centers as train_pool

                '''collect cluster_losses/accs for all models'''
                cluster_accs, cluster_losses = [[] for _ in range(len(models))], [[] for _ in range(len(models))]
                for valset in valsets:
                    print(f"==>> collect classes from {valset}.")
                    for j in tqdm(range(args['train.eval_size']), ncols=50):
                        with torch.no_grad():
                            '''obtain 1 task from val_loader'''
                            sample_numpy = val_loader.get_validation_task(session, valset, d='numpy')
                            images_numpy = np.concatenate(
                                [sample_numpy['context_images'], sample_numpy['target_images']])
                            re_labels_numpy = np.concatenate(
                                [sample_numpy['context_labels'], sample_numpy['target_labels']])
                            gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                            domain = sample_numpy['domain'].item()

                            val_pool.clustering(images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                                train_loader, devices, models, i, update_cluster_centers=False)

                            '''check if any cluster have sufficient class to construct 1 task'''
                            for idx, classes in enumerate(val_pool.current_classes()):
                                num_imgs = np.array([cls[1] for cls in classes])
                                available_way = len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']])
                                if available_way >= args['train.n_way']:
                                    # enough classes to construct 1 task
                                    # then use all available classes to construct 1 task
                                    numpy_task = val_pool.episodic_sample(
                                        idx, n_way=available_way, remove_sampled_classes=True)

                                    '''put to device and forward model to obtain val_acc/loss'''
                                    d = devices[idx]
                                    model = models[idx]
                                    torch_task = val_pool.to_torch(numpy_task, device_list=[d])

                                    context_features = model.embed(torch_task[d]['context_images'])
                                    target_features = model.embed(torch_task[d]['target_images'])
                                    context_labels = torch_task[d]['context_labels']
                                    target_labels = torch_task[d]['target_labels']
                                    _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                                      target_features, target_labels)

                                    cluster_losses[idx].append(stats_dict['loss'])
                                    cluster_accs[idx].append(stats_dict['acc'])

                '''write and print'''
                for cluster_idx, (loss_list, acc_list) in enumerate(zip(cluster_losses, cluster_accs)):
                    if len(loss_list) > 0:
                        cluster_acc, cluster_loss = np.mean(acc_list) * 100, np.mean(loss_list)

                        epoch_val_loss[model_names[cluster_idx]].append(cluster_loss)
                        epoch_val_acc[model_names[cluster_idx]].append(cluster_acc)
                        writer.add_scalar(f"loss/{model_names[cluster_idx]}/val_loss", cluster_loss, i+1)
                        writer.add_scalar(f"accuracy/{model_names[cluster_idx]}/val_acc", cluster_acc, i+1)
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc {cluster_acc:.2f}%, val_loss {cluster_loss:.3f}")

                    else:   # no class(task) is assign to this cluster
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc No value, val_loss No value")

                # write summaries averaged over clusters
                avg_val_loss, avg_val_acc = np.mean(np.concatenate(cluster_losses)), np.mean(np.concatenate(cluster_accs))
                writer.add_scalar(f"loss/avg_val_loss", avg_val_loss, i+1)
                writer.add_scalar(f"accuracy/avg_val_acc", avg_val_acc, i+1)

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc,
                              'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                for cluster_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
                    checkpointers[cluster_idx].save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=optimizer,
                        state_dict=model.get_state_dict(), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                np.save(os.path.join(args['out.dir'], 'summary', 'val_log.npy'),
                        {'loss': epoch_val_loss, 'acc': epoch_val_acc})

                for _model in models:
                    _model.train()
                print(f"====>> Trained and evaluated at {i + 1}.\n")

                # saving pool
                pool.store(i, train_loader, is_best)

                # write pool
                images = pool.current_images()
                for cluster_id, cluster in enumerate(images):
                    if len(cluster) > 0:
                        img_in_cluster = np.concatenate(cluster)
                        writer.add_images(f"image/pool-{cluster_id}", img_in_cluster, i+1)

                # write pure and mixed tasks
                for task_id, task in enumerate(numpy_tasks):
                    imgs = np.concatenate([task['context_images'], task['target_images']])
                    writer.add_images(f"image/task-{task_id}", imgs, i+1)

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

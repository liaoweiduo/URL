"""
This code allows you to train clustering network and  multi learned domain learning networks with pool mo technique.

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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader, MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator, device, devices, cluster_device, set_determ, check_dir
from config import args

from pmo_utils import Pool, Mixer, prototype_similarity, cal_hv_loss, cal_hv, draw_objs

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

        print(f'Devices: {devices}.')
        print(f'Cluster network device: {cluster_device}.')
        print(f'Mult-obj NCC loss calculation device: {device}.')

        train_loaders = dict()
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                'train', [trainset], valsets, testsets, test_type='1shot')
            num_train_classes[trainset] = train_loaders[trainset].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        # train_loader = MetaDatasetEpisodeReader('train', trainsets, valsets, testsets, test_type='5shot')
        # num_train_classes = train_loader.num_classes('train')
        # print(f'num_train_classes: {num_train_classes}')

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['train.type'])

        '''initialize models and optimizer'''
        models = []
        optimizers = []
        checkpointers = []
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        lr_managers = []
        # init all starting issues for M(8) models.
        model_names = ["imagenet-net", "omniglot-net", "aircraft-net", "birds-net", "textures-net", "quickdraw-net",
                       "fungi-net", "vgg_flower-net"]
        cluster_model_name = 'C-net'
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]
        # M0-net - M7-net
        for m_indx in range(args['model.num_clusters']):        # 8
            model_args_with_name = copy.deepcopy(args)
            model_args_with_name['model.name'] = model_names[m_indx]
            _model = get_model(None, model_args_with_name, d=devices[m_indx])  # distribute model to multi-devices
            models.append(_model)
            _optimizer = get_optimizer(_model, model_args_with_name, params=_model.get_parameters())
            optimizers.append(_optimizer)

            # restoring the last checkpoint
            _checkpointer = CheckPointer(model_args_with_name, _model, optimizer=_optimizer)
            checkpointers.append(_checkpointer)

            _checkpointer.restore_model(ckpt='best', optimizer=False, strict=False)

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
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'])

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']

        def init_train_log():
            epoch_loss = {name: [] for name in trainsets}
            epoch_loss.update({model_name: {name: [] for name in cluster_names} for model_name in model_names})
            epoch_loss.update({
                obj_idx: {
                    pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_loss['hv_loss'], epoch_loss['hv'] = [], []

            epoch_acc = {name: [] for name in trainsets}
            epoch_acc.update({model_name: {name: [] for name in cluster_names} for model_name in model_names})
            epoch_acc.update({
                obj_idx: {
                    pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_acc['hv'] = []
            return epoch_loss, epoch_acc

        def model_train():
            # train mode
            for _model in models:
                _model.train()

        def model_eval():
            # eval mode
            for _model in models:
                _model.eval()

        def zero_grad():
            for optimizer in optimizers:
                optimizer.zero_grad()

        def update_step(idx):
            for optimizer in optimizers:
                optimizer.step()

            for lr_manager in lr_managers:
                lr_manager.step(idx)

        epoch_loss, epoch_acc = init_train_log()
        # epoch_val_loss = {model_name: {name: [] for name in valsets} for model_name in model_names}
        # epoch_val_acc = {model_name: {name: [] for name in valsets} for model_name in model_names}
        epoch_val_loss = {model_name: [] for model_name in model_names}
        epoch_val_acc = {model_name: [] for model_name in model_names}

        print(f'\n>>>> Train start from {start_iter}.')
        for i in tqdm(range(max_iter), ncols=100):

            if i < start_iter:
                continue

            zero_grad()
            model_train()

            '''select args['train.n_obj'] clusters'''
            selected_cluster_idxs = sorted(np.random.choice(
                np.arange(args['model.num_clusters']), args['train.n_obj'], replace=False))
            # which is also devices idx
            device_list = list(set([devices[idx] for idx in selected_cluster_idxs]))    # unique devices

            '''sample pure tasks from clusters in selected_cluster_idxs'''
            numpy_tasks = []
            pure_tasks = []
            for idx in selected_cluster_idxs:
                numpy_pure_task = train_loaders[trainsets[idx]].get_train_task(session, d='numpy')
                grad_ones = {'context_grad_ones': torch.ones(numpy_pure_task['context_images'].shape),
                             'target_grad_ones': torch.ones(numpy_pure_task['target_images'].shape)}

                numpy_tasks.append(numpy_pure_task)
                pure_tasks.append(pool.to_torch(numpy_pure_task, grad_ones, device_list=device_list))

            '''sample mix tasks by mixer'''
            mix_tasks = []
            for mix_id in range(args['train.n_mix']):
                [numpy_mix_task, grad_ones], _ = mixer.mix(
                    task_list=[
                        (train_loaders[trainsets[idx]].get_train_task(session, d='numpy'),
                         {'context_grad_ones': torch.ones(numpy_pure_task['context_images'].shape),
                          'target_grad_ones': torch.ones(numpy_pure_task['target_images'].shape)})
                        for idx in selected_cluster_idxs
                    ],
                    mix_id=mix_id
                )
                numpy_tasks.append(numpy_mix_task)
                mix_tasks.append(pool.to_torch(numpy_mix_task, grad_ones, device_list=device_list))

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
            epoch_loss['hv_loss'].append(hv_loss.item())

            '''calculate HV value for mutli-obj loss and acc'''
            obj = np.array([[
                epoch_loss[obj_idx][task_idx][-1] for task_idx in range(len(tasks))     # this iter
            ] for obj_idx in range(len(selected_cluster_idxs))])
            hv = cal_hv(obj, ref, target='loss')
            epoch_loss['hv'].append(hv)
            obj = np.array([[
                epoch_acc[obj_idx][task_idx][-1] for task_idx in range(len(tasks))
            ] for obj_idx in range(len(selected_cluster_idxs))])
            hv = cal_hv(obj, 0, target='acc')
            epoch_acc['hv'].append(hv)

            hv_loss = hv_loss * args['train.hv_coefficient']
            hv_loss.backward()

            update_step(i)

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
                for dataset_name in trainsets:
                    writer.add_scalar(f"loss/train/{dataset_name}",
                                      np.mean(epoch_loss[dataset_name]), i+1)
                    writer.add_scalar(f"accuracy/train/{dataset_name}",
                                      np.mean(epoch_acc[dataset_name]), i+1)

                '''log multi-objective loss and accuracy'''
                objs_loss, objs_acc = [], []
                for obj_idx in range(args['train.n_obj']):
                    obj_loss, obj_acc = [], []
                    for pop_idx in range(args['train.n_mix'] + args['train.n_obj']):
                        loss_values = epoch_loss[obj_idx][pop_idx]
                        writer.add_scalar(f"loss/train/{obj_idx}/{pop_idx}",
                                          np.mean(loss_values), i+1)
                        obj_loss.append(np.mean(loss_values))
                        acc_values = epoch_acc[obj_idx][pop_idx]
                        writer.add_scalar(f"accuracy/train/{obj_idx}/{pop_idx}",
                                          np.mean(acc_values), i+1)
                        obj_acc.append(np.mean(acc_values))
                    objs_loss.append(obj_loss)
                    objs_acc.append(obj_acc)

                '''log objs'''
                pop_labels = [
                    f"p{idx}" if idx < args['train.n_obj'] else f"m{idx-args['train.n_obj']}"
                    for idx in range(args['train.n_mix'] + args['train.n_obj'])
                ]       # ['p0', 'p1', 'm0', 'm1']
                objs = np.array(objs_loss)     # [2, 4]
                figure = draw_objs(objs, pop_labels)
                writer.add_figure(f"image/train_objs_loss", figure, i+1)
                objs = np.array(objs_acc)     # [2, 4]
                figure = draw_objs(objs, pop_labels)
                writer.add_figure(f"image/train_objs_acc", figure, i+1)

                '''log loss and accuracy on all models on all clusters'''
                for model_name in model_names:
                    for cluster_name in cluster_names:
                        loss_values = epoch_loss[model_name][cluster_name]
                        acc_values = epoch_acc[model_name][cluster_name]
                        if len(loss_values) > 0:
                            writer.add_scalar(f"loss/train/{model_name}/{cluster_name}",
                                              np.mean(loss_values), i+1)
                            writer.add_scalar(f"accuracy/train/{model_name}/{cluster_name}",
                                              np.mean(acc_values), i+1)
                writer.add_scalar('loss/train/train_hv', np.mean(epoch_loss['hv_loss']), i+1)
                writer.add_scalar('loss/train/hv', np.mean(epoch_loss['hv']), i+1)
                writer.add_scalar('accuracy/train/hv', np.mean(epoch_acc['hv']), i+1)
                writer.add_scalar('learning_rate',
                                  optimizers[0].param_groups[0]['lr'], i+1)
                print(f"==>> loss/train/hv {np.mean(epoch_loss['hv']):.3f}, "
                      f"accuracy/train/hv {np.mean(epoch_acc['hv']):.3f}.")

                epoch_loss, epoch_acc = init_train_log()

                '''write pure and mixed tasks'''
                for task_id, task in enumerate(numpy_tasks):
                    imgs = np.concatenate([task['context_images'], task['target_images']])
                    writer.add_images(f"image/task-{task_id}", imgs, i+1)

            '''----------'''
            print(f"====>> Trained and evaluated at {i + 1}.\n")

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

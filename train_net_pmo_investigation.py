"""
This code allows you to train a domain classifier.

Author: Weiduo Liao
Date: 2023.07.31
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
from models.model_helpers import get_model, get_optimizer, get_model_moe
from utils import Accumulator, device, set_determ, check_dir
from config import args

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

        train_loaders = dict()
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                'train', [trainset], valsets, testsets, test_type=args['train.type'])
            num_train_classes[trainset] = train_loaders[trainset].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['test.type'])

        '''initialize models and optimizer'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0

        # pmo model load from url
        pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe

        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_film_parameters())
        optimizer_selector = torch.optim.Adam(pmo.get_trainable_selector_parameters(True),
                                              lr=args['train.selector_learning_rate'],
                                              weight_decay=args['train.selector_learning_rate'] / 50
                                              )
        checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                checkpointer.restore_model(ckpt='last', strict=False)
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
            lr_manager_selector = UniformStepLR(optimizer_selector, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
            lr_manager_selector = ExpDecayLR(optimizer_selector, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, 0)
            lr_manager_selector = CosineAnnealRestartLR(optimizer_selector, args, 0)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'],
                      num_sources=args['train.n_mix_source'], num_mixes=args['train.n_mix'])

        def model_train(mode=pmo):
            # train mode
            mode.train()
            if mode.feature_extractor is not None:
                mode.feature_extractor.eval()        # to extract task features
            pool.train()

        def model_eval(mode=pmo):
            # eval mode
            mode.eval()

        '''--------------------------------'''
        '''fill random pool and investigate'''
        '''--------------------------------'''
        '''obtain tasks from train_loaders and put to buffer'''
        # loading images and labels
        for t_indx, (name, train_loader) in enumerate(train_loaders.items()):
            sample = train_loader.get_train_task(session, d=device)

            context_images, target_images = sample['context_images'], sample['target_images']
            context_labels, target_labels = sample['context_labels'], sample['target_labels']
            context_gt_labels, target_gt_labels = sample['context_gt'], sample['target_gt']

            '''samples put to buffer'''
            task_images = torch.cat([context_images, target_images]).cpu()
            gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
            domain = np.array([t_indx] * len(gt_labels))        # [domain, domain, domain,...]
            with torch.no_grad():
                task_features = pmo.embed(torch.cat([context_images, target_images]))
                _, selection_info = pmo.selector(
                    task_features, gumbel=False, average=False)  # [bs, n_clusters]
                similarities = selection_info['y_soft'].cpu().numpy()  # [bs, n_clusters]

            not_full = pool.put_buffer(
                task_images, {'domain': domain, 'gt_labels': gt_labels,
                              'similarities': similarities, 'features': task_features.cpu().numpy()},
                maintain_size=False)

        '''buffer -> clusters'''
        print(f'Buffer contains {len(pool.buffer)} classes.')
        pool.clear_clusters()
        pool.buffer2cluster()
        pool.clear_buffer()

        debugger.write_pool(pool, i=0, writer=writer)

        '''multiple mo sampling'''
        pop_labels = [
            f"p{idx}" if idx < args['train.n_obj'] else f"m{idx - args['train.n_obj']}"
            for idx in range(args['train.n_mix'] + args['train.n_obj'])
        ]  # ['p0', 'p1', 'm0', 'm1']
        num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]
        mo_ncc_dict = {'acc': {}, 'loss': {}}  # acc/loss: {mo_idx: {pop_idx(4): {inner_idx: [2]}}}
        train_dict = {'acc': {'inner': {idx: [] for idx in range(args['train.n_obj'] + args['train.n_mix'])}},
                      'loss': {'inner': {idx: [] for idx in range(args['train.n_obj'] + args['train.n_mix'])}}}

        for mo_train_idx in range(args['train.n_mo']):
            mo_ncc_dict['acc'][mo_train_idx] = {}     # pop_idx(4): {inner_idx: [2]}
            mo_ncc_dict['loss'][mo_train_idx] = {}    # pop_idx(4): {inner_idx: [2]}
            '''check pool has enough samples and generate 1 setting'''
            n_way, n_shot, n_query = available_setting(num_imgs_clusters, args['train.mo_task_type'],
                                                       min_available_clusters=args['train.n_obj'])
            if n_way == -1:  # not enough samples
                print(f"==>> pool has not enough samples. skip MO")
                break

            available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

            selected_cluster_idxs = sorted(np.random.choice(
                available_cluster_idxs, args['train.n_obj'], replace=False))

            torch_tasks = []
            '''sample pure tasks from clusters in selected_cluster_idxs'''
            for cluster_idx in selected_cluster_idxs:
                pure_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
                torch_tasks.append(pure_task)

            '''sample mix tasks by mixer'''
            for mix_id in range(args['train.n_mix']):
                numpy_mix_task, _ = mixer.mix(
                    task_list=[pool.episodic_sample(idx, n_way, n_shot, n_query)
                               for idx in selected_cluster_idxs],
                    mix_id=mix_id
                )
                torch_tasks.append(task_to_device(numpy_mix_task, device))

            '''obtain ncc loss multi-obj matrix'''
            for task_idx, task in enumerate(torch_tasks):
                mo_ncc_dict['acc'][mo_train_idx][task_idx] = {}     # inner_idx: [2]
                mo_ncc_dict['loss'][mo_train_idx][task_idx] = {}    # inner_idx: [2]
                train_dict['acc']['inner'][task_idx].append([])
                train_dict['loss']['inner'][task_idx].append([])

                context_images = task['context_images']
                context_labels = task['context_labels']
                # target_images = task['target_images']
                # target_labels = task['target_labels']

                # debugger.write_task(pool, task, task_label, i=0, writer=writer)

                '''new a url with one film for inner update'''
                args_num_clusters1 = copy.deepcopy(args)
                args_num_clusters1['model.num_clusters'] = 1
                film_url = get_model_moe(None, args_num_clusters1, base_network_name='url')
                inner_lr = 1e-4
                optimizer_film_url = torch.optim.Adam(
                    film_url.get_trainable_film_parameters(), lr=inner_lr, weight_decay=inner_lr / 50)

                selection = torch.ones(1, 1).to(device)
                model_train(film_url)
                for inner_idx in range(5 + 1):      # 0 is before inner loop
                    mo_ncc_dict['acc'][mo_train_idx][task_idx][inner_idx] = []     # [2]
                    mo_ncc_dict['loss'][mo_train_idx][task_idx][inner_idx] = []    # [2]

                    '''forward with no grad for mo matrix'''
                    for obj_idx in range(len(selected_cluster_idxs)):       # 2
                        obj_context_images = torch_tasks[obj_idx]['context_images']
                        obj_target_images = torch_tasks[obj_idx]['target_images']
                        obj_context_labels = torch_tasks[obj_idx]['context_labels']
                        obj_target_labels = torch_tasks[obj_idx]['target_labels']

                        with torch.no_grad():
                            model_eval(film_url)
                            obj_context_features = film_url.embed(obj_context_images, selection=selection)
                            obj_target_features = film_url.embed(obj_target_images, selection=selection)
                            model_train(film_url)

                        _, stats_dict, _ = prototype_loss(
                            obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
                            distance=args['test.distance'])
                        mo_ncc_dict['acc'][mo_train_idx][task_idx][inner_idx].append(stats_dict['acc'])
                        mo_ncc_dict['loss'][mo_train_idx][task_idx][inner_idx].append(stats_dict['loss'])

                    '''inner update using context set'''
                    context_features = film_url.embed(context_images, selection=selection)
                    loss, stats_dict, _ = prototype_loss(
                        context_features, context_labels, context_features, context_labels,
                        distance=args['test.distance'])

                    optimizer_film_url.zero_grad()
                    loss.backward()
                    optimizer_film_url.step()

                    train_dict['acc']['inner'][task_idx][-1].append(stats_dict['acc'])
                    train_dict['loss']['inner'][task_idx][-1].append(stats_dict['loss'])

                    '''write inner loss/acc for 4 tasks averaging over multiple mo sampling'''
                    debugger.write_scale(stats_dict['acc'], f'inner_acc/taskid{task_idx}',
                                         i=inner_idx, writer=writer)
                    debugger.write_scale(stats_dict['loss'], f'inner_loss/taskid{task_idx}',
                                         i=inner_idx, writer=writer)

            '''write mo image'''
            debugger.write_mo(mo_ncc_dict['acc'][mo_train_idx], pop_labels, i=mo_train_idx, writer=writer, prefix='acc')
            debugger.write_mo(mo_ncc_dict['loss'][mo_train_idx], pop_labels, i=mo_train_idx, writer=writer, prefix='loss')


if __name__ == '__main__':
    train()

    '''nvidia-smi'''
    print(os.system('nvidia-smi'))
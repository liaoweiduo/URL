"""
This code allows you to train a domain classifier.

Author: Weiduo Liao
Date: 2023.07.31
"""

import os
import sys
import pickle
import copy

import pandas as pd
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
from models.pa import apply_selection, pa
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

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'], max_num_classes=20)

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'],
                      num_sources=args['train.n_mix_source'], num_mixes=args['train.n_mix'])

        def model_train(mode):
            # train mode
            mode.train()
            if mode.feature_extractor is not None:
                mode.feature_extractor.eval()        # to extract task features
            pool.train()

        def model_eval(mode):
            # eval mode
            mode.eval()

        url = get_model(None, args, base_network_name='url', freeze_fe=True)
        # pmo model load from url
        pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe
        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_film_parameters())
        optimizer_selector = torch.optim.Adam(pmo.get_trainable_selector_parameters(True),
                                              lr=args['train.selector_learning_rate'],
                                              weight_decay=args['train.selector_learning_rate'] / 50)

        '''load many exps to check their clustering's hv'''
        exp = 'pmo-domain_selector-lr0_0001'
        # for exp in ['pmo-inner0_01-1-tune-lr5e-06-gumbelTrue-hvc1',
        #             'pmo-mov_avg-tkcph-gumbelFalse-clce1-kdkernelcka1-pc1-hvc1',    # OOD 70
        #             'pmo-mov_avg-tkcph-gumbelTrue-clce1-kdkernelcka1-pc0-hvc0']:
        args_temp = copy.deepcopy(args)
        args_temp['model.dir'] = '../URL-experiments/saved_results/' + exp

        checkpointer = CheckPointer(args_temp, pmo, optimizer=optimizer, save_all=True)
        checkpointer.restore_model(ckpt='best', strict=False)       # load selector

        # model_train(model)
        model_eval(pmo)
        model_eval(url)

        mo_ncc_df = pd.DataFrame(columns=['Type', 'Pop_id', 'Obj_id', 'Inner_id',
                                          'Inner_lr', 'Exp', 'Logit_scale',
                                          'Value'])
        # Type: ['acc', 'loss']
        train_df = pd.DataFrame(columns=['Type', 'Tag', 'Task_id', 'Idx',
                                         'Inner_lr', 'Exp', 'Logit_scale',
                                         'Value'])

        pool.clear_clusters()
        pool.clear_buffer()
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
            domain = np.array([t_indx] * len(gt_labels))  # [domain, domain, domain,...]
            with torch.no_grad():
                task_features = pmo.embed(torch.cat([context_images, target_images]))
                # _, selection_info = pmo.selector(
                #     task_features, gumbel=True, hard=True, average=False,
                #     logit_scale=logit_scale)  # [bs, n_clusters]
                # similarities = selection_info['y_soft'].cpu().numpy()  # [bs, n_clusters]
            similarities = np.array([0] * len(gt_labels))  # no use

            not_full = pool.put_buffer(
                task_images, {'domain': domain, 'gt_labels': gt_labels,
                              'similarities': similarities, 'features': task_features.cpu().numpy()},
                maintain_size=False)
        print(f'Buffer contains {len(pool.buffer)} classes.')
        pool.buffer_backup = copy.deepcopy(pool.buffer)

        for logit_scale in [-10, -1, 0, 1]:
            print(f'logit_scale: {logit_scale}')
            for pool_idx in range(10):      # try different gumbel randomness
                print(f'pool construction idx: {pool_idx}')

                pool.buffer = copy.deepcopy(pool.buffer_backup)

                '''cal similarities for specific logit_scale (gumbel)'''
                for cls in pool.buffer:
                    # cal sim from stored features
                    features = torch.from_numpy(cls['features']).to(device)
                    # images = torch.from_numpy(cls['images'])

                    with torch.no_grad():
                        # features = pmo.embed(images.to(device))
                        _, selection_info = pmo.selector(
                            features, gumbel=True, average=False, logit_scale=logit_scale)  # [bs, n_clusters]
                        similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                    cls['similarities'] = similarities

                '''buffer -> clusters'''
                print(f'Buffer contains {len(pool.buffer)} classes.')
                pool.clear_clusters()
                pool.buffer2cluster()
                pool.clear_buffer()

                '''multiple mo sampling'''
                pop_labels = [
                    f"p{idx}" if idx < args['train.n_obj'] else f"m{idx - args['train.n_obj']}"
                    for idx in range(args['train.n_mix'] + args['train.n_obj'])
                ]  # ['p0', 'p1', 'm0', 'm1']
                num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]
                # Tag: ['inner'], Task_id: 0,1,2,3
                for mo_train_idx in range(args['train.n_mo']):
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
                        # context_images = task['context_images']
                        # context_labels = task['context_labels']
                        # target_images = task['target_images']
                        # target_labels = task['target_labels']

                        # debugger.write_task(pool, task, task_label, i=0, writer=writer)

                        '''use url with pa'''
                        model = url
                        with torch.no_grad():
                            context_features = model.embed(task['context_images'])
                            context_labels = task['context_labels']

                        inner_lr = 1
                        '''new a url with one film for inner update'''
                        # args_num_clusters1 = copy.deepcopy(args)
                        # args_num_clusters1['model.num_clusters'] = 1
                        # model = get_model_moe(None, args_num_clusters1, base_network_name='url')
                        # # inner_lr = args['train.inner_learning_rate']
                        # # optimizer_model = torch.optim.Adam(
                        # #     model.get_trainable_film_parameters(), lr=inner_lr, weight_decay=inner_lr / 50)
                        # optimizer_model = torch.optim.Adadelta(model.get_trainable_film_parameters(), lr=inner_lr)
                        # selection = torch.ones(1, 1).to(device)

                        for inner_idx, selection_params in enumerate(
                                pa(context_features, context_labels, max_iter=40, lr=inner_lr,
                                   distance=args['test.distance'], return_iterator=True)):
                            # '''record at certain iter'''
                            # if
                            '''inner acc/loss'''
                            with torch.no_grad():
                                selected_context = apply_selection(context_features, selection_params)
                            _, stats_dict, _ = prototype_loss(
                                selected_context, context_labels,
                                selected_context, context_labels, distance=args['test.distance'])
                            train_df = train_df.append({
                                'Tag': 'inner', 'Task_id': task_idx, 'Idx': inner_idx,
                                'Inner_lr': inner_lr, 'Exp': exp, 'Logit_scale': logit_scale,
                                'Type': 'acc', 'Value': stats_dict['acc']}, ignore_index=True)
                            train_df = train_df.append({
                                'Tag': 'inner', 'Task_id': task_idx, 'Idx': inner_idx,
                                'Inner_lr': inner_lr, 'Exp': exp, 'Logit_scale': logit_scale,
                                'Type': 'loss', 'Value': stats_dict['loss']}, ignore_index=True)

                            '''forward with no grad for mo matrix'''
                            for obj_idx in range(len(selected_cluster_idxs)):       # 2
                                obj_context_images = torch_tasks[obj_idx]['context_images']
                                obj_target_images = torch_tasks[obj_idx]['target_images']
                                obj_context_labels = torch_tasks[obj_idx]['context_labels']
                                obj_target_labels = torch_tasks[obj_idx]['target_labels']

                                with torch.no_grad():
                                    obj_context_features = apply_selection(model.embed(obj_context_images),
                                                                           selection_params)
                                    obj_target_features = apply_selection(model.embed(obj_target_images),
                                                                          selection_params)

                                _, stats_dict, _ = prototype_loss(
                                    obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
                                    distance=args['test.distance'])

                                mo_ncc_df = mo_ncc_df.append({
                                    'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx,
                                    'Inner_lr': inner_lr, 'Exp': exp, 'Logit_scale': logit_scale,
                                    'Type': 'acc', 'Value': stats_dict['acc']}, ignore_index=True)
                                mo_ncc_df = mo_ncc_df.append({
                                    'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx,
                                    'Inner_lr': inner_lr, 'Exp': exp, 'Logit_scale': logit_scale,
                                    'Type': 'loss', 'Value': stats_dict['loss']}, ignore_index=True)

            debugger.write_pool(pool, i=0, writer=writer,
                                prefix=f'pool_logit_scale{logit_scale}')    # only record last pool

            # for inner_idx in range(5 + 1):      # 0 is before inner loop
            #
            #     '''forward with no grad for mo matrix'''
            #     for obj_idx in range(len(selected_cluster_idxs)):       # 2
            #         obj_context_images = torch_tasks[obj_idx]['context_images']
            #         obj_target_images = torch_tasks[obj_idx]['target_images']
            #         obj_context_labels = torch_tasks[obj_idx]['context_labels']
            #         obj_target_labels = torch_tasks[obj_idx]['target_labels']
            #
            #         with torch.no_grad():
            #             model_eval(model)
            #             obj_context_features = model.embed(obj_context_images, selection=selection)
            #             obj_target_features = model.embed(obj_target_images, selection=selection)
            #             model_train(model)
            #
            #         _, stats_dict, _ = prototype_loss(
            #             obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
            #             distance=args['test.distance'])
            #         mo_ncc_df = mo_ncc_df.append({
            #             'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx, 'Inner_lr': inner_lr,
            #             'Type': 'acc', 'Value': stats_dict['acc']}, ignore_index=True)
            #         mo_ncc_df = mo_ncc_df.append({
            #             'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx, 'Inner_lr': inner_lr,
            #             'Type': 'loss', 'Value': stats_dict['loss']}, ignore_index=True)
            #
            #     '''inner update using context set'''
            #     context_features = model.embed(context_images, selection=selection)
            #     loss, stats_dict, _ = prototype_loss(
            #         context_features, context_labels, context_features, context_labels,
            #         distance=args['test.distance'])
            #
            #     optimizer_model.zero_grad()
            #     loss.backward()
            #     optimizer_model.step()
            #
            #     train_df = train_df.append({
            #         'Tag': 'inner', 'Task_id': task_idx, 'Idx': inner_idx, 'Inner_lr': inner_lr,
            #         'Type': 'acc', 'Value': stats_dict['acc']}, ignore_index=True)
            #     train_df = train_df.append({
            #         'Tag': 'inner', 'Task_id': task_idx, 'Idx': inner_idx, 'Inner_lr': inner_lr,
            #         'Type': 'loss', 'Value': stats_dict['loss']}, ignore_index=True)

        '''save'''
        debugger.save_df(mo_ncc_df, writer=writer, name=f'mo_dict.json')
        '''write mo image'''
        debugger.write_mo(mo_ncc_df, pop_labels, i=0, writer=writer, target='acc')
        debugger.write_mo(mo_ncc_df, pop_labels, i=0, writer=writer, target='loss')
        '''write hv acc/loss'''
        debugger.write_hv(mo_ncc_df, ref='relative', writer=writer, target='acc')     # 0
        debugger.write_hv(mo_ncc_df, ref='relative', writer=writer, target='loss')    # args['train.ref']
        '''write avg_span acc/loss: E_i(max(f_i) - min(f_i))'''
        debugger.write_avg_span(mo_ncc_df, writer=writer, target='acc')
        debugger.write_avg_span(mo_ncc_df, writer=writer, target='loss')

        '''write inner loss/acc for 4 tasks averaging over multiple mo sampling(and inner lr settings)'''
        for inner_idx in range(len(set(train_df.Idx))):
            for task_idx in range(len(set(train_df.Task_id))):
                t_df = train_df[(train_df.Task_id == task_idx) & (train_df.Idx == inner_idx) &
                                (train_df.Tag == 'inner')]
                debugger.write_scale(t_df[t_df.Type == 'acc'].Value.mean(),
                                     f'inner_acc_{exp}/taskid{task_idx}',
                                     i=inner_idx, writer=writer)
                debugger.write_scale(t_df[t_df.Type == 'loss'].Value.mean(),
                                     f'inner_loss_{exp}/taskid{task_idx}',
                                     i=inner_idx, writer=writer)


if __name__ == '__main__':
    train()

    '''nvidia-smi'''
    print(os.system('nvidia-smi'))

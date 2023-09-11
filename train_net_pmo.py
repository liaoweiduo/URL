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
from models.model_helpers import get_model, get_model_moe, get_optimizer
from utils import Accumulator, device, set_determ, check_dir
from config import args

from pmo_utils import (Pool, Mixer,
                       cal_hv_loss, cal_hv, draw_objs, draw_heatmap, available_setting, check_available, task_to_device)

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
        assert (device != 'cpu'), f'device is cpu'

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

        # pmo model, fe load from url
        pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe

        optimizer = get_optimizer(pmo, args, params=pmo.get_trainable_film_parameters())    # for films
        optimizer_selector = torch.optim.Adam(pmo.get_trainable_selector_parameters(True),
                                              lr=args['train.selector_learning_rate'],
                                              weight_decay=args['train.selector_learning_rate'] / 50
                                              )
        # optimizer_selector = torch.optim.Adadelta(pmo.get_trainable_selector_parameters(True),
        #                                           lr=args['train.selector_learning_rate'])
        checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
        if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            lr_manager = UniformStepLR(optimizer, args, start_iter)
            lr_manager_selector = UniformStepLR(optimizer_selector, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            lr_manager = ExpDecayLR(optimizer, args, start_iter)
            lr_manager_selector = ExpDecayLR(optimizer_selector, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            lr_manager = CosineAnnealRestartLR(optimizer, args, start_iter)
            lr_manager_selector = CosineAnnealRestartLR(optimizer_selector, args, start_iter)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
        center_pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'],
                           max_num_classes=5)       # visualize history class closest to cluster centers
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
            epoch_loss[f'task/gumbel_sim'] = []
            epoch_loss[f'task/softmax_sim'] = []
            # epoch_loss[f'task/selection_ce_loss'] = []
            epoch_loss[f'pool/selection_ce_loss'] = []
            epoch_loss[f'pure/selection_ce_loss'] = []
            epoch_loss.update({f'task/{name}': [] for name in trainsets})
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
            pool.train()

        def model_eval():
            # eval mode
            pmo.eval()
            pool.eval()

        def zero_grad():
            optimizer.zero_grad()
            optimizer_selector.zero_grad()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.zero_grad()

        def update_step(idx):
            optimizer.step()
            optimizer_selector.step()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.step()

            lr_manager.step(idx)
            lr_manager_selector.step(idx)
            if args['train.cluster_center_mode'] == 'trainable':
                pool.lr_manager.step(idx)

        epoch_loss, epoch_acc = init_train_log()
        epoch_val_loss = {}
        epoch_val_acc = {}

        print(f'\n>>>> Train start from {start_iter}.')
        verbose = True
        for i in tqdm(range(start_iter, max_iter), ncols=100):

            zero_grad()
            model_train()

            '''iteratively obtain tasks from train_loaders'''
            p = np.ones(len(trainsets))
            if 'ilsvrc_2012' in trainsets:
                p[trainsets.index('ilsvrc_2012')] = 2.0
            p = p / sum(p)
            # while True:
            t_indx = np.random.choice(len(trainsets), p=p)
            trainset = trainsets[t_indx]

            samples = train_loaders[trainset].get_train_task(session, d=device)
            context_images, target_images = samples['context_images'], samples['target_images']
            context_labels, target_labels = samples['context_labels'], samples['target_labels']
            context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
            domain = t_indx

            '''samples put to buffer'''
            task_images = torch.cat([context_images, target_images]).cpu()
            gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
            domain = np.array([domain] * len(gt_labels))
            similarities = np.array([0] * len(gt_labels))       # no use
            not_full = pool.put_buffer(
                task_images, {'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})



            # todo: check label keeps the same
            pool.buffer_during_sampling = copy.deepcopy(pool.buffer)
            if not_full:        # have put into the buffer
                for pre_img_idx, pre_img in enumerate(task_images.numpy()):
                    pre_gt_label = gt_labels[pre_img_idx]
                    pre_domain = domain[pre_img_idx]
                    pre_sim = similarities[pre_img_idx]
                    checked = False
                    for cls in pool.buffer:
                        for post_image_idx, post_image in enumerate(cls['images']):
                            sim = cls['similarities'][post_image_idx]
                            label = cls['label']
                            if (pre_img == post_image).all():
                                checked = True
                                assert ((label[0] == pre_gt_label
                                         ) and label[1] == pre_domain and (
                                        sim == pre_sim).all()
                                        ), f'put_during sampling: ' \
                                           f'incorrect info: gt_label {label[0]} vs {pre_gt_label}, ' \
                                           f'domain {label[1]} vs {pre_domain}, ' \
                                           f'sim {sim} vs {pre_sim}.'
                    assert checked, f'no img find in buffer.'





            if not not_full and verbose:  # full buffer
                print(f'Buffer is full at iter: {i}.')
                verbose = False
                # print(f'Buffer is full num classes in buffer: {len(pool.buffer)}..')
            # if not not_full:    # enough sampling
            #     break

            # need to check how many classes in 1 samples and need a buffer size
            # about 10 iters can obtain 200 classes
            # print(f'num classes in buffer: {len(pool.buffer)}.')

            # '''fill pool from train_loaders'''
            # for t in tqdm(range(args['train.max_sampling_iter_for_pool']), ncols=100):
            #     for t_indx, trainset in enumerate(trainsets):
            #         num_task_per_batch = 1 if trainset != 'ilsvrc_2012' else 2
            #         for _ in range(num_task_per_batch):
            #             samples = train_loaders[trainset].get_train_task(session, d='cpu')
            #             images = torch.cat([samples['context_images'], samples['target_images']])
            #             # re_labels = torch.cat([samples['context_labels'], samples['target_labels']]).numpy()
            #             gt_labels = torch.cat([samples['context_gt'], samples['target_gt']]).numpy()
            #             domain = np.array([t_indx] * len(gt_labels))
            #
            #             # put in sequence
            #             # '''obtain selection vec for images'''
            #             # with torch.no_grad():
            #             #     _, selection_info = pmo.selector(
            #             #         pmo.embed(images.to(device)), gumbel=True)  # [bs, n_clusters]
            #             #     similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]
            #             #     cluster_idxs = np.argmax(similarities, axis=1)  # [bs]
            #             #     similarities = selection_info['normal_soft'].detach().cpu().numpy()
            #             #     # using gumbel to determine which cluster to put, but similarity use normal softmax
            #             #
            #             # pool.put_batch(
            #             #     images, cluster_idxs, {
            #             #         'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})
            #
            #             # put to buffer then put to cluster
            #             '''obtain selection vec for images'''
            #             with torch.no_grad():
            #                 _, selection_info = pmo.selector(
            #                     pmo.embed(images.to(device)), gumbel=False)  # [bs, n_clusters]
            #                 similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]
            #
            #             pool.put_buffer(images, {
            #                 'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})

            '''maintain pool'''
            if (i + 1) % args['train.pool_freq'] == 0:
                print(f"\n>> Iter: {i + 1}, update pool: ")

                '''collect samples in the buffer'''
                pool.clear_clusters()       # do not need last iter's center
                buffer_samples = [cls for cls in pool.buffer]

                verbose = True
                if verbose:
                    print(f'Buffer contains {len(buffer_samples)} classes.')

                '''re-cal sim and re-put samples into pool buffer'''
                pool.clear_buffer()
                if len(buffer_samples) > 0:
                    images = torch.from_numpy(np.concatenate([cls['images'] for cls in buffer_samples]))
                    gt_labels = np.array([cls['label'][0] for cls in buffer_samples for img in cls['images']])
                    domain = np.array([cls['label'][1] for cls in buffer_samples for img in cls['images']])

                    # need to check num of images, maybe need to reshape to batch to calculate
                    # less than 1w images for 200 classes
                    # print(f'num images in buffer (cal sim): {len(images)}.')

                    with torch.no_grad():
                        _, selection_info = pmo.selector(
                            pmo.embed(images.to(device)), gumbel=False, average=False)  # [bs, n_clusters]
                        similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                    # ignore buffer size and put into buffer
                    pool.put_buffer(
                        images, {'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities},
                        maintain_size=False)

                    # todo: check sim keeps the same
                    pool.buffer_copy = copy.deepcopy(pool.buffer)
                    for cls in pool.buffer:
                        for image_idx, image in enumerate(cls['images']):
                            sim = cls['similarities'][image_idx]
                            label = cls['label']
                            for img_idx, img in enumerate(images.numpy()):
                                if (image == img).all():
                                    assert ((label[0] == gt_labels[img_idx]
                                             ) and label[1] == domain[img_idx] and (
                                            sim == similarities[img_idx]).all()
                                            ), f'incorrect info: gt_label {label[0]} vs {gt_labels[img_idx]}, ' \
                                               f'domain {label[1]} vs {domain[img_idx]}, ' \
                                               f'sim {sim} vs {similarities[img_idx]}.'


                '''collect cluster'''
                current_clusters = center_pool.clear_clusters()
                current_clusters = [cls for clses in current_clusters for cls in clses]       # cat all clusters

                '''re-cal sim and re-put samples into center pool's buffer'''
                center_pool.clear_buffer()
                if len(current_clusters) > 0:
                    current_images = torch.from_numpy(np.concatenate([cls['images'] for cls in current_clusters]))
                    current_gt_labels = np.array([cls['label'][0] for cls in current_clusters for img in cls['images']])
                    current_domain = np.array([cls['label'][1] for cls in current_clusters for img in cls['images']])

                    with torch.no_grad():
                        _, selection_info = pmo.selector(
                            pmo.embed(current_images.to(device)), gumbel=False, average=False)  # [bs, n_clusters]
                        current_similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                    '''cat into pool's buffer samples'''
                    images = torch.cat([images, current_images])
                    domain = np.concatenate([domain, current_domain])
                    gt_labels = np.concatenate([gt_labels, current_gt_labels])
                    similarities = np.concatenate([similarities, current_similarities])

                if len(buffer_samples) > 0:
                    # ignore buffer size and put into buffer
                    center_pool.put_buffer(
                        images, {'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities},
                        maintain_size=False)

                '''buffer -> clusters'''
                pool.buffer2cluster()
                pool.clear_buffer()
                center_pool.buffer2cluster()
                center_pool.clear_buffer()


                '''check pool image similarity'''
                # todo: for debug, remove
                if (i + 1) % args['train.mo_freq'] == 0:    # only draw every 200 iter

                    '''write image similarities in the pool'''
                    similarities = pool.current_similarities(image_wise=True)
                    for cluster_id, cluster in enumerate(similarities):
                        if len(cluster) > 0:
                            sim_in_cluster = np.concatenate(cluster)  # [num_cls*num_img, 8]
                            figure = draw_heatmap(sim_in_cluster, verbose=False)
                            writer.add_figure(f"pool-img-sim-in-the-pool/{cluster_id}", figure, i + 1)

                    '''write re-called image similarities in the pool'''
                    numpy_images = pool.current_images()
                    for cluster_idx, cluster in enumerate(numpy_images):
                        if len(cluster) > 0:
                            image_batch = torch.from_numpy(
                                np.concatenate(cluster)
                            ).to(device)
                            with torch.no_grad():
                                img_features = pmo.embed(image_batch)    # [img_size, 512]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                                img_sim = selection_info['y_soft']        # [img_size, 10]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                                tsk_sim = selection_info['y_soft']        # [1, 10]
                            sim = torch.cat([img_sim, *[tsk_sim]*(img_sim.shape[0]//10)]).cpu().numpy()
                            figure = draw_heatmap(sim, verbose=False)
                            writer.add_figure(f"pool-img-sim-re-cal-after-buffer2cluster/{cluster_idx}", figure, i+1)

                    cases = []
                    for _ in range(100):
                        # todo: track a specific image sample
                        anchor_cluster_index = np.random.choice(len(pool.clusters))
                        anchor_cls_index = np.random.choice(len(pool.clusters[anchor_cluster_index]))
                        anchor_img_index = np.random.choice(
                            len(pool.clusters[anchor_cluster_index][anchor_cls_index]['images']))
                        anchor_img = pool.clusters[anchor_cluster_index][anchor_cls_index]['images'][anchor_img_index]
                        anchor_label = pool.clusters[anchor_cluster_index][anchor_cls_index]['label']
                        anchor_sim = pool.clusters[anchor_cluster_index][anchor_cls_index]['similarities'][
                            anchor_img_index]
                        # print(f'debug: anchor img shape: {anchor_img.shape}, '
                        #       f'label: {anchor_label}, '
                        #       f'\nsim: {anchor_sim}. ')

                        # todo: track a specific image sample
                        found = False
                        correct = False
                        for cls in pool.buffer_copy:
                            if (cls['label'] == anchor_label).all():
                                for i, img in enumerate(cls['images']):
                                    if (img == anchor_img).all():
                                        found = True
                                        found_sim = cls['similarities'][i]
                                        # print(f'debug: find anchor img in the buffer with \nsim: {found_sim}.')
                                        # assert (found_sim == anchor_sim).all(), f'debug: sim does not match.'
                                        if (found_sim == anchor_sim).all():
                                            correct = True

                        cases.append([found, correct])

                    assert (np.array(cases).sum(0) == np.array([100, 100])).all(), f'{np.array(cases).sum(0)}'




                '''selection CE loss on all clusters'''
                if 'ce' in args['train.loss_type']:
                    numpy_images = pool.current_images()
                    '''random select a batch of samples'''
                    # batch_size_each_cluster = 50        # bs = 50*10
                    image_batch, cluster_labels = [], []
                    for cluster_idx, cluster in enumerate(numpy_images):
                        if len(cluster) > 0:
                            imgs = np.concatenate(cluster)
                            image_batch.append(imgs)   # np
                            cluster_labels.append([cluster_idx] * imgs.shape[0])
                            # select_idxs = np.random.permutation(len(imgs))[:batch_size_each_cluster]
                            # image_batch.append(imgs[select_idxs])   # np
                            # cluster_labels.append([cluster_idx] * len(select_idxs))
                    image_batch = torch.from_numpy(np.concatenate(image_batch)).to(device)
                    cluster_labels = torch.from_numpy(np.concatenate(cluster_labels)).long().to(device)
                    # image_batch = torch.from_numpy(
                    #     np.concatenate([np.concatenate(cluster) for cluster in numpy_images if len(cluster) > 0])
                    # ).to(device)
                    # cluster_labels = torch.from_numpy(
                    #     np.array([
                    #         cluster_idx
                    #         for cluster_idx, cluster in enumerate(numpy_images)
                    #         for cls in cluster
                    #         for _ in range(cls.shape[0])])
                    # ).long().to(device)
                    _, selection_info = pmo.selector(pmo.embed(image_batch), gumbel=False, average=False)
                    fn = torch.nn.CrossEntropyLoss()
                    y_soft = selection_info['y_soft']  # [img_size, 8]
                    # select_idx = selected_cluster_idxs[task_idx]
                    # labels = torch.ones(
                    #     (y_soft.shape[0],), dtype=torch.long, device=y_soft.device) * select_idx
                    selection_ce_loss = fn(y_soft, cluster_labels)

                    '''log ce loss'''
                    epoch_loss[f'pool/selection_ce_loss'].append(selection_ce_loss.item())

                    '''ce loss coefficient'''
                    # selection_ce_loss = selection_ce_loss * 1000
                    selection_ce_loss.backward()

                    ''''''
                    '''pure task selection CE loss on all clusters'''
                    num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]
                    for cluster_idx in range(len(num_imgs_clusters)):
                        n_way, n_shot, n_query = available_setting([num_imgs_clusters[cluster_idx]],
                                                                   args['train.type'])
                        if n_way == -1:
                            continue    # not enough samples to construct a task
                        else:
                            pure_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
                            _, selection_info = pmo.selector(
                                pmo.embed(
                                    torch.cat([pure_task['context_images'], pure_task['target_images']])),
                                gumbel=False, average=True)
                            y_soft = selection_info['y_soft']  # [1, 8]
                            labels = torch.ones(
                                (y_soft.shape[0],), dtype=torch.long, device=y_soft.device) * cluster_idx
                            selection_ce_loss = fn(y_soft, labels)

                            '''log pure ce loss'''
                            epoch_loss[f'pure/selection_ce_loss'].append(selection_ce_loss.item())

                            '''ce loss to average'''
                            selection_ce_loss = selection_ce_loss / len(num_imgs_clusters)
                            # selection_ce_loss = selection_ce_loss * 1000
                            selection_ce_loss.backward()

            '''----------------'''
            '''Task Train Phase'''
            '''----------------'''
            if 'task' in args['train.loss_type']:
                [enriched_context_features, enriched_target_features], selection_info = pmo(
                    [context_images, target_images], torch.cat([context_images, target_images]),
                    gumbel=True, hard=False)
                # task_cluster_idx = torch.argmax(selection_info['y_soft'], dim=1).squeeze()
                # # supervision to be softmax for CE loss
            else:
                with torch.no_grad():
                    [enriched_context_features, enriched_target_features], selection_info = pmo(
                        [context_images, target_images], torch.cat([context_images, target_images]),
                        gumbel=True, hard=False)

            task_loss, stats_dict, _ = prototype_loss(
                enriched_context_features, context_labels,
                enriched_target_features, target_labels,
                distance=args['test.distance'])

            if 'task' in args['train.loss_type']:
                task_loss.backward()

            '''log task loss and acc'''
            epoch_loss[f'task/{trainset}'].append(stats_dict['loss'])
            epoch_acc[f'task/{trainset}'].append(stats_dict['acc'])
            # ilsvrc_2012 has 2 times larger len than others.

            '''log task sim (softmax and gumbel)'''
            epoch_loss[f'task/gumbel_sim'].append(selection_info['y_soft'].detach().cpu().numpy())    # [1,8]
            epoch_loss[f'task/softmax_sim'].append(selection_info['normal_soft'].detach().cpu().numpy())

            '''log img sim in the task'''
            with torch.no_grad():
                img_features = pmo.embed(torch.cat([context_images, target_images]))  # [img_size, 512]
                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                img_sim = selection_info['y_soft']  # [img_size, 10]
                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                tsk_sim = selection_info['y_soft']  # [1, 10]
            sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
            epoch_loss[f'task/image_softmax_sim'] = sim

            # '''selection CE loss on training task'''
            # if 'ce' in args['train.loss_type']:
            #     image_batch = torch.cat([context_images, target_images])
            #     cluster_labels = torch.ones_like(
            #         torch.cat([context_labels, target_labels])).long() * task_cluster_idx
            #     _, selection_info = pmo.selector(pmo.embed(image_batch), gumbel=False)
            #     fn = torch.nn.CrossEntropyLoss()
            #     y_soft = selection_info['y_soft']  # [img_size, 8]
            #     # select_idx = selected_cluster_idxs[task_idx]
            #     # labels = torch.ones(
            #     #     (y_soft.shape[0],), dtype=torch.long, device=y_soft.device) * select_idx
            #     selection_ce_loss = fn(y_soft, cluster_labels)
            #
            #     '''ce loss coefficient'''
            #     # selection_ce_loss = selection_ce_loss * args['train.task_ce_coefficient']
            #     selection_ce_loss.backward()
            #
            #     '''log ce loss'''
            #     epoch_loss[f'task/selection_ce_loss'].append(selection_ce_loss.item())

            '''----------------'''
            '''MO Train Phase  '''
            '''----------------'''
            if (i + 1) % args['train.mo_freq'] == 0:
                print(f"\n>> Iter: {i + 1}, MO phase: "
                      f"({'train' if 'hv' in args['train.loss_type'] else 'eval'})")

                num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]

                '''pure loss on all clusters'''
                if 'pure' in args['train.loss_type']:
                    epoch_loss[f'pure/task_softmax_sim'] = []
                    epoch_loss[f'pure/task_dist'] = []
                    epoch_loss[f'pure/image_softmax_sim'] = {}
                    pure_task_images = {}
                    for cluster_idx in range(len(num_imgs_clusters)):
                        n_way, n_shot, n_query = available_setting([num_imgs_clusters[cluster_idx]],
                                                                   args['train.type'])
                        if n_way == -1:
                            continue    # not enough samples to construct a task
                        else:
                            pure_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
                            context_images, target_images = pure_task['context_images'], pure_task['target_images']
                            context_labels, target_labels = pure_task['context_labels'], pure_task['target_labels']
                            pure_task_images[cluster_idx] = torch.cat([context_images, target_images]).cpu()

                            [enriched_context_features, enriched_target_features], selection_info = pmo(
                                [context_images, target_images], torch.cat([context_images, target_images]),
                                gumbel=False, hard=True)

                            pure_loss, stats_dict, _ = prototype_loss(
                                enriched_context_features, context_labels,
                                enriched_target_features, target_labels,
                                distance=args['test.distance'])

                            '''pure_loss to average'''
                            pure_loss = pure_loss / len(num_imgs_clusters)
                            pure_loss.backward()

                            '''log pure loss'''
                            epoch_loss[f'pure/C{cluster_idx}'].append(stats_dict['loss'])
                            epoch_acc[f'pure/C{cluster_idx}'].append(stats_dict['acc'])

                            '''log pure selection info for pure task'''
                            epoch_loss[f'pure/task_softmax_sim'].append(
                                selection_info['normal_soft'].detach().cpu().numpy())   # [1, 10]
                            epoch_loss[f'pure/task_dist'].append(
                                selection_info['dist'].detach().cpu().numpy())   # [1, 10]

                            '''log pure img sim'''
                            with torch.no_grad():
                                img_features = pmo.embed(torch.cat([context_images, target_images]))  # [img_size, 512]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                                img_sim = selection_info['y_soft']  # [img_size, 10]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                                tsk_sim = selection_info['y_soft']  # [1, 10]
                            sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
                            epoch_loss[f'pure/image_softmax_sim'][cluster_idx] = sim



                '''check pool image similarity'''
                # todo: for debug, remove
                numpy_images = pool.current_images()
                for cluster_idx, cluster in enumerate(numpy_images):
                    if len(cluster) > 0:
                        image_batch = torch.from_numpy(
                            np.concatenate(cluster)
                        ).to(device)
                        with torch.no_grad():
                            img_features = pmo.embed(image_batch)    # [img_size, 512]
                            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                            img_sim = selection_info['y_soft']        # [img_size, 10]
                            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                            tsk_sim = selection_info['y_soft']        # [1, 10]
                        sim = torch.cat([img_sim, *[tsk_sim]*(img_sim.shape[0]//10)]).cpu().numpy()
                        figure = draw_heatmap(sim, verbose=False)
                        writer.add_figure(f"pool-img-sim-re-cal-before-update/{cluster_idx}", figure, i+1)



                '''repeat collecting MO loss'''
                for mo_train_idx in range(args['train.n_mo']):
                    '''check pool has enough samples and generate 1 setting'''
                    n_way, n_shot, n_query = available_setting(num_imgs_clusters, args['train.mo_task_type'],
                                                               min_available_clusters=args['train.n_obj'])
                    if n_way == -1:         # not enough samples
                        print(f"==>> pool has not enough samples. skip MO training")
                        break
                    else:
                        available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

                        selected_cluster_idxs = sorted(np.random.choice(
                            available_cluster_idxs, args['train.n_obj'], replace=False))
                        # which is also devices idx
                        # device_list = list(set([devices[idx] for idx in selected_cluster_idxs]))    # unique devices

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
                        ncc_losses_multi_obj = []  # [4, 2]

                        for task_idx, task in enumerate(torch_tasks):
                            '''obtain task-specific selection'''
                            gumbel = False
                            hard = task_idx < len(selected_cluster_idxs)    # pure use hard, mixed use soft
                            if 'hv' in args['train.loss_type']:
                                selection, selection_info = pmo.selector(
                                    pmo.embed(torch.cat([task['context_images'], task['target_images']])),
                                    gumbel=gumbel, hard=hard)
                            else:
                                with torch.no_grad():
                                    selection, selection_info = pmo.selector(
                                        pmo.embed(torch.cat([task['context_images'], task['target_images']])),
                                        gumbel=gumbel, hard=hard)

                            '''forward 2 pure tasks as 2 objs'''
                            losses = []  # [2,]
                            for obj_idx in range(len(selected_cluster_idxs)):
                                context_images = torch_tasks[obj_idx]['context_images']
                                target_images = torch_tasks[obj_idx]['target_images']
                                context_labels = torch_tasks[obj_idx]['context_labels']
                                target_labels = torch_tasks[obj_idx]['target_labels']
                                if 'hv' in args['train.loss_type']:
                                    context_features = pmo.embed(context_images, selection=selection)
                                    target_features = pmo.embed(target_images, selection=selection)
                                else:
                                    with torch.no_grad():
                                        context_features = pmo.embed(context_images, selection=selection)
                                        target_features = pmo.embed(target_images, selection=selection)

                                loss, stats_dict, _ = prototype_loss(
                                    context_features, context_labels, target_features, target_labels,
                                    distance=args['test.distance'])
                                losses.append(loss)

                                epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['loss'])  # [2, 4]
                                epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['acc'])

                            ncc_losses_multi_obj.append(torch.stack(losses))
                        ncc_losses_multi_obj = torch.stack(ncc_losses_multi_obj)   # shape [num_tasks, num_objs], [4, 2]

                        '''calculate HV loss'''
                        ref = args['train.ref']
                        ncc_losses_multi_obj = ncc_losses_multi_obj.T       # [2, 4]
                        hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
                        epoch_loss['hv/loss'].append(hv_loss.item())

                        if 'hv' in args['train.loss_type']:
                            '''hv loss to average'''
                            hv_loss = hv_loss / args['train.n_mo']

                            '''step coefficient from 0 to hv_coefficient (default: 1.0)'''
                            hv_loss = hv_loss * (args['train.hv_coefficient'] * i / max_iter)
                            '''since no torch is saved in the pool, do not need to retain_graph'''
                            # retain_graph = True if mo_train_idx < args['train.n_mo'] - 1 else False
                            # hv_loss.backward(retain_graph=retain_graph)
                            hv_loss.backward()

                        '''calculate HV value for mutli-obj loss and acc'''
                        obj = np.array([[
                            epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                            for task_idx in range(len(torch_tasks))
                        ] for obj_idx in range(len(selected_cluster_idxs))])
                        hv = cal_hv(obj, ref, target='loss')
                        epoch_loss['hv'].append(hv)
                        obj = np.array([[
                            epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                            for task_idx in range(len(torch_tasks))
                        ] for obj_idx in range(len(selected_cluster_idxs))])
                        hv = cal_hv(obj, 0, target='acc')
                        epoch_acc['hv'].append(hv)

            # '''try prototypes' grad * 1000'''
            # for k, p in pmo.named_parameters():
            #     if 'selector.prototypes' in k and p.grad is not None:
            #         p.grad = p.grad * 1000

            update_step(i)

            '''log iter-wise params change'''
            writer.add_scalar('params/learning_rate', optimizer.param_groups[0]['lr'], i+1)
            writer.add_scalar('params/gumbel_tau', pmo.selector.tau.item(), i+1)
            writer.add_scalar('params/sim_logit_scale', pmo.selector.logit_scale.item(), i+1)

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

                if len(average_loss) > 0:      # did task train process
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

                '''store pool'''
                pool.store(i, train_loaders, trainsets, False,
                           class_filename=f'pool-{i+1}.json', center_filename=f'pool-{i+1}.npy')
                center_pool.store(i, train_loaders, trainsets, False,
                                  class_filename=f'center_pool-{i+1}.json', center_filename=f'center_pool-{i+1}.npy')

                '''write pool images'''
                images = pool.current_images(single_image=True)
                for cluster_id, cluster in enumerate(images):
                    if len(cluster) > 0:
                        writer.add_image(f"pool-image/{cluster_id}", cluster, i+1)
                    #     img_in_cluster = np.concatenate(cluster)
                    #     writer.add_images(f"train_image/pool-{cluster_id}", img_in_cluster, i+1)
                images = center_pool.current_images(single_image=True)
                for cluster_id, cluster in enumerate(images):
                    if len(cluster) > 0:
                        writer.add_image(f"center_pool-image/{cluster_id}", cluster, i+1)

                '''write pool similarities (class-wise)'''
                similarities = pool.current_similarities()
                for cluster_id, cluster in enumerate(similarities):
                    if len(cluster) > 0:
                        sim_in_cluster = np.stack(cluster)  # [num_cls, 8]
                        figure = draw_heatmap(sim_in_cluster, verbose=False)
                        writer.add_figure(f"pool-sim/{cluster_id}", figure, i+1)
                similarities = center_pool.current_similarities()
                for cluster_id, cluster in enumerate(similarities):
                    if len(cluster) > 0:
                        sim_in_cluster = np.stack(cluster)  # [num_cls, 8]
                        figure = draw_heatmap(sim_in_cluster, verbose=False)
                        writer.add_figure(f"center_pool-sim/{cluster_id}", figure, i+1)

                '''write image similarities in the pool'''
                similarities = pool.current_similarities(image_wise=True)
                for cluster_id, cluster in enumerate(similarities):
                    if len(cluster) > 0:
                        sim_in_cluster = np.concatenate(cluster)  # [num_cls*num_img, 8]
                        figure = draw_heatmap(sim_in_cluster, verbose=False)
                        writer.add_figure(f"pool-img-sim/{cluster_id}", figure, i+1)

                '''write image similarities in the pool after update iter'''
                numpy_images = pool.current_images()
                for cluster_idx, cluster in enumerate(numpy_images):
                    if len(cluster) > 0:
                        image_batch = torch.from_numpy(
                            np.concatenate(cluster)
                        ).to(device)
                        with torch.no_grad():
                            img_features = pmo.embed(image_batch)    # [img_size, 512]
                            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                            img_sim = selection_info['y_soft']        # [img_size, 10]
                            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                            tsk_sim = selection_info['y_soft']        # [1, 10]
                        sim = torch.cat([img_sim, *[tsk_sim]*(img_sim.shape[0]//10)]).cpu().numpy()
                        figure = draw_heatmap(sim, verbose=False)
                        writer.add_figure(f"pool-img-sim-re-cal/{cluster_idx}", figure, i+1)

                '''write task images'''
                writer.add_images(f"task-image/image", task_images, i+1)     # task images
                sim = epoch_loss['task/image_softmax_sim']
                figure = draw_heatmap(sim, verbose=False)
                writer.add_figure(f"task-image/sim", figure, i+1)
                with torch.no_grad():
                    img_features = pmo.embed(task_images.to(device))    # [img_size, 512]
                    _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                    img_sim = selection_info['y_soft']        # [img_size, 10]
                    _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                    tsk_sim = selection_info['y_soft']        # [1, 10]
                sim = torch.cat([img_sim, *[tsk_sim]*(img_sim.shape[0]//10)]).cpu().numpy()
                figure = draw_heatmap(sim, verbose=False)
                writer.add_figure(f"task-image/sim-re-cal", figure, i+1)

                '''write task similarities'''
                if len(epoch_loss[f'task/gumbel_sim']) > 0:
                    similarities = np.concatenate(epoch_loss[f'task/gumbel_sim'][-10:])      # [num_tasks, 8]
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/task-gumbel-sim", figure, i+1)
                    similarities = np.concatenate(epoch_loss[f'task/softmax_sim'][-10:])      # [num_tasks, 8]
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/task-softmax-sim", figure, i+1)

                '''write pure task image sim'''
                for cluster_idx, sim in epoch_loss[f'pure/image_softmax_sim'].items():
                    writer.add_images(f"pure-image/image{cluster_idx}", pure_task_images[cluster_idx], i + 1)  # pure images
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"pure-image/sim{cluster_idx}", figure, i + 1)

                '''write pure task similarities   10*10'''
                if len(epoch_loss[f'pure/task_softmax_sim']) > 0:
                    similarities = np.concatenate(epoch_loss[f'pure/task_softmax_sim'])
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/pure-task-softmax-sim", figure, i+1)
                    similarities = np.concatenate(epoch_loss[f'pure/task_dist'])
                    figure = draw_heatmap(similarities, verbose=True)
                    writer.add_figure(f"train_image/pure-task-dist", figure, i+1)

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
                    for task_id, task in enumerate(torch_tasks):
                        imgs = np.concatenate([task['context_images'].cpu().numpy(),
                                               task['target_images'].cpu().numpy()])
                        writer.add_images(f"mo-image/{pop_labels[task_id]}", imgs, i+1)

                '''log pool ce loss'''
                if len(epoch_loss[f'pool/selection_ce_loss']) > 0:      # did selection loss on pool samples
                    writer.add_scalar('train_loss/selection_ce_loss/pool',
                                      np.mean(epoch_loss[f'pool/selection_ce_loss']), i+1)

                '''log pure ce loss'''
                if len(epoch_loss[f'pure/selection_ce_loss']) > 0:  # did selection loss on pool samples
                    writer.add_scalar('train_loss/selection_ce_loss/pure',
                                      np.mean(epoch_loss[f'pure/selection_ce_loss']), i + 1)

                # '''log task ce loss'''
                # if len(epoch_loss[f'task/selection_ce_loss']) > 0:      # did selection loss on training tasks
                #     writer.add_scalar('train_loss/selection_ce_loss/task',
                #                       np.mean(epoch_loss[f'task/selection_ce_loss']), i+1)

                '''log pure ncc loss'''
                for cluster_idx in range(args['model.num_clusters']):
                    if f'pure/C{cluster_idx}' in epoch_loss.keys() and len(epoch_loss[f'pure/C{cluster_idx}']) > 0:
                        writer.add_scalar(f'train_loss/pure/C{cluster_idx}',
                                          np.mean(epoch_loss[f'pure/C{cluster_idx}']), i+1)
                        writer.add_scalar(f'train_accuracy/pure/C{cluster_idx}',
                                          np.mean(epoch_acc[f'pure/C{cluster_idx}']), i+1)

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
                val_pool.centers = pool.centers     # same centers and device as train_pool; no use
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

                            [enriched_context_features, enriched_target_features], _ = pmo(
                                [context_images, target_images], torch.cat([context_images, target_images]),
                                gumbel=False, hard=False)
                            # enriched_context_features, _ = pmo(context_images, gumbel=False)
                            # enriched_target_features, _ = pmo(target_images, gumbel=False)

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
                                pmo.embed(images), gumbel=False, average=False)  # [bs, n_clusters]
                            similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]
                            # if similarities.shape[0] == 1 and images.shape[0] != 1:    # repeat to match num of samples
                            #     similarities = np.concatenate(([similarities for _ in range(images.shape[0])]))
                            cluster_idxs = np.argmax(similarities, axis=1)  # [bs]

                            val_pool.put_batch(
                                images.cpu(), cluster_idxs, {
                                    'domain': domain, 'gt_labels': gt_labels, 'similarities': similarities})

                            '''check if any cluster have sufficient class to construct 1 task'''
                            num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in
                                                 val_pool.current_classes()]
                            n_way, n_shot, n_query = available_setting(num_imgs_clusters, args['test.type'],
                                                                       min_available_clusters=1,
                                                                       use_max_shot=True)

                            if n_way != -1:
                                # enough classes to construct 1 task
                                available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)
                                # then use all available classes to construct 1 task
                                for idx in available_cluster_idxs:
                                    task = val_pool.episodic_sample(
                                        idx, n_way, n_shot, n_query,
                                        remove_sampled_classes=True,
                                        d=device
                                    )

                                    [enriched_context_features, enriched_target_features], _ = pmo(
                                        [task['context_images'], task['target_images']],
                                        torch.cat([context_images, target_images]), gumbel=False, hard=True)

                                    # enriched_context_features, _ = pmo(task['context_images'], gumbel=False)
                                    # enriched_target_features, _ = pmo(task['target_images'], gumbel=False)

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

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.3f}""")
    else:
        print(f"""Training not completed. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

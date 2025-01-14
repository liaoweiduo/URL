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
        start_iter, best_val_loss, best_val_acc = 0, 999999999, -1

        # pmo model, fe load from url
        pmo = get_model_moe(None, args, base_network_name='url')    # resnet18_moe

        if 'kd' in args['train.loss_type']:
            # KL-divergence loss
            criterion_div = DistillKL(T=4)
            url = get_model(None, args, base_network_name='url', freeze_fe=True)
            url.eval()

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

        def lr_manager_step(idx):
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
            task_features = pmo.embed(torch.cat([context_images, target_images]))
            similarities = np.array([0] * len(gt_labels))       # no use

            not_full = pool.put_buffer(
                task_images, {'domain': domain, 'gt_labels': gt_labels,
                              'similarities': similarities, 'features': task_features.cpu().numpy()})

            if not not_full and verbose:  # full buffer
                print(f'Buffer is full at iter: {i}.')
                verbose = False
                # print(f'Buffer is full num classes in buffer: {len(pool.buffer)}..')
            # if not not_full:    # enough sampling
            #     break

            # need to check how many classes in 1 samples and need a buffer size
            # about 10 iters can obtain 200 classes
            # print(f'num classes in buffer: {len(pool.buffer)}.')

            '''----------------'''
            '''Task Train Phase'''
            '''----------------'''
            if 'task' in args['train.loss_type']:
                [enriched_context_features, enriched_target_features], selection_info = pmo(
                    [context_images, target_images], task_features,
                    gumbel=args['train.sim_gumbel'], hard=args['train.sim_gumbel'])
                # task_cluster_idx = torch.argmax(selection_info['y_soft'], dim=1).squeeze()
                # # supervision to be softmax for CE loss
            else:
                with torch.no_grad():
                    [enriched_context_features, enriched_target_features], selection_info = pmo(
                        [context_images, target_images], task_features,
                        gumbel=args['train.sim_gumbel'], hard=args['train.sim_gumbel'])

            task_loss, stats_dict, pred_dict = prototype_loss(
                enriched_context_features, context_labels,
                enriched_target_features, target_labels,
                distance=args['test.distance'])

            if 'kd' in args['train.loss_type']:

                '''forward url obtain url features'''
                url_context_features = url.embed(context_images)
                url_target_features = url.embed(target_images)

                if args['train.kd_type'] == 'kl':
                    fs = pred_dict['logits_tensor']     # NCC logits
                    _, _, url_pred_dict = prototype_loss(
                        url_context_features, context_labels,
                        url_target_features, target_labels,
                        distance=args['test.distance'])
                    ft = url_pred_dict['logits_tensor']     # NCC logits
                    # kd_losses = criterion_div(F.softmax(fs, dim=1),
                    #                           F.softmax(ft, dim=1))
                    kd_losses = criterion_div(fs, ft)
                elif args['train.kd_type'] == 'film_param_l2':
                    film_gamma_params = torch.cat([
                        v.flatten() for k, v in pmo.named_parameters()
                        if v.requires_grad and 'film' in k and 'gamma' in k]).unsqueeze(0)      # [1, dim]
                    film_beta_params = torch.cat([
                        v.flatten() for k, v in pmo.named_parameters()
                        if v.requires_grad and 'film' in k and 'beta' in k]).unsqueeze(0)       # [1, dim]
                    film_gamma_labels = torch.ones_like(film_gamma_params)
                    film_beta_labels = torch.zeros_like(film_beta_params)
                    kd_losses = (torch.cdist(film_gamma_params, film_gamma_labels, p=2.0).squeeze() +
                                 torch.cdist(film_beta_params, film_beta_labels, p=2.0).squeeze()) / 2
                else:   # kernelCKA
                    fs = torch.cat([enriched_context_features, enriched_target_features])
                    ft = torch.cat([url_context_features, url_target_features]).detach()
                    kd_losses = distillation_loss(F.normalize(fs, p=2, dim=1, eps=1e-12),
                                                  F.normalize(ft, p=2, dim=1, eps=1e-12),
                                                  opt=args['train.kd_type'])

                '''log kd_losses'''
                epoch_loss['kd'].append(kd_losses.item())

                '''kd_losses to coeff'''
                kd_weight_annealing = WeightAnnealing(
                    T=int(args['train.cosine_anneal_freq'] * args['train.kd_T_extent']))
                kd_weight = max(kd_weight_annealing(t=i, opt='linear'), 0) * args['train.kd_coefficient']
                # kd_weight = args['train.kd_coefficient']
                writer.add_scalar('params/kd_weight', kd_weight, i+1)

                task_loss = task_loss + kd_losses * kd_weight

            '''log task loss and acc'''
            epoch_loss[f'task/{trainset}'].append(stats_dict['loss'])
            epoch_acc[f'task/{trainset}'].append(stats_dict['acc'])
            # ilsvrc_2012 has 2 times larger len than others.

            '''log task sim (softmax and gumbel)'''
            epoch_loss[f'task/gumbel_sim'].append(selection_info['y_soft'].detach().cpu().numpy())    # [1,8]
            epoch_loss[f'task/softmax_sim'].append(selection_info['normal_soft'].detach().cpu().numpy())

            '''log img sim in the task'''
            with torch.no_grad():
                img_features = task_features  # [img_size, 512]
                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                img_sim = selection_info['y_soft']  # [img_size, 10]
                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                tsk_sim = selection_info['y_soft']  # [1, 10]
            sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
            epoch_loss[f'task/image_softmax_sim'] = sim

            if 'task' in args['train.loss_type']:
                zero_grad()
                task_loss.backward()

                '''debug'''
                debugger.print_grad(pmo, key='film', prefix=f'iter{i} after task_loss (with kd_loss) backward:\n')

                optimizer.step()

            '''-------------------'''
            '''Cluster Train Phase'''
            '''-------------------'''
            '''maintain pool'''
            if (i + 1) % args['train.pool_freq'] == 0:
                print(f"\n>> Iter: {i}, update pool: ")

                '''collect samples in the buffer'''
                pool.clear_clusters()       # do not need last iter's center

                verbose = True
                if verbose:
                    print(f'Buffer contains {len(pool.buffer)} classes.')

                '''re-cal sim and re-put samples into pool buffer'''
                if len(pool.buffer) > 0:
                    # need to check num of images, maybe need to reshape to batch to calculate
                    # less than 1w images for 200 classes
                    # print(f'num images in buffer (cal sim): {len(images)}.')

                    for cls in pool.buffer:
                        # cal sim from stored features
                        features = torch.from_numpy(cls['features']).to(device)
                        images = torch.from_numpy(cls['images'])

                        with torch.no_grad():
                            # features = pmo.embed(images.to(device))
                            _, selection_info = pmo.selector(
                                features, gumbel=False, average=False)  # [bs, n_clusters]
                            similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                        cls['similarities'] = similarities

                '''collect cluster for center_pool'''
                current_clusters = center_pool.clear_clusters()
                current_clusters = [cls for clses in current_clusters for cls in clses]       # cat all clusters

                '''re-cal sim and re-put samples into center pool's buffer'''
                center_pool.clear_buffer()
                center_pool.buffer = copy.deepcopy(pool.buffer)
                for current_cls in current_clusters:
                    current_images = torch.from_numpy(current_cls['images'])
                    current_features = current_cls['features']
                    current_gt_labels = np.array([current_cls['label'][0]] * len(current_images))
                    current_domain = np.array([current_cls['label'][1]] * len(current_images))

                    with torch.no_grad():
                        # current_features = current_images.to(device)
                        _, selection_info = pmo.selector(
                            torch.from_numpy(current_features).to(device), gumbel=False, average=False)  # [bs, n_clusters]
                        current_similarities = selection_info['y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                    '''put current cases into center_pool.buffer'''
                    center_pool.put_buffer(
                        current_images, {
                            'domain': current_domain, 'gt_labels': current_gt_labels,
                            'similarities': current_similarities, 'features': current_features},
                        maintain_size=False)

                '''buffer -> clusters'''
                pool.buffer2cluster()
                pool.clear_buffer()
                center_pool.buffer2cluster()
                center_pool.clear_buffer()

                if args['train.cluster_center_mode'] == 'mov_avg':
                    print(f"\n>> Iter: {i}, update prototypes: ")

                    centers = []
                    for cluster in pool.clusters:
                        # cat features and forward with average embedding
                        features = torch.cat([torch.from_numpy(cls['features']) for cls in cluster]).to(device)
                        with torch.no_grad():
                            _, selection_info = pmo.selector(
                                features, gumbel=False, average=True)
                            center = selection_info['embeddings']  # [1, 64, 1, 1]
                        centers.append(center)
                    centers = torch.stack(centers)

                    pmo.selector.update_prototypes(centers)

                '''debug'''
                debugger.print_prototype_change(pmo, i=i, writer=writer)

                '''selection CE loss on all clusters'''
                if 'ce' in args['train.loss_type']:
                    print(f"\n>> Iter: {i}, clustering loss calculation: ")
                    features_batch, cluster_labels = [], []
                    for cluster_idx, cluster in enumerate(pool.clusters):
                        if len(cluster) > 0:
                            features = np.concatenate([cls['features'] for cls in cluster])
                            features_batch.append(features)
                            cluster_labels.append([cluster_idx] * features.shape[0])
                    features_batch = torch.from_numpy(np.concatenate(features_batch)).to(device)
                    cluster_labels = torch.from_numpy(np.concatenate(cluster_labels)).long().to(device)

                    _, selection_info = pmo.selector(features_batch, gumbel=False, average=False)

                    if args['train.cluster_loss_type'] == 'kl':
                        fn = DistillKL(T=4)
                        cluster_labels = F.one_hot(cluster_labels, num_classes=args['model.num_clusters']).float()
                    else:
                        fn = torch.nn.CrossEntropyLoss()
                    dist = selection_info['dist']  # [img_size, 8]
                    selection_ce_loss = fn(dist, cluster_labels)

                    '''log ce loss'''
                    epoch_loss[f'pool/selection_ce_loss'].append(selection_ce_loss.item())

                    zero_grad()
                    '''ce loss coefficient'''
                    selection_ce_loss = selection_ce_loss * args['train.ce_coefficient']
                    selection_ce_loss.backward()

                    '''debug'''
                    debugger.print_grad(pmo, key='film', prefix=f'iter{i} after selection_ce_loss backward:\n')

                    optimizer_selector.step()

            '''----------------'''
            '''MO Train Phase  '''
            '''----------------'''
            if (i + 1) % args['train.mo_freq'] == 0:
                print(f"\n>> Iter: {i + 1}, MO phase: "
                      f"({'train' if 'hv' in args['train.loss_type'] else 'eval'})")

                zero_grad()

                # model_eval()
                # if 'pure' not in args['train.loss_type'] and 'hv' not in args['train.loss_type']:
                #     model_eval()

                num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]

                '''pure loss on all clusters'''
                epoch_loss[f'pure/task_softmax_sim'] = []
                epoch_loss[f'pure/task_dist'] = []
                epoch_loss[f'pure/image_softmax_sim'] = {}
                if 'pure' in args['train.loss_type']:
                    pure_task_images = {}
                    for cluster_idx in range(len(num_imgs_clusters)):
                        n_way, n_shot, n_query = available_setting([num_imgs_clusters[cluster_idx]],
                                                                   args['train.type'])
                        if n_way == -1:
                            continue    # not enough samples to construct a task
                        else:
                            pure_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
                            context_images, target_images = pure_task['context_images'], pure_task['target_images']
                            context_features, target_features = pure_task['context_features'], pure_task['target_features']
                            context_labels, target_labels = pure_task['context_labels'], pure_task['target_labels']
                            pure_task_images[cluster_idx] = torch.cat([context_images, target_images]).cpu()

                            [enriched_context_features, enriched_target_features], selection_info = pmo(
                                [context_images, target_images], torch.cat([context_features, target_features]),
                                gumbel=args['train.sim_gumbel'], hard=args['train.sim_gumbel'])

                            pure_loss, stats_dict, _ = prototype_loss(
                                enriched_context_features, context_labels,
                                enriched_target_features, target_labels,
                                distance=args['test.distance'])

                            '''pure_loss to average'''
                            pure_loss = pure_loss / len(num_imgs_clusters)
                            '''step coefficient from 0 to pure_coefficient (default: 1.0)'''
                            pure_loss = pure_loss * (args['train.pure_coefficient'] * min(i * 5, max_iter) / max_iter)

                            zero_grad()
                            pure_loss.backward()
                            optimizer.step()
                            optimizer_selector.step()

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
                                img_features = torch.cat([context_features, target_features])  # [img_size, 512]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                                img_sim = selection_info['y_soft']  # [img_size, 10]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                                tsk_sim = selection_info['y_soft']  # [1, 10]
                            sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
                            epoch_loss[f'pure/image_softmax_sim'][cluster_idx] = sim

                    '''debug'''
                    debugger.print_grad(pmo, key='film', prefix=f'iter{i} after pure_loss backward:\n')

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
                        epoch_loss[f'mo/image_softmax_sim'] = {}
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
                            gumbel = args['train.sim_gumbel']
                            hard = args['train.sim_gumbel']
                            # hard = task_idx < len(selected_cluster_idxs)    # pure use hard, mixed use soft
                            if task_idx < len(selected_cluster_idxs):   # pure use feature->sim, mixed use img->sim
                                torch_task_features = torch.cat([task['context_features'], task['target_features']])
                            else:
                                torch_task_features = pmo.embed(
                                    torch.cat([task['context_images'], task['target_images']]))
                            if 'hv' in args['train.loss_type']:
                                selection, selection_info = pmo.selector(
                                    torch_task_features,
                                    gumbel=gumbel, hard=hard)
                            else:
                                with torch.no_grad():
                                    selection, selection_info = pmo.selector(
                                        torch_task_features,
                                        gumbel=gumbel, hard=hard)

                            '''log img sim in the task'''
                            with torch.no_grad():
                                img_features = torch_task_features  # [img_size, 512]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                                img_sim = selection_info['y_soft']  # [img_size, 10]
                                _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                                tsk_sim = selection_info['y_soft']  # [1, 10]
                            sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
                            epoch_loss[f'mo/image_softmax_sim'][task_idx] = sim

                            '''do inner loop on task's sup set itself for pmo's film then cal 2 objs'''
                            pmo_clone = copy.deepcopy(pmo)
                            inner_step = 1
                            inner_lr = args['train.inner_learning_rate']
                            pmo_clone_opt = torch.optim.Adam(pmo_clone.get_trainable_film_parameters(),
                                                             lr=inner_lr, weight_decay=inner_lr / 50)
                            for inner_idx in range(inner_step):
                                # perform update of model weights
                                inner_context_features = pmo_clone.embed(task['context_images'], selection=selection)
                                inner_context_labels = task['context_labels']
                                loss, stats_dict, _ = prototype_loss(
                                    inner_context_features, inner_context_labels,
                                    inner_context_features, inner_context_labels,
                                    distance=args['test.distance'])
                                # gradients = torch.autograd.grad(loss, pmo_clone.get_trainable_film_parameters(),
                                #                                 create_graph=False)     # FOMAML

                                # update weights manually
                                pmo_clone_opt.zero_grad()
                                loss.backward(retain_graph=True)
                                pmo_clone_opt.step()

                            '''forward 2 pure tasks as 2 objs'''
                            losses = []  # [2,]
                            for obj_idx in range(len(selected_cluster_idxs)):
                                context_images = torch_tasks[obj_idx]['context_images']
                                target_images = torch_tasks[obj_idx]['target_images']
                                context_labels = torch_tasks[obj_idx]['context_labels']
                                target_labels = torch_tasks[obj_idx]['target_labels']
                                if 'hv' in args['train.loss_type']:
                                    context_features = pmo_clone.embed(context_images, selection=selection)
                                    target_features = pmo_clone.embed(target_images, selection=selection)
                                else:
                                    with torch.no_grad():
                                        context_features = pmo_clone.embed(context_images, selection=selection)
                                        target_features = pmo_clone.embed(target_images, selection=selection)

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
                        if args['train.n_obj'] > 1:
                            hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
                            epoch_loss['hv/loss'].append(hv_loss.item())

                            '''hv loss to average'''
                            hv_loss = hv_loss / args['train.n_mo']

                        if 'hv' in args['train.loss_type']:

                            '''step coefficient from 0 to hv_coefficient (default: 1.0)'''
                            hv_loss = hv_loss * (args['train.hv_coefficient'] * min(i * 5, max_iter) / max_iter)
                            '''since no torch is saved in the pool, do not need to retain_graph'''
                            # retain_graph = True if mo_train_idx < args['train.n_mo'] - 1 else False
                            # hv_loss.backward(retain_graph=retain_graph)
                            zero_grad()
                            hv_loss.backward()

                            optimizer.step()
                            optimizer_selector.step()

                        '''calculate HV value for mutli-obj loss and acc'''
                        if args['train.n_obj'] > 1:
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

                model_train()
                # if 'pure' not in args['train.loss_type'] and 'hv' not in args['train.loss_type']:
                #     model_train()

                '''debug'''
                debugger.print_grad(pmo, key='film', prefix=f'iter{i} after hv_loss backward:\n')

            # '''try prototypes' grad * 1000'''
            # for k, p in pmo.named_parameters():
            #     if 'selector.prototypes' in k and p.grad is not None:
            #         p.grad = p.grad * 1000

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

                '''log task loss and accuracy'''
                average_loss, average_accuracy = [], []
                for dataset_name in trainsets:
                    if f'task/{dataset_name}' in epoch_loss.keys() and len(epoch_loss[f'task/{dataset_name}']) > 0:
                        writer.add_scalar(f"train_task_loss/{dataset_name}",
                                          np.mean(epoch_loss[f'task/{dataset_name}']), i+1)
                        writer.add_scalar(f"train_task_accuracy/{dataset_name}",
                                          np.mean(epoch_acc[f'task/{dataset_name}']), i+1)
                        average_loss.append(epoch_loss[f'task/{dataset_name}'])
                        average_accuracy.append(epoch_acc[f'task/{dataset_name}'])

                if len(average_loss) > 0:      # did task train process
                    average_loss = np.mean(np.concatenate(average_loss))
                    average_accuracy = np.mean(np.concatenate(average_accuracy))
                    writer.add_scalar(f"train_task_loss/average", average_loss, i+1)
                    writer.add_scalar(f"train_task_accuracy/average", average_accuracy, i+1)
                    print(f"==>> task: loss {average_loss:.3f}, "
                          f"accuracy {average_accuracy:.3f}.")

                '''log kd_loss'''
                if len(epoch_loss['kd']) > 0:
                    average_loss = np.mean(epoch_loss['kd'])
                    writer.add_scalar(f"train_kd_loss", average_loss, i + 1)

                # '''log task_rec'''
                # writer.add_scalar(f"loss/train/task_rec",
                #                   np.mean(epoch_loss['task_rec']), i+1)

                pop_labels = [
                    f"p{idx}" if idx < args['train.n_obj'] else f"m{idx-args['train.n_obj']}"
                    for idx in range(args['train.n_mix'] + args['train.n_obj'])
                ]       # ['p0', 'p1', 'm0', 'm1']

                if len(epoch_loss['hv/loss']) > 0:      # did mo process
                    '''log multi-objective loss and accuracy'''
                    objs_loss, objs_acc = [], []        # for average figure visualization
                    for obj_idx in range(args['train.n_obj']):
                        obj_loss, obj_acc = [], []
                        for pop_idx in range(args['train.n_mix'] + args['train.n_obj']):
                            loss_values = epoch_loss[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"train_mo_loss/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(loss_values), i+1)
                            obj_loss.append(np.mean(loss_values))
                            acc_values = epoch_acc[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"train_mo_accuracy/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(acc_values), i+1)
                            obj_acc.append(np.mean(acc_values))
                        objs_loss.append(obj_loss)
                        objs_acc.append(obj_acc)

                    '''log objs figure'''
                    objs = np.array(objs_loss)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"train_image/objs_loss", figure, i+1)
                    objs = np.array(objs_acc)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"train_image/objs_acc", figure, i+1)

                    '''log hv'''
                    writer.add_scalar('train_mo_loss/hv_loss', np.mean(epoch_loss['hv/loss']), i+1)
                    writer.add_scalar('train_mo_loss/hv', np.mean(epoch_loss['hv']), i+1)
                    writer.add_scalar('train_mo_accuracy/hv', np.mean(epoch_acc['hv']), i+1)
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

                # '''write image similarities in the pool'''
                # similarities = pool.current_similarities(image_wise=True)
                # for cluster_id, cluster in enumerate(similarities):
                #     if len(cluster) > 0:
                #         sim_in_cluster = np.concatenate(cluster)  # [num_cls*num_img, 8]
                #         figure = draw_heatmap(sim_in_cluster, verbose=False)
                #         writer.add_figure(f"pool-img-sim/{cluster_id}", figure, i+1)

                # '''write image similarities in the pool after update iter'''
                # for cluster_idx, cluster in enumerate(pool.clusters):
                #     if len(cluster) > 0:
                #         features_batch = torch.from_numpy(
                #             np.concatenate([cls['features'] for cls in cluster])
                #         ).to(device)
                #         with torch.no_grad():
                #             img_features = features_batch      # [img_size, 512]
                #             _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
                #             img_sim = selection_info['y_soft']        # [img_size, 10]
                #             _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                #             tsk_sim = selection_info['y_soft']        # [1, 10]
                #         sim = torch.cat([img_sim, *[tsk_sim]*(img_sim.shape[0]//10)]).cpu().numpy()
                #         figure = draw_heatmap(sim, verbose=False)
                #         writer.add_figure(f"pool-img-sim-(fea)-re-cal/{cluster_idx}", figure, i+1)

                '''write task images'''
                writer.add_images(f"task-image/image", task_images, i+1)     # task images
                sim = epoch_loss['task/image_softmax_sim']
                figure = draw_heatmap(sim, verbose=False)
                writer.add_figure(f"task-image/sim", figure, i+1)
                with torch.no_grad():
                    img_features = task_features    # [img_size, 512]
                    # img_features = pmo.embed(task_images.to(device))    # [img_size, 512]
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

                # '''write pure task image sim'''
                # for cluster_idx, sim in epoch_loss[f'pure/image_softmax_sim'].items():
                #     writer.add_images(f"pure-image/image{cluster_idx}", pure_task_images[cluster_idx], i + 1)  # pure images
                #     figure = draw_heatmap(sim, verbose=False)
                #     writer.add_figure(f"pure-image/sim{cluster_idx}", figure, i + 1)

                '''write pure task similarities   10*10'''
                if len(epoch_loss[f'pure/task_softmax_sim']) > 0:
                    similarities = np.concatenate(epoch_loss[f'pure/task_softmax_sim'])
                    figure = draw_heatmap(similarities, verbose=False)
                    writer.add_figure(f"train_image/pure-task-softmax-sim", figure, i+1)
                    similarities = np.concatenate(epoch_loss[f'pure/task_dist'])
                    figure = draw_heatmap(similarities, verbose=True)
                    writer.add_figure(f"train_image/pure-task-dist", figure, i+1)

                '''write mo: (pure+mixed) task image sim'''
                for task_id, sim in epoch_loss[f'mo/image_softmax_sim'].items():
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"mo-image/{pop_labels[task_id]}/sim", figure, i + 1)

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
                    writer.add_scalar('train_ce_loss/pool',
                                      np.mean(epoch_loss[f'pool/selection_ce_loss']), i+1)

                '''log pure ce loss'''
                if len(epoch_loss[f'pure/selection_ce_loss']) > 0:  # did selection loss on pool samples
                    writer.add_scalar('train_ce_loss/pure',
                                      np.mean(epoch_loss[f'pure/selection_ce_loss']), i + 1)

                # '''log task ce loss'''
                # if len(epoch_loss[f'task/selection_ce_loss']) > 0:      # did selection loss on training tasks
                #     writer.add_scalar('train_loss/selection_ce_loss/task',
                #                       np.mean(epoch_loss[f'task/selection_ce_loss']), i+1)

                '''log pure ncc loss'''
                for cluster_idx in range(args['model.num_clusters']):
                    if f'pure/C{cluster_idx}' in epoch_loss.keys() and len(epoch_loss[f'pure/C{cluster_idx}']) > 0:
                        writer.add_scalar(f'train_pure_loss/C{cluster_idx}',
                                          np.mean(epoch_loss[f'pure/C{cluster_idx}']), i+1)
                        writer.add_scalar(f'train_pure_accuracy/C{cluster_idx}',
                                          np.mean(epoch_acc[f'pure/C{cluster_idx}']), i+1)

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

                val_pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
                val_pool.centers = pool.centers     # same centers and device as train_pool; no use
                val_pool.eval()

                '''collect val_losses/accs for all sources and cluster_losses/accs for all FiLMs'''
                val_accs, val_losses = {f'{name}': [] for name in valsets}, {f'{name}': [] for name in valsets}
                cluster_accs, cluster_losses = [[] for _ in range(args['model.num_clusters'])], \
                                               [[] for _ in range(args['model.num_clusters'])]
                epoch_val_acc[f'mo/image_softmax_sim'] = {}
                epoch_val_acc['hv'] = []
                epoch_val_acc.update({
                    f'hv/obj{obj_idx}': {
                        f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                    } for obj_idx in range(args['train.n_obj'])})
                epoch_val_loss['hv'] = []
                epoch_val_loss.update({
                    f'hv/obj{obj_idx}': {
                        f'hv/pop{pop_idx}': [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                    } for obj_idx in range(args['train.n_obj'])})
                pop_labels = [
                    f"p{idx}" if idx < args['train.n_obj'] else f"m{idx-args['train.n_obj']}"
                    for idx in range(args['train.n_mix'] + args['train.n_obj'])
                ]       # ['p0', 'p1', 'm0', 'm1']
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
                            [enriched_context_features, enriched_target_features], _ = pmo(
                                [context_images, target_images], features,
                                gumbel=False, hard=args['train.sim_gumbel'])

                            _, stats_dict, _ = prototype_loss(
                                enriched_context_features, context_labels,
                                enriched_target_features, target_labels,
                                distance=args['test.distance'])

                            val_losses[valset].append(stats_dict['loss'])
                            val_accs[valset].append(stats_dict['acc'])

                            '''samples put to val buffer'''
                            task_images = torch.cat([context_images, target_images]).cpu()
                            gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
                            domain = np.array([domain] * len(gt_labels))
                            task_features = pmo.embed(torch.cat([context_images, target_images]))

                            _, selection_info = pmo.selector(
                                features, gumbel=False, average=False)  # [bs, n_clusters]
                            similarities = selection_info[
                                'y_soft'].detach().cpu().numpy()  # [bs, n_clusters]

                            not_full = val_pool.put_buffer(
                                task_images, {'domain': domain, 'gt_labels': gt_labels,
                                              'similarities': similarities, 'features': task_features.cpu().numpy()},
                                maintain_size=False)

                        '''collect samples in the val buffer'''
                        val_pool.clear_clusters()  # do not need last iter's center

                        verbose = False
                        if verbose:
                            print(f'Buffer contains {len(val_pool.buffer)} classes.')

                        '''buffer -> clusters'''
                        val_pool.buffer2cluster()
                        val_pool.clear_buffer()

                        '''repeat collecting MO acc on val pool'''
                        num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in
                                             val_pool.current_classes()]
                        for mo_train_idx in range(args['train.n_mo']):
                            '''check pool has enough samples and generate 1 setting'''
                            n_way, n_shot, n_query = available_setting(num_imgs_clusters, args['train.mo_task_type'],
                                                                       min_available_clusters=args['train.n_obj'])
                            if n_way == -1:  # not enough samples
                                print(f"==>> val_pool has not enough samples. skip MO evaluation this iter.")
                                break
                            else:
                                available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

                                selected_cluster_idxs = sorted(np.random.choice(
                                    available_cluster_idxs, args['train.n_obj'], replace=False))

                                torch_tasks = []
                                epoch_val_acc[f'mo/image_softmax_sim'] = {}
                                '''sample pure tasks from clusters in selected_cluster_idxs'''
                                for cluster_idx in selected_cluster_idxs:
                                    pure_task = val_pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d=device)
                                    torch_tasks.append(pure_task)

                                '''sample mix tasks by mixer'''
                                for mix_id in range(args['train.n_mix']):
                                    numpy_mix_task, _ = mixer.mix(
                                        task_list=[val_pool.episodic_sample(idx, n_way, n_shot, n_query)
                                                   for idx in selected_cluster_idxs],
                                        mix_id=mix_id
                                    )
                                    torch_tasks.append(task_to_device(numpy_mix_task, device))

                                for task_idx, task in enumerate(torch_tasks):
                                    '''obtain task-specific selection'''
                                    # pure use feature->sim, mixed use img->sim
                                    if task_idx < len(selected_cluster_idxs):
                                        torch_task_features = torch.cat(
                                            [task['context_features'], task['target_features']])
                                    else:
                                        torch_task_features = pmo.embed(
                                            torch.cat([task['context_images'], task['target_images']]))
                                    selection, selection_info = pmo.selector(
                                        torch_task_features,
                                        gumbel=False, hard=args['train.sim_gumbel'])

                                    '''log img sim in the task'''
                                    img_features = torch_task_features  # [img_size, 512]
                                    _, selection_info = pmo.selector(img_features, gumbel=False, hard=False,
                                                                     average=False)
                                    img_sim = selection_info['y_soft']  # [img_size, 10]
                                    _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
                                    tsk_sim = selection_info['y_soft']  # [1, 10]
                                    sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
                                    epoch_val_acc[f'mo/image_softmax_sim'][task_idx] = sim

                                    '''forward 2 pure tasks as 2 objs'''
                                    for obj_idx in range(len(selected_cluster_idxs)):
                                        context_images = torch_tasks[obj_idx]['context_images']
                                        target_images = torch_tasks[obj_idx]['target_images']
                                        context_labels = torch_tasks[obj_idx]['context_labels']
                                        target_labels = torch_tasks[obj_idx]['target_labels']
                                        context_features = pmo.embed(context_images, selection=selection)
                                        target_features = pmo.embed(target_images, selection=selection)

                                        _, stats_dict, _ = prototype_loss(
                                            context_features, context_labels, target_features, target_labels,
                                            distance=args['test.distance'])

                                        if task_idx == obj_idx:     # forward on itself
                                            cluster_losses[torch.argmax(selection).item()].append(stats_dict['loss'])
                                            cluster_accs[torch.argmax(selection).item()].append(stats_dict['acc'])

                                        epoch_val_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['loss'])
                                        epoch_val_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'].append(stats_dict['acc'])

                                '''calculate HV value for mutli-obj loss acc'''
                                if args['train.n_obj'] > 1:
                                    obj = np.array([[
                                        epoch_val_loss[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                                        for task_idx in range(len(torch_tasks))
                                    ] for obj_idx in range(len(selected_cluster_idxs))])
                                    ref = args['train.ref']
                                    hv = cal_hv(obj, ref, target='loss')
                                    epoch_val_loss['hv'].append(hv)
                                    obj = np.array([[
                                        epoch_val_acc[f'hv/obj{obj_idx}'][f'hv/pop{task_idx}'][-1]
                                        for task_idx in range(len(torch_tasks))
                                    ] for obj_idx in range(len(selected_cluster_idxs))])
                                    hv = cal_hv(obj, 0, target='acc')
                                    epoch_val_acc['hv'].append(hv)

                '''write mo: (pure+mixed) task image sim for val'''
                for task_id, sim in epoch_val_acc[f'mo/image_softmax_sim'].items():
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"val-mo-image/{pop_labels[task_id]}/sim", figure, i + 1)

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

                '''write and print val on cluster'''
                for cluster_idx, (loss_list, acc_list) in enumerate(zip(cluster_losses, cluster_accs)):
                    if len(loss_list) > 0:
                        cluster_acc, cluster_loss = np.mean(acc_list).item(), np.mean(loss_list).item()

                        epoch_val_loss[f"C{cluster_idx}"] = cluster_loss
                        epoch_val_acc[f"C{cluster_idx}"] = cluster_acc
                        writer.add_scalar(f"val-cluster-loss/C{cluster_idx}", cluster_loss, i+1)
                        writer.add_scalar(f"val-cluster-accuracy/C{cluster_idx}", cluster_acc, i+1)
                        print(f"==>> val C{cluster_idx}: "
                              f"val_loss {cluster_loss:.3f}, accuracy {cluster_acc:.3f}")

                    else:   # no class(task) is assign to this cluster
                        print(f"==>> val C{cluster_idx}: "
                              f"val_loss No value, val_acc No value")

                # write summaries averaged over clusters
                avg_val_cluster_loss = np.mean(np.concatenate(cluster_losses))
                avg_val_cluster_acc = np.mean(np.concatenate(cluster_accs))
                writer.add_scalar(f"val-cluster-loss/avg_val_cluster_loss", avg_val_cluster_loss, i+1)
                writer.add_scalar(f"val-cluster-accuracy/avg_val_cluster_acc", avg_val_cluster_acc, i+1)
                print(f"==>> val: avg_loss {avg_val_cluster_loss:.3f}, "
                      f"avg_accuracy {avg_val_cluster_acc:.3f}.")

                if len(epoch_val_acc['hv']) > 0:      # did mo process
                    '''log multi-objective loss/accuracy'''
                    objs_loss, objs_acc = [], []        # for average figure visualization
                    for obj_idx in range(args['train.n_obj']):
                        obj_loss, obj_acc = [], []
                        for pop_idx in range(args['train.n_mix'] + args['train.n_obj']):
                            loss_values = epoch_val_loss[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"val-mo-loss/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(loss_values), i+1)
                            obj_loss.append(np.mean(loss_values))
                            acc_values = epoch_val_acc[f'hv/obj{obj_idx}'][f'hv/pop{pop_idx}']
                            writer.add_scalar(f"val-mo-accuracy/obj{obj_idx}/pop{pop_idx}",
                                              np.mean(acc_values), i+1)
                            obj_acc.append(np.mean(acc_values))
                        objs_loss.append(obj_loss)
                        objs_acc.append(obj_acc)

                    '''log objs figure'''
                    objs = np.array(objs_loss)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"val-image/objs_loss", figure, i+1)
                    objs = np.array(objs_acc)     # [2, 4]
                    figure = draw_objs(objs, pop_labels)
                    writer.add_figure(f"val-image/objs_acc", figure, i+1)

                    '''log hv'''
                    writer.add_scalar('val-mo-loss/hv', np.mean(epoch_val_loss['hv']), i+1)
                    writer.add_scalar('val-mo-accuracy/hv', np.mean(epoch_val_acc['hv']), i+1)
                    print(f"==>> "
                          f"loss {np.mean(epoch_val_loss['hv']):.3f}, "
                          f"accuracy {np.mean(epoch_val_acc['hv']):.3f}.")

                if args['train.best_criteria'] == 'cluster':
                    '''evaluation acc based on cluster acc'''
                    avg_val_loss, avg_val_acc = avg_val_cluster_loss, avg_val_cluster_acc
                elif args['train.best_criteria'] == 'hv':
                    '''evaluation acc based on hv acc/loss (the larger the better)'''
                    # epoch_val_acc or epoch_val_loss
                    avg_val_loss, avg_val_acc = avg_val_cluster_loss, np.mean(epoch_val_loss['hv'])
                else:
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

    # run testing
    from test_extractor_pa import main as test
    test(test_model='best')
    print("↑↑ best model")
    test(test_model='last')
    print("↑↑ last model")

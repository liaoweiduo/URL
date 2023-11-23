"""
This code allows you to train a domain classifier.

Author: Weiduo Liao
Date: 2023.11.20
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
from models.losses import cross_entropy_loss, prototype_loss, DistillKL
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

    debugger = Debugger(level='DEBUG')

    '''--------------------'''
    '''Initialization Phase'''
    '''--------------------'''
    # defining the summary writer
    writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

    '''initialize pool'''
    pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
    pool.clear_clusters()
    pool.clear_buffer()

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
    if 'film' in args['train.cond_mode']:
        params = pmo.get_trainable_film_parameters()
    elif 'pa' in args['train.cond_mode']:
        params = pmo.get_trainable_pa_parameters()
    else:
        raise Exception(f"Un-implemented train.cond_mode {args['train.cond_mode']}")
    optimizer = get_optimizer(pmo, args, params=params)
    optimizer_selector = torch.optim.Adam(pmo.get_trainable_selector_parameters(True),
                                          lr=args['train.selector_learning_rate'],
                                          weight_decay=args['train.selector_learning_rate'] / 50)

    checkpointer = CheckPointer(args, pmo, optimizer=optimizer, save_all=True)
    start_iter, best_val_loss, best_val_acc = 0, 999999999, -1
    if os.path.isfile(checkpointer.last_ckpt) and args['train.resume']:
        start_iter, best_val_loss, best_val_acc = checkpointer.restore_model(ckpt='best', strict=False)
        # since only store film and selector
    else:
        print('No checkpoint restoration for pmo.')
    if args['train.lr_policy'] == "step":
        lr_manager = UniformStepLR(optimizer, args, start_iter)
        lr_manager_selector = UniformStepLR(optimizer_selector, args, start_iter)
    elif "exp_decay" in args['train.lr_policy']:
        lr_manager = ExpDecayLR(optimizer, args, start_iter)
        lr_manager_selector = ExpDecayLR(optimizer_selector, args, start_iter)
    else:       # elif "cosine" in args['train.lr_policy']:
        lr_manager = CosineAnnealRestartLR(optimizer, args, 0)       # start_iter
        lr_manager_selector = CosineAnnealRestartLR(optimizer_selector, args, 0)       # start_iter

    model_eval(pmo)
    model_eval(url)

    def init_train_log():
        log = {}
        # Tag: acc/loss
        log['mo_df'] = pd.DataFrame(columns=['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value'])
        # 'Inner_lr', 'Exp', 'Logit_scale',
        log['scaler_df'] = pd.DataFrame(columns=['Tag', 'Idx', 'Value'])

        return log

    epoch_log = init_train_log()
    val_log = init_train_log()

    pop_labels = [
        f"p{idx}" if idx < args['train.n_obj'] else f"m{idx - args['train.n_obj']}"
        for idx in range(args['train.n_mix'] + args['train.n_obj'])
    ]  # ['p0', 'p1', 'm0', 'm1']

    '''Load training data'''
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
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

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']
        print(f'\n>>>> Train start from {start_iter}.')
        for i in tqdm(range(start_iter, max_iter), ncols=100):      # every iter, load one task from all loaders
            print(f"\n>> Iter: {i}, collect training samples: ")
            '''obtain tasks from train_loaders and put to buffer'''
            # loading images and labels
            numpy_samples, sample_domain_names = [], []
            for t_indx, (name, train_loader) in enumerate(train_loaders.items()):
                numpy_sample = train_loader.get_train_task(session, d='numpy')
                numpy_samples.append(numpy_sample)
                sample_domain_names.append(name)
                sample = task_to_device(numpy_sample, device)

                context_images, target_images = sample['context_images'], sample['target_images']
                context_labels, target_labels = sample['context_labels'], sample['target_labels']
                context_gt_labels, target_gt_labels = sample['context_gt'], sample['target_gt']

                '''samples put to buffer'''
                task_images = torch.cat([context_images, target_images]).cpu()
                gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
                domain = np.array([t_indx] * len(gt_labels))  # [domain, domain, domain,...]
                with torch.no_grad():
                    task_features = pmo.embed(torch.cat([context_images, target_images]))
                    _, selection_info = pmo.selector(
                        task_features, gumbel=False, hard=False, average=False)  # [bs, n_clusters]
                    similarities = selection_info['y_soft'].cpu().numpy()  # [bs, n_clusters]
                # similarities = np.array([0] * len(gt_labels))  # no use

                not_full = pool.put_buffer(
                    task_images, {'domain': domain, 'gt_labels': gt_labels,
                                  'similarities': similarities, 'features': task_features.cpu().numpy()},
                    maintain_size=False)
            print(f'Buffer contains {len(pool.buffer)} classes.')
            # pool.buffer_backup = copy.deepcopy(pool.buffer)
            # pool.buffer = copy.deepcopy(pool.buffer_backup)

            '''buffer -> clusters'''
            pool.clear_clusters()
            pool.buffer2cluster()
            pool.clear_buffer()

            '''selection CE loss on all clusters'''
            print(f"\n>> Iter: {i}, clustering ce loss calculation: ")
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
            ce_loss = fn(dist, cluster_labels)
            epoch_log['scaler_df'] = pd.concat([
                epoch_log['scaler_df'], pd.DataFrame.from_records([{
                    'Tag': 'ce/loss', 'Idx': 0, 'Value': ce_loss.item()}])])

            '''multiple mo sampling'''
            print(f"\n>> Iter: {i}, start mo sampling: ")
            num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]
            ncc_losses_mo = dict()  # f'p{task_idx}_o{obj_idx}'
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

                    '''use url with pa'''
                    model = url
                    with torch.no_grad():
                        task_features = pmo.embed(
                            torch.cat([task['context_images'], task['target_images']]))
                        context_features = model.embed(task['context_images'])
                        context_labels = task['context_labels']

                    selection, selection_info = pmo.selector(
                        task_features, gumbel=False, hard=False)
                    vartheta = [torch.mm(selection, pmo.pas.detach().flatten(1)).view(512, 512, 1, 1)]
                    # detach from pas to only train clustering and not train pas

                    max_iter, inner_lr = 1, 1
                    selection_params = pa(context_features, context_labels, max_iter=max_iter, lr=inner_lr,
                                          distance=args['test.distance'],
                                          vartheta_init=[vartheta, torch.optim.Adadelta(vartheta, lr=inner_lr)],
                                          create_graph=True)
                    '''forward to get mo matrix'''
                    for obj_idx in range(len(selected_cluster_idxs)):       # 2
                        obj_context_images = torch_tasks[obj_idx]['context_images']
                        obj_target_images = torch_tasks[obj_idx]['target_images']
                        obj_context_labels = torch_tasks[obj_idx]['context_labels']
                        obj_target_labels = torch_tasks[obj_idx]['target_labels']

                        obj_context_features = apply_selection(model.embed(obj_context_images),
                                                               selection_params)
                        obj_target_features = apply_selection(model.embed(obj_target_images),
                                                              selection_params)

                        obj_loss, stats_dict, _ = prototype_loss(
                            obj_context_features, obj_context_labels, obj_target_features, obj_target_labels,
                            distance=args['test.distance'])
                        if f'p{task_idx}_o{obj_idx}' in ncc_losses_mo.keys():       # collect n_mo data
                            ncc_losses_mo[f'p{task_idx}_o{obj_idx}'].append(obj_loss)
                        else:
                            ncc_losses_mo[f'p{task_idx}_o{obj_idx}'] = [obj_loss]

                        epoch_log['mo_df'] = pd.concat([
                            epoch_log['mo_df'], pd.DataFrame.from_records([
                                {'Tag': 'loss', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': 0,
                                 'Value': stats_dict['loss']},
                                {'Tag': 'acc', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': 0,
                                 'Value': stats_dict['acc']}])])

            for task_idx, task in enumerate(torch_tasks):  # only visual the last task
                debugger.write_task(pmo, task, pop_labels[task_idx], i=i, writer=writer)

            '''calculate HV loss for n_mo matrix'''
            ref = args['train.ref']
            ncc_losses_multi_obj = torch.stack([torch.stack([
                torch.mean(torch.stack(ncc_losses_mo[f'p{task_idx}_o{obj_idx}']))
                for task_idx in range(args['train.n_obj'] + args['train.n_mix'])
            ]) for obj_idx in range(args['train.n_obj'])])      # [2, 4]
            hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
            epoch_log['scaler_df'] = pd.concat([
                epoch_log['scaler_df'], pd.DataFrame.from_records([{
                    'Tag': 'hv_loss', 'Idx': 0, 'Value': hv_loss.item()}])])

            # '''hv loss to average'''
            # hv_loss = hv_loss / args['train.n_mo']

            '''backward ce_coef * clustering_ce_loss + hv_coef * hv_loss'''
            ce_loss = ce_loss * args['train.ce_coefficient']
            hv_loss = hv_loss * args['train.hv_coefficient']
            loss = ce_loss + hv_loss
            # '''step coefficient from 0 to hv_coefficient (default: 1.0)'''
            # hv_loss = hv_loss * (args['train.hv_coefficient'] * min(i * 5, max_iter) / max_iter)

            optimizer_selector.zero_grad()
            loss.backward()

            '''debug'''
            debugger.print_grad(pmo, key='selector', prefix=f'iter{i} after ce_loss and hv_loss backward:\n')

            optimizer_selector.step()

            '''mov_avg update pool's centers'''
            if args['train.cluster_center_mode'] == 'mov_avg':
                print(f"\n>> Iter: {i}, update prototypes: ")
                centers = []
                for cluster in pool.clusters:
                    # cat features and forward with average embedding
                    features = torch.cat([torch.from_numpy(cls['features']) for cls in cluster]).to(device)
                    with torch.no_grad():
                        _, selection_info = pmo.selector(
                            features, gumbel=False, average=True)
                        center = selection_info['embeddings']  # [1, 64]
                    centers.append(center)
                centers = torch.stack(centers)

                pmo.selector.update_prototypes(centers)

            debugger.print_prototype_change(pmo, i=i, writer=writer)

            '''Update condition model with some training samples'''
            print(f"\n>> Iter: {i}, update pmo.pas: len(tasks): {len(numpy_samples)}.")
            task_losses = []
            for task_id, numpy_sample in enumerate(numpy_samples):
                task = task_to_device(numpy_sample, device)
                '''use url with pa'''
                model = url
                with torch.no_grad():
                    context_features = model.embed(task['context_images'])
                    context_labels = task['context_labels']
                    target_features = model.embed(task['target_images'])
                    target_labels = task['target_labels']

                    task_features = pmo.embed(
                        torch.cat([task['context_images'], task['target_images']]))
                    selection, selection_info = pmo.selector(
                        task_features, gumbel=False, hard=False)

                selection_params = [torch.mm(selection.detach(), pmo.pas.flatten(1)).view(512, 512, 1, 1)]
                # detach from selection to prevent train the selector

                selected_context = apply_selection(context_features, selection_params)
                selected_target = apply_selection(target_features, selection_params)

                task_loss, stats_dict, _ = prototype_loss(
                    selected_context, context_labels, selected_target, target_labels,
                    distance=args['test.distance'])
                task_losses.append(task_loss)
                epoch_log['scaler_df'] = pd.concat([
                    epoch_log['scaler_df'], pd.DataFrame.from_records([
                        {'Tag': 'task/loss', 'Idx': 0, 'Value': stats_dict['loss']},
                        {'Tag': 'task/acc', 'Idx': 0, 'Value': stats_dict['acc']}])])

            loss = torch.mean(torch.stack(task_losses))
            optimizer.zero_grad()
            loss.backward()

            '''debug'''
            debugger.print_grad(pmo, key='pas', prefix=f'iter{i} after task_loss backward:\n')

            optimizer.step()

            lr_manager.step(i)
            lr_manager_selector.step(i)

            '''log iter-wise params change'''
            writer.add_scalar('params/learning_rate', optimizer.param_groups[0]['lr'], i + 1)

            if (i + 1) % args['train.summary_freq'] == 0:
                print(f">> Iter: {i + 1}, train summary:")
                '''save train_log'''
                epoch_train_history = dict()
                if os.path.exists(os.path.join(args['out.dir'], 'summary', 'train_log.pickle')):
                    epoch_train_history = pickle.load(
                        open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'rb'))
                epoch_train_history[i + 1] = epoch_log.copy()
                with open(os.path.join(args['out.dir'], 'summary', 'train_log.pickle'), 'wb') as f:
                    pickle.dump(epoch_train_history, f)

                '''write pool'''
                debugger.write_pool(pool, i=i, writer=writer, prefix=f'pool')

                '''write mo image'''
                debugger.write_mo(epoch_log['mo_df'], pop_labels, i=i, writer=writer, target='acc')
                debugger.write_mo(epoch_log['mo_df'], pop_labels, i=i, writer=writer, target='loss')

                '''write hv acc/loss'''
                debugger.write_hv(epoch_log['mo_df'], ref=0, writer=writer, target='acc')  # 0
                debugger.write_hv(epoch_log['mo_df'], ref=args['train.ref'], writer=writer, target='loss')
                '''write avg_span acc/loss: E_i(max(f_i) - min(f_i))'''
                debugger.write_avg_span(epoch_log['mo_df'], writer=writer, target='acc')
                debugger.write_avg_span(epoch_log['mo_df'], writer=writer, target='loss')

                debugger.write_scaler(epoch_log['scaler_df'], key='ce/loss', i=i, writer=writer)
                debugger.write_scaler(epoch_log['scaler_df'], key='hv_loss', i=i, writer=writer)
                debugger.write_scaler(epoch_log['scaler_df'], key='task/loss', i=i, writer=writer)
                debugger.write_scaler(epoch_log['scaler_df'], key='task/acc', i=i, writer=writer)

                epoch_log = init_train_log()

            if (i + 1) % args['train.eval_freq'] == 0:   #  or i == 0:          # eval at init
                print(f"\n>> Iter: {i + 1}, evaluation:")

                # eval mode
                model_eval(pmo)
                model_eval(url)

                '''nvidia-smi'''
                print(os.system('nvidia-smi'))

                val_pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
                val_pool.centers = pool.centers     # same centers and device as train_pool; no use
                val_pool.eval()
                val_pool.clear_clusters()
                val_pool.clear_buffer()

                for j in tqdm(range(args['train.eval_size']), ncols=100):
                    '''obtain 1 task from all val_loader'''
                    for v_indx, valset in enumerate(valsets):
                        samples = val_loader.get_validation_task(session, valset, d=device)
                        context_images, target_images = samples['context_images'], samples['target_images']
                        context_labels, target_labels = samples['context_labels'], samples['target_labels']
                        context_gt_labels, target_gt_labels = samples['context_gt'], samples['target_gt']
                        domain = v_indx

                        '''eval task loss use url with pa'''
                        model = url
                        with torch.no_grad():
                            context_features = model.embed(context_images)
                            target_features = model.embed(target_images)
                            task_features = pmo.embed(torch.cat([context_images, target_images]))
                            selection, selection_info = pmo.selector(
                                task_features, gumbel=False, hard=False)

                            selection_params = [torch.mm(selection, pmo.pas.flatten(1)).view(512, 512, 1, 1)]

                            selected_context = apply_selection(context_features, selection_params)
                            selected_target = apply_selection(target_features, selection_params)

                        _, stats_dict, _ = prototype_loss(
                            selected_context, context_labels, selected_target, target_labels,
                            distance=args['test.distance'])
                        val_log['scaler_df'] = pd.concat([
                            val_log['scaler_df'], pd.DataFrame.from_records([
                                {'Tag': f'task_loss/{valset}', 'Idx': 0, 'Value': stats_dict['loss']},
                                {'Tag': f'task_acc/{valset}', 'Idx': 0, 'Value': stats_dict['acc']}])])

                        '''samples put to val buffer'''
                        task_images = torch.cat([context_images, target_images]).cpu()
                        gt_labels = torch.cat([context_gt_labels, target_gt_labels]).cpu().numpy()
                        domain = np.array([domain] * len(gt_labels))

                        with torch.no_grad():
                            _, selection_info = pmo.selector(
                                task_features, gumbel=False, average=False)  # [bs, n_clusters]
                            similarities = selection_info[
                                'y_soft'].cpu().numpy()  # [bs, n_clusters]

                        not_full = val_pool.put_buffer(
                            task_images, {'domain': domain, 'gt_labels': gt_labels,
                                          'similarities': similarities, 'features': task_features.cpu().numpy()},
                            maintain_size=False)

                    verbose = False
                    if verbose:
                        print(f'Buffer contains {len(val_pool.buffer)} classes.')

                    '''buffer -> clusters'''
                    val_pool.clear_clusters()
                    val_pool.buffer2cluster()
                    val_pool.clear_buffer()

                    '''repeat collecting MO acc on val pool'''
                    num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in
                                         val_pool.current_classes()]
                    ncc_losses_mo = dict()  # f'p{task_idx}_o{obj_idx}'
                    for mo_train_idx in range(args['train.n_mo']):
                        '''check pool has enough samples and generate 1 setting'''
                        n_way, n_shot, n_query = available_setting(num_imgs_clusters, args['train.mo_task_type'],
                                                                   min_available_clusters=args['train.n_obj'])
                        if n_way == -1:  # not enough samples
                            print(f"==>> val_pool has not enough samples. skip MO evaluation this iter.")
                            break

                        available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

                        selected_cluster_idxs = sorted(np.random.choice(
                            available_cluster_idxs, args['train.n_obj'], replace=False))

                        torch_tasks = []
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

                        '''obtain ncc loss multi-obj matrix'''
                        for task_idx, task in enumerate(torch_tasks):

                            '''use url with pa'''
                            model = url
                            with torch.no_grad():
                                task_features = pmo.embed(
                                    torch.cat([task['context_images'], task['target_images']]))
                                context_features = model.embed(task['context_images'])
                                context_labels = task['context_labels']

                            selection, selection_info = pmo.selector(
                                task_features, gumbel=False, hard=False)
                            vartheta = [
                                torch.mm(
                                    selection.detach(), pmo.pas.detach().flatten(1)
                                ).view(512, 512, 1, 1).requires_grad_(True)]
                            # detach from the learned one

                            inner_lr = 1
                            for inner_idx, selection_params in enumerate(
                                    pa(context_features, context_labels, max_iter=40, lr=inner_lr,
                                       distance=args['test.distance'],
                                       vartheta_init=[vartheta, torch.optim.Adadelta(vartheta, lr=inner_lr)],
                                       return_iterator=True)):
                                '''inner acc/loss'''
                                with torch.no_grad():
                                    selected_context = apply_selection(context_features, selection_params)
                                _, stats_dict, _ = prototype_loss(
                                    selected_context, context_labels,
                                    selected_context, context_labels, distance=args['test.distance'])

                                '''log'''
                                val_log['scaler_df'] = pd.concat([
                                    val_log['scaler_df'], pd.DataFrame.from_records([
                                        {'Tag': f'inner/loss/{task_idx}', 'Idx': inner_idx, 'Value': stats_dict['loss']},
                                        {'Tag': f'inner/acc/{task_idx}', 'Idx': inner_idx, 'Value': stats_dict['acc']}])])

                                '''forward to get mo'''
                                for obj_idx in range(len(selected_cluster_idxs)):  # 2
                                    obj_context_images = torch_tasks[obj_idx]['context_images']
                                    obj_target_images = torch_tasks[obj_idx]['target_images']
                                    obj_context_labels = torch_tasks[obj_idx]['context_labels']
                                    obj_target_labels = torch_tasks[obj_idx]['target_labels']
                                    with torch.no_grad():
                                        obj_context_features = apply_selection(model.embed(obj_context_images),
                                                                               selection_params)
                                        obj_target_features = apply_selection(model.embed(obj_target_images),
                                                                              selection_params)

                                    obj_loss, stats_dict, _ = prototype_loss(
                                        obj_context_features, obj_context_labels,
                                        obj_target_features, obj_target_labels,
                                        distance=args['test.distance'])
                                    if f'p{task_idx}_o{obj_idx}' in ncc_losses_mo.keys():  # collect n_mo data
                                        ncc_losses_mo[f'p{task_idx}_o{obj_idx}'].append(obj_loss)
                                    else:
                                        ncc_losses_mo[f'p{task_idx}_o{obj_idx}'] = [obj_loss]

                                    val_log['mo_df'] = pd.concat([
                                        val_log['mo_df'], pd.DataFrame.from_records([
                                            {'Tag': 'loss', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx,
                                             'Value': stats_dict['loss']},
                                            {'Tag': 'acc', 'Pop_id': task_idx, 'Obj_id': obj_idx, 'Inner_id': inner_idx,
                                             'Value': stats_dict['acc']}])])

                '''write val pool'''
                debugger.write_pool(val_pool, i=i, writer=writer, prefix=f'val_pool')

                '''write mo image'''
                debugger.write_mo(val_log['mo_df'], pop_labels, i=i, writer=writer, target='acc', prefix='val_image')
                debugger.write_mo(val_log['mo_df'], pop_labels, i=i, writer=writer, target='loss', prefix='val_image')

                '''write hv acc/loss'''
                debugger.write_hv(val_log['mo_df'], ref=0, writer=writer, target='acc', prefix='val_hv')  # 0
                debugger.write_hv(val_log['mo_df'], ref=args['train.ref'], writer=writer, target='loss', prefix='val_hv')
                '''write avg_span acc/loss: E_i(max(f_i) - min(f_i))'''
                val_avg_span_acc = debugger.write_avg_span(
                    val_log['mo_df'], writer=writer, target='acc', prefix='val_avg_span')
                val_avg_span_loss = debugger.write_avg_span(
                    val_log['mo_df'], writer=writer, target='loss', prefix='val_avg_span')

                '''task/loss'''
                avg_log = {'loss': [], 'acc': []}
                for v_indx, valset in enumerate(valsets):
                    avg_log['loss'].append(debugger.write_scaler(
                        val_log['scaler_df'], key=f'task_loss/{valset}',
                        i=i, writer=writer, prefix='val_'))
                    avg_log['acc'].append(debugger.write_scaler(
                        val_log['scaler_df'], key=f'task_acc/{valset}',
                        i=i, writer=writer, prefix='val_'))

                '''write summaries averaged over sources'''
                avg_val_source_loss = np.mean(avg_log['loss'])
                avg_val_source_acc = np.mean(avg_log['acc'])
                writer.add_scalar(f"val_task_loss/avg_val_source_loss", avg_val_source_loss, i + 1)
                writer.add_scalar(f"val_task_acc/avg_val_source_acc", avg_val_source_acc, i + 1)
                print(f"==>> val: avg_loss {avg_val_source_loss:.3f}, "
                      f"avg_accuracy {avg_val_source_acc:.3f}.")

                '''f'inner/loss/{task_idx}' and acc'''
                for task_idx in range(args['train.n_obj'] + args['train.n_mix']):
                    debugger.write_inner(val_log['scaler_df'], key=f'inner/loss/{task_idx}',
                                         i=i, writer=writer, prefix='val_')
                    debugger.write_inner(val_log['scaler_df'], key=f'inner/acc/{task_idx}',
                                         i=i, writer=writer, prefix='val_')

                '''best_criteria'''
                if args['train.best_criteria'] == 'avg_span':
                    avg_val_loss, avg_val_acc = avg_val_source_loss, val_avg_span_loss
                else:
                    avg_val_loss, avg_val_acc = avg_val_source_loss, avg_val_source_acc

                # saving checkpoints
                if avg_val_acc > best_val_acc and i > 0:    # do not consider init
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_log': epoch_log, 'val_log': val_log}
                checkpointer.save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=optimizer,
                        state_dict=pmo.get_state_dict(whole=False), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': i + 1, 'val_log': val_log}, f)

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

    # '''nvidia-smi'''
    # print(os.system('nvidia-smi'))

    # run testing
    from test_extractor_pa import main as test
    test(test_model='best')
    print("↑↑ best model")
    test(test_model='last')
    print("↑↑ last model")

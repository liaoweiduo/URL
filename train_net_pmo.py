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
from models.adaptors import adaptor
from utils import Accumulator, device, devices, cluster_device, set_determ, check_dir
from config import args

from pmo_utils import Pool, Clusterer, Mixer, prototype_similarity, cal_hv_loss, cal_hv, draw_objs, draw_heatmap, pmo_embed

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

        print(f'PMO devices: {devices}.')
        print(f'Cluster network device: {cluster_device}.')
        print(f'Mult-obj NCC loss calculation: {device}.')

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

        val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type=args['train.type'])

        '''initialize models and optimizer'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        # init all starting issues for M(8) models.
        model_names = [f'M{m_indx}' for m_indx in range(args['model.num_clusters'])]    # M0-M7
        cluster_model_name = 'C-net'
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]        # C0-C7

        # load url model as the frozen feature extractor
        url_args = copy.deepcopy(args)
        url_args['model.pretrained'] = True
        cluster_url = get_model(None, url_args, d=cluster_device, freeze_fe=True, base_network_name='url')
        model_url = get_model(None, url_args, d=device, freeze_fe=True, base_network_name='url')
        cluster_url.eval()
        model_url.eval()

        # pmo model
        pmo = adaptor(num_datasets=args['model.num_clusters'],
                      dim_in=512, opt=args['pmo.opt'])
        pmo.to_device(devices)
        pmo_optimizer = get_optimizer(pmo, args, params=pmo.get_parameters())
        pmo_checkpointer = CheckPointer(args, pmo, optimizer=pmo_optimizer)
        if os.path.isfile(pmo_checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                pmo_checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration for pmo.')
        if args['train.lr_policy'] == "step":
            pmo_lr_manager = UniformStepLR(pmo_optimizer, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            pmo_lr_manager = ExpDecayLR(pmo_optimizer, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            pmo_lr_manager = CosineAnnealRestartLR(pmo_optimizer, args, start_iter)

        '''initialize the clustering model and optimizer'''
        model_args_with_name = copy.deepcopy(args)
        model_args_with_name['model.name'] = cluster_model_name     # 'C-net'
        clusterer = Clusterer().to(cluster_device)
        cluster_optimizer = get_optimizer(clusterer, model_args_with_name, params=clusterer.get_parameters())
        # restoring the last checkpoint
        cluster_checkpointer = CheckPointer(model_args_with_name, clusterer, optimizer=cluster_optimizer)
        if os.path.isfile(cluster_checkpointer.last_ckpt) and args['train.resume']:
            start_iter, best_val_loss, best_val_acc = \
                cluster_checkpointer.restore_model(ckpt='last')
        else:
            print('No checkpoint restoration for cluster model.')
        # define learning rate policy
        if args['train.lr_policy'] == "step":
            cluster_lr_manager = UniformStepLR(cluster_optimizer, args, start_iter)
        elif "exp_decay" in args['train.lr_policy']:
            cluster_lr_manager = ExpDecayLR(cluster_optimizer, args, start_iter)
        elif "cosine" in args['train.lr_policy']:
            cluster_lr_manager = CosineAnnealRestartLR(cluster_optimizer, args, start_iter)

        # defining the summary writer
        writer = SummaryWriter(check_dir(os.path.join(args['out.dir'], 'summary'), False))

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'], mode=args['train.cluster_center_mode'])
        if start_iter > 0:
            pool.restore(start_iter)      # restore pool cluster centers.
        pool = pool.to(cluster_device)

        '''initialize mixer'''
        mixer = Mixer(mode=args['train.mix_mode'])

        '''-------------'''
        '''Training loop'''
        '''-------------'''
        max_iter = args['train.max_iter']

        def init_train_log():
            epoch_loss = {model_name: {name: [] for name in cluster_names} for model_name in model_names}
            if 'task' in args['train.loss_type']:
                epoch_loss.update({name: [] for name in trainsets})
            epoch_loss.update({
                obj_idx: {
                    pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_loss['hv_loss'], epoch_loss['hv'] = [], []

            epoch_acc = {model_name: {name: [] for name in cluster_names} for model_name in model_names}
            if 'task' in args['train.loss_type']:
                epoch_acc.update({name: [] for name in trainsets})
            epoch_acc.update({
                obj_idx: {
                    pop_idx: [] for pop_idx in range(args['train.n_mix'] + args['train.n_obj'])
                } for obj_idx in range(args['train.n_obj'])})
            epoch_acc['hv'] = []
            return epoch_loss, epoch_acc

        def model_train():
            # train mode
            pmo.train()
            clusterer.train()
            pool.train()

        def model_eval():
            # eval mode
            pmo.eval()
            clusterer.eval()
            pool.eval()

        def zero_grad():
            pmo_optimizer.zero_grad()
            cluster_optimizer.zero_grad()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.zero_grad()

        def update_step(idx):
            pmo_optimizer.step()
            cluster_optimizer.step()
            if args['train.cluster_center_mode'] == 'trainable':
                pool.optimizer.step()

            pmo_lr_manager.step(idx)
            cluster_lr_manager.step(idx)
            if args['train.cluster_center_mode'] == 'trainable':
                pool.lr_manager.step(idx)

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

            '''----------------'''
            '''Clustering Phase'''
            '''----------------'''
            '''put the classes in the clusters to the buffer'''
            pool.cluster_put_into_buffer()

            '''obtain tasks from train_loaders'''
            for t_indx, trainset in enumerate(trainsets):
                num_task_per_batch = 1
                if trainset == 'ilsvrc_2012':
                    num_task_per_batch = 2

                for _ in range(num_task_per_batch):
                    sample_numpy = train_loaders[trainset].get_train_task(session, d='numpy')
                    images_numpy = np.concatenate([sample_numpy['context_images'], sample_numpy['target_images']])
                    re_labels_numpy = np.concatenate([sample_numpy['context_labels'], sample_numpy['target_labels']])
                    gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                    domain = t_indx

                    if (
                            'task' in args['train.loss_type'] and i < args['train.num_iters_cal_task_loss']
                    ) or args['train.loss_type'] == 'task':
                        '''cluster for task'''
                        cluster_idx, grad_one, similarity, task_centroid = pool.cluster_for_task(
                            images_numpy, cluster_url, clusterer
                        )
                        grad_ones = {
                            'context_grad_ones': grad_one.repeat(sample_numpy['context_images'].shape),
                            # shape [n_shot*n_way, 3, 84, 84]
                            'target_grad_ones': grad_one.repeat(sample_numpy['target_images'].shape),
                            # shape [n_query*n_way, 3, 84, 84]
                        }

                        '''obtain and backward task ncc loss'''
                        context_features_list, context_labels_list = pmo_embed(
                            sample_numpy['context_images'],
                            sample_numpy['context_labels'],
                            grad_ones['context_grad_ones'],
                            model_url, pmo, [cluster_idx])
                        target_features_list, target_labels_list = pmo_embed(
                            sample_numpy['target_images'],
                            sample_numpy['target_labels'],
                            grad_ones['target_grad_ones'],
                            model_url, pmo, [cluster_idx])
                        task_loss, stats_dict, _ = prototype_loss(
                            context_features_list[0], context_labels_list[0],
                            target_features_list[0], target_labels_list[0],
                            distance=args['test.distance'])

                        task_loss.backward()

                        '''log task loss and acc'''
                        epoch_loss[trainset].append(stats_dict['loss'])
                        epoch_acc[trainset].append(stats_dict['acc'])
                        # ilsvrc_2012 has 2 times larger len than others.

                    '''pmo method: assign buffer'''
                    pool.task_put_into_buffer(images_numpy, re_labels_numpy, gt_labels_numpy, domain)

            '''perform clustering on buffer'''
            if args['train.loss_type'] == 'task':
                with torch.no_grad():
                    pool.cluster_and_assign_from_buffer(cluster_url, clusterer)
            else:
                pool.cluster_and_assign_from_buffer(cluster_url, clusterer)

            '''--------------'''
            '''Training Phase'''
            '''--------------'''
            '''select args['train.n_obj'] clusters'''
            # assert args['train.n_obj'] == 2
            available_cluster_idxs = []
            for idx, classes in enumerate(pool.current_classes()):
                # if len(classes) >= args['train.n_way']:
                num_imgs = np.array([cls[1] for cls in classes])
                if len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']]) >= args['train.n_way']:
                    available_cluster_idxs.append(idx)
            if len(available_cluster_idxs) < args['train.n_obj']:      # not enough available clusters
                print(f'Iter: {i + 1}, not enough available clusters.')
                continue
            else:
                for mo_train_idx in range(args['train.n_mo']):         # repeat collecting MO loss
                    selected_cluster_idxs = sorted(np.random.choice(
                        available_cluster_idxs, args['train.n_obj'], replace=False))
                    # which is also devices idx
                    # device_list = list(set([devices[idx] for idx in selected_cluster_idxs]))    # unique devices

                    '''sample pure tasks from clusters in selected_cluster_idxs'''
                    numpy_tasks, grad_ones_list = [], []
                    for cluster_idx in selected_cluster_idxs:
                        numpy_pure_task, grad_ones = pool.episodic_sample(cluster_idx)
                        numpy_tasks.append(numpy_pure_task)
                        grad_ones_list.append(grad_ones)

                    '''sample mix tasks by mixer'''
                    for mix_id in range(args['train.n_mix']):
                        [numpy_mix_task, grad_ones], _ = mixer.mix(
                            task_list=[pool.episodic_sample(idx) for idx in selected_cluster_idxs],
                            mix_id=mix_id
                        )
                        numpy_tasks.append(numpy_mix_task)
                        grad_ones_list.append(grad_ones)

                    '''obtain ncc loss multi-obj matrix and put to last device'''
                    ncc_losses_multi_obj = []    # [4, 2]
                    for task_idx, (task, grad_ones) in enumerate(zip(numpy_tasks, grad_ones_list)):
                        if args['train.loss_type'] == 'task':
                            with torch.no_grad():
                                context_features_list, context_labels_list = pmo_embed(
                                    task['context_images'],
                                    task['context_labels'],
                                    grad_ones['context_grad_ones'],
                                    model_url, pmo, selected_cluster_idxs)
                                target_features_list, target_labels_list = pmo_embed(
                                    task['target_images'],
                                    task['target_labels'],
                                    grad_ones['target_grad_ones'],
                                    model_url, pmo, selected_cluster_idxs)
                        else:
                            context_features_list, context_labels_list = pmo_embed(
                                task['context_images'],
                                task['context_labels'],
                                grad_ones['context_grad_ones'],
                                model_url, pmo, selected_cluster_idxs)
                            target_features_list, target_labels_list = pmo_embed(
                                task['target_images'],
                                task['target_labels'],
                                grad_ones['target_grad_ones'],
                                model_url, pmo, selected_cluster_idxs)

                        losses = []     # [2,]
                        for obj_idx in range(len(selected_cluster_idxs)):       # obj_idx is the selected model
                            loss, stats_dict, _ = prototype_loss(
                                context_features_list[obj_idx], context_labels_list[obj_idx],
                                target_features_list[obj_idx], target_labels_list[obj_idx],
                                distance=args['test.distance'])

                            # loss for all tasks on 1 model on 1 device. to last device
                            losses.append(loss.to(device))
                            epoch_loss[obj_idx][task_idx].append(stats_dict['loss'])    # [2, 4]
                            epoch_acc[obj_idx][task_idx].append(stats_dict['acc'])
                            if task_idx < len(selected_cluster_idxs):       # [2, 2]

                                if task_idx == obj_idx and 'pure' in args['train.loss_type']:
                                    # backward pure loss on the corresponding model and cluster.
                                    loss.backward(retain_graph=True)

                                '''log model-specific pure loss'''
                                epoch_loss[model_names[selected_cluster_idxs[obj_idx]]][
                                    cluster_names[selected_cluster_idxs[task_idx]]
                                ].append(stats_dict['loss'])
                                epoch_acc[model_names[selected_cluster_idxs[obj_idx]]][
                                    cluster_names[selected_cluster_idxs[task_idx]]
                                ].append(stats_dict['acc'])
                        ncc_losses_multi_obj.append(torch.stack(losses))
                    ncc_losses_multi_obj = torch.stack(ncc_losses_multi_obj)   # shape [num_tasks, num_objs], [4, 2]
                    ncc_losses_multi_obj = ncc_losses_multi_obj.T       # [2, 4]

                    '''calculate HV loss'''
                    ref = args['train.ref']
                    hv_loss = cal_hv_loss(ncc_losses_multi_obj, ref)
                    epoch_loss['hv_loss'].append(hv_loss.item())

                    '''calculate HV value for mutli-obj loss and acc'''
                    obj = np.array([[
                        epoch_loss[obj_idx][task_idx][-1] for task_idx in range(len(numpy_tasks))     # this iter
                    ] for obj_idx in range(len(selected_cluster_idxs))])
                    hv = cal_hv(obj, ref, target='loss')
                    epoch_loss['hv'].append(hv)
                    obj = np.array([[
                        epoch_acc[obj_idx][task_idx][-1] for task_idx in range(len(numpy_tasks))
                    ] for obj_idx in range(len(selected_cluster_idxs))])
                    hv = cal_hv(obj, 0, target='acc')
                    epoch_acc['hv'].append(hv)

                    if 'hv' in args['train.loss_type']:
                        retain_graph = True if mo_train_idx < args['train.n_mo'] - 1 else False
                        hv_loss = hv_loss * args['train.hv_coefficient']
                        hv_loss.backward(retain_graph=retain_graph)

            update_step(i)

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
                for dataset_name in trainsets:
                    if len(epoch_loss[dataset_name]) > 0:
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
                                  cluster_optimizer.param_groups[0]['lr'], i+1)
                print(f"==>> loss/train/hv {np.mean(epoch_loss['hv']):.3f}, "
                      f"accuracy/train/hv {np.mean(epoch_acc['hv']):.3f}.")

                epoch_loss, epoch_acc = init_train_log()

                '''write pool images'''
                images = pool.current_images()
                for cluster_id, cluster in enumerate(images):
                    if len(cluster) > 0:
                        img_in_cluster = np.concatenate(cluster)
                        writer.add_images(f"image/pool-{cluster_id}", img_in_cluster, i+1)

                '''write pool similarities'''
                similarities = pool.current_similarities()
                for cluster_id, cluster in enumerate(similarities):
                    if len(cluster) > 0:
                        sim_in_cluster = np.stack(cluster)  # [num_cls, 8]
                        figure = draw_heatmap(sim_in_cluster)
                        writer.add_figure(f"image/pool-{cluster_id}-similarities", figure, i+1)

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
                            writer.add_figure(f"image/pool-{cluster_id}-assigns", figure, i+1)
                            figure = draw_heatmap(gat_in_cluster)
                            writer.add_figure(f"image/pool-{cluster_id}-gates", figure, i+1)

                '''write pure and mixed tasks'''
                for task_id, task in enumerate(numpy_tasks):
                    imgs = np.concatenate([task['context_images'], task['target_images']])
                    writer.add_images(f"image/task-{task_id}", imgs, i+1)

            '''----------'''
            '''Eval Phase'''
            '''----------'''
            # Evaluation inside the training loop
            if (i + 1) % args['train.eval_freq'] == 0:      # args['train.eval_freq']; 10 for DEBUG
                print(f"\n>> Iter: {i + 1}, evaluation:")
                # eval mode
                model_eval()

                val_pool = Pool(capacity=args['model.num_clusters'])
                val_pool.centers = pool.centers     # same centers and device as train_pool
                val_pool.eval()

                '''collect cluster_losses/accs for all models'''
                cluster_accs, cluster_losses = [[] for _ in range(args['model.num_clusters'])], \
                                               [[] for _ in range(args['model.num_clusters'])]
                for v_indx, valset in enumerate(valsets):
                    print(f"==>> collect classes from {valset}.")
                    for j in tqdm(range(args['train.eval_size']), ncols=100):
                        with torch.no_grad():
                            '''obtain 1 task from val_loader'''
                            sample_numpy = val_loader.get_validation_task(session, valset, d='numpy')
                            images_numpy = np.concatenate(
                                [sample_numpy['context_images'], sample_numpy['target_images']])
                            re_labels_numpy = np.concatenate(
                                [sample_numpy['context_labels'], sample_numpy['target_labels']])
                            gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                            domain = v_indx

                            val_pool.cluster_and_assign(
                                images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                cluster_url, clusterer, softmax_mode='softmax')

                            '''check if any cluster have sufficient class to construct 1 task'''
                            for idx, classes in enumerate(val_pool.current_classes()):
                                num_imgs = np.array([cls[1] for cls in classes])
                                available_way = len(num_imgs[num_imgs >= args['train.n_shot'] + args['train.n_query']])
                                if available_way >= args['train.n_way']:
                                    # enough classes to construct 1 task
                                    # then use all available classes to construct 1 task
                                    numpy_task, grad_ones = val_pool.episodic_sample(
                                        idx, n_way=available_way, remove_sampled_classes=True)

                                    context_features_list, context_labels_list = pmo_embed(
                                        numpy_task['context_images'],
                                        numpy_task['context_labels'],
                                        grad_ones['context_grad_ones'],
                                        model_url, pmo, [idx])
                                    target_features_list, target_labels_list = pmo_embed(
                                        numpy_task['target_images'],
                                        numpy_task['target_labels'],
                                        grad_ones['target_grad_ones'],
                                        model_url, pmo, [idx])

                                    context_features = context_features_list[0]
                                    target_features = target_features_list[0]
                                    context_labels = context_labels_list[0]
                                    target_labels = target_labels_list[0]
                                    _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                                      target_features, target_labels)

                                    cluster_losses[idx].append(stats_dict['loss'])
                                    cluster_accs[idx].append(stats_dict['acc'])

                '''write and print'''
                for cluster_idx, (loss_list, acc_list) in enumerate(zip(cluster_losses, cluster_accs)):
                    if len(loss_list) > 0:
                        cluster_acc, cluster_loss = np.mean(acc_list).item() * 100, np.mean(loss_list).item()

                        epoch_val_loss[model_names[cluster_idx]].append(cluster_loss)
                        epoch_val_acc[model_names[cluster_idx]].append(cluster_acc)
                        writer.add_scalar(f"loss/val/{model_names[cluster_idx]}", cluster_loss, i+1)
                        writer.add_scalar(f"accuracy/val/{model_names[cluster_idx]}", cluster_acc, i+1)
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc {cluster_acc:.2f}%, val_loss {cluster_loss:.3f}")

                    else:   # no class(task) is assign to this cluster
                        print(f"==>> {model_names[cluster_idx]}: "
                              f"val_acc No value, val_loss No value")

                # write summaries averaged over clusters
                avg_val_loss, avg_val_acc = np.mean(np.concatenate(cluster_losses)), np.mean(np.concatenate(cluster_accs))
                writer.add_scalar(f"loss/val/avg_val_loss", avg_val_loss, i+1)
                writer.add_scalar(f"accuracy/val/avg_val_acc", avg_val_acc, i+1)

                # saving checkpoints
                if avg_val_acc > best_val_acc:
                    best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                    is_best = True
                    print('====>> Best model so far!')
                else:
                    is_best = False
                extra_dict = {'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc,
                              'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                pmo_checkpointer.save_checkpoint(
                        i, best_val_acc, best_val_loss,
                        is_best, optimizer=pmo_optimizer,
                        state_dict=pmo.get_state_dict(), extra=extra_dict)
                cluster_checkpointer.save_checkpoint(
                    i, best_val_acc, best_val_loss,
                    is_best, optimizer=cluster_optimizer,
                    state_dict=clusterer.get_state_dict(), extra=extra_dict)

                '''save epoch_val_loss and epoch_val_acc'''
                with open(os.path.join(args['out.dir'], 'summary', 'val_log.pickle'), 'wb') as f:
                    pickle.dump({'epoch': start_iter + 1, 'loss': epoch_val_loss, 'acc': epoch_val_acc}, f)

                print(f"====>> Trained and evaluated at {i + 1}.\n")

                # saving pool
                pool.store(i, train_loaders, trainsets, is_best)

    '''Close the writers'''
    writer.close()

    if start_iter < max_iter:
        print(f"""Done training with best_mean_val_loss: {best_val_loss:.3f}, 
        best_avg_val_acc: {best_val_acc:.2f}%""")
    else:
        print(f"""No training happened. Loaded checkpoint at {start_iter}, while max_iter was {max_iter}""")


if __name__ == '__main__':
    train()

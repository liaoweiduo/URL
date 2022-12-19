"""
This code allows you to obtain classes mapping for trained pmo models.
A mapping json will be stored for further use.

Author: Weiduo Liao
Date: 2022.12.6
"""

import os
import sys
import pickle
import json
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
from utils import Accumulator, cluster_device, set_determ, check_dir
from config import args

from pmo_utils import Pool, map_re_label


def obtain():
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
        print(f'Test on: {testsets}.')

        print(f'Cluster network device: {cluster_device}.')

        train_loaders = dict()
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                args['map.target'], [trainset], valsets, testsets, test_type='5shot')
            num_train_classes[trainset] = train_loaders[trainset].num_classes(args['map.target'])
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        '''initialize and load model'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]

        cluster_model = get_model(None, args, d=cluster_device)
        cluster_optimizer = get_optimizer(cluster_model, args, params=cluster_model.get_parameters())
        # restoring the last checkpoint
        cluster_checkpointer = CheckPointer(args, cluster_model, optimizer=cluster_optimizer)
        if os.path.isfile(cluster_checkpointer.best_ckpt):
            start_iter, best_val_loss, best_val_acc = \
                cluster_checkpointer.restore_model(ckpt='best')
        else:
            raise Exception('No checkpoint restoration.')

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'])
        pool.restore(start_iter, center_filename='pool_best.npy')      # restore best pool cluster centers.
        pool.to(cluster_device)

        '''--------------'''
        '''Obtain mapping'''
        '''--------------'''
        class_mapping = dict()      # {trainset: {gt_label: cluster_idx}}
        info = {'labels': [], 'cluster_idxs': [], 'class_centroids': [], 'sample_images': [], 'similarities': []}
        for t_indx, trainset in enumerate(trainsets):
            print(f"\n>> Obtain classes for {trainset}.")
            loader = train_loaders[trainset]
            class_mapping[trainset] = dict()
            num_classes = num_train_classes[trainset]
            with tqdm(total=num_classes, ncols=100) as pbar:
                classes = np.zeros(num_classes)     # [0, 0, ..., 0]
                while classes.sum() < num_classes:
                    '''obtain a task from loader'''
                    with torch.no_grad():
                        sample_numpy = loader.get_train_task(session, d='numpy')
                        images_numpy = np.concatenate([sample_numpy['context_images'], sample_numpy['target_images']])
                        re_labels_numpy = np.concatenate([sample_numpy['context_labels'], sample_numpy['target_labels']])
                        gt_labels_numpy = np.concatenate([sample_numpy['context_gt'], sample_numpy['target_gt']])
                        domain = t_indx

                        gt_label_set = np.unique(gt_labels_numpy)
                        unseen_labels = gt_label_set[classes[gt_label_set] == 0]      # e.g., [3]

                        if len(unseen_labels) != 0:
                            '''filter unseen classes to do clustering'''
                            unseen_mask = np.isin(gt_labels_numpy, unseen_labels)
                            images_numpy = images_numpy[unseen_mask]
                            re_labels_numpy = re_labels_numpy[unseen_mask]   # e.g., [1,1,1,3,3,3,7,7,7]
                            gt_labels_numpy = gt_labels_numpy[unseen_mask]

                            '''map re_label in the form [0,0,0,1,1,1,2,2,2]'''
                            re_labels_numpy = map_re_label(re_labels_numpy)

                            cluster_info = pool.cluster_and_assign(
                                images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                cluster_model,
                                softmax_mode='softmax',
                                update_cluster_centers=False, put=False)

                            labels = cluster_info['labels']                 # n_way * (gt, domain)
                            cluster_idxs = cluster_info['cluster_idxs']     # numpy [n_way,]

                            '''add mapping'''
                            class_mapping[trainset].update(
                                {label[0].item(): cluster_idx.item()
                                 for label, cluster_idx in zip(labels, cluster_idxs)})

                            classes[unseen_labels] = 1      # mark as seen
                            pbar.update(len(unseen_labels))

                            '''collect information for tsne'''
                            info['labels'].append(labels)
                            info['cluster_idxs'].append(cluster_idxs)
                            info['class_centroids'].append(cluster_info['class_centroids'])
                            info['sample_images'].append(cluster_info['sample_images'])
                            info['similarities'].append(cluster_info['similarities'])

        info['labels'] = np.concatenate(info['labels'])                     # [num_cls, gt, domain]
        info['cluster_idxs'] = np.concatenate(info['cluster_idxs'])         # [num_cls]
        info['class_centroids'] = np.concatenate(info['class_centroids'])   # [num_cls, emb_dim]
        info['sample_images'] = np.concatenate(info['sample_images'])       # [num_cls, c, h, w]
        info['similarities'] = np.concatenate(info['similarities'])         # [num_cls, num_cluster]

        '''assign relative label for each cluster'''
        '''collect labels for each cluster'''
        classes_in_cluster = {cluster_names[cluster_idx]: [] for cluster_idx in range(args['model.num_clusters'])}
        for t_idx, (trainset, class_map_dict) in enumerate(class_mapping.items()):
            for gt_label, cluster_idx in class_map_dict.items():
                gt_label_str = train_loaders[trainset].label_to_str((gt_label, 0))[0]
                re_label = len(classes_in_cluster[cluster_names[cluster_idx]])
                # domain is always 0 for single domain
                classes_in_cluster[cluster_names[cluster_idx]].append(
                    (
                        gt_label,
                        gt_label_str,
                        trainset,
                        re_label
                    )
                )
                class_map_dict[gt_label] = (
                    gt_label_str,
                    re_label,
                    cluster_names[cluster_idx]
                )

        '''save class_mapping'''
        with open(os.path.join(args['out.dir'], 'weights', 'pool',
                               f"class_mapping_{args['map.target']}.json"), 'w') as f:
            json.dump(
                {'epoch': start_iter + 1,
                 'classes_in_cluster': classes_in_cluster,
                 'class_mapping': class_mapping},
                f)
        '''save info for tsne'''
        with open(os.path.join(args['out.dir'], 'weights', 'pool',
                               f"class_mapping_info_{args['map.target']}.pickle"), 'wb') as f:
            pickle.dump({'info': info}, f)


if __name__ == '__main__':
    obtain()

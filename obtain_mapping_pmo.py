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
from utils import Accumulator, device, set_determ, check_dir
from config import args

from pmo_utils import Pool, Mixer, prototype_similarity, cal_hv_loss, cal_hv


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
        # print(f'Val on: {valsets}.')
        # print(f'Test on: {testsets}.')

        print(f'Cluster network device: {device}.')

        train_loaders = []
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders[trainset] = MetaDatasetEpisodeReader(
                'train', [trainset], valsets, testsets, test_type='5shot')
            num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        '''initialize and load model'''
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]

        cluster_model = get_model(None, args, d=device)
        cluster_optimizer = get_optimizer(cluster_model, args, params=cluster_model.get_parameters())
        # restoring the last checkpoint
        cluster_checkpointer = CheckPointer(args, cluster_model, optimizer=cluster_optimizer)
        if os.path.isfile(cluster_checkpointer.best_ckpt):
            start_iter, best_val_loss, best_val_acc = \
                cluster_checkpointer.restore_model(ckpt='best')
        else:
            raise Exception('No checkpoint restoration')

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'])
        pool.restore(start_iter, center_filename='pool_best.npy')      # restore best pool cluster centers.
        pool.to(device)

        '''--------------'''
        '''Obtain mapping'''
        '''--------------'''
        class_mapping = dict()      # {trainset: {gt_label: cluster_idx}}
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
                            re_labels_numpy = re_labels_numpy[unseen_mask]
                            gt_labels_numpy = gt_labels_numpy[unseen_mask]

                            cluster_info = pool.clustering(
                                images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                cluster_model,
                                softmax_mode='argmax',  # use argmax
                                update_cluster_centers=False, put=False)

                            labels = cluster_info['labels']
                            cluster_idxs = cluster_info['cluster_idxs']

                            '''add mapping'''
                            class_mapping[trainset].update(
                                {label[0]: cluster_idx for label, cluster_idx in zip(labels, cluster_idxs)})

                            '''collect information for tsne'''

                            TBD

                            classes[unseen_labels] = 1      # mark as seen
                            pbar.update(len(unseen_labels))

        '''assign relative label for each cluster'''
        '''collect labels for each cluster'''
        classes_in_cluster = {cluster_names[cluster_idx]: [] for cluster_idx in range(args['model.num_clusters'])}
        for t_idx, (trainset, class_map_dict) in enumerate(class_mapping.items()):
            for gt_label, cluster_idx in class_map_dict.items():
                classes_in_cluster[cluster_names[cluster_idx]].append(
                    (
                        gt_label,
                        train_loaders[t_idx].label_to_str((gt_label, 0))[0],   # domain is always 0 for single domain
                        trainset,
                        len(classes_in_cluster[cluster_names[cluster_idx]])
                    )
                )

        '''save classes_in_cluster'''
        with open(os.path.join(args['out.dir'], 'summary', 'class_mapping.json'), 'w') as f:
            json.dump({'epoch': start_iter, 'classes_in_cluster': classes_in_cluster}, f)


if __name__ == '__main__':
    obtain()

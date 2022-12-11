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
from utils import Accumulator, device, devices, set_determ, check_dir
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

        print(f'Devices: {devices}')

        train_loaders = []
        num_train_classes = dict()
        for t_indx, trainset in enumerate(trainsets):
            train_loaders.append(MetaDatasetEpisodeReader('train', [trainset], valsets, testsets, test_type='5shot'))
            num_train_classes[trainset] = train_loaders[t_indx].num_classes('train')
        print(f'num_train_classes: {num_train_classes}')
        # {'ilsvrc_2012': 712, 'omniglot': 883, 'aircraft': 70, 'cu_birds': 140, 'dtd': 33, 'quickdraw': 241,
        #  'fungi': 994, 'vgg_flower': 71}

        # val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets, testsets, test_type='5shot')

        '''initialize and load model'''
        models = []
        optimizers = []
        checkpointers = []
        start_iter, best_val_loss, best_val_acc = 0, 999999999, 0
        # init all starting issues for M(8) models.
        model_names = [args['model.name'].format(m_indx) for m_indx in range(args['model.num_clusters'])]
        cluster_names = [f'C{idx}' for idx in range(args['model.num_clusters'])]
        # M0-net - M7-net
        for m_indx in range(args['model.num_clusters']):        # 8
            model_args_with_name = copy.deepcopy(args)
            model_args_with_name['model.name'] = model_names[m_indx]
            _model = get_model(None, model_args_with_name, multi_device_id=m_indx)  # distribute model to multi-devices
            _model.eval()   # eval mode
            models.append(_model)

            _optimizer = get_optimizer(_model, model_args_with_name, params=_model.get_parameters())
            optimizers.append(_optimizer)

            # restoring the best checkpoint
            _checkpointer = CheckPointer(model_args_with_name, _model, optimizer=_optimizer)
            checkpointers.append(_checkpointer)

            if os.path.isfile(_checkpointer.best_ckpt):
                start_iter, best_val_loss, best_val_acc =\
                    _checkpointer.restore_model(ckpt='best')    # all 3 things are the same for all models
            else:
                raise Exception('No checkpoint restoration')

        '''initialize pool'''
        pool = Pool(capacity=args['model.num_clusters'])
        pool.restore()      # restore pool cluster centers.

        '''--------------'''
        '''Obtain mapping'''
        '''--------------'''
        class_mapping = dict()      # {trainset: {gt_label: cluster_idx}}
        for t_indx, (trainset, loader) in enumerate(zip(trainsets, train_loaders)):
            print(f"\n>> Obtain classes for {trainset}.")
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
                        domain = sample_numpy['domain'].item()

                        gt_label_set = np.unique(gt_labels_numpy)
                        unseen_labels = gt_label_set[classes[gt_label_set] == 0]      # e.g., [3]

                        if len(unseen_labels) != 0:
                            '''filter unseen classes to do clustering'''
                            unseen_mask = np.isin(gt_labels_numpy, unseen_labels)
                            images_numpy = images_numpy[unseen_mask]
                            re_labels_numpy = re_labels_numpy[unseen_mask]
                            gt_labels_numpy = gt_labels_numpy[unseen_mask]

                            labels, cluster_idxs = pool.clustering(
                                images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                                loader, devices, models, iter=1e5, mode='argmax',  # use argmax
                                update_cluster_centers=False,
                                only_return_cluster_idx=True)

                            '''add mapping'''
                            class_mapping[trainset].update(
                                {label[0]: cluster_idx for label, cluster_idx in zip(labels, cluster_idxs)})

                            ''''''
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

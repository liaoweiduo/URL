from typing import List, Dict, Any, Optional
import os
import shutil
import json
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from pymoo.util.ref_dirs import get_reference_directions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.losses import compute_prototypes
from models.model_helpers import get_optimizer
from models.adaptors import adaptor
from models.hierarchical_clustering import HierarchicalClustering
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from data.meta_dataset_reader import MetaDatasetReader
from utils import device, to_device

from config import args
from utils import check_dir


class Pool(nn.Module):
    """
    Pool stored class samples for the current clustering.

    A class instance contains (a set of image samples, class_label, class_label_str).
    """
    def __init__(self, capacity=8, max_num_classes=20, max_num_images=30,
                 thres_num_images=15,
                 mode='hierarchical', buffer_size=200):
        """
        :param capacity: Number of clusters. Typically, 8 columns of classes.
        :param max_num_classes: Maximum number of classes can be stored in each cluster.
        :param max_num_images: Maximum number of images can be stored in each class.
        :param thres_num_images: a class with more than thres_num_images is a valid class ready for sampling.
        :param mode: mode for cluster centers, choice=[kmeans, hierarchical].
        """
        super(Pool, self).__init__()
        self.capacity = capacity
        self.max_num_classes = max_num_classes
        self.max_num_images = max_num_images
        self.thres_num_images = thres_num_images
        self.mode = mode
        self.emb_dim = 512
        self.load_path = os.path.join(args['model.dir'], 'weights', 'pool')
        self.out_path = os.path.join(args['out.dir'], 'weights', 'pool')
        self.clusters: List[List[Dict[str, Any]]] = [[] for _ in range(self.capacity)]
        self.centers = None
        self.buffer = []
        self.buffer_size = buffer_size

        self.cluster_device = device
        self.init(0)

    def init(self, start_iter):
        self.clear_clusters()
        if self.mode == 'learnable':
            self.centers: torch.Tensor = nn.Parameter(torch.randn((self.capacity, self.emb_dim)))
            nn.init.xavier_uniform_(self.centers)
            args_with_lr = copy.deepcopy(args)
            args_with_lr['train.learning_rate'] = 3e-2
            self.optimizer = get_optimizer(self, args_with_lr, params=self.get_parameters())
            if start_iter > 0:
                ckpt_path = os.path.join(self.load_path, 'optimizer.pth.tar')
                ch = torch.load(ckpt_path, map_location=self.cluster_device)
                self.optimizer.load_state_dict(ch['optimizer'])
            if args['train.lr_policy'] == "step":
                self.lr_manager = UniformStepLR(self.optimizer, args, start_iter)
            elif "exp_decay" in args['train.lr_policy']:
                self.lr_manager = ExpDecayLR(self.optimizer, args, start_iter)
            elif "cosine" in args['train.lr_policy']:
                self.lr_manager = CosineAnnealRestartLR(self.optimizer, args, start_iter)
        # elif self.mode == 'mov_avg':
        #     self.centers: List[Optional[torch.Tensor]] = [None for _ in range(self.capacity)]
        elif self.mode in ['kmeans', 'hierarchical']:
            self.centers: Optional[torch.Tensor] = None
            self.clear_buffer()
        else:
            print(f'mode: {self.mode} does not need centers. Pool only store samples.')

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]

    def clear_clusters(self):
        clusters = self.clusters
        self.clusters: List[List[Dict[str, Any]]] = [[] for _ in range(self.capacity)]
        return clusters

    def clear_buffer(self):
        self.buffer = []

    def store(self, epoch, loaders, trainsets, is_best, class_filename='pool.json', center_filename='pool.npy'):
        """
        Store pool to json file.
        Only label information is stored for analysis, images are not stored for saving storage.
        Cluster centers are also stored.
        """
        check_dir(self.out_path, False)
        pool_dict = dict(epoch=epoch + 1)

        cu_cl = self.current_classes()
        for cluster_idx in range(len(cu_cl)):
            pool_dict[cluster_idx] = []
            pool_dict[f'{cluster_idx}_str'] = []
            for cls_idx in range(len(cu_cl[cluster_idx])):
                label = cu_cl[cluster_idx][cls_idx][0].tolist()
                # print('label', label)
                domain = label[1]
                str_label = loaders[trainsets[domain]].label_to_str(label, domain=0)
                pool_dict[cluster_idx].append(label)
                pool_dict[f'{cluster_idx}_str'].append(str_label)

        path = os.path.join(self.out_path, class_filename)
        with open(path, 'w') as f:
            json.dump(pool_dict, f)

        if is_best:
            shutil.copyfile(os.path.join(self.out_path, class_filename),
                            os.path.join(self.out_path, 'pool_best.json'))

        if self.mode not in ['learnable', 'kmeans']:        # , 'mov_avg'
            return

        path = os.path.join(self.out_path, center_filename)
        if self.mode == 'learnable':
            centers = self.centers.detach().cpu().numpy()
        # elif self.mode == 'mov_avg':
        #     centers = torch.stack(self.centers).numpy()     # may raise exception if contains None.
        elif self.mode == 'kmeans':
            centers = self.centers.cpu().numpy()
        else:
            raise Exception(f'Un implemented mode: {self.mode} for Pool.')

        np.save(path, centers)

        if is_best:
            shutil.copyfile(os.path.join(self.out_path, center_filename),
                            os.path.join(self.out_path, 'pool_best.npy'))

        if self.mode == 'learnable':
            '''store optimizer'''
            state = {'epoch': epoch + 1, 'optimizer': self.optimizer.state_dict()}
            torch.save(state, os.path.join(self.out_path, 'optimizer.pth.tar'))

    def restore(self, start_iter, center_filename='pool.npy'):
        """
        Restore pool's centers from npy file.
        """
        self.init(start_iter)

        # if self.mode == 'hierarchical':
        if self.mode not in ['learnable', 'kmeans']:    # , 'mov_avg'
            return

        centers = np.load(os.path.join(self.load_path, center_filename))

        if self.mode == 'learnable':
            self.centers.data = torch.from_numpy(centers)    # tensor: 8*512
        # elif self.mode == 'mov_avg':
        #     self.centers = [item for item in torch.from_numpy(centers)]     # tensor: 8*512 -> list: 8 *[512]
        elif self.mode == 'kmeans':
            self.centers = torch.from_numpy(centers).to(self.cluster_device)
        else:
            raise Exception(f'Un implemented mode: {self.mode} for Pool.')

    def put_buffer(self, images, info_dict, maintain_size=True):
        """
        Put samples (batch of torch cpu or numpy images) into buffer.
            info_dict should contain `cat_labels`, `similarities`,     # numpy
        If maintain_size, then check buffer size before put into buffer.
        """
        if len(self.buffer) >= self.buffer_size and maintain_size:     # do not exceed buffer size
            return False

        '''unpack'''
        labels = info_dict['cat_labels']
        similarities, features = info_dict['similarities'], info_dict['features']

        for label in np.unique(labels, axis=0):     # unique along first axis
            mask = (labels[:, 0] == label[0]) & (labels[:, 1] == label[1])      # gt label and domain all the same
            class_images = images[mask].numpy() if type(images) == torch.Tensor else images[mask]
            class_features = features[mask]
            class_similarities = similarities[mask]

            '''pop stored images and cat new images'''
            position = self.find_label(label, target='buffer')
            if position != -1:  # find exist label, cat onto it and re-put
                stored = self.buffer[position]
                # stored = self.buffer.pop(position)
                assert (stored['label'] == label).all()
                stored_images = np.concatenate([stored['images'], class_images])
                stored_features = np.concatenate([stored['features'], class_features])
                stored_similarities = np.concatenate([stored['similarities'], class_similarities])
            else:
                stored_images = class_images
                stored_features = class_features
                stored_similarities = class_similarities

            '''remove same image'''
            stored_images, img_idxes = np.unique(stored_images, return_index=True, axis=0)
            stored_features = stored_features[img_idxes]
            stored_similarities = stored_similarities[img_idxes]

            class_dict = {
                'images': stored_images, 'label': label,  # 'selection': stored_selection,
                'similarities': stored_similarities,
                'features': stored_features,
                # 'class_similarity': np.mean(stored_similarities, axis=0),  # mean over all samples [n_clusters]
            }

            '''put into buffer'''
            if position == -1:
                self.buffer.append(class_dict)
            else:
                self.buffer[position] = class_dict

        return True

    def buffer2cluster(self):
        """
        For each cluster, find max_num_classes.
        And for each class, find max_num_images with corresponding average sim.
        """
        for cluster_idx in range(self.capacity):
            '''preprocess buffer according to cluster_idx'''
            for cls in self.buffer:
                '''sort within class '''
                indexes = np.argsort(cls['similarities'][:, cluster_idx])[::-1]     # descending order
                chosen_images = cls['images'][indexes][:self.max_num_images]
                remain_images = cls['images'][indexes][self.max_num_images:]
                chosen_features = cls['features'][indexes][:self.max_num_images]
                remain_features = cls['features'][indexes][self.max_num_images:]
                # can be empty array([], shape=(0, 3, 84, 84)) len(remain_images) = 0
                chosen_similarities = cls['similarities'][indexes][:self.max_num_images]
                remain_similarities = cls['similarities'][indexes][self.max_num_images:]
                class_similarity = np.mean(chosen_similarities, axis=0)
                # mean over max_num_img samples [n_clusters]
                cls['remain_images'], cls['remain_similarities'] = remain_images, remain_similarities
                cls['chosen_images'], cls['chosen_similarities'] = chosen_images, chosen_similarities
                cls['chosen_features'], cls['remain_features'] = chosen_features, remain_features
                cls['class_similarity'] = class_similarity

            self.buffer.sort(
                key=lambda x: x['class_similarity'][cluster_idx], reverse=True)   # descending order

            '''put cls to cluster and modify cls'''
            for cls in self.buffer[:self.max_num_classes]:      # other clses in the buffer are not considered
                self.clusters[cluster_idx].append({
                    'images': cls['chosen_images'], 'label': cls['label'],
                    'similarities': cls['chosen_similarities'],
                    'features': cls['chosen_features'],
                    'class_similarity': cls['class_similarity'],
                })
                cls['images'], cls['similarities'] = cls['remain_images'], cls['remain_similarities']
                cls['features'] = cls['remain_features']
            '''remove empty cls'''
            self.buffer = [cls for cls in self.buffer if len(cls['images']) > 0]

    def put(self, images, info_dict):
        """
        Put samples (batch of torch cpu or numpy images) into clusters.
            info_dict should contain `cat_labels`, `similarities`,     # numpy

        """
        '''unpack'''
        labels = info_dict['cat_labels']
        similarities, features = info_dict['similarities'], info_dict['features']

        for label in np.unique(labels, axis=0):     # unique along first axis
            mask = (labels[:, 0] == label[0]) & (labels[:, 1] == label[1])      # gt label and domain all the same
            class_images = images[mask].numpy() if type(images) == torch.Tensor else images[mask]
            class_features = features[mask]
            class_similarities = similarities[mask]

            '''put to each cluster'''
            for cluster_idx, cluster in enumerate(self.clusters):
                position = self.find_label(label, cluster_idx=cluster_idx)
                if position != -1:  # find exist label, cat onto it
                    _, cls_idx = position       # cluster_idx, cls_idx
                    # stored = self.clusters[cluster_idx][cls_idx]
                    stored = cluster.pop(cls_idx)
                    # assert (stored['label'] == label).all()
                    stored_images = np.concatenate([stored['images'], class_images])
                    stored_features = np.concatenate([stored['features'], class_features])
                    stored_similarities = np.concatenate([stored['similarities'], class_similarities])
                else:
                    stored_images = class_images
                    stored_features = class_features
                    stored_similarities = class_similarities

                '''remove same image'''
                stored_images, img_idxes = np.unique(stored_images, return_index=True, axis=0)
                stored_features = stored_features[img_idxes]
                stored_similarities = stored_similarities[img_idxes]

                '''sort within class '''
                indexes = np.argsort(stored_similarities[:, cluster_idx])[::-1]  # descending order
                stored_images = stored_images[indexes][:self.max_num_images]
                stored_features = stored_features[indexes][:self.max_num_images]
                stored_similarities = stored_similarities[indexes][:self.max_num_images]
                # class_similarity = np.mean(stored_similarities, axis=0)
                # mean over max_num_img samples [n_clusters]

                '''put into cluster'''
                cluster.append({
                    'images': stored_images, 'label': label,  # 'selection': stored_selection,
                    'similarities': stored_similarities,
                    'features': stored_features,
                    # 'class_similarity': class_similarity,
                })

                '''maintain num valid cls'''
                while self.num_valid_class(cluster_idx) > self.max_num_classes:
                    '''remove one image with smallest similarity and maintain class_similarity'''
                    min_idx = np.argmin([cls['similarities'][-1, cluster_idx] for cls in cluster])

                    if len(cluster[min_idx]['images']) > 1:
                        cluster[min_idx]['images'] = cluster[min_idx]['images'][:-1]
                        cluster[min_idx]['similarities'] = cluster[min_idx]['similarities'][:-1]
                        cluster[min_idx]['features'] = cluster[min_idx]['features'][:-1]
                        # cluster[min_idx]['class_similarity'] = np.mean(
                        #     cluster[min_idx]['similarities'], axis=0)
                    else:
                        cluster.pop(min_idx)

                '''maintain class_similarity'''
                for cls in cluster:
                    cls['class_similarity'] = np.mean(cls['similarities'], axis=0)

                '''sort cluster according to class_similarity'''
                cluster.sort(
                    key=lambda x: x['class_similarity'][cluster_idx], reverse=True)   # descending order

    def num_valid_class(self, cluster_idx):
        return len([cls for cls in self.clusters[cluster_idx] if len(cls['images']) >= self.thres_num_images])

    def update_cluster_centers(self, cluster_idx, class_centroid):
        """
        Exponential Moving Average, EMA
        Update current cluster center of index cluster_idx with class_centroid.
        Shapes:
            class_centroid: [vec_size, ]
        """
        assert isinstance(class_centroid, torch.Tensor)

        '''if first, just assign'''
        if self.centers[cluster_idx] is None:
            self.centers[cluster_idx] = class_centroid.detach().cpu()
            return

        '''moving average update center calculation'''
        alpha = args['train.mov_avg_alpha']
        self.centers[cluster_idx] = alpha * class_centroid.detach().cpu() + (1-alpha) * self.centers[cluster_idx]

    # not use
    def get_distances(self, class_centroids, cluster_idx, distance):
        """
        Return similarities for class_centroids comparing with all centers.
        :param class_centroids: tensor centroid of classes with shape [n_classes, vec_size]
        :param cluster_idx: which cluster_center to compare.
        :param distance: string of distance [cos, l2, lin, corr]

        :return a tensor of distances with shape [n_classes, n_clusters]
        """
        if len(class_centroids.shape) == 1:     # 1 class
            class_centroids = class_centroids.unsqueeze(0)

        # class_centroids = class_centroids.unsqueeze(1)       # shape[n_classes, 1, vec_size]
        if self.centers[cluster_idx] is None:
            # centers = torch.rand(1, class_centroids.shape[-1]) * torch.ceil(class_centroids.max()).item()
            # uniform random from [0, ceil(class_centroids.max())]
            # centers = centers.to(class_centroids.device)
            centers = class_centroids   # always put to empty cluster.
        else:
            centers = self.centers[cluster_idx].to(class_centroids.device).unsqueeze(0)    # shape [1, vec_size]

        if distance == 'l2':
            logits = -torch.pow(class_centroids - centers, 2).sum(-1)  # shape [n_classes, n_clusters]
        elif distance == 'cos':
            logits = F.cosine_similarity(class_centroids, centers, dim=-1, eps=1e-30) * 10
        elif distance == 'lin':
            logits = torch.einsum('izd,zjd->ij', class_centroids, centers)
        elif distance == 'corr':
            logits = F.normalize((class_centroids * centers).sum(-1), dim=-1, p=2) * 10
        else:
            raise Exception(f"Un-implemented distance {distance}.")

        return logits

    def find_label(self, label: np.ndarray, target='clusters', cluster_idx=None):
        """
        Find label in pool, return position with (cluster_idx, cls_idx)
        If not in pool, return -1.
        If target == 'buffer', return the position (idx) in the buffer.
        """
        if target == 'clusters':
            if cluster_idx is not None:
                for cls_idx, cls in enumerate(self.clusters[cluster_idx]):
                    if (cls['label'] == label).all():       # (0, str) == (0, str) ? int
                        return cluster_idx, cls_idx
            else:
                for cluster_idx, cluster in enumerate(self.clusters):
                    for cls_idx, cls in enumerate(cluster):
                        if (cls['label'] == label).all():       # (0, str) == (0, str) ? int
                            return cluster_idx, cls_idx
        elif target == 'buffer':
            for buf_idx, cls in enumerate(self.buffer):
                if (cls['label'] == label).all():
                    return buf_idx
        return -1

    def current_classes(self):
        """
        Return current classes stored in the pool (name, num_images)
        """
        clses = []
        for cluster in self.clusters:
            clses_in_cluster = []
            for cls in cluster:
                clses_in_cluster.append((cls['label'], cls['images'].shape[0]))
            clses.append(clses_in_cluster)
        return clses

    def current_invalid_classes(self):
        classes = [cls for cluster in self.clusters
                    for cls in cluster if len(cls['images']) < self.thres_num_images]
        if len(classes) == 0:
            return []

        labels = np.concatenate([np.stack([cls['label'] for _ in range(len(cls['images']))])
                                 for cls in classes])  # [n_imgs, 2]
        images = np.concatenate([cls['images'] for cls in classes])
        features = np.concatenate([cls['features'] for cls in classes])

        classes = []
        for label in np.unique(labels, axis=0):  # unique along first axis
            mask = (labels[:, 0] == label[0]) & (labels[:, 1] == label[1])  # gt label and domain all the same
            class_images = images[mask]     # numpy images
            class_features = features[mask]

            '''remove same image'''
            class_images, img_idxes = np.unique(class_images, return_index=True, axis=0)
            class_features = class_features[img_idxes]

            classes.append({
                'images': class_images, 'labels': np.stack([label for _ in range(len(class_images))]),
                'features': class_features,
            })
        return classes

    def current_images(self, single_image=False):
        """
        # Return a batch of images (torch.Tensor) in the current pool with pool_montage.
        # batch of images => (10, 3, 84, 84)
        # class_montage => (3, 84, 84*10)
        # cluster montage => (3, 84*max_num_classes, 84*10)
        # pool montage => (3, 84*max_num_classes, 84*10*capacity).

        first return raw list, [8 * [num_class_each_cluster * numpy [10, 3, 84, 84]]]
        with labels
        """
        images = []
        for cluster in self.clusters:
            imgs = []
            for cls in cluster:
                imgs.append(cls['images'])      # cls['images'] shape [10, 3, 84, 84]
            images.append(imgs)

        if single_image:
            '''obtain width of images'''
            # max_num_imgs = self.max_num_images
            max_num_imgs = 0
            for cluster_idx, cluster in enumerate(images):
                for cls_idx, cls in enumerate(cluster):
                    num_imgs = cls.shape[0]
                    if num_imgs > max_num_imgs:
                        max_num_imgs = num_imgs

            '''construct a single image for each cluster'''
            for cluster_idx, cluster in enumerate(images):
                for cls_idx, cls in enumerate(cluster):
                    imgs = np.zeros((max_num_imgs, *cls.shape[1:]))
                    if len(cls) > 0:    # contain images
                        imgs[:cls.shape[0]] = cls
                    cluster[cls_idx] = np.concatenate([
                        imgs[img_idx] for img_idx in range(max_num_imgs)], axis=-1)
                if len(cluster) > 0:
                    images[cluster_idx] = np.concatenate(cluster, axis=-2)
                    # [3, 84*num_class, 84*max_num_images_in_class]
                    # [3, 84*50, 84*20]
                # else:   # empty cluster
                #     images[cluster_idx] = np.zeros((3, 84, 84))

        return images

    def current_embeddings(self):
        """
        first return raw list, [8 * [num_class_each_cluster * numpy [10, 512]]]
        """
        embeddings = []
        for cluster in self.clusters:
            embs = []
            for cls in cluster:
                embs.append(cls['embeddings'])      # cls['embeddings'] shape [10, 512]
            embeddings.append(embs)
        return embeddings

    def current_similarities(self, image_wise=False):
        """
        first return raw list, [8 * [num_class_each_cluster * numpy [8,]]]
        """
        similarities = []
        for cluster in self.clusters:
            similarity = []
            for cls in cluster:
                if image_wise:
                    similarity.append(cls['similarities'])      # cls['similarities'] shape [num_img, 8]
                else:
                    similarity.append(cls['class_similarity'])      # cls['class_similarity'] shape [8,]
            similarities.append(similarity)
        return similarities

    def current_assigns_gates(self):
        """
        first return raw list,
        assigns: [8 * [num_class_each_cluster * numpy [8,]]]
        gates: [8 * [num_class_each_cluster * numpy [8,4,]]]
        """
        assigns, gates = [], []
        for cluster in self.clusters:
            assign, gate = [], []
            for cls in cluster:
                assign.append(cls['assigns'])      # cls['assigns'] shape [8,]
                gate.append(cls['gates'])      # cls['gates'] shape [8,4,]
            assigns.append(assign)
            gates.append(gate)
        return assigns, gates

    def episodic_sample(
            self,
            cluster_idx,
            n_way,
            n_shot,
            n_query,
            remove_sampled_classes=False,
            d='numpy',
    ):
        """
        Sample a task from the specific cluster_idx.
        length of this cluster needs to be guaranteed larger than n_way.
        Random issue may occur, highly recommended to use np.rng.
        Return numpy if d is `numpy`, else tensor on d
        """
        candidate_class_idxs = np.arange(len(self.clusters[cluster_idx]))
        num_imgs = np.array([cls[1] for cls in self.current_classes()[cluster_idx]])
        candidate_class_idxs = candidate_class_idxs[num_imgs >= n_shot + n_query]
        assert len(candidate_class_idxs) >= n_way

        selected_class_idxs = np.random.choice(candidate_class_idxs, n_way, replace=False)
        context_images, target_images, context_labels, target_labels, context_gt, target_gt = [], [], [], [], [], []
        context_features, target_features = [], []
        # context_selection, target_selection = [], []
        for re_idx, idx in enumerate(selected_class_idxs):
            images = self.clusters[cluster_idx][idx]['images']              # [bs, c, h, w]
            tuple_label = self.clusters[cluster_idx][idx]['label']          # (gt_label, domain)
            features = self.clusters[cluster_idx][idx]['features']
            # selection = self.clusters[cluster_idx][idx]['selection']        # [bs, n_clusters]

            perm_idxs = np.random.permutation(np.arange(len(images)))
            context_images.append(images[perm_idxs[:n_shot]])
            target_images.append(images[perm_idxs[n_shot:n_shot+n_query]])
            context_features.append(features[perm_idxs[:n_shot]])
            target_features.append(features[perm_idxs[n_shot:n_shot+n_query]])
            context_labels.append([re_idx for _ in range(n_shot)])
            target_labels.append([re_idx for _ in range(n_query)])
            context_gt.append([tuple_label for _ in range(n_shot)])         # [(gt_label, domain)*n_shot]
            target_gt.append([tuple_label for _ in range(n_query)])         # [(gt_label, domain)*n_query]
            # context_selection.append(selection[perm_idxs[:n_shot]])
            # target_selection.append(selection[perm_idxs[n_shot:n_shot+n_query]])

        context_images = np.concatenate(context_images)
        target_images = np.concatenate(target_images)
        context_features = np.concatenate(context_features)
        target_features = np.concatenate(target_features)
        context_labels = np.concatenate(context_labels)
        target_labels = np.concatenate(target_labels)
        context_gt = np.concatenate(context_gt)
        target_gt = np.concatenate(target_gt)
        # context_selection = torch.cat(context_selection)
        # target_selection = torch.cat(target_selection)

        if d is None:
            d = device
        '''to tensor on divice d'''
        if d != 'numpy':
            context_images = torch.from_numpy(context_images).to(d)
            target_images = torch.from_numpy(target_images).to(d)
            context_features = torch.from_numpy(context_features).to(d)
            target_features = torch.from_numpy(target_features).to(d)
            context_labels = torch.from_numpy(context_labels).long().to(d)
            target_labels = torch.from_numpy(target_labels).long().to(d)

        task_dict = {
            'context_images': context_images,           # shape [n_shot*n_way, 3, 84, 84]
            'context_features': context_features,       # shape [n_shot*n_way, 512]
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'context_gt': context_gt,                   # shape [n_shot*n_way, 2]: [local, domain]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
            'target_features': target_features,         # shape [n_query*n_way, 512]
            'target_labels': target_labels,             # shape [n_query*n_way,]
            'target_gt': target_gt,                     # shape [n_query*n_way, 2]: [local, domain]
            'domain': cluster_idx,                      # 0-7: C0-C7, num_clusters
            # 'context_selection': context_selection,     # shape [n_shot*n_way, n_clusters]
            # 'target_selection': target_selection,       # shape [n_query*n_way, n_clusters]
        }

        if remove_sampled_classes:
            class_items = []
            for idx in range(len(self.clusters[cluster_idx])):
                if idx not in selected_class_idxs:
                    class_items.append(self.clusters[cluster_idx][idx])
            self.clusters[cluster_idx] = class_items

        return task_dict

class Mixer:
    """
    Mixer used to generate mixed tasks.
    """
    def __init__(self, mode='cutmix', num_sources=2, num_mixes=2):
        """
        :param mode indicates how to generate mixed tasks.
        :param num_sources indicates how many tasks are used to generate 1 mixed task.
        :param num_mixes indicates how many mixed tasks needed to be generated.
        """
        self.mode = mode
        if mode not in ['cutmix']:      # 'mixup'
            raise Exception(f'Un implemented mixer mode: {mode}.')
        self.num_sources = num_sources
        self.num_mixes = num_mixes
        self.ref = get_reference_directions("energy", num_sources, num_sources+num_mixes, seed=1234)

        '''eliminate num_obj extreme cases.'''
        check = np.sum(self.ref == 1, axis=1) == 0             # [[0,0,1]] == 1 => [[False, False, True]]
        # np.sum(weights == 1, axis=1): array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.ref = self.ref[check]      # shape [num_mixes, num_sources]    e.g. [[0.334, 0.666], [0.666, 0.334]]
        # self.ref = get_reference_directions("energy", num_obj, num_mix, seed=1)  # use those [1, 0, 0]
        assert self.ref.shape[0] == num_mixes

    def _cutmix(self, task_list, mix_id):
        """
        Apply cutmix on the task_list.
        task_list contains a list of task_dicts:
            {context_images, context_labels, context_gt, target_images, target_labels, target_gt, domain,
             context_selection, target_selection}
        mix_id is used to identify which ref to use as a probability.

        task sources should have same size, so that the mixed image is corresponding to the same position in sources.

        return:
        task_dict = {
            'context_images': context_images,           # shape [n_shot*n_way, 3, 84, 84]
            'context_features': context_features,       # shape [n_shot*n_way, 512]
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
            'target_features': target_features,         # shape [n_query*n_way, 512]
            'target_labels': target_labels,             # shape [n_query*n_way,]
            }
        meta_info: {'probability': probability of chosen which background,
         'lam': the chosen background for each image [(n_shot+n_query)* n_way,]}
        """
        # identify image size
        _, c, h, w = task_list[0]['context_images'].shape
        _, fs = task_list[0]['context_features'].shape
        context_size_list = [task_list[idx]['context_images'].shape[0] for idx in range(len(task_list))]
        target_size_list = [task_list[idx]['target_images'].shape[0] for idx in range(len(task_list))]
        assert np.min(context_size_list) == np.max(context_size_list)   # assert all contexts have same size
        assert np.min(target_size_list) == np.max(target_size_list)     # assert all targets have same size
        context_size = np.min(context_size_list)
        target_size = np.min(target_size_list)
        # print(f'context_size: {context_size}, target_size: {target_size}.')

        # generate num_sources masks for imgs with size [c, h, w]
        cutmix_prop = 0.3   # for (84*84), cut region is int(84*0.3)= (25*25)
        cuth, cutw = int(h * cutmix_prop), int(w * cutmix_prop)  # 84*0.3 [25, 25]
        cutfs = int(fs * cutmix_prop)  # 84*0.3 [25, 25]

        # generate lam, which is the index of img to be background. other imgs are foreground.
        # based on weight as probability.
        probability = self.ref[mix_id]      # shape [num_sources, ], sum = 1
        lam = np.random.choice(self.num_sources, context_size+target_size, p=probability, replace=True)
        # lam with shape [context_size+target_size,] is the decision to use which source as background.

        mix_imgs = []   # mix images batch
        mix_feas = []   # mix features batch
        mix_labs = []   # mix relative labels batch, same [0,0,1,1,2,2,...]
        # mix_gtls = []   # mix gt labels batch, str((weighted local label, domain=-1))
        for idx in range(context_size+target_size):
            if idx < context_size:
                set_name = 'context_images'
                set_nafs = 'context_features'
                lab_name = 'context_labels'
            else:
                set_name = 'target_images'
                set_nafs = 'target_features'
                lab_name = 'target_labels'
            # gtl_name = 'context_gt' if img_idx < context_size else 'target_gt'

            img_idx = idx if idx < context_size else idx - context_size     # local img idx in context and target set.
            # mix img is first cloned with background.
            mix_img = task_list[lam[idx]][set_name][img_idx].copy()
            mix_fea = task_list[lam[idx]][set_nafs][img_idx].copy()

            # for other foreground, cut the specific [posihs: posihs+cuth, posiws: posiws+cutw] region to
            # mix_img's [posiht: posiht+cuth, posiwt: posiwt+cutw] region
            for fore_img_idx in np.delete(np.arange(self.num_sources), lam[idx]):  # idxs for other imgs
                # pick pixels from [posihs, posiws, cuth, cutw], then paste to [posiht, posiwt, cuth, cutw]
                posihs = np.random.randint(h - cuth)
                posiws = np.random.randint(w - cutw)
                posiht = np.random.randint(h - cuth)
                posiwt = np.random.randint(w - cutw)
                posifss = np.random.randint(fs - cutfs)
                posifst = np.random.randint(fs - cutfs)

                fore = task_list[fore_img_idx][set_name][img_idx][:, posihs: posihs + cuth, posiws: posiws + cutw]
                mix_img[:, posiht: posiht + cuth, posiwt: posiwt + cutw] = fore
                mix_imgs.append(mix_img)
                fofs = task_list[fore_img_idx][set_nafs][img_idx][posifss: posifss + cutfs]
                mix_fea[posifst: posifst + cutfs] = fofs
                mix_feas.append(mix_fea)

            # determine mix_lab  same as the chosen img
            mix_labs.append(task_list[lam[idx]][lab_name][img_idx])

            # # determine mix_gtl  str((weighted local label, domain=-1))
            # local_label = np.sum(
            #     probability * np.array([task_list[idx][gtl_name][img_idx][0] for idx in range(self.num_sources)]))
            # domain = -1
            # mix_gtls.append('({:.2f}, {})'.format(local_label, domain))

        # formulate to task
        task_dict = {
            'context_images': np.stack(mix_imgs[:context_size]),   # shape [n_shot*n_way, 3, 84, 84]
            'context_features': np.stack(mix_feas[:context_size]),       # shape [n_shot*n_way, 512]
            'context_labels': np.array(mix_labs[:context_size]),   # shape [n_shot*n_way,]
            'target_images': np.stack(mix_imgs[context_size:]),     # shape [n_query*n_way, 3, 84, 84]
            'target_features': np.stack(mix_feas[context_size:]),         # shape [n_query*n_way, 512]
            'target_labels': np.array(mix_labs[context_size:]),     # shape [n_query*n_way,]
        }

        return task_dict, {'probability': probability, 'lam': lam}

    def mix(self, task_list, mix_id=0):
        """
        Numpy task task_list, len(task_list) should be same with self.num_sources
        """
        assert len(task_list) == self.num_sources
        assert mix_id < self.num_mixes
        assert isinstance(task_list[0]['context_images'], np.ndarray)

        if self.mode == 'cutmix':
            return self._cutmix(task_list, mix_id)

    def visualization(self, task_list):
        """
        Visualize the generated mixed task for all num_mixes cases.
        """
        pass


# return cos similarities for each prot (class_centroid) to cluster_centers
def prototype_similarity(embeddings, labels, centers, distance='cos'):
    """
    :param embeddings: shape [task_img_size, emb_dim]
    :param labels: relative labels shape [task_img_size,], e.g., [0, 0, 0, 1, 1, 1]
    :param centers: shape [n_clusters, emb_dim]
    :param distance: similarity [cos, l2, lin, corr]

    :return similarities shape [n_way, n_clusters] and class_centroids shape [n_way, emb_dim]
    """
    n_way = len(labels.unique())

    class_centroids = compute_prototypes(embeddings, labels, n_way)
    prots = class_centroids.unsqueeze(1)      # shape [n_way, 1, emb_dim]
    centers = centers.unsqueeze(0)         # shape [1, n_clusters, emb_dim]

    if distance == 'l2':
        logits = -torch.pow(centers - prots, 2).sum(-1)    # shape [n_way, n_clusters]
    elif distance == 'cos':
        logits = F.cosine_similarity(centers, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', prots, centers)
    elif distance == 'corr':
        logits = F.normalize((centers * prots).sum(-1), dim=-1, p=2) * 10
    else:
        raise Exception(f"Un-implemented distance {distance}.")

    return logits, class_centroids


def cal_hv_loss(objs, ref=2):
    """
    HV loss calculation: weighted loss
    code function from HV maximization:
    https://github.com/timodeist/multi_objective_learning/tree/06217d0ce024b92d52cdeb0390b1afb29ee59819

    Args:
        objs: Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)
        ref:

    Returns:
        weighted loss, for which the weights are based on HV gradients.
    """
    from mo_optimizers import hv_maximization

    num_obj, num_sol = objs.shape[0], objs.shape[1]
    ref_np = np.array([ref for _ in range(num_obj)])

    # obtain weights for the points in this front
    mo_opt = hv_maximization.HvMaximization(num_sol, num_obj, ref_np)

    # obtain np objs
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs

    # compute weight for each solution
    weights = mo_opt.compute_weights(objs_np)       # check if use objs_np.transpose()
    if type(objs) is torch.Tensor:
        weights = weights.to(objs.device)
        # weights = weights.permute([1, 0]).to(objs.device)
        weighted_loss = torch.sum(objs * weights)
    else:
        weights = weights.numpy()
        # weights = weights.permute([1, 0]).numpy()
        weighted_loss = np.sum(objs * weights)

    return weighted_loss


def cal_hv(objs, ref=2, target='loss'):
    """
    Calculate HV value for multi-objective losses and accs.

    Args:
        objs : Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)
        ref: 2 for loss, 0 for acc
        target:

    Returns:
        hv value
    """
    from pymoo.indicators.hv import HV

    num_obj, num_sol = objs.shape[0], objs.shape[1]
    if type(ref) is not list:
        ref_point = np.array([ref for _ in range(num_obj)])
    else:
        ref_point = np.array(ref)

    assert len(ref_point.shape) == 1

    # obtain np objs
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs

    # for acc reverse objs
    if target == 'acc':
        objs_np = -objs_np

    ind = HV(ref_point=ref_point)
    hv = ind(objs_np.T)

    if type(hv) is not float:
        hv = hv.item()

    return hv


def cal_min_crowding_distance(objs):
    """
    code from pymoo, remove normalization part, return the min cd.
    Args:
        objs: Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)

    Returns:

    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    # obtain np objs: F
    if type(objs) is torch.Tensor:
        F = objs.detach().cpu().numpy().T
    else:
        F = objs.T

    non_dom = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F = np.copy(F[non_dom, :])

    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # # calculate the norm for each objective - set to NaN if all values are equal
    # norm = np.max(F, axis=0) - np.min(F, axis=0)
    # norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    # dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
    dist_to_last, dist_to_next = dist_to_last[:-1], dist_to_next[1:]

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return min(cd)


def draw_objs(objs, labels):
    """
    return a figure of objs.
    objs: numpy with shape [obj_size, pop_size] or [n_iter, obj_size, pop_size] with gradient color
    labels: list of labels: ['p0', 'p1', 'm0', 'm1']
    """
    n_iter = 1
    if len(objs.shape) == 2:
        obj_size, pop_size = objs.shape
        objs = objs[np.newaxis, :, :]
    else:
        n_iter, obj_size, pop_size = objs.shape

    assert obj_size == 2

    '''generate pandas DataFrame for objs'''
    data = pd.DataFrame({       # for all points
        'f1': [objs[i_idx, 0, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
        'f2': [objs[i_idx, 1, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
        'Iter': [i_idx for i_idx in range(n_iter) for pop_idx in range(pop_size)],
        'Label': [labels[pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
    })

    fig, ax = plt.subplots()
    ax.grid(True)
    # c = plt.get_cmap('rainbow', pop_size)

    sns.scatterplot(data, x='f1', y='f2',
                    hue='Label', size='Iter', sizes=(100, 200), alpha=1., ax=ax)

    # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.1), ncol=1)
    return fig


def draw_heatmap(data, verbose=True):
    """
    return a figure of heatmap.
    :param data: 2-D Numpy
    :param verbose: whether to use annot
    """
    fig, ax = plt.subplots()
    if verbose:
        sns.heatmap(
            data, cmap=plt.get_cmap('Greens'), annot=True, fmt=".3f", cbar=False,
        )
    else:
        sns.heatmap(
            data, cmap=plt.get_cmap('Greens'), cbar=True,
        )
    return fig


def map_re_label(re_labels):
    """
    For masked re_label, it can be [1,1,1,3,3,3,7,7,7]
    As a valid episodic task, re_label should be in the form [0,0,0,1,1,1,2,2,2]
    Return valid re_label
    """
    label_set = np.unique(re_labels)
    re_label_map = {
        origin_label: correct_label for origin_label, correct_label in zip(label_set, np.arange(len(label_set)))}
    correct_labels = np.array(list(map(lambda x: re_label_map[x], re_labels)))
    return correct_labels


def pmo_embed(images, labels, grad_ones, feature_extractor, pmo, cluster_idxs):
    """
    Apply grad_ones on sample and forward to feature_extractor and
    the specific pmo branches indicated by `cluster_idxs`.
    :param images: Numpy with shape [bs, c, h, w]
    :param labels: Numpy with shape [bs]
    :param grad_ones: Tensor with same shape as images.
    :param feature_extractor: an ResNet18 feature extractor.
    :param pmo: an adaptor with `num_clusters` branches.
    :param cluster_idxs: a list of idxs for which pmo branches to access.
    Return a list of embeddings .
    """
    grad_one_device = grad_ones.device
    images = torch.from_numpy(images).to(grad_one_device) * grad_ones
    labels = torch.from_numpy(labels).long()

    embeddings = feature_extractor.embed(images.to(device))
    embeddings_list = pmo([embeddings.to(device) for cluster_idx in cluster_idxs], cluster_idxs)
    # devices[cluster_idx]
    return embeddings_list, [labels.to(device) for cluster_idx in cluster_idxs]


def to_torch(sample, grad_ones, device_list=None):
    """
    Put samples to a list of devices specified by device_list.
    :param sample:
        {context_images, context_labels, context_gt,
         target_images, target_labels, target_gt, domain}
    :param grad_ones:
        {context_grad_ones, target_grad_ones}
    :param device_list: a list of devices.
    Return a dict keying by device.
    """
    if device_list is None:
        device_list = [device]

    sample_dict = {d: dict() for d in device_list}

    grad_one_device = grad_ones['context_grad_ones'].device

    for key, val in sample.items():
        if isinstance(val, str):
            for s in sample_dict.values():
                s[key] = val
            continue
        val = torch.from_numpy(np.array(val))
        if key == 'context_images':
            val = val.to(grad_one_device) * grad_ones['context_grad_ones']
        elif key == 'target_images':
            val = val.to(grad_one_device) * grad_ones['target_grad_ones']
        elif 'image' not in key:
            val = val.long()

        for d, s in sample_dict.items():
            s[key] = val.to(d)

    return sample_dict


def available_setting(num_imgs_clusters, task_type, min_available_clusters=1, use_max_shot=False):
    """Check whether pool has enough samples for specific task_type and return a valid setting.
    :param num_imgs_clusters: list of Numpy array with shape [num_clusters * [num_classes]]
                              indicating number of images for specific class in specific clusters.
    :param task_type: `standard`: vary-way-vary-shot-ten-query
                      `1shot`: five-way-one-shot-ten-query
                      `5shot`: vary-way-five-shot-ten-query
    :param min_available_clusters: minimum number of available clusters to apply that setting.
    :param use_max_shot: if True, return max_shot rather than random shot
    :return a valid setting.
    """
    n_way, n_shot, n_query = -1, -1, -1
    for _ in range(10):     # try 10 times, if still not available setting, return -1
        n_query = 10

        min_shot = 5 if task_type == '5shot' else 1
        min_way = 5
        max_way = sorted([len(num_images[num_images >= min_shot + n_query]) for num_images in num_imgs_clusters]
                         )[::-1][min_available_clusters - 1]

        if max_way < min_way:
            return -1, -1, -1   # do not satisfy the minimum requirement.

        n_way = 5 if task_type == '1shot' else np.random.randint(min_way, max_way + 1)

        # shot depends on chosen n_way
        available_shots = []
        for num_images in num_imgs_clusters:
            shots = sorted(num_images[num_images >= min_shot + n_query])[::-1][:n_way]
            available_shots.append(0 if len(shots) < n_way else (shots[-1] - n_query))
        max_shot = np.min(sorted(available_shots)[::-1][:min_available_clusters])

        if max_shot < min_shot:
            return -1, -1, -1   # do not satisfy the minimum requirement.

        n_shot = 1 if task_type == '1shot' else 5 if task_type == '5shot' else max_shot if (
            use_max_shot) else np.random.randint(min_shot, max_shot + 1)

        available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

        if len(available_cluster_idxs) < min_available_clusters:
            print(f"available_setting error with information: \n"
                  f"way [{min_way}, {max_way}]:{n_way} shot [{min_shot}, {max_shot}]:{n_shot}, \n"
                  f"pool: {num_imgs_clusters}, \n"
                  f"avail: {available_cluster_idxs}")
        else:
            break

    return n_way, n_shot, n_query


def check_available(num_imgs_clusters, n_way, n_shot, n_query):
    """Check whether pool has enough samples for specific setting and return available cluster idxes.
    :param num_imgs_clusters: list of Numpy array with shape [num_clusters * [num_classes]]
                              indicating number of images for specific class in specific clusters.
    :param n_way:
    :param n_shot:
    :param n_query:

    :return available cluster idxes which can satisfy sampling n_way, n_shot, n_query.
    """
    available_cluster_idxs = []
    for idx, num_imgs in enumerate(num_imgs_clusters):
        if len(num_imgs[num_imgs >= n_shot + n_query]) >= n_way:
            available_cluster_idxs.append(idx)

    return available_cluster_idxs


def task_to_device(task, d='numpy'):
    new_task = {}
    if d == 'numpy':
        new_task['context_images'] = task['context_images'].cpu().numpy()
        new_task['context_labels'] = task['context_labels'].cpu().numpy()
        new_task['target_images'] = task['target_images'].cpu().numpy()
        new_task['target_labels'] = task['target_labels'].cpu().numpy()
    else:
        new_task['context_images'] = torch.from_numpy(task['context_images']).to(d)
        new_task['context_labels'] = torch.from_numpy(task['context_labels']).long().to(d)
        new_task['target_images'] = torch.from_numpy(task['target_images']).to(d)
        new_task['target_labels'] = torch.from_numpy(task['target_labels']).long().to(d)

    return new_task


def inner_update(context_features, context_labels, model, max_iter=40, lr=0.1, distance='cos', return_iterator=False):
    """
    apply inner updates on film inited with params in model.
    Args:
        context_features:
        context_labels:
        model:
        max_iter:
        lr:
        distance:
        return_iterator: True support yield.

    Returns:

    """
    '''init film params'''

    '''iter'''
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
    #     optimizer_model.zero_grad()
    #     loss.backward()
    #     optimizer_model.step()
    #


if __name__ == '__main__':
    cal_hv_loss(np.array([[1,0], [0,1]]), 2)

    _n_way, _n_shot, _n_query = available_setting([np.array([20,20,20,20,20]), np.array([])], task_type='standard')

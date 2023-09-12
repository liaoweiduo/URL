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
    def __init__(self, capacity=8, max_num_classes=10, max_num_images=20, mode='hierarchical', buffer_size=200):
        """
        :param capacity: Number of clusters. Typically 8 columns of classes.
        :param max_num_classes: Maximum number of classes can be stored in each cluster.
        :param max_num_images: Maximum number of images can be stored in each class.
        :param mode: mode for cluster centers, choice=[kmeans, hierarchical].
        """
        super(Pool, self).__init__()
        self.capacity = capacity
        self.max_num_classes = max_num_classes
        self.max_num_images = max_num_images
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
        elif self.mode == 'mov_avg':
            self.centers: List[Optional[torch.Tensor]] = [None for _ in range(self.capacity)]
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

        if self.mode not in ['learnable', 'mov_avg', 'kmeans']:
            return

        path = os.path.join(self.out_path, center_filename)
        if self.mode == 'learnable':
            centers = self.centers.detach().cpu().numpy()
        elif self.mode == 'mov_avg':
            centers = torch.stack(self.centers).numpy()     # may raise exception if contains None.
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
        if self.mode not in ['learnable', 'mov_avg', 'kmeans']:
            return

        centers = np.load(os.path.join(self.load_path, center_filename))

        if self.mode == 'learnable':
            self.centers.data = torch.from_numpy(centers)    # tensor: 8*512
        elif self.mode == 'mov_avg':
            self.centers = [item for item in torch.from_numpy(centers)]     # tensor: 8*512 -> list: 8 *[512]
        elif self.mode == 'kmeans':
            self.centers = torch.from_numpy(centers).to(self.cluster_device)
        else:
            raise Exception(f'Un implemented mode: {self.mode} for Pool.')

    def put_batch(self, images, cluster_idxs, info_dict):
        """
        Put samples (batch of torch cpu images) into clusters.
            info_dict should contain `domain`, `gt_labels`, `similarities`,     # numpy
                              #  `selection`.  # torch cuda
        """
        '''unpack'''
        domain, gt_labels = info_dict['domain'], info_dict['gt_labels']
        similarities = info_dict['similarities']
        # similarities, selection = info_dict['similarities'], info_dict['selection']

        for sample_idx in range(len(cluster_idxs)):
            cluster_idx = cluster_idxs[sample_idx]
            '''pop stored images and cat new images'''
            label = np.array([gt_labels[sample_idx], domain[sample_idx]])
            position = self.find_label(label, cluster_idx=cluster_idx)
            if position != -1:      # find exist label, cat onto it and re-put
                stored = self.clusters[position[0]].pop(position[1])
                stored_images = stored['images']
                stored_images = np.concatenate([stored_images, images[sample_idx:sample_idx+1].numpy()])
                stored_similarities = stored['similarities']
                stored_similarities = np.concatenate([stored_similarities, similarities[sample_idx:sample_idx+1]])
                # stored_selection = stored['selection']
                # stored_selection = torch.cat([stored_selection, selection[sample_idx:sample_idx+1]])
            else:
                stored_images = images[sample_idx:sample_idx+1].numpy()
                stored_similarities = similarities[sample_idx:sample_idx+1]
                # stored_selection = selection[sample_idx:sample_idx+1]

            '''sort within class '''
            indexs = np.argsort(stored_similarities[:, cluster_idx])[::-1]      # descending order
            stored_images = stored_images[indexs]
            stored_similarities = stored_similarities[indexs]

            '''remove several images with smaller sim to satisfy max_num_images'''
            if stored_images.shape[0] > self.max_num_images:
                stored_images = stored_images[:self.max_num_images]
                stored_similarities = stored_similarities[:self.max_num_images]
                # stored_selection = stored_selection[:self.max_num_images]

            '''put to cluster: cluster_idx'''
            self.clusters[cluster_idx].append({
                'images': stored_images, 'label': label,    # 'selection': stored_selection,
                'similarities': stored_similarities,
                'class_similarity': np.mean(stored_similarities, axis=0),       # mean over all samples [n_clusters]
            })

        '''after put all samples into pool, handle for full clusters'''
        for cluster_idx, cluster in enumerate(self.clusters):
            '''sort class according to the corresponding similarity (mean over all samples in the class)'''
            self.clusters[cluster_idx].sort(
                key=lambda x: x['class_similarity'][cluster_idx], reverse=True)   # descending order

            if len(self.clusters[cluster_idx]) > self.max_num_classes:
                '''need to remove all classes only contain 1 image, since they are high probability to be outliers'''
                self.clusters[cluster_idx] = [cls for cls in cluster if cls['images'].shape[0] > 1]

            if len(self.clusters[cluster_idx]) > self.max_num_classes:
                '''need to remove one with smallest similarity: last one to satisfy max_num_classes'''
                self.clusters[cluster_idx] = self.clusters[cluster_idx][:self.max_num_classes]

    def put_buffer(self, images, info_dict, maintain_size=True):
        """
        Put samples (batch of torch cpu images) into buffer.
            info_dict should contain `domain`, `gt_labels`, `similarities`,     # numpy
        If maintain_size, then check buffer size before put into buffer.
        """
        if len(self.buffer) >= self.buffer_size and maintain_size:     # do not exceed buffer size
            return False

        '''unpack'''
        domains, gt_labels = info_dict['domain'], info_dict['gt_labels']
        similarities = info_dict['similarities']

        '''images for one class'''
        labels = np.stack([gt_labels, domains], axis=1)     # [n_img, 2]

        for label in np.unique(labels, axis=0):     # unique along first axis
            mask = (labels[:, 0] == label[0]) & (labels[:, 1] == label[1])      # gt label and domain all the same
            assert len(np.unique(gt_labels[mask])) == len(np.unique(domains[mask])) == 1
            assert gt_labels[mask][0] == label[0] and domains[mask][0] == label[1]
            class_images = images[mask].numpy()
            class_similarities = similarities[mask]

            '''pop stored images and cat new images'''
            position = self.find_label(label, target='buffer')
            if position != -1:  # find exist label, cat onto it and re-put
                stored = self.buffer[position]
                # stored = self.buffer.pop(position)
                assert (stored['label'] == label).all()

                # todo: check label keeps the same
                print(f'debug: find exist cls in buffer at position: {position} for label: {label}, '
                      f'with img len {len(stored["images"])} and sim len {len(stored["similarities"])}, '
                      f'current img len{len(class_images)} and sim len {len(class_similarities)}')

                stored_images = np.concatenate([stored['images'], class_images])
                stored_similarities = np.concatenate([stored['similarities'], class_similarities])
            else:
                stored_images = class_images
                stored_similarities = class_similarities

            '''remove same image'''
            stored_images, img_idxes = np.unique(stored_images, return_index=True, axis=0)

            # todo: check label keeps the same
            if len(stored_images) != len(stored_similarities):
                print(f'debug: after remove the same image and img_idxs: {img_idxes} for label: {label}'
                      f'current len{len(stored_images)} and before len {len(stored_similarities)}')

            stored_similarities = stored_similarities[img_idxes]

            class_dict = {
                'images': stored_images, 'label': label,  # 'selection': stored_selection,
                'similarities': stored_similarities,
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
                # can be empty array([], shape=(0, 3, 84, 84)) len(remain_images) = 0
                chosen_similarities = cls['similarities'][indexes][:self.max_num_images]
                remain_similarities = cls['similarities'][indexes][self.max_num_images:]
                class_similarity = np.mean(chosen_similarities, axis=0)
                # mean over max_num_img samples [n_clusters]
                cls['remain_images'], cls['remain_similarities'] = remain_images, remain_similarities
                cls['chosen_images'], cls['chosen_similarities'] = chosen_images, chosen_similarities
                cls['class_similarity'] = class_similarity

            self.buffer.sort(
                key=lambda x: x['class_similarity'][cluster_idx], reverse=True)   # descending order

            '''put cls to cluster and modify cls'''
            for cls in self.buffer[:self.max_num_classes]:      # other clses in the buffer are not considered
                self.clusters[cluster_idx].append({
                    'images': cls['chosen_images'], 'label': cls['label'],
                    'similarities': cls['chosen_similarities'],
                    'class_similarity': cls['class_similarity'],
                })
                cls['images'], cls['similarities'] = cls['remain_images'], cls['remain_similarities']

            '''remove empty cls'''
            self.buffer = [cls for cls in self.buffer if len(cls['images']) > 0]

    '''
    OLD PUT
    '''

    '''Buffer'''
    def put_into_buffer(self, images_list, label_list):
        """Put a list of classes to the buffer
        :param images_list: list of numpy images with shape [bs, c, h, w]
        :param label_list: list of int tuple: (gt_label, domain)
        """
        for images, label in zip(images_list, label_list):
            '''pop stored images and cat new images'''
            position = self.find_label(label, target='buffer')
            if position != -1:      # find exist label, cat onto it and reput into buffer
                stored = self.buffer.pop(position)
                assert stored['label'] == label
                stored_images = np.concatenate([stored['images'], images])
            else:
                stored_images = images

            '''remove first several images to satisfy max_num_images'''
            if stored_images.shape[0] > self.max_num_images:
                stored_images = stored_images[-self.max_num_images:]

            class_dict = {
                'images': stored_images,
                'label': label,
            }

            '''put into buffer'''
            self.buffer.append(class_dict)

    def task_put_into_buffer(self, images, re_labels, gt_labels, domain):
        """Put the classes in the tasks to the buffer. Classes are separated by re_labels.
        :param images: Numpy images with shape [bs, c, h, w]
        :param re_labels: Numpy relative labels with shape [bs]
        :param gt_labels: Numpy ground truth labels with shape [bs]
        :param domain: Int domain
        """
        images_list, label_list = [], []
        for re_label in range(len(np.unique(re_labels))):
            mask = re_labels == re_label
            class_images, class_gt_label = images[mask], gt_labels[mask][0].item()
            images_list.append(class_images)
            label_list.append((class_gt_label, domain))

        self.put_into_buffer(images_list, label_list)

    def cluster_put_into_buffer(self):
        """Put the classes in the clusters to the buffer."""
        images_list, label_list = [], []
        for cluster_idx, (cls_list_in_cluster, img_list_in_cluster) in enumerate(
                zip(self.current_classes(), self.current_images())):
            images_list.extend(img_list_in_cluster)
            label_list.extend([cls_with_img_num[0] for cls_with_img_num in cls_list_in_cluster])

        self.clear_clusters()

        self.put_into_buffer(images_list, label_list)

    def batch_put_into_buffer(self, sample, class_mapping, domain_name, cluster_name, loader):
        """put batch sample into the buffer with specific domain_str and cluster_name"""
        images = sample['images']       # Tensor images [bs, c, h, w] in device
        labels = sample['labels']       # Tensor labels [bs] in device
        for idx, label in enumerate(labels):
            if class_mapping[domain_name][str(label.item())][2] == cluster_name:
                re_label = class_mapping[domain_name][str(label.item())][1]
                label_str = class_mapping[domain_name][str(label.item())][0]
                str_label = loader.label_to_str((label.item(), _), domain=0)[0]
                assert label_str == str_label

                image_dict = {
                    'image': images[idx],
                    'label': re_label,
                }

                '''put into buffer'''
                self.buffer.append(image_dict)

    '''Cluster'''
    def cluster_and_assign_from_buffer(self, feature_extractor, model):
        """Perform clustering on classes in the buffer"""

        '''construct a big task for all classes in the buffer'''
        images, re_labels, gt_labels, sampled_images = [], [], [], []
        for re_label, class_dict in enumerate(self.buffer):
            n_img = class_dict['images'].shape[0]
            images.append(class_dict['images'])
            sampled_images.append(class_dict['images'][0])
            re_labels.append(np.repeat(re_label, n_img))
            gt_labels.append(class_dict['label'])

        n_cls = len(self.buffer)
        images_numpy = np.concatenate(images)           # [bs, c, h, w]
        re_labels_numpy = np.concatenate(re_labels)     # [bs]

        '''move images to device where model locates'''
        images_torch = to_device(images_numpy, self.cluster_device)
        labels_torch = to_device(re_labels_numpy, self.cluster_device)

        '''clear buffer'''
        self.clear_buffer()

        # obtain embeddings after url
        embeddings = feature_extractor.embed(images_torch)

        if self.mode == 'kmeans':
            embeddings = model([embeddings.to(self.cluster_device)])[0]

            with torch.no_grad():
                class_centroids = compute_prototypes(embeddings, labels_torch, n_cls).cpu().numpy()     # [n_cls, emb_dim]

            '''get cluster centers'''
            centers = self.centers.cpu().numpy() if self.centers is not None else 'k-means++'

            '''do k-means on class_centroids init with centers'''
            km = KMeans(n_clusters=self.capacity, init=centers, n_init=1)
            km.fit(class_centroids)
            # similarities = km.fit_transform(class_centroids)        # [n_cls, n_cluster]
            updated_centers = km.cluster_centers_
            self.centers = torch.from_numpy(updated_centers).to(self.cluster_device)

            cluster_idxs, grad_ones, similarities, class_centroids = self.cluster_from_emb(
                embeddings, labels_torch)

            '''put into clusters'''
            for re_label in range(len(cluster_idxs)):
                class_images = images[re_label]
                label = gt_labels[re_label]
                # embeddings_numpy = embeddings[re_labels_numpy == re_label].detach().cpu().numpy()
                similarities_numpy = similarities[re_label]  # [n_cluster]
                self.put(class_images, label, grad_ones[re_label],
                         {'similarities': similarities_numpy},
                         cluster_idxs[re_label], class_centroids[re_label])

        elif self.mode == 'hierarchical':
            class_centroids = compute_prototypes(embeddings, labels_torch, n_cls)       # [n_cls, 512]
            similarities, loss_rec, assigns, gates = model(class_centroids)     # [n_cls, 8]

            cluster_idxs, grad_ones = self.cluster_from_similarities(similarities)

            similarities = similarities.detach().cpu().numpy()

            '''put into clusters'''
            for re_label in range(len(cluster_idxs)):
                class_images = images[re_label]
                label = gt_labels[re_label]
                # embeddings_numpy = embeddings[re_labels_numpy == re_label].detach().cpu().numpy()
                similarities_numpy = similarities[re_label]  # [n_cluster]
                self.put(class_images, label, grad_ones[re_label],
                         {'similarities': similarities[re_label],
                          'assigns': assigns[:, re_label],
                          'gates': gates[:, :, re_label]},
                         cluster_idxs[re_label], class_centroids[re_label])

        else:
            raise Exception('')

        '''return info for tsne'''
        return {
            'labels': gt_labels,  # n_cls * (gt, domain)
            'cluster_idxs': cluster_idxs,  # [n_cls,]
            'class_centroids': class_centroids.detach().cpu().numpy(),  # [n_cls, emb_dim]
            'sample_images': np.stack(sampled_images),  # [n_cls, c, h, w]
            'similarities': similarities,  # [n_cls, n_cluster]
            'loss_rec': loss_rec,
        }

    def cluster_from_similarities(self, similarities, softmax_mode='gumbel'):
        """Compute cluster_idxs and grad_ones with specific softmax mode"""

        '''cluster_idx is obtained with based on similarities'''
        if softmax_mode == 'gumbel':
            hard_gumbel_softmax = F.gumbel_softmax(similarities, tau=args['train.gumbel_tau'], hard=True)
            cluster_idxs = torch.argmax(hard_gumbel_softmax, dim=1).detach().cpu().numpy()    # numpy [n_way,]
            grad_ones = torch.stack([
                hard_gumbel_softmax[idx, cluster_idx] for idx, cluster_idx in enumerate(cluster_idxs)])
            # Tensor [1, 1,...] shape [n_way,]
        elif softmax_mode == 'softmax':
            sftmx = F.softmax(similarities, dim=1).detach().cpu().numpy()
            cluster_idxs = np.argmax(sftmx, axis=1)
            grad_ones = torch.ones(similarities.shape[0]).to(similarities.device)
        else:
            raise Exception(f'Un implemented softmax_mode: {softmax_mode} for Pool to do clustering.')

        return cluster_idxs, grad_ones

    def cluster_from_emb(self, embeddings, labels_torch, softmax_mode='gumbel'):
        """Apply clustering on embeddings with specific softmax mode"""
        centers = self.centers if self.centers is not None else torch.randn(self.capacity, self.emb_dim).to(self.cluster_device)
        similarities, class_centroids = prototype_similarity(
            embeddings, labels_torch, centers, distance=args['test.distance'])
        # similarities shape [n_way, n_clusters] and class_centroids shape [n_way, emb_dim]

        cluster_idxs, grad_ones = self.cluster_from_similarities(similarities, softmax_mode)

        return cluster_idxs, grad_ones, similarities.detach().cpu().numpy(), class_centroids

    def cluster_and_assign(
            self, images_numpy, re_labels_numpy, gt_labels_numpy, domain,
            feature_extractor, model,
            update_cluster_centers=False,
            softmax_mode='gumbel',
            put=True):
        """
        Do clustering based on class centroid similarities with cluster centers.
        :param images_numpy: numpy/tensor with shape [bs, c, h, w]
        :param re_labels_numpy: related labels, numpy/tensor with shape [bs,]
        :param gt_labels_numpy: true labels in the domain, numpy with shape [bs,]
        :param domain: int domain or same size numpy as labels [bs,].
        :param feature_extractor: feature_extractor to obtain image features.
        :param model: clustering model to obtain img embedding and class centroid.
        :param update_cluster_centers: whether to update the cluster center.
            only activated for mov avg approach, not for trainable cluster centers.
        :param softmax_mode: how similarity do softmax.
            choices=[gumbel, softmax]
        :param put: whether to put classes into clusters.
        """
        '''move images to device where model locates'''
        images_torch = to_device(images_numpy, self.cluster_device)
        labels_torch = to_device(re_labels_numpy, self.cluster_device)

        embeddings = feature_extractor.embed(images_torch)

        if self.mode == 'kmeans':
            embeddings = model([embeddings.to(self.cluster_device)])[0]

            cluster_idxs, grad_ones, similarities, class_centroids = self.cluster_from_emb(
                embeddings, labels_torch, softmax_mode)

        elif self.mode == 'hierarchical':
            n_cls = len(labels_torch.unique())
            class_centroids = compute_prototypes(embeddings, labels_torch, n_cls)       # [n_cls, 512]
            similarities, loss_rec, assigns, gates = model(class_centroids)     # [n_cls, 8]

            cluster_idxs, grad_ones = self.cluster_from_similarities(similarities, softmax_mode)

            similarities = similarities.detach().cpu().numpy()

        else:
            raise Exception('')

        labels = []
        sample_images = []
        sims = []
        for re_label in range(len(cluster_idxs)):
            images = images_numpy[re_labels_numpy == re_label]
            # embeddings_numpy = embeddings[re_labels_numpy == re_label].detach().cpu().numpy()
            gt_label = gt_labels_numpy[re_labels_numpy == re_label][0].item()
            similarities_numpy = similarities[re_label]  # [num_cluster]
            if type(domain) is np.ndarray:
                image_domains = domain[re_labels_numpy == re_label]
                label = (gt_label, image_domains[0].item())
            else:
                label = (gt_label, domain)
            labels.append(label)
            sample_images.append(images[0])
            sims.append(similarities_numpy)

            '''put samples into pool with (gt_label, domain)'''
            if put:
                self.put(images, label, grad_ones[re_label], {'similarities': similarities_numpy},
                         cluster_idxs[re_label], class_centroids[re_label], update_cluster_centers)

        '''return info for tsne'''
        return {
            'labels': np.array(labels),                                     # [n_way, gt, domain]
            'cluster_idxs': cluster_idxs,                                   # [n_way,]
            'class_centroids': class_centroids.detach().cpu().numpy(),      # [n_way, emb_dim]
            'sample_images': np.stack(sample_images),                       # [n_way, c, h, w]
            'similarities': np.stack(sims)                                  # [n_way, num_cluster]
        }

    def cluster_and_assign_with_class_mapping(
            self, images_numpy, gt_labels_numpy, domain, domain_name,
            class_mapping, cluster_name, cluster_idx):
        """
        Use class_mapping to filter class belongs to specific cluster_name
        :param images_numpy: numpy/tensor with shape [bs, c, h, w]
        :param gt_labels_numpy: true labels in the domain, numpy with shape [bs,]
        :param domain: int domain or same size numpy as labels [bs,]. Should be consistent with domain_name.
        :param domain_name: string domain or list of string with same size as labels [bs, ]
        :param class_mapping: {domain_name: {gt_label: [label_str, re_label, cluster_name]}}
        :param cluster_name: C0-C7.
        :param cluster_idx: 0-7
        """
        label_set = np.unique(gt_labels_numpy)
        for label in label_set:
            if class_mapping[domain_name][str(label.item())][2] == cluster_name:
                re_label = class_mapping[domain_name][str(label.item())][1]
                label_str = class_mapping[domain_name][str(label.item())][0]
                images = images_numpy[gt_labels_numpy == label]
                tuple_label = (label, domain)

                '''put to clusters'''
                self.put(images=images, label=tuple_label, grad_one=torch.ones(1)[0],
                         info_dict=None,
                         cluster_idx=cluster_idx, class_centroid=None)

    def cluster_for_task(self, images_numpy, feature_extractor, model, softmax_mode='gumbel'):
        """Clustering for task centroid.
        :param images_numpy: numpy/tensor with shape [bs, c, h, w]
        :param model: clustering model to obtain img embedding and class centroid.
        :param softmax_mode: how similarity do softmax.
            choices=[gumbel, softmax]
        Return cluster_idx: numpy(), grad_one: tensor(), similarity: numpy(8), task_centroid: tensor(512)
        """
        re_labels_numpy = np.array([0 for _ in range(images_numpy.shape[0])])
        images_torch = to_device(images_numpy, self.cluster_device)
        labels_torch = to_device(re_labels_numpy, self.cluster_device)

        if self.mode == 'kmeans':
            embeddings = model.embed(images_torch)
            cluster_idxs, grad_ones, similarities, task_centroids = self.cluster_from_emb(
                embeddings, labels_torch, softmax_mode)
            # [1], [1], [1, 8], [1, 512]

        elif self.mode == 'hierarchical':
            # obtain embeddings after url
            embeddings = feature_extractor.embed(images_torch)
            task_centroids = compute_prototypes(embeddings, labels_torch, n_way=1)  # [1, 512]
            similarities, loss_rec, assigns, gates = model(task_centroids)  # [1, 8]

            cluster_idxs, grad_ones = self.cluster_from_similarities(similarities, softmax_mode)

            similarities = similarities.detach().cpu().numpy()

        return cluster_idxs[0], grad_ones[0], similarities[0], task_centroids[0], loss_rec

    def re_clustering(self, model):
        """
        collect classes in the pool then clear clusters and cluster for those classes
        """
        images, re_labels, gt_labels, domains = [], [], [], []
        re_idx = 0
        for cluster_idx, (cls_list_in_cluster, img_list_in_cluster) in enumerate(
                zip(self.current_classes(), self.current_images())):
            for class_idx, (cls, img) in enumerate(zip(cls_list_in_cluster, img_list_in_cluster)):
                images.append(img)                                          # [20, 3, 84, 84]
                re_labels.append(np.repeat(re_idx, img.shape[0]))           # [20]
                gt_labels.append(np.repeat(cls[0][0], img.shape[0]))        # [20]
                domains.append(np.repeat(cls[0][1], img.shape[0]))          # [20]
                re_idx += 1

        if len(images) > 0:
            self.clear_clusters()
            self.cluster_and_assign(
                np.concatenate(images), np.concatenate(re_labels), np.concatenate(gt_labels), np.concatenate(domains),
                model
            )

    def put(self, images, label, grad_one,
            info_dict,
            cluster_idx, class_centroid,
            update_cluster_centers=False):
        """
        Put class samples (batch of numpy images) into clusters.
        Issues to handle:
            Maximum number of classes.
                just remove the earliest one.
            Already stored class in same cluster or other cluster.
                cat images with max size: max_num_images.
            Update cluster centers with class_centroid: [feature_size, ]
            info_dict should contain `similarities`.
        """
        '''pop stored images and cat new images'''
        position = self.find_label(label)
        if position != -1:      # find exist label, cat onto it and change to current cluster_idx
            stored = self.clusters[position[0]].pop(position[1])
            stored_images = stored['images']
            stored_images = np.concatenate([stored_images, images])
            # stored_embeddings = stored['embeddings']
            # stored_embeddings = np.concatenate([stored_embeddings, embeddings])
        else:
            stored_images = images
            # stored_embeddings = embeddings

        '''remove first several images to satisfy max_num_images'''
        if stored_images.shape[0] > self.max_num_images:
            stored_images = stored_images[-self.max_num_images:]
            # stored_embeddings = stored_embeddings[-self.max_num_images:]

        '''put to cluster: cluster_idx'''
        self.clusters[cluster_idx].append({'images': stored_images, 'label': label,
                                           'grad_one': grad_one.repeat(*images.shape[-3:]),     # c,h,w
                                           'class_centroid': class_centroid.detach().cpu().numpy() if class_centroid is not None else None,
                                           **info_dict
                                           })
        '''sort according to the corresponding similarity'''
        self.clusters[cluster_idx].sort(key=lambda x: x['similarities'][cluster_idx], reverse=True)   # descending order
        if len(self.clusters[cluster_idx]) == self.max_num_classes + 1:
            '''need to remove one with smallest similarity: last one'''
            self.clusters[cluster_idx].pop(-1)

        '''update cluster centers: mov avg'''
        if self.mode == 'mov_avg' and update_cluster_centers:
            self.update_cluster_centers(cluster_idx, class_centroid)

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
            '''construct a single image for each cluster'''
            for cluster_idx, cluster in enumerate(images):
                for cls_idx, cls in enumerate(cluster):
                    imgs = np.zeros((self.max_num_images, *cls.shape[1:]))
                    if len(cls) > 0:    # contain images
                        imgs[:cls.shape[0]] = cls
                    cluster[cls_idx] = np.concatenate([
                        imgs[img_idx] for img_idx in range(self.max_num_images)], axis=-1)
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
        # context_selection, target_selection = [], []
        for re_idx, idx in enumerate(selected_class_idxs):
            images = self.clusters[cluster_idx][idx]['images']              # [bs, c, h, w]
            tuple_label = self.clusters[cluster_idx][idx]['label']          # (gt_label, domain)
            # selection = self.clusters[cluster_idx][idx]['selection']        # [bs, n_clusters]

            perm_idxs = np.random.permutation(np.arange(len(images)))
            context_images.append(images[perm_idxs[:n_shot]])
            target_images.append(images[perm_idxs[n_shot:n_shot+n_query]])
            context_labels.append([re_idx for _ in range(n_shot)])
            target_labels.append([re_idx for _ in range(n_query)])
            context_gt.append([tuple_label for _ in range(n_shot)])         # [(gt_label, domain)*n_shot]
            target_gt.append([tuple_label for _ in range(n_query)])         # [(gt_label, domain)*n_query]
            # context_selection.append(selection[perm_idxs[:n_shot]])
            # target_selection.append(selection[perm_idxs[n_shot:n_shot+n_query]])

        context_images = np.concatenate(context_images)
        target_images = np.concatenate(target_images)
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
            context_labels = torch.from_numpy(context_labels).long().to(d)
            target_labels = torch.from_numpy(target_labels).long().to(d)

        task_dict = {
            'context_images': context_images,           # shape [n_shot*n_way, 3, 84, 84]
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'context_gt': context_gt,                   # shape [n_shot*n_way, 2]: [local, domain]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
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

    def batch_sample(self, cluster_idx):
        """
        Batch sampler for train.
        """
        pass

    def batch_sample_from_buffer(self, batch_size):
        """
        Batch sampler for train from buffer.
        Buffer is a list of image_dict = {'image': image, 'label': re_label}.
        The first batch_size sample is pop and return.
        """
        image_dicts = self.buffer[:batch_size]  # can be less than batch_size
        images = torch.stack([image_dict['image'] for image_dict in image_dicts])
        labels = torch.stack([image_dict['label'] for image_dict in image_dicts])

        '''pop these sample'''
        self.buffer = self.buffer[batch_size:]  # can be less than batch_size

        return {'images': images, 'labels': labels}


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
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
            'target_labels': target_labels,             # shape [n_query*n_way,]
            }
        meta_info: {'probability': probability of chosen which background,
         'lam': the chosen background for each image [(n_shot+n_query)* n_way,]}
        """
        # identify image size
        _, c, h, w = task_list[0]['context_images'].shape
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

        # generate lam, which is the index of img to be background. other imgs are foreground.
        # based on weight as probability.
        probability = self.ref[mix_id]      # shape [num_sources, ], sum = 1
        lam = np.random.choice(self.num_sources, context_size+target_size, p=probability, replace=True)
        # lam with shape [context_size+target_size,] is the decision to use which source as background.

        mix_imgs = []   # mix images batch
        mix_labs = []   # mix relative labels batch, same [0,0,1,1,2,2,...]
        # mix_gtls = []   # mix gt labels batch, str((weighted local label, domain=-1))
        for idx in range(context_size+target_size):
            if idx < context_size:
                set_name = 'context_images'
                lab_name = 'context_labels'
            else:
                set_name = 'target_images'
                lab_name = 'target_labels'
            # gtl_name = 'context_gt' if img_idx < context_size else 'target_gt'

            img_idx = idx if idx < context_size else idx - context_size     # local img idx in context and target set.
            # mix img is first cloned with background.
            mix_img = task_list[lam[idx]][set_name][img_idx].copy()

            # for other foreground, cut the specific [posihs: posihs+cuth, posiws: posiws+cutw] region to
            # mix_img's [posiht: posiht+cuth, posiwt: posiwt+cutw] region
            for fore_img_idx in np.delete(np.arange(self.num_sources), lam[idx]):  # idxs for other imgs
                # pick pixels from [posihs, posiws, cuth, cutw], then paste to [posiht, posiwt, cuth, cutw]
                posihs = np.random.randint(h - cuth)
                posiws = np.random.randint(w - cutw)
                posiht = np.random.randint(h - cuth)
                posiwt = np.random.randint(w - cutw)

                fore = task_list[fore_img_idx][set_name][img_idx][:, posihs: posihs + cuth, posiws: posiws + cutw]
                mix_img[:, posiht: posiht + cuth, posiwt: posiwt + cutw] = fore

                mix_imgs.append(mix_img)

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
            'context_labels': np.array(mix_labs[:context_size]),   # shape [n_shot*n_way,]
            'target_images': np.stack(mix_imgs[context_size:]),     # shape [n_query*n_way, 3, 84, 84]
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
    ref_point = np.array([ref for _ in range(num_obj)])

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


def draw_objs(objs, labels):
    """
    return a figure of objs.
    objs: numpy with shape [obj_size, pop_size]
    labels: list of labels: ['p0', 'p1', 'm0', 'm1']
    """
    obj_size, pop_size = objs.shape
    fig, ax = plt.subplots()
    ax.grid(True)
    c = plt.get_cmap('rainbow', pop_size)
    for pop_idx in range(pop_size):
        ax.scatter(objs[0, pop_idx], objs[1, pop_idx], s=200, color=c(pop_idx), label=labels[pop_idx])
    ax.legend()
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


if __name__ == '__main__':
    cal_hv_loss(np.array([[1,0], [0,1]]), 2)

    _n_way, _n_shot, _n_query = available_setting([np.array([20,20,20,20,20]), np.array([])], task_type='standard')

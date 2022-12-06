from typing import List, Dict, Any, Optional
import os
import shutil
import json

import numpy as np
import torch
import torch.nn.functional as F
from pymoo.util.ref_dirs import get_reference_directions

from models.losses import compute_prototypes
from data.meta_dataset_reader import MetaDatasetReader

from config import args
from utils import check_dir


class Pool:
    """
    Pool stored class samples for the current clustering.

    A class instance contains (a set of image samples, class_label, class_label_str).
    """
    def __init__(self, capacity=8, max_num_classes=20, max_num_images=20):
        """
        :param capacity: Number of clusters. Typically 8 columns of classes.
        :param max_num_classes: Maximum number of classes can be stored in each cluster.
        :param max_num_images: Maximum number of images can be stored in each class.
        """
        self.capacity = capacity
        self.max_num_classes = max_num_classes
        self.max_num_images = max_num_images
        self.emb_dim = 512
        self.load_path = os.path.join(args['model.dir'], 'weights', 'pool')
        self.out_path = os.path.join(args['out.dir'], 'weights', 'pool')
        self.out_path = check_dir(self.out_path, False)
        self.clusters: List[List[Dict[str, Any]]] = [[] for _ in range(self.capacity)]
        self.centers: List[Optional[torch.Tensor]] = [None for _ in range(self.capacity)]
        # instance with key {images, label}
        self.init()

    def init(self):
        self.clusters = [[] for _ in range(self.capacity)]
        self.centers = [None for _ in range(self.capacity)]

    def store(self, epoch, loader, is_best, class_filename='pool.json', center_filename='pool.npy'):
        """
        Store pool to json file.
        Only label information is stored for analysis, images are not stored for saving storage.
        Cluster centers are also stored.
        """
        pool_dict = dict(epoch=epoch + 1)

        cu_cl = self.current_classes()
        for cluster_idx in range(len(cu_cl)):
            pool_dict[cluster_idx] = []
            pool_dict[f'{cluster_idx}_str'] = []
            for cls_idx in range(len(cu_cl[cluster_idx])):
                str_label = loader.label_to_str(cu_cl[cluster_idx][cls_idx][0])
                pool_dict[cluster_idx].append(cu_cl[cluster_idx][cls_idx][0])
                pool_dict[f'{cluster_idx}_str'].append(str_label)

        path = os.path.join(self.out_path, class_filename)
        with open(path, 'w') as f:
            json.dump(pool_dict, f)

        if is_best:
            shutil.copyfile(os.path.join(self.out_path, class_filename),
                            os.path.join(self.out_path, 'pool_best.json'))

        centers = torch.stack(self.centers).numpy()     # may raise exception if contains None.
        path = os.path.join(self.out_path, center_filename)
        np.save(path, centers)

        if is_best:
            shutil.copyfile(os.path.join(self.out_path, center_filename),
                            os.path.join(self.out_path, 'pool_best.npy'))

    def restore(self, center_filename='pool.npy'):
        """
        Restore pool's centers from npy file.
        """
        self.init()
        centers = np.load(os.path.join(self.load_path, center_filename))
        self.centers = [item for item in torch.from_numpy(centers)]     # tensor: 8*512 -> list: 8 *[512]

    def clustering(self, images_numpy, re_labels_numpy, gt_labels_numpy, domain,
                   loader, devices, models, iter, mode=args['train.cluster_mode'],
                   update_cluster_centers=True,
                   only_return_cluster_idx=False):
        """
        Do clustering based on class centroid similarities with cluster centers.
        """
        '''move images to all devices, dict {device, images}'''
        images_all = loader.to_all_devices(images_numpy)
        # labels_all = train_loader.to_all_devices(re_labels_numpy)   # related labels
        re_label_set = np.unique(re_labels_numpy)       # unique also do sorting

        '''for each label, obtain class_centroid on all models and calculate similarities to all centers'''
        cluster_idxs = []
        labels = []
        for re_label in re_label_set:
            similarities = []
            centroids = []
            for model_id, (d, model) in enumerate(zip(devices, models)):
                images = images_all[d][re_labels_numpy == re_label]
                features = model.embed(images)
                centroid = features.mean(0)  # [emb_dim, ]
                similarity = self.get_distances(
                    centroid.unsqueeze(0), model_id, distance=args['test.distance']).item()
                similarities.append(similarity)
                centroids.append(centroid)

            '''cluster_idx is obtained with probability'''
            similarities = F.softmax(torch.tensor(similarities), dim=0).numpy()
            if mode == 'probability' or iter < 50:  # for cluster init
                cluster_idx = np.random.choice(len(similarities), p=similarities)
            elif mode == 'argmax':
                cluster_idx = np.argmax(similarities)
            else:
                raise Exception(f"Un-implemented clustering mode {args['train.cluster_mode']}")

            '''put samples into pool with (gt_label, domain)'''
            images = images_numpy[re_labels_numpy == re_label]
            gt_label = gt_labels_numpy[re_labels_numpy == re_label][0].item()
            label = (gt_label, domain)
            labels.append(label)
            cluster_idxs.append(cluster_idx)
            if not only_return_cluster_idx:
                self.put(images, label, cluster_idx, centroids[cluster_idx], update_cluster_centers)

        return labels, cluster_idxs

    def put(self, images, label, cluster_idx, class_centroid, update_cluster_centers=True):
        """
        Put class samples (batch of numpy images) into clusters.
        Issues to handle:
            Maximum number of classes.
                just remove the earliest one.
            Already stored class in same cluster or other cluster.
                cat images with max size: max_num_images.
            Update cluster centers with class_centroid: [feature_size, ]
        """
        '''pop stored images and cat new images'''
        position = self.find_label(label)
        if position != -1:      # find exist label, cat onto it and change to current cluster_idx
            stored_images = self.clusters[position[0]].pop(position[1])['images']
            stored_images = np.concatenate([stored_images, images])
        else:
            stored_images = images

        '''remove first several images to satisfy max_num_images'''
        while stored_images.shape[0] > self.max_num_images:
            stored_images = np.delete(stored_images, 0, axis=0)

        '''put to cluster: cluster_idx'''
        if len(self.clusters[cluster_idx]) == self.max_num_classes:
            self.clusters[cluster_idx].pop(0)   # remove the earliest cls.
        self.clusters[cluster_idx].append({'images': stored_images, 'label': label})

        '''update cluster centers'''
        if update_cluster_centers:
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

    def find_label(self, label):
        """
        Find label in pool, return position with (cluster_idx, cls_idx)
        If not in pool, return -1.
        """
        for cluster_idx, cluster in enumerate(self.clusters):
            for cls_idx, cls in enumerate(cluster):
                if cls['label'] == label:
                    return cluster_idx, cls_idx
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

    def current_images(self):
        """
        Return a batch of images (torch.Tensor) in the current pool with pool_montage.
        batch of images => (10, 3, 84, 84)
        class_montage => (3, 84, 84*10)
        cluster montage => (3, 84*max_num_classes, 84*10)
        pool montage => (3, 84*max_num_classes, 84*10*capacity).

        first return raw list, (8, num_class_each_cluster, 10, 3, 84, 84)
        """
        images = []
        for cluster in self.clusters:
            imgs = []
            for cls in cluster:
                imgs.append(cls['images'])      # cls['images'] shape [10, 3, 84, 84]
            images.append(imgs)
        return images

    def episodic_sample(
            self,
            cluster_idx,
            n_way=args['train.n_way'],
            n_shot=args['train.n_shot'],
            n_query=args['train.n_query'],
            remove_sampled_classes=False
    ):
        """
        Sample a task from the specific cluster_idx.
        length of this cluster needs to be guaranteed larger than n_way.
        Random issue may occur, highly recommended to use np.rng.
        Return numpy, need to put to devices
        """
        candidate_class_idxs = np.arange(len(self.clusters[cluster_idx]))
        num_imgs = np.array([cls[1] for cls in self.current_classes()[cluster_idx]])
        candidate_class_idxs = candidate_class_idxs[num_imgs >= args['train.n_shot'] + args['train.n_query']]
        assert len(candidate_class_idxs) >= n_way

        selected_class_idxs = np.random.choice(candidate_class_idxs, n_way, replace=False)
        context_images, target_images, context_labels, target_labels, context_gt, target_gt = [], [], [], [], [], []
        for re_idx, idx in enumerate(selected_class_idxs):
            images = self.clusters[cluster_idx][idx]['images']
            tuple_label = self.clusters[cluster_idx][idx]['label']

            perm_idxs = np.random.permutation(np.arange(len(images)))
            context_images.append(images[perm_idxs[:n_shot]])
            target_images.append(images[perm_idxs[n_shot:n_shot+n_query]])
            context_labels.append([re_idx for _ in range(n_shot)])
            target_labels.append([re_idx for _ in range(n_query)])
            context_gt.append([tuple_label for _ in range(n_shot)])
            target_gt.append([tuple_label for _ in range(n_query)])

        context_images = np.concatenate(context_images)
        target_images = np.concatenate(target_images)
        context_labels = np.concatenate(context_labels)
        target_labels = np.concatenate(target_labels)
        context_gt = np.concatenate(context_gt)
        target_gt = np.concatenate(target_gt)

        task_dict = {
            'context_images': context_images,   # shape [n_shot*n_way, 3, 84, 84]
            'context_labels': context_labels,   # shape [n_shot*n_way,]
            'context_gt': context_gt,           # shape [n_shot*n_way, 2]: [local, domain]
            'target_images': target_images,     # shape [n_query*n_way, 3, 84, 84]
            'target_labels': target_labels,     # shape [n_query*n_way,]
            'target_gt': target_gt,             # shape [n_query*n_way, 2]: [local, domain]
            'domain': cluster_idx,              # C0 - C7, num_clusters
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

    def to_torch(self, sample, device_list=None):
        """
        Put samples to a list of devices specified by device_list.
        Return a dict keying by device.
        """
        if device_list is None:
            from utils import device
            device_list = [device]

        sample_dict = {}

        for d in device_list:
            s = dict()
            for key, val in sample.items():
                if isinstance(val, str):
                    s[key] = val
                    continue
                val = torch.from_numpy(np.array(val))
                if 'image' not in key:
                    val = val.long()

                s[key] = val.to(d)

            sample_dict[d] = s

        return sample_dict


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
        # eliminate num_obj extreme cases.
        check = np.sum(self.ref == 1, axis=1) == 0             # [[0,0,1]] == 1 => [[False, False, True]]
        # np.sum(weights == 1, axis=1): array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.ref = self.ref[check]      # shape [num_mixes, num_sources]    e.g. [[0.334, 0.666], [0.666, 0.334]]
        # self.ref = get_reference_directions("energy", num_obj, num_mix, seed=1)  # use those [1, 0, 0]
        assert self.ref.shape[0] == num_mixes

    def _cutmix(self, task_list, mix_id):
        """
        Apply cutmix on the task_list.
        mix_id is used to identify which ref to use as a probability.

        return:
        task_dict = {
            'context_images': context_images,   # shape [n_shot*n_way, 3, 84, 84]
            'context_labels': context_labels,   # shape [n_shot*n_way,]
            'target_images': target_images,     # shape [n_query*n_way, 3, 84, 84]
            'target_labels': target_labels,     # shape [n_query*n_way,]
        }
        {'probability': probability of chosen which background,
         'lam': the chosen background for each image [(n_shot+n_query)* n_way,]}
        """
        # identify image size
        _, c, h, w = task_list[0]['context_images'].shape
        context_size = np.min([task_list[idx]['context_images'].shape[0] for idx in range(len(task_list))])
        target_size = np.min([task_list[idx]['target_images'].shape[0] for idx in range(len(task_list))])
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
            set_name = 'context_images' if idx < context_size else 'target_images'
            lab_name = 'context_labels' if idx < context_size else 'target_labels'
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

    return hv

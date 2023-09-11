import numpy as np
import torch
import copy


class Pool:
    def __init__(self):
        self.capacity = 10
        self.max_num_images = 20
        self.max_num_classes = 10
        self.buffer_size = 200

        self.buffer = []
        self.clusters = [[] for _ in range(self.capacity)]

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
            # assert len(np.unique(gt_labels[mask])) == len(np.unique(domains[mask])) == 1
            class_images = images[mask].numpy()
            class_similarities = similarities[mask]

            '''pop stored images and cat new images'''
            position = self.find_label(label, target='buffer')
            if position != -1:  # find exist label, cat onto it and re-put
                stored = self.buffer.pop(position)
                assert (stored['label'] == label).all()
                stored_images = np.concatenate([stored['images'], class_images])
                stored_similarities = np.concatenate([stored['similarities'], class_similarities])
            else:
                stored_images = class_images
                stored_similarities = class_similarities

            '''remove same image'''
            stored_images, img_idxes = np.unique(stored_images, return_index=True, axis=0)
            stored_similarities = stored_similarities[img_idxes]

            class_dict = {
                'images': stored_images, 'label': label,  # 'selection': stored_selection,
                'similarities': stored_similarities,
                # 'class_similarity': np.mean(stored_similarities, axis=0),  # mean over all samples [n_clusters]
            }

            '''put into buffer'''
            self.buffer.append(class_dict)

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


pool = Pool()

domain_dict = []
gt_labels_dict = []
images_dict = []
similarities_dict = []
for cls_id in range(100):
    domain = np.array([cls_id for _ in range(30)])
    gt_labels = np.array([cls_id for _ in range(30)])
    images = np.random.randn(30, 3, 5, 5)
    similarities = np.random.randn(30, 10)
    domain_dict.append(domain)
    gt_labels_dict.append(gt_labels)
    images_dict.append(images)
    similarities_dict.append(similarities)

domain_dict = np.concatenate(domain_dict)
gt_labels_dict = np.concatenate(gt_labels_dict)
images_dict = np.concatenate(images_dict)
similarities_dict = np.concatenate(similarities_dict)

perm = np.random.permutation(len(domain_dict))
domain_dict = domain_dict[perm]
gt_labels_dict = gt_labels_dict[perm]
images_dict = images_dict[perm]
similarities_dict = similarities_dict[perm]

pool.put_buffer(
    torch.from_numpy(images_dict), {'domain': domain_dict, 'gt_labels': gt_labels_dict, 'similarities': similarities_dict},
    maintain_size=False)


# todo: check sim keeps the same
checks = []
for cls in pool.buffer:
    for image_idx, image in enumerate(cls['images']):
        sim = cls['similarities'][image_idx]
        label = cls['label']
        for img_idx, img in enumerate(images_dict):
            if (image == img).all():
                if (label[0] == gt_labels_dict[img_idx]
                ) and label[1] == domain_dict[img_idx] and (sim == similarities_dict[img_idx]).all():
                    checks.append(True)
                else:
                    checks.append(False)
print(f'checks: {np.sum(checks)}.')

pool.buffer_copy = copy.deepcopy(pool.buffer)

pool.buffer2cluster()

cases = []
for _ in range(100):
    # todo: track a specific image sample
    anchor_cluster_index = np.random.choice(len(pool.clusters))
    anchor_cls_index = np.random.choice(len(pool.clusters[anchor_cluster_index]))
    anchor_img_index = np.random.choice(len(pool.clusters[anchor_cluster_index][anchor_cls_index]['images']))
    anchor_img = pool.clusters[anchor_cluster_index][anchor_cls_index]['images'][anchor_img_index]
    anchor_label = pool.clusters[anchor_cluster_index][anchor_cls_index]['label']
    anchor_sim = pool.clusters[anchor_cluster_index][anchor_cls_index]['similarities'][anchor_img_index]
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

print(np.array(cases).sum(0))

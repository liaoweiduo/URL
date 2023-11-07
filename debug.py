import copy
import os

import numpy as np
import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from pmo_utils import Pool, draw_heatmap, draw_objs


class Debugger:
    def __init__(self, activate=True):
        self.activate = activate
        self.storage = {}

    def print_prototype_change(self, model: nn.Module, i, writer: Optional[SummaryWriter] = None):
        """

        Args:
            model: target model contain prototypes
            writer: None or obj: writer
            i: iter index

        Returns:

        """
        if not self.activate:
            return

        proto = model.selector.prototypes.detach().cpu().numpy()
        if 'proto' in self.storage:
            dif = np.linalg.norm((proto - self.storage['proto']).ravel(), 2)  # l2 distance
        else:
            dif = 0
        self.storage['proto'] = proto

        print(f'proto diff (l2) is {dif}.\nmov_avg_alpha is {model.selector.mov_avg_alpha.item()}.')
        if writer is not None:
            writer.add_scalar('train_image/cluster-centers-dif', dif, i + 1)
            writer.add_scalar('params/mov_avg_alpha', model.selector.mov_avg_alpha.item(), i + 1)

    def print_grad(self, model: nn.Module, key=None, prefix=''):
        """

        Args:
            model: target model
            key: for parameter name and also for storage name for diff
            prefix: print prefix

        Returns:

        """
        if not self.activate:
            return
        vs = []
        with torch.no_grad():
            for k, v in model.named_parameters():
                if (key is None) or (key in k):
                    if v.grad is not None:
                        vs.append(v.grad.flatten())
            vs = torch.cat(vs).detach().cpu().numpy()
            if key in self.storage:
                dif = vs - self.storage[key]
            else:
                dif = vs
            self.storage[key] = vs
            dif = np.abs(dif)
            print(f'{prefix}mean abs grad diff for {key} is {np.mean(dif)}.')

    def write_pool(self, pool: Pool, i, writer: Optional[SummaryWriter] = None, prefix='pool'):
        """

        Args:
            pool: pool after call buffer2cluster()
            i: iter index
            writer: None or obj: writer
            prefix:

        Returns:

        """
        if not self.activate:
            return

        print(f'iter {i}: {prefix} num_cls info: '
              f'{[f"{idx}[{len(sim)}]" for idx, sim in enumerate(pool.current_similarities())]}.')

        if writer is not None:
            '''all images and img_sims and class_sims '''
            images = pool.current_images(single_image=True)
            for cluster_id, cluster in enumerate(images):
                if len(cluster) > 0:
                    writer.add_image(f"{prefix}-img/{cluster_id}", cluster, i + 1)
            similarities = pool.current_similarities(image_wise=True)
            class_similarities = pool.current_similarities()
            for cluster_id, (img_sim, cls_sim) in enumerate(zip(similarities, class_similarities)):
                if len(img_sim) > 0:
                    # img_sim [num_cls * [num_img, 8]]; cls_sim [num_cls * [8]]
                    sim = np.concatenate([
                        np.concatenate([img_sim[cls_idx], *[cls_sim[cls_idx][np.newaxis, :]] * len(img_sim[cls_idx])])
                        for cls_idx in range(len(img_sim))
                    ])
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"{prefix}-img-sim/{cluster_id}", figure, i+1)

    def write_scale(self, value, key, i, writer: Optional[SummaryWriter] = None):
        """

        Args:
            value:
            key: name
            i:
            writer:

        Returns:

        """
        if not self.activate:
            return

        writer.add_scalar(key, value, i + 1)

    def write_mo(self, mo_dict, pop_labels, i, writer: Optional[SummaryWriter] = None, prefix='acc'):
        """
        draw mo graph for different inner step.
        Args:
            mo_dict: {pop_idx: {inner_idx: [n_obj]}}
            i:
            writer:
            prefix:

        Returns:

        """
        if not self.activate:
            return

        n_pop = len(mo_dict)
        n_inner = len(mo_dict[0])
        n_obj = len(mo_dict[0][0])

        '''log objs figure'''
        objs = np.array([
            mo_dict[pop_idx][inner_idx][obj_idx]
            for inner_idx in range(n_inner) for obj_idx in range(n_obj) for pop_idx in range(n_pop)
        ])  # [n_inner, n_obj, n_pop]
        figure = draw_objs(objs, pop_labels)
        writer.add_figure(f"train_image/objs_{prefix}", figure, i + 1)


    # def write_task(self, pool: Pool, task: dict, i, writer: Optional[SummaryWriter] = None, prefix='pool'):
    #
    #     '''log img sim in the task'''
    #     with torch.no_grad():
    #         img_features = torch_task_features  # [img_size, 512]
    #         _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
    #         img_sim = selection_info['y_soft']  # [img_size, 10]
    #         _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
    #         tsk_sim = selection_info['y_soft']  # [1, 10]
    #     sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
    #     epoch_loss[f'mo/image_softmax_sim'][task_idx] = sim
    #
    #     '''write task images'''
    #     writer.add_images(f"task-image/image", task_images, i + 1)  # task images
    #     sim = epoch_loss['task/image_softmax_sim']
    #     figure = draw_heatmap(sim, verbose=False)
    #     writer.add_figure(f"task-image/sim", figure, i + 1)
    #     with torch.no_grad():
    #         img_features = task_features  # [img_size, 512]
    #         # img_features = pmo.embed(task_images.to(device))    # [img_size, 512]
    #         _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
    #         img_sim = selection_info['y_soft']  # [img_size, 10]
    #         _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
    #         tsk_sim = selection_info['y_soft']  # [1, 10]
    #     sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
    #     figure = draw_heatmap(sim, verbose=False)
    #     writer.add_figure(f"task-image/sim-re-cal", figure, i + 1)


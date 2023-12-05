import copy
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from pmo_utils import Pool, draw_heatmap, draw_objs, cal_hv, cal_min_crowding_distance


class Debugger:
    def __init__(self, level='DEBUG'):
        self.levels = ['DEBUG', 'INFO']
        self.level = self.levels.index(level)   # 0 or 1
        self.storage = {}

    def print_prototype_change(self, model: nn.Module, i, writer: Optional[SummaryWriter] = None):
        """

        Args:
            model: target model contain prototypes
            writer: None or obj: writer
            i: iter index

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        proto = model.selector.prototypes.detach().cpu().numpy()
        if 'proto' in self.storage:
            old_proto = self.storage['proto']
        else:
            old_proto = 0
        dif = np.linalg.norm((proto - old_proto).ravel(), 2)  # l2 distance
        self.storage['proto'] = proto

        print(f'proto diff (l2) is {dif}.\nmov_avg_alpha is {model.selector.mov_avg_alpha.item()}.')
        if writer is not None:
            writer.add_scalar('params/cluster-centers-dif', dif, i + 1)
            writer.add_scalar('params/mov_avg_alpha', model.selector.mov_avg_alpha.item(), i + 1)

    def print_grad(self, model: nn.Module, key=None, prefix=''):
        """

        Args:
            model: target model
            key: for parameter name and also for storage name for diff
            prefix: print prefix

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
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
        level = self.levels.index('INFO')
        if level < self.level:
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
                        np.concatenate([img_sim[cls_idx],
                                        *[cls_sim[cls_idx][np.newaxis, :]] * max(10, len(img_sim[cls_idx]) // 2)])
                        for cls_idx in range(len(img_sim))
                    ])
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"{prefix}-img-sim/{cluster_id}", figure, i + 1)

    def write_scaler(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix=''):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:

        Returns: last Idx's avg_value or -1

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = df[df.Tag == key]
        value = -1
        for idx in sorted(set(t_df.Idx)):
            value = t_df[t_df.Idx == idx].Value.mean()
            value = np.nan_to_num(value)
            writer.add_scalar(f'{prefix}{key}/{idx}', value, i + 1)

        print(f'{prefix}{key}: {value:.5f}.')

        return value

    def write_inner(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix=''):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = df[df.Tag == key]

        fig, ax = plt.subplots()
        ax.grid(True)
        sns.lineplot(t_df, x='Idx', y='Value', ax=ax)

        writer.add_figure(f"{prefix}{key}", fig, i + 1)

    def write_hv(self, mo_dict, i, ref=0, writer: Optional[SummaryWriter] = None, target='acc',
                 prefix='hv'):
        """

        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            ref: ref for cal hv
            writer:
            target: also for mo_dict's Tag selector.
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''cal hv for each inner mo'''
        hv = -1
        if ref == 'relative':
            ref = np.mean(objs[0], axis=-1).tolist()  # [n_obj]   to be list
        for inner_step in range(n_inner):
            hv = cal_hv(objs[inner_step], ref, target=target)
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', hv, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', hv, i + 1)

        print(f"==>> {prefix}: {target} {hv:.3f}.")

    def write_avg_span(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc',
                       prefix='avg_span'):
        """
        E_i(max(f_i) - min(f_i))
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
        max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
        objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal avg span for each inner mo'''
        avg_span = -1
        for inner_step in range(n_inner):
            avg_span = np.mean(
                [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                 range(n_obj)])
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', avg_span, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', avg_span, i + 1)

        print(f"==>> {prefix}: {target} {avg_span:.5f}.")

        return avg_span

    def write_min_crowding_distance(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc',
                                    prefix='min_cd'):
        """
        only for nd solutions. if min cd is inf, use avg_span instead.
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
        max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
        objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal min crowding distance for each inner mo (after nd sort)'''
        cd = -1
        for inner_step in range(n_inner):
            cd = cal_min_crowding_distance(objs[inner_step])
            if cd == np.inf:
                avg_span = np.mean(
                    [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                     range(n_obj)])
                cd = avg_span
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', cd, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', cd, i + 1)

        print(f"==>> {prefix}: {target} {cd:.5f}.")

        return cd

    def write_mo(self, mo_dict, pop_labels, i, writer: Optional[SummaryWriter] = None, target='acc',
                 prefix='train_image'):
        """
        draw mo graph for different inner step.
        Args:
            mo_dict: {pop_idx: {inner_idx: [n_obj]}} or
                dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            pop_labels:
            i:
            writer:
            target: for mo_dict's Tag selector.
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        if type(mo_dict) is dict:
            n_pop, n_inner, n_obj = len(mo_dict), len(mo_dict[0]), len(mo_dict[0][0])
            objs = np.array([
                [[mo_dict[pop_idx][inner_idx][obj_idx] for pop_idx in range(n_pop)]
                 for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
            ])  # [n_inner, n_obj, n_pop]

            '''log objs figure'''
            figure = draw_objs(objs, pop_labels)
            writer.add_figure(f"{prefix}/objs_{target}", figure, i + 1)

            with open(os.path.join(writer.log_dir, f'{prefix}_mo_dict_{target}.json'), 'w') as f:
                json.dump(mo_dict, f)
        else:
            # for exp in set(mo_dict.Exp):
            #     for inner_lr in set(mo_dict.Inner_lr):
            #         for logit_scale in set(mo_dict.Logit_scale):
            #             t_df = mo_dict[(mo_dict.Tag == target) &
            #                            (mo_dict.Exp == exp) & (mo_dict.Inner_lr == inner_lr) &
            #                            (mo_dict.Logit_scale == logit_scale)]
            t_df = mo_dict[mo_dict.Tag == target]
            n_pop = len(set(t_df.Pop_id))
            n_inner = len(set(t_df.Inner_id))
            n_obj = len(set(t_df.Obj_id))
            objs = np.array([[[
                t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                            t_df.Inner_id == inner_idx)].Value.mean()
                for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
            ])  # [n_inner, n_obj, n_pop]
            objs = np.nan_to_num(objs)

            '''log objs figure'''
            figure = draw_objs(objs, pop_labels)
            # writer.add_figure(f"objs_{target}_{exp}_innerlr_{inner_lr}{prefix}/logit_scale_{logit_scale}",
            #                   figure, i + 1)
            writer.add_figure(f"{prefix}/objs_{target}", figure, i + 1)

    def write_task(self, pmo, task: dict, task_title, i, writer: Optional[SummaryWriter] = None, prefix='task'):
        """

        Args:
            pmo:
            task: ['context_images', 'target_images', 'context_labels', 'target_labels']
            task_title:
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        '''write images'''
        imgs = torch.cat([task['context_images'], task['target_images']]).cuda()
        numpy_imgs = imgs.cpu().numpy()
        writer.add_images(f"{prefix}/{task_title}", numpy_imgs, i+1)

        '''log img sim in the task'''
        with torch.no_grad():
            img_features = pmo.embed(imgs)
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
            img_sim = selection_info['y_soft']  # [img_size, 10]
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
            tsk_sim = selection_info['y_soft']  # [1, 10]
        sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
        figure = draw_heatmap(sim, verbose=False)
        writer.add_figure(f"{prefix}/{task_title}/sim", figure, i + 1)

import copy
import os

import numpy as np
import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


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
            dif = np.linalg.norm(proto - self.storage['proto'], 2)  # l2 distance
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

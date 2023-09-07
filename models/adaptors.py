"""
This code allows you to use adaptors for aligning features 
between multi-domain learning network and single domain learning networks.
The code is adapted from https://github.com/VICO-UoE/KD4MTL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from typing import Optional


class adaptor(torch.nn.Module):
    def __init__(self, num_datasets, dim_in, dim_out=None, opt='linear'):
        super(adaptor, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.num_datasets = num_datasets

        for i in range(num_datasets):
            if opt == 'linear':
                setattr(self, 'conv{}'.format(i), torch.nn.Conv2d(dim_in, dim_out, 1, bias=False))
            else:
                # setattr(self, 'conv{}'.format(i), nn.Sequential(
                #     torch.nn.Conv2d(dim_in, 2*dim_in, 1, bias=False),
                #     torch.nn.ReLU(True),
                #     torch.nn.Conv2d(2*dim_in, dim_out, 1, bias=False),
                #     )
                setattr(self, 'conv{}'.format(i), nn.Sequential(
                    torch.nn.Conv2d(dim_in, dim_in // 4, 1, bias=False),    # 512 -> 128
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(dim_in // 4, dim_out, 1, bias=False),     # 128 -> 32
                    )
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs, idxs: Optional[list] = None):
        if idxs is None:
            idxs = list(range(self.num_datasets))
        assert len(inputs) == len(idxs)
        results = []
        for i, idx in enumerate(idxs):
            ad_layer = getattr(self, 'conv{}'.format(idx))
            if len(list(inputs[i].size())) < 4:
                input_ = inputs[i].view(inputs[i].size(0), -1, 1, 1)
            else:
                input_ = inputs[i]
            results.append(ad_layer(input_).flatten(1))
            # results.append(ad_layer(inputs[i]))
        return results

    def to_device(self, device_list):
        assert len(device_list) == self.num_datasets
        for i in range(self.num_datasets):
            setattr(self, 'conv{}'.format(i), getattr(self, 'conv{}'.format(i)).to(device_list[i]))

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.named_parameters()]







# -*- coding: utf-8 -*-
# @Time: 2026/1/4
# @File: unpooling.py
# @Author: fwb
import torch.nn as nn
from functools import partial
from model.components import DataDict
from model.layers.sequential import DataModule, DataSequential


class GraphUnpooling(DataModule):
    def __init__(
            self,
            in_chs,
            skip_chs,
            out_chs,
            norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            act_layer=nn.GELU,
            traceable=False,  # record parent and cluster.
    ):
        super().__init__()
        self.proj = DataSequential(nn.Linear(in_chs, out_chs))
        self.proj_skip = DataSequential(nn.Linear(skip_chs, out_chs))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_chs))
            self.proj_skip.add(norm_layer(out_chs))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, data_dict: DataDict):
        assert 'pooling_parent' in data_dict.keys()
        assert 'pooling_inverse' in data_dict.keys()
        parent = data_dict.pop('pooling_parent')
        inverse = data_dict.pop('pooling_inverse')
        data_dict = self.proj(data_dict)
        parent = self.proj_skip(parent)
        parent.x = parent.x + data_dict.x[inverse]

        if self.traceable:
            parent['unpooling_parent'] = data_dict

        return parent

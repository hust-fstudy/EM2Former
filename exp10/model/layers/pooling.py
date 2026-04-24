# -*- coding: utf-8 -*-
# @Time: 2026/3/17
# @File: pooling.py
# @Author: fwb
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import voxel_grid, max_pool, avg_pool
from functools import partial
from model.components import DataDict
from model.layers.sequential import DataModule, DataSequential


class GraphPooling(DataModule):
    def __init__(
            self,
            in_chs,
            out_chs,
            pooling_size,
            width,
            height,
            batch_size,
            transform=None,
            pooling_aggr='max',
            traceable=True,  # record parent and cluster.
            norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            act_layer=nn.GELU,
    ):
        super(GraphPooling, self).__init__()
        assert pooling_aggr in ['mean', 'max']
        self.pooling_aggr = pooling_aggr
        self.register_buffer("pooling_size", pooling_size, persistent=False)

        self.transform = transform
        self.traceable = traceable

        self.register_buffer("start", torch.Tensor([0, 0, 0, 0]), persistent=False)
        self.register_buffer("end", torch.Tensor([0.9999999, 0.9999999, 0.9999999]), persistent=False)
        self.register_buffer("wh_inv", 1 / torch.Tensor([[width, height]]), persistent=False)

        self.max_num_voxels = batch_size * self.num_grid_cells
        self.register_buffer("sorted_cluster", torch.arange(self.max_num_voxels), persistent=False)

        self.proj = nn.Linear(in_chs, out_chs)

        self.norm = norm_layer
        self.act = act_layer
        if norm_layer is not None:
            self.norm = DataSequential(norm_layer(out_chs))
        if act_layer is not None:
            self.act = DataSequential(act_layer())

    @property
    def num_grid_cells(self):
        return (1 / self.pooling_size + 1e-3).int().prod()

    def round_to_pixel(self, pos, wh_inv):
        pos = torch.div(pos + 1e-5, wh_inv, rounding_mode='floor')
        return pos * wh_inv

    def forward(self, data_dict: DataDict):
        cluster = voxel_grid(pos=data_dict.pos, batch=data_dict.batch, size=self.pooling_size,
                             start=self.start, end=self.end)
        new_data_dict = Data(x=self.proj(data_dict.x), pos=data_dict.pos,
                             batch=data_dict.batch, edge_index=data_dict.edge_index)
        if self.transform is not None:
            if new_data_dict.edge_index.numel() > 0:
                if self.pooling_aggr == 'max':
                    new_data_dict = max_pool(cluster, new_data_dict, transform=self.transform)
                else:
                    new_data_dict = avg_pool(cluster, new_data_dict, transform=self.transform)
        else:
            if self.pooling_aggr == 'max':
                new_data_dict = max_pool(cluster, new_data_dict)
            else:
                new_data_dict = avg_pool(cluster, new_data_dict)

        if hasattr(data_dict, 'height'):
            new_data_dict.height = data_dict.height
            new_data_dict.width = data_dict.width

        # Round x and y coordinates to the center of the voxel grid.
        new_data_dict.pos[:, :2] = self.round_to_pixel(new_data_dict.pos[:, :2], wh_inv=self.wh_inv)

        if self.traceable:
            _, inv, _ = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
            new_data_dict['pooling_inverse'] = inv
            new_data_dict['pooling_parent'] = data_dict

        # Collect information.
        new_data_dict = DataDict(new_data_dict)
        if self.norm is not None:
            new_data_dict = self.norm(new_data_dict)
        if self.act is not None:
            new_data_dict = self.act(new_data_dict)
        new_data_dict['grid_size'] = self.pooling_size[:3].clone().detach().to(new_data_dict.x.device)
        new_data_dict.sparsify()

        return new_data_dict

# -*- coding: utf-8 -*-
# @Time: 2026/3/17
# @File: net_utils.py
# @Author: fwb
import torch
import torch_geometric.transforms as T


def calc_pooling_at_each_stage(pooling_dim, num_stages):
    px, py = map(int, pooling_dim)
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    pooling_size = []
    for i in range(num_stages):
        pooling = pooling_base / 2 ** (num_stages - 1 - i)
        pooling[-1] = 1
        pooling_size.append(pooling)
    pooling_size = torch.stack(pooling_size)
    return pooling_size


class Cartesian(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        T.Cartesian.__init__(self, *args, **kwargs)

    def forward(self, data):
        if data.edge_index.shape[1] > 0:
            return T.Cartesian.forward(self, data)
        else:
            data.edge_attr = torch.zeros((0, 3), dtype=data.x.dtype, device=data.x.device)
            return data

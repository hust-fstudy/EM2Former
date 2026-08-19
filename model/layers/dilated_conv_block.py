# -*- coding: utf-8 -*-
# @Time: 2026/3/18
# @File: dilated_conv_block.py
# @Author: fwb
from torch import nn
from model.components import DataDict
from model.layers.sequential import DataModule, DataSequential
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm
from model.layers.sparse_conv import SparseDilatedLayer


class BatchNormData(BatchNorm):
    def forward(self, data: Data):
        data.x = BatchNorm.forward(self, data.x)
        return data


class Linear(nn.Module):
    def __init__(self, in_chs, out_chs, bias=True):
        nn.Module.__init__(self)
        self.mlp = nn.Linear(in_chs, out_chs, bias=bias)

    def forward(self, data: Data):
        data.x = self.mlp(data.x)
        return data


class DilatedConvBlock(DataModule):
    def __init__(
            self,
            dim,
            down_kernel_size=(3, 3, 3),
            down_stride=(1, 2, 2),
            num_sparse_block=(2, 1, 1),
            act_layer=nn.ReLU,
            xy_only=True
    ):
        super().__init__()

        self.sd_layer = SparseDilatedLayer(
            dim=dim,
            down_kernel_size=list(down_kernel_size),
            down_stride=list(down_stride),
            num_sparse_block=list(num_sparse_block),
            indice_key='sedlayer2',
            xy_only=xy_only
        )

        self.lin = Linear(dim, dim, bias=False)
        self.norm_lin = BatchNormData(in_channels=dim)

        self.activation = act_layer()

    def forward(self, data: Data, pooling_size) -> DataDict:
        skip_dict = DataDict(
            x=data.x.clone(),
            pos=data.pos[:, :2],
            grid_size=pooling_size[:2].clone().detach().to(data.x.device),
            sparse_shape=(int(1 / pooling_size[0]), int(1 / pooling_size[1])),
            batch=data.batch
        )
        skip_dict.sparsify()

        skip_dict.sparse_conv_feat = self.sd_layer(skip_dict.sparse_conv_feat)
        skip_dict.x = skip_dict.sparse_conv_feat.features

        data = self.lin(data)
        data = self.norm_lin(data)
        data_dict = DataDict(data)

        data_dict.x = self.activation(data_dict.x + skip_dict.x)

        return data_dict

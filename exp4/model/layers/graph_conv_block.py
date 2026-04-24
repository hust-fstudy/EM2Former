# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: conv.py
# @Author: fwb
import torch
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Data
from model.layers.graph_conv import GraphConv
from model.model_utils import shallow_copy


class BatchNormData(BatchNorm):
    def forward(self, data: Data):
        data.x = BatchNorm.forward(self, data.x)
        return data


class Linear(torch.nn.Module):
    def __init__(self, in_chs, out_chs, bias=True):
        torch.nn.Module.__init__(self)
        self.mlp = torch.nn.Linear(in_chs, out_chs, bias=bias)

    def forward(self, data: Data):
        data.x = self.mlp(data.x)
        return data


class GraphConvLayer(torch.nn.Module):
    def __init__(self, args, in_chs: int, out_chs: int, degree=1) -> None:
        super(GraphConvLayer, self).__init__()
        self.dim = args.edge_attr_dim
        self.activation = args.activation

        self.graph_conv = GraphConv(
            args=args,
            in_chs=in_chs,
            out_chs=out_chs,
            degree=degree,
            bias=False
        )

        self.norm = BatchNormData(in_channels=out_chs)
        self.activation = getattr(torch.nn.functional, self.activation, torch.nn.functional.elu)

    def forward(self, data: Data) -> torch.Tensor:
        data = self.graph_conv(data)
        data = self.norm(data)
        data.x = self.activation(data.x)

        return data


class GraphConvLayerWithSkip(torch.nn.Module):
    def __init__(self, args, skip_in_chs: int, out_chs: int) -> None:
        super(GraphConvLayerWithSkip, self).__init__()
        self.dim = args.edge_attr_dim
        self.activation = args.activation

        self.lin = Linear(skip_in_chs, out_chs, bias=False)
        self.norm_skip = BatchNormData(in_channels=out_chs)
        self.activation = getattr(torch.nn.functional, self.activation, torch.nn.functional.elu)

    def forward(self, data: Data, data_skip: Data):
        data_skip = self.lin(data_skip)
        data_skip = self.norm_skip(data_skip)

        data.x = self.activation(data.x + data_skip.x)

        return data


class GraphConvBlock(torch.nn.Module):
    def __init__(self, args, in_chs: int, out_chs: int) -> None:
        super(GraphConvBlock, self).__init__()
        self.in_channel = in_chs
        self.out_channel = out_chs

        self.graph_conv_layer1 = GraphConvLayer(args=args, in_chs=in_chs, out_chs=out_chs)
        self.graph_conv_layer2 = GraphConvLayerWithSkip(args=args, skip_in_chs=in_chs, out_chs=out_chs)

    def forward(self, data: Data) -> torch.Tensor:
        data_skip = shallow_copy(data)
        data = self.graph_conv_layer1(data)
        output = self.graph_conv_layer2(data, data_skip)

        return output

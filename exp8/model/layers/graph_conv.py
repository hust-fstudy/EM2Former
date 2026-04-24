# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: graph_conv.py
# @Author: fwb
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, GMMConv, SplineConv


class GraphConv(nn.Module):
    def __init__(self, args, in_chs, out_chs, degree=1, bias=False):
        super(GraphConv, self).__init__()
        # Init gnn parameters.
        self.sel_gnn = args.sel_gnn
        self.gat_heads = args.gat_heads
        self.aggr = args.aggr
        self.kernel_size = args.kernel_size
        self.edge_attr_dim = args.edge_attr_dim

        match self.sel_gnn:
            case 'GCN':
                self.conv = GCNConv(in_channels=in_chs, out_channels=out_chs, bias=bias)
            case 'GAT':
                self.conv = GATConv(in_channels=in_chs, out_channels=out_chs, heads=self.gat_heads,
                                    concat=False, bias=bias)
            case 'SAGE':
                self.conv = SAGEConv(in_channels=in_chs, out_channels=out_chs, aggr=self.aggr, bias=bias)
            case 'GMM':
                self.conv = GMMConv(in_channels=in_chs, out_channels=out_chs, dim=self.edge_attr_dim,
                                    kernel_size=self.kernel_size, aggr=self.aggr, bias=bias)
            case 'Spline':
                self.conv = SplineConv(in_channels=in_chs, out_channels=out_chs, dim=self.edge_attr_dim,
                                       kernel_size=self.kernel_size, degree=degree, aggr=self.aggr, bias=bias)
            case _:
                print(f"The {self.sel_gnn} does not exist!")

    def forward(self, data: Data) -> Data:
        if not hasattr(data, 'adj_t'):
            data.edge_attr = data.edge_attr[:, :self.edge_attr_dim]
        if self.sel_gnn in ['GCN', 'SAGE']:
            data.x = self.conv(x=data.x, edge_index=data.edge_index)
        elif self.sel_gnn in ['GAT', 'GMM', 'Spline']:
            data.x = self.conv(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        else:
            data.x = None
            print(f"The {self.sel_gnn} does not exist!")

        return data


def to_dense(self, x, pos, pooling, batch=None, batch_size=None, proj_mode='sum'):
    if batch_size is not None:
        self.batch_size = batch_size
        B = batch_size
    elif batch is None:
        batch = torch.zeros(size=(len(x),), dtype=torch.long, device=x.device)
        B = 1
        self.batch_size = B
    else:
        B = batch.max().item() + 1
        self.batch_size = B

    if not hasattr(self, "dense"):
        W, H = (1 / pooling[:2] + 1e-3).long()
        C = x.shape[-1]
        if proj_mode in ['overwrite', 'mean', 'sum']:
            self.dense = torch.zeros(size=(B, C, H, W), dtype=x.dtype, device=x.device)
        elif proj_mode in ['max']:
            self.dense = torch.full((B, C, H, W), torch.inf, dtype=x.dtype, device=x.device)

    est_x, est_y = (pos[:, :2] / pooling[:2]).t().long()

    self.dense = self.dense.detach()
    self.dense.zero_()

    dense = self.dense[:B] if B < self.dense.shape[0] else self.dense
    B, C, H, W = dense.shape

    if proj_mode == 'overwrite':
        dense[batch.long(), :, est_y, est_x] = x
    else:
        dense = dense.view(-1, C)
        flat_indices = batch * (H * W) + est_y * W + est_x
        unique_idx = flat_indices.unsqueeze(-1).expand(-1, C)
        if proj_mode in ['mean', 'sum']:
            dense = torch.scatter_reduce(
                dense, dim=0, index=unique_idx,
                src=x, reduce=proj_mode
            )
        elif proj_mode in ['max']:
            dense = torch.scatter_reduce(
                dense, dim=0, index=unique_idx,
                src=x, reduce='amax'
            )
            dense = torch.where(dense == -torch.inf, torch.tensor(0.0, device=x.device), dense)
        dense = dense.view(B, H, W, C).permute(0, 3, 1, 2)

    return dense


class GraphConvToDense(GraphConv):
    def __init__(self, args, in_chs, out_chs, bias):
        self.proj_mode = args.proj_mode
        super().__init__(args=args, in_chs=in_chs, out_chs=out_chs, bias=bias)

    def forward(self, data: Data, batch_size: int = None) -> torch.Tensor:
        data = super().forward(data)
        if data.batch is None:
            data.batch = torch.zeros(len(data.x), dtype=torch.long, device=data.x.device)
        return self.to_dense(data.x, data.pos, data.pooling, data.batch,
                             batch_size=batch_size, proj_mode=self.proj_mode)

    def to_dense(self, x, pos, pooling, batch=None, batch_size=None, proj_mode='sum'):
        return to_dense(self, x, pos, pooling, batch, batch_size=batch_size, proj_mode=proj_mode)

# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: det_backbone.py
# @Author: fwb
import torch
from torch import nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from functools import partial
from model.components import DataDict
from model.layers.event2graph import Event2Graph
from model.layers.pooling import GraphPooling
from model.layers.graph_conv_block import GraphConvBlock
from model.layers.attention import Block
from model.layers.sequential import DataModule, DataSequential
from model.model_utils import shallow_copy
from model.networks.net_utils import calc_pooling_at_each_stage, Cartesian


class DetBackbone(torch.nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        # Encoder.
        self.enable_flash = args.enable_flash
        self.enable_rope = args.enable_rope
        self.in_chs = args.in_chs
        self.embed_chs = args.embed_chs
        self.enc_depths = args.enc_depths
        self.enc_chs = args.enc_chs
        self.enc_heads = args.enc_heads
        self.enc_patches = args.enc_patches
        self.enc_type = args.enc_type
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.drop_path = args.drop_path

        # Dilated conv block.
        self.down_kernel_size = args.down_kernel_size
        self.down_stride = args.down_stride
        self.num_sparse_block = args.num_sparse_block
        self.xy_only = args.xy_only

        # To graph.
        self.radius = args.radius
        self.height = height
        self.width = width

        # Graph pooling.
        self.pooling_dim = args.pooling_dim
        self.pooling_aggr = args.pooling_aggr
        self.traceable = args.traceable

        # Postprocessing.
        self.batch_size = args.batch_size
        self.num_scales = args.num_scales
        self.num_classes = args.num_classes

        self.num_stages = len(self.enc_depths)
        assert self.num_stages == len(self.enc_chs)
        assert self.num_stages == len(self.enc_heads)
        assert self.num_stages == len(self.enc_patches)
        assert self.num_stages == len(self.enc_type)

        # Encoder chs.
        self.enc_chs = [self.embed_chs] + self.enc_chs
        self.out_chs = self.enc_chs[-2:]

        # Graph edges.
        self.events_to_graph = Event2Graph(args)
        effective_radius = 2 * float(int(self.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        # Pooling size at each stage.
        self.pooling_size = calc_pooling_at_each_stage(self.pooling_dim, num_stages=self.num_stages)
        max_vals_for_cartesian = 2 * self.pooling_size[:, :2].max(-1).values
        self.strides = torch.ceil(self.pooling_size[-2:, 1] * height).numpy().astype('int32').tolist()
        self.strides = self.strides[-self.num_scales:]

        # Norm layers.
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Activation layers.
        act_layer = nn.GELU

        # Stem.
        self.stem = GraphConvBlock(
            args=args,
            in_chs=self.in_chs + 2,
            out_chs=self.enc_chs[0]
        )

        # Encoder.
        enc_drop_path = [
            x.item() for x in torch.linspace(0, self.drop_path, sum(self.enc_depths))
        ]
        self.enc = DataSequential()
        self.down = DataSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                             sum(self.enc_depths[:s]): sum(self.enc_depths[: s + 1])
                             ]
            enc = DataSequential()
            down = DataSequential()
            down.add(
                GraphPooling(
                    in_chs=self.enc_chs[s] + 2,
                    out_chs=self.enc_chs[s + 1],
                    pooling_size=self.pooling_size[s],
                    width=self.width,
                    height=self.height,
                    batch_size=self.batch_size,
                    transform=T.Cartesian(
                        norm=True, cat=False, max_value=max_vals_for_cartesian[s] if s > 0 else 2 * effective_radius
                    ),
                    pooling_aggr=self.pooling_aggr,
                    traceable=self.traceable,
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="down",
            )
            for i in range(self.enc_depths[s]):
                enc.add(
                    Block(
                        chs=self.enc_chs[s + 1],
                        down_kernel_size=self.down_kernel_size,
                        down_stride=self.down_stride,
                        num_sparse_block=self.num_sparse_block,
                        xy_only=self.xy_only,
                        num_heads=self.enc_heads[s],
                        patch_size=self.enc_patches[s],
                        drop_path=enc_drop_path_[i],
                        attn_type=self.enc_type[s],
                        enable_flash=self.enable_flash,
                        enable_rope=self.enable_rope,
                        d_state=self.d_state,
                        d_conv=self.d_conv,
                        expand=self.expand
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")
            if len(down) != 0:
                self.down.add(module=down, name=f"down{s}")

    def get_output_sizes(self):
        pooling_size = [self.down[-2][0].pooling_size[:2], self.down[-1][0].pooling_size[:2]]
        output_sizes = [(1 / p + 1e-3).cpu().int().numpy().tolist()[::-1] for p in pooling_size]
        return output_sizes

    def forward(self, data: Data, reset=True):
        if hasattr(data, 'reset'):
            reset = data.reset

        data = self.events_to_graph(data, reset=reset)

        data = self.edge_attrs(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        data.x = torch.cat((data.x, data.pos[:, :2]), dim=1)

        # Stem.
        data = self.stem(data)  # 1+2 -> 16
        data_dict = DataDict(data)

        # Encoder.
        output = []
        out_layers = self.num_stages - self.num_scales
        for s in range(self.num_stages):
            data_dict.x = torch.cat((data_dict.x, data_dict.pos[:, :2]), dim=1)
            data_dict = self.down[s](data_dict)
            data_dict = self.enc[s](data_dict)
            if s >= out_layers:
                out = shallow_copy(Data(**data_dict))
                out.pooling = self.down[s][0].pooling_size[:3]
                output.append(out)

        return output[-self.num_scales:]


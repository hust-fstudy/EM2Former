# -*- coding: utf-8 -*-
# @Time: 2025/6/27
# @File: net.py
# @Author: fwb
import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric.transforms as T
from functools import partial
from model.components import DataDict
from model.layers.event2graph import Event2Graph
from model.layers.graph_conv_block import GraphConvBlock
from model.layers.sequential import DataModule, DataSequential
from model.layers.pooling import GraphPooling
from model.layers.unpooling import GraphUnpooling
from model.layers.seg_head import SegHead
from model.layers.attention import Block
from model.networks.net_utils import calc_pooling_at_each_stage, Cartesian


class SegNet(DataModule):
    def __init__(self, args):
        super().__init__()
        # To graph.
        self.width = args.img_w
        self.height = args.img_h
        self.radius = args.radius

        # Encoder.
        self.enable_flash = args.enable_flash
        self.enable_rope = args.enable_rope
        self.in_chs = args.in_chs
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
        self.xy_only = args.xy_only
        self.down_kernel_size = args.down_kernel_size
        self.down_stride = args.down_stride
        self.num_sparse_block = args.num_sparse_block

        # Graph pooling.
        self.traceable = args.traceable
        self.pooling_dim = args.pooling_dim
        self.pooling_aggr = args.pooling_aggr

        # Decoder.
        self.dec_depths = args.dec_depths
        self.dec_chs = args.dec_chs
        self.dec_heads = args.dec_heads
        self.dec_patches = args.dec_patches
        self.dec_type = args.dec_type

        # Postprocessing.
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes

        self.num_stages = len(self.enc_depths)
        assert self.num_stages == len(self.enc_chs)
        assert self.num_stages == len(self.enc_heads)
        assert self.num_stages == len(self.enc_patches)
        assert self.num_stages == len(self.enc_type)
        assert self.num_stages == len(self.dec_depths) + 1
        assert self.num_stages == len(self.dec_chs) + 1
        assert self.num_stages == len(self.dec_heads) + 1
        assert self.num_stages == len(self.dec_patches) + 1
        assert self.num_stages == len(self.dec_type) + 1

        # Graph edges.
        self.events_to_graph = Event2Graph(args)
        effective_radius = 2 * float(int(self.radius * self.width + 2) / self.width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        # Pooling size at each stage.
        self.pooling_size = calc_pooling_at_each_stage(self.pooling_dim, num_stages=self.num_stages)
        max_vals_for_cartesian = 2 * self.pooling_size[:, :2].max(-1).values

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
            if s > 0:
                down.add(
                    GraphPooling(
                        in_chs=self.enc_chs[s - 1] + 2,
                        out_chs=self.enc_chs[s],
                        pooling_size=self.pooling_size[s - 1],
                        width=self.width,
                        height=self.height,
                        batch_size=self.batch_size,
                        transform=T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[s]),
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
                        chs=self.enc_chs[s],
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

        # Decoder.
        dec_drop_path = [
            x.item() for x in torch.linspace(0, self.drop_path, sum(self.dec_depths))
        ]
        self.dec = DataSequential()
        self.up = DataSequential()
        self.dec_chs = list(self.dec_chs) + [self.enc_chs[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                             sum(self.dec_depths[:s]): sum(self.dec_depths[: s + 1])
                             ]
            dec_drop_path_.reverse()
            dec = DataSequential()
            up = DataSequential()
            up.add(
                GraphUnpooling(
                    in_chs=self.dec_chs[s + 1] + 2,
                    skip_chs=self.enc_chs[s] + 2,
                    out_chs=self.dec_chs[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(self.dec_depths[s]):
                dec.add(
                    Block(
                        chs=self.dec_chs[s],
                        down_kernel_size=self.down_kernel_size,
                        down_stride=self.down_stride,
                        num_sparse_block=self.num_sparse_block,
                        xy_only=self.xy_only,
                        num_heads=self.dec_heads[s],
                        patch_size=self.dec_patches[s],
                        attn_type=self.dec_type[s],
                        drop_path=dec_drop_path_[i],
                        enable_flash=self.enable_flash,
                        enable_rope=self.enable_rope,
                        d_state=self.d_state,
                        d_conv=self.d_conv,
                        expand=self.expand
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")
            self.up.add(module=up, name=f"up{s}")

        # Task head.
        self.seg = SegHead(self.dec_chs[0], self.num_classes)

    def forward(self, data: Data, reset=True):
        if hasattr(data, 'reset'):
            reset = data.reset

        # Calc graph edge and attr.
        data = self.events_to_graph(data, reset=reset)
        data = self.edge_attrs(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)

        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)

        # Stem.
        data = self.stem(data)
        data_dict = DataDict(data, grid_size=self.pooling_size[0].clone().detach().to(data.x.device))
        data_dict.sparsify()

        # Encoder.
        for s in range(self.num_stages):
            if s > 0:
                data_dict.x = torch.cat((data_dict.x, data_dict.pos[:, :2]), dim=1)
                data_dict = self.down[s - 1](data_dict)
            data_dict = self.enc[s](data_dict)

        # Decoder.
        for s in range(self.num_stages - 1):
            data_dict.x = torch.cat((data_dict.x, data_dict.pos[:, :2]), dim=1)
            data_dict = self.up[s](data_dict)
            data_dict = self.dec[s](data_dict)

        # Task head.
        out = self.seg(data_dict.x)

        return out

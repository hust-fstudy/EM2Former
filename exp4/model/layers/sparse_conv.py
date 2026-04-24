# -*- coding: utf-8 -*-
# @Time: 2026/3/18
# @File: sparse_conv.py
# @Author: fwb
import spconv
from functools import partial

if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)


def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SparseBasicBlock2D(spconv.SparseModule):

    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d, bias=True):
        super(SparseBasicBlock2D, self).__init__()

        self.conv1 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv2d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


class SparseBasicBlock3D(spconv.SparseModule):

    def __init__(self, dim, indice_key, norm_fn=norm_fn_1d, bias=True):
        super(SparseBasicBlock3D, self).__init__()

        self.conv1 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn1 = norm_fn(dim)

        self.conv2 = spconv.SubMConv3d(dim, dim, 3, 1, 1, bias=bias, indice_key=indice_key)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + x.features))
        return out


def post_act_block_sparse_2d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d,
                             conv_type='subm', indice_key=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


def post_act_block_sparse_3d(input_dim, output_dim, kernel_size, stride=1, padding=0, norm_fn=norm_fn_1d,
                             conv_type='subm', indice_key=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(input_dim, output_dim, kernel_size, bias=False, indice_key=indice_key)

    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)

    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(input_dim, output_dim, kernel_size, indice_key=indice_key, bias=False)

    else:
        raise NotImplementedError

    return spconv.SparseSequential(conv, norm_fn(output_dim), nn.ReLU())


class SparseDilatedLayer(spconv.SparseModule):

    def __init__(
            self,
            dim: int,
            down_kernel_size: list,
            down_stride: list,
            num_sparse_block: list,
            indice_key,
            xy_only=False
    ):
        super().__init__()

        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_0") for _ in range(num_sparse_block[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [
                post_act_block(
                    dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),

                *[block(dim, indice_key=f"{indice_key}_{idx}") for _ in range(num_sparse_block[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    dim, dim, down_kernel_size[idx],
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x

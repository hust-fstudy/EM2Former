# -*- coding: utf-8 -*-
# @Time: 2025/7/12
# @File: sequential.py
# @Author: fwb
import sys
import torch.nn as nn
import spconv.pytorch as spconv
from collections import OrderedDict
from model.components import DataDict


class DataModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DataSequential(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, in_x):
        for k, module in self._modules.items():
            # Point module.
            if isinstance(module, DataModule):
                in_x = module(in_x)
            # Spconv module.
            elif spconv.modules.is_spconv_module(module):
                if isinstance(in_x, DataDict):
                    in_x.sparse_conv_feat = module(in_x.sparse_conv_feat)
                    in_x.x = in_x.sparse_conv_feat.features
                else:
                    in_x = module(in_x)
            # PyTorch module.
            else:
                if isinstance(in_x, DataDict):
                    in_x.x = module(in_x.x)
                    if "sparse_conv_feat" in in_x.keys():
                        in_x.sparse_conv_feat = in_x.sparse_conv_feat.replace_feature(
                            in_x.x
                        )
                elif isinstance(in_x, spconv.SparseConvTensor):
                    if in_x.indices.shape[0] != 0:
                        in_x = in_x.replace_feature(module(in_x.features))
                else:
                    in_x = module(in_x)
        return in_x

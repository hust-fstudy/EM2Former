# -*- coding: utf-8 -*-
# @Time: 2025/7/13
# @File: classifier.py
# @Author: fwb
from torch import nn


class SegHead(nn.Module):
    def __init__(self, in_dim, num_classes=1):
        super().__init__()
        self.det_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        x = self.det_head(x).squeeze()
        return x

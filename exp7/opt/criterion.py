# -*- coding: utf-8 -*-
# @Time: 2025/3/29
# @File: criterion.py
# @Author: fwb
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy


def build_criterion(args):
    criterion_name = args.criterion_name.lower()
    match criterion_name:
        case 'smooth':
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        case 'cross':
            criterion = nn.CrossEntropyLoss()
        case 'bce':
            criterion = nn.BCEWithLogitsLoss()
        case _:
            raise ValueError(f"The {criterion_name} criterion does not exist!")
    return criterion

# -*- coding: utf-8 -*-
# @Time: 2025/2/12
# @File: optimizer.py
# @Author: fwb
import numpy as np
import torch.optim as optim


def build_optimizer(args, model):
    """
    Build optimizer.
    """
    parameters = model.parameters()
    opt_name = args.opt_name.lower()
    match opt_name:
        case 'adam':
            optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        case 'adamw':
            lr = args.lr * np.sqrt(args.batch_size) / np.sqrt(args.norm_batch_size)
            optimizer = optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)
        case 'sgd':
            optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay, nesterov=True)
        case _:
            raise ValueError(f"The {opt_name} optimizer does not exist!")
    return optimizer

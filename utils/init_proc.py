# -*- coding: utf-8 -*-
# @Time: 2026/3/12
# @File: init_proc.py
# @Author: fwb
import random
import numpy as np
import torch
import torch_geometric
from model.networks.det_net import DetNet


def init_seed(random_seed):
    print(f"Random seed ID is: {random_seed}")
    torch_geometric.seed.seed_everything(random_seed)
    torch.random.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def init_model(args):
    if args.task in ['det']:
        model = DetNet(args=args)
    else:
        model = None
        print(f"Task {args.task} does not exist!")
    return model

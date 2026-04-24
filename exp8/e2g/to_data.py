# -*- coding: utf-8 -*-
# @Time: 2026/3/15
# @File: to_data.py
# @Author: fwb
import torch
from torch_geometric.data import Data
from e2g.to_graph_dict import to_seg_dict, to_det_dict


def to_data(**kwargs):
    for k, v in kwargs.items():
        if k in ['t', 'idx', 'loc']:
            kwargs[k] = torch.from_numpy(v)
    kwargs['x'] = torch.from_numpy(kwargs['x'])
    kwargs['pos'] = torch.from_numpy(kwargs['pos'])
    kwargs['y'] = torch.from_numpy(kwargs['y'])

    return Data(**kwargs)


def create_graph_data(args, events_dict):
    if args.task in ['seg']:
        graph_data_dict = to_seg_dict(args, events_dict)
    elif args.task in ['det']:
        graph_data_dict = to_det_dict(args, events_dict)
    else:
        raise ValueError(f"Task {args.task} in create graph stage does not exist!")
    graph_data = to_data(**graph_data_dict)
    if args.task in ['det']:
        graph_data['bbox'] = graph_data.pop('y')
        graph_data.t -= (graph_data.t[-1] - args.tw + 1)

    return graph_data

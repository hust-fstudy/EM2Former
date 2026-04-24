# -*- coding: utf-8 -*-
# @Time: 2026/2/18
# @File: to_graph_dict.py
# @Author: fwb
import numpy as np


def to_seg_dict(args, events_dict):
    assert 'img_w' in args.keys()
    assert 'img_h' in args.keys()
    # Graph data info.
    graph_data_dict = {
        'x': np.where(events_dict['evs_norm'][:, 3] == 0, -1,
                      events_dict['evs_norm'][:, 3].astype(int)).reshape(-1, 1).astype(np.float32),  # [p]
        'pos': events_dict['evs_norm'][:, :3].astype(np.float32),  # [n_x, n_y, n_t]
        'y': events_dict['seg_label'].astype(np.int64),
        'idx': events_dict['idx'].astype(np.int64),
        'loc': events_dict['ev_loc'].astype(np.int64),
        'width': args.img_w,
        'height': args.img_h,
        'time_window': args.tw
    }

    return graph_data_dict


def to_det_dict(args, events_dict):
    assert 'img_w' in args.keys()
    assert 'img_h' in args.keys()
    # Graph data info.
    graph_data_dict = {
        'x': events_dict['p'].reshape(-1, 1).astype(np.float32),  # [p]
        'pos': np.hstack((
            events_dict['x'].reshape(-1, 1),
            events_dict['y'].reshape(-1, 1),
        )).astype(np.int64),  # [x, y]
        'y': np.hstack((
            events_dict['bbox'].reshape(-1, 4),
            np.full(len(events_dict['bbox']), events_dict['label']).reshape(-1, 1)
        )).astype(np.float32),  # [x_min, y_min, h, w, class_id]
        't': events_dict['t'].astype(np.float32),  # [t]
        'width': args.img_w,
        'height': args.img_h,
        'time_window': args.tw
    }

    return graph_data_dict

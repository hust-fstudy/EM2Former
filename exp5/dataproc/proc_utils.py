# -*- coding: utf-8 -*-
# @Time: 2025/6/26
# @File: proc_utils.py
# -*- coding: utf-8 -*-
# @Time: 2026/3/18
# @File: proc_utils.py
# @Author: fwb
import os
import os.path as osp
from pathlib import Path
import numpy as np


def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)


def min_max_normalization(arr, mx, mi=0):
    arr = arr.astype(float)
    epsilon = 1e-8
    min_val = np.min(arr, axis=0)
    max_val = np.max(arr, axis=0)
    de = max_val - min_val
    if np.any(de < epsilon):
        if arr.ndim == 1:
            arr[:] = 0.5
        else:
            small_index = np.where(de < epsilon)[0]
            big_index = np.where(de >= epsilon)[0]
            arr[:, small_index] = 0.5
            arr[:, big_index] = (mx - mi) * ((arr[:, big_index] - min_val[big_index]) /
                                             (max_val[big_index] - min_val[big_index])) + mi
        normalize_data = arr
    else:
        normalize_data = (mx - mi) * ((arr - min_val) / (max_val - min_val)) + mi
    return normalize_data


def get_label_dict(is_split, data_dir: Path, dataset_name: str):
    if is_split:
        train_dir = data_dir.joinpath('total')
    else:
        train_dir = data_dir.joinpath('train')
    if dataset_name in ['pedro']:
        label_dict = {'person': 0}
    elif dataset_name in ['rseod']:
        label_dict = {
            'car': 0,
            'two-wheel': 1,
            'pedestrian': 2,
            'bus': 3,
            'truck': 4,
        }
    else:
        class_list = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        label_dict = {class_name: label for label, class_name in enumerate(class_list)}
    return label_dict

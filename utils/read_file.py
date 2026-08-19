# -*- coding: utf-8 -*-
# @Time: 2024/11/22
# @File: read_file.py
# @Author: fwb
import hdf5plugin
import h5py
import random
from pathlib import Path
import numpy as np
import json

"""
In all events, the unit for t is us, and p is {+1, -1}.
For all BBoxes, the coordinates denote the top-left corner (x_min, y_min, w, h),
correspond to the absolute pixel position in the image.
"""


def read_way(total_len, num_nodes, is_rand):
    if is_rand:
        diff_num = total_len - num_nodes
        if diff_num >= 0:
            start_idx = random.randint(0, diff_num)
            end_idx = start_idx + num_nodes
        else:
            start_idx, end_idx = None, None
    else:
        start_idx, end_idx = -num_nodes, None
    return start_idx, end_idx


class ReadFile:
    def cal101det_data_reader(self, filepath, num_nodes, is_rand):
        with h5py.File(filepath) as f:
            events = f['events']
            start_idx, end_idx = read_way(len(events['x']), num_nodes, is_rand)
            x = np.array(events['x'][start_idx:end_idx])
            y = np.array(events['y'][start_idx:end_idx])
            t = np.array(events['t'][start_idx:end_idx])  # us
            p = np.array(events['p'][start_idx:end_idx])  # -1, 1
        # Load annotations.
        mode_dir = Path(filepath).parent.parent
        rel_path = str(Path(filepath).relative_to(mode_dir))
        rel_path = rel_path.replace('image_', 'annotation_').replace('.h5', '.bin')
        annotation_file = mode_dir.parent / 'annotations' / rel_path
        with annotation_file.open() as f:
            annotations = np.fromfile(f, dtype=np.int16)
            annotations = np.array(annotations[2:10])
        bbox = np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
        ]).astype('float32').reshape(-1, 4)
        return [x, y, t, p, bbox]

    def pedro_data_reader(self, filepath, num_nodes, is_rand, width, height):
        events = np.load(filepath)
        start_idx, end_idx = read_way(len(events), num_nodes, is_rand)
        events = events[start_idx:end_idx]
        t, x, y, p = events.T
        t -= min(t)  # 0~duration, us
        p = np.where(p == 0, -1, p.astype(int))  # -1, 1
        # Load annotations.
        mode_dir = Path(filepath).parent.parent
        rel_path = str(Path(filepath).relative_to(mode_dir))
        rel_path = rel_path.replace('.npy', '.txt')
        annotation_file = mode_dir.joinpath('annotations', 'yolo', rel_path)
        annotation_data = np.loadtxt(annotation_file, dtype=float)
        if annotation_data.ndim == 1:
            annotation_data = annotation_data[np.newaxis, :]
        sample_label = annotation_data[:, 0].astype(int)
        bbox = annotation_data[:, 1:].reshape(-1, 4)
        # (center_x, center_y, w, h) --> (x_min, y_min, w, h)
        bbox[:, :4] *= np.array([width, height, width, height])
        bbox[:, :2] -= 0.5 * bbox[:, 2:4]
        bbox = bbox.astype('float32')
        return [x, y, t, p, bbox, sample_label]

    def rseod_data_reader(self, filepath, num_nodes, is_rand, label_dict, width, height):
        events = np.load(filepath)
        start_idx, end_idx = read_way(len(events), num_nodes, is_rand)
        events = events[start_idx:end_idx]
        x, y, p, t = events.T
        t -= min(t)  # 0~duration, us
        p = np.where(p == 0, -1, p.astype(int))  # -1, 1
        # Load annotations.
        mode_dir = Path(filepath).parent.parent.parent.parent
        rel_path = str(Path(filepath).relative_to(mode_dir))
        rel_path = rel_path.replace('.npy', '.json')
        annotation_file = mode_dir.joinpath('annotations', rel_path)
        with open(annotation_file, 'r') as json_file:
            data = json.load(json_file)
            objects = data['shapes']
            sample_label = []
            bbox = []
            for i in range(len(objects)):
                bbox_points = objects[i]['points']
                if 'label' in objects[i]:
                    bbox_class = objects[i]['label']
                else:
                    bbox_class = objects[i]['lable']
                bbox_label = int(label_dict[bbox_class])
                single_bbox = [int(bbox_points[0][0]), int(bbox_points[0][1]),
                               int(bbox_points[2][0]), int(bbox_points[2][1])]
                sample_label.append(bbox_label)
                bbox.append(single_bbox)
        sample_label = np.array(sample_label).astype(int)
        bbox = np.array(bbox).astype('float32')
        bbox[:, 0::2].clip(min=0, max=width)
        bbox[:, 1::2].clip(min=0, max=height)
        keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
        bbox = bbox[keep]
        sample_label = sample_label[keep]
        # (x_min, y_min, x_max, y_max) --> (x_min, y_min, w, h)
        bbox[:, 2:4] -= bbox[:, 0:2]
        return [x, y, t, p, bbox, sample_label]

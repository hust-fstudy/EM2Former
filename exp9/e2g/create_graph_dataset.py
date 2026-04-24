# -*- coding: utf-8 -*-
# @Time: 2025/3/4
# @File: create_graph_dataset.py
# @Author: fwb
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset
from utils.read_file import ReadFile
from utils.uniform_event import Event
from dataproc.augment import init_transforms
from dataproc.proc_utils import get_label_dict
from e2g.to_data import create_graph_data


class CreateGraphDataset(Dataset):
    def __init__(self, args, data_path: Path, mode: str, transform=None):
        super(CreateGraphDataset, self).__init__()
        self.args = args
        self.dataset_name = Path(data_path).parent.name
        # Obtain dataset info.
        self.mode_dir = data_path.joinpath(mode)
        self.sample_path_list = sorted(p for p in self.mode_dir.rglob('*') if p.is_file())
        self.label_dict = get_label_dict(is_split=args.is_split, data_dir=data_path, dataset_name=self.dataset_name)
        # Data reader.
        self.RF = ReadFile()
        # Data augments.
        if args.task in ['det']:
            if transform is not None and hasattr(transform, 'transforms'):
                init_transforms(transform.transforms, args.img_h, args.img_w)
        self.transform = transform

    def __len__(self):
        return len(self.sample_path_list)

    def __getitem__(self, index):
        sample_path = self.sample_path_list[index]
        x, y, t, p = [], [], [], []
        evs_norm, ev_loc, seg_label, idx = [], [], [], []
        bbox = []
        sample_label = None
        # Data read.
        if self.dataset_name in ['ev-uav']:
            [evs_norm, ev_loc, seg_label, idx] = self.RF.uav_data_reader(sample_path)
        elif self.dataset_name in ['ncaltech101det']:
            [x, y, t, p, bbox] = self.RF.cal101det_data_reader(
                filepath=sample_path,
                num_nodes=int(self.args.num_nodes),
                is_rand=self.args.is_rand
            )
        elif self.dataset_name in ['pedro']:
            [x, y, t, p, bbox, sample_label] = self.RF.pedro_data_reader(
                filepath=sample_path,
                num_nodes=int(self.args.num_nodes),
                is_rand=self.args.is_rand,
                width=self.args.img_w,
                height=self.args.img_h
            )
        elif self.dataset_name in ['rseod']:
            [x, y, t, p, bbox, sample_label] = self.RF.rseod_data_reader(
                filepath=sample_path,
                num_nodes=int(self.args.num_nodes),
                is_rand=self.args.is_rand,
                label_dict=self.label_dict,
                width=self.args.img_w,
                height=self.args.img_h
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        # Convert graph data.
        if self.args.task in ['seg']:
            events_dict = {
                'evs_norm': evs_norm,
                'ev_loc': ev_loc,
                'seg_label': seg_label,
                'idx': idx
            }
        elif self.args.task in ['det']:
            sample_label = np.array(self.label_dict[str(sample_path.parent.name)]) \
                if sample_label is None else sample_label
            events_dict = Event(x, y, t, p, sample_label).to_uniform_format()
            events_dict['bbox'] = bbox
        else:
            raise ValueError(f"Task {self.args.task} in dataset read stage does not exist!")
        graph_data = create_graph_data(self.args, events_dict)
        # Data augments.
        graph_data = self.transform(graph_data) if self.transform is not None else graph_data

        return graph_data

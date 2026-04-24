# -*- coding: utf-8 -*-
# @Time: 2026/3/18
# @File: dataset_loader.py
# @Author: fwb
from torch_geometric.loader import DataLoader
from dataproc.augment import Augmentations
from e2g.create_graph_dataset import CreateGraphDataset


def dataset_loader(args, data_path):
    # Data augments.
    aug = Augmentations(args)
    # Load graph dataset.
    train_graph_dataset = CreateGraphDataset(
        args=args,
        data_path=data_path,
        mode='train',
        transform=aug.transform_train
    )
    val_graph_dataset = CreateGraphDataset(
        args=args,
        data_path=data_path,
        mode='val',
        transform=aug.transform_test
    )
    test_graph_dataset = CreateGraphDataset(
        args=args,
        data_path=data_path,
        mode='test',
        transform=aug.transform_test
    )
    print(
        f"train len: {len(train_graph_dataset)}, "
        f"val len: {len(val_graph_dataset)}, "
        f"test len: {len(test_graph_dataset)}"
    )
    if args.task in ['det']:
        follow_batch = ['bbox']
        drop_last = True
    else:
        follow_batch = None
        drop_last = False
    train_loader = DataLoader(
        train_graph_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        follow_batch=follow_batch,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_graph_dataset,
        batch_size=args.batch_size,
        follow_batch=follow_batch,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        test_graph_dataset,
        batch_size=args.batch_size,
        follow_batch=follow_batch,
        drop_last=drop_last
    )

    return train_loader, val_loader, test_loader

# -*- coding: utf-8 -*-
# @Time: 2024/12/10
# @File: model_performance.py
# @Author: fwb
import torch
import numpy as np
from tqdm import tqdm
from metrics.seg_metrics import SegMetrics
from model.batch_tools import offset2batch
from utils.init_proc import init_model


def model_train(args, model, train_loader, criterion, optimizer, lr_scheduler, epoch, device):
    # Define variables.
    train_loss = []
    num_steps = len(train_loader)
    # Start training.
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(tqdm(train_loader, desc=f"train", position=0)):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Calculate loss.
        loss = criterion(output, data.y.to(dtype=torch.float32))
        # Backward propagation.
        loss.backward()
        optimizer.step()
        if args.is_lr_scheduler:
            if args.lr_scheduler.lower() in ['cosine', 'linear', 'step', 'multistep']:
                lr_scheduler.step_update(epoch * num_steps + i)
            elif args.lr_scheduler.lower() in ['cycle', 'lambda']:
                lr_scheduler.step()
        train_loss.append(loss.item())
    train_loss = round(np.mean(train_loss), 8)

    return train_loss


def model_val(args, model, val_loader, device):
    # Define variables.
    cal_metrics = SegMetrics(args)
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc=f"val", position=0)):
            data = data.to(device)
            output = model(data)
            # Statistics.
            cal_metrics.matches[str(i)] = {}
            cal_metrics.matches[str(i)]['seg_pred'] = output
            cal_metrics.matches[str(i)]['seg_gt'] = data.y
        iou = cal_metrics.evaluate_semantic_segmentation_iou()
        iou = round(iou.item(), 4)

        return iou


def model_test(args, model_path, test_loader, device):
    model = init_model(args)
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # Define variables.
    cal_metrics = SegMetrics(args)
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc=f"final test", position=0)):
            data = data.to(device)
            output = model(data)
            # Statistics.
            cal_metrics.matches[str(i)] = {}
            cal_metrics.matches[str(i)]['seg_pred'] = output
            cal_metrics.matches[str(i)]['seg_gt'] = data.y
            if args.roc:
                data.batch = offset2batch(data.ptr) - 1
                ev_locs = torch.cat([data.batch.reshape(-1, 1), data.loc], dim=1)
                cal_metrics.roc_update(ev_locs[:, 3].cpu(), output.cpu(), data.idx.cpu(), data.y.cpu(), ev_locs.cpu())

        iou = cal_metrics.evaluate_semantic_segmentation_iou()
        seg_acc = cal_metrics.evaluate_semantic_segmentation_accuracy()
        if args.roc:
            pd, fa = cal_metrics.cal_roc()
        iou = round(iou.item(), 4)
        seg_acc = round(seg_acc.item(), 4)
        pd = round(pd, 4)
        fa = round(fa * 1e4, 4)
        stats = {
            'iou': iou,
            'seg_acc': seg_acc,
            'pd': pd,
            'fa': fa
        }

        return stats


def model_test_epoch(args, model, test_loader, device):
    # Define variables.
    cal_metrics = SegMetrics(args)
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc=f"test", position=0)):
            data = data.to(device)
            output = model(data)
            # Statistics.
            cal_metrics.matches[str(i)] = {}
            cal_metrics.matches[str(i)]['seg_pred'] = output
            cal_metrics.matches[str(i)]['seg_gt'] = data.y
            if args.roc:
                data.batch = offset2batch(data.ptr) - 1
                ev_locs = torch.cat([data.batch.reshape(-1, 1), data.loc], dim=1)
                cal_metrics.roc_update(ev_locs[:, 3].cpu(), output.cpu(), data.idx.cpu(), data.y.cpu(), ev_locs.cpu())

        iou = cal_metrics.evaluate_semantic_segmentation_iou()
        seg_acc = cal_metrics.evaluate_semantic_segmentation_accuracy()
        if args.roc:
            pd, fa = cal_metrics.cal_roc()
        iou = round(iou.item(), 4)
        seg_acc = round(seg_acc.item(), 4)
        pd = round(pd, 4)
        fa = round(fa * 1e4, 4)
        stats = {
            'iou': iou,
            'seg_acc': seg_acc,
            'pd': pd,
            'fa': fa
        }

        return stats

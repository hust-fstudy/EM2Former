# -*- coding: utf-8 -*-
# @Time: 2024/12/10
# @File: det_one_epoch.py
# @Author: fwb
import torch
import numpy as np
from tqdm import tqdm
from utils.box_ops import format_input
from utils.init_proc import init_model
from metrics.det_buffer import DetBuffer


def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


def model_train(args, model, ema, train_loader, optimizer, lr_scheduler, epoch, device):
    # Define variables.
    train_loss = []
    num_steps = len(train_loader)
    # Start training.
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(tqdm(train_loader, desc=f"train", position=0)):
        data = data.to(device)
        data = format_input(data)
        optimizer.zero_grad()
        output = model(data)
        # Calculate loss.
        loss_dict = {k: v for k, v in output.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")
        # Backward propagation.
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
        fix_gradients(model)
        optimizer.step()
        if args.is_lr_scheduler:
            if args.lr_scheduler.lower() in ['cosine', 'linear', 'step', 'multistep']:
                lr_scheduler.step_update(epoch * num_steps + i)
            elif args.lr_scheduler.lower() in ['cycle', 'lambda']:
                lr_scheduler.step()
        ema.update(model)
        train_loss.append(loss.item())
    train_loss = round(np.mean(train_loss), 8)

    return train_loss


def model_val(args, model, val_loader, device):
    # Define variables.
    calc = DetBuffer(height=args.img_h, width=args.img_w, classes=list(val_loader.dataset.label_dict.keys()))
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc=f"val", position=0)):
            data = data.to(device)
            data = format_input(data)
            detections, targets = model(data)
            calc.update(detections, targets)
        metrics = calc.compute()

        return metrics


def model_test(args, model_path, test_loader, device):
    model = init_model(args)
    model.to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # Define variables.
    calc = DetBuffer(height=args.img_h, width=args.img_w, classes=list(test_loader.dataset.label_dict.keys()))
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc=f"final test", position=0)):
            data = data.to(device)
            data = format_input(data)
            detections, targets = model(data)
            calc.update(detections, targets)
        metrics = calc.compute()

        return metrics


def model_test_epoch(args, model, test_loader, device):
    # Define variables.
    calc = DetBuffer(height=args.img_h, width=args.img_w, classes=list(test_loader.dataset.label_dict.keys()))
    # Start testing.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc=f"test", position=0)):
            data = data.to(device)
            data = format_input(data)
            detections, targets = model(data)
            calc.update(detections, targets)
        metrics = calc.compute()

        return metrics

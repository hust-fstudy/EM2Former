# -*- coding: utf-8 -*-
# @Time: 2025/2/12
# @File: lr_scheduler.py
# @Author: fwb
import torch
from functools import partial
import math
from typing import List
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
import torch.optim.lr_scheduler as lr_sche


class LRSchedule:
    def __init__(self,
                 warmup_epochs: float,
                 num_iters_per_epoch: int,
                 tot_num_epochs: int,
                 min_lr_ratio: float = 0.05,
                 warmup_lr_start: float = 0,
                 steps_at_iteration=[50000],
                 reduction_at_step=0.5):
        warmup_total_iters = num_iters_per_epoch * warmup_epochs
        total_iters = tot_num_epochs * num_iters_per_epoch
        no_aug_iters = 0
        self.lr_func = partial(_yolox_warm_cos_lr, min_lr_ratio, total_iters, warmup_total_iters, warmup_lr_start,
                               no_aug_iters, steps_at_iteration, reduction_at_step)

    def __call__(self, *args, **kwargs) -> float:
        return self.lr_func(*args, **kwargs)


def _yolox_warm_cos_lr(
        min_lr_ratio: float,
        total_iters: int,
        warmup_total_iters: int,
        warmup_lr_start: float,
        no_aug_iter: int,
        steps_at_iteration: List[int],
        reduction_at_step: float,
        iters: int) -> float:
    """Cosine learning rate with warm up."""
    min_lr = min_lr_ratio
    if iters < warmup_total_iters:
        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
        lr = (1 - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    else:
        lr = min_lr + 0.5 * (1 - min_lr) * (1.0 + math.cos(
            math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))

    for step in steps_at_iteration:
        if iters >= step:
            lr *= reduction_at_step

    return lr


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def build_scheduler(args, optimizer, n_iter_per_epoch):
    decay_rate = args.decay_rate
    min_lr = args.min_lr
    warmup_lr_init = args.warmup_lr_init
    epochs = args.epochs
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
    decay_steps = int(args.decay_epochs * n_iter_per_epoch)
    scheduler_name = args.lr_scheduler.lower()
    match scheduler_name:
        case 'cosine':
            lr_scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=num_steps,
                lr_min=min_lr,
                cycle_limit=1,
                warmup_t=warmup_steps,
                warmup_lr_init=warmup_lr_init,
                t_in_epochs=False
            )
        case 'linear':
            lr_scheduler = LinearLRScheduler(
                optimizer=optimizer,
                t_initial=num_steps,
                lr_min_rate=0.01,
                warmup_t=warmup_steps,
                warmup_lr_init=warmup_lr_init,
                t_in_epochs=False
            )
        case 'step':
            lr_scheduler = StepLRScheduler(
                optimizer=optimizer,
                decay_t=decay_steps,
                decay_rate=decay_rate,
                warmup_t=warmup_steps,
                warmup_lr_init=warmup_lr_init,
                t_in_epochs=False
            )
        case 'cycle':
            lr_scheduler = lr_sche.OneCycleLR(
                optimizer=optimizer,
                max_lr=args.lr,
                total_steps=num_steps
            )
        case 'lambda':
            lr_scheduler = lr_sche.LambdaLR(
                optimizer=optimizer,
                lr_lambda=LRSchedule(
                    warmup_epochs=.3,
                    num_iters_per_epoch=n_iter_per_epoch,
                    tot_num_epochs=epochs
                )
            )
        case _:
            raise ValueError(f"The {scheduler_name} lr_scheduler does not exist!")
    return lr_scheduler

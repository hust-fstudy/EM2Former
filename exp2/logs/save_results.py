# -*- coding: utf-8 -*-
# @Time: 2024/12/12
# @File: save_results.py
# @Author: fwb
import os
import os.path as osp
from pathlib import Path
import torch
import scipy.io as sio
from dataproc.proc_utils import create_path


class SaveResults:
    def __init__(self, args=None, dataset_name=None, seed=None):
        self.args = args
        # Create results save path.
        self.results_path = osp.join(Path.cwd().parent, 'output', Path.cwd().name, dataset_name)
        self.config_path = osp.join(self.results_path, 'config.txt')
        self.config_info = '-'.join([f"{k}[{v}]" for k, v in args.items()])
        self.config_info = f"seed[{seed}]-" + self.config_info
        create_path(self.results_path)

    def save_model(self, model_weight, name):
        # Save model.
        model_save_path = osp.join(self.results_path, name + '.pth')
        torch.save(model_weight, model_save_path)
        return model_save_path

    def save_results(self, results_dict):
        # Save config info.
        with open(self.config_path, 'w', encoding='utf-8') as file:
            file.write(self.config_info + '\n')
        # Save training and test results to results.mat file.
        sio.savemat(osp.join(self.results_path, 'results.mat'), results_dict)

    def create_results_dict(self):
        if self.args.task in ['seg']:
            keys = ['train_epoch_loss', 'val_epoch_iou',
                    'test_epoch_iou', 'test_epoch_acc', 'test_epoch_pd', 'test_epoch_fa',
                    'test_best_loss', 'test_best_iou', 'test_best_epoch']
        elif self.args.task in ['det']:
            keys = ['train_epoch_loss', 'val_epoch_AP',
                    'test_epoch_AP', 'test_epoch_AP50', 'test_epoch_AP75',
                    'test_best_loss', 'test_best_AP', 'test_best_epoch']
        else:
            raise ValueError(f"Task {self.args.task} in save results stage does not exist!")
        results_dict = dict((key, []) for key in keys)
        return results_dict

# -*- coding: utf-8 -*-
# @Time: 2025/6/25
# @File: run_recognition.py
# @Author: fwb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['WANDB_API_KEY'] = 'de6f3a4bf172a1e8e28f43cb88a65f379586a8c4'
import wandb
from pathlib import Path
from tqdm import tqdm
import torch
from dataproc.dataset_loader import dataset_loader
from opt.criterion import build_criterion
from opt.optimizer import build_optimizer
from opt.lr_scheduler import build_scheduler
from utils.parse_yaml import parse_yaml
from logs.save_results import SaveResults
from utils.init_proc import init_seed, init_model
import phase.seg_one_epoch as seg_phase
import phase.det_one_epoch as det_phase
from model.networks.ema import ModelEMA
from tests.epoch_test import seg_test_epoch, det_test_epoch
from tests.final_test import seg_test, det_test
from logs.print_logs import print_epoch_logs

EVAL_DATASET_DICT = {
    1: 'EV-UAV',
    2: 'NCaltech101det',
    3: 'PEDRo',
    4: 'RSEOD'
}
SEL_DATASET_IDX = 3


class TaskMain:
    def __init__(self, cfgs_dir: Path, data_root_dir: Path, dataset_name: str, random_seed: int):
        self.random_seed = random_seed
        # Parse dataset config info.
        self.dataset_name = dataset_name.lower()
        dataset_cfgs_path = cfgs_dir.joinpath(self.dataset_name + '.yaml')
        self.args = parse_yaml(dataset_cfgs_path)
        # Load the corresponding dataset.
        print(f"The current dataset used is {self.dataset_name}")
        self.data_path = data_root_dir.joinpath(self.dataset_name, 'raw')
        # Start a new wandb run to track this script.
        config_info = {
            "architecture": "EM2Former",
            "dataset": dataset_name
        }
        for k, v in self.args.items():
            config_info[k] = v
        wandb.login()
        self.exp_log = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hust_fwb",
            # Set the wandb project where this run will be logged.
            project="EM2Former",
            # Track hyperparameters and run metadata.
            config=config_info
        )

    def run(self):
        train_loader, val_loader, test_loader = dataset_loader(args=self.args, data_path=self.data_path)
        # Initialization Model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = init_model(self.args)
        model.to(device)
        if self.args.task in ['seg']:
            criterion = build_criterion(self.args)
            ema = None
        elif self.args.task in ['det']:
            criterion = None
            ema = ModelEMA(model)
        else:
            raise ValueError(f"Task {self.args.task} in init stage does not exist!")
        optimizer = build_optimizer(self.args, model)
        lr_scheduler = None
        if self.args.is_lr_scheduler:
            lr_scheduler = build_scheduler(self.args, optimizer, len(train_loader))
        # Model train-val-test.
        SR = SaveResults(self.args, self.dataset_name, self.random_seed)
        results_dict = SR.create_results_dict()
        best_loss_epoch = 0
        best_val_epoch = 0
        best_test_epoch = 0
        best_train_loss = 1.0e+6
        best_val_metrics = 0
        best_test_metrics = 0
        best_test_save_path = None
        for epoch in range(self.args.epochs):
            with tqdm(desc=f"epoch {epoch + 1}/{self.args.epochs}", position=0):
                if self.args.task in ['seg']:
                    train_temp_loss = seg_phase.model_train(
                        args=self.args,
                        model=model,
                        train_loader=train_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        device=device
                    )
                    results_dict['train_epoch_loss'].append(train_temp_loss)
                    print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss})")
                    if (epoch + 1) % self.args.val_step == 0 or (epoch + 1) >= self.args.min_val_epoch:
                        val_temp_iou = seg_phase.model_val(
                            args=self.args,
                            model=model,
                            val_loader=val_loader,
                            device=device
                        )
                        results_dict['val_epoch_iou'].append(val_temp_iou)
                        best_val_iou = max(results_dict['val_epoch_iou'])
                        print(f"best val iou: (best_iou: {best_val_iou})")
                        # Test at each epoch.
                        if self.args.is_test_epoch:
                            seg_test_epoch(
                                args=self.args,
                                seg_phase=seg_phase,
                                model=model,
                                test_loader=test_loader,
                                device=device,
                                epoch=epoch,
                                results_dict=results_dict
                            )
                            test_temp_iou = results_dict['test_epoch_iou'][-1]
                            if test_temp_iou >= best_test_metrics:
                                best_test_save_path = SR.save_model(model.state_dict(), 'best_test_model')
                                best_test_epoch = epoch + 1
                                best_test_metrics = test_temp_iou
                        # Logs.
                        if train_temp_loss <= best_train_loss:
                            best_loss_save_path = SR.save_model(model.state_dict(), 'best_loss_model')
                            best_loss_epoch = epoch + 1
                            best_train_loss = train_temp_loss
                        if val_temp_iou >= best_val_metrics:
                            best_val_save_path = SR.save_model(model.state_dict(), 'best_val_model')
                            best_val_epoch = epoch + 1
                            best_val_metrics = val_temp_iou
                elif self.args.task in ['det']:
                    train_temp_loss = det_phase.model_train(
                        args=self.args,
                        model=model,
                        ema=ema,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        epoch=epoch,
                        device=device
                    )
                    results_dict['train_epoch_loss'].append(train_temp_loss)
                    print(f"train info for epoch {epoch + 1}: (loss: {train_temp_loss})")
                    if (epoch + 1) % self.args.val_step == 0 or (epoch + 1) >= self.args.min_val_epoch:
                        metrics = det_phase.model_val(
                            args=self.args,
                            model=model,
                            val_loader=val_loader,
                            device=device
                        )
                        val_temp_AP = metrics['AP']
                        results_dict['val_epoch_AP'].append(val_temp_AP)
                        best_val_AP = max(results_dict['val_epoch_AP'])
                        print(f"best val AP: (best_AP: {best_val_AP})")
                        # Test at each epoch.
                        if self.args.is_test_epoch:
                            det_test_epoch(
                                args=self.args,
                                det_phase=det_phase,
                                model=model,
                                test_loader=test_loader if self.dataset_name in ['pedro'] else val_loader,
                                device=device,
                                epoch=epoch,
                                results_dict=results_dict
                            )
                            test_temp_AP = results_dict['test_epoch_AP'][-1]
                            if test_temp_AP >= best_test_metrics:
                                best_test_save_path = SR.save_model(model.state_dict(), 'best_test_model')
                                best_test_epoch = epoch + 1
                                best_test_metrics = test_temp_AP
                        elif self.dataset_name in ['ncaltech101det', 'pedro', 'rseod']:
                            results_dict['test_epoch_AP'].append(metrics['AP'])
                            results_dict['test_epoch_AP50'].append(metrics['AP_50'])
                            results_dict['test_epoch_AP75'].append(metrics['AP_75'])
                            print_epoch_logs(
                                results_dict=results_dict,
                                test_epoch_results=metrics,
                                epoch=epoch
                            )
                        # Logs.
                        if train_temp_loss <= best_train_loss:
                            best_loss_save_path = SR.save_model(model.state_dict(), 'best_loss_model')
                            best_loss_epoch = epoch + 1
                            best_train_loss = train_temp_loss
                        if val_temp_AP >= best_val_metrics:
                            best_val_save_path = SR.save_model(model.state_dict(), 'best_val_model')
                            best_val_epoch = epoch + 1
                            best_val_metrics = val_temp_AP
                else:
                    raise ValueError(f"Task {self.args.task} in train stage does not exist!")
                    # Save logs.
                SR.save_results(results_dict)
                torch.cuda.empty_cache()
                # Wandb logs.
                self.exp_log.log({"train_epoch_loss": train_temp_loss})
        """
        Test Phase!!!
        The corresponding dataset for the test.
        """
        # Test phase.
        if self.dataset_name in ['ev-uav']:
            seg_test(
                args=self.args,
                seg_phase=seg_phase,
                best_loss_epoch=best_loss_epoch,
                best_val_epoch=best_val_epoch,
                best_test_epoch=best_test_epoch,
                best_loss_save_path=best_loss_save_path,
                best_val_save_path=best_val_save_path,
                best_test_save_path=best_test_save_path,
                test_loader=test_loader,
                device=device,
                results_dict=results_dict
            )
        if self.dataset_name in ['ncaltech101det', 'pedro', 'rseod']:
            det_test(
                args=self.args,
                det_phase=det_phase,
                best_loss_epoch=best_loss_epoch,
                best_val_epoch=best_val_epoch,
                best_test_epoch=best_test_epoch,
                best_loss_save_path=best_loss_save_path,
                best_val_save_path=best_val_save_path,
                best_test_save_path=best_test_save_path,
                test_loader=test_loader if self.dataset_name in ['xxx'] else val_loader,
                device=device,
                results_dict=results_dict
            )
        # Save final results.
        SR.save_results(results_dict)
        self.exp_log.finish()


if __name__ == '__main__':
    # Load common config info.
    cfgs_dir = Path(r'./configs')
    com_cfgs_path = cfgs_dir.joinpath('com_params' + '.yaml')
    com_cfgs = parse_yaml(com_cfgs_path)
    # Set the random seed and initialize it.
    random_seed = com_cfgs.random_seed
    init_seed(random_seed)
    # Evaluate on the selected dataset.
    assert SEL_DATASET_IDX in EVAL_DATASET_DICT.keys()
    RM = TaskMain(
        cfgs_dir=cfgs_dir,
        data_root_dir=Path(com_cfgs.data_root_dir),
        dataset_name=EVAL_DATASET_DICT[SEL_DATASET_IDX],
        random_seed=random_seed
    )
    RM.run()

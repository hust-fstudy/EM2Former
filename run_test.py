# -*- coding: utf-8 -*-
# @Time: 2026/5/21
# @File: run_test.py
# @Author: fwb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path as osp
from pathlib import Path
import torch
from dataproc.dataset_loader import dataset_loader
from utils.parse_yaml import parse_yaml
from logs.save_results import SaveResults
from utils.init_proc import init_seed, init_model
import phase.det_one_epoch as det_phase
from tests.final_test import det_test

EVAL_DATASET_DICT = {
    1: 'NCaltech101det',
    2: 'PEDRo',
    3: 'RSEOD'
}
SEL_DATASET_IDX = 1


class TestTask:
    def __init__(self, cfgs_dir: Path, data_root_dir: Path, dataset_name: str, random_seed: int):
        self.random_seed = random_seed
        # Parse dataset config info.
        self.dataset_name = dataset_name.lower()
        dataset_cfgs_path = cfgs_dir.joinpath(self.dataset_name + '.yaml')
        self.args = parse_yaml(dataset_cfgs_path)
        # Load the corresponding dataset.
        print(f"The current dataset used is {self.dataset_name}")
        self.data_path = data_root_dir.joinpath(self.dataset_name, 'raw')
        # Model weight path.
        self.results_path = osp.join(os.getcwd(), 'results', self.dataset_name)
        self.best_loss_save_path = osp.join(self.results_path, 'best_loss_model.pth')
        self.best_val_save_path = osp.join(self.results_path, 'best_val_model.pth')

    def run(self):
        train_loader, val_loader, test_loader = dataset_loader(args=self.args, data_path=self.data_path)
        # Initialization Model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = init_model(self.args)
        model.to(device)
        # Model train-val-test.
        SR = SaveResults(self.args, self.dataset_name, self.random_seed)
        results_dict = SR.create_results_dict()
        best_loss_epoch = 0
        best_val_epoch = 0
        best_test_epoch = 0
        best_test_save_path = None
        """
        Test Phase!!!
        The corresponding dataset for the test.
        """
        # Test phase.
        if self.dataset_name in ['ncaltech101det', 'pedro', 'rseod']:
            det_test(
                args=self.args,
                det_phase=det_phase,
                best_loss_epoch=best_loss_epoch,
                best_val_epoch=best_val_epoch,
                best_test_epoch=best_test_epoch,
                best_loss_save_path=self.best_loss_save_path,
                best_val_save_path=self.best_val_save_path,
                best_test_save_path=best_test_save_path,
                test_loader=test_loader if self.dataset_name in ['xxx'] else val_loader,
                device=device,
                results_dict=results_dict
            )
        # Save final results.
        SR.save_results(results_dict)


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
    RM = TestTask(
        cfgs_dir=cfgs_dir,
        data_root_dir=Path(com_cfgs.data_root_dir),
        dataset_name=EVAL_DATASET_DICT[SEL_DATASET_IDX],
        random_seed=random_seed
    )
    RM.run()

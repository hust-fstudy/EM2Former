# -*- coding: utf-8 -*-
# @Time: 2026/3/26
# @File: epoch_test.py
# @Author: fwb
from logs.print_logs import print_epoch_logs


def det_test_epoch(
        args,
        det_phase,
        model,
        test_loader,
        device,
        epoch,
        results_dict
):
    metrics = det_phase.model_test_epoch(
        args=args,
        model=model,
        test_loader=test_loader,
        device=device
    )
    results_dict['test_epoch_AP'].append(metrics['AP'])
    results_dict['test_epoch_AP50'].append(metrics['AP_50'])
    results_dict['test_epoch_AP75'].append(metrics['AP_75'])
    print_epoch_logs(
        results_dict=results_dict,
        test_epoch_results=metrics,
        epoch=epoch
    )

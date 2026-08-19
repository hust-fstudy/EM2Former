# -*- coding: utf-8 -*-
# @Time: 2026/3/26
# @File: final_test.py
# @Author: fwb
from logs.print_logs import print_final_logs


def det_test(
        args,
        det_phase,
        best_loss_epoch,
        best_val_epoch,
        best_test_epoch,
        best_loss_save_path,
        best_val_save_path,
        best_test_save_path,
        test_loader,
        device,
        results_dict
):
    # Test best loss.
    metrics = det_phase.model_test(
        args=args,
        model_path=best_loss_save_path,
        test_loader=test_loader,
        device=device
    )
    results_dict['test_best_loss'].append(
        {'best_loss_AP': metrics['AP'],
         'best_loss_AP50': metrics['AP_50'],
         'best_loss_AP75': metrics['AP_75'],
         'best_loss_APS': metrics['AP_S'],
         'best_loss_APM': metrics['AP_M'],
         'best_loss_APL': metrics['AP_L']}
    )
    print_final_logs(
        test_final_results=metrics,
        key_phase_item='train',
        key_metric_item='loss',
        best_epoch=best_loss_epoch
    )
    # Test best val.
    metrics = det_phase.model_test(
        args=args,
        model_path=best_val_save_path,
        test_loader=test_loader,
        device=device
    )
    results_dict['test_best_AP'].append(
        {'best_AP_AP': metrics['AP'],
         'best_AP_AP50': metrics['AP_50'],
         'best_AP_AP75': metrics['AP_75'],
         'best_AP_APS': metrics['AP_S'],
         'best_AP_APM': metrics['AP_M'],
         'best_AP_APL': metrics['AP_L']}
    )
    print_final_logs(
        test_final_results=metrics,
        key_phase_item='val',
        key_metric_item='AP',
        best_epoch=best_val_epoch
    )
    # Test best epoch.
    if best_test_save_path is not None:
        metrics = det_phase.model_test(
            args=args,
            model_path=best_test_save_path,
            test_loader=test_loader,
            device=device
        )
        results_dict['test_best_epoch'].append(
            {'best_epoch_AP': metrics['AP'],
             'best_epoch_AP50': metrics['AP_50'],
             'best_epoch_AP75': metrics['AP_75'],
             'best_epoch_APS': metrics['AP_S'],
             'best_epoch_APM': metrics['AP_M'],
             'best_epoch_APL': metrics['AP_L']}
        )
        print_final_logs(
            test_final_results=metrics,
            key_phase_item='test',
            key_metric_item='epoch',
            best_epoch=best_test_epoch
        )

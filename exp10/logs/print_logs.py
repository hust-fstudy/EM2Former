# -*- coding: utf-8 -*-
# @Time: 2026/3/26
# @File: print_logs.py
# @Author: fwb


def print_epoch_logs(results_dict, test_epoch_results, epoch):
    if any('iou' in key for key in results_dict.keys()):
        best_test_iou = max(results_dict['test_epoch_iou'])
        best_idx = results_dict['test_epoch_iou'].index(best_test_iou) + 1
        print(f"test info for epoch {epoch + 1}: "
              f"(iou: {test_epoch_results['iou']}, "
              f"seg_acc: {test_epoch_results['seg_acc']}, "
              f"pd: {test_epoch_results['pd']}, "
              f"fa: {test_epoch_results['fa']}e-4)")
        print(f"best test iou: "
              f"(best_iou: {best_test_iou}, "
              f"seg_acc: {results_dict['test_epoch_acc'][best_idx - 1]}, "
              f"pd: {results_dict['test_epoch_pd'][best_idx - 1]}, "
              f"fa: {results_dict['test_epoch_fa'][best_idx - 1]}e-4)")
    elif any('AP' in key for key in results_dict.keys()):
        best_test_AP = max(results_dict['test_epoch_AP'])
        best_idx = results_dict['test_epoch_AP'].index(best_test_AP) + 1
        print(f"test info for epoch {epoch + 1}: "
              f"(AP: {test_epoch_results['AP']}, "
              f"AP50: {test_epoch_results['AP_50']}, "
              f"AP75: {test_epoch_results['AP_75']})")
        print(f"best test AP: "
              f"(best_AP: {best_test_AP}, "
              f"AP50: {results_dict['test_epoch_AP50'][best_idx - 1]}, "
              f"AP75: {results_dict['test_epoch_AP75'][best_idx - 1]})")
    else:
        raise ValueError(f"The printing {results_dict} in epoch stage does not meet the requirements!")


def print_final_logs(test_final_results, key_phase_item, key_metric_item, best_epoch):
    if any('iou' in key for key in test_final_results.keys()):
        print(f"test for best {key_phase_item} {key_metric_item} in epoch {best_epoch}: "
              f"(best_{key_metric_item}_iou: {test_final_results['iou']}, "
              f"best_{key_metric_item}_seg_acc: {test_final_results['seg_acc']}, "
              f"best_{key_metric_item}_pd: {test_final_results['pd']}, "
              f"best_{key_metric_item}_fa: {test_final_results['fa']}e-4)")
    elif any('AP' in key for key in test_final_results.keys()):
        print(f"test for best {key_phase_item} {key_metric_item} in epoch {best_epoch}: "
              f"(best_{key_metric_item}_AP: {test_final_results['AP']}, "
              f"best_{key_metric_item}_AP50: {test_final_results['AP_50']}, "
              f"best_{key_metric_item}_AP75: {test_final_results['AP_75']}")
    else:
        raise ValueError(f"The printing {test_final_results} in final stage does not meet the requirements!")

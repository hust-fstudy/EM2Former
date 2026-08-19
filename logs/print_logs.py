# -*- coding: utf-8 -*-
# @Time: 2026/3/26
# @File: print_logs.py
# @Author: fwb


def print_epoch_logs(results_dict, test_epoch_results, epoch):
    if any('AP' in key for key in results_dict.keys()):
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
    if any('AP' in key for key in test_final_results.keys()):
        print(f"test for best {key_phase_item} {key_metric_item} in epoch {best_epoch}: "
              f"(best_{key_metric_item}_AP: {test_final_results['AP']}, "
              f"best_{key_metric_item}_AP50: {test_final_results['AP_50']}, "
              f"best_{key_metric_item}_AP75: {test_final_results['AP_75']}")
    else:
        raise ValueError(f"The printing {test_final_results} in final stage does not meet the requirements!")

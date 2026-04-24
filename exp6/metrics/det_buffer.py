# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: det_buffer.py
# @Author: fwb
import torch
from typing import List, Dict
from metrics.coco_eval import evaluate_detection


def to_cpu(data_list: List[Dict[str, torch.Tensor]]):
    return [{k: v.cpu() for k, v in d.items()} for d in data_list]


class Buffer:
    def __init__(self):
        self.buffer = []

    def extend(self, elements: List[Dict[str, torch.Tensor]]):
        self.buffer.extend(to_cpu(elements))

    def clear(self):
        self.buffer.clear()

    def __iter__(self):
        return iter(self.buffer)

    def __next__(self):
        return next(self.buffer)


class DetBuffer:
    def __init__(self, height: int, width: int, classes: List[str]):
        self.height = height
        self.width = width
        self.classes = classes
        self.detections = Buffer()
        self.ground_truth = Buffer()

    def compile(self, sequences, timestamps):
        detections = compile(self.detections, sequences, timestamps)
        ground_truth = compile(self.ground_truth, sequences, timestamps)
        return detections, ground_truth

    def update(self, detections: List[Dict[str, torch.Tensor]], ground_truth: List[Dict[str, torch.Tensor]]):
        self.detections.extend(detections)
        self.ground_truth.extend(ground_truth)

    def compute(self) -> Dict[str, float]:
        output = evaluate_detection(self.ground_truth.buffer, self.detections.buffer, height=self.height,
                                    width=self.width, classes=self.classes)
        self.detections.clear()
        self.ground_truth.clear()
        return output

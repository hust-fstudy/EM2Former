# -*- coding: utf-8 -*-
# @Time: 2026/1/21
# @File: box_ops.py
# @Author: fwb
import torch


def box_xywh_to_xyxy(x):
    x0, y0, w, h = x.unbind(-1)
    b = [x0, y0, (x0 + w), (y0 + h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [x0, y0, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def format_input(data, normalizer=None):
    if normalizer is None:
        normalizer = torch.tensor([data.width[0], data.height[0], data.time_window[0]],
                                  dtype=torch.float32, device=data.x.device)
    data.pos = torch.cat([data.pos, data.t.view(-1, 1)], dim=-1)
    data.pos = data.pos / normalizer
    del data.t

    return data

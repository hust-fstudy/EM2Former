# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: model_utils.py
# @Author: fwb
import torchvision
import torch
import numpy as np
from torch_geometric.data import Data


def batched_nms_coordinate_trick(boxes, scores, idx, iou_threshold, width, height):
    # adopted from torchvision nms, but faster
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_dim = max([width, height])
    offsets = idx * float(max_dim + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep


def to_eval_format(data):
    targets = []
    for d in data.to_data_list():
        bbox = d.bbox.clone()
        bbox[:, 2:4] += bbox[:, :2]
        targets.append({
            "boxes": bbox[:, :4],
            "labels": bbox[:, 4].long()  # class 0 is background class
        })
    return targets


def _sequential_counter(counts: torch.LongTensor):
    """
    Returns a torch tensor which counts up for each count
    Example: counts = [2,4,6,2,4] then the output will be
    output = [0,1,0,1,2,3,0,1,2,3,4,5,0,1,0,1,2,3]
    """
    assert counts.dtype == torch.long
    assert len(counts.shape) > 0
    assert (counts >= 0).all()

    len_counter = counts.sum()
    tensors_kwargs = dict(device=counts.device, dtype=torch.long)

    # first construct delta function, which has value c_N at position sum_k=0^N c_k
    delta = torch.zeros(size=(len_counter,), **tensors_kwargs)
    x_coord = counts.cumsum(dim=0)
    delta[x_coord[:-1]] = counts[:-1]

    # next construct step function, and the result it a linear function minus this step function
    step = delta.cumsum(dim=0)
    counter = torch.arange(len_counter, **tensors_kwargs) - step

    return counter


def to_train_format(bbox, batch, batch_size):
    max_detections = 100
    targets = torch.zeros(size=(batch_size, max_detections, 5), dtype=torch.float32, device=bbox.device)
    unique, counts = torch.unique(batch, return_counts=True)
    counter = _sequential_counter(counts)

    bbox = bbox.clone()
    # xywhlc pix -> lcxcywh pix
    bbox[:, :2] += bbox[:, 2:4] * .5
    bbox = torch.roll(bbox[:, :5], dims=1, shifts=1)

    targets[batch, counter] = bbox

    return targets


def post_proc_net(prediction, num_classes, conf_thr=0.01, nms_thr=0.65, height=640, width=640,
                  filtering=True):
    prediction[..., :2] -= prediction[..., 2:4] / 2  # cxcywh->xywh
    prediction[..., 2:4] += prediction[..., :2]

    output = []
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if len(image_pred) == 0:
            device = prediction.device
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=device),
                "scores": torch.zeros(0, dtype=torch.float, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        image_pred[:, 4:5] *= class_conf

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thr).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_pred), 1)

        if filtering:
            detections = detections[conf_mask]

        if len(detections) == 0:
            device = prediction.device
            output.append({
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=device),
                "scores": torch.zeros(0, dtype=torch.float, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device)
            })
            continue

        nms_out_index = batched_nms_coordinate_trick(detections[:, :4], detections[:, 4], detections[:, 5],
                                                     nms_thr, width=width, height=height)

        if filtering:
            detections = detections[nms_out_index]

        output.append({
            "boxes": detections[:, :4],
            "scores": detections[:, 4],
            "labels": detections[:, -1].long()
        })

    return output


def grid_to_params(pooling_layer, height, width):
    rx = int(np.ceil(2 * pooling_layer.voxel_size[0].cpu().numpy() * width))
    ry = int(np.ceil(2 * pooling_layer.voxel_size[1].cpu().numpy() * height))
    M = pooling_layer.transform.max
    return rx, ry, M


def init_grid(hw, strides, dtype):
    grids = []
    all_strides = []
    for (hsize, wsize), stride in zip(hw, strides):
        yv, xv = torch.meshgrid(torch.arange(hsize), torch.arange(wsize), indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        all_strides.append(torch.full((*shape, 1), stride))

    grid_cache = torch.cat(grids, dim=1).type(dtype)
    stride_cache = torch.cat(all_strides, dim=1).type(dtype)

    return grid_cache, stride_cache


def shallow_copy(data):
    out = Data(x=data.x.clone(), edge_index=data.edge_index, edge_attr=data.edge_attr, pos=data.pos, batch=data.batch)
    for key in ["active_clusters", "_changed_attr", "_changed_attr_indices", "diff_idx", "diff_pos_idx", "pooling",
                "num_image_channels", "skipped", "pooled"]:
        if hasattr(data, key):
            setattr(out, key, getattr(data, key))
    for key in ["diff_idx", "diff_pos_idx"]:
        if hasattr(data, key):
            setattr(out, key, getattr(data, key).clone())
    return out

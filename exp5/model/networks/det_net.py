# -*- coding: utf-8 -*-
# @Time: 2026/3/11
# @File: det_net.py
# @Author: fwb
import torch
from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss
from model.networks.det_backbone import DetBackbone
from model.layers.graph_conv_block import GraphConvLayer
from model.layers.graph_conv import GraphConvToDense
from model.batch_tools import batch2offset
from model.model_utils import (
    shallow_copy,
    grid_to_params,
    post_proc_net,
    to_eval_format,
    init_grid,
    to_train_format
)


class DetNet(YOLOX):
    def __init__(self, args):
        # Dataset info.
        self.height = args.img_h
        self.width = args.img_w

        # Postprocessing.
        self.conf_thr = args.conf_thr
        self.nms_thr = args.nms_thr

        # Backbone.
        backbone = DetBackbone(
            args,
            height=self.height,
            width=self.width
        )

        # Detection head.
        head = DetHead(
            num_classes=backbone.num_classes,
            in_chs=backbone.out_chs,
            strides=backbone.strides,
            args=args
        )

        super().__init__(backbone=backbone, head=head)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = to_train_format(x.bbox, x.bbox_batch, x.num_graphs)

            outputs = YOLOX.forward(self, x, targets)

            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        detections = post_proc_net(outputs, self.backbone.num_classes, self.conf_thr, self.nms_thr,
                                   filtering=filtering, height=self.height, width=self.width)

        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = to_eval_format(x)
            ret.append(targets)

        return ret


class DetHead(YOLOXHead):
    def __init__(
            self,
            num_classes,
            strides=(8, 16, 32),
            in_chs=(256, 512, 1024),
            act="silu",
            depth_wise=False,
            args=None
    ):
        YOLOXHead.__init__(self, num_classes, args.yolo_stem_width, strides, in_chs, act, depth_wise)

        self.num_scales = args.num_scales
        self.batch_size = args.batch_size

        self.in_channels = in_chs
        self.n_anchors = 1
        self.num_classes = num_classes

        n_reg = max(in_chs)
        self.stem1 = GraphConvLayer(args=args, in_chs=in_chs[0], out_chs=n_reg)
        self.cls_conv1 = GraphConvLayer(args=args, in_chs=n_reg, out_chs=n_reg)
        self.cls_pred1 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=self.n_anchors * self.num_classes,
                                          bias=True)
        self.reg_conv1 = GraphConvLayer(args=args, in_chs=n_reg, out_chs=n_reg)
        self.reg_pred1 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=4, bias=True)
        self.obj_pred1 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=self.n_anchors, bias=True)

        if self.num_scales > 1:
            self.stem2 = GraphConvLayer(args=args, in_chs=in_chs[1], out_chs=n_reg)
            self.cls_conv2 = GraphConvLayer(args=args, in_chs=n_reg, out_chs=n_reg)
            self.cls_pred2 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=self.n_anchors * self.num_classes,
                                              bias=True)
            self.reg_conv2 = GraphConvLayer(args=args, in_chs=n_reg, out_chs=n_reg)
            self.reg_pred2 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=4, bias=True)
            self.obj_pred2 = GraphConvToDense(args=args, in_chs=n_reg, out_chs=self.n_anchors, bias=True)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_chs)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, obj_output

    def forward(self, xin: Data, labels=None, imgs=None):
        # Outputs.
        hybrid_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        if 'offset' not in xin[-1].keys():
            xin[-1]['offset'] = batch2offset(xin[-1].batch)
        batch_size = len(xin[-1].offset)
        cls_output, reg_output, obj_output = self.process_feature(xin[0], self.stem1, self.cls_conv1, self.reg_conv1,
                                                                  self.cls_pred1, self.reg_pred1, self.obj_pred1,
                                                                  batch_size=batch_size, cache=self.cache)

        self.collect_outputs(cls_output, reg_output, obj_output, 0, self.strides[0], ret=hybrid_out)

        if self.num_scales > 1:
            cls_output, reg_output, obj_output = self.process_feature(xin[1], self.stem2, self.cls_conv2,
                                                                      self.reg_conv2, self.cls_pred2, self.reg_pred2,
                                                                      self.obj_pred2, batch_size=batch_size,
                                                                      cache=self.cache)

            self.collect_outputs(cls_output, reg_output, obj_output, 1, self.strides[1], ret=hybrid_out)

        if self.training:
            # Minimize the loss at detections from the image branch.

            return self.get_losses(
                imgs,
                hybrid_out['x_shifts'],
                hybrid_out['y_shifts'],
                hybrid_out['expanded_strides'],
                labels,
                torch.cat(hybrid_out['outputs'], 1),
                hybrid_out['origin_preds'],
                dtype=xin[0].x.dtype,
            )
        else:
            out = hybrid_out['outputs']

            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=out[0].type())

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs

import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F


class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in ['voxels', 'num_points', ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key in ['coordinates', ]:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in ['img_meta', 'gt_labels', 'gt_bboxes', 'gt_types', ]:
                ret[key] = elems
            else:
                if isinstance(elems, dict):
                    ret[key] = {k: torch.stack(v, dim=0) for k, v in elems.items()}
                else:
                    ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        x, conv6, point_misc = self.neck(vx, ret['coordinates'], batch_size, is_test=False)

        losses = dict()

        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['gt_types'],\
                            ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            guided_anchors, _ = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'],\
                        ret['anchors_mask'], ret['gt_bboxes'], ret['gt_labels'], thr=self.train_cfg.rpn.anchor_thr)
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            bbox_score = self.extra_head(conv6, guided_anchors)
            refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
            refine_losses = self.extra_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        guided_anchors, anchor_labels = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, None, thr=.1)

        bbox_score = self.extra_head(conv6, guided_anchors, is_test=True)

        det_bboxes, det_scores, det_labels = self.extra_head.get_rescore_bboxes(
            guided_anchors, bbox_score, anchor_labels, img_meta, self.test_cfg.extra)

        results = [kitti_bbox2results(*param, class_names=self.class_names) for param in zip(det_bboxes, det_scores, det_labels, img_meta)]

        return results




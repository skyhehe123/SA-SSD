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

class TwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 points_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_single_stage_head(rpn_head)

        if bbox_head is not None:
            self.bbox_head = builder.build_bbox_head(bbox_head)
	
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)

        if points_roi_extractor is not None:
            self.points_roi_extractor = builder.build_roi_extractor(
                points_roi_extractor)

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
            if key in [
                'voxels', 'num_points',
            ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6), point_misc = self.neck(vx, ret['coordinates'], batch_size)

        losses = dict()

        aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
        losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)
            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], ret['gt_bboxes'], thr=0.1)
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.with_bbox:
            x = self.bbox_head(conv6)
            rois3d = rbbox2roi(guided_anchors)
            roi_feats = self.bbox_roi_extractor([x], rois3d[:, [0, 1, 2, 4, 5, 7]])
            bbox_score = F.avg_pool2d(roi_feats, roi_feats.shape[-2:])

            refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.rcnn)
            refine_losses = self.bbox_head.loss(*refine_loss_inputs)
            losses.update(refine_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):


        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)

        vx = self.backbone(ret['voxels'], ret['num_points'])
        (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)

        rpn_outs = self.rpn_head.forward(x)

        # proposal_inputs = rpn_outs + (ret['anchors'], ret['anchors_mask'], img_meta, self.test_cfg.rpn)
        # proposal_list = self.rpn_head.get_proposals(*proposal_inputs)
        # return self.rpn_head.get_det_bboxes_nms(*rpn_outs, ret['anchors'], ret['anchors_mask'], img_meta, self.test_cfg.rcnn)

        guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'],
                                                                       None, thr=.1)

        rois3d = rbbox2roi(guided_anchors)
        # bbox_feats = self.bbox_roi_extractor([x], rois3d[:, [0, 1, 2, 4, 5, 7]])
        # misc_preds = self.bbox_head(bbox_feats)

        det_bboxes = [None, ] * batch_size
        det_scores = [None, ] * batch_size

        if len(rois3d) != 0:
            x = self.bbox_head(conv6)
            roi_feats = self.bbox_roi_extractor([x], rois3d[:, [0, 1, 2, 4, 5, 7]])
            bbox_score = F.avg_pool2d(roi_feats, roi_feats.shape[-2:])

            det_bboxes, det_scores = self.bbox_head.get_det_bboxes(
                rois3d, bbox_score, None, img_meta, self.test_cfg.rcnn)

        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]


        return results



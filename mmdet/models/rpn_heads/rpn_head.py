from __future__ import division
from mmdet.models.utils import one_hot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.ops.iou3d.iou3d_utils import nms_gpu, boxes3d_to_bev_torch
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, delta2rbbox3d, add_sin_difference,
                        multi_apply, weighted_cross_entropy, weighted_smoothl1, weighted_sigmoid_focal_loss,
                        weighted_binary_cross_entropy)
from ..utils import normal_init


class RPNHead(nn.Module):
    """Network head of RPN.

                                  / - rpn_cls (1x1 conv)
    input - rpn_conv (3x3 conv) -
                                  \ - rpn_reg (1x1 conv)

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels for the RPN feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        use_sigmoid_cls (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 num_anchors=2,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 use_sigmoid_cls=False,
                 ):
        super(RPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = use_sigmoid_cls

        self.rpn_conv = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.num_anchors = num_anchors
        out_channels = (self.num_anchors
                        if self.use_sigmoid_cls else self.num_anchors * 2)
        self.rpn_cls = nn.Conv2d(feat_channels, out_channels, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, self.num_anchors * 7, 1)
        self.rpn_dir = nn.Conv2d(feat_channels, self.num_anchors * 2, 1)
        self.debug_imgs = None

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_dir, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward(self, x):
        rpn_feat = self.relu(self.rpn_conv(x))
        rpn_cls_score = self.rpn_cls(rpn_feat)
        rpn_bbox_pred = self.rpn_reg(rpn_feat)
        rpn_dir_pred = self.rpn_dir(rpn_feat)
        return rpn_cls_score, rpn_bbox_pred, rpn_dir_pred

    def loss_single(self, rpn_cls_score, rpn_bbox_pred, rpn_dir_pred, labels, label_weights,
                    bbox_targets, bbox_weights, dir_labels, dir_weights, num_total_samples, cfg):

        # classification loss
        labels = labels.contiguous().view(-1)
        label_weights = label_weights.contiguous().view(-1)

        if self.use_sigmoid_cls:
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(-1)
            #criterion = weighted_binary_cross_entropy
            criterion = weighted_sigmoid_focal_loss
        else:
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            criterion = weighted_cross_entropy

        loss_cls = criterion(
            rpn_cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.contiguous().view(-1, 7)
        bbox_weights = bbox_weights.contiguous().view(-1, 7)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 7)
        rpn_bbox_pred, bbox_targets = add_sin_difference(rpn_bbox_pred, bbox_targets)

        loss_reg = weighted_smoothl1(
            rpn_bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        # direction loss
        dir_logits = rpn_dir_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        loss_dir = weighted_cross_entropy(dir_logits, dir_labels, dir_weights, avg_factor=num_total_samples)

        loss_reg *= 2
        loss_dir *= .2

        return loss_cls, loss_reg, loss_dir

    def loss(self, rpn_cls_scores, rpn_bbox_preds, rpn_dir_preds, gt_bboxes, gt_labels, anchors, anchors_mask, cfg):

        cls_reg_targets = multi_apply(anchor_target,
            anchors, anchors_mask, gt_bboxes, gt_labels, target_means=self.target_means, target_stds=self.target_stds, cfg=cfg, sampling=False)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, dir_labels_list, dir_weights_list,
         pos_inds_list, neg_inds_list) = cls_reg_targets

        num_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        losses_cls, losses_reg, losses_dir = self.loss_single(
            rpn_cls_scores,
            rpn_bbox_preds,
            rpn_dir_preds,
            torch.cat(labels_list),
            torch.cat(label_weights_list),
            torch.cat(bbox_targets_list),
            torch.cat(bbox_weights_list),
            torch.cat(dir_labels_list),
            torch.cat(dir_weights_list),
            #num_total_samples=num_pos + num_neg,
            num_total_samples=num_pos,
            cfg=cfg)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_reg=losses_reg, loss_rpn_dir=losses_dir)

    def get_proposals(self, rpn_cls_scores, rpn_bbox_preds, rpn_dir_preds, anchors, anchors_mask, img_meta, cfg):
        num_imgs = len(img_meta)

        proposal_list = []
        for img_id in range(num_imgs):
            rpn_cls_score = rpn_cls_scores[img_id].detach()
            rpn_bbox_pred = rpn_bbox_preds[img_id].detach()
            rpn_dir_pred = rpn_dir_preds[img_id].detach()
            anchor = anchors[img_id]
            anchor_mask = anchors_mask[img_id]
            proposals = self._get_proposals_single(
                rpn_cls_score,
                rpn_bbox_pred,
                rpn_dir_pred,
                anchor,
                anchor_mask,
                img_meta[img_id], cfg)
            proposal_list.append(proposals)
        return proposal_list

    def _get_proposals_single(self, rpn_cls_score, rpn_bbox_pred, rpn_dir_pred, anchors, anchor_mask, img_meta, cfg):

        if self.use_sigmoid_cls:
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0).contiguous().view(-1)
            scores = rpn_cls_score.sigmoid()
        else:
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0).contiguous().view(-1, 2)
            scores = F.softmax(rpn_cls_score, dim=1)[:, 1]

        rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).contiguous().view(-1, 7)

        rpn_dir_label = torch.max(rpn_dir_pred.view(-1, 2), dim=-1)[1]

        scores = scores[anchor_mask]
        rpn_bbox_pred = rpn_bbox_pred[anchor_mask]
        dir_label = rpn_dir_label[anchor_mask]
        anchors = anchors[anchor_mask]

        #####################################
        # borrow from bbox_head, for expriments

        # select = scores > .3
        #
        # bbox_pred = rpn_bbox_pred[select, :]
        # anchors = anchors[select, :]
        # scores = scores[select]
        #
        # if scores.numel() == 0:
        #     return bbox_pred, scores
        #
        # bboxes = delta2rbbox3d(anchors, bbox_pred, self.target_means,
        #                        self.target_stds)
        #
        # keep = nms_gpu(
        #     boxes3d_to_bev_torch(bboxes), scores, .1)

        # bboxes = bboxes.cpu().numpy()
        # scores = scores.cpu().numpy()
        # from mmdet.models.single_stage_heads.ssd_rotate_head import rotate_nms
        # keep = rotate_nms(bboxes[:, [0, 1, 3, 4, 6]], scores, iou_threshold=.01)

        # det_bboxes = bboxes[keep, :]
        # det_scores = scores[keep]
        #
        # return (det_bboxes, det_scores)

        #####################################

        _, order = scores.sort(0, descending=True)

        if cfg.nms_pre > 0:
            order = order[:cfg.nms_pre]
            rpn_bbox_pred = rpn_bbox_pred[order, :]
            anchors = anchors[order, :]
            scores = scores[order]
            dir_label = dir_label[order]

        proposals = delta2rbbox3d(anchors, rpn_bbox_pred, self.target_means,
                               self.target_stds)

        keep = nms_gpu(
            boxes3d_to_bev_torch(proposals), scores, cfg.nms_thr)

        proposals = proposals[keep,:]
        scores = scores[keep]
        dir_label = dir_label[keep]

        proposals = proposals[:cfg.nms_post, :]
        scores = scores[:cfg.nms_post]
        dir_label = dir_label[:cfg.nms_post]

        opp_labels = (proposals[..., -1] > 0) ^ dir_label.byte()
        proposals[opp_labels, -1] += np.pi

        return proposals


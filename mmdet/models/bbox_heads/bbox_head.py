import torch
import torch.nn as nn

from mmdet.core import (delta2bbox, delta2rbbox3d, bbox_target, weighted_binary_cross_entropy,
                        weighted_cross_entropy, weighted_smoothl1, accuracy, add_sin_difference)
from mmdet.core.post_processing.bbox_nms import rotate_nms_torch
from mmdet.ops.iou3d.iou3d_utils import boxes3d_to_bev_torch

class BBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False):
        super(BBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = roi_feat_size

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
        if self.with_reg:
            out_dim_reg = 7 if reg_class_agnostic else 7 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        iou_targets = torch.cat([res.overs for res in sampling_results])
        return cls_reg_targets + (iou_targets, )

    def loss(self, cls_score, bbox_pred, labels, label_weights, bbox_targets,
             bbox_weights, iou_targets):
        soft_label = torch.clamp(2*iou_targets-0.5, 0, 1)
        labels = soft_label * labels.float()

        losses = dict()
        if cls_score is not None:
            losses['loss_cls'] = weighted_binary_cross_entropy(
                cls_score.view(-1), labels, label_weights)

        if bbox_pred is not None:
            bbox_pred, bbox_targets = add_sin_difference(bbox_pred, bbox_targets)
            losses['loss_reg'] = weighted_smoothl1(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                avg_factor=bbox_targets.size(0))

        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_scores,
                       bbox_preds,
                       img_metas,
                       cfg):

        det_bboxes = list()
        det_scores = list()

        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            roi = rois[inds, 1:]
            scores = cls_scores[inds,:]

            if bbox_preds is not None:
                bbox_pred = bbox_preds[inds, :]
            else:
                bbox_pred = roi

            scores = torch.sigmoid(scores).view(-1)

            select = scores > cfg.score_thr
            bbox_pred = bbox_pred[select, :]
            roi = roi[select, :]
            scores = scores[select]

            if scores.numel() == 0:
                det_bboxes.append(bbox_pred)
                det_scores.append(scores)
                continue

            if bbox_preds is not None:
                bbox_pred = delta2rbbox3d(roi, bbox_pred, self.target_means,
                                   self.target_stds)

            boxes_for_nms = boxes3d_to_bev_torch(bbox_pred)
            keep = rotate_nms_torch(boxes_for_nms, scores, iou_threshold=cfg.nms.iou_thr)

            bbox_pred = bbox_pred[keep, :]
            scores = scores[keep]

            det_bboxes.append(bbox_pred.detach().cpu().numpy())
            det_scores.append(scores.detach().cpu().numpy())

        return det_bboxes, det_scores

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()
            bboxes_ = rois[inds, 1:]

            bbox_pred_ = bbox_preds[inds]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = delta2rbbox3d(bboxes_, bbox_pred_, self.target_means,  self.target_stds)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list


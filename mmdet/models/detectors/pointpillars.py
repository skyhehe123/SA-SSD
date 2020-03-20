import logging
import torch
from .. import builder
from mmcv.runner import load_checkpoint
from .base import BaseDetector
import torch.nn.functional as F

class PointPillars(BaseDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head=None,
                 bbox_head=None,
                 rcnn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointPillars, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_single_stage_head(bbox_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_rpn_head(rpn_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if rcnn_head is not None:
            self.rcnn_head = builder.build_bbox_head(rcnn_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
    def freeze_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False

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

        losses = dict()

        canvas = self.backbone(ret['voxels'], ret['coordinates'], ret['num_points'], batch_size)

        x = self.neck(canvas)

        bbox_outs = self.bbox_head(x)
        bbox_loss_inputs = bbox_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg)
        bbox_losses = self.bbox_head.loss(*bbox_loss_inputs)
        losses.update(bbox_losses)

        return losses

    def forward_test(self, img, img_meta, **kwargs):

        batch_size = len(img_meta)
        ret = self.merge_second_batch(kwargs)
        canvas = self.backbone(ret['voxels'], ret['coordinates'], ret['num_points'], batch_size)
        x = self.neck(canvas)

        rpn_outs = self.bbox_head.forward(x)
        proposal_inputs = rpn_outs + (ret['anchors'], ret['anchors_mask'], img_meta, self.test_cfg)

        return self.bbox_head.get_det_bboxes_nms(*proposal_inputs)







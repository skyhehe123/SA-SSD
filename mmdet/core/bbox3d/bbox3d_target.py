from mmdet.core.bbox3d.target_ops import create_target_np
from mmdet.core.bbox3d import region_similarity as regionSimilarity
from mmdet.core.bbox3d import box_coders as boxCoders

class TargetEncoder:
    def __init__(self,
                 box_coders,
                 region_similarity):

        self._similarity_fn = getattr(regionSimilarity, region_similarity)()
        self._box_coder = getattr(boxCoders, box_coders)()

    @property
    def box_coder(self):
        return self._box_coder

    def assign(self,
               anchors,
               gt_boxes,
               anchors_mask=None,
               gt_classes=None,
               pos_iou_thr=0.6,
               neg_iou_thr=0.45,
               positive_fraction=None,
               sample_size=512,
               ):

        return create_target_np(
            anchors,
            gt_boxes,
            anchors_mask,
            gt_classes,
            similarity_fn=self._similarity_fn,
            box_encoding_fn = self._box_coder.encode,
            matched_threshold=pos_iou_thr,
            unmatched_threshold=neg_iou_thr,
            positive_fraction=positive_fraction,
            rpn_batch_size=sample_size,
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size)


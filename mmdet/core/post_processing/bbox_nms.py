import torch
from mmdet.ops.iou3d.iou3d_utils import nms_gpu

def rotate_nms_torch(rbboxes,
                     scores,
                     pre_max_size=None,
                     post_max_size=None,
                     iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]

    if len(rbboxes) == 0:
        keep = torch.empty((0,), dtype=torch.int64)
    else:
        ret = nms_gpu(rbboxes, scores, iou_threshold)
        keep = ret[:post_max_size]

    if keep.shape[0] == 0:
        return None

    if pre_max_size is not None:
        return indices[keep]
    else:
        return keep
import torch
import mmdet.ops.iou3d.iou3d_cuda as iou3d_cuda
import math

def limit_period(val, offset=0.5, period=math.pi):
    return val - torch.floor(val / period + offset) * period

def boxes3d_to_near_torch(boxes3d):
    rboxes = boxes3d[:, [0, 1, 3, 4, 6]]
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        boxes_near: [N, 4(xmin, ymin, xmax, ymax)] nearest boxes
    """
    rots = rboxes[..., -1]
    rots_0_pi_div_2 = torch.abs(limit_period(rots, 0.5, math.pi))
    cond = (rots_0_pi_div_2 > math.pi / 4)[..., None]
    boxes_center = torch.where(cond, rboxes[:, [0, 1, 3, 2]], rboxes[:, :4])
    boxes_near = torch.cat([boxes_center[:, :2] - boxes_center[:, 2:] / 2, \
                        boxes_center[:, :2] + boxes_center[:, 2:] / 2], dim=-1)
    return boxes_near

def boxes_iou(bboxes1, bboxes2, mode='iou', eps=0.0):
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new(rows, cols)

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]
    wh = (rb - lt + eps).clamp(min=0)  # [rows, cols, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + eps) * (
        bboxes1[:, 3] - bboxes1[:, 1] + eps)
    if mode == 'iou':
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + eps) * (
            bboxes2[:, 3] - bboxes2[:, 1] + eps)
        ious = overlap / (area1[:, None] + area2 - overlap)
    else:
        ious = overlap / (area1[:, None])
    return ious

def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 1]
    half_l, half_w = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev

def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a_bev.shape[0], boxes_b_bev.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5]).view(-1, 1)
    boxes_a_height_min = boxes_a[:, 2].view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5]).view(1, -1)
    boxes_b_height_min = boxes_b[:, 2].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d


def nms_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()

def nms_normal_gpu(boxes, scores, thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()

class RotateIou2dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        return boxes_iou_bev(boxes1, boxes2)

class RotateIou3dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        return boxes_iou3d_gpu(boxes1, boxes2)


class NearestIouSimilarity(object):
    """Class to compute similarity based on the squared distance metric.

    This class computes pairwise similarity between two BoxLists based on the
    negative squared distance metric.
    """

    def __call__(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """

        boxes1_near = boxes3d_to_near_torch(boxes1)
        boxes2_near = boxes3d_to_near_torch(boxes2)
        return boxes_iou(boxes1_near, boxes2_near)

if __name__ == '__main__':
    pass
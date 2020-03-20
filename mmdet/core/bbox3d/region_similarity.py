# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""

from mmdet.core.bbox3d.geometry import rbbox2d_to_near_bbox, iou_jit, distance_similarity
from mmdet.core.post_processing.rotate_nms_gpu import rotate_iou_gpu, rotate_iou_gpu_eval
import numba

@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 2], qboxes[j, 2]) - max(
                    boxes[i, 2] - boxes[i, 5], qboxes[j, 2] - qboxes[j, 5]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0

class RotateIou2dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        boxes1_rbv = boxes1[:, [0, 1, 3, 4, 6]]
        boxes2_rbv = boxes2[:, [0, 1, 3, 4, 6]]
        return rotate_iou_gpu(boxes1_rbv, boxes2_rbv)

class RotateIou3dSimilarity(object):
    """Class to compute similarity based on Intersection over Union (IOU) metric.

    This class computes pairwise similarity between two BoxLists based on IOU.
    """
    def __call__(self, boxes1, boxes2):
        boxes1_rbv = boxes1[:, [0, 1, 3, 4, 6]]
        boxes2_rbv = boxes2[:, [0, 1, 3, 4, 6]]
        rinc = rotate_iou_gpu_eval(boxes1_rbv, boxes2_rbv, criterion=2)
        d3_box_overlap_kernel(boxes1, boxes2, rinc)
        return rinc

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
        boxes1_rbv = boxes1[:, [0, 1, 3, 4, 6]]
        boxes2_rbv = boxes2[:, [0, 1, 3, 4, 6]]
        boxes1_bv = rbbox2d_to_near_bbox(boxes1_rbv)
        boxes2_bv = rbbox2d_to_near_bbox(boxes2_rbv)
        ret = iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
        return ret


class DistanceSimilarity(object):
    """Class to compute similarity based on Intersection over Area (IOA) metric.

    This class computes pairwise similarity between two BoxLists based on their
    pairwise intersections divided by the areas of second BoxLists.
    """

    def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
        self._distance_norm = distance_norm
        self._with_rotation = with_rotation
        self._rotation_alpha = rotation_alpha

    def __call__(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise squared distance.
        """
        boxes1_rbv = boxes1[:, [0, 1, 3, 4, 6]]
        boxes2_rbv = boxes2[:, [0, 1, 3, 4, 6]]
        return distance_similarity(
            boxes1_rbv[..., [0, 1, -1]],
            boxes2_rbv[..., [0, 1, -1]],
            dist_norm=self._distance_norm,
            with_rotation=self._with_rotation,
            rot_alpha=self._rotation_alpha)


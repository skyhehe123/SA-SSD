from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import torch
from mmdet.core.bbox3d.geometry import center_to_corner_box3d

def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la**2 + wa**2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal

    zt = (zg - za) / ha  # 1.6
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, zt, wt, lt, ht, rtx, rty], axis=-1)
    else:
        rt = rg - ra
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)

def second_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
            box_encodings, 1, dim=-1)

    else:
        xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    # xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    # need to convert boxes to z-center format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    xg, yg, wg, lg, rg = np.split(boxes, 5, axis=-1)
    diagonal = np.sqrt(la**2 + wa**2)  # 4.3
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, wt, lt, rtx, rty], axis=-1)
    else:
        rt = rg - ra
        return np.concatenate([xt, yt, wt, lt, rt], axis=-1)

def bev_box_decode(box_encodings, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    # need to convert box_encodings to z-bottom format
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = np.split(box_encodings, 6, axis=-1)
    else:
        xt, yt, wt, lt, rt = np.split(box_encodings, 5, axis=-1)
    diagonal = np.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    return np.concatenate([xg, yg, wg, lg, rg], axis=-1)

class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 8 if self.vec_encode else 7

    def _encode(self, boxes, anchors):
        return second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        return second_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)


class BevBoxCoder(BoxCoder):
    """WARNING: this coder will return encoding with size=5, but 
    takes size=7 boxes, anchors
    """
    def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0):
        super().__init__()
        self.linear_dim = linear_dim
        self.z_fixed = z_fixed
        self.h_fixed = h_fixed
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 6 if self.vec_encode else 5

    def _encode(self, boxes, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        boxes = boxes[..., [0, 1, 3, 4, 6]]
        return bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        anchors = anchors[..., [0, 1, 3, 4, 6]]
        ret = bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
        z_fixed = np.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
        h_fixed = np.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
        return np.concatenate([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], axis=-1)


class BoxCornerCoder(BoxCoder):
    def __init__(self, ):
        super(BoxCornerCoder).__init__()

    @property
    def code_size(self):
        return 24

    def _encode(self, boxes, anchors):
        assert len(boxes) == len(anchors)
        N = len(boxes)
        boxes = center_to_corner_box3d(boxes)
        anchors = center_to_corner_box3d(anchors)
        offset = boxes - anchors
        return offset.reshape(N, -1)

    def _decode(self, encodings, anchors):
        NotImplementedError
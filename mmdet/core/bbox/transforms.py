import mmcv
import numpy as np
import torch
from mmdet.datasets.kitti_utils import project_rect_to_image, project_velo_to_rect
from mmdet.core.bbox3d.geometry import center_to_corner_box3d, limit_period

def rbbox3d2delta(anchors, boxes, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    lt = torch.log(lg / la)
    wt = torch.log(wg / wa)
    ht = torch.log(hg / ha)
    rt = rg - ra
    deltas = torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)
    return deltas

def delta2rbbox3d(anchors, deltas, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)
    za = za + ha / 2
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha
    rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)

def add_sin_difference(boxes1, boxes2):
     rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
         boxes2[..., -1:])
     rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
     boxes1 = torch.cat((boxes1[..., :-1], rad_pred_encoding), dim=-1)
     boxes2 = torch.cat((boxes2[..., :-1], rad_tg_encoding), dim=-1)
     return boxes1, boxes2

def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 8), [batch_ind, x, y, z, w, l, h, r]
    """

    rois_list = []
    for img_id, rois3d in enumerate(bbox_list):
        if rois3d.size(0) > 0:
            img_inds = rois3d.new_full((rois3d.size(0), 1), img_id)
            rois = torch.cat([img_inds, rois3d], dim=-1)
        else:
            rois = rois3d.new_zeros((0, 8))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

def tensor2points(tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
    indices = tensor.indices.float()
    offset = torch.Tensor(offset).to(indices.device)
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
    return tensor.features, indices

def kitti_empty_results(sample_id):
    return {"bbox": None, "box3d_camera": None,
                            "box3d_lidar": None, "scores": None,
                            "label_preds": None, "image_idx": sample_id}

def kitti_bbox2results(boxes_lidar, scores, meta, labels=None):
    calib = meta['calib']
    sample_id = meta['sample_idx']

    if scores is None \
        or len(scores) == 0 \
            or boxes_lidar is None \
                or len(boxes_lidar) == 0:

        predictions_dict = kitti_empty_results(sample_id)

    else:
        if labels is None:
            labels = np.zeros(len(boxes_lidar), dtype=np.long)

        boxes_lidar[:, -1] = limit_period(
            boxes_lidar[:, -1], offset=0.5, period=np.pi * 2,
        )

        boxes_cam = np.zeros_like(boxes_lidar)
        boxes_cam[:, :3] = project_velo_to_rect(boxes_lidar[:, :3], calib)
        boxes_cam[:, 3:] = boxes_lidar[:, [4, 5, 3, 6]]

        corners_cam = center_to_corner_box3d(boxes_cam, origin=[0.5, 1.0, 0.5], axis=1)
        corners_rgb = project_rect_to_image(corners_cam, calib)

        minxy = np.min(corners_rgb, axis=1)
        maxxy = np.max(corners_rgb, axis=1)

        box2d_rgb = np.concatenate([minxy, maxxy], axis=1)

        alphas = -np.arctan2(-boxes_lidar[:, 1], boxes_lidar[:, 0]) + boxes_lidar[:, 6]

        predictions_dict = {
            "bbox": box2d_rgb, "box3d_camera": boxes_cam, "box3d_lidar": boxes_lidar,\
            "alphas": alphas, "scores": scores, "label_preds": labels, "image_idx": sample_id
        }

    return predictions_dict


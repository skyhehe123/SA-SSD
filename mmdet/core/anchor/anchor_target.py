import torch
from ..bbox import assign_and_sample, BBoxAssigner, SamplingResult, bbox2delta, rbbox3d2delta

def anchor_target(flat_anchors,
                  inside_flags,
                  gt_bboxes,
                  gt_labels,
                  target_means,
                  target_stds,
                  cfg,
                  cls_out_channels=1,
                  sampling=True):

    # assign gt and sample anchors

    anchors = flat_anchors[inside_flags]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, None, None, cfg)
    else:
        bbox_assigner = BBoxAssigner(**cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, None, gt_labels)
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = anchors.new_zeros(anchors.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, anchors,
                                         gt_bboxes, assign_result, gt_flags)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    dir_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = rbbox3d2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        dir_weights[pos_inds] = 1.
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    num_total_anchors = flat_anchors.shape[0]
    labels = unmap(labels, num_total_anchors, inside_flags)
    label_weights = unmap(label_weights, num_total_anchors, inside_flags)
    if cls_out_channels > 1:
        labels, label_weights = expand_binary_labels(labels, label_weights,
                                                     cls_out_channels)
    bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    dir_labels = get_direction_target(flat_anchors, bbox_targets)
    dir_weights = unmap(dir_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, dir_labels, dir_weights, pos_inds,
            neg_inds)


def expand_binary_labels(labels, label_weights, cls_out_channels):
    bin_labels = labels.new_full(
        (labels.size(0), cls_out_channels), 0, dtype=torch.float32)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), cls_out_channels)
    return bin_labels, bin_label_weights

def get_direction_target(anchors, reg_targets):
    anchors = anchors.view(-1, 7)
    rot_gt = reg_targets[:, -1] + anchors[:, -1]
    dir_cls_targets = (rot_gt > 0).long()
    return dir_cls_targets

def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

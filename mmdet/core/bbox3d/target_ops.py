import numpy as np
import numpy.random as npr
import torch

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def create_target_np(all_anchors,
                     gt_boxes,
                     anchors_mask,
                     gt_classes,
                     similarity_fn,
                     box_encoding_fn,
                     matched_threshold=0.6,
                     unmatched_threshold=0.45,
                     positive_fraction=None,
                     rpn_batch_size=300,
                     norm_by_num_examples=False,
                     box_code_size=7):
    total_anchors = all_anchors.shape[0]
    if anchors_mask is not None:
        inds_inside = np.where(anchors_mask)[0]  # prune_anchor_fn(all_anchors)
        anchors = all_anchors[inds_inside, :]
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[inds_inside]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[inds_inside]
    else:
        anchors = all_anchors
        inds_inside = None
    num_inside = len(inds_inside) if inds_inside is not None else total_anchors

    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside,), dtype=np.int32)
    gt_ids = np.empty((num_inside,), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]  #
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max)[0]
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0]

    # subsample positive labels if we have too many
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels > 0)[0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = rpn_batch_size - np.sum(labels > 0)
        # print(num_fg, num_bg, len(bg_inds) )
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
    else:
        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            # re-enable anchors_with_max_overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
    bbox_targets = np.zeros(
        (num_inside, box_code_size), dtype=all_anchors.dtype)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        bbox_targets[fg_inds, :] = box_encoding_fn(
            gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])
    bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)

    # uniform weighting of examples (given non-uniform sampling)
    if norm_by_num_examples:
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0

    # Map up to original set of anchors
    if inds_inside is not None:
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)

    return (labels, bbox_targets, anchor_to_gt_max)


def create_target_torch(all_anchors,
                        gt_boxes,
                        anchor_mask,
                        gt_classes,
                        similarity_fn,
                        box_encoding_fn,
                        matched_threshold=0.6,
                        unmatched_threshold=0.45,
                        positive_fraction=None,
                        rpn_batch_size=300,
                        norm_by_num_examples=False,
                        box_code_size=7):

    def _unmap(data, count, inds, fill=0):
        """ Unmap a subset of item (data) back to the original set of items (of
        size count) """
        if data.dim() == 1:
            ret = data.new_full((count,), fill)
            ret[inds] = data
        else:
            new_size = (count,) + data.size()[1:]
            ret = data.new_full(new_size, fill)
            ret[inds, :] = data
        return ret

    total_anchors = all_anchors.shape[0]
    if anchor_mask is not None:
        #inds_inside = np.where(anchors_mask)[0]  # prune_anchor_fn(all_anchors)
        anchors = all_anchors[anchor_mask, :]
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[anchor_mask]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[anchor_mask]
    else:
        anchors = all_anchors
        #inds_inside = None
    num_inside = len(torch.nonzero(anchor_mask)) if anchor_mask is not None else total_anchors

    if gt_classes is None:
        gt_classes = torch.ones([gt_boxes.shape[0]], dtype=torch.int64, device=gt_boxes.device)
    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = torch.empty((num_inside,), dtype=torch.int64, device=gt_boxes.device).fill_(-1)
    gt_ids = torch.empty((num_inside,), dtype=torch.int64, device=gt_boxes.device).fill_(-1)

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_inside),
                                                anchor_to_gt_argmax]  #
        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            torch.arange(anchor_by_gt_overlap.shape[1])]
        # must remove gt which doesn't match any anchor.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        # Find all anchors that share the max overlap amount
        # (this includes many ties)

        anchors_with_max_overlap = torch.nonzero(
            anchor_by_gt_overlap == gt_to_anchor_max)[:,0]
        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        # Fg label: above threshold IOU
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        #bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
        bg_inds = torch.nonzero(anchor_to_gt_max < unmatched_threshold)[:, 0]
    else:
        bg_inds = torch.arange(num_inside)

    #fg_inds = np.where(labels > 0)[0]
    fg_inds = torch.nonzero(labels > 0)[:, 0]

    # subsample positive labels if we have too many
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
            #fg_inds = np.where(labels > 0)[0]
            fg_inds = torch.where(labels > 0)[:, 0]

        # subsample negative labels if we have too many
        # (samples with replacement, but since the set of bg inds is large most
        # samples will not have repeats)
        num_bg = rpn_batch_size - np.sum(labels > 0)
        # print(num_fg, num_bg, len(bg_inds) )
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
    else:
        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            # re-enable anchors_with_max_overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

    bbox_targets = torch.zeros(
        (num_inside, box_code_size), dtype=all_anchors.dtype, device=gt_boxes.device)

    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        bbox_targets[fg_inds, :] = box_encoding_fn(
            gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])

    bbox_outside_weights = torch.zeros((num_inside,), dtype=all_anchors.dtype, device=gt_boxes.device)

    # uniform weighting of examples (given non-uniform sampling)
    if norm_by_num_examples:
        num_examples = torch.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0

    # Map up to original set of anchors
    if anchor_mask is not None:
        labels = _unmap(labels, total_anchors, anchor_mask, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, anchor_mask, fill=0)

    return (labels, bbox_targets, anchor_to_gt_max)

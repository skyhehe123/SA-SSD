import numpy as np
import pathlib
from mmdet.core.bbox3d.geometry import (center_to_corner_box2d,\
                                        center_to_corner_box3d,\
                                        box2d_to_corner_jit,\
                                        points_in_convex_polygon_3d_jit,\
                                        corner_to_surfaces_3d_jit,\
                                        rotation_box2d_jit,\
                                        rotation_points_single_angle,\
                                        box_collision_test)
import copy
import pickle
import numba
from mmdet.datasets.kitti_utils import project_velo_to_rect, project_rect_to_velo

def select_transform(transform, indices):
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result

@numba.njit
def rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos

@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform

@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                rotation_box2d_jit(current_corners, rot_noises[i, j],
                                   rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask

class BatchSampler:
    def __init__(self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        if self._name is not None:
            print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


class PointAugmentor:
    def __init__(self, root_path, \
                 info_path, \
                 sample_classes, \
                 min_num_points, \
                 sample_max_num, \
                 removed_difficulties,
                 gt_rot_range=None, \
                 global_rot_range=None,
                 center_noise_std=None, \
                 scale_range=None):

        with open(info_path, 'rb') as f:
            db_infos_all = pickle.load(f)

        self._samplers = list()

        if isinstance(min_num_points, int):
            min_num_points = [min_num_points] * len(sample_classes)

        for i, sample_class in enumerate(sample_classes):
            db_infos = db_infos_all[sample_class]
            print(f"load {len(db_infos)} {sample_class} database infos")

            filtered_infos = []
            for info in db_infos:
                if info["num_points_in_gt"] >= min_num_points[i]:
                    filtered_infos.append(info)
            db_infos = filtered_infos

            new_db_infos = [
                info for info in db_infos
                if info["difficulty"] not in removed_difficulties
            ]

            print("After filter database:")
            print(f"load {len(new_db_infos)} {sample_class} database infos")

            self._samplers.append(BatchSampler(new_db_infos, sample_class))

        self.root_path = root_path
        # self._db_infos = new_db_infos
        self._sample_classes = sample_classes

        if isinstance(sample_max_num, int):
            self._sample_max_num = [sample_max_num] * len(sample_classes)
        else:
            self._sample_max_num = sample_max_num

        self._global_rot_range = global_rot_range
        self._gt_rot_range = gt_rot_range
        self._center_noise_std = center_noise_std
        self._min_scale = scale_range[0]
        self._max_scale = scale_range[1]

    def sample_all(self, gt_boxes, gt_types, road_planes=None, calib=None):
        avoid_coll_boxes = gt_boxes

        sampled = []
        sampled_gt_boxes = []

        for i, class_name in enumerate(self._sample_classes):
            sampled_num_per_class = int(self._sample_max_num[i] - np.sum([n == class_name for n in gt_types]))
            if sampled_num_per_class > 0:
                sampled_cls = self.sample(avoid_coll_boxes, sampled_num_per_class, i)
            else:
                sampled_cls = []

            sampled += sampled_cls

            if len(sampled_cls) > 0:
                if len(sampled_cls) == 1:
                    sampled_gt_box = sampled_cls[0]["box3d_lidar"][
                        np.newaxis, ...]
                else:
                    sampled_gt_box = np.stack(
                        [s["box3d_lidar"] for s in sampled_cls], axis=0)

                sampled_gt_boxes += [sampled_gt_box]
                avoid_coll_boxes = np.concatenate(
                    [avoid_coll_boxes, sampled_gt_box], axis=0)

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            if road_planes is not None:
                center = sampled_gt_boxes[:, 0:3]
                a, b, c, d = road_planes
                center_cam = project_velo_to_rect(center, calib)
                cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
                center_cam[:, 1] = cur_height_cam
                lidar_tmp_point = project_rect_to_velo(center_cam, calib)
                cur_lidar_height = lidar_tmp_point[:, 2]
                mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                sampled_gt_boxes[:, 2] -= mv_height  # lidar view
            s_points_list = []
            sampled_gt_types = []
            for i, info in enumerate(sampled):
                s_points = np.fromfile(
                    str(pathlib.Path(self.root_path) / info["path"]),
                    dtype=np.float32)
                s_points = s_points.reshape([-1, 4])
                s_points[:, :3] += info["box3d_lidar"][:3]
                if road_planes is not None:
                    s_points[:, 2] -= mv_height[i]
                s_points_list.append(s_points)
                sampled_gt_types.append(info['name'])

            return sampled_gt_boxes.astype(np.float32), sampled_gt_types, np.concatenate(s_points_list, axis=0)

        else:
            return np.empty((0, 7), dtype=np.float32), [], np.empty((0, 4), dtype=np.float32)

    def sample(self, gt_boxes, num, i):
        sampled = self._samplers[i].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled)
        gt_boxes_bv = center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, 6])

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)

        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_boxes.shape[0]:]
        sp_boxes_bv = center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)

        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

    def random_flip(self, gt_boxes, points, probability=0.5):
        enable = np.random.choice(
            [False, True], replace=False, p=[1 - probability, probability])
        if enable:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
            points[:, 1] = -points[:, 1]
        return gt_boxes, points

    def global_rotation(self, gt_boxes, points):
        noise_rotation = np.random.uniform(self._global_rot_range[0], \
                                           self._global_rot_range[1])
        points[:, :3] = rotation_points_single_angle(
            points[:, :3], noise_rotation, axis=2)
        gt_boxes[:, :3] = rotation_points_single_angle(
            gt_boxes[:, :3], noise_rotation, axis=2)
        gt_boxes[:, 6] += noise_rotation
        return gt_boxes, points

    def global_scaling(self, gt_boxes, points):
        noise_scale = np.random.uniform(self._min_scale, self._max_scale)
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return gt_boxes, points


    def noise_per_object_(self,
                          gt_boxes,
                          points=None,
                          valid_mask=None,
                          num_try=100):
        """random rotate or remove each groundtrutn independently.
        use kitti viewer to test this function points_transform_

        Args:
            gt_boxes: [N, 7], gt box in lidar.points_transform_
            points: [M, 4], point cloud in lidar.
        """
        num_boxes = gt_boxes.shape[0]

        if valid_mask is None:
            valid_mask = np.ones((num_boxes,), dtype=np.bool_)
        center_noise_std = np.array(self._center_noise_std, dtype=gt_boxes.dtype)
        loc_noises = np.random.normal(
            scale=center_noise_std, size=[num_boxes, num_try, 3])

        rot_noises = np.random.uniform(
            self._global_rot_range[0], self._global_rot_range[1], size=[num_boxes, num_try])

        origin = [0.5, 0.5, 0]
        gt_box_corners = center_to_corner_box3d(gt_boxes, origin=origin, axis=2)

        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises)

        loc_transforms = select_transform(loc_noises, selected_noise)
        rot_transforms = select_transform(rot_noises, selected_noise)
        surfaces = corner_to_surfaces_3d_jit(gt_box_corners)

        if points is not None:
            point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
            points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                              rot_transforms, valid_mask)

        box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


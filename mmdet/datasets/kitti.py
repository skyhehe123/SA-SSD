import os.path as osp
import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
import torch
from torch.utils.data import Dataset
from mmdet.datasets.transforms import (ImageTransform, BboxTransform)
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.kitti_utils import read_label, read_lidar, \
    project_rect_to_velo, Calibration, get_lidar_in_image_fov, \
    project_rect_to_image, project_rect_to_right, load_proposals
from mmdet.core.bbox3d.geometry import rbbox2d_to_near_bbox, filter_gt_box_outside_range, \
    sparse_sum_for_anchors_mask, fused_get_anchors_area, limit_period, center_to_corner_box3d, points_in_rbbox
import os
from mmdet.core.point_cloud.voxel_generator import VoxelGenerator
from mmdet.ops.points_op import points_op_cpu

class KittiLiDAR(Dataset):
    def __init__(self, root, ann_file,
                 img_prefix,
                 img_norm_cfg,
                 img_scale=(1242, 375),
                 size_divisor=32,
                 proposal_file=None,
                 flip_ratio=0.5,
                 with_point=False,
                 with_mask=False,
                 with_label=True,
                 with_plane=False,
                 class_names = ['Car', 'Van'],
                 augmentor=None,
                 generator=None,
                 anchor_generator=None,
                 anchor_area_threshold=1,
                 target_encoder=None,
                 out_size_factor=2,
                 test_mode=False):
        self.root = root
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # flip ratio
        self.flip_ratio = flip_ratio

        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        self.class_names = class_names
        self.test_mode = test_mode
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_point = with_point
        self.with_plane = with_plane
        self.img_prefix = osp.join(root, 'image_2')
        self.right_prefix = osp.join(root, 'image_3')
        self.lidar_prefix = osp.join(root, 'velodyne_reduced')
        self.calib_prefix = osp.join(root, 'calib')
        self.label_prefix = osp.join(root, 'label_2')
        self.plane_prefix = osp.join(root, 'planes')

        with open(ann_file, 'r') as f:
            self.sample_ids = list(map(int, f.read().splitlines()))

        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)

        # voxel
        self.augmentor = augmentor
        self.generator = generator
        self.target_encoder = target_encoder
        self.out_size_factor = out_size_factor
        self.anchor_area_threshold = anchor_area_threshold

        # anchor
        if anchor_generator is not None:
            feature_map_size = self.generator.grid_size[:2] // self.out_size_factor
            feature_map_size = [*feature_map_size, 1][::-1]

            if self.test_mode:
                self.anchors = np.concatenate([v(feature_map_size).reshape(-1, 7) for v in anchor_generator.values()],
                                              0)
                self.anchors_bv = rbbox2d_to_near_bbox(self.anchors[..., [0, 1, 3, 4, 6]])
            else:
                self.anchors = {k: v(feature_map_size).reshape(-1, 7) for k, v in anchor_generator.items()}
                self.anchors_bv = {k: rbbox2d_to_near_bbox(v[:, [0, 1, 3, 4, 6]]) for k, v in self.anchors.items()}

        else:
            self.anchors = None

    def get_road_plane(self, plane_file):
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.sample_ids)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        sample_id = self.sample_ids[idx]

        # load image
        img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, 1, False)

        objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        gt_bboxes = [object.box3d for object in objects if object.type not in ["DontCare"]]
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_types = [object.type for object in objects if object.type not in ["DontCare"]]

        # transfer from cam to lidar coordinates
        if len(gt_bboxes) != 0:
            gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)

        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta = DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = {k: DC(to_tensor(v.astype(np.float32))) for k, v in self.anchors.items()}

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if self.with_plane:
            plane = self.get_road_plane(osp.join(self.plane_prefix, '%06d.txt' % sample_id))
        else:
            plane = None

        if self.augmentor is not None and self.test_mode is False:
            sampled_gt_boxes, sampled_gt_types, sampled_points = self.augmentor.sample_all(gt_bboxes, gt_types, plane, calib)
            assert sampled_points.dtype == np.float32
            gt_bboxes = np.concatenate([gt_bboxes, sampled_gt_boxes])
            gt_types = gt_types + sampled_gt_types
            assert len(gt_types) == len(gt_bboxes)

            # to avoid overlapping point (option)
            masks = points_in_rbbox(points, sampled_gt_boxes)
            #masks = points_op_cpu.points_in_bbox3d_np(points[:,:3], sampled_gt_boxes)
            points = points[np.logical_not(masks.any(-1))]

            # paste sampled points to the scene
            points = np.concatenate([sampled_points, points], axis=0)

            # force van to have same type as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_types = np.array(gt_types)

            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = gt_types[selected]
            gt_labels = np.array([self.class_names.index(n) + 1 for n in gt_types], dtype=np.int64)

            self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
            gt_bboxes, points = self.augmentor.random_flip(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_rotation(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_scaling(gt_bboxes, points)

        if isinstance(self.generator, VoxelGenerator):
            voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            #keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            #voxels = points[keep, :]
            #coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2,1,0]], dtype=np.float32)) / np.array(
            #    voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            #num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            if self.anchor_area_threshold >= 0 and self.anchors is not None:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                anchors_mask = dict()
                for k, v in self.anchors_bv.items():
                    mask = fused_get_anchors_area(
                        dense_voxel_map, v, voxel_size, pc_range,
                        grid_size) > self.anchor_area_threshold
                    anchors_mask[k] = DC(to_tensor(mask.astype(np.bool)))
                data['anchors_mask'] = anchors_mask

            # filter gt_bbox out of range
            bv_range = self.generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(gt_bboxes, bv_range)
            gt_bboxes = gt_bboxes[mask]
            gt_types = gt_types[mask]
            gt_labels = gt_labels[mask]

        else:
            NotImplementedError

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # limit rad to [-pi, pi]
        gt_bboxes[:, 6] = limit_period(
            gt_bboxes[:, 6], offset=0.5, period=2 * np.pi)

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes))
            data['gt_types'] = DC(gt_types, cpu_only=True)

        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        sample_id = self.sample_ids[idx]
        # sample_id=8
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, 1, False)

        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        if self.with_label:
            objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
            gt_bboxes = [object.box3d for object in objects if object.type not in ["DontCare"]]
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_types = [object.type for object in objects if object.type not in ["DontCare"]]

            # transfer from cam to lidar coordinates
            if len(gt_bboxes) != 0:
                gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)

            # force van to have same type as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_types = np.array(gt_types)
            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = gt_types[selected]
            gt_labels = np.array([self.class_names.index(n) + 1 for n in gt_types], dtype=np.int64)


        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta=DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = DC(to_tensor(self.anchors.astype(np.float32)))

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        if isinstance(self.generator, VoxelGenerator):
            voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            #keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            #voxels = points[keep, :]

            #coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2, 1, 0]], dtype=np.float32)) / np.array(
            #    voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            #num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))

            if self.anchor_area_threshold >= 0 and self.anchors is not None:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)

                anchors_mask = fused_get_anchors_area(
                    dense_voxel_map, self.anchors_bv, voxel_size, pc_range,
                    grid_size) > self.anchor_area_threshold

                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.bool)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels), cpu_only=True)
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes), cpu_only=True)
            data['gt_types'] = DC(gt_types, cpu_only=True)
        else:
            data['gt_labels'] = DC(None, cpu_only=True)
            data['gt_bboxes'] = DC(None, cpu_only=True)
            data['gt_types'] = DC(None, cpu_only=True)

        return data

class KittiVideo(KittiLiDAR):
    ''' Load data for KITTI videos '''

    def __init__(self, img_dir, lidar_dir, calib_dir, **kwargs):
        super(KittiVideo, self).__init__(**kwargs)

        self.calib = Calibration(os.path.join(self.root, calib_dir), from_video=True)
        self.img_dir = os.path.join(self.root, img_dir)
        self.lidar_dir = os.path.join(self.root, lidar_dir)
        self.img_filenames = sorted([os.path.join(self.img_dir, filename) \
                                     for filename in os.listdir(self.img_dir)])

        self.lidar_filenames = sorted([os.path.join(self.lidar_dir, filename) \
                                       for filename in os.listdir(self.lidar_dir)])

        sample_ids = sorted([os.path.splitext(filename)[0] \
                             for filename in os.listdir(self.img_dir)])
        self.sample_ids = list(map(int, sample_ids))

    def prepare_test_img(self, idx):
        sample_id = self.sample_ids[idx]
        # load image
        img = mmcv.imread(self.img_filenames[idx])
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, 1, False)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_shape=DC(img_shape, cpu_only=True),
            sample_idx=DC(sample_id, cpu_only=True),
            calib=DC(self.calib, cpu_only=True)
        )

        if self.with_mask:
            NotImplemented

        if self.with_point:
            points = read_lidar(self.lidar_filenames[idx])
            points = get_lidar_in_image_fov(points, self.calib, 0, 0, img_shape[1], img_shape[0], clip_distance=0.1)

        if self.generator is not None:
            voxels, coordinates, num_points = self.generator.generate(points)
            data['voxels'] = DC(to_tensor(voxels))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))
            data['anchors'] = DC(to_tensor(self.anchors))

        return data





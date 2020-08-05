import spconv
from torch import nn
from mmdet.models.utils import change_default_args, Sequential
from mmdet.ops.pointnet2 import pointnet2_utils
import torch
from mmdet.ops import pts_in_boxes3d
from mmdet.core.loss.losses import weighted_smoothl1, weighted_sigmoid_focal_loss
from mmdet.core import tensor2points
import torch.nn.functional as F


class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=4,
                 num_hidden_features=128,
                 ):

        super(SpMiddleFHD, self).__init__()

        print(output_shape)
        self.sparse_shape = output_shape

        self.backbone = VxNet(num_input_features)
        self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)

        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)

    def _make_layer(self, conv2d, bachnorm2d, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            conv2d(inplanes, planes, 3, stride=stride),
            bachnorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(conv2d(planes, planes, 3, padding=1))
            block.add(bachnorm2d(planes))
            block.add(nn.ReLU())
        return block, planes

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )

    def forward(self, voxel_features, coors, batch_size, is_test=False):

        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        x, middle = self.backbone(x)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        x, conv6 = self.fcn(x)

        if is_test:
            return x, conv6
        else:
            # auxiliary network
            vx_feat, vx_nxyz = tensor2points(middle[0], (0, -40., -3.), voxel_size=(.1, .1, .2))
            p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            vx_feat, vx_nxyz = tensor2points(middle[1], (0, -40., -3.), voxel_size=(.2, .2, .4))
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            vx_feat, vx_nxyz = tensor2points(middle[2], (0, -40., -3.), voxel_size=(.4, .4, .8))
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            pointwise = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
            point_cls = self.point_cls(pointwise)
            point_reg = self.point_reg(pointwise)

            return x, conv6, (points_mean, point_cls, point_reg)


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
    )

def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 3, (2, 2, 2), padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
    )

def nearest_neighbor_interpolate(unknown, known, known_feats):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    dist, idx = pointnet2_utils.three_nn(unknown, known)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

    return interpolated_feats


class VxNet(nn.Module):

    def __init__(self, num_input_features):
        super(VxNet, self).__init__()

        self.conv0 = double_conv(num_input_features, 16, 'subm0')

        self.down0 = stride_conv(16, 32, 'down0')
        self.conv1 = double_conv(32, 32, 'subm1')

        self.down1 = stride_conv(32, 64, 'down1')
        self.conv2 = triple_conv(64, 64, 'subm2')

        self.down2 = stride_conv(64, 64, 'down2')
        self.conv3 = triple_conv(64, 64, 'subm3')  # middle line

        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(64, 64, (1, 1, 1), (1, 1, 1), bias=False),  # shape no change
            nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        middle = list()

        x = self.conv0(x)
        x = self.down0(x)  # sp
        x = self.conv1(x)  # 2x sub
        middle.append(x)

        x = self.down1(x)
        x = self.conv2(x)
        middle.append(x)

        x = self.down2(x)
        x = self.conv3(x)
        middle.append(x)

        out = self.extra_conv(x)
        return out, middle

class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        BatchNorm2d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)

        self.conv0 = Conv2d(in_features, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)

        self.conv6 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = BatchNorm2d(num_filters)

        self.conv7 = Conv2d(num_filters, num_filters, 1)
        self.bn7 = BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.conv5(x)
        x = F.relu(self.bn5(x), inplace=True)
        x = self.conv6(x)
        x = F.relu(self.bn6(x), inplace=True)
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6




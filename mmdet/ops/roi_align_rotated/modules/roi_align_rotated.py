from torch.nn.modules.module import Module
from ..functions.roi_align_rotated import RoIAlignRotatedFunction, \
    PSRoIAlignRotatedFunction


class RoIAlignRotated(Module):

    def __init__(self, out_size, spatial_scale, x_offset=0., y_offset=40., sample_num=0):
        super(RoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.x_offset = x_offset
        self.y_offset = y_offset

    def forward(self, features, rois):
        rois[:, 1] += self.x_offset
        rois[:, 2] += self.y_offset
        return RoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)

class PSRoIAlignRotated(Module):

    def __init__(self, out_size, spatial_scale, x_offset=0., y_offset=40., sample_num=0):
        super(PSRoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.x_offset = x_offset
        self.y_offset = y_offset

    def forward(self, features, rois):
        rois[:, 1] += self.x_offset
        rois[:, 2] += self.y_offset
        return PSRoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)


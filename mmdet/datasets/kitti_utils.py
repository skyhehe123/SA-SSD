import numpy as np
import os
import cv2


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        try:
            self.score = data[15]
        except:
            self.score = 1.
        self.box3d = np.array(
                [data[11], data[12], data[13], data[9], data[10], data[8], data[14]]).astype(np.float32)
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])

        # Projection matrix from rect camera coord to image3 coord
        self.P3 = calibs['P3']
        self.P3 = np.reshape(self.P3, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])

        self.C2V = np.zeros_like(self.V2C)  # 3x4
        self.C2V[0:3, 0:3] = np.transpose(self.V2C[0:3, 0:3])
        self.C2V[0:3, 3] = np.dot(-np.transpose(self.V2C[0:3, 0:3]), \
                                  self.V2C[0:3, 3])

        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P2[0, 2]
        self.c_v = self.P2[1, 2]
        self.f_u = self.P2[0, 0]
        self.f_v = self.P2[1, 1]
        self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P2[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        data['P3'] = cam2cam['P_rect_03']
        return data

def read_lidar(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = list(pts_3d.shape[0:-1])
    pts_3d_hom = np.concatenate([pts_3d, np.ones(n + [1])], axis=-1)
    return pts_3d_hom

# ===========================
# ------- 3d to 3d ----------
# ===========================
def project_velo_to_ref(pts_3d_velo, calib):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return pts_3d_velo @ calib.V2C.T

def project_ref_to_velo(pts_3d_ref,calib):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return pts_3d_ref @ calib.C2V.T

def project_rect_to_ref(pts_3d_rect, calib):
    ''' Input and Output are nx3 points '''
    return pts_3d_rect @ np.linalg.inv(calib.R0).T
    #return np.transpose(np.dot(np.linalg.inv(calib.R0), np.transpose(pts_3d_rect)))

def project_ref_to_rect(pts_3d_ref, calib):
    ''' Input and Output are nx3 points '''
    return pts_3d_ref @ calib.R0.T

def project_velo_to_rect(pts_3d_velo, calib):
    pts_3d_ref = project_velo_to_ref(pts_3d_velo, calib)
    return project_ref_to_rect(pts_3d_ref, calib)

def project_rect_to_velo(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect, calib)
    return project_ref_to_velo(pts_3d_ref, calib)

# ===========================
# ------- 3d to 2d ----------
# ===========================
def project_rect_to_image(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = pts_3d_rect @ calib.P2.T
    pts_2d[..., 0] /= pts_2d[..., 2]
    pts_2d[..., 1] /= pts_2d[..., 2]
    return pts_2d[..., 0:2]

def project_velo_to_image(pts_3d_velo, calib):
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, calib)
    return project_rect_to_image(pts_3d_rect, calib)

def project_rect_to_right(pts_3d_rect, calib):
    ''' Input: nx3 points in rect camera coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = pts_3d_rect @ calib.P3.T
    pts_2d[..., 0] /= pts_2d[..., 2]
    pts_2d[..., 1] /= pts_2d[..., 2]
    return pts_2d[..., 0:2]

def project_velo_to_right(pts_3d_velo, calib):
    ''' Input: nx3 points in velodyne coord.
        Output: nx2 points in image2 coord.
    '''
    pts_3d_rect = project_velo_to_rect(pts_3d_velo, calib)
    return project_rect_to_right(pts_3d_rect, calib)

# ===========================
# ------- 2d to 3d ----------
# ===========================
def project_image_to_rect(uv_depth, calib):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - calib.c_u) * uv_depth[:, 2]) / calib.f_u + calib.b_x
    y = ((uv_depth[:, 1] - calib.c_v) * uv_depth[:, 2]) / calib.f_v + calib.b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect

def project_image_to_velo(uv_depth, calib):
    pts_3d_rect = project_image_to_rect(uv_depth, calib)
    return project_rect_to_velo(pts_3d_rect, calib)


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = project_velo_to_image(pc_velo[:,:3], calib)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def draw_projected_boxes3d(image, boxes3d, color, thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    boxes3d = boxes3d.astype(np.int32)
    for qs in boxes3d:
        # import random
        # color = random.choice(colors)
        for k in range(0,4):
           # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
           i,j=k,(k+1)%4
           # use LINE_AA for opencv3
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

           i,j=k+4,(k+1)%4 + 4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

           i,j=k,k+4
           cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image

def load_proposals(proosal_file, cls='Car'):
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    d= dict()
    for line in open(proosal_file, 'r'):
        t = line.rstrip().split(" ")
        id = int(os.path.basename(t[0]).rstrip('.png'))
        if det_id2str[int(t[1])] == cls:
            if id in d.keys():
                    d[id].append(np.array([float(t[i]) for i in range(2, 7)],dtype=np.float32))
            else:
                d[id] = [np.array([float(t[i]) for i in range(2, 7)],dtype=np.float32)]
        else:
            continue
    d = {k: np.vstack(v) for k, v in d.items()}
    return d

def draw_lidar(pc, fig=None, color=None, scale=1., axis=True, show=False):
    import mayavi.mlab as mlab
    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    ''' Draw lidar points. simplest set up. '''
    if color is None: color = pc[:, 2]
    # draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=scale,
                  figure=fig)
    if axis:
        # draw origin
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        # draw axis
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[15, 0, 0], distance=50.0, figure=fig)
    if show:
        mlab.show()
    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=False, scores=None, text_scale=(1,1,1), color_list=None, show=False):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    import mayavi.mlab as mlab
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)

        if scores is not None: mlab.text3d(b[4, 0], b[4, 1], b[4, 2], '%02f' % scores[n], scale=(.25, .25, .25), color=color, figure=fig)

        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180, elevation=70, focalpoint=[0, 0, 0], distance=62.0, figure=fig)
    if show:
        mlab.show()
    return fig




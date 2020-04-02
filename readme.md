## SA-SSD: Structure Aware Single-stage 3D Object Detection from Point Cloud (CVPR 2020) [\[paper\]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf)
Currently 1st place in KITTI BEV and 3rd in KITTI 3D. The detector can run at 25 FPS. 

**Authors**: [Chenhang He](https://github.com/skyhehe123), [Zeng Hui](https://github.com/HuiZeng), Jianqiang Huang, Xiansheng Hua, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/).

## Demo Video
[![Demo](https://github.com/skyhehe123/SA-SSD/blob/master/doc/hqdefault.jpg)](https://www.youtube.com/watch?v=jrAb3ts4tAs)

# Introduction
![model](https://github.com/skyhehe123/SA-SSD/blob/master/doc/model.png)
Current single-stage detectors are efficient by progressively downscaling the 3D point clouds in a fully convolutional manner. However, the downscaled features inevitably lose spatial information and cannot make full use of the structure information of 3D point cloud, degrading their localization precision. In this work, we propose to improve the localization precision of single-stage detectors by explicitly leveraging the structure information of 3D point cloud. Specifically, we design an auxiliary network which converts the convolutional features in the backbone network back to point-level representations. The auxiliary network is jointly optimized, by two point-level supervisions, to guide the convolutional features in the backbone network to be aware of the object structure. The auxiliary network can be detached after training and therefore introduces no extra computation in the inference stage. Besides, considering that single-stage detectors suffer from the discordance between the predicted bounding boxes and corresponding classification confidences, we develop an efficient part-sensitive warping operation to align the confidences to the predicted bounding boxes.

# Dependencies
- `python3.5+`
- `pytorch` (tested on 1.1.0)
- `opencv`
- `shapely`
- `mayavi`
- `spconv` (v1.0)

# Installation
1. Clone this repository.
2. Compile C++/CUDA modules in mmdet/ops by running the following command at each directory, e.g.
```bash
$ cd mmdet/ops/points_op
$ python3 setup.py build_ext --inplace
```
3. Setup following Environment variables, you may add them to ~/.bashrc:
```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
export LD_LIBRARY_PATH=/home/billyhe/anaconda3/lib/python3.7/site-packages/spconv;
```

# Data Preparation
1. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Data to download include:
    * Velodyne point clouds (29 GB): input data to VoxelNet
    * Training labels of object data set (5 MB): input label to VoxelNet
    * Camera calibration matrices of object data set (16 MB): for visualization of predictions
    * Left color images of object data set (12 GB): for visualization of predictions

2. Create cropped point cloud and sample pool for data augmentation, please refer to [SECOND](https://github.com/traveller59/second.pytorch).
```bash
$ python3 tools/create_data.py
```

3. Split the training set into training and validation set according to the protocol [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz).
```plain
└── DATA_DIR
       ├── training   <-- training data
       |   ├── image_2
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced
       └── testing  <--- testing data
       |   ├── image_2
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced
```

# Pretrained Model
You can download the pretrained model [here](https://drive.google.com/file/d/1WJnJDMOeNKszdZH3P077wKXcoty7XOUb/view?usp=sharing), 
which is trained on the train split (3712 samples) and evaluated on the val split (3769 samples) and test split (7518 samples). 
The performance (using 40 recall poisitions) on validation set is as follows:
```
Car  AP@0.70, 0.70, 0.70:
bbox AP:99.12, 96.09, 93.61
bev  AP:96.55, 92.79, 90.32
3d   AP:93.13, 84.54, 81.71
```
# Train
To train the SA-SSD with single GPU, run the following command:
```
cd mmdet/tools
python3 train.py ../configs/car_cfg.py
```
To train the SA-SSD with multiple GPUs, run the following command:
```
bash dist_train.sh
```
# Eval
To evaluate the model, run the following command:
```
cd mmdet/tools
python3 test.py ../configs/car_cfg.py ../saved_model_vehicle/epoch_50.pth
```
## Citation
If you find this work useful in your research, please consider cite:
```
@inproceedings{he2020sassd,
title={Structure Aware Single-stage 3D Object Detection from Point Cloud},
author={He, Chenhang and Zeng, Hui and Huang, Jianqiang and Hua, Xian-Sheng and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgement
The code is devloped based on mmdetection, some part of codes are borrowed from SECOND and PointRCNN.
* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)



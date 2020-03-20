import os
import os.path as osp
import shutil
import time

import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from mmdet import datasets
from .class_names import get_classes
import tools.kitti_common as kitti
from mmdet.core.evaluation.kitti_eval import get_official_eval_result
from mmdet.datasets import utils, loader
from .mean_ap import eval_map


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))

        self.interval = interval
        self.lock_dir = None

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        if runner.rank == 0:
            shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            for _ in range(batch_size):
                prog_bar.update()

        if runner.rank == 0:
            print('\n')
            self._barrier(runner.rank, runner.world_size)
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            self._barrier(runner.rank, runner.world_size)
        self._barrier(runner.rank, runner.world_size)

    def evaluate(self):
        raise NotImplementedError


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True

class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            field = '{}_mAP'.format(res_type)
            runner.log_buffer.output[field] = cocoEval.stats[0]
        runner.log_buffer.ready = True
        os.remove(tmp_file)

class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True

class KittiEvalmAPHook(Hook):
    def __init__(self, dataset, interval=5):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = utils.get_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=4, num_workers=2, shuffle=False, \
                                                      collate_fn=utils.merge_second_batch)
        self.interval = interval
        self.lock_dir = None

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        # self.det_dir = osp.join(runner.work_dir, '.det_2')
        # if osp.exists(self.det_dir):
        #     shutil.rmtree(self.det_dir)
        # mmcv.mkdir_or_exist(self.det_dir)

        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        #shutil.rmtree(self.det_dir)
        if runner.rank == 0:
            shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if runner.rank == 0:
            runner.model.eval()
            prog_bar = mmcv.ProgressBar(len(self.dataset))
            class_names = get_classes('kitti')
            results = []
            for i, data in enumerate(self.dataloader):
                # compute output
                with torch.no_grad():
                    result = runner.model(
                        return_loss=False, **data)

                image_shape = (375, 1242)

                for re in result:
                    img_idx = re['image_idx']
                    if re['bbox'] is not None:
                        box2d = re['bbox']
                        box3d = re['box3d_camera']
                        labels = re['label_preds']
                        scores = re['scores']
                        alphas = re['alphas']

                        anno = kitti.get_start_result_anno()
                        num_example = 0
                        for bbox2d, bbox3d, label, score, alpha in zip(box2d, box3d, labels, scores, alphas):

                            if bbox2d[0] > image_shape[1] or bbox2d[1] > image_shape[0]:
                                continue
                            if bbox2d[2] < 0 or bbox2d[3] < 0:
                                continue

                            bbox2d[2:] = np.minimum(bbox2d[2:], image_shape[::-1])
                            bbox2d[:2] = np.maximum(bbox2d[:2], [0, 0])

                            anno["name"].append(class_names[label])
                            anno["truncated"].append(0.0)
                            anno["occluded"].append(0)
                            #anno["alpha"].append(-10)
                            anno["alpha"].append(alpha)
                            anno["bbox"].append(bbox2d)

                            #anno["dimensions"].append(np.array([-1,-1,-1]))
                            anno["dimensions"].append(bbox3d[[3,4,5]])
                            #anno["location"].append(np.array([-1000,-1000,-1000]))
                            anno["location"].append(bbox3d[:3])
                            #anno["rotation_y"].append(-10)
                            anno["rotation_y"].append(bbox3d[6])

                            anno["score"].append(score)
                            num_example += 1
                        if num_example != 0:
                            anno = {n: np.stack(v) for n, v in anno.items()}
                            results.append(anno)
                        else:
                            results.append(kitti.empty_result_anno())
                    else:
                        results.append(kitti.empty_result_anno())

                    num_example = results[-1]["name"].shape[0]
                    results[-1]["image_idx"] = np.array(
                        [img_idx] * num_example, dtype=np.int64)


                batch_size = len(data['sample_idx'])
                for _ in range(batch_size):
                    prog_bar.update()

            self._barrier(runner.rank, runner.world_size)
            self.evaluate(runner, results)
        else:
            self._barrier(runner.rank, runner.world_size)

    def evaluate(self, runner, results):

        gt_annos = kitti.get_label_annos(self.dataset.label_prefix, self.dataset.sample_ids)

        result = get_official_eval_result(gt_annos, results, current_classes=0)

        runner.logger.info(result)

        runner.log_buffer.ready = True

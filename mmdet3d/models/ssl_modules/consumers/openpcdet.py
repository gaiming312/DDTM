import numpy as np

import torch
from torch.nn import functional as F


from mmdet.core.bbox import build_assigner
from mmdet.models import FocalLoss, MSELoss, build_loss
from mmseg.core import add_prefix
from mmdet3d.core import LiDARInstance3DBoxes, bbox3d2result
import re
import time

from ...builder import SSL_MODULES
from ..bbox_utils import filter_by_nms_2d
from ..utils import mlvl_get, mlvl_getattr

@SSL_MODULES.register_module
class Opd_SimpleTest_3D():
    def __init__(self,
                 ssl_obj_attr='teacher',
                 batch_dict_key='tea',
                 out_bboxes_key='3d_simple_test'):

        self.ssl_obj_attr = ssl_obj_attr
        self.batch_dict_key = batch_dict_key
        self.out_bboxes_key = out_bboxes_key

    def forward(self, ssl_obj, batch_dict):
        """
        Args:
            ssl_obj: a pointer to the main SSL class object
            batch_dict:
        Returns: Additions:
            batch_dict:
                [batch_dict_key]:
                    [bboxes_key]: list[tuple(bboxes, scores)]
                        bboxes is bbox class N x 7
                        scores is tensor N x num_classes

        Should not be NMS'ed or anything
        """
        detector = mlvl_getattr(ssl_obj, self.ssl_obj_attr)
        curr_batch_dict = mlvl_get(batch_dict, self.batch_dict_key)

        points = curr_batch_dict['points']
        img_metas = curr_batch_dict['img_metas']

        #### Basically copy openpcdet.py 
        openpcdet_batch = dict()

        ### Voxelize
        voxels, num_points, coors = detector.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        openpcdet_batch['batch_size'] = batch_size
        openpcdet_batch['voxels'] = voxels
        openpcdet_batch['voxel_num_points'] = num_points
        openpcdet_batch['voxel_coords'] = coors

        ### Pad points
        points_batch = []
        for k in range(batch_size):
            points_pad = F.pad(points[k], (1, 0), mode='constant', value=k)
            points_batch.append(points_pad)
        points_batch = torch.cat(points_batch, dim=0)
        openpcdet_batch['points'] = points_batch

        ### Extras
        # print(img_metas)
        openpcdet_batch['frame_id'] = np.array(
            [tmp['sample_idx'] for tmp in img_metas])

        ### Send to OpenPCDet
        pred_dicts, _ = detector.model(openpcdet_batch)
        
        #### Now, format as list[tuple(bboxes, scores)] to be returned by this class
        res = []
        for pred_dict in pred_dicts:
            tmp = pred_dict['pred_boxes'][:, 3].clone()
            pred_dict['pred_boxes'][:, 3] = pred_dict['pred_boxes'][:, 4]
            pred_dict['pred_boxes'][:, 4] = tmp

            pred_dict['pred_boxes'][:, 6] = (
                -pred_dict['pred_boxes'][:, 6] - np.pi / 2)
            # just to deal with the yaw hack in mm3d that openpcdet doesnt have
            # pred_dict['pred_boxes'][:, 6] += np.pi

            pred_dict['pred_boxes'] = LiDARInstance3DBoxes(
                pred_dict['pred_boxes'],
                origin=(0.5, 0.5, 0.5))

            assert len(pred_dict['pred_boxes'].tensor) == len(pred_dict['pred_sem_scores_full'])

            res.append((pred_dict['pred_boxes'], pred_dict['pred_sem_scores_full']))
        
        # Should already be sigmoid'ed. Checked that it is indeed, in detector3d_template post_process
        curr_batch_dict[self.out_bboxes_key] = res

        return batch_dict

@SSL_MODULES.register_module
class Opd_HardPseudoLabel_3D():
    """
    This might be generalizable to non-opd - I'm just pulling out box targets and 
    running forward_train after all.
    
    Main thing is - in addition to regular forward, get the actual boxes.
    This is to be run on student on labeled samples
    """

    def __init__(self,
                 score_thr,
                 cls_includes_bg_pred=False, # I believe is false for Opd models
                 loss_detach_keys=[],
                 ssl_obj_attr='student',
                 target_bboxes_key='tea.placeholder',
                 target_batch_dict_key='stu',
                 name='hard_pseudo_3d',
                 weight=1,
                 out_bboxes_key=None, # if this is None, don't return boxes
                 no_nms=True,
                 box_dim=7):
        
        self.score_thr = score_thr
        self.cls_includes_bg_pred = cls_includes_bg_pred
        self.loss_detach_keys = loss_detach_keys
        assert len(loss_detach_keys) == 0, "Not supported yet, requires changes elsewhere."
        self.ssl_obj_attr = ssl_obj_attr
        self.target_bboxes_key = target_bboxes_key
        self.target_batch_dict_key = target_batch_dict_key
        self.name = name
        self.weight = weight
        self.out_bboxes_key = out_bboxes_key
        self.no_nms = no_nms
        self.iou_key = "iou_th"

        self.box_dim = box_dim

    def forward(self, ssl_obj, batch_dict):
        detector = mlvl_getattr(ssl_obj, self.ssl_obj_attr)
        curr_batch_dict = mlvl_get(batch_dict, self.target_batch_dict_key)


        pseudo_labels = []
        pseudo_bboxes = []
        for batch_idx, (curr_bboxes, curr_scores) in \
                enumerate(mlvl_get(batch_dict, self.target_bboxes_key)):

            if len(curr_scores) == 0:
                curr_labels = curr_scores.new_zeros((0, ), dtype=torch.long)
                curr_bboxes = LiDARInstance3DBoxes(curr_scores.new_zeros((0, self.box_dim)))
            else:
                if self.cls_includes_bg_pred:
                    curr_max_scores, curr_labels = \
                        curr_scores[:, :-1].max(dim=1)
                else:
                    curr_max_scores, curr_labels = curr_scores.max(dim=1)
                    
                curr_labels = curr_labels[curr_max_scores > self.score_thr]
                curr_bboxes = curr_bboxes[curr_max_scores > self.score_thr]


        #stu forward inference
        detector.model.eval()

        points = curr_batch_dict['points']
        img_metas = curr_batch_dict['img_metas']

        #### Basically copy openpcdet.py
        openpcdet_batch = dict()

        ### Voxelize
        voxels, num_points, coors = detector.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        openpcdet_batch['batch_size'] = batch_size
        openpcdet_batch['voxels'] = voxels
        openpcdet_batch['voxel_num_points'] = num_points
        openpcdet_batch['voxel_coords'] = coors

        ### Pad points
        points_batch = []
        for k in range(batch_size):
            points_pad = F.pad(points[k], (1, 0), mode='constant', value=k)
            points_batch.append(points_pad)
        points_batch = torch.cat(points_batch, dim=0)
        openpcdet_batch['points'] = points_batch

        ### Extras
        # print(img_metas)
        openpcdet_batch['frame_id'] = np.array(
            [tmp['sample_idx'] for tmp in img_metas])

        ### Send to OpenPCDet
        pred_dicts, _ = detector.model(openpcdet_batch)
        batch_dict['stu.3d_bboxes_nms'] = pred_dicts
        #get stu boxes with NMS
        res = []
        for pred_dict in pred_dicts:
            tmp = pred_dict['pred_boxes'][:, 3].clone()
            pred_dict['pred_boxes'][:, 3] = pred_dict['pred_boxes'][:, 4]
            pred_dict['pred_boxes'][:, 4] = tmp

            pred_dict['pred_boxes'][:, 6] = (
                    -pred_dict['pred_boxes'][:, 6] - np.pi / 2)
            # just to deal with the yaw hack in mm3d that openpcdet doesnt have
            # pred_dict['pred_boxes'][:, 6] += np.pi

            pred_dict['pred_boxes'] = LiDARInstance3DBoxes(
                pred_dict['pred_boxes'],
                origin=(0.5, 0.5, 0.5))

            assert len(pred_dict['pred_boxes'].tensor) == len(pred_dict['pred_sem_scores_full'])

            res.append((pred_dict['pred_boxes'], pred_dict['pred_sem_scores_full']))
            #stu boxes MaxScoreFilter
            res_out = []
            cls_includes_bg_pred = False
            for batch_idx, curr_in_bboxes in enumerate(res):
                if cls_includes_bg_pred:
                    scores = curr_in_bboxes[1][:, :-1]
                else:
                    scores = curr_in_bboxes[1]

                if len(scores) == 0:
                    score_mask = scores.new_tensor([]) > 0.1
                else:
                    max_score = scores.max(dim=1)[0]
                    score_mask = max_score > 0.1

                curr_out_bboxes = tuple(
                    [tmp[score_mask] for tmp in curr_in_bboxes])

            res_out.append(curr_out_bboxes)

            for batch_idx, (pred_boxes, pred_sem_scores_full,) in \
                enumerate(res_out):

                if len(pred_sem_scores_full) == 0:
                    pred_labels = pred_sem_scores_full.new_zeros((0,), dtype=torch.long)
                    pred_boxes = LiDARInstance3DBoxes(pred_sem_scores_full.new_zeros((0, self.box_dim)))
                else:
                    if self.cls_includes_bg_pred:
                        pred_max_scores, pred_labels = \
                            pred_sem_scores_full[:, :-1].max(dim=1)
                    else:
                        pred_max_scores, pred_labels = pred_sem_scores_full.max(dim=1)
        #matching box
        stu_boxes_tensor = pred_boxes
        tea_boxes_tensor = curr_bboxes
        matching_indices = []
        iou_threshold = mlvl_get(batch_dict, self.iou_key)

        for i in range(len(stu_boxes_tensor)):
            for j in range(len(tea_boxes_tensor)):
                iou = compute_3d_iou(stu_boxes_tensor.tensor[i], tea_boxes_tensor.tensor[j])
                if iou > iou_threshold:
                    matching_indices.append((i, j))


        if len(matching_indices) == 0:
            pseudo_labels.append(curr_labels)
            pseudo_bboxes.append(curr_bboxes)
        else:
            for idx in matching_indices:
                stu_idx, tea_idx = idx

                stu_scores = pred_max_scores[stu_idx].item()
                tea_scores = curr_max_scores[tea_idx].item()
                tea_boxes = curr_bboxes
                stu_boxes = pred_boxes.detach()
                tea_labels = curr_labels
                stu_labels = pred_labels

                if stu_scores > tea_scores :
                    tea_boxes.tensor[tea_idx] = stu_boxes.tensor[stu_idx]
                    tea_labels[tea_idx] = stu_labels[stu_idx]

            pseudo_labels.append(tea_labels)
            pseudo_bboxes.append(tea_boxes)

        detector.model.train()

        openpcdet_batch = detector.train_to_openpcdet(
            curr_batch_dict['points'],
            curr_batch_dict['img_metas'],
            pseudo_bboxes,
            pseudo_labels)

        # From Spconv-OpenPCDet/pcdet/models/detectors/pv_rcnn.py
        for cur_module in detector.model.module_list:
            openpcdet_batch = cur_module(openpcdet_batch)


        # First deal with loss
        loss, _, _ = detector.model.get_training_loss()

        losses = dict(loss=loss.mean())
        losses = add_prefix(losses, self.name)
        batch_dict['ssl_losses'] = \
            ssl_obj._sum_update_losses(batch_dict['ssl_losses'], losses)

        # Then, optionally get the boxes as well.
        if self.out_bboxes_key is not None:
            pred_dicts, _ = detector.model.post_processing(openpcdet_batch, no_nms=self.no_nms)

            # from mmdetection3d/mmdet3d/models/detectors/openpcdet.py simple_test.py opd -> mm3d box conversion
            # Also done right upstairs by simple_test
            #### Now, format as list[tuple(bboxes, scores)]
            res = []
            for pred_dict in pred_dicts:
                tmp = pred_dict['pred_boxes'][:, 3].clone()
                pred_dict['pred_boxes'][:, 3] = pred_dict['pred_boxes'][:, 4]
                pred_dict['pred_boxes'][:, 4] = tmp

                pred_dict['pred_boxes'][:, 6] = (
                    -pred_dict['pred_boxes'][:, 6] - np.pi / 2)
                # just to deal with the yaw hack in mm3d that openpcdet doesnt have
                # pred_dict['pred_boxes'][:, 6] += np.pi

                pred_dict['pred_boxes'] = LiDARInstance3DBoxes(
                    pred_dict['pred_boxes'],
                    origin=(0.5, 0.5, 0.5))
                    
                assert len(pred_dict['pred_boxes'].tensor) == len(pred_dict['pred_sem_scores_full'])

                res.append((pred_dict['pred_boxes'], pred_dict['pred_sem_scores_full']))

            curr_batch_dict[self.out_bboxes_key] = res

        return batch_dict





@SSL_MODULES.register_module
class Opd_Supervised_3D():
    """
    Simple forward train.
    """

    def __init__(self,
                 ssl_obj_attr='student',
                 batch_dict_key='stu',
                 name='sup_3d',
                 weight=1):

        self.ssl_obj_attr = ssl_obj_attr
        self.batch_dict_key = batch_dict_key
        self.name = name
        self.weight = weight

    def forward(self, ssl_obj, batch_dict):
        """
        
        """
        detector = mlvl_getattr(ssl_obj, self.ssl_obj_attr)
        curr_batch_dict = mlvl_get(batch_dict, self.batch_dict_key)

        losses = detector.forward_train(
            curr_batch_dict['points'],
            curr_batch_dict['img_metas'],
            curr_batch_dict['gt_bboxes_3d'],
            curr_batch_dict['gt_labels_3d'],
            curr_batch_dict.get('gt_bboxes_ignore', None))

        if self.weight != 1:
            for k in list(losses.keys()):
                losses[k] = losses[k] * self.weight

        losses = add_prefix(losses, self.name)
        if 'sup_losses' in batch_dict:
            batch_dict['sup_losses'] = \
                ssl_obj._sum_update_losses(batch_dict['sup_losses'], losses)
        else:
            batch_dict['ssl_losses'] = \
                ssl_obj._sum_update_losses(batch_dict['ssl_losses'], losses)

        return batch_dict

def compute_3d_iou(box1, box2):
    # 计算两个box的交集体积
    inter_vol = compute_intersection_volume(box1,box2)  # 这里你需要计算box1和box2的交集体积

    # 计算每个box的体积
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]

    # 计算两个box的并集体积
    union_vol = vol1 + vol2 - inter_vol

    # 计算3D IoU
    iou = inter_vol / union_vol
    return iou


def compute_intersection_volume(box1, box2):
    # 计算两个框在三个维度上的重叠长度
    dx = min(box1[0] + 0.5 * box1[3], box2[0] + 0.5 * box2[3]) - max(box1[0] - 0.5 * box1[3], box2[0] - 0.5 * box2[3])
    dy = min(box1[1] + 0.5 * box1[4], box2[1] + 0.5 * box2[4]) - max(box1[1] - 0.5 * box1[4], box2[1] - 0.5 * box2[4])
    dz = min(box1[2] + 0.5 * box1[5], box2[2] + 0.5 * box2[5]) - max(box1[2] - 0.5 * box1[5], box2[2] - 0.5 * box2[5])

    # 如果有任何维度上的重叠长度小于等于零，则交集体积为0
    if (dx <= 0) or (dy <= 0) or (dz <= 0):
        return 0

    # 计算交集体积
    intersection_volume = dx * dy * dz
    return intersection_volume



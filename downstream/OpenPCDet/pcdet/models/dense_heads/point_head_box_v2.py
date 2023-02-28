import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate

import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate

from ...utils import common_utils, loss_utils

import torch.nn.functional as F


class PointHeadBoxV2(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class + 1 # for the background class (default index: 0)
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SoftmaxFocalClassificationLoss(gamma=2.0)
        )
        self.p2_supervision = losses_cfg.get(
            "P2_SUPERVISION", None)

        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class + 1)

        if tb_dict is None:
            tb_dict = {}
        if self.p2_supervision is not None:
            raise NotImplementedError()
            p2_score = self.forward_ret_dict['p2_score']
            ignore_mask = (p2_score >= self.p2_supervision.FG_THRESHOLD) & \
                (p2_score < self.p2_supervision.BG_THRESHOLD)

            preserved_mask = torch.logical_not(ignore_mask)
            point_cls_labels = point_cls_labels[preserved_mask]
            point_cls_preds = point_cls_preds[preserved_mask]
            p2_score = p2_score[preserved_mask]

            point_cls_labels[
                (p2_score >= self.p2_supervision.BG_THRESHOLD) & \
                    (point_cls_labels >= 0)
                    ] = 0
            one_hot_targets = point_cls_preds.new_zeros(
                *list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            positives = (p2_score < self.p2_supervision.FG_THRESHOLD) & \
                (point_cls_labels >= 0)
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights = torch.ones_like(p2_score)
            cls_weights[point_cls_labels < 0] = 0
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_loss_src = self.cls_loss_func(
                point_cls_preds, one_hot_targets, weights=cls_weights)
            fg_exclude_mask = (p2_score < self.p2_supervision.FG_THRESHOLD) & \
                (point_cls_labels == 0)
            cls_loss_src = cls_loss_src[torch.logical_not(fg_exclude_mask)]

            point_loss_cls = cls_loss_src.sum()

            # points with low p2 score and no pseudolabel
            point_loss_cls_margin_p2 = torch.clamp((self.p2_supervision.CLASS_THRESHOLD - torch.sigmoid(point_cls_preds[fg_exclude_mask]).max(dim=1)[0]), min=0).sum() / torch.clamp(pos_normalizer, min=1.0) * self.p2_supervision.MARGIN_WEIGHT

            tb_dict.update({
                'ignored_cls_num': ignore_mask.sum().item(),
                'extra_fg_cls_num': fg_exclude_mask.sum().item()
            })
            for _c in range(self.num_class):
                tb_dict.update({
                    f'class_{_c+1}_pos': (point_cls_labels == (_c + 1)).sum().item(),
                })
        else:
            target_idx = point_cls_labels >= 0 # remove the ignored points 
            point_cls_labels = point_cls_labels[target_idx]
            point_cls_preds = point_cls_preds[target_idx]

            cls_weights = torch.ones_like(point_cls_labels)
            positives = (point_cls_labels > 0).float()
            pos_normalizer = positives.sum(dim=0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = F.one_hot(point_cls_labels, num_classes=self.num_class + 1)
            cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()
            point_loss_cls_margin_p2 = torch.zeros_like(point_loss_cls)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        point_loss_cls_margin_p2 = point_loss_cls_margin_p2 * loss_weights_dict['point_cls_weight']

        total_loss = point_loss_cls + point_loss_cls_margin_p2
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_loss_cls_margin_p2': point_loss_cls_margin_p2.item(),
            'point_total_loss': total_loss.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return total_loss, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class + 1)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (M, num_class)
            point_box_preds: (M, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        foreground_ind = (pred_classes > 0)
        point_box_preds = self.box_coder.decode_torch(point_box_preds[foreground_ind], 
                                    points[foreground_ind], pred_classes[foreground_ind])

        return point_cls_preds[foreground_ind, 1:], point_box_preds

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class + 1)
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = F.softmax(point_cls_preds_max, dim=-1)

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            if 'p2_score' in batch_dict:
                ret_dict['p2_score'] = batch_dict['p2_score']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = F.softmax(point_cls_preds, dim=-1)
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = True

        self.forward_ret_dict = ret_dict
        return batch_dict

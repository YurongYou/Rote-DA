import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        if self.p2_supervision:
            feature_map_stride = self.model_cfg.ANCHOR_GENERATOR_CONFIG[0]['feature_map_stride']
            align_center = self.model_cfg.ANCHOR_GENERATOR_CONFIG[0]['align_center']
            self.feature_map_size = grid_size[:2] // feature_map_stride
            self.feature_map_size = self.feature_map_size.tolist()
            self.point_cloud_range = point_cloud_range
            if align_center:
                self.x_stride = (point_cloud_range[3] - point_cloud_range[0]) / self.feature_map_size[0]
                self.y_stride = (point_cloud_range[4] - point_cloud_range[1]) / self.feature_map_size[0]
                x_offset, y_offset = self.x_stride / 2, self.y_stride / 2
            else:
                self.x_stride = (point_cloud_range[3] - point_cloud_range[0]) / (self.feature_map_size[0] - 1)
                self.y_stride = (point_cloud_range[4] - point_cloud_range[1]) / (self.feature_map_size[1] - 1)
                x_offset, y_offset = 0, 0
            x_shifts = torch.arange(
                point_cloud_range[0] + x_offset, point_cloud_range[3] + 1e-5, step=self.x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                point_cloud_range[1] + y_offset, point_cloud_range[4] + 1e-5, step=self.y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(0)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            self.feature_map_grid = torch.stack(
                (x_shifts, y_shifts, z_shifts), dim=-1).squeeze(2)
            # print(self.feature_map_grid.shape)
            self.feature_map_grid = self.feature_map_grid.permute(1, 0, 2).contiguous()  # [y, x, 3]

    def generate_gt_fg_mask(self, gt_boxes, extra_width=(1., 1., 0)):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        batch_size = gt_boxes.shape[0]
        gt_boxes = gt_boxes[:, :, :-1]
        # print(gt_boxes.shape)
        masks = []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            fg_mask = torch.zeros(self.feature_map_grid.shape[:2],
                                  dtype=torch.bool,
                                device=gt_boxes.device)
            if len(cur_gt) > 0:
                cur_gt = cur_gt.clone()
                # make the boxes z on 0
                cur_gt[:, 2] = 0.
                cur_gt[:, 4] = 1.
                cur_gt = box_utils.enlarge_box3d(cur_gt, extra_width=extra_width)
                box_idxs_of_pts = points_in_boxes_gpu(
                    self.feature_map_grid.view(-1, 3).unsqueeze(dim=0),
                    cur_gt.unsqueeze(dim=0)
                    )
                fg_mask = box_idxs_of_pts >= 0
                fg_mask = fg_mask.view(self.feature_map_grid.shape[:2])
            masks.append(fg_mask)
        return torch.stack(masks)


    def generate_p2_sup_mask(self, p2_score, points, batch_size):
        with torch.no_grad():
            assert p2_score.shape[0] == points.shape[0]
            masks = []
            BG_THRESHOLD = self.p2_supervision.BG_THRESHOLD
            FG_THRESHOLD = self.p2_supervision.FG_THRESHOLD
            for k in range(batch_size):
                mask = torch.ones(self.feature_map_size, dtype=torch.int32,
                                device=points.device) * -1
                batch_mask = (points[:, 0] == k)
                points_k = points[batch_mask]
                p2_score_k = p2_score[batch_mask]
                points_k_x_coord = (
                    points_k[:, 1] - self.point_cloud_range[0]) / self.x_stride
                points_k_y_coord = (
                    points_k[:, 2] - self.point_cloud_range[1]) / self.y_stride
                points_k_x_coord = points_k_x_coord.long()
                points_k_y_coord = points_k_y_coord.long()
                # points_k_x_coord = torch.clamp(
                #     points_k_x_coord, min=0, max=self.feature_map_size[0]-1)
                # points_k_y_coord = torch.clamp(
                #     points_k_y_coord, min=0, max=self.feature_map_size[1]-1)

                bg_mask = p2_score_k >= BG_THRESHOLD
                ignore_mask = (p2_score_k >= FG_THRESHOLD) & (p2_score_k < BG_THRESHOLD)
                fg_mask = (p2_score_k < FG_THRESHOLD) & (p2_score_k >= 0)
                mask[points_k_x_coord[ignore_mask], points_k_y_coord[ignore_mask]] = -2
                mask[points_k_x_coord[bg_mask], points_k_y_coord[bg_mask]] = 1
                mask[points_k_x_coord[fg_mask], points_k_y_coord[fg_mask]] = 0
                # the anchors are in (1, y, x), so we need to transpose
                mask = mask.permute(1, 0).contiguous()
                masks.append(mask)
            return torch.stack(masks)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.p2_supervision = losses_cfg.get(
            "P2_SUPERVISION", None)
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        tb_dict = {}
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])

        if self.num_class == 1:
            # class agnostic
            box_cls_labels[box_cls_labels > 0] = 1
        if self.p2_supervision:
            box_cls_labels_fg_mask = self.generate_gt_fg_mask(self.forward_ret_dict['gt_boxes'])
            fg_target_template = torch.zeros((1, 6, 3), dtype=cls_preds.dtype,
                                             device=box_cls_labels.device)
            fg_target_template[:, 0:2, 0] = 1.
            fg_target_template[:, 2:4, 1] = 1.
            fg_target_template[:, 4:6, 2] = 1.
            p2_sup_mask = self.generate_p2_sup_mask(
                self.forward_ret_dict['p2_score'],
                self.forward_ret_dict['points'],
                batch_size=batch_size)
            # torch.save((self.forward_ret_dict['box_cls_labels'],
            #             self.forward_ret_dict['p2_score'],
            #             self.forward_ret_dict['points'],
            #             box_cls_labels_fg_mask,
            #             self.feature_map_size,
            #             self.point_cloud_range),
            #            "/home/yy785/tmp/anchor_vis.pth.tar")
            # assert False
            assert p2_sup_mask.shape[1:3] == cls_preds.shape[1:3], f"{p2_sup_mask.shape}, {cls_preds.shape}"
            p2_sup_mask_src = p2_sup_mask  # (B, y, x)
            # make p2_sup_mask into (B, y, x, 1)
            p2_sup_mask = p2_sup_mask.unsqueeze(-1)
            # box_cls_labels is in shape (B, y*x*num_anchors)
            # we make box_cls_labels into (B, y, x, num_anchors)
            box_cls_labels = box_cls_labels.view(*p2_sup_mask.shape[:-1], -1)
            # it can be 0 (clear background),
            # -1 (don't care, e.g. some overlap w/ gt but not large)
            # 1-3 (class labels)

            # make p2_sup_mask into (B, y, x, num_anchors)
            p2_sup_mask = p2_sup_mask.repeat(1, 1, 1, box_cls_labels.shape[-1])

            # first change the cls_labels with FG indicator
            tb_dict.update({
                'extra_bg_cls_num': ((box_cls_labels > 0) & (p2_sup_mask == 1)).sum().item(),
            })
            box_cls_labels[box_cls_labels_fg_mask & (p2_sup_mask_src == 1)] = 0

            # ignore ambiguous ones
            tb_dict.update({
                'extra_ignore_cls_num': ((box_cls_labels >= 0) & (p2_sup_mask == -2)).sum().item(),
            })
            box_cls_labels[torch.logical_not(box_cls_labels_fg_mask) & (p2_sup_mask_src == -2)] = -1

            cls_targets = box_cls_labels
            # to avoid -1 in scattering
            cls_targets = cls_targets * (cls_targets >= 0).type_as(cls_targets)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1,
                                     cls_targets.unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            fg_exclude_mask = torch.logical_not(box_cls_labels_fg_mask) & (p2_sup_mask_src == 0)
            # TODO: remove the dirty brute-force here
            if not self.p2_supervision.margin_loss:
                one_hot_targets[fg_exclude_mask] = fg_target_template

            one_hot_targets = one_hot_targets.view(batch_size, -1, self.num_class)
            fg_exclude_mask = fg_exclude_mask.unsqueeze(-1).repeat(
                1, 1, 1, box_cls_labels.shape[-1])
            fg_exclude_mask = fg_exclude_mask.view(batch_size, -1)
            # print(f"{fg_exclude_mask.shape}, {fg_exclude_mask.sum()}")

            tb_dict.update({
                'ignored_cls_num': (box_cls_labels == -1).sum().item(),
                'fg_cls_num': torch.any(cls_targets > 0, dim=-1).sum(),
                'extra_fg_cls_num': fg_exclude_mask.sum().item(),
            })

            # positives = one_hot_targets.sum()
            cls_weights = torch.ones_like(box_cls_labels).float()
            cls_weights[box_cls_labels < 0] = 0
            cls_weights = cls_weights.view(batch_size, -1)
            # positives = positives.view(batch_size, -1)
            # pos_normalizer = positives.sum(1, keepdim=True).float()
            pos_normalizer = one_hot_targets.sum((1, 2)).float().unsqueeze(-1)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            # make cls_preds into (B, y*x*num_anchors, C)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            cls_loss_src = self.cls_loss_func(
                cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src[torch.logical_not(
                    fg_exclude_mask)].sum() / batch_size
            if not self.p2_supervision.margin_loss:
                cls_loss_margin_p2 = cls_loss_src[fg_exclude_mask].sum(
                ) * self.p2_supervision.MARGIN_WEIGHT / batch_size
            else:
                cls_preds = cls_preds.view(*p2_sup_mask_src.shape, -1)
                fg_exclude_map = torch.logical_not(box_cls_labels_fg_mask) & (p2_sup_mask_src == 0)
                if self.p2_supervision.get('temperature', None):
                    log_softmax = F.log_softmax(cls_preds[fg_exclude_map] / self.p2_supervision.temperature) # N * 18
                    softmax = log_softmax.exp()
                    entropy = -softmax * log_softmax
                    cls_loss_margin_p2 = entropy.sum() / batch_size / torch.clamp(pos_normalizer, min=1.0).sum() * self.p2_supervision.MARGIN_WEIGHT
                else:
                    cls_loss_margin_p2 = torch.clamp((self.p2_supervision.CLASS_THRESHOLD - torch.sigmoid(cls_preds[fg_exclude_map]).max(
                        dim=1)[0]), min=0).sum() / batch_size / torch.clamp(pos_normalizer, min=1.0).sum() * self.p2_supervision.MARGIN_WEIGHT
        else:
            cared = box_cls_labels >= 0  # [N, num_anchors]
            positives = box_cls_labels > 0
            negatives = box_cls_labels == 0
            negative_cls_weights = negatives * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            reg_weights = positives.float()

            pos_normalizer = positives.sum(1, keepdim=True).float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
            cls_targets = cls_targets.unsqueeze(dim=-1)

            cls_targets = cls_targets.squeeze(dim=-1)
            one_hot_targets = torch.zeros(
                *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
            )
            one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
            one_hot_targets = one_hot_targets[..., 1:]
            cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size
            cls_loss_margin_p2 = torch.zeros_like(cls_loss)

        total_loss = (cls_loss + cls_loss_margin_p2) * \
            self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict.update({
            'total_loss': total_loss.item(),
            'rpn_loss_cls': cls_loss.item(),
            'rpn_loss_cls_margin_p2': cls_loss_margin_p2.item(),
            'pos_normalizer': pos_normalizer.sum().item(),
        })
        return total_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError

from torch import nn
import torch

from .utils import matching_strategy, matching_strategy_quads
from ....._utils import _check_norm, _check_ins
from ....utils.codec import CodecBase, EncoderBase, DecoderBase

import torchvision


class Codec(CodecBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means
        self.norm_stds = norm_stds

        super().__init__(Encoder(self.norm_means, self.norm_stds), Decoder(self.norm_means, self.norm_stds))


class Encoder(EncoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)


    def forward(self, targets, dboxes, batch_num):
        """
        :param targets: Tensor, shape is (batch*object num(batch), 1+4+class_labels)
        :param dboxes: Tensor, shape is (total_dbox_nums, 4=(cx,cy,w,h))
        :param batch_num: int
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            encoded_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                           gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                           gt_w = train(gt_w / dbox_w), gt_h = train(gt_h / dbox_h)
                           shape = (batch, default boxes num, 4)
        """
        # matching
        # pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        # targets: Tensor, shape = (batch, default box num, 4+class_num) including background
        pos_indicator, targets = matching_strategy(targets, dboxes, batch_num=batch_num)

        # encoding
        # targets_loc: Tensor, shape = (batch, default boxes num, 4)
        targets_loc = targets[:, :, :4]

        assert targets_loc.shape[1:] == dboxes.shape, "targets_loc and default_boxes must be same shape"

        gt_cx = (targets_loc[:, :, 0] - dboxes[:, 0]) / dboxes[:, 2]
        gt_cy = (targets_loc[:, :, 1] - dboxes[:, 1]) / dboxes[:, 3]
        gt_w = torch.log(targets_loc[:, :, 2] / dboxes[:, 2])
        gt_h = torch.log(targets_loc[:, :, 3] / dboxes[:, 3])

        encoded_boxes = torch.cat((gt_cx.unsqueeze(2),
                                   gt_cy.unsqueeze(2),
                                   gt_w.unsqueeze(2),
                                   gt_h.unsqueeze(2)), dim=2)

        # normalization
        targets[:, :, :4] = (encoded_boxes - self.norm_means) / self.norm_stds

        return pos_indicator, targets

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)


class Decoder(DecoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0), norm_stds=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)

    def forward(self, predicts, default_boxes):
        """
        Opposite to above procession
        :param predicts: Tensor, shape = (batch, default boxes num, 4 + class_nums)
        :param default_boxes: Tensor, shape = (default boxes num, 4)
        Note that 4 means (cx, cy, w, h)
        :return:
            inf_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                      inf_cx = pred_cx * dbox_w + dbox_cx, inf_cy = pred_cy * dbox_h + dbox_cy,
                      inf_w = exp(pred_w) * dbox_w, inf_h = exp(pred_h) * dbox_h
                      shape = (batch, default boxes num, 4)
        """
        pred_locs = predicts[:, :, :4]

        assert pred_locs.shape[1:] == default_boxes.shape, "predicts and default_boxes must be same shape"

        pred_unnormalized = pred_locs * self.norm_stds + self.norm_means

        inf_cx = pred_unnormalized[:, :, 0] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_cy = pred_unnormalized[:, :, 1] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_w = torch.exp(pred_unnormalized[:, :, 2]) * default_boxes[:, 2]
        inf_h = torch.exp(pred_unnormalized[:, :, 3]) * default_boxes[:, 3]

        predicts[:, :, :4] = torch.cat((inf_cx.unsqueeze(2),
                                        inf_cy.unsqueeze(2),
                                        inf_w.unsqueeze(2),
                                        inf_h.unsqueeze(2)), dim=2)

        return predicts

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)


class TextBoxCodec(CodecBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0),
                       norm_stds=(0.1, 0.1, 0.2, 0.2,
                                  0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1, 0.1, 0.1)):
        # shape = (1, 1, 12=(cx, cy, w, h, x1, y1,...)) or (1, 1, 1)
        self.norm_means = norm_means
        self.norm_stds = norm_stds

        super().__init__(TextBoxEncoder(self.norm_means, self.norm_stds),
                         TextBoxDecoder(self.norm_means, self.norm_stds))


class TextBoxEncoder(EncoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0),
                       norm_stds=(0.1, 0.1, 0.2, 0.2,
                                  0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1, 0.1, 0.1)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 12=(cx, cy, w, h, x1, y1,...)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)


    def forward(self, targets, dboxes, batch_num):
        """
        :param targets: Tensor, shape is (batch*object num(batch), 4=(cx,cy,w,h)+8=(x1,y1,x2,y2,...)+class_labels)
        :param dboxes: Tensor, shape is (total_dbox_nums, 4=(cx,cy,w,h))
        :param batch_num: int
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            encoded_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                           gt_cx = (gt_cx - dbox_cx)/dbox_w, gt_cy = (gt_cy - dbox_cy)/dbox_h,
                           gt_w = train(gt_w / dbox_w), gt_h = train(gt_h / dbox_h)
                           shape = (batch, default boxes num, 4)
        """
        # matching
        # pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        # targets: Tensor, shape = (batch, default box num, 12+class_num) including background
        pos_indicator, targets = matching_strategy_quads(targets, dboxes, batch_num=batch_num)

        # encoding
        # targets_loc: Tensor, shape = (batch, default boxes num, 4)
        # targets_quad: Tensor, shape = (batch, default boxes num, 8)
        targets_loc, targets_quad = targets[:, :, :4], targets[:, :, 4:12]

        assert targets_loc.shape[1:] == dboxes.shape, "targets_loc and default_boxes must be same shape"

        # bounding box
        gt_cx = (targets_loc[:, :, 0] - dboxes[:, 0]) / dboxes[:, 2]
        gt_cy = (targets_loc[:, :, 1] - dboxes[:, 1]) / dboxes[:, 3]
        gt_w = torch.log(targets_loc[:, :, 2] / dboxes[:, 2])
        gt_h = torch.log(targets_loc[:, :, 3] / dboxes[:, 3])

        # quad
        gt_x1 = (targets_quad[:, :, 0] - dboxes[:, 0]) / dboxes[:, 2]
        gt_y1 = (targets_quad[:, :, 1] - dboxes[:, 1]) / dboxes[:, 3]
        gt_x2 = (targets_quad[:, :, 2] - dboxes[:, 0]) / dboxes[:, 2]
        gt_y2 = (targets_quad[:, :, 3] - dboxes[:, 1]) / dboxes[:, 3]
        gt_x3 = (targets_quad[:, :, 4] - dboxes[:, 0]) / dboxes[:, 2]
        gt_y3 = (targets_quad[:, :, 5] - dboxes[:, 1]) / dboxes[:, 3]
        gt_x4 = (targets_quad[:, :, 6] - dboxes[:, 0]) / dboxes[:, 2]
        gt_y4 = (targets_quad[:, :, 7] - dboxes[:, 1]) / dboxes[:, 3]


        encoded_boxes = torch.cat((gt_cx.unsqueeze(2), gt_cy.unsqueeze(2), gt_w.unsqueeze(2), gt_h.unsqueeze(2),
                                   gt_x1.unsqueeze(2), gt_y1.unsqueeze(2), gt_x2.unsqueeze(2), gt_y2.unsqueeze(2),
                                   gt_x3.unsqueeze(2), gt_y3.unsqueeze(2), gt_x4.unsqueeze(2), gt_y4.unsqueeze(2)),
                                   dim=2)

        # normalization
        targets[:, :, :12] = (encoded_boxes - self.norm_means) / self.norm_stds

        return pos_indicator, targets

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)


class TextBoxDecoder(DecoderBase):
    def __init__(self, norm_means=(0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0),
                       norm_stds=(0.1, 0.1, 0.2, 0.2,
                                  0.1, 0.1, 0.1, 0.1,
                                  0.1, 0.1, 0.1, 0.1)):
        super().__init__()

        norm_means = _check_norm('norm_means', norm_means)
        norm_stds = _check_norm('norm_stds', norm_stds)

        # shape = (1, 1, 4=(cx, cy, w, h)) or (1, 1, 1)
        self.norm_means = norm_means.unsqueeze(0).unsqueeze(0)
        self.norm_stds = norm_stds.unsqueeze(0).unsqueeze(0)

    def forward(self, predicts, default_boxes):
        """
        Opposite to above procession
        :param predicts: Tensor, shape = (batch, default boxes num, 14=(4+8+2))
        :param default_boxes: Tensor, shape = (default boxes num, 4)
        Note that 4 means (cx, cy, w, h)
        :return:
            inf_boxes: Tensor, calculate ground truth value considering default boxes. The formula is below;
                      inf_cx = pred_cx * dbox_w + dbox_cx, inf_cy = pred_cy * dbox_h + dbox_cy,
                      inf_w = exp(pred_w) * dbox_w, inf_h = exp(pred_h) * dbox_h
                      shape = (batch, default boxes num, 4)
        """
        predicts_locations = predicts[:, :, :12]

        assert predicts_locations.shape[1] == default_boxes.shape[0], "predicts and default_boxes must be same number"

        pred_unnormalized = predicts_locations * self.norm_stds + self.norm_means
        pred_loc, pred_quad = pred_unnormalized[:, :, :4], pred_unnormalized[:, :, 4:12]

        # bounding box
        inf_cx = pred_loc[:, :, 0] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_cy = pred_loc[:, :, 1] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_w = torch.exp(pred_loc[:, :, 2]) * default_boxes[:, 2]
        inf_h = torch.exp(pred_loc[:, :, 3]) * default_boxes[:, 3]

        # quad
        inf_x1 = pred_quad[:, :, 0] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_y1 = pred_quad[:, :, 1] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_x2 = pred_quad[:, :, 2] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_y2 = pred_quad[:, :, 3] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_x3 = pred_quad[:, :, 4] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_y3 = pred_quad[:, :, 5] * default_boxes[:, 3] + default_boxes[:, 1]
        inf_x4 = pred_quad[:, :, 6] * default_boxes[:, 2] + default_boxes[:, 0]
        inf_y4 = pred_quad[:, :, 7] * default_boxes[:, 3] + default_boxes[:, 1]

        predicts[:, :, :12] = torch.cat((inf_cx.unsqueeze(2), inf_cy.unsqueeze(2), inf_w.unsqueeze(2), inf_h.unsqueeze(2),
                                         inf_x1.unsqueeze(2), inf_y1.unsqueeze(2), inf_x2.unsqueeze(2), inf_y2.unsqueeze(2),
                                         inf_x3.unsqueeze(2), inf_y3.unsqueeze(2), inf_x4.unsqueeze(2), inf_y4.unsqueeze(2)),
                                         dim=2)

        return predicts

    def to(self, *args, **kwargs):
        self.norm_means = self.norm_means.to(*args, **kwargs)
        self.norm_stds = self.norm_stds.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.norm_means = self.norm_means.cuda(device)
        self.norm_stds = self.norm_stds.cuda(device)

        return super().cuda(device)
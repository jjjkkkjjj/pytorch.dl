import torch
from torch import nn
from torch.nn import functional as F

from ..data.utils.boxes import dice, dists2corners

class FOTSLoss(nn.Module):
    def __init__(self, theta_coef=10, reg_coef=1, recog_coef=1, blankIndex=0):
        super().__init__()

        self.theta_coef = theta_coef
        self.reg_coef = reg_coef
        self.recog_coef = recog_coef

        self.detn_loss = DetectionLoss(theta_coef, reg_coef)
        self.reg_loss = RegressionLoss(theta_coef)
        self.recog_loss = nn.CTCLoss(blank=blankIndex, reduction='sum', zero_infinity=True)

    def forward(self, detn, recog):
        """
        :param detn:
                pos_indicator: bool Tensor, shape = (b, c, h/4, w/4)
                pred_confs: confidence Tensor, shape = (b, 1, h/4, w/4)
                pred_locs: predicted Tensor, shape = (b, 5=(conf, t, l, b, r, angle), h/4, w/4)
                    distances: distances Tensor, shape = (b, 4=(t, l, b, r), h/4, w/4) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, 1, h/4, w/4)
                true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
        :param recog:
                pred_texts: list(b) of predicted text number Tensor, shape = (times, text nums, class_nums)
                true_texts: list(b) of true text number Tensor, shape = (true text nums, char nums)
                pred_txtlens: list(b) of length Tensor, shape = (text nums)
                true_txtlens: list(b) of true length Tensor, shape = (true text nums)
        :return:
        """
        detn_loss = self.detn_loss(*detn)
        recog_loss = self.recog_loss(*recog)

        return detn_loss.mean() + self.recog_coef * recog_loss.mean()


class DetectionLoss(nn.Module):
    def __init__(self, theta_coef, reg_coef):
        super().__init__()
        self.theta_coef = theta_coef
        self.reg_coef = reg_coef

        self.cls_loss = ClassificationLoss()
        self.reg_loss = RegressionLoss(theta_coef)

    def forward(self, pos_indicator, pred_confs, pred_locs, true_locs):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param pred_locs: predicted Tensor, shape = (b, h/4, w/4, 5=(t, l, b, r, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, l, b, r)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
        :param true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
        :return:
        """
        # convert distances into corners representation
        pred_dists = pred_locs[..., :4]
        pred_boxes = dists2corners(pred_dists)
        pred_angles = pred_locs[..., -1:]

        true_boxes = [t_loc[:, :4] for t_loc in true_locs]
        true_angles = [t_loc[:, -1:] for t_loc in true_locs]

        cls_loss = self.cls_loss(pos_indicator, pred_confs)
        reg_loss = self.reg_loss(pos_indicator, pred_boxes, true_boxes, pred_angles, true_angles)

        return cls_loss + self.reg_coef * reg_loss


class ClassificationLoss(nn.Module):
    def __init__(self, hard_neg_num=512, random_neg_num=512):
        super().__init__()

        self.hard_neg_num = hard_neg_num
        self.random_neg_num = random_neg_num

    def forward(self, pos_indicator, pred_confs):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :return:
        """
        batch_nums, h, w = pos_indicator.shape
        neg_indicator = torch.logical_not(pos_indicator)
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=pos_indicator.device)
        for b in range(batch_nums):
            sample_nums = 0

            # shape = (pos num, 1)
            pos_confs = pred_confs[b][pos_indicator[b]]
            pos_loss = F.binary_cross_entropy(pos_confs, torch.ones_like(pos_confs), reduction='sum')

            # online hard example mining
            # hard negative
            neg_confs = pred_confs[b][neg_indicator[b]]
            _, indices = torch.sort(neg_confs, dim=0, descending=True)
            hardn_confs = neg_confs[indices[:self.hard_neg_num]]
            hardn_loss = F.binary_cross_entropy(hardn_confs, torch.zeros_like(hardn_confs), reduction='sum')

            loss[b] += pos_loss + hardn_loss
            sample_nums += pos_confs.numel() + hardn_confs.numel()
            # random negative
            indices = indices[self.hard_neg_num:]
            if indices.numel() > 0:
                randindices = torch.randperm(indices.numel()).unsqueeze(-1)
                indices = indices[randindices]
                hardnrand_confs = neg_confs[indices[:self.random_neg_num]]
                hardnrand_loss = F.binary_cross_entropy(hardnrand_confs, torch.zeros_like(hardnrand_confs), reduction='sum')

                loss[b] += hardnrand_loss
                sample_nums += hardn_confs.numel()

            loss[b] = loss[b] / sample_nums

        return loss.mean()

class RegressionLoss(nn.Module):
    def __init__(self, theta_coef=10):
        super().__init__()

        self.theta_coef = theta_coef

    def forward(self, pos_indicator, pred_confs, p_boxes, t_boxes, p_theta, t_theta):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param p_boxes: predicted boxes Tensor, shape = (b, h/4, w/4, 4=(xmin, ymin, xmax, ymax))
        :param t_boxes: true boxes list(b) of Tensor, shape = (text numbers, 4=(xmin, ymin, xmax, ymax))
        :param p_theta: predicted angles Tensor, shape = (b, h/4, w/4, 1)
        :param t_theta: true angle list(b) of Tensor, shape = (text numbers, 1)
        :return:
        """
        batch_nums, h, w = pos_indicator.shape
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=pos_indicator.device)
        for b in range(batch_nums):
            sample_nums = 0

            pos_indicator[b]


        loc_loss = dice(p_boxes, t_boxes)
        orient_loss = F.cosine_similarity(p_theta, t_theta, dim=1)
        return loc_loss + self.theta_coef*orient_loss

class RecognitionLoss(nn.Module):
    def forward(self, predicts, targets, predict_lengths, target_lengths):
        """
        :param predicts: list(b) of predicted text number Tensor, shape = (times, text nums, class_nums)
        :param targets: list(b) of true text number Tensor, shape = (true text nums, char nums)
        :param predict_lengths: list(b) of length Tensor, shape = (text nums)
        :param target_lengths: list(b) of true length Tensor, shape = (true text nums)
        :return:
        """
        batch_nums = len(predicts)
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=predicts[0].device)
        for p, t, p_l, t_l in zip(predicts, targets, predict_lengths, target_lengths):
            loss += [F.ctc_loss(p, t, p_l, t_l, reduction='mean')]

        return loss.mean()
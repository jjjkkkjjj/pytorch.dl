import torch, types
from torch import nn
from torch.nn import functional as F

from ..data.utils.boxes import dice, iou, dists2corners, iou_dists
from ..data.utils.quads import quad2mask
from .utils import ohem

class FOTSLoss(nn.Module):
    def __init__(self, theta_coef=10, reg_coef=1, recog_coef=1, blankIndex=0):
        super().__init__()

        self.theta_coef = theta_coef
        self.reg_coef = reg_coef
        self.recog_coef = recog_coef

        self.detn_loss = DetectionLoss(theta_coef, reg_coef)
        self.reg_loss = RegressionLoss(theta_coef)
        self.recog_loss = RecognitionLoss(blankIndex=blankIndex)

    def forward(self, detn, recog):
        """
        :param detn:
                pos_indicator: bool Tensor, shape = (b, h/4, w/4)
                pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
                pred_rboxes: predicted Tensor, shape = (b, h/4, w/4, 5=(t, r, b, l, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, r, b, l)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
                true_rboxes: true Tensor, shape = (text b, h/4, w/4, 5=(t, r, b, l, angle))
        :param recog:
                pred_texts: list(b) of predicted text number Tensor, shape = (times, text nums, class_nums)
                true_texts: list(b) of true text number Tensor, shape = (true text nums, char nums)
                pred_txtlens: list(b) of length Tensor, shape = (text nums)
                true_txtlens: list(b) of true length Tensor, shape = (true text nums)
        :returns:
            total_loss: Tensor, shape = (1,)
            detn_loss: Tensor, shape = (1,)
            recog_loss: Tensor, shape = (1,)
        """
        detn_loss = self.detn_loss(*detn)
        recog_loss = self.recog_loss(*recog)

        detn_loss = detn_loss.mean()
        recog_loss = recog_loss.mean()
        return detn_loss + self.recog_coef * recog_loss, detn_loss, recog_loss


class DetectionLoss(nn.Module):
    def __init__(self, theta_coef, reg_coef):
        super().__init__()
        self.theta_coef = theta_coef
        self.reg_coef = reg_coef

        self.cls_loss = ClassificationLoss()
        self.reg_loss = RegressionLoss(theta_coef)

    def forward(self, pos_indicator, pred_confs, pred_rboxes, true_rboxes):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param pred_rboxes: predicted Tensor, shape = (b, h/4, w/4, 5=(t, r, b, l, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, r, b, l)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
        :param true_rboxes: true Tensor, shape = (text b, h/4, w/4, 5=(t, r, b, l, angle))
        :return: loss: loss Tensor, shape = (b,)
        """
        # get distances and angles
        pred_dists = pred_rboxes[..., :4]
        pred_angles = pred_rboxes[..., -1:]

        true_dists = true_rboxes[..., :4]
        true_angles = true_rboxes[..., -1:]

        cls_loss = self.cls_loss(pos_indicator, pred_confs)
        reg_loss = self.reg_loss(pos_indicator, pred_confs, pred_dists, true_dists, pred_angles, true_angles)

        return cls_loss + self.reg_coef * reg_loss


class ClassificationLoss(nn.Module):
    def __init__(self, hard_neg_num=512, random_neg_num=512, lossfunc='bce'):
        super().__init__()

        self.hard_neg_num = hard_neg_num
        self.random_neg_num = random_neg_num

        funclist = ['dice', 'stable_bce', 'bce']

        if lossfunc == 'dice':
            self.loss_func = dice_coef
            self.loss_funckwargs = {}
        elif lossfunc == 'stable_bce':
            self.loss_func = stableBCE
            self.loss_funckwargs = {'reduction': 'sum'}
        elif lossfunc == 'bce':
            self.loss_func = F.binary_cross_entropy
            self.loss_funckwargs = {'reduction': 'sum'}
        else:
            raise ValueError('lossfunc must be {}, but got {}'.format(funclist, lossfunc))

    def forward(self, pos_indicator, pred_confs):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :return: loss: loss Tensor, shape = (b,)
        """
        batch_nums, h, w = pos_indicator.shape
        neg_indicator = torch.logical_not(pos_indicator)
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=pos_indicator.device)
        for b in range(batch_nums):
            # shape = (pos num, 1)
            pos_confs = pred_confs[b][pos_indicator[b]]
            #pos_loss = stableBCE(pos_confs, torch.ones_like(pos_confs), reduction='sum')
            pos_loss = self.loss_func(pos_confs, torch.ones_like(pos_confs), **self.loss_funckwargs)

            sample_nums = pos_indicator[b].sum().numel()

            # shape = (neg num, 1)
            neg_confs = pred_confs[b][neg_indicator[b]]
            # online hard example mining
            hard_neg_indices, rand_neg_indices, _sample_nums = ohem(neg_confs, hard_sample_nums=self.hard_neg_num,
                                                                   random_sample_nums=self.random_neg_num)

            sample_nums += _sample_nums

            # hard negative
            hardn_confs = neg_confs[hard_neg_indices]
            #hardn_loss = stableBCE(hardn_confs, torch.zeros_like(hardn_confs), reduction='sum')
            hardn_loss = self.loss_func(hardn_confs, torch.zeros_like(hardn_confs), **self.loss_funckwargs)

            # avoid in-place
            #loss[b] = loss[b] + pos_loss + hardn_loss
            if rand_neg_indices.numel() > 0:
                # random negative
                hardnrand_confs = neg_confs[rand_neg_indices]
                #hardnrand_loss = stableBCE(hardnrand_confs, torch.zeros_like(hardnrand_confs), reduction='sum')
                hardnrand_loss = self.loss_func(hardnrand_confs, torch.zeros_like(hardnrand_confs), **self.loss_funckwargs)

                loss[b] = (pos_loss + hardn_loss + hardnrand_loss) / sample_nums

            else:
                loss[b] = (pos_loss + hardn_loss) / sample_nums

        return loss

class RegressionLoss(nn.Module):
    def __init__(self, theta_coef=10, hard_pos_num=128, random_pos_num=128):
        super().__init__()

        self.theta_coef = theta_coef
        self.hard_pos_num = hard_pos_num
        self.random_pos_num = random_pos_num

    def forward(self, pos_indicator, pred_confs, pred_dists, true_dists, pred_angles, true_angles):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
                              or None. if it's None, use pos_creater(t_quad, h, w)
                                :param t_quad: Tensor, shape = (8=(x1,y1,...))
                                :param h: int
                                :param w: int
                                :return mask_per_rect: Bool Tensor, shape = (h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param pred_dists: predicted distances Tensor, shape = (b, h/4, w/4, 4=(t,r,b,l))
        :param true_dists: true distances Tensor, shape = (b, h/4, w/4, 4=(t,r,b,l))
        :param pred_angles: predicted angles Tensor, shape = (b, h/4, w/4, 1). range = (-pi/4, pi/4). unit is radian.
        :param true_angles: true angle Tensor, shape = (b, h/4, w/4, 1). range = [-pi/4, pi/2)
        :return: loss: loss Tensor, shape = (b,)
        """
        device = pred_dists.device

        batch_nums, h, w, _ = pred_dists.shape
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=device)
        for b in range(batch_nums):
            mask = pos_indicator[b]
            # shape=(masked number, 4=(t,r,b,l))
            pos_p_dists = pred_dists[b][mask]
            # shape=(masked number, 1)
            pos_p_angles = pred_angles[b][mask]
            # shape=(masked number, 1)
            pos_p_confs = pred_confs[b][mask]

            # shape=(masked number, 4=(t,r,b,l))
            pos_t_dists = true_dists[b][mask]
            # shape=(masked number, 1)
            pos_t_angles = true_angles[b][mask]

            orient_loss = 1 - torch.cos(pos_p_angles - pos_t_angles)
            loc_loss = -torch.log(iou_dists(pos_p_dists, pos_t_dists))

            # online hard example mining
            hard_pos_indices, rand_pos_indices, sample_nums = ohem(pos_p_confs, hard_sample_nums=self.hard_pos_num,
                                                                   random_sample_nums=self.random_pos_num)
            hardn_loc_loss = loc_loss[hard_pos_indices]
            hardn_orient_loss = orient_loss[hard_pos_indices]

            if rand_pos_indices.numel() > 0:
                # random positive
                hardnrand_loc_loss = loc_loss[rand_pos_indices]
                hardnrand_orient_loss = orient_loss[rand_pos_indices]
                loss[b] = ((hardn_loc_loss.sum() + hardnrand_loc_loss.sum()) +
                           self.theta_coef * (hardn_orient_loss.sum() + hardnrand_orient_loss.sum())) / sample_nums

            else:
                loss[b] = (hardn_loc_loss.sum() + self.theta_coef * hardn_orient_loss.sum()) / sample_nums

        return loss

class RecognitionLoss(nn.Module):
    def __init__(self, blankIndex=0):
        super().__init__()
        self.blankIndex = blankIndex

    def forward(self, predicts, targets, predict_lengths, target_lengths):
        """
        :param predicts: list(b) of predicted text number Tensor, shape = (times, text nums, class_nums)
        :param targets: list(b) of true text number Tensor, shape = (true text nums, char nums)
        :param predict_lengths: list(b) of length Tensor, shape = (text nums)
        :param target_lengths: list(b) of true length Tensor, shape = (true text nums)
        :return: loss: loss Tensor, shape = (b,)
        """
        batch_nums = len(predicts)
        loss = torch.zeros((batch_nums,), dtype=torch.float, device=predicts[0].device)
        for b, (p, t, p_l, t_l) in enumerate(zip(predicts, targets, predict_lengths, target_lengths)):
            loss[b] = F.ctc_loss(p, t, p_l, t_l, blank=self.blankIndex, reduction='mean', zero_infinity=True)

        return loss


def stableBCE(predict, target, reduction='none'):
    return F.binary_cross_entropy(torch.clamp(predict, min=1e-7, max=1-1e-7), target, reduction=reduction)

def dice_coef(predict, target):
    """
    return summed dice coefficient
    :param predict:
    :param target:
    :return:
    """
    eps = 1e-7
    intersection = torch.sum(predict*target) + eps
    union = torch.sum(predict) + torch.sum(target) + eps
    dice = 1 - (2 * intersection / union)
    return dice
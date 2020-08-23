import torch, types
from torch import nn
from torch.nn import functional as F

from ..data.utils.boxes import dice, iou, dists2corners, poscreator_quads
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

    def forward(self, pos_indicator, pred_confs, pred_locs, true_locs):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param pred_locs: predicted Tensor, shape = (b, h/4, w/4, 5=(t, l, b, r, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, l, b, r)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
        :param true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
        :return: loss: loss Tensor, shape = (b,)
        """
        # convert distances into corners representation
        pred_dists = pred_locs[..., :4]
        pred_boxes = dists2corners(pred_dists)
        pred_angles = pred_locs[..., -1:]

        true_rects = [t_loc[:, :12] for t_loc in true_locs]
        true_angles = [t_loc[:, -1:] for t_loc in true_locs]

        cls_loss = self.cls_loss(pos_indicator, pred_confs)
        reg_loss = self.reg_loss(None, pred_confs, pred_boxes, true_rects, pred_angles, true_angles)

        return cls_loss + self.reg_coef * reg_loss


class ClassificationLoss(nn.Module):
    def __init__(self, hard_neg_num=512, random_neg_num=512, lossfunc='dice'):
        super().__init__()

        self.hard_neg_num = hard_neg_num
        self.random_neg_num = random_neg_num

        if lossfunc == 'dice':
            self.loss_func = dice_coef
            self.loss_funckwargs = {}
        elif lossfunc == 'bce':
            self.loss_func = stableBCE
            self.loss_funckwargs = {'reduction': 'sum'}
        else:
            raise ValueError('lossfunc must be [\'dice\', \'bce\'], but got {}'.format(lossfunc))

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

            # shape = (neg num, 1)
            neg_confs = pred_confs[b][neg_indicator[b]]
            # online hard example mining
            hard_neg_indices, rand_neg_indices, sample_nums = ohem(neg_confs, hard_sample_nums=self.hard_neg_num,
                                                                   random_sample_nums=self.random_neg_num)
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
            if torch.isinf(loss[b]).sum().item() > 0:
                print('class loss is inf!!!')
                print(pos_confs)
                print(neg_confs)
                print(pos_loss, hardn_loss, sample_nums)
                exit()
        return loss

class RegressionLoss(nn.Module):
    def __init__(self, theta_coef=10, hard_pos_num=128, random_pos_num=128):
        super().__init__()

        self.theta_coef = theta_coef
        self.hard_pos_num = hard_pos_num
        self.random_pos_num = random_pos_num

    def forward(self, pos_indicator, pred_confs, pred_boxes, true_rects, pred_thetas, true_thetas):
        """
        :param pos_indicator: bool Tensor, shape = (b, h/4, w/4)
                              or None. if it's None, use pos_creater(t_quad, h, w)
                                :param t_quad: Tensor, shape = (8=(x1,y1,...))
                                :param h: int
                                :param w: int
                                :return mask_per_rect: Bool Tensor, shape = (h/4, w/4)
        :param pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        :param pred_boxes: predicted boxes Tensor, shape = (b, h/4, w/4, 4=(xmin, ymin, xmax, ymax))
        :param true_rects: true boxes list(b) of Tensor, shape = (text numbers, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...))
        :param pred_thetas: predicted angles Tensor, shape = (b, h/4, w/4, 1). range = (-pi/2, pi/2). unit is radian.
        :param true_thetas: true angle list(b) of Tensor, shape = (text numbers, 1)
        :return: loss: loss Tensor, shape = (b,)
        """
        device = pred_boxes.device

        if pos_indicator:
            raise ValueError('Not Supported')
        else:
            batch_nums, h, w, _ = pred_boxes.shape
            loss = torch.zeros((batch_nums,), dtype=torch.float, device=device)
            for b in range(batch_nums):
                text_nums = true_rects[b].shape[0]

                loc_loss = []
                orient_loss = []
                pos_confs = []

                for t in range(text_nums):
                    t_quad = true_rects[b][t, 4:12]  # shape=(8,)
                    t_box = true_rects[b][t, :4].unsqueeze(0)  # shape=(1,4) unsqueeze means for broadcast
                    t_thetas = true_thetas[b][t, :].unsqueeze(0)  # shape=(1,1) unsqueeze means for broadcast

                    # convert percentage representation into true box coordinates
                    t_box = torch.cat((t_box[:, 0:1]*w, t_box[:, 1:2]*h, t_box[:, 2:3]*w, t_box[:, 3:4]*h), dim=-1)
                    """
                    mask = torch.zeros((batch_nums, h, w, 1), dtype=torch.bool, device=device)

                    # shape = (h, w)
                    pos = poscreator_quads(t_quad, h, w, device)
                    mask[b, :, :, 0] = pos

                    p_boxes = pred_boxes.masked_select(mask).reshape(-1, 4)  # shape=(masked number, 4)
                    p_thetas = pred_thetas.masked_select(mask).reshape(-1, 1)  # shape=(masked number, 1)
                    """
                    mask = poscreator_quads(t_quad, h, w, device)
                    p_boxes = pred_boxes[b][mask] # shape=(masked number, 4)
                    p_thetas = pred_thetas[b][mask] # shape=(masked number, 1)
                    p_confs = pred_confs[b][mask] # shape = (masked number, 1)

                    # calculate loss for each true text box
                    # iou's shape = (masked number, 1)
                    loc_loss += [iou(p_boxes, t_box)]
                    # cosine_similarity's shape = (masked number,)
                    # convert to (masked number, 1)
                    # p_thetas is between [-pi/2, pi/2]
                    orient_loss += [1 - torch.cos(p_thetas - t_thetas)]

                    pos_confs += [p_confs]

                # convert list into tensor, shape = (pos num, 1)
                loc_loss = torch.cat(loc_loss, dim=0)
                orient_loss = torch.cat(orient_loss, dim=0)
                pos_confs = torch.cat(pos_confs, dim=0)

                # online hard example mining
                hard_pos_indices, rand_pos_indices, sample_nums = ohem(pos_confs, hard_sample_nums=self.hard_pos_num,
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
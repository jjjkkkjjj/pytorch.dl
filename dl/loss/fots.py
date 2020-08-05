import torch, types
from torch import nn
from torch.nn import functional as F

from ..data.utils.boxes import dice, dists2corners, poscreator_quads

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
    def __init__(self, hard_neg_num=512, random_neg_num=512):
        super().__init__()

        self.hard_neg_num = hard_neg_num
        self.random_neg_num = random_neg_num

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

            # avoid in-place
            #loss[b] = loss[b] + pos_loss + hardn_loss
            sample_nums += pos_confs.numel() + hardn_confs.numel()
            # random negative
            indices = indices[self.hard_neg_num:]
            if indices.numel() > 0:
                randindices = torch.randperm(indices.numel()).unsqueeze(-1)
                indices = indices[randindices]
                hardnrand_confs = neg_confs[indices[:self.random_neg_num]]
                hardnrand_loss = F.binary_cross_entropy(hardnrand_confs, torch.zeros_like(hardnrand_confs), reduction='sum')

                loss[b] = pos_loss + hardn_loss + hardnrand_loss
                sample_nums += hardn_confs.numel()
            else:
                loss[b] = (pos_loss + hardn_loss) / sample_nums

        return loss

class RegressionLoss(nn.Module):
    def __init__(self, theta_coef=10):
        super().__init__()

        self.theta_coef = theta_coef

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
        :param pred_thetas: predicted angles Tensor, shape = (b, h/4, w/4, 1)
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
                sample_nums = 0
                text_nums = true_rects[b].shape[0]

                loc_loss = torch.zeros((text_nums,), dtype=torch.float, device=device)
                orient_loss = torch.zeros((text_nums,), dtype=torch.float, device=device)

                for t in range(text_nums):
                    t_quad = true_rects[b][t, 4:12]  # shape=(8,)
                    t_box = true_rects[b][t, :4].unsqueeze(0)  # shape=(1,4) unsqueeze means for broadcast
                    t_thetas = true_thetas[b][t, :].unsqueeze(0)  # shape=(1,1) unsqueeze means for broadcast
                    """
                    mask = torch.zeros((batch_nums, h, w, 1), dtype=torch.bool, device=device)

                    # shape = (h, w)
                    pos = poscreator_quads(t_quad, h, w, device)
                    mask[b, :, :, 0] = pos

                    p_boxes = pred_boxes.masked_select(mask).reshape(-1, 4)  # shape=(masked number, 4)
                    p_thetas = pred_thetas.masked_select(mask).reshape(-1, 1)  # shape=(masked number, 1)
                    """
                    mask = poscreator_quads(t_quad, h, w, device)
                    p_boxes = pred_boxes[b][mask]
                    p_thetas = pred_thetas[b][mask]

                    # calculate loss for each true text box
                    loc_loss[t] = dice(p_boxes, t_box).sum()
                    orient_loss[t] = F.cosine_similarity(p_thetas, t_thetas, dim=1).sum()

                    sample_nums += mask.sum().item()

                loss[b] = (loc_loss.sum() + self.theta_coef * orient_loss.sum()) / sample_nums

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
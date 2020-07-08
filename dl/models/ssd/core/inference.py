from ....data.utils.boxes import centroids2corners, iou, quads_iou
from ...._utils import _check_ins

from torch.nn import Module
from torch.nn import functional as F
import torch, cv2
import math
import numpy as np

class InferenceBoxBase(Module):
    def __init__(self, class_nums_with_background, filter_func, val_config):
        super().__init__()
        self.class_nums_with_background = class_nums_with_background
        self.filter_func = filter_func

        from ..base import SSDValConfig
        self.val_config = _check_ins('val_config', val_config, SSDValConfig)
        
        self.device = torch.device('cpu')

class InferenceBox(InferenceBoxBase):
    def __init__(self, class_nums_with_background, filter_func, val_config):
        super().__init__(class_nums_with_background, filter_func, val_config)

    @property
    def conf_threshold(self):
        return self.val_config.conf_threshold

    def forward(self, predicts, conf_threshold=None):
        """
        :param predicts: Tensor, shape = (batch number, default boxes number, 4 + class_num)
        :param conf_threshold: float or None, if it's None, passed default value with 0.01
        :return:
            ret_boxes: list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
        """
        # alias
        batch_num = predicts.shape[0]
        class_num = self.class_nums_with_background
        ret_num = predicts.shape[2] - class_num + 1 + 1 # loc num + 1=(class index) + 1=(conf)

        predicts[:, :, -class_num:] = F.softmax(predicts[:, :, -class_num:], dim=-1)

        conf_threshold = conf_threshold if conf_threshold else self.conf_threshold

        ret_boxes = []
        for b in range(batch_num):
            ret_box = []
            pred = predicts[b] # shape = (default boxes number, *)
            for c in range(class_num - 1): # last index means background
                # filter out less than threshold
                indicator = pred[:, -class_num+c] > conf_threshold
                if indicator.sum().item() == 0:
                    continue

                # shape = (filtered default boxes num, *=loc+1=conf)
                filtered_pred = torch.cat((pred[indicator, :-class_num], pred[indicator, -class_num+c].unsqueeze(1)), dim=1)

                # inferred_indices: Tensor, shape = (inferred boxes num,)
                # inferred_confs: Tensor, shape = (inferred boxes num,)
                inferred_indices, inferred_confs, inferred_locs = self.filter_func(filtered_pred, self.val_config)
                if inferred_indices.nelement() == 0:
                    continue
                else:
                    # append class flag
                    # shape = (inferred boxes num, 1)
                    flag = np.broadcast_to([c], shape=(inferred_indices.nelement(), 1))
                    flag = torch.from_numpy(flag).float().to(self.device)

                    # shape = (inferred box num, 2+loc=(class index, confidence, *))
                    ret_box += [torch.cat((flag, inferred_confs.unsqueeze(1), inferred_locs), dim=1)]

            if len(ret_box) == 0:
                ret_boxes += [torch.from_numpy(np.ones((1, ret_num))*np.nan)]
            else:
                ret_boxes += [torch.cat(ret_box, dim=0)]

        # list of tensor, shape = (box num, ret_num=(class index, confidence, *=loc))
        return ret_boxes


def non_maximum_suppression(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 4=loc + 1=conf)
    Note that filtered default boxes number must be more than 1
    :param val_config: SSDValConfig
    :return:
        inferred_indices: Tensor, shape = (inferred box num,)
        inferred_confs: Tensor, shape = (inferred box num,)
        inferred_locs: Tensor, shape = (inferred box num, 4)
    """
    loc, conf = pred[:, :-1], pred[:, -1]
    iou_threshold = val_config.iou_threshold
    topk = val_config.topk

    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]
    # converted into minmax coordinates
    loc_mm = centroids2corners(loc)

    inferred_indices = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]

        largest_conf_loc = loc[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_indices.append(largest_conf_index)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = iou(centroids2corners(largest_conf_loc), loc_mm[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold

        conf_des_inds = conf_des_inds[indicator]

    inferred_indices = torch.Tensor(inferred_indices).long()
    return inferred_indices, conf[inferred_indices], loc[inferred_indices]


def non_maximum_suppression_quads(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 12=bbox+quad + 1=conf)
    :param val_config: SSDValConfig
    :return:
    """
    topk = val_config.topk
    iou_threshold2 = val_config.iou_threshold2

    loc, quad, conf = pred[:, :4], pred[:, 4:12], pred[:, -1]

    # sort confidence and default boxes with descending order
    c, conf_des_inds = conf.sort(dim=0, descending=True)
    # get topk indices
    conf_des_inds = conf_des_inds[:topk]

    inferred_indices = []
    while conf_des_inds.nelement() > 0:
        largest_conf_index = conf_des_inds[0]
        # conf[largest_conf_index]'s shape = []
        largest_conf = conf[largest_conf_index].unsqueeze(0).unsqueeze(0)  # shape = (1, 1)
        largest_conf_quad = quad[largest_conf_index, :].unsqueeze(0)  # shape = (1, 4=(xmin, ymin, xmax, ymax))
        # append to result
        inferred_indices.append(largest_conf_index)

        # remove largest element
        conf_des_inds = conf_des_inds[1:]

        if conf_des_inds.nelement() == 0:
            break

        # get iou, shape = (1, loc_des num)
        overlap = quads_iou(largest_conf_quad, quad[conf_des_inds])
        # filter out overlapped boxes for box with largest conf, shape = (loc_des num)
        indicator = overlap.reshape((overlap.nelement())) <= iou_threshold2

        conf_des_inds = conf_des_inds[indicator]

    inferred_indices = torch.Tensor(inferred_indices).long()
    return inferred_indices, conf[inferred_indices], torch.cat((loc[inferred_indices], quad[inferred_indices]), dim=1)


def textbox_non_maximum_suppression(pred, val_config):
    """
    :param pred: tensor, shape = (filtered default boxes num, 12=bbox+quad + 1=conf)
    Note that filtered default boxes number must be more than 1
    :param val_config: SSDValConfig
    :return:
        inferred_indices: Tensor, shape = (inferred box num,)
        inferred_confs: Tensor, shape = (inferred box num,)
        inferred_locs: Tensor, shape = (inferred box num, 4)
    """
    loc, quad, conf = pred[:, :4], pred[:, 4:12], pred[:, -1]

    indices, _, _ = non_maximum_suppression(torch.cat((loc, conf.unsqueeze(1)), dim=1), val_config)
    if indices.nelement() == 0:
        return indices, conf[indices], torch.cat((loc[indices], quad[indices]), dim=1)

    non_maximum_suppression_quads(pred[indices], val_config)


    return indices, conf[indices], torch.cat((loc[indices], quad[indices]), dim=1)



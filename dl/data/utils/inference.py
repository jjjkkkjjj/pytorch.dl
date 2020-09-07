import torch

from .boxes import centroids2corners, iou
from .quads import quads_iou

def non_maximum_suppression(confs, values, topk, threshold, compare_func, **funckwargs):
    """
    :param confs: Tensor, shape = (val num,)
    :param values: Tensor, shape = (val num, ?)
        Note that val num must be more than 1
    :param topk: int
    :param threshold: float
    :param compare_func: function,
            arguments: (a: Tensor, shape=(a_nums, ?), b: Tensor, shape=(b_nums, ?))
            return: ret: Tensor, shape=(a_nums, b_nums)
    :return inferred_indices: Tensor, shape = (inferred box num,)
    """

    # sort confidence and default boxes with descending order
    c, conf_descending_inds = confs.sort(dim=0, descending=True)
    # get topk indices
    conf_descending_inds = conf_descending_inds[:topk]

    inferred_indices = []
    while conf_descending_inds.nelement() > 0:
        largest_conf_index = conf_descending_inds[0]

        largest_conf_val = values[largest_conf_index, :].unsqueeze(0)  # shape = (1, ?)
        # append to result
        inferred_indices.append(largest_conf_index)

        # remove largest element
        conf_descending_inds = conf_descending_inds[1:]

        if conf_descending_inds.nelement() == 0:
            break

        # get iou, shape = (1, conf_descending_inds num)
        overlap = compare_func(largest_conf_val, values[conf_descending_inds], **funckwargs)
        # filter out overlapped boxes for box with largest conf, shape = (conf_descending_inds num)
        indicator = overlap.reshape((overlap.nelement())) <= threshold

        conf_descending_inds = conf_descending_inds[indicator]

    inferred_indices = torch.Tensor(inferred_indices).long()
    return inferred_indices


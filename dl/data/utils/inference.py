import torch

from .boxes import centroids2corners, iou
from .quads import quads_iou
from .._utils import _check_ins

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
    topk = _check_ins('topk', topk, int)
    threshold = _check_ins('threshold', threshold, float)

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


def weighted_merge(conf1, values1, conf2, values2):
    """
    :param conf1: Tensor, shape=(1,)
    :param values1: Tensor, shape=(?,)
    :param conf2: Tensor, shape=(1,)
    :param values2: Tensor, shape=(?,)
    :return:
    """
    weighted_values = (conf1 * values1 + conf2 * values2) / (conf1 + conf2)
    weighted_conf = conf1 + conf2
    return weighted_conf, weighted_values

def locally_aware_nms(confs, values, topk, threshold, compare_func, **funckwargs):
    """
    :param confs: Tensor, shape=(num,)
    :param values: Tensor, shape=(num, ?)
    :param topk: int
    :param threshold: float
    :param compare_func: function,
            arguments: (a: Tensor, shape=(a_nums, ?), b: Tensor, shape=(b_nums, ?))
            return: ret: Tensor, shape=(a_nums, b_nums)
    :param funckwargs:
    :return inferred_indices: Tensor, shape = (inferred box num,)
    """
    topk = _check_ins('topk', topk, int)
    threshold = _check_ins('threshold', threshold, float)
    val_nums = confs.shape[0]

    if val_nums == 0:
        return torch.Tensor([]).bool()

    new_confs, new_values, indices = [], [], []
    prev_conf, prev_value = confs[0], values[0]
    for n in range(1, val_nums):
        if compare_func(prev_value.unsqueeze(0), values[n].unsqueeze(0), **funckwargs).item() > threshold:
            prev_conf, prev_value = weighted_merge(prev_conf, prev_value, confs[n], values[n])
        else:
            new_confs += [confs[n]]
            new_values += [values[n].unsqueeze(0)]
            indices += [n]
            prev_conf, prev_value = confs[n], values[n]

    new_confs, new_values, inferred_indices = torch.tensor(new_confs, dtype=torch.float), torch.cat(new_values, dim=0), torch.tensor(indices, dtype=torch.long)

    if inferred_indices.numel() == 0:
        return torch.Tensor([]).bool()


    standard_inds = non_maximum_suppression(new_confs, new_values, topk, threshold, compare_func, **funckwargs)
    inferred_indices = inferred_indices[standard_inds]

    return inferred_indices

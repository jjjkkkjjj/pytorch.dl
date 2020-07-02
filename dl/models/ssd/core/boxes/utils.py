import numpy as np
import torch
from torch import nn
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

def matching_strategy(targets, dboxes, **kwargs):
    """
    :param targets: Tensor, shape is (batch*object num(batch), 1+4+class_labels)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        matched_targets: Tensor, shape = (batch, default box num, 4+class_num) including background
    """
    threshold = kwargs.pop('threshold', 0.5)
    batch_num = kwargs.pop('batch_num')
    device = dboxes.device



    dboxes_num = dboxes.shape[0]
    # minus 'box number per image' and 'localization=(cx, cy, w, h)'
    class_num = targets[0].shape[1] - 4

    # convert centered coordinated to minmax coordinates
    dboxes_mm = centroids2corners(dboxes)

    # create returned empty Tensor
    pos_indicator, matched_targets = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty((batch_num, dboxes_num, 4 + class_num), device=device)

    # matching for each batch
    index = 0
    for b, target in enumerate(targets):
        targets_loc, targets_conf = target[:, :4], target[:, 4:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(centroids2corners(targets_loc), dboxes_mm.clone())
        """
        best_overlap_per_object, best_dbox_ind_per_object = overlaps.max(dim=1)
        best_overlap_per_dbox, best_object_ind_per_dbox = overlaps.max(dim=0)
        for object_ind, dbox_ind in enumerate(best_dbox_ind_per_object):
            best_object_ind_per_dbox[dbox_ind] = object_ind
        best_overlap_per_dbox.index_fill_(0, best_dbox_ind_per_object, 999)

        pos_ind = best_overlap_per_dbox > threshold
        pos_indicator[b] = pos_ind
        gt_loc[b], gt_conf[b] = targets[best_object_ind_per_dbox], targets_conf[best_object_ind_per_dbox]

        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1
        """
        # get maximum overlap value for each default box
        # shape = (batch num, dboxes num)
        overlaps_per_dbox, object_indices = overlaps.max(dim=0)
        #object_indices = object_indices.long() # for fancy indexing

        # get maximum overlap values for each object
        # shape = (batch num, object num)
        overlaps_per_object, dbox_indices = overlaps.max(dim=1)
        for obj_ind, dbox_ind in enumerate(dbox_indices):
            object_indices[dbox_ind] = obj_ind
        overlaps_per_dbox.index_fill_(0, dbox_indices, threshold + 1)# ensure N!=0

        pos_ind = overlaps_per_dbox > threshold

        # assign targets
        matched_targets[b, :, :4], matched_targets[b, :, 4:] = targets_loc[object_indices], targets_conf[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        matched_targets[b, neg_ind, 4:] = 0
        matched_targets[b, neg_ind, -1] = 1



    return pos_indicator, matched_targets

def iou(a, b):
    """
    :param a: Box Tensor, shape is (nums, 4)
    :param b: Box Tensor, shape is (nums, 4)
    IMPORTANT: Note that 4 means (xmin, ymin, xmax, ymax)
    :return:
        iou: Tensor, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """

    # get intersection's xmin, ymin, xmax, ymax
    # xmin = max(a_xmin, b_xmin)
    # ymin = max(a_ymin, b_ymin)
    # xmax = min(a_xmax, b_xmax)
    # ymax = min(a_ymax, b_ymax)
    """
    >>> b
    tensor([2., 6.])
    >>> c
    tensor([1., 5.])
    >>> torch.cat((b.unsqueeze(1),c.unsqueeze(1)),1)
    tensor([[2., 1.],
            [6., 5.]])
    """
    # convert for broadcast
    # a's shape = (a_num, 1, 4), b's shape = (1, b_num, 4)
    a, b = a.unsqueeze(1), b.unsqueeze(0)
    intersection = torch.cat((torch.max(a[:, :, 0], b[:, :, 0]).unsqueeze(2),
                              torch.max(a[:, :, 1], b[:, :, 1]).unsqueeze(2),
                              torch.min(a[:, :, 2], b[:, :, 2]).unsqueeze(2),
                              torch.min(a[:, :, 3], b[:, :, 3]).unsqueeze(2)), dim=2)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, :, 2] - intersection[:, :, 0], intersection[:, :, 3] - intersection[:, :, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = torch.clamp(intersection_w, min=0), torch.clamp(intersection_h, min=0)

    intersectionArea = intersection_w * intersection_h

    # get a and b's area
    # area = (xmax - xmin) * (ymax - ymin)
    A, B = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1]), (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])

    return intersectionArea / (A + B - intersectionArea)

def iou_numpy(a, b):
    """
    :param a: Box ndarray, shape is (nums, 4)
    :param b: Box ndarray, shape is (nums, 4)
    IMPORTANT: Note that 4 means (xmin, ymin, xmax, ymax)
    :return:
        iou: ndarray, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """

    # get intersection's xmin, ymin, xmax, ymax
    # xmin = max(a_xmin, b_xmin)
    # ymin = max(a_ymin, b_ymin)
    # xmax = min(a_xmax, b_xmax)
    # ymax = min(a_ymax, b_ymax)

    # convert for broadcast
    # a's shape = (a_num, 1, 4), b's shape = (1, b_num, 4)
    a, b = np.expand_dims(a, 1), np.expand_dims(b, 0)
    intersection = np.concatenate((np.expand_dims(np.maximum(a[:, :, 0], b[:, :, 0]), 2),
                                   np.expand_dims(np.maximum(a[:, :, 1], b[:, :, 1]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 2], b[:, :, 2]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 3], b[:, :, 3]), 2)), axis=2)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, :, 2] - intersection[:, :, 0], intersection[:, :, 3] - intersection[:, :, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = np.clip(intersection_w, a_min=0, a_max=None), np.clip(intersection_h, a_min=0, a_max=None)

    intersectionArea = intersection_w * intersection_h

    # get a and b's area
    # area = (xmax - xmin) * (ymax - ymin)
    A, B = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1]), (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])

    return intersectionArea / (A + B - intersectionArea)

def centroids2corners(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    :return:
        a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.cat((a[:, :2] - a[:, 2:]/2, a[:, :2] + a[:, 2:]/2), dim=1)

def corners2centroids(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    :return:
        a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    """
    return torch.cat(((a[:, 2:] + a[:, :2])/2, a[:, 2:] - a[:, :2]), dim=1)

def centroids2corners_numpy(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    :return:
        a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    """
    return np.concatenate((a[:, :2] - a[:, 2:]/2, a[:, :2] + a[:, 2:]/2), axis=1)

def corners2centroids_numpy(a):
    """
    :param a: Box Tensor, shape is (nums, 4=(xmin, ymin, xmax, ymax))
    :return:
        a: Box Tensor, shape is (nums, 4=(cx, cy, w, h))
    """
    return np.concatenate(((a[:, 2:] + a[:, :2])/2, a[:, 2:] - a[:, :2]), axis=1)


"""
repeat_interleave is similar to numpy.repeat
>>> a = torch.Tensor([[1,2,3,4],[5,6,7,8]])
>>> a
tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.]])
>>> torch.repeat_interleave(a, 3, dim=0)
tensor([[1., 2., 3., 4.],
        [1., 2., 3., 4.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [5., 6., 7., 8.],
        [5., 6., 7., 8.]])
>>> torch.cat(3*[a])
tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [1., 2., 3., 4.],
        [5., 6., 7., 8.]])
"""
def tensor_tile(a, repeat, dim=0):
    return torch.cat([a]*repeat, dim=dim)


def matching_strategy(targets, dboxes, **kwargs):
    """
    :param targets: Tensor, shape is (batch*object num(batch), 4=(cx,cy,w,h)+8=(x1,y1,x2,y2,...)+class_labels)
    :param dboxes: shape is (default boxes num, 4)
    IMPORTANT: Note that means (cx, cy, w, h)
    :param kwargs:
        threshold: (Optional) float, threshold for returned indicator
        batch_num: (Required) int, batch size
    :return:
        pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
        matched_targets: Tensor, shape = (batch, default box num, 4+class_num) including background
    """
    threshold = kwargs.pop('threshold', 0.5)
    batch_num = kwargs.pop('batch_num')
    device = dboxes.device

    dboxes_num = dboxes.shape[0]
    # minus 'box number per image' and 'localization=(cx, cy, w, h)'
    class_num = targets[0].shape[1] - 4 - 8

    # convert centered coordinated to minmax coordinates
    dboxes_mm = centroids2corners(dboxes)

    # create returned empty Tensor
    pos_indicator, matched_targets = torch.empty((batch_num, dboxes_num), device=device, dtype=torch.bool), torch.empty(
        (batch_num, dboxes_num, 4 + 8 + class_num), device=device)

    # matching for each batch
    index = 0
    for b, target in enumerate(targets):
        targets_loc, targets_quad, targets_conf = target[:, :4], target[:, 4:12], target[:, 12:]

        # overlaps' shape = (object num, default box num)
        overlaps = iou(centroids2corners(targets_loc), dboxes_mm.clone())
        """
        best_overlap_per_object, best_dbox_ind_per_object = overlaps.max(dim=1)
        best_overlap_per_dbox, best_object_ind_per_dbox = overlaps.max(dim=0)
        for object_ind, dbox_ind in enumerate(best_dbox_ind_per_object):
            best_object_ind_per_dbox[dbox_ind] = object_ind
        best_overlap_per_dbox.index_fill_(0, best_dbox_ind_per_object, 999)

        pos_ind = best_overlap_per_dbox > threshold
        pos_indicator[b] = pos_ind
        gt_loc[b], gt_conf[b] = targets[best_object_ind_per_dbox], targets_conf[best_object_ind_per_dbox]

        neg_ind = torch.logical_not(pos_ind)
        gt_conf[b, neg_ind] = 0
        gt_conf[b, neg_ind, -1] = 1
        """
        # get maximum overlap value for each default box
        # shape = (batch num, dboxes num)
        overlaps_per_dbox, object_indices = overlaps.max(dim=0)
        # object_indices = object_indices.long() # for fancy indexing

        # get maximum overlap values for each object
        # shape = (batch num, object num)
        overlaps_per_object, dbox_indices = overlaps.max(dim=1)
        for obj_ind, dbox_ind in enumerate(dbox_indices):
            object_indices[dbox_ind] = obj_ind
        overlaps_per_dbox.index_fill_(0, dbox_indices, threshold + 1)  # ensure N!=0

        pos_ind = overlaps_per_dbox > threshold

        # assign targets
        matched_targets[b, :, :4], matched_targets[b, :, 4:12], matched_targets[b, :, 12:] = \
            targets_loc[object_indices], targets_quad[object_indices], targets_conf[object_indices]
        pos_indicator[b] = pos_ind

        # set background flag
        neg_ind = torch.logical_not(pos_ind)
        matched_targets[b, neg_ind, 12:] = 0
        matched_targets[b, neg_ind, -1] = 1

    return pos_indicator, matched_targets


def coverage_numpy(a, b, divide_b=False):
    """
    :param a: Box ndarray, shape is (nums, 4)
    :param b: Box ndarray, shape is (nums, 4)
    IMPORTANT: Note that 4 means (xmin, ymin, xmax, ymax)
    :param divide_b: bool, if true, |a ^ b| / |b|, otherwise, |a ^ b| / |a|
    :return:
        iou: ndarray, shape is (a_num, b_num)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """

    # get intersection's xmin, ymin, xmax, ymax
    # xmin = max(a_xmin, b_xmin)
    # ymin = max(a_ymin, b_ymin)
    # xmax = min(a_xmax, b_xmax)
    # ymax = min(a_ymax, b_ymax)

    # convert for broadcast
    # a's shape = (a_num, 1, 4), b's shape = (1, b_num, 4)
    a, b = np.expand_dims(a, 1), np.expand_dims(b, 0)
    intersection = np.concatenate((np.expand_dims(np.maximum(a[:, :, 0], b[:, :, 0]), 2),
                                   np.expand_dims(np.maximum(a[:, :, 1], b[:, :, 1]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 2], b[:, :, 2]), 2),
                                   np.expand_dims(np.minimum(a[:, :, 3], b[:, :, 3]), 2)), axis=2)
    # get intersection's area
    # (w, h) = (xmax - xmin, ymax - ymin)
    intersection_w, intersection_h = intersection[:, :, 2] - intersection[:, :, 0], intersection[:, :,
                                                                                    3] - intersection[:, :, 1]
    # if intersection's width or height is negative, those will be converted to zero
    intersection_w, intersection_h = np.clip(intersection_w, a_min=0, a_max=None), np.clip(intersection_h, a_min=0,
                                                                                           a_max=None)

    intersectionArea = intersection_w * intersection_h

    # get a and b's area
    # area = (xmax - xmin) * (ymax - ymin)
    """
    >>> a = np.array([-1, 0, 1, 2, 3], dtype=float)
    >>> b = np.array([ 0, 0, 0, 2, 2], dtype=float)

    # If you don't pass `out` the indices where (b == 0) will be uninitialized!
    >>> c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    >>> print(c)
    [ 0.   0.   0.   1.   1.5]
    """
    if divide_b:
        B = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])
        # return intersectionArea / B
        return np.divide(intersectionArea, B, out=np.zeros_like(intersectionArea), where=B != 0)
    else:
        A = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
        # return intersectionArea / A
        return np.divide(intersectionArea, A, out=np.zeros_like(intersectionArea), where=A != 0)


# ref: https://github.com/MhLiao/TextBoxes_plusplus/blob/master/examples/text/nms.py
def quads_iou(a, b):
    """
    :param a: Box Tensor, shape is (nums, 8)
    :param b: Box Tensor, shape is (nums, 8)
    IMPORTANT: Note that 8 means (topleft=(x1, y1), x2, y2,..clockwise)
    :return:
        iou: Tensor, shape is (a_num, b_num)
             formula is
             iou = intersection / union
    """
    # convert Tensor to numpy for using shapely
    a_numpy, b_numpy = a.cpu().numpy(), b.cpu().numpy()

    a_number, b_number = a_numpy.shape[0], b_numpy.shape[0]
    ret = np.zeros(shape=(a_number, b_number), dtype=np.float32)

    a_numpy, b_numpy = a_numpy.reshape((-1, 4, 2)), b_numpy.reshape((-1, 4, 2))
    for i in range(a_number):
        a_polygon = Polygon(a_numpy[i]).convex_hull
        for j in range(b_number):
            b_polygon = Polygon(b_numpy[j]).convex_hull

            if not a_polygon.intersects(b_polygon):
                continue

            intersectionArea = a_polygon.intersection(b_polygon).area
            unionArea = MultiPoint(np.concatenate((a_numpy[i], b_numpy[j]))).convex_hull.area
            if unionArea == 0:
                continue
            ret[i, j] = intersectionArea / unionArea

    return torch.from_numpy(ret)
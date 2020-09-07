import torch
import numpy as np

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

def iou_dists(a, b):
    """
    :param a: dists Tensor, shape is (*, 4=(t,r,b,l))
    :param b: dists Tensor, shape is (*, 4=(t,r,b,l))
    :return:
        iou: Tensor, shape is (*,)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """
    assert a.shape == b.shape, "must be same shape, but fot {} and {}".format(a.shape, b.shape)
    A, B = (a[..., 0] + a[..., 2])*(a[..., 1] + a[..., 3]), (b[..., 0] + b[..., 2])*(b[..., 1] + b[..., 3])
    intersects = torch.min(a, b)
    intersectionArea = (intersects[..., 0] + intersects[..., 2])*(intersects[..., 1] + intersects[..., 3])

    return intersectionArea / (A + B - intersectionArea)


def iou_dists_numpy(a, b):
    """
    :param a: dists ndarray, shape is (*, 4=(t,r,b,l))
    :param b: dists ndarray, shape is (*, 4=(t,r,b,l))
    :return:
        iou: ndarray, shape is (*,)
             formula is
             iou = intersection / union = intersection / (A + B - intersection)
    """
    assert a.shape == b.shape, "must be same shape, but fot {} and {}".format(a.shape, b.shape)
    A, B = (a[..., 0] + a[..., 2]) * (a[..., 1] + a[..., 3]), (b[..., 0] + b[..., 2]) * (b[..., 1] + b[..., 3])
    intersects = np.minimum(a, b)
    intersectionArea = (intersects[..., 0] + intersects[..., 2]) * (intersects[..., 1] + intersects[..., 3])

    return intersectionArea / (A + B - intersectionArea)

def dice(a, b):
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

    return 2*intersectionArea / (A + B)

def dice_numpy(a, b):
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

    return 2*intersectionArea / (A + B)

def sort_corners(a):
    """
    Sort corners points (xmin, ymin, xmax, ymax)
    :param a: Box Tensor, shape is ([nums, ]*, 4=(x1,y1,x2,y2))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.cat((a[:, ::2].min(dim=-1, keepdims=True),
                      a[:, 1::2].min(dim=-1, keepdims=True),
                      a[:, ::2].max(dim=-1, keepdims=True),
                      a[:, 1::2].max(dim=-1, keepdims=True)), dim=-1)

def sort_corners_numpy(a):
    """
    Sort corners points (xmin, ymin, xmax, ymax)
    :param a: Box ndarray, shape is ([nums, ]*, 4=(x1,y1,x2,y2))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return np.concatenate((a[:, ::2].min(axis=-1, keepdims=True),
                           a[:, 1::2].min(axis=-1, keepdims=True),
                           a[:, ::2].max(axis=-1, keepdims=True),
                           a[:, 1::2].max(axis=-1, keepdims=True)), axis=-1)

def corners2centroids(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(cx, cy, w, h))
    """
    return torch.cat(((a[..., 2:] + a[..., :2])/2, a[..., 2:] - a[..., :2]), dim=-1)

def corners2centroids_numpy(a):
    """
    :param a: Box ndarray, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(cx, cy, w, h))
    """
    return np.concatenate(((a[..., 2:] + a[..., :2])/2, a[..., 2:] - a[..., :2]), axis=-1)

def corners2minmax(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    """
    return torch.index_select(a, dim=-1, index=torch.tensor([0, 2, 1, 3]))

def corners2minmax_numpy(a):
    """
    :param a: Box ndarray, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    """
    return a[..., np.array((0, 2, 1, 3))]

def centroids2corners(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(cx, cy, w, h))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.cat((a[..., :2] - a[..., 2:]/2, a[..., :2] + a[..., 2:]/2), dim=-1)

def centroids2corners_numpy(a):
    """
    :param a: Box ndarray, shape is ([nums, ]*, 4=(cx, cy, w, h))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return np.concatenate((a[..., :2] - a[..., 2:]/2, a[..., :2] + a[..., 2:]/2), axis=-1)

def centroids2minmax(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(cx, cy, w, h))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    """
    return torch.cat((a[..., 0] - a[..., 2]/2,
                      a[..., 0] + a[..., 2]/2,
                      a[..., 1] - a[..., 3]/2,
                      a[..., 1] + a[..., 3]/2), dim=-1)

def centroids2minmax_numpy(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(cx, cy, w, h))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    """
    return np.concatenate((a[..., 0] - a[..., 2]/2,
                           a[..., 0] + a[..., 2]/2,
                           a[..., 1] - a[..., 3]/2,
                           a[..., 1] + a[..., 3]/2), axis=-1)

def minmax2centroids(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(cx, cy, w, h))
    """
    return torch.cat((a[..., 0] + (a[..., 1] - a[..., 0])/2,
                      a[..., 2] + (a[..., 3] - a[..., 2])/2,
                      a[..., 1] - a[..., 0],
                      a[..., 3] - a[..., 2]), dim=-1)

def minmax2centroids_numpy(a):
    """
    :param a: Box ndarray, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(cx, cy, w, h))
    """
    return np.concatenate((a[..., 0] + (a[..., 1] - a[..., 0])/2,
                           a[..., 2] + (a[..., 3] - a[..., 2])/2,
                           a[..., 1] - a[..., 0],
                           a[..., 3] - a[..., 2]), axis=-1)

def minmax2corners(a):
    """
    :param a: Box Tensor, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    :return a: Box Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.index_select(a, dim=-1, index=torch.tensor([0, 2, 1, 3]))

def minmax2corners_numpy(a):
    """
    :param a: Box ndarray, shape is ([nums, ]*, 4=(xmin, xmax, ymin, ymax))
    :return a: Box ndarray, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return a[..., np.array((0, 2, 1, 3))]

def dists2corners(a):
    """
    :param a: dist Tensor, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box Tensor, shape is (*, h, w, 4=(xmin, ymin, xmax, ymax))
    """
    assert a.ndim >= 3, 'must be greater than 3d'
    h, w, _ = a.shape[-3:]
    device = a.device
    # shape = (*, h, w, 4=(xmin, ymin, xmax, ymax))
    ret = torch.zeros_like(a, device=device, dtype=torch.float)

    heights, widths = torch.meshgrid(torch.arange(h), torch.arange(w))
    # shape = (h, w, 1)
    heights = heights.to(device)
    widths = widths.to(device)

    widths, rights, lefts = torch.broadcast_tensors(widths, a[..., 1], a[..., 3])
    heights, tops, bottoms = torch.broadcast_tensors(heights, a[..., 0], a[..., 2])
    xmin = (widths - lefts).unsqueeze(-1) # xmin
    ymin = (heights - tops).unsqueeze(-1) # ymin
    xmax = (widths + rights).unsqueeze(-1) # xmax
    ymax = (heights + bottoms).unsqueeze(-1) # ymax

    ret[..., ::2] = torch.clamp(torch.cat((xmin, xmax), dim=-1), 0, w)
    ret[..., 1::2] = torch.clamp(torch.cat((ymin, ymax), dim=-1), 0, h)

    return ret

def dists2corners_numpy(a):
    """
    :param a: dist ndarray, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box ndarray, shape is (*, h, w, 4=(xmin, ymin, xmax, ymax))
    """
    assert a.ndim >= 3, 'must be greater than 3d'
    h, w, _ = a.shape[-3:]

    # shape = (*, h, w, 4=(xmin, ymin, xmax, ymax))
    ret = np.zeros_like(a)

    widths, heights = np.meshgrid(np.arange(w), np.arange(h))
    # shape = (h, w, 1)
    widths, rights, lefts = np.broadcast_arrays(widths, a[..., 1], a[..., 3])
    heights, tops, bottoms = np.broadcast_arrays(heights, a[..., 0], a[..., 2])
    xmin = np.expand_dims(widths - lefts, axis=-1) # xmin
    ymin = np.expand_dims(heights - tops, axis=-1) # ymin
    xmax = np.expand_dims(widths + rights, axis=-1) # xmax
    ymax = np.expand_dims(heights + bottoms, axis=-1) # ymax

    ret[..., ::2] = np.clip(np.concatenate((xmin, xmax), axis=-1), 0, w)
    ret[..., 1::2] = np.clip(np.concatenate((ymin, ymax), axis=-1), 0, h)

    return ret

def dists2centroids(a):
    """
    :param a: dist Tensor, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box Tensor, shape is (*, h, w, 4=(cx, cy, w, h))
    """
    return corners2centroids(dists2corners(a))

def dists2centroids_numpy(a):
    """
    :param a: dist ndarray, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box ndarray, shape is (*, h, w, 4=(cx, cy, w, h))
    """
    return corners2centroids_numpy(dists2corners_numpy(a))

def dists2minmax(a):
    """
    :param a: dist Tensor, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box Tensor, shape is (*, h, w, 4=(xmin, xmax, ymin, ymax))
    """
    return corners2minmax(dists2corners(a))

def dists2minmax_numpy(a):
    """
    :param a: dist ndarray, shape = (*, h, w, 4=(t, r, b, l))
    :return a: Box ndarray, shape is (*, h, w, 4=(xmin, xmax, ymin, ymax))
    """
    return corners2minmax_numpy(dists2corners_numpy(a))

def dists_pt2line_numpy(line_pt1, line_pt2, pt):
    """
    :param line_pt1: ndarray, shape = (*, 2)
    :param line_pt2: ndarray, shape = (*, 2)
    :param pt: ndarray, shape = (..., 2)
    :return: distances: ndarray, shape = (..., *)
    """
    assert line_pt1.shape == line_pt2.shape, "shape of line_pt1 and line_pt2 must be same, but got {} and {}".format(line_pt1.shape, line_pt2.shape)
    assert line_pt1.shape[-1] == pt.shape[-1] == 2, "last dimension must be 2"

    # convert shape for broadcasting
    # >>> a=np.arange(216*2).reshape(2,3,6,3,2,2)
    # >>> b=np.arange(5*2).reshape(5,2)
    # >>> np.expand_dims(b, (0,1,2,3,4)).shape
    # (1, 1, 1, 1, 1, 5, 2)
    # >>> np.expand_dims(b, (-2,-3,-4,-5,-6)).shape
    # (5, 1, 1, 1, 1, 1, 2)

    line_dim = line_pt1.ndim - 1
    # shape = (..., (1,...,1)=line_dim, 2)
    broadcasted_pt = np.expand_dims(pt, axis=tuple(i for i in range(-2, -(2+line_dim), -1)))

    pt_dim = pt.ndim - 1
    # shape = ((1,...,1)=pt_dim, *, 2)
    broadcasted_line_pt1 = np.expand_dims(line_pt1, axis=tuple(i for i in range(0, pt_dim)))
    broadcasted_line_pt2 = np.expand_dims(line_pt2, axis=tuple(i for i in range(0, pt_dim)))

    # note that np.cross returns scalar value with shape = (..., *)
    return np.abs(np.cross(broadcasted_line_pt2 - broadcasted_line_pt1, broadcasted_line_pt1 - broadcasted_pt)) \
           / np.linalg.norm(broadcasted_line_pt2 - broadcasted_line_pt1, axis=-1)

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

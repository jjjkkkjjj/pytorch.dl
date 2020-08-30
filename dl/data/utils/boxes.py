import torch, cv2
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

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

def dists2corners(a):
    """
    :param a: dist Tensor, shape = (*, h, w, 4=(t, l, b, r))
    :return:
        a: Box Tensor, shape is (*, h, w, 4=(xmin, ymin, xmax, ymax))
    """
    assert a.ndim > 3, 'must be greater than 3d'
    h, w, _ = a.shape[-3:]
    device = a.device
    # shape = (*, h, w, 4=(xmin, ymin, xmax, ymax))
    ret = torch.zeros_like(a, device=device, dtype=torch.float)

    heights, widths = torch.meshgrid(torch.arange(h), torch.arange(w))
    # shape = (h, w, 1)
    heights = heights.to(device)
    widths = widths.to(device)

    widths, lefts, rights = torch.broadcast_tensors(widths, a[..., 1], a[..., 3])
    heights, tops, bottoms = torch.broadcast_tensors(heights, a[..., 0], a[..., 2])
    xmin = (widths - lefts).unsqueeze(-1) # xmin
    ymin = (heights - tops).unsqueeze(-1) # ymin
    xmax = (widths + rights).unsqueeze(-1) # xmax
    ymax = (heights + bottoms).unsqueeze(-1) # ymax

    ret[..., ::2] = torch.clamp(torch.cat((xmin, xmax), dim=-1), 0, w)
    ret[..., 1::2] = torch.clamp(torch.cat((ymin, ymax), dim=-1), 0, h)

    return ret


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


def poscreator_quads(quad, h, w, device):
    """
    :param quad: Tensor, shape = (8=(x1,y1,...))
    :param h: int
    :param w: int
    :param device: device
    :return: mask_per_rect: Bool Tensor, shape = (h/4, w/4)
    """
    _quad = quad.clone() # avoid in-place operation
    _quad[::2] *= w
    _quad[1::2] *= h

    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(_quad.cpu().numpy(), outline=255, fill=255)
    # img.show()
    return torch.from_numpy(np.array(img, dtype=np.bool)).to(device=device)


def shrink_quads_numpy(reshaped_quads, ref_lengths, scale=0.3):
    """
    convert quads into rbox, see fig4 in EAST paper
    :param reshaped_quads: ndarray, shape = (box nums, 4=(clockwise from top-left), 2=(x,y))
    :param ref_lengths: ndarray, shape = (box nums, 4)
    :param scale: int, shrink scale
    :return: rboxes: ndarray, shape = (box nums, 4=(clockwise from top-left), 2=(x, y))
    """
    assert reshaped_quads.shape[-2:] == (4, 2), "reshaped_quads' shape must be (box nums, 4, 2)"

    def _shrink_h(quad, ref_len):
        """
        :param quad: ndarray, shape = (4, 2)
        :param ref_len: ndarray, shape = (4,)
        """
        # get angle
        adj_quad = np.roll(quad[::-1], 2, axis=0)  # adjacent points
        # shape = (4,)
        angles = np.arctan2(adj_quad[:, 1] - quad[:, 1], adj_quad[:, 0] - quad[:, 0])

        # shape = (4,2)
        trigonometric = np.array([np.cos(angles),
                                  np.sin(angles)]).T

        quad += np.expand_dims(ref_len, axis=-1) * trigonometric * scale

        return quad

    def _shrink_v(quad, ref_len):
        """
        :param quad: ndarray, shape = (4, 2)
        :param ref_len: ndarray, shape = (4,)
        """
        # get angle
        adj_quad = quad[::-1]  # adjacent points
        # shape = (4,)
        angles = np.arctan2(adj_quad[:, 0] - quad[:, 0], adj_quad[:, 1] - quad[:, 1])

        # shape = (4,2)
        trigonometric = np.array([np.sin(angles),
                                  np.cos(angles)]).T

        quad += np.expand_dims(ref_len, axis=-1) * trigonometric * scale

        return quad

    def _shrink(quad, ref_len, horizontal_first):
        """
        :param quad: ndarray, shape = (4, 2)
        :param ref_len: ndarray, shape = (4,)
        :param horizontal_first: boolean, if True, horizontal edges will be shrunk first, otherwise vertical ones will be shrunk first
        :return:
        """
        if horizontal_first:
            quad = _shrink_h(quad, ref_len)
            quad = _shrink_v(quad, ref_len)
        else:
            quad = _shrink_v(quad, ref_len)
            quad = _shrink_h(quad, ref_len)

        return quad

    box_nums = reshaped_quads.shape[0]

    # lengths, clockwise from horizontal top edge
    # shape = (box nums, 4)
    lengths = np.linalg.norm(reshaped_quads - np.roll(reshaped_quads, 1, axis=1), axis=-1)

    h_lens, v_lens = np.mean(lengths[:, ::2], axis=-1), np.mean(lengths[:, 1::2], axis=-1)
    horizontal_firsts = h_lens > v_lens

    shrinked_quads = np.array([_shrink(reshaped_quads[b], ref_lengths[b], horizontal_firsts[b]) for b in range(box_nums)])
    return shrinked_quads



def quads2rboxes_numpy(quads, scale):
    """
    convert quads into rbox, see fig4 in EAST paper
    https://github.com/Masao-Taketani/FOTS_OCR/blob/5c214bf2e3d815d6f826f7771da92ba4d899d08b/data_provider/data_utils.py#L575

    Brief summary of rbox creation from quads

    1. compute reference lengths (ref_lengths) by getting shorter edge adjacent one point
    2. shrink longer edge pair* with scale value
        *: longer pair is got by comparing between two opposing edges following;
            (vertical edge1 + 2)ave <=> (horizontal edge1 + 2)ave
        Note that shrinking way is different between vertical edges pair and horizontal one
        horizontal: (x_i, y_i) += scale*(ref_lengths_i*cos + ref_lengths_(i mod 4 + 1)*sin)
        vertical:   (x_i, y_i) += scale*(ref_lengths_i*sin + ref_lengths_(i mod 4 + 1)*cos)
    3. create minimum rectangle surrounding quads points and angle. these values are created by opencv's minAreaRect

    :param quads: ndarray, shape = (box nums, 8)
    :param scale: int, shrink scale
    :return: rboxes: ndarray, shape=(box nums, 9=(8=(x1,y1,...clockwise order)+1=(angle))
    """
    reshaped_quads = quads.reshape((-1, 4, 2))

    # reference lengths, clockwise from horizontal top edge
    # shape = (box nums, 4)
    ref_lengths = np.minimum(np.linalg.norm(reshaped_quads - np.roll(reshaped_quads, 1, axis=1), axis=-1),
                             np.linalg.norm(reshaped_quads - np.roll(reshaped_quads, -1, axis=1), axis=-1))

    shrinked_quads = shrink_quads_numpy(reshaped_quads, ref_lengths, scale)

    box_nums = reshaped_quads.shape[0]
    rboxes = np.zeros((box_nums, 9))  # 9=(8=(x1,y1,...clockwise order)+1=(angle))
    rboxes[:, :8] = shrinked_quads.reshape((-1, 8))
    for b in range(box_nums):
        rect = cv2.minAreaRect(reshaped_quads[b].astype(np.int))
        _, _, angle = rect
        # box = cv2.boxPoints(rect)
        angle += 90 - 45
        angle = np.deg2rad(angle)
        rboxes[b, -1] = angle

    return rboxes
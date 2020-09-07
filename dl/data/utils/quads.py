import torch, cv2
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

from .boxes import dists_pt2line_numpy, dists2corners_numpy

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

def sort_clockwise_topleft(a):
    """
    Sort corners points (xmin, ymin, xmax, ymax)
    :param a: Quads Tensor, shape is ([nums, ]*, 4=(x1,y1,x2,y2))
    :return a: Quads Tensor, shape is ([nums, ]*, 4=(xmin, ymin, xmax, ymax))
    """
    return torch.from_numpy(sort_clockwise_topleft_numpy(a.numpy()))

def sort_clockwise_topleft_numpy(a):
    """
    Sort corners points (x1, y1, x2, y2, ... clockwise from topleft)
    :ref https://stackoverflow.com/questions/10846431/ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python
    :param a: Quads ndarray, shape is (box nums, 8=(x1,y1,x2,y2,...))
    :return a: Quads ndarray, shape is (box nums, 8=(x1,y1,x2,y2,... clockwise from topleft))
    """
    box_nums = a.shape[0]
    centroids = np.concatenate((a[:, ::2].mean(axis=-1, keepdims=True),
                                a[:, 1::2].mean(axis=-1, keepdims=True)), axis=-1)
    sorted_a = np.zeros_like(a, dtype=np.float32)
    for b in range(box_nums):
        sorted_a[b] = np.array(sorted(a[b], key=lambda pt: np.arctan2(pt[1]-centroids[b,1], pt[0]-centroids[b,0])))

    return sorted_a

def quad2mask(quad, w, h, device):
    """
    :param quad: Tensor, shape = (8=(x1,y1,...)), percent style
    :param w: int
    :param h: int
    :param device: device
    :return: mask_per_rect: Bool Tensor, shape = (h, w)
    """
    return torch.from_numpy(quad2mask_numpy(quad.numpy(), w, h)).to(device=device)

def quad2mask_numpy(quad, w, h):
    """
    :param quad: ndarray, shape = (8=(x1,y1,...)), percent style
    :param w: int
    :param h: int
    :return: mask_per_rect: Bool ndarray, shape = (h, w)
    """
    _quad = quad.copy()
    _quad[::2] *= w
    _quad[1::2] *= h

    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(_quad, outline=255, fill=255)
    # img.show()
    return np.array(img, dtype=np.bool)

def quads2allmask(quads, w, h, device):
    """
    :param quads: Tensor, shape = (box num, 8=(x1,y1,...)), percent style
    :param w: int
    :param h: int
    :param device: device
    :return: mask_per_rect: Bool Tensor, shape = (h, w)
    """
    return torch.from_numpy(quads2allmask_numpy(quads.numpy(), w, h)).to(device=device)

def quads2allmask_numpy(quads, w, h):
    """
    :param quads: ndarray, shape = (box num, 8=(x1,y1,...)), percent style
    :param w: int
    :param h: int
    :return: mask_per_rect: Bool ndarray, shape = (h, w)
    """
    box_nums = quads.shape[0]

    ret = np.zeros((h, w), dtype=np.bool)
    for b in range(box_nums):
        ret = np.logical_or(ret, quad2mask_numpy(quads[b], w, h))

    return ret

def angles_from_quads_numpy(quads):
    """
    :param quads: ndarray, shape = (box nums, 8=(x1,y1,... clockwose from top-left))
    :return angles: ndarray, shape = (box nums, 1)
            Note that angle range is [-pi/4, pi/4)
    """
    box_nums = quads.shape[0]

    angles = np.zeros((box_nums, 1))
    reshaped_quads = quads.reshape((-1, 4, 2))
    for b in range(box_nums):
        rect = cv2.minAreaRect(reshaped_quads[b])
        # note that angle range is (0, 90]
        angle = -rect[-1]
        if angle == 90:
            angle = 0
        angles[b, 0] = angle if angle < 45 else -(90 - angle)
    return np.deg2rad(angles)

def shrink_quads_numpy(quads, scale=0.3):
    """
    convert quads into rbox, see fig4 in EAST paper

    Brief summary of rbox creation from quads

    1. compute reference lengths (ref_lengths) by getting shorter edge adjacent one point
    2. shrink longer edge pair* with scale value
        *: longer pair is got by comparing between two opposing edges following;
            (vertical edge1 + 2)ave <=> (horizontal edge1 + 2)ave
        Note that shrinking way is different between vertical edges pair and horizontal one
        horizontal: (x_i, y_i) += scale*(ref_lengths_i*cos + ref_lengths_(i mod 4 + 1)*sin)
        vertical:   (x_i, y_i) += scale*(ref_lengths_i*sin + ref_lengths_(i mod 4 + 1)*cos)

    :param quads: ndarray, shape = (box nums, 8=(x1,y1,...clockwise order))
    :param scale: int, shrink scale
    :return: shrinked_quads: ndarray, shape = (box nums, 8=(x1,y1,...clockwise order))
    """
    reshaped_quads = quads.reshape((-1, 4, 2))

    # reference lengths, clockwise from horizontal top edge
    # shape = (box nums, 4)
    ref_lengths = np.minimum(np.linalg.norm(reshaped_quads - np.roll(reshaped_quads, 1, axis=1), axis=-1),
                             np.linalg.norm(reshaped_quads - np.roll(reshaped_quads, -1, axis=1), axis=-1))

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
    return shrinked_quads.reshape((-1, 8))



def quads2rboxes_numpy(quads, w, h, shrink_scale=0.3):
    """
    convert quads into rbox, see fig4 in EAST paper
    https://github.com/Masao-Taketani/FOTS_OCR/blob/5c214bf2e3d815d6f826f7771da92ba4d899d08b/data_provider/data_utils.py#L575

    1. create minimum rectangle surrounding quads points and angle. these values are created by opencv's minAreaRect

    :param quads: ndarray, shape = (box nums, 8)
    :param w: int
    :param h: int
    :param shrink_scale: None or int, use raw quads to calculate dists if None or 1, use shrinked ones otherwise
    :returns:
        pos: ndarray, shape = (h, w)
        rbox: ndarray, shape=(h, w, 5=(4=(t, r, b, l)+1=(angle))
            Note that angle is between [-pi/4, pi/4)
    """
    reshaped_quads = quads.reshape((-1, 4, 2)).copy()

    reshaped_quads[:, :, 0] *= w
    reshaped_quads[:, :, 1] *= h

    box_nums, _, _ = reshaped_quads.shape

    # initialization
    rbox = np.zeros((h, w, 5), dtype=np.float32)  # shape=(h, w, 5=(4=(t, r, b, l)+1=(angle))
    pos = np.zeros((h, w), dtype=np.bool)

    # shrink
    if shrink_scale and shrink_scale != 1:
        shrunk_quads = shrink_quads_numpy(reshaped_quads.reshape(-1, 8).copy(), shrink_scale)
    else:
        shrunk_quads = reshaped_quads.reshape(-1, 8).copy()
    shrunk_quads[:, ::2] /= w
    shrunk_quads[:, 1::2] /= h

    for b in range(box_nums):
        rect = cv2.minAreaRect(reshaped_quads[b])
        # note that angle range is (0, 90]
        angle = -rect[-1]

        # shape = (4, 2)
        # clockwise from ymax point: https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return/51952289
        box_4pts = cv2.boxPoints(rect)

        # shift box_4pts for clockwise order from top-left
        # and convert angle range into [0, 90)
        if angle == 90:
            angle = 0
            # horizontal and vertical lines are parallel to x-axis and y-axis respectively
            # box is clockwise order from bottom-right, i.e. index 0 is bottom-right
            shift = -2
        elif angle < 45:
            # box is clockwise order from bottom-left, i.e. index 0 is bottom-left
            shift = -1
        else:
            # box is clockwise order from bottom-right, i.e. index 0 is bottom-right
            shift = -2
        box_4pts = np.roll(box_4pts, shift, axis=0)

        # compute distance from each point
        # >>> widths, heights = np.meshgrid(np.arange(3), np.arange(7))
        # >>> heights.shape
        # (7, 3)
        # >>> k=np.concatenate((np.expand_dims(widths, -1), np.expand_dims(heights, -1)), axis=-1)
        # >>> k[0,1,:]
        # array([1, 0])
        widths, heights = np.meshgrid(np.arange(w), np.arange(h))
        # shape = (h, w, 2)
        origins = np.concatenate((np.expand_dims(widths, -1), np.expand_dims(heights, -1)), axis=-1)
        # shape = (h, w, 4=(t,r,b,l))
        dists = dists_pt2line_numpy(box_4pts, np.roll(box_4pts, -1, axis=0), origins)

        # compute pos
        shrunk_quad = shrunk_quads[b]
        mask = quad2mask_numpy(shrunk_quad, w, h)
        pos = np.logical_or(pos, mask)

        # assign dists
        rbox[mask, :4] = dists[mask]
        # assign angle
        # the reason of below process is https://github.com/argman/EAST/issues/210
        angle = angle if angle < 45 else -(90 - angle)
        rbox[mask, -1] = np.deg2rad(angle)


    return pos, rbox

#ref https://github.com/Masao-Taketani/FOTS_OCR/blob/5c214bf2e3d815d6f826f7771da92ba4d899d08b/data_provider/data_utils.py#L498
def rboxes2quads_numpy(rboxes):
    """
    :param rboxes: ndarray, shape = (*, h, w, 5=(4=(t,r,b,l) + 1=angle))
        Note that angle is between [-pi/4, pi/4)
    :return: quads: ndarray, shape = (*, h, w, 8=(x1, y1,... clockwise order from top-left))
    """
    # dists, shape = (*, h, w, 4=(t,r,b,l))
    # angles, shape = (*, h, w)
    h, w, _ = rboxes.shape[-3:]
    dists, angles = rboxes[..., :4], rboxes[..., 4]

    # shape = (*, h, w, 5=(t,r,b,l,offset), 2=(x,y))
    pts = np.zeros(list(dists.shape[:-1]) + [5, 2], dtype=np.float32)

    # assign pts for angle >= 0
    dists_pos = dists[angles >= 0]
    if dists_pos.size > 0:
        # shape = (*, h, w)
        tops, rights, bottoms, lefts = np.rollaxis(dists_pos, axis=-1)
        shape = tops.shape
        pts[angles >= 0] = np.moveaxis(np.array([[np.zeros(shape), -(tops+bottoms)],
                                                 [lefts+rights, -(tops+bottoms)],
                                                 [lefts+rights, np.zeros(shape)],
                                                 [np.zeros(shape), np.zeros(shape)],
                                                 [lefts, -bottoms]]), [0, 1], [-2, -1])

    # assign pts for angle < 0
    dists_neg = dists[angles < 0]
    if dists_neg.size > 0:
        # shape = (*, h, w)
        tops, rights, bottoms, lefts = np.rollaxis(dists_neg, axis=-1)
        shape = tops.shape
        pts[angles < 0] = np.moveaxis(np.array([[-(lefts+rights), -(tops+bottoms)],
                                                [np.zeros(shape), -(tops+bottoms)],
                                                [np.zeros(shape), np.zeros(shape)],
                                                [-(lefts+rights), np.zeros(shape)],
                                                [-rights, -bottoms]]), [0, 1], [-2, -1])

    # note that rotate clockwise is positive, otherwise, negative
    angles *= -1

    # rotate
    # shape = (*, h, w, 2, 2)
    R = np.moveaxis(np.array([[np.cos(angles), -np.sin(angles)],
                              [np.sin(angles), np.cos(angles)]]), [0, 1], [-2, -1])
    # shape = (*, h, w, 2=(x, y), 5=(t,r,b,l,offset))
    pts = np.swapaxes(pts, -1, -2)
    # shape = (*, h, w, 2=(x, y), 5=(t,r,b,l,offset))
    rotated_pts = R @ pts

    # quads, shape = (*, h, w, 2=(x, y), 4=(t,r,b,l))
    # offsets, shape = (*, h, w, 2=(x, y), 1=(offset))
    quads, offsets = rotated_pts[..., :4], rotated_pts[..., 4:5]

    # align
    widths, heights = np.meshgrid(np.arange(w), np.arange(h))
    # shape = (h, w, 2)
    origins = np.concatenate((np.expand_dims(widths, -1), np.expand_dims(heights, -1)), axis=-1)
    # shape = (*, h, w, 2=(x,y), 1)
    origins = np.expand_dims(origins, axis=tuple(i for i in range(-1, rboxes.ndim - 3)))
    quads += origins - offsets

    quads[..., 0, :] = np.clip(quads[..., 0, :], 0, w)
    quads[..., 1, :] = np.clip(quads[..., 1, :], 0, h)

    # reshape
    quads = np.swapaxes(quads, -1, -2).reshape(list(rboxes.shape[:-1]) + [8])

    return quads
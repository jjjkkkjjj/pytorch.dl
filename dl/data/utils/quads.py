import torch, cv2
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import Polygon, MultiPoint, MultiPolygon

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

    :param reshaped_quads: ndarray, shape = (box nums, 4=(clockwise from top-left), 2=(x,y))
    :param ref_lengths: ndarray, shape = (box nums, 4)
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



def quads2rboxes_numpy(quads, angle_mode='symmetry'):
    """
    convert quads into rbox, see fig4 in EAST paper
    https://github.com/Masao-Taketani/FOTS_OCR/blob/5c214bf2e3d815d6f826f7771da92ba4d899d08b/data_provider/data_utils.py#L575

    1. create minimum rectangle surrounding quads points and angle. these values are created by opencv's minAreaRect

    :param quads: ndarray, shape = (box nums, 8)
    :param angle_mode: str, 'x-anticlock', 'y-clock', 'symmetry'
    :return: rboxes: ndarray, shape=(box nums, 9=(8=(x1,y1,...clockwise order)+1=(angle))
    Note that angle is between
        [-pi/2, 0) anti-clockwise angle from x-axis (same as opencv output) if angle_mode='x-anticlock'
        [0, pi/2) clockwise angle from y-axis if angle_mode='y-clock'
        [-pi/4, pi/4) this mode may be useful for sigmoid output? if angle_mode='symmetry'
    """
    reshaped_quads = quads.reshape((-1, 4, 2))

    box_nums = reshaped_quads.shape[0]
    rboxes = np.zeros((box_nums, 9))  # 9=(8=(x1,y1,...clockwise order)+1=(angle))

    for b in range(box_nums):
        rect = cv2.minAreaRect(reshaped_quads[b])
        angle = rect[-1]
        # shape = (4, 2)
        # clockwise from ymax point: https://stackoverflow.com/questions/29739411/what-does-cv2-cv-boxpointsrect-return/51952289
        box = cv2.boxPoints(rect)

        if angle == -90:
            # horizontal and vertical lines are parallel to x-axis and y-axis respectively
            # box is clockwise order from bottom-right, i.e. index 0 is bottom-right
            shift = 2
        elif angle < -45:
            # box is clockwise order from bottom-right, i.e. index 0 is bottom-right
            shift = 2
        else:
            # box is clockwise order from bottom-left, i.e. index 0 is bottom-left
            shift = 1

        rboxes[b, :8] = np.roll(box, shift, axis=0).reshape(8)

        angle = np.deg2rad(angle)
        if angle_mode == 'x-anticlock':
            rboxes[b, -1] = angle
        elif angle_mode == 'y-clock':
            rboxes[b, -1] = angle + np.pi/2
        elif angle_mode == 'symmetry':
            # the reason of below process is https://github.com/argman/EAST/issues/210
            # I think this process may be useful for sigmoid output?
            angle += np.pi/2
            rboxes[b, -1] = angle if angle < np.pi/4 else -(np.pi/2 - angle)

    return rboxes
import cv2, math, torch
import numpy as np

from .boxes import centroids2corners

def tensor2cvrgbimg(img, to8bit=True):
    if to8bit:
        img = img * 255.
    return img.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

def toVisualizeRectRGBimg(img, locs, thickness=2, rgb=(255, 0, 0), tensor2cvimg=True, verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param locs: Tensor, centered coordinates, shape = (box num, 4=(cx, cy, w, h)).
    :param thickness: int
    :param rgb: tuple of int, order is rgb and range is 0~255
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    # convert (c, h, w) to (h, w, c)
    if tensor2cvimg:
        img = tensor2cvrgbimg(img, to8bit=True).copy()
    else:
        if not isinstance(img, np.ndarray):
            raise ValueError('img must be Tensor, but got {}. if you pass \'Tensor\' img, set tensor2cvimg=True'.format(type(img).__name__))

    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    locs_mm = centroids2corners(locs).cpu().numpy()

    h, w, c = img.shape
    locs_mm[:, 0::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm[:, 0::2] = np.clip(locs_mm[:, 0::2], 0, w).astype(int)
    locs_mm[:, 1::2] = np.clip(locs_mm[:, 1::2], 0, h).astype(int)
    locs_mm = locs_mm.astype(int)

    if verbose:
        print(locs_mm)
    for bnum in range(locs_mm.shape[0]):
        topleft = locs_mm[bnum, :2]
        bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(topleft), tuple(bottomright))

        cv2.rectangle(img, tuple(topleft), tuple(bottomright), rgb, thickness)

    return img

def toVisualizeRectLabelRGBimg(img, locs, inf_labels, classe_labels, inf_confs=None, tensor2cvimg=True, verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param locs: Tensor, centered coordinates, shape = (box num, 4=(cx, cy, w, h)).
    :param inf_labels:
    :param classe_labels: list of str
    :param inf_confs: Tensor, (box_num,)
    :param verbose:
    :return:
    """
    # convert (c, h, w) to (h, w, c)
    if tensor2cvimg:
        img = tensor2cvrgbimg(img)
    else:
        if not isinstance(img, np.ndarray):
            raise ValueError('img must be Tensor, but got {}. if you pass \'Tensor\' img, set tensor2cvimg=True'.format(type(img).__name__))

    class_num = len(classe_labels)
    box_num = locs.shape[0]
    assert box_num == inf_labels.shape[0], 'must be same boxes number'
    if inf_confs is not None:
        if isinstance(inf_confs, torch.Tensor):
            inf_confs = inf_confs.cpu().numpy()
        elif not isinstance(inf_confs, np.ndarray):
            raise ValueError(
                'Invalid \'inf_confs\' argment were passed. inf_confs must be ndarray or Tensor, but got {}'.format(
                    type(inf_confs).__name__))
        assert inf_confs.ndim == 1 and inf_confs.size == box_num, "Invalid inf_confs"

    # color
    angles = np.linspace(0, 255, class_num).astype(np.uint8)
    # print(angles.shape)
    hsvs = np.array((0, 255, 255))[np.newaxis, np.newaxis, :].astype(np.uint8)
    hsvs = np.repeat(hsvs, class_num, axis=0)
    # print(hsvs.shape)
    hsvs[:, 0, 0] += angles
    rgbs = cv2.cvtColor(hsvs, cv2.COLOR_HSV2RGB).astype(np.int)

    # Line thickness of 2 px
    thickness = 1



    h, w, c = img.shape
    # print(locs)
    locs_mm = centroids2corners(locs).cpu().numpy()
    locs_mm[:, ::2] *= w
    locs_mm[:, 1::2] *= h
    locs_mm = locs_mm
    locs_mm[:, 0::2] = np.clip(locs_mm[:, 0::2], 0, w).astype(int)
    locs_mm[:, 1::2] = np.clip(locs_mm[:, 1::2], 0, h).astype(int)
    locs_mm = locs_mm.astype(int)

    if verbose:
        print(locs_mm)
    for bnum in range(box_num):# box num
        img = img.copy()

        rect_topleft = locs_mm[bnum, :2]
        rect_bottomright = locs_mm[bnum, 2:]

        if verbose:
            print(tuple(rect_topleft), tuple(rect_bottomright))

        index = inf_labels[bnum].item()
        if math.isnan(index):
            continue
        index = int(index)

        if inf_confs is not None:
            text = classe_labels[index] + ':{:.2f}'.format(inf_confs[bnum])
        else:
            text = classe_labels[index]

        labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)

        rect_bottomright = tuple(rect_bottomright)
        rect_topleft = tuple(rect_topleft)
        rgb = tuple(rgbs[index, 0].tolist())

        # text area
        text_bottomleft = (rect_topleft[0], rect_topleft[1] + int(labelSize[0][1] * 1.5))
        text_topright = (rect_topleft[0] + labelSize[0][0], rect_topleft[1])
        cv2.rectangle(img, text_bottomleft, text_topright, rgb, cv2.FILLED)

        text_bottomleft = (rect_topleft[0], rect_topleft[1] + labelSize[0][1])
        cv2.putText(img, text, text_bottomleft, cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)

        # rectangle
        cv2.rectangle(img, rect_topleft, rect_bottomright, rgb, thickness)

    return img

def toVisualizeQuadsRGBimg(img, poly_pts, thickness=2, rgb=(255, 0, 0), verbose=False, tensor2cvimg=True):
    """
    :param img: Tensor, shape = (c, h, w)
    :param poly_pts: list of Tensor, centered coordinates, shape = (box num, ?*2=(x1, y1, x2, y2,...)).
    :param thickness: int
    :param rgb: tuple of int, order is rgb and range is 0~255
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    if tensor2cvimg:
        img = tensor2cvrgbimg(img)
    else:
        if not isinstance(img, np.ndarray):
            raise ValueError('img must be Tensor, but got {}. if you pass \'Tensor\' img, set tensor2cvimg=True'.format(
                type(img).__name__))

    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    poly_pts_mm = poly_pts.detach().numpy()

    h, w, c = img.shape

    if verbose:
        print(poly_pts)
    for bnum in range(poly_pts_mm.shape[0]):
        img = img.copy()
        pts = poly_pts_mm[bnum]
        pts[0::2] *= w
        pts[1::2] *= h
        pts[0::2] = np.clip(pts[0::2], 0, w)
        pts[1::2] = np.clip(pts[1::2], 0, h)
        #print(pts)
        pts = pts.reshape((-1, 1, 2)).astype(int)
        #print(pts)
        if verbose:
            print(pts)
        cv2.polylines(img, [pts], isClosed=True, color=rgb, thickness=thickness)

    return img

def toVisualizeQuadsLabelRGBimg(img, poly_pts, inf_labels, classe_labels, inf_confs=None, tensor2cvimg=True, verbose=False):
    """
    :param img: Tensor, shape = (c, h, w)
    :param poly_pts: list of Tensor, centered coordinates, shape = (box num, ?*2=(x1, y1, x2, y2,...)).
    :param inf_labels:
    :param classe_labels: list of str
    :param inf_confs: Tensor, (box_num,)
    :param tensor2cvimg: bool, whether to convert Tensor to cvimg
    :param verbose: bool, whether to show information
    :return:
        img: RGB order
    """
    # convert (c, h, w) to (h, w, c)
    if tensor2cvimg:
        img = tensor2cvrgbimg(img, to8bit=True).copy()
    else:
        if not isinstance(img, np.ndarray):
            raise ValueError('img must be Tensor, but got {}. if you pass \'Tensor\' img, set tensor2cvimg=True'.format(type(img).__name__))


    #cv2.imshow('a', img)
    #cv2.waitKey()
    # print(locs)
    poly_pts_mm = poly_pts.cpu().numpy()

    class_num = len(classe_labels)
    box_num = poly_pts.shape[0]
    assert box_num == inf_labels.shape[0], 'must be same boxes number'
    if inf_confs is not None:
        if isinstance(inf_confs, torch.Tensor):
            inf_confs = inf_confs.cpu().numpy()
        elif not isinstance(inf_confs, np.ndarray):
            raise ValueError(
                'Invalid \'inf_confs\' argment were passed. inf_confs must be ndarray or Tensor, but got {}'.format(
                    type(inf_confs).__name__))
        assert inf_confs.ndim == 1 and inf_confs.size == box_num, "Invalid inf_confs"

    # color
    angles = np.linspace(0, 255, class_num).astype(np.uint8)
    # print(angles.shape)
    hsvs = np.array((0, 255, 255))[np.newaxis, np.newaxis, :].astype(np.uint8)
    hsvs = np.repeat(hsvs, class_num, axis=0)
    # print(hsvs.shape)
    hsvs[:, 0, 0] += angles
    rgbs = cv2.cvtColor(hsvs, cv2.COLOR_HSV2RGB).astype(np.int)

    # Line thickness of 2 px
    thickness = 1


    h, w, c = img.shape

    if verbose:
        print(poly_pts)
    for bnum in range(poly_pts_mm.shape[0]):
        img = img.copy()
        pts = poly_pts_mm[bnum]
        pts[0::2] *= w
        pts[1::2] *= h
        pts[0::2] = np.clip(pts[0::2], 0, w)
        pts[1::2] = np.clip(pts[1::2], 0, h)
        #print(pts)
        pts = pts.reshape((-1, 1, 2)).astype(int)
        #print(pts)

        index = inf_labels[bnum].item()
        if math.isnan(index):
            continue
        index = int(index)

        rgb = tuple(rgbs[index, 0].tolist())

        if verbose:
            print(pts)
        cv2.polylines(img, [pts], isClosed=True, color=rgb, thickness=thickness)

    return img
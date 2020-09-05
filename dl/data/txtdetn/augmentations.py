from numpy import random
import numpy as np
import logging, cv2

from .._utils import _check_ins
from ...data.utils.boxes import iou_numpy, coverage_numpy, corners2centroids_numpy
from ..objrecog.augmentations import *

class RandomRotate(object):
    def __init__(self, fill_rgb=(103.939, 116.779, 123.68), center=(0, 0), amin=-10, amax=10, fit=True, p=0.5):
        self.fill_rgb = fill_rgb
        self.center = center
        self.amin = amin
        self.amax = amax
        self.fit = fit
        self.p = p

    def __call__(self, img, labels, bboxes, flags, quads, texts):
        """
        :param img: ndarray
        :param bboxes: ndarray, shape = (box num, 4=(xmin, ymin, xmax, ymax))
        :param labels: ndarray, shape = (box num, class num)
        :param flags: list of dict, whose length is box num
        :param quads: ndarray, shape = (box num, 8=(top-left(x,y),... clockwise))
        :param texts: list of str, whose length is box num
        :return:
        """
        if decision(self.p):
            h, w, _ = img.shape

            quads[:, ::2] *= w
            quads[:, 1::2] *= h

            box_nums = bboxes.shape[0]
            # shape = (box nums, 3=(x,y,1), 4)
            mat_quads = np.concatenate((np.swapaxes(quads.reshape((-1, 4, 2)), -2, -1),
                                        np.ones((box_nums, 1, 4))), axis=1)

            angle = random.uniform(self.amin, self.amax)
            if self.fit:
                radian = np.radians(angle)
                sine = np.abs(np.sin(radian))
                cosine = np.abs(np.cos(radian))
                tri_mat = np.array([[cosine, sine], [sine, cosine]], np.float32)
                old_size = np.array([w, h], np.float32)
                new_size = np.ravel(tri_mat @ old_size.reshape(-1, 1))


                affine = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                # move
                affine[:2, 2] += (new_size - old_size) / 2.0
                # resize
                affine[:2, :] *= (old_size / new_size).reshape(-1, 1)
            else:
                affine = cv2.getRotationMatrix2D(self.center, angle, 1.0)

            img = cv2.warpAffine(img, affine, (w, h), borderValue=self.fill_rgb)
            R = np.concatenate((affine, np.array([[0, 0, 1]])), axis=0)

            mat_quads = R @ mat_quads
            # shape = (box nums, 4, 2=(x,y))
            mat_quads = np.swapaxes(mat_quads[..., :2, :], -2, -1)

            quads = mat_quads.reshape(box_nums, 8).astype(np.float32)

            quads[:, ::2] /= w
            quads[:, 1::2] /= h

            # xmin and ymin
            bboxes[:, 0:2] = np.min(mat_quads, axis=1)
            # xmax and ymax
            bboxes[:, 2:4] = np.max(mat_quads, axis=1)

        return img, (labels, bboxes, flags, quads, texts)

class _SampledPatchOp(object):
    class UnSatisfy(Exception):
        pass


class EntireSample(_SampledPatchOp):
    def __call__(self, img, labels, bboxes, flags, *args):
        return img, (labels, bboxes, flags, *args)


class RandomThresSampledPatch(_SampledPatchOp):
    def __init__(self, overlap_func=iou_numpy, thres_min=None, thres_max=None, ar_min=0.5, ar_max=2):
        """
        :param overlap_func: overlap function. Argument must be (bboxes, patch)
        :param thres_min: float or None, if it's None, set thres_min as -inf
        :param thres_max: float or None, if it's None, set thres_max as inf
        :param ar_min: float, if it's None, set thres_min as -inf
        :param ar_max: float, if it's None, set thres_max as inf
        """
        self.overlap_func = overlap_func
        self.thres_min = _check_ins('thres_min', thres_min, float, allow_none=True, default=float('-inf'))
        self.thres_max = _check_ins('thres_max', thres_max, float, allow_none=True, default=float('inf'))
        self.aspect_ration_min = _check_ins('ar_min', ar_min, (float, int))
        self.aspect_ration_max = _check_ins('ar_max', ar_max, (float, int))

    def __call__(self, img, labels, bboxes, flags, quads, texts):
        """
        :param img: ndarray
        :param bboxes: ndarray, shape = (box num, 4=(xmin, ymin, xmax, ymax))
        :param labels: ndarray, shape = (box num, class num)
        :param flags: list of dict, whose length is box num
        :param quads: ndarray, shape = (box num, 8=(top-left(x,y),... clockwise))
        :param texts: list of str, whose length is box num
        :return:
        """
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        h, w, _ = img.shape

        ret_img = img.copy()
        ret_bboxes = bboxes.copy()
        ret_quads = quads.copy()
        ret_texts = np.array(texts) # convert list to ndarray to mask

        # get patch width and height, and aspect ratio randomly
        patch_w = random.randint(int(0.3 * w), w)
        patch_h = random.randint(int(0.3 * h), h)
        aspect_ratio = patch_h / float(patch_w)

        # aspect ratio constraint b/t .5 & 2
        if not (aspect_ratio >= 0.5 and aspect_ratio <= 2):
            raise _SampledPatchOp.UnSatisfy
        # aspect_ratio = random.uniform(self.aspect_ration_min, self.aspect_ration_max)

        # patch_h, patch_w = int(aspect_ratio*h), int(aspect_ratio*w)
        patch_topleft_x = random.randint(w - patch_w)
        patch_topleft_y = random.randint(h - patch_h)
        # shape = (1, 4)
        patch = np.array((patch_topleft_x, patch_topleft_y,
                          patch_topleft_x + patch_w, patch_topleft_y + patch_h))
        patch = np.expand_dims(patch, 0)

        # e.g.) IoU
        overlaps = self.overlap_func(bboxes, patch)
        if overlaps.min() < self.thres_min and overlaps.max() > self.thres_max:
            raise _SampledPatchOp.UnSatisfy
            # return None

        # cut patch
        ret_img = ret_img[patch_topleft_y:patch_topleft_y + patch_h, patch_topleft_x:patch_topleft_x + patch_w]

        # reconstruct box coordinates
        ret_bboxes[:, 0::2] *= float(w)
        ret_bboxes[:, 1::2] *= float(h)

        ret_quads[:, 0::2] *= float(w)
        ret_quads[:, 1::2] *= float(h)

        # convert minmax to centroids coordinates of bboxes
        # shape = (*, 4=(cx, cy, w, h))
        centroids_boxes = corners2centroids_numpy(ret_bboxes)

        # check if centroids of boxes is in patch
        mask_box = (centroids_boxes[:, 0] > patch_topleft_x) * (centroids_boxes[:, 0] < patch_topleft_x + patch_w) * \
                   (centroids_boxes[:, 1] > patch_topleft_y) * (centroids_boxes[:, 1] < patch_topleft_y + patch_h)
        if not mask_box.any():
            raise _SampledPatchOp.UnSatisfy
            # return None

        # filtered out the boxes with unsatisfied above condition
        ret_bboxes = ret_bboxes[mask_box, :]
        ret_labels = labels[mask_box]
        ret_quads = ret_quads[mask_box, :]
        ret_texts = ret_texts[mask_box]

        # adjust boxes within patch
        ret_bboxes[:, :2] = np.maximum(ret_bboxes[:, :2], patch[:, :2])
        ret_bboxes[:, 2:] = np.minimum(ret_bboxes[:, 2:], patch[:, 2:])

        # boxes, patch => xmin, ymin, xmax, ymax
        # quads => x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl
        ret_quads[:, 0::2] = np.clip(ret_quads[:, 0::2], a_min=patch[:, 0], a_max=patch[:, 2])
        ret_quads[:, 1::2] = np.clip(ret_quads[:, 1::2], a_min=patch[:, 1], a_max=patch[:, 3])

        # move new position
        ret_bboxes -= np.tile(patch[:, :2], reps=(2))

        ret_quads -= np.tile(patch[:, :2], reps=(4))

        # to percent
        ret_bboxes[:, 0::2] /= float(patch_w)
        ret_bboxes[:, 1::2] /= float(patch_h)

        ret_quads[:, 0::2] /= float(patch_w)
        ret_quads[:, 1::2] /= float(patch_h)

        return ret_img, (ret_labels, ret_bboxes, flags, ret_quads, ret_texts.tolist())


class RandomIoUSampledPatch(RandomThresSampledPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(iou_numpy, *args, **kwargs)

class RandomCoverageSampledPatch(RandomThresSampledPatch):
    def __init__(self, *args, **kwargs):
        super().__init__(coverage_numpy, *args, **kwargs)

class RandomSampledPatch(RandomThresSampledPatch):
    def __init__(self):
        super().__init__(iou_numpy, None, None)

class RandomSampled(object):
    def __init__(self, options=(
            EntireSample(),
            RandomIoUSampledPatch(0.1, None),
            RandomIoUSampledPatch(0.3, None),
            RandomIoUSampledPatch(0.5, None),
            RandomIoUSampledPatch(0.7, None),
            RandomIoUSampledPatch(0.9, None),
            RandomCoverageSampledPatch(0.1, None),
            RandomCoverageSampledPatch(0.3, None),
            RandomCoverageSampledPatch(0.5, None),
            RandomCoverageSampledPatch(0.7, None),
            RandomCoverageSampledPatch(0.9, None),
            RandomSampledPatch()
    ), max_iteration=20):

        # check argument is proper
        for op in options:
            if not isinstance(op, _SampledPatchOp):
                raise ValueError('All of option\'s element must be inherited to _SampledPatchOp')

        if not any([isinstance(op, EntireSample) for op in options]):
            logging.warning("Option does not contain entire sample. Could not return value in worst case")

        self.options = options
        self.max_iteration = max_iteration

    def __call__(self, img, labels, bboxes, flags, *args):
        import time
        s = time.time()
        while True:
            # select option randomly
            op = random.choice(self.options)

            if isinstance(op, EntireSample):
                return op(img, labels, bboxes, flags, *args)

            for _ in range(self.max_iteration):
                try:
                    return op(img, labels, bboxes, flags, *args)
                except _SampledPatchOp.UnSatisfy:
                    continue

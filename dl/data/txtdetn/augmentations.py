from numpy import random
import numpy as np
import logging

from .._utils import _check_ins
from ...models.ssd.core.boxes.utils import iou_numpy, coverage_numpy, corners2centroids_numpy
from ..objdetn.augmentations import Compose

class _SampledPatchOp(object):
    class UnSatisfy(Exception):
        pass


class EntireSample(_SampledPatchOp):
    def __call__(self, img, bboxes, labels, flags, *args):
        return img, (bboxes, labels, flags, *args)


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

    def __call__(self, img, bboxes, labels, flags, *args):
        """
        :param img: ndarray
        :param bboxes: ndarray, shape = (box num, 4=(xmin, ymin, xmax, ymax))
        :param labels: ndarray, shape = (box num, class num)
        :param flags: list of dict, whose length is box num
        :param args:
                quads: ndarray, shape = (box num, 8=(top-left(x,y),... clockwise))
                texts: list of str, whose length is box num
        :return:
        """
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        h, w, _ = img.shape

        ret_img = img.copy()
        ret_bboxes = bboxes.copy()
        ret_quads = args[0].copy()
        ret_texts = np.array(args[1]) # convert list to ndarray to mask

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

        return ret_img, (ret_bboxes, ret_labels, flags, ret_quads, ret_texts.tolist())


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

    def __call__(self, img, bboxes, labels, flags, *args):
        import time
        s = time.time()
        while True:
            # select option randomly
            op = random.choice(self.options)

            if isinstance(op, EntireSample):
                return op(img, bboxes, labels, flags, *args)

            for _ in range(self.max_iteration):
                try:
                    return op(img, bboxes, labels, flags, *args)
                except _SampledPatchOp.UnSatisfy:
                    continue

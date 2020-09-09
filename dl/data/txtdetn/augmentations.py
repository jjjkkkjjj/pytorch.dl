from numpy import random
import numpy as np
import logging, cv2

from .._utils import _check_ins
from ...data.utils.boxes import iou_numpy, coverage_numpy, corners2centroids_numpy
from ...data.utils.points import apply_affine
from ...data.utils.quads import quads2allmask_numpy
from ..objrecog.augmentations import *
from ..objdetn.augmentations import (
    RandomExpand,
    RandomFlip,
    RandomScaleH,
    RandomScaleV
)


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

            box_nums = bboxes.shape[0]

            angle = random.uniform(self.amin, self.amax)
            if self.fit:
                radian = np.radians(angle)
                sine = np.sin(radian)
                cosine = np.cos(radian)
                tri_mat = np.array([[cosine, sine], [-sine, cosine]], np.float32)
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

            # shape = (box nums, 4, 2=(x,y))
            affined_quads = apply_affine(affine, (w, h), (w, h), quads.reshape(-1, 4, 2))

            quads = affined_quads.reshape(box_nums, 8)

            # xmin and ymin
            bboxes[:, 0:2] = np.min(affined_quads, axis=1)
            # xmax and ymax
            bboxes[:, 2:4] = np.max(affined_quads, axis=1)

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
        ret_flags = np.array(flags)  # convert list to ndarray to mask
        ret_texts = np.array(texts)  # convert list to ndarray to mask

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
        ret_flags = ret_flags[mask_box]
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

        return ret_img, (ret_labels, ret_bboxes, ret_flags.tolist(), ret_quads, ret_texts.tolist())


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
    ), max_iteration=50):

        # check argument is proper
        for op in options:
            if not isinstance(op, _SampledPatchOp):
                raise ValueError('All of option\'s element must be inherited to _SampledPatchOp')

        if not any([isinstance(op, EntireSample) for op in options]):
            logging.warning("Option does not contain EntireSample. May not be able to return value in worst case")

        self.options = options
        self.max_iteration = max_iteration

    def __call__(self, img, labels, bboxes, flags, *args):
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


class RandomSimpleCropPatch(_SampledPatchOp):
    def __init__(self, thres_ratio=0.1, padding=None):
        """
        :param thres_ratio: int or float
        :param padding: None or int, this argument means cropping entirely. when this argument is big, cropping entirely is done more easily.
                        padding values are quotient by 10 of h and w respectively if it's None.
        """
        self.thres_ratio = _check_ins('thres_ratio', thres_ratio, (int, float))
        self.padding = _check_ins('padding', padding, int, allow_none=True)

    def __call__(self, img, labels, bboxes, flags, quads, texts):
        h, w, _ = img.shape

        ret_bboxes = bboxes.copy()
        ret_quads = quads.copy()
        ret_flags = np.array(flags)  # convert list to ndarray to mask
        ret_texts = np.array(texts)  # convert list to ndarray to mask

        mask = quads2allmask_numpy(quads, w, h)

        # reconstruct bboxes and quads
        ret_bboxes[:, ::2] *= w
        ret_bboxes[:, 1::2] *= h
        ret_quads[:, ::2] *= w
        ret_quads[:, 1::2] *= h

        # text flag, whose true means non-text flag
        nontxtflag_h = np.logical_not(np.any(mask, axis=1))  # shape = (h,)
        nontxtflag_w = np.logical_not(np.any(mask, axis=0))  # shape = (w,)

        # insert flag for cropping entirely
        if self.padding:
            pad_w, pad_h = self.padding, self.padding
        else:
            pad_w, pad_h = w // 10, h // 10
        nontxtflag_h = np.insert(nontxtflag_h, h, [True] * pad_h)
        nontxtflag_h = np.insert(nontxtflag_h, 0, [True] * pad_h)  # shape = (h+2*pad_h,)
        nontxtflag_w = np.insert(nontxtflag_w, w, [True] * pad_w)
        nontxtflag_w = np.insert(nontxtflag_w, 0, [True] * pad_w)  # shape = (w+2*pad_w,)

        # search non-text coordinates
        nontxt_h_inds = np.where(nontxtflag_h)[0]
        nontxt_w_inds = np.where(nontxtflag_w)[0]

        # select 2 coordinates randomly
        # note that -pad_[h or w] means revert correct coordinates of boxes(quads) for inserting flag previously
        selected_x = random.choice(nontxt_w_inds, size=2) - pad_w
        selected_y = random.choice(nontxt_h_inds, size=2) - pad_h

        selected_x = np.clip(selected_x, 0, w)
        selected_y = np.clip(selected_y, 0, h)

        cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax = selected_x.min(), selected_y.min(), selected_x.max(), selected_y.max()
        new_w, new_h = cropped_xmax - cropped_xmin, cropped_ymax - cropped_ymin
        if new_w < self.thres_ratio * w or new_h < self.thres_ratio * h:
            # too small
            raise _SampledPatchOp.UnSatisfy

        # avoid tiny error
        ret_bboxes[:, ::2] = np.clip(ret_bboxes[:, ::2], 0, w)
        ret_bboxes[:, 1::2] = np.clip(ret_bboxes[:, 1::2], 0, h)

        # count up boxes inside cropped box
        insidebox_inds = (ret_bboxes[:, 0] >= cropped_xmin) & \
                         (ret_bboxes[:, 1] >= cropped_ymin) & \
                         (ret_bboxes[:, 2] < cropped_xmax) & \
                         (ret_bboxes[:, 3] < cropped_ymax)

        if insidebox_inds.sum() == 0:
            raise _SampledPatchOp.UnSatisfy

        img = img[cropped_ymin:cropped_ymax, cropped_xmin:cropped_xmax]

        # move and convert to percent
        ret_bboxes[:, ::2] = (ret_bboxes[:, ::2] - cropped_xmin)/new_w
        ret_bboxes[:, 1::2] = (ret_bboxes[:, 1::2] - cropped_ymin)/new_h
        ret_quads[:, ::2] = (ret_quads[:, ::2] - cropped_xmin)/new_w
        ret_quads[:, 1::2] = (ret_quads[:, 1::2] - cropped_ymin)/new_h

        # cut off boxes outside cropped box
        ret_bboxes = ret_bboxes[insidebox_inds]
        ret_labels = labels[insidebox_inds]
        ret_quads = ret_quads[insidebox_inds]
        ret_flags = ret_flags[insidebox_inds]
        ret_texts = ret_texts[insidebox_inds]

        return img, (ret_labels, ret_bboxes, ret_flags.tolist(), ret_quads, ret_texts.tolist())


class RandomSimpleCrop(RandomSampled):
    def __init__(self, options=None, max_iteration=50):
        if options is None:
            options = (EntireSample(),
                       RandomSimpleCropPatch(thres_ratio=0.1),)
        super().__init__(options=options, max_iteration=max_iteration)

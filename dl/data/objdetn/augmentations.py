from numpy import random
import numpy as np
import logging

from dl.data.base.augmentations import decision, Compose
from dl.data.objrecog.augmentations import *
from ...data.utils.boxes import iou_numpy, corners2centroids_numpy, sort_corners_numpy
from ...data.utils.quads import sort_clockwise_topleft_numpy
from ...data.utils.points import apply_affine

"""
IMPORTANT: augmentation will be ran before transform and target_transform

ref > http://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
"""
class RandomExpand(object):
    def __init__(self, filled_rgb=(103.939, 116.779, 123.68), rmin=1, rmax=4, p=0.5):
        self.filled_rgb = filled_rgb
        self.ratio_min = rmin
        self.ratio_max = rmax
        self.p = p

        assert self.ratio_min >= 0, "must be more than 0"
        assert self.ratio_max >= self.ratio_min, "must be more than factor min"

    def __call__(self, img, labels, bboxes, flags, *args):
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        if decision(self.p):
            h, w, c = img.shape
            # get ratio randomly
            ratio = random.uniform(self.ratio_min, self.ratio_max)

            new_h = int(h*ratio)
            new_w = int(w*ratio)

            # get top left coordinates of original image randomly
            topleft_x = int(random.uniform(0, new_w - w))
            topleft_y = int(random.uniform(0, new_h - h))

            affine = cv2.getAffineTransform(src=np.array([[0,0], [0,h], [w,0]], dtype=np.float32),
                                            dst=np.array([[topleft_x, topleft_y], [topleft_x, topleft_y+h], [topleft_x+w, topleft_y]], dtype=np.float32))
            img = cv2.warpAffine(img, affine, (new_w, new_h), borderValue=self.filled_rgb)

            if len(args) > 0 and isinstance(args[0], np.ndarray):
                quads = args[0]
                bboxes, quads = apply_affine(affine, (w, h), (new_w, new_h), bboxes.reshape(-1, 2, 2), quads.reshape(-1, 4, 2))

                bboxes = bboxes.reshape((-1, 8))
                quads = quads.reshape((-1, 8))
                return img, (labels, bboxes, flags, quads, *args[1:])
            else:
                bboxes = apply_affine(affine, (w, h), (new_w, new_h), bboxes.reshape(-1, 2, 2)).reshape((-1, 4))

            """
            # filled with normalized mean value
            new_img = np.ones((new_h, new_w, c)) * np.broadcast_to(self.filled_rgb, shape=(1, 1, c))

            # put original image to selected topleft coordinates
            new_img[topleft_y:topleft_y+h, topleft_x:topleft_x+w] = img
            img = new_img

            # convert box coordinates
            # bbox shape = (*, 4=(xmin, ymin, xmax, ymax))
            # reconstruct original coordinates
            bboxes[:, 0::2] *= float(w)
            bboxes[:, 1::2] *= float(h)
            # move new position
            bboxes[:, 0::2] += topleft_x
            bboxes[:, 1::2] += topleft_y
            # to percent
            bboxes[:, 0::2] /= float(new_w)
            bboxes[:, 1::2] /= float(new_h)
            """

        return img, (labels, bboxes, flags, *args)

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels, bboxes, flags, *args):
        if decision(self.p):
            h, w, _ = img.shape
            """
            copy because ->>>>
            ValueError: some of the strides of a given numpy array are negative.
             This is currently not supported, but will be added in future releases.
            """

            """
            img = img[:, ::-1].copy()

            ret_bboxes = bboxes.copy()
            ret_bboxes[:, 0::2] = 1 - ret_bboxes[:, 2::-2]
            bboxes = ret_bboxes.clip(min=0, max=1)
            """
            src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
            dst = src.copy()
            dst[:, 0] = w - src[:, 0]
            affine = cv2.getAffineTransform(src=src, dst=dst)

            img = cv2.warpAffine(img, affine, (w, h))

            if len(args) > 0 and isinstance(args[0], np.ndarray):
                quads = args[0]
                bboxes, quads = apply_affine(affine, (w, h), (w, h), bboxes.reshape(-1, 2, 2), quads.reshape(-1, 4, 2))

                bboxes = bboxes.reshape((-1, 4))
                # re-sort to (xmin,ymin,xmax,ymax)
                bboxes = sort_corners_numpy(bboxes)

                quads = quads.reshape((-1, 8))
                # re-sort to (x1,y1,x2,y2,... clockwise from topleft)
                quads = sort_clockwise_topleft_numpy(quads)

                return img, (labels, bboxes, flags, quads, *args[1:])
            else:
                bboxes = apply_affine(affine, (w, h), (w, h), bboxes.reshape((-1, 2, 2))).reshape(-1, 4)
                # re-sort to (xmin,ymin,xmax,ymax)
                bboxes = sort_corners_numpy(bboxes)


        return img, (labels, bboxes, flags, *args)

class RandomScaleH(object):
    def __init__(self, smin=0.8, smax=1.2, keep_aspect=True, p=0.5):
        self.smin = smin
        self.smax = smax
        self.keep_aspect = keep_aspect
        self.p = p

    def __call__(self, img, labels, bboxes, flags, *args):
        if decision(self.p):
            h, w, _ = img.shape
            scaleh = random.uniform(self.smin, self.smax)
            scalev = scaleh if self.keep_aspect else 1
            #print(img.shape)
            img = cv2.resize(img, dsize=None, fx=scaleh, fy=scalev)
            #print(img.shape)
        return img, (labels, bboxes, flags, *args)

class RandomScaleV(object):
    def __init__(self, smin=0.8, smax=1.2, keep_aspect=True, p=0.5):
        self.smin = smin
        self.smax = smax
        self.keep_aspect = keep_aspect
        self.p = p

    def __call__(self, img, labels, bboxes, flags, *args):
        if decision(self.p):
            h, w, _ = img.shape
            scalev = random.uniform(self.smin, self.smax)
            scaleh = scalev if self.keep_aspect else 1
            #print(img.shape)
            img = cv2.resize(img, dsize=None, fx=scaleh, fy=scalev)
            #print(img.shape)
        return img, (labels, bboxes, flags, *args)

class _SampledPatchOp(object):
    class UnSatisfy(Exception):
        pass

class EntireSample(_SampledPatchOp):
    def __call__(self, img, labels, bboxes, flags, *args):
        return img, (labels, bboxes, flags, *args)

class RandomIoUSampledPatch(_SampledPatchOp):
    def __init__(self, iou_min=None, iou_max=None, ar_min=0.5, ar_max=2):
        """
        :param iou_min: float or None, if it's None, set iou_min as -inf
        :param iou_max: float or None, if it's None, set iou_max as inf
        """
        self.iou_min = iou_min if iou_min else float('-inf')
        self.iou_max = iou_max if iou_max else float('inf')
        self.aspect_ration_min = ar_min
        self.aspect_ration_max = ar_max

    def __call__(self, img, labels, bboxes, flags, *args):
        # IMPORTANT: img = rgb order, bboxes: minmax coordinates with PERCENT
        h, w, _ = img.shape

        ret_img = img.copy()
        ret_bboxes = bboxes.copy()
        ret_flags = np.array(flags)  # convert list to ndarray to mask

        # get patch width and height, and aspect ratio randomly
        patch_w = random.randint(int(0.3 * w), w)
        patch_h = random.randint(int(0.3 * h), h)
        aspect_ratio = patch_h / float(patch_w)

        # aspect ratio constraint b/t .5 & 2
        if not (aspect_ratio >= 0.5 and aspect_ratio <= 2):
            raise _SampledPatchOp.UnSatisfy
        #aspect_ratio = random.uniform(self.aspect_ration_min, self.aspect_ration_max)

        #patch_h, patch_w = int(aspect_ratio*h), int(aspect_ratio*w)
        patch_topleft_x = random.randint(w - patch_w)
        patch_topleft_y = random.randint(h - patch_h)
        # shape = (1, 4)
        patch = np.array((patch_topleft_x, patch_topleft_y,
                          patch_topleft_x + patch_w, patch_topleft_y + patch_h))
        patch = np.expand_dims(patch, 0)

        # IoU
        overlaps = iou_numpy(bboxes, patch)
        if overlaps.min() < self.iou_min and overlaps.max() > self.iou_max:
            raise _SampledPatchOp.UnSatisfy
            #return None

        # cut patch
        ret_img = ret_img[patch_topleft_y:patch_topleft_y+patch_h, patch_topleft_x:patch_topleft_x+patch_w]

        # reconstruct box coordinates
        ret_bboxes[:, 0::2] *= float(w)
        ret_bboxes[:, 1::2] *= float(h)

        # convert minmax to centroids coordinates of bboxes
        # shape = (*, 4=(cx, cy, w, h))
        centroids_boxes = corners2centroids_numpy(ret_bboxes)

        # check if centroids of boxes is in patch
        mask_box = (centroids_boxes[:, 0] > patch_topleft_x) * (centroids_boxes[:, 0] < patch_topleft_x+patch_w) *\
                   (centroids_boxes[:, 1] > patch_topleft_y) * (centroids_boxes[:, 1] < patch_topleft_y+patch_h)
        if not mask_box.any():
            raise _SampledPatchOp.UnSatisfy
            #return None

        # filtered out the boxes with unsatisfied above condition
        ret_bboxes = ret_bboxes[mask_box, :].copy()
        ret_labels = labels[mask_box]
        ret_flags = ret_flags[mask_box]

        # adjust boxes within patch
        ret_bboxes[:, :2] = np.maximum(ret_bboxes[:, :2], patch[:, :2])
        ret_bboxes[:, 2:] = np.minimum(ret_bboxes[:, 2:], patch[:, 2:])

        # move new position
        ret_bboxes[:, :2] -= patch[:, :2]
        ret_bboxes[:, 2:] -= patch[:, :2]

        # to percent
        ret_bboxes[:, 0::2] /= float(patch_w)
        ret_bboxes[:, 1::2] /= float(patch_h)

        return ret_img, (ret_labels, ret_bboxes, ret_flags.tolist(), *args)

class RandomSampledPatch(RandomIoUSampledPatch):
    def __init__(self):
        super().__init__(None, None)


class RandomSampled(object):
    def __init__(self, options=(
            EntireSample(),
            RandomIoUSampledPatch(0.1, None),
            RandomIoUSampledPatch(0.3, None),
            RandomIoUSampledPatch(0.5, None),
            RandomIoUSampledPatch(0.7, None),
            RandomIoUSampledPatch(0.9, None),
            RandomSampledPatch()
            ), max_iteration=50):

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
                return op(img, labels, bboxes, flags, args)

            for _ in range(self.max_iteration):
                try:
                    return op(img, labels, bboxes, flags, args)
                except _SampledPatchOp.UnSatisfy:
                    continue
                """
                ret = op(img, labels, bboxes, flags)
                if ret:
                    print(time.time()-s)
                    return ret
                """


class GeometricDistortions(Compose):
    def __init__(self):
        gmdists = [
            RandomExpand(),
            RandomSampled(),
            RandomFlip()
        ]
        super().__init__(gmdists)

class AugmentationOriginal(Compose):
    def __init__(self):
        augmentations = [
            PhotometricDistortions(),
            GeometricDistortions()
        ]
        super().__init__(augmentations)
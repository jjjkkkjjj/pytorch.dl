from ..base.augmentations import decision, Compose
from numpy import random
import numpy as np
import cv2, logging
from itertools import permutations

class RandomBrightness(object):
    def __init__(self, dmin=-32, dmax=32, p=0.5):
        self.delta_min = dmin
        self.delta_max = dmax
        self.p = p

        assert abs(self.delta_min) >= 0 and abs(self.delta_min) < 256, "must be range between -255 and 255"
        assert abs(self.delta_max) >= 0 and abs(self.delta_max) < 256, "must be range between -255 and 255"
        assert self.delta_max >= self.delta_min, "must be more than delta min"

    def __call__(self, img, *targets):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            delta = random.uniform(self.delta_min, self.delta_max)
            img += delta
            img = np.clip(img, a_min=0, a_max=255)

        return img, (*targets,)

class RandomContrast(object):
    def __init__(self, fmin=0.5, fmax=1.5, p=0.5):
        self.factor_min = fmin
        self.factor_max = fmax
        self.p = p

        assert self.factor_min >= 0, "must be more than 0"
        assert self.factor_max >= self.factor_min, "must be more than factor min"

    def __call__(self, img, *targets):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            factor = random.uniform(self.factor_min, self.factor_max)
            img *= factor
            img = np.clip(img, a_min=0, a_max=255)

        return img, (*targets,)

class RandomHue(object):
    def __init__(self, dmin=-18, dmax=18, p=0.5):
        self.delta_min = dmin
        self.delta_max = dmax
        self.p = p

        assert abs(self.delta_min) >= 0 and abs(self.delta_min) < 180, "must be range between -179 and 179"
        assert abs(self.delta_max) >= 0 and abs(self.delta_max) < 180, "must be range between -179 and 179"
        assert self.delta_max >= self.delta_min, "must be more than delta min"

    def __call__(self, img, *targets):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            delta = random.uniform(self.delta_min, self.delta_max)
            img[:, :, 0] += delta

            # clip 0 to 180, note that opencv's hue range is [0, 180]
            over_mask = img[:, :, 0] >= 180
            img[over_mask, 0] -= 180

            under_mask = img[:, :, 0] < 0
            img[under_mask, 0] += 180

        return img, (*targets,)

class RandomSaturation(object):
    def __init__(self, fmin=0.5, fmax=1.5, p=0.5):
        self.factor_min = fmin
        self.factor_max = fmax
        self.p = p

        assert self.factor_min >= 0, "must be more than 0"
        assert self.factor_max >= self.factor_min, "must be more than factor min"

    def __call__(self, img, *targets):
        if decision(self.p):
            # get delta randomly between delta_min and delta_max
            factor = random.uniform(self.factor_min, self.factor_max)
            img[:, :, 1] *= factor
            img = np.clip(img, a_min=0, a_max=255)

        return img, (*targets,)

class RandomLightingNoise(object):
    def __init__(self, perms=None, p=0.5):
        self.p = p
        if perms:
            self.permutations = perms
        else:
            self.permutations = tuple(permutations([0, 1, 2]))

    def __call__(self, img, *targets):
        if decision(self.p):
            # get transposed indices randomly
            index = random.randint(0, len(self.permutations))
            t = SwapChannels(self.permutations[index])
            img, targets = t(img, *targets)

        return img, (*targets,)

class RandomResize(object):
    def __init__(self, w_min=None, w_max=None, h_min=None, h_max=None, p=0.5):
        self.w_min = w_min
        self.w_max = w_max
        self.h_min = h_min
        self.h_max = h_max
        if self.h_min is None and self.h_max is None and self.w_min is None and self.w_max is None:
            logging.warning("No meaning when all arguments are None")
        self.p = p

    def __call__(self, img, *targets):
        if decision(self.p):
            h, w, _ = img.shape

            w_min = self.w_min if self.w_min else w
            w_max = self.w_max if self.w_max else w
            h_min = self.h_min if self.h_min else h
            h_max = self.h_max if self.h_max else h

            w_new = random.randint(w_min, w_max+1)
            h_new = random.randint(h_min, h_max+1)

            img = cv2.resize(img, (w_new, h_new))

        return img, (*targets,)

class RandomLongerResize(object):
    def __init__(self, smin, smax, p=0.5):
        self.smin = smin
        self.smax = smax
        self.p = p

    def __call__(self, img, *targets):
        if decision(self.p):
            h, w, _ = img.shape

            new_size = random.randint(self.smin, self.smax+1)
            if h > w:
                img = cv2.resize(img, (w, new_size))
            else:
                img = cv2.resize(img, (new_size, h))

        return img, (*targets,)

class SwapChannels(object):
    def __init__(self, trans_indices):
        self.trans_indices = trans_indices

    def __call__(self, img, *targets):
        return img[:, :, self.trans_indices], (*targets,)


class ConvertImgOrder(object):
    def __init__(self, src='rgb', dst='hsv'):
        self.src_order = src.upper()
        self.dst_order = dst.upper()

    def __call__(self, img, *targets):
        try:
            img = cv2.cvtColor(img, eval('cv2.COLOR_{}2{}'.format(self.src_order, self.dst_order)))
        except:
            raise ValueError('Invalid src:{} or dst:{}'.format(self.src_order, self.dst_order))

        return img, (*targets,)


class PhotometricDistortions(Compose):
    def __init__(self, p=0.5):
        self.p = p

        self.brigtness = RandomBrightness()
        self.cotrast = RandomContrast()
        self.lightingnoise = RandomLightingNoise()

        pmdists = [
            ConvertImgOrder(src='rgb', dst='hsv'),
            RandomSaturation(),
            RandomHue(),
            ConvertImgOrder(src='hsv', dst='rgb')
        ]
        super().__init__(pmdists)

    def __call__(self, img, *targets):
        img, targets = self.brigtness(img, *targets)

        if decision(self.p): # random contrast first
            img, targets = self.cotrast(img, *targets)
            img, targets = super().__call__(img, *targets)

        else: # random contrast last
            img, targets = super().__call__(img, *targets)
            img, targets = self.cotrast(img, *targets)

        img, targets = self.lightingnoise(img, *targets)

        return img, targets
import numpy as np
import torch
import cv2
import logging
from dl.data._utils import _check_ins

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

"""
bellow classes are consisted of
    :param img: Tensor
    :param bboxes: ndarray of bboxes
    :param labels: ndarray of bboxes' indices
    :param flags: list of flag's dict
    :return: Tensor of img, ndarray of bboxes, ndarray of labels, dict of flags
"""

class ToTensor(object):
    """
    Note that convert ndarray to tensor and [0-255] to [0-1]
    """
    def __call__(self, img):
        # convert ndarray into Tensor
        # transpose img's tensor (h, w, c) to pytorch's format (c, h, w). (num, c, h, w)
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float() / 255.

class Resize(object):
    def __init__(self, size):
        """
        :param size: 2d-array-like, (height, width)
        """
        self._size = size

    def __call__(self, img):
        return cv2.resize(img, self._size)


class Grayscale(object):
    def __init__(self, last_dims=None):
        """
        :param last_dims: int or None, if last_dims is None, return image with (h, w), 
                          otherwise, with (h, w, last_dims)
        """
        self._last_dims = _check_ins('last_dims', last_dims, int, allow_none=True)
        
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self._last_dims:
            img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(np.broadcast_to(img, self._last_dims))

        return img


class Normalize(object):
    #def __init__(self, rgb_means=(103.939, 116.779, 123.68), rgb_stds=(1.0, 1.0, 1.0)):
    def __init__(self, rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225)):
        self.means = np.array(rgb_means, dtype=np.float32).reshape((-1, 1, 1))
        if np.any(np.abs(self.means) > 1):
            logging.warning("In general, mean value should be less than 1 because img's range is [0-1]")

        self.stds = np.array(rgb_stds, dtype=np.float32).reshape((-1, 1, 1))

    def __call__(self, img, *args):
        if isinstance(img, torch.Tensor):
            return (img.float() - torch.from_numpy(self.means)) / torch.from_numpy(self.stds)
        else:
            return (img.astype(np.float32) - self.means) / self.stds

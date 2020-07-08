import numpy as np
import torch

from ..base.target_transforms import _one_hot_encode, Compose

class OneHot(object):

    def __init__(self, class_nums, add_nolabel=True):
        self._class_nums = class_nums
        self._add_nolabel = add_nolabel
        if add_nolabel:
            self._class_nums += 1

    def __call__(self, labels, *args):
        if labels.ndim != 1:
            raise ValueError('labels might have been already one-hotted or be invalid shape')

        labels = _one_hot_encode(labels.astype(np.int), self._class_nums)
        labels = np.array(labels, dtype=np.float32)

        return (labels, *args)

class ToTensor(object):
    def __call__(self, labels, *args):
        """
        :param labels: ndarray
        :return:
        """
        return (torch.from_numpy(labels), *args)

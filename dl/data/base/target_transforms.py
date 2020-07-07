import numpy as np

from .._utils import _one_hot_encode

class Compose(object):
    def __init__(self, target_transforms):
        self.target_transforms = target_transforms

    def __call__(self, *targets):
        for t in self.target_transforms:
            targets = t(*targets)
        return targets

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.target_transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class _IgnoreBase(object):
    def __call__(self, *args):
        pass


class OneHot(object):

    def __init__(self, class_nums, add_background=True):
        self._class_nums = class_nums
        self._add_background = add_background
        if add_background:
            self._class_nums += 1

    def __call__(self, bboxes, labels, flags, *args):
        if labels.ndim != 1:
            raise ValueError('labels might have been already one-hotted or be invalid shape')

        labels = _one_hot_encode(labels.astype(np.int), self._class_nums)
        labels = np.array(labels, dtype=np.float32)

        return (bboxes, labels, flags, *args)
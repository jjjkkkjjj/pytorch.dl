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


import numpy as np

from ..objrecog.target_transforms import (
    Compose,
    OneHot,
    ToTensor
)
from .._utils import _check_ins

class Text2Number(object):
    def __init__(self, class_labels, blankIndex=None, ignore_nolabel=True, toLower=True):
        self._class_labels = class_labels

        blankIndex = _check_ins('blankIndex', blankIndex, int, allow_none=True)
        if blankIndex:
            self._class_labels.insert(blankIndex, '-')

        self._ignore_nolabel = ignore_nolabel
        self._toLower = toLower

    def __call__(self, labels, *args):
        ret_labels = []
        for c in labels:
            try:
                c = c.lower() if self._toLower else c
                ret_labels += [self._class_labels.index(c)]
            except ValueError:
                if self._ignore_nolabel:
                    continue
                else:
                    raise ValueError('{} didn\'t contain ({})'.format(labels, ''.join(self._class_labels)))

        return (np.array(ret_labels), *args)

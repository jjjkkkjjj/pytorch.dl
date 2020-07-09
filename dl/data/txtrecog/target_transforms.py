import numpy as np

from ..objrecog.target_transforms import (
    Compose,
    OneHot,
    ToTensor
)

class Text2Number(object):
    def __init__(self, class_labels, ignore_nolabel=False):
        self._class_labels = class_labels
        self._ignore_nolabel = ignore_nolabel

    def __call__(self, labels, *args):
        ret_labels = []
        for c in labels:
            try:
                ret_labels += [self._class_labels.index(c.lower())]
            except ValueError:
                if self._ignore_nolabel:
                    continue
                else:
                    raise ValueError('{} didn\'t contain ({})'.format(labels, ''.join(self._class_labels)))

        return (np.array(ret_labels), *args)

import torch
import numpy as np
import abc

from ..._utils import _check_ins, _contain_ignore
from ...base.target_transforms import _IgnoreBase
from ...base.datasets import ImageDatasetBase, Dataset
from ...objrecog.datasets import ObjectRecognitionDatasetBase
"""
ref > https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample

"""

class ObjectDetectionDatasetMixin:

    @property
    @abc.abstractmethod
    def class_nums(self):
        pass

    @property
    @abc.abstractmethod
    def class_labels(self):
        pass

class ObjectDetectionDatasetBase(ObjectDetectionDatasetMixin, ImageDatasetBase):
    """
    class_nums, class_labels, _get_image(index), _get_target(index), __len__
    must be implemented at least
    """
    def __init__(self, ignore=None, transform=None, target_transform=None, augmentation=None):
        """
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        super().__init__(transform, target_transform, augmentation)
        self.ignore = _check_ins('ignore', ignore, _IgnoreBase, allow_none=True)


    def __getitem__(self, index):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets : Tensor or ndarray of bboxes and labels [box, label]
            = [xmin, ymin, xmamx, ymax, label index(or relu_one-hotted label)]
            or
            = [cx, cy, w, h, label index(or relu_one-hotted label)]
        """
        img, targets = self.get_imgtarget(index)

        if len(targets) >= 3:
            labels, bboxes, flags = targets[:3]
        else:
            raise ValueError(
                'ValueError: not enough values to unpack (expected more than 3, got {})'.format(len(targets)))

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(labels, torch.Tensor):
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            targets = torch.cat((bboxes, labels), dim=1)
        else:
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            targets = np.concatenate((bboxes, labels), axis=1)

        return img, targets

    def apply_transform(self, img, *targets):
        """
        IMPORTATANT: apply transform function in order with ignore, augmentation, transform and target_transform
        :param img:
        :param labels:
        :param bboxes:
        :param flags:
        :return:
            Transformed img, bboxes, linds, flags
        """
        if len(targets) >= 3:
            labels, bboxes, flags = targets[:3]
            args = targets[3:]
        else:
            raise ValueError(
                'ValueError: not enough values to unpack (expected more than 3, got {})'.format(len(targets)))

        # To Percent mode
        height, width, channel = img.shape
        # bbox = [xmin, ymin, xmax, ymax]
        # [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        bboxes[:, 0::2] /= float(width)
        bboxes[:, 1::2] /= float(height)

        if self.ignore:
            targets = self.ignore(labels, bboxes, flags, *args)
        else:
            targets = (labels, bboxes, flags, *args)

        return super().apply_transform(img, *targets)



class Compose(ImageDatasetBase):
    def __init__(self, datasets, **kwargs):
        """
        :param datasets: tuple of Dataset
        :param kwargs:
            :param ignore:
            :param transform:
            :param target_transform:
            :param augmentation:
        """
        transform = kwargs.get('transform', None)
        target_transform = kwargs.get('target_transform', None)
        augmentation = kwargs.get('augmentation', None)
        super().__init__(transform, target_transform, augmentation)
        
        datasets = _check_ins('datasets', datasets, (tuple, list))

        _datasets, _lens = [], []
        _class_labels = None
        for dataset in datasets:
            try:
                dataset = dataset(**kwargs)
            except Exception as e:
                raise ValueError('Invalid arguments were passed. {} could not be initialized because\n{}'.format(dataset.__name__, e))
            dataset = _check_ins('element of datasets', dataset, ImageDatasetBase)
            if _class_labels is None:
                _class_labels = dataset.class_labels
            else:
                #if set(_class_labels) != set(dataset.class_labels):
                if _class_labels != dataset.class_labels:
                    raise ValueError('all of datasets must be same class labels')

            # initialization
            _datasets += [dataset]

            _lens += [len(_datasets[-1])]

        self.datasets = _datasets
        self.lens = _lens
        self._class_labels = _class_labels

    @property
    def class_labels(self):
        return self._class_labels
    @property
    def class_nums(self):
        return len(self._class_labels)

    def __getitem__(self, index):
        for i in range(len(self.lens)):
            if index < sum(self.lens[:i+1]):
                return self.datasets[i][index - sum(self.lens[:i])]

        raise ValueError('Index out of range')

    def __len__(self):
        return sum(self.lens)



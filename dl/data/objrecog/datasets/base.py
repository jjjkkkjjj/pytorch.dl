import abc
import numpy as np

from ...base.datasets import ImageDatasetBase

class ObjectRecognitionDatasetMixin:

    @property
    @abc.abstractmethod
    def class_nums(self):
        pass
    @property
    @abc.abstractmethod
    def class_labels(self):
        pass

class ObjectRecognitionDatasetBase(ObjectRecognitionDatasetMixin, ImageDatasetBase):
    """
    class_nums, class_labels, _get_image(index), _get_target(index), __len__
    must be implemented at least
    """
    pass

import abc
import numpy as np

from ...base.datasets import _DatasetBase, reapply_in_exception, maximum_reapply
from ...base.exceptions import _TargetTransformBaseException, MaximumReapplyError

class ObjectRecognitionDatasetBase(_DatasetBase):
    def __init__(self, transform=None, target_transform=None, augmentation=None):
        """
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        self.transform = transform
        self.target_transform = target_transform #_contain_ignore(target_transform)
        self.augmentation = augmentation

    @property
    @abc.abstractmethod
    def class_nums(self):
        pass
    @property
    @abc.abstractmethod
    def class_labels(self):
        pass

    @abc.abstractmethod
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(Tensor)
        """
        raise NotImplementedError('\'_get_image\' must be overridden')

    @abc.abstractmethod
    def _get_target(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated])
        """
        raise NotImplementedError('\'_get_target\' must be overridden')

    def get_imgtarget(self, index, count=0):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets : Tensor or array-like labels
        """
        try:
            img = self._get_image(index)
            targets = self._get_target(index)

            img, targets = self.apply_transform(img, *targets)

            return img, targets
        except _TargetTransformBaseException as e:
            if count == maximum_reapply:
                raise MaximumReapplyError('Maximum Reapplying reached: {}. last error was {}'.format(count, str(e)))
            elif reapply_in_exception:
                return self.get_imgtarget(np.random.randint(len(self)), count+1)
            else:
                raise e

    def __getitem__(self, index):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets : Tensor or array-like labels
        """
        return self.get_imgtarget(index)

    def apply_transform(self, img, *targets):
        """
        IMPORTATANT: apply transform function in order with ignore, augmentation, transform and target_transform
        :param img:
        :param targets:
        :return:
            Transformed img, targets, args
        """

        if self.augmentation:
            img, targets = self.augmentation(img, *targets)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            targets = self.target_transform(*targets)

        return img, targets

    @abc.abstractmethod
    def __len__(self):
        pass
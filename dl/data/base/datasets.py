from torch.utils.data import Dataset
import abc, os, cv2
import numpy as np
from pycocotools.coco import COCO
from xml.etree import ElementTree as ET

from .exceptions import _TargetTransformBaseException, MaximumReapplyError
from .._utils import DATA_ROOT, _get_xml_et_value, _check_ins

reapply_in_exception = True
maximum_reapply = 10

class ImageDatasetBase(Dataset):
    def __init__(self, transform=None, target_transform=None, augmentation=None):
        """
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        self.transform = transform
        self.target_transform = target_transform  # _contain_ignore(target_transform)
        self.augmentation = augmentation


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
                return self.get_imgtarget(np.random.randint(len(self)), count + 1)
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


class COCODatasetMixin:
    _coco_dir: str
    _focus: str
    _coco: COCO
    _imageids: list

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', self._focus, filename)


    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(ndarray)
        """

        """
        self._coco.loadImgs(self._imageids[index]): list of dict, contains;
            license: int
            file_name: str
            coco_url: str
            height: int
            width: int
            date_captured: str
            flickr_url: str
            id: int
        """
        filename = self._coco.loadImgs(self._imageids[index])[0]['file_name']
        img = cv2.imread(self._jpgpath(filename))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

class VOCDatasetMixin:
    _voc_dir: str
    _annopaths: list

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._voc_dir, 'JPEGImages', filename)

    """
    Detail of contents in voc > https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    VOC bounding box (xmin, ymin, xmax, ymax)
    """
    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(ndarray)
        """
        root = ET.parse(self._annopaths[index]).getroot()
        img = cv2.imread(self._jpgpath(_get_xml_et_value(root, 'filename')))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)
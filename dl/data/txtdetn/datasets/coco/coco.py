import cv2, os, logging
import numpy as np
from .coco_text import COCO_Text
from ..base import TextDetectionDatasetBase, Compose, TextDetectionDatasetMixin
from ....base.datasets import COCODatasetMixin

from ...._utils import DATA_ROOT, _check_ins

COCOText_class_labels = ['text']
COCOText_class_nums = len(COCOText_class_labels)

COCO2014Text_ROOT = os.path.join(DATA_ROOT, 'coco', 'coco2014')

class COCOTextDatasetMixin(TextDetectionDatasetMixin, COCODatasetMixin):
    _image_dir: str
    _coco: COCO_Text
    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', self._image_dir, filename)

    def _get_target(self, index):
        """
        :param index: int
        :return:
            list of bboxes' label index, list of bboxes, list of flags([difficult, truncated,...])
        """
        linds = []
        bboxes = []
        flags = []
        texts = []
        quads = []

        # anno_ids is list
        _imgid = self._coco.loadImgs(self._imageids[index])[0]['id']
        anno_ids = self._coco.getAnnIds(_imgid)

        # annos is list of dict
        annos = self._coco.loadAnns(anno_ids)
        for anno in annos:
            """
            anno's  keys are;
                polygon: list of float, whose length must be 8=
                language: str
                area: float
                id: int
                utf8_string, str, if it's illegible, this will not exist
                image_id: int
                bbox: list of float, whose length is 4=(xmin,ymin,w,h)
                legibility: int
                class: int
            """
            linds += [0]# 0 means text, 1 means background

            # bbox = [xmin, ymin, w, h]
            xmin, ymin, w, h = anno['bbox']
            # convert to corners
            xmax, ymax = xmin + w, ymin + h
            bboxes.append([xmin, ymin, xmax, ymax])

            #print(len(anno['polygon']))
            assert len(anno['polygon']) == 8, 'Invalid polygon length, must be 8, but got {}'.format(len(anno['polygon']))
            x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl = anno['polygon']
            quads.append([x_tl, y_tl, x_tr, y_tr,
                          x_br, y_br, x_bl, y_bl])

            flags += [{'illegible': anno['legibility'] == 'illegible'}]

            try:
                texts += [anno['utf8_string']]
            except KeyError:
                texts += ['']
            """
            if anno['legibility'] == 'illegible':
                texts += ['']
            else:
                texts += [anno['utf8_string']]
            """

        return np.array(linds, dtype=np.float32), np.array(bboxes, dtype=np.float32), flags, np.array(quads, dtype=np.float32), texts


class COCOTextSingleDatasetBase(COCOTextDatasetMixin, TextDetectionDatasetBase):
    def __init__(self, coco_dir, focus, image_dir, datasetTypes, ignore=None, transform=None, target_transform=None, augmentation=None):
        """
        :param coco_dir: str, coco directory path above 'annotations' and 'images'
                e.g.) coco_dir = '~~~~/coco2007/trainval'
        :param focus: str or str, directory name under images
                e.g.) focus = 'train2014'
        :param image_dir: str
        :param datasetType: list of str, train, val or test
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        super().__init__(ignore=ignore, transform=transform, target_transform=target_transform,
                         augmentation=augmentation)

        self._coco_dir = coco_dir
        self._focus = focus
        self._image_dir = image_dir

        self._class_labels = COCOText_class_labels

        self._annopath = os.path.join(self._coco_dir, 'annotations', self._focus + '.json')
        if os.path.exists(self._annopath):
            self._coco = COCO_Text(self._annopath)
        else:
            raise FileNotFoundError('json: {} was not found'.format(self._focus + '.json'))

        # get all images containing at least one instance of legible text
        datasetTypes = _check_ins('datasetTypes', datasetTypes, (list, tuple))
        imgIds = []
        _dataset_types = ['train', 'val', 'test']
        for datasettype in datasetTypes:
            if not datasettype in _dataset_types:
                raise ValueError('Invalid argument: datasettype must be list of str, are {}, but got {}'.format(_dataset_types, datasettype))
            imgIds.extend(self._coco.getImgIds(imgIds=eval('self._coco.{}'.format(datasettype)),
                                               catIds=[('legibility', 'legible')]))
            #imgIds.extend(self._coco.getImgIds(imgIds=eval('self._coco.{}'.format(datasettype)),
            #                                   catIds=[('legibility', 'illegible')]))
            #imgIds.extend(self._coco.getImgIds(imgIds=eval('self._coco.{}'.format(datasettype)),
            #                                   catIds=[('class', 'machine printed')]))
            #imgIds.extend(self._coco.getImgIds(imgIds=eval('self._coco.{}'.format(datasettype)),
            #                                   catIds=[('class', 'handwritten')]))

        self._imageids = list(set(imgIds))

    def __len__(self):
        return len(self._imageids)

    @property
    def class_nums(self):
        return len(self._class_labels)
    @property
    def class_labels(self):
        return self._class_labels


class COCOTextMultiDatasetBase(Compose):
    def __init__(self, **kwargs):
        """
        :param datasets: tuple of Dataset
        :param kwargs:
            :param ignore:
            :param transform:
            :param target_transform:
            :param augmentation:
        """
        super().__init__(datasets=(), **kwargs)

        coco_dir = _check_ins('coco_dir', kwargs.pop('coco_dir'), (tuple, list, str))
        focus = _check_ins('focus', kwargs.pop('focus'), (tuple, list, str))
        datasetTypes = _check_ins('datasetTypes', kwargs.pop('datasetTypes'), (tuple, list))
        image_dir = _check_ins('image_dir', kwargs.pop('image_dir'), str)

        if isinstance(coco_dir, str) and isinstance(focus, str) and isinstance(datasetTypes, str):
            datasets = [COCOTextSingleDatasetBase(coco_dir, focus, image_dir, datasetTypes, **kwargs)]
            lens = [len(datasets[0])]

        elif isinstance(coco_dir, (list, tuple)) and isinstance(focus, (list, tuple)) and isinstance(datasetTypes, (list, tuple)):
            if not (len(coco_dir) == len(focus)):
                raise ValueError('coco_dir, focus and datasetTypes must be same length, but got {} and {}'.format(len(coco_dir), len(focus)))

            datasets = [COCOTextSingleDatasetBase(cdir, f, image_dir, datasetTypes, **kwargs) for cdir, f in zip(coco_dir, focus)]
            lens = [len(d) for d in datasets]
        else:
            raise ValueError('Invalid coco_dir and focus combination')

        self.datasets = datasets
        self.lens = lens
        self._class_labels = datasets[0].class_labels


class COCO2014Text_Dataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', 'train2014', ('train', 'val', 'test'), **kwargs)


class COCO2014Text_TrainDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', 'train2014', ('train',), **kwargs)


class COCO2014Text_ValDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', 'train2014', ('val',), **kwargs)


class COCO2014Text_TestDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', 'train2014', ('test',), **kwargs)

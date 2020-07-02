import cv2, os, logging
import numpy as np
from .coco_text import COCO_Text
from ..base import TextDetectionDatasetBase

from ...._utils import DATA_ROOT, _check_ins

COCOText_class_labels = ['text']
COCOText_class_nums = len(COCOText_class_labels)

class COCOTextSingleDatasetBase(TextDetectionDatasetBase):
    def __init__(self, coco_dir, focus, datasetTypes, ignore=None, transform=None, target_transform=None, augmentation=None):
        """
        :param coco_dir: str, coco directory path above 'annotations' and 'images'
                e.g.) coco_dir = '~~~~/coco2007/trainval'
        :param focus: str or str, directory name under images
                e.g.) focus = 'train2014'
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


    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', self._focus, filename)

    def __len__(self):
        return len(self._imageids)

    @property
    def class_nums(self):
        return len(self._class_labels)
    @property
    def class_labels(self):
        return self._class_labels

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

    def _get_target(self, index):
        """
        :param index: int
        :return:
            list of bboxes, list of bboxes' label index, list of flags([difficult, truncated,...])
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

        return np.array(bboxes, dtype=np.float32), np.array(linds, dtype=np.float32), flags, np.array(quads, dtype=np.float32), texts


class COCO2014Text_Dataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', ('train', 'val', 'test'), **kwargs)

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', 'train2014', filename)

class COCO2014Text_TrainDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', ('train',), **kwargs)

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', 'train2014', filename)

class COCO2014Text_ValDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', ('val',), **kwargs)

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', 'train2014', filename)

class COCO2014Text_TestDataset(COCOTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(DATA_ROOT + '/coco/coco2014/trainval', 'COCO_Text', ('test',), **kwargs)

    def _jpgpath(self, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._coco_dir, 'images', 'train2014', filename)
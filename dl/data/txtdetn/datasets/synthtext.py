import cv2, os, glob, csv, logging, time
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
from ...txtdetn.datasets.base import TextDetectionDatasetBase, Compose
from ...base.synthtext import *

from ..._utils import DATA_ROOT, _check_ins, _get_xml_et_value

class _FoundNonAlphaNumeric(Exception):
    pass

class SynthTextDetectionSingleDatasetBase(TextDetectionDatasetBase):
    def __init__(self, synthtext_dir, ignore=None, transform=None, target_transform=None, augmentation=None,
                 onlyAlphaNumeric=False):
        """
        :param synthtext_dir: str, synthtext directory path above 'Annotations' and 'SynthText'
        :param ignore: target_transforms.Ignore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        :param onlyAlphaNumeric: bool, whether to return img with words containing non-alphanumeric
        """
        super().__init__(ignore=ignore, transform=transform, target_transform=target_transform,
                         augmentation=augmentation)

        self._synthtext_dir = synthtext_dir
        self._class_labels = SynthText_class_labels

        if not os.path.exists(os.path.join(self._synthtext_dir, 'Annotations')):
            raise FileNotFoundError('{} was not found'.format(os.path.join(self._synthtext_dir, 'Annotations')))

        self._annopaths = [str(path.absolute()) for path in Path(os.path.join(self._synthtext_dir, 'Annotations')).rglob('*.xml')]
        self._onlyAlphaNumeric = onlyAlphaNumeric

    def _jpgpath(self, dirname, filename):
        """
        :param filename: path containing .jpg
        :return: path of jpg
        """
        return os.path.join(self._synthtext_dir, 'SynthText', dirname, filename)

    def __len__(self):
        return len(self._annopaths)

    @property
    def class_nums(self):
        return len(self._class_labels)
    @property
    def class_labels(self):
        return self._class_labels

    def __getitem__(self, index):
        start = time.time()
        ind = index
        while True:
            try:
                return super().__getitem__(ind)
            except _FoundNonAlphaNumeric:
                ind = np.random.randint(0, len(self))
                if time.time() - start > 10:
                    logging.warning('10s passed...\nMany non-alphanumeric words may exist.')
                    start = time.time()
                pass

    def _get_image(self, index):
        """
        :param index: int
        :return:
            rgb image(ndarray)
        """
        root = ET.parse(self._annopaths[index]).getroot()
        dirname = _get_xml_et_value(root, 'folder')
        filename = _get_xml_et_value(root, 'filename')
        img = cv2.imread(self._jpgpath(dirname, filename))
        # pytorch's image order is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

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

        root = ET.parse(self._annopaths[index]).getroot()
        for obj in root.iter('object'):
            linds.append(0) # 0 means text, 1 means background

            bndbox = obj.find('bndbox')

            # bbox = [xmin, ymin, xmax, ymax]
            bboxes.append([_get_xml_et_value(bndbox, 'xmin', float), _get_xml_et_value(bndbox, 'ymin', float),
                           _get_xml_et_value(bndbox, 'xmax', float), _get_xml_et_value(bndbox, 'ymax', float)])

            quads.append([_get_xml_et_value(bndbox, 'x1', float), _get_xml_et_value(bndbox, 'y1', float),
                          _get_xml_et_value(bndbox, 'x2', float), _get_xml_et_value(bndbox, 'y2', float),
                          _get_xml_et_value(bndbox, 'x3', float), _get_xml_et_value(bndbox, 'y3', float),
                          _get_xml_et_value(bndbox, 'x4', float), _get_xml_et_value(bndbox, 'y4', float)])

            text = _get_xml_et_value(obj, 'name', str)
            if self._onlyAlphaNumeric and not text.isalnum():
                raise _FoundNonAlphaNumeric()
            texts.append(text)

            flags.append({'difficult': _get_xml_et_value(obj, 'difficult', int) == 1})

        return np.array(linds, dtype=np.float32), np.array(bboxes, dtype=np.float32), flags, np.array(quads, dtype=np.float32), texts

class SynthTextDetectionMultiDatasetBase(Compose):
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

        synthtext_dir = _check_ins('synthtext_dir', kwargs.pop('synthtext_dir'), (tuple, list, str))

        if isinstance(synthtext_dir, str):
            datasets = [SynthTextDetectionSingleDatasetBase(synthtext_dir, **kwargs)]
            lens = [len(datasets[0])]

        elif isinstance(synthtext_dir, (list, tuple)):
            datasets = [SynthTextDetectionSingleDatasetBase(sdir, **kwargs) for sdir in synthtext_dir]
            lens = [len(d) for d in datasets]
        else:
            assert False

        self.datasets = datasets
        self.lens = lens
        self._class_labels = datasets[0].class_labels

class SynthTextDetectionDataset(SynthTextDetectionSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(synthtext_dir=DATA_ROOT + '/text/SynthText', **kwargs)
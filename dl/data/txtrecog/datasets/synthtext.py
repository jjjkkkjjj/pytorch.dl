import os, cv2, logging, time, csv
import numpy as np

from .base import TextRecognitionDatasetBase, Alphanumeric_labels
from ...base.synthtext import *
from ..._utils import _check_ins, DATA_ROOT

class SynthTextRecongnitionDatasetMixin:
    _gts: list
    _synthtext_dir: str
    def _get_image(self, index):
        line = self._gts[index]
        folder, filename, text = line[:3]
        xmin, ymin, xmax, ymax = map(float, line[3:7])
        #x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line[7:15])
        img = cv2.imread(os.path.join(self._synthtext_dir, 'SynthText', folder, filename))
        h, w, _ = img.shape
        # clip
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, w), min(ymax, h)

        # crop
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    def _get_target(self, index):
        line = self._gts[index]
        folder, filename, text = line[:3]
        return text,


class SynthTextRecognitionSingleDatasetBase(SynthTextRecongnitionDatasetMixin, TextRecognitionDatasetBase):
    def __init__(self, synthtext_dir, transform=None, target_transform=None, augmentation=None, class_labels=None):
        """
        :param synthtext_dir: str, synthtext directory path above 'Annotations' and 'SynthText'
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        :param class_labels: None or list or tuple, if it's None use ALPHABET
        """
        super().__init__(transform=transform, target_transform=target_transform,
                         augmentation=augmentation)

        self._synthtext_dir = synthtext_dir
        self._class_labels = _check_ins('class_labels', class_labels, (list, tuple), allow_none=True, default=Alphanumeric_labels)

        annopaths = os.path.join(self._synthtext_dir, 'Annotations', 'gt_alphanumeric.csv')
        if not os.path.exists(annopaths):
            raise FileNotFoundError('{} was not found'.format(annopaths))

        logging.basicConfig(level=logging.INFO)
        logging.info('Loading ground truth...')
        start = time.time()
        self._annopaths = annopaths
        with open(self._annopaths, 'r') as f:
            lines = csv.reader(f)
            self._gts = list(lines)[1:] # remove header # use too much memory about 8GB...
        logging.info('Loaded! {}s'.format(time.time() - start))

    def __len__(self):
        return len(self._gts)

class SynthTextRecognitionDataset(SynthTextRecognitionSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(synthtext_dir=DATA_ROOT + '/text/SynthText', **kwargs)
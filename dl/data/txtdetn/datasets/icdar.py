import cv2, os, glob
import numpy as np
from xml.etree import ElementTree as ET
from .base import TextDetectionDatasetBase, Compose

from ..._utils import DATA_ROOT, _check_ins, _get_xml_et_value

SynthText_class_labels = ['text']
SynthText_class_nums = len(SynthText_class_labels)

ICDARText_ROOT = os.path.join(DATA_ROOT, 'text', 'ICDAR2015')
class ICDARTextSingleDatasetBase(TextDetectionDatasetBase):
    def __init__(self, icdar_dir, image_ext='.jpg', ignore=None, transform=None, target_transform=None, augmentation=None):
        """
        :param icdar_dir: str, ICDAR directory path above 'Annotations' and 'Images'
        :param image_ext: str or None, if None, the extension will be inferred
        :param ignore: target_transforms.TextDetectionIgnore
        :param transform: instance of transforms
        :param target_transform: instance of target_transforms
        :param augmentation:  instance of augmentations
        """
        super().__init__(ignore=ignore, transform=transform, target_transform=target_transform,
                         augmentation=augmentation)

        self._icdar_dir = icdar_dir
        self._class_labels = SynthText_class_labels

        if not os.path.exists(os.path.join(self._icdar_dir, 'Annotations')):
            raise FileNotFoundError('{} was not found'.format(os.path.join(self._icdar_dir, 'Annotations')))

        if not os.path.exists(os.path.join(self._icdar_dir, 'Images')):
            raise FileNotFoundError('{} was not found'.format(os.path.join(self._icdar_dir, 'Images')))


        self._annopaths = glob.glob(os.path.join(self._icdar_dir, 'Annotations', '*.txt'))
        self._image_ext = image_ext



    def _imgpath(self, annopath):
        """
        :param annopath: path containing .txt
        :return: path of jpg
        """
        filename, _ = os.path.splitext(os.path.basename(annopath))
        # remove gt_
        filename = filename.replace('gt_', '')
        if self._image_ext:
            return os.path.join(self._icdar_dir, 'Images', filename + self._image_ext)
        else:
            path = glob.glob(os.path.join(self._icdar_dir, 'Images', filename + '.*'))
            if len(path) != 1:
                raise FileExistsError('plural \'{}\' were found\n{}'.format(filename, path))
            else:
                return path[0]

    def __len__(self):
        return len(self._annopaths)

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
        imgpath = self._imgpath(self._annopaths[index])
        # pytorch's image order is rgb
        _, ext = os.path.splitext(imgpath)
        if ext == '.png':
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif ext == '.gif':
            gif = cv2.VideoCapture(imgpath)
            _, img = gif.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(imgpath)
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

        with open(self._annopaths[index], 'r', encoding='utf-8-sig') as f:
            """
            x1,y1,x2,y2,x3,y3,x4,y4,text(### means illegible)
            
            377,117,463,117,465,130,378,130,Genaxis Theatre
            493,115,519,115,519,131,493,131,[06]
            374,155,409,155,409,170,374,170,###
            """
            lines = f.readlines()
            for line in lines:
                element = line.rstrip().split(',')

                linds.append(0) # 0 means text, 1 means background

                quad = np.array(element[:8]).astype(np.float32)

                # bbox = [xmin, ymin, xmax, ymax]
                bboxes.append([quad[::2].min(), quad[1::2].min(),
                               quad[::2].max(), quad[1::2].max()])

                quads.append(quad)

                texts.append(element[-1])

                flags.append({'difficult': element[-1] == '###'})

        return np.array(bboxes, dtype=np.float32), np.array(linds, dtype=np.float32), flags, np.array(quads, dtype=np.float32), texts

class ICDARTextMultiDatasetBase(Compose):
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

        icdar_dir = _check_ins('icdar_dir', kwargs.pop('icdar_dir'), (tuple, list, str))

        if isinstance(icdar_dir, str):
            datasets = [ICDARTextSingleDatasetBase(icdar_dir, image_ext=None, **kwargs)]
            lens = [len(datasets[0])]

        elif isinstance(icdar_dir, (list, tuple)):
            datasets = [ICDARTextSingleDatasetBase(idir, image_ext=None, **kwargs) for idir in icdar_dir]
            lens = [len(d) for d in datasets]
        else:
            assert False

        self.datasets = datasets
        self.lens = lens
        self._class_labels = datasets[0].class_labels

class ICDAR2015TextDataset(ICDARTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(icdar_dir=DATA_ROOT + '/text/ICDAR2015', image_ext='.jpg', **kwargs)

class ICDARBornDigitalTextDataset(ICDARTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(icdar_dir=DATA_ROOT + '/text/Born-Digital-Images', image_ext=None, **kwargs)

class ICDARFocusedSceneTextDataset(ICDARTextSingleDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(icdar_dir=DATA_ROOT + '/text/Focused-Scene-Text', image_ext='.jpg', **kwargs)
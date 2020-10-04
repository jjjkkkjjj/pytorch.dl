from ...objdetn.datasets.base import ObjectDetectionDatasetBase
from ...objdetn.datasets.base import Compose

import torch
import numpy as np


class TextDetectionDatasetBase(ObjectDetectionDatasetBase):
    def __getitem__(self, index):
        """
        :param index: int
        :return:
            img : rgb image(Tensor or ndarray)
            targets and texts:
                targets : Tensor or ndarray of bboxes, quads and labels [box, quads, label]
                = [xmin, ymin, xmamx, ymax, x1, y1, x2, y2,..., label index(or one-hotted label)]
                or
                = [cx, cy, w, h, x1, y1, x2, y2,..., label index(or relu_one-hotted label)]
                texts: list of str, if it's illegal, str = ''
        """
        img, targets = self.get_imgtarget(index)

        labels, bboxes, flags, quads, texts = targets[:]

        # concatenate bboxes and linds
        if isinstance(bboxes, torch.Tensor) and isinstance(labels, torch.Tensor):
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            targets = torch.cat((bboxes, quads, labels), dim=1)
        else:
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            targets = np.concatenate((bboxes, quads, labels), axis=1)

        return img, targets, texts
        #return img, targets

    def apply_transform(self, img, *targets):
        labels, bboxes, flags, quads, texts = targets[:]

        height, width, channel = img.shape

        quads[:, 0::2] /= float(width)
        quads[:, 1::2] /= float(height)

        return super().apply_transform(img, labels, bboxes, flags, quads, texts)


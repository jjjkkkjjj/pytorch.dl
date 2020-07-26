import logging

from ..base import TextSpottingModelBase
from .detn import SharedConv, Detector
from .roi import RoIRotate


class FOTS(TextSpottingModelBase):
    def __init__(self, input_shape):
        super().__init__(input_shape)

        self.sharedConv = SharedConv(out_channels=64)
        self.detector = Detector(in_channels=64, dist_scale=512)

        self.roi_rotate = RoIRotate()

    def forward(self, x, labels=None, texts=None):
        """
        :param x: img Tensor, shape = (b, c, h, w)
        :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
        :param texts: list(b) of list(text number) of Tensor, shape = (1, characters number)
        :return:
        """
        if self.training and labels is None and texts is None:
            raise ValueError("pass \'labels\' and \'texts\' for training mode")

        elif not self.training and (labels is not None or texts is None):
            logging.warning("forward as eval mode, but passed \'labels\' and \'texts\'")

        # detection branch
        fmaps = self.sharedConv(x)
        conf, distances, angle = self.detector(fmaps)

        if self.training:
            # RoI Rotate Branch
            rotated_features = self.roi_rotate(fmaps, distances, angle, labels)

        # recognition branch
        
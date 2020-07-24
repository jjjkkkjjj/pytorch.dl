from ..base import TextSpottingModelBase

from .detn import SharedConv, Detector
from .roi import RoIRotate


class FOTS(TextSpottingModelBase):
    def __init__(self, input_shape):
        super().__init__(input_shape)

        self.sharedConv = SharedConv(out_channels=64)
        self.detector = Detector(in_channels=64, dist_scale=512)

        self.roi_rotate = RoIRotate()

    def forward(self, x, targets=None):
        # detection branch
        fmaps = self.sharedConv(x)
        conf, distances, angle = self.detector(fmaps)

        if self.training:
            # RoI Rotate Branch
            self.roi_rotate

        # recognition branch
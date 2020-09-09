from .base import FOTSBase
from dl.models.fots.modules.detn import Detector
from dl.models.fots.modules.featextr import SharedConvRes34
from dl.models.fots.modules.recog import CRNN


class FOTSRes34(FOTSBase):
    def __init__(self, chars, input_shape, val_config=None):
        super().__init__(chars, input_shape, shrink_scale=0.3, val_config=val_config)

    def build_feature_extractor(self):
        return SharedConvRes34(out_channels=32)

    def build_detector(self):
        return Detector(in_channels=32, dist_scale=160)

    def build_recognizer(self):
        return CRNN(self.chars, (8, None, 32), 0)
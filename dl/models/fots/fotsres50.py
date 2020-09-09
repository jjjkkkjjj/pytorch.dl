from .base import FOTSBase
from .modules.detn import Detector
from .modules.featextr import SharedConvRes50
from .modules.recog import CRNN


class FOTSRes50(FOTSBase):
    def __init__(self, chars, input_shape, val_config=None):
        super().__init__(chars, input_shape, shrink_scale=0.3, val_config=val_config)

    def build_feature_extractor(self):
        return SharedConvRes50(out_channels=32)

    def build_detector(self):
        return Detector(in_channels=32, dist_scale=160)

    def build_recognizer(self):
        return CRNN(self.chars, (8, None, 32), 0)


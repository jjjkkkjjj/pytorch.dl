from .base import FOTSBase
from dl.models.fots.modules.detn import Detector
from dl.models.fots.modules.featextr import SharedConvRes50
from dl.models.fots.modules.recog import CRNN


class FOTSRes50(FOTSBase):
    def __init__(self, chars, input_shape):
        super().__init__(chars, input_shape)

    def build_feature_extractor(self):
        return SharedConvRes50(out_channels=32)

    def build_detector(self):
        return Detector(in_channels=32, dist_scale=512)

    def build_recognizer(self):
        return CRNN(self.chars, (8, None, 32), 0)


from .base import FOTSBase, FOTSTrainConfig, FOTSValConfig
from .modules.detn import Detector
from .modules.featextr import SharedConvRes50
from .modules.recog import CRNN


class FOTSRes50(FOTSBase):
    def __init__(self, chars, input_shape, feature_height=32):
        train_config = FOTSTrainConfig(chars=chars, input_shape=input_shape, shrink_scale=0.3, feature_height=feature_height,
                                       rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))

        val_config = FOTSValConfig(conf_threshold=0.5, iou_threshold=0.1, topk=200)

        super().__init__(train_config, val_config)

    def build_feature_extractor(self):
        return SharedConvRes50(out_channels=32)

    def build_detector(self):
        return Detector(in_channels=32, dist_scale=self.dist_scale)

    def build_recognizer(self):
        return CRNN(self.chars, (self.feature_height, None, 32), 0)


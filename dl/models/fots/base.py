import logging, abc
from torch import nn

from ..base.model import TextSpottingModelBase
from .roi import RoIRotate
from .recog import CRNNBase
from .utils import matching_strategy
from ..._utils import _check_retval

class FeatureExtractorBase(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

class DetectorBase(nn.Module):
    pass

class FOTSBase(TextSpottingModelBase):
    def __init__(self, chars, input_shape, shrink_scale=0.3):
        super().__init__(chars, input_shape)

        self.feature_extractor = _check_retval('build_feature_extractor', self.build_feature_extractor(), FeatureExtractorBase)
        self.detector = _check_retval('build_detector', self.build_detector(), DetectorBase)

        self.roi_rotate = RoIRotate()
        self.recognizer = _check_retval('build_recognizer', self.build_recognizer(), CRNNBase)

        self._shrink_scale = shrink_scale

    @abc.abstractmethod
    def build_feature_extractor(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def build_detector(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def build_recognizer(self):
        raise NotImplementedError()


    def forward(self, x, labels=None, texts=None):
        """
        :param x: img Tensor, shape = (b, c, h, w)
        :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
        :param texts: list(b) of list(text number) of Tensor, shape = (characters number,)
        :returns:
            detn:
                pos_indicator: bool Tensor, shape = (b, h/4, w/4)
                pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
                pred_rboxes: predicted Tensor, shape = (b, h/4, w/4, 5=(t, r, b, l, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, r, b, l)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
                true_rboxes: true Tensor, shape = (text b, h/4, w/4, 5=(t, r, b, l, angle))
            recog:
                pred_texts: list(b) of predicted text number Tensor, shape = (times, text nums, class_nums)
                true_texts: list(b) of true text number Tensor, shape = (true text nums, char nums)
                pred_txtlens: list(b) of length Tensor, shape = (text nums)
                true_txtlens: list(b) of true length Tensor, shape = (true text nums)
        """
        if self.training and labels is None and texts is None:
            raise ValueError("pass \'labels\' and \'texts\' for training mode")

        elif not self.training and (labels is not None or texts is None):
            logging.warning("forward as eval mode, but passed \'labels\' and \'texts\'")

        # detection branch
        fmaps = self.feature_extractor(x)
        #  predicts: predicted Tensor, shape = (b, 6=(conf, t, l, b, r, angle), h/4, w/4)
        pred_confs, pred_rboxes = self.detector(fmaps)

        if self.training:
            # create pos_indicator and rboxes from label
            _, _, h, w = fmaps.shape
            device = fmaps.device
            pos_indicator, true_rboxes = matching_strategy(labels, w, h, device, scale=self._shrink_scale)

            # RoI Rotate Branch
            # list(b) of Tensor, shape = (text nums, c, height=8, non-fixed width)
            true_quads = [l[:, 4:12] for l in labels]
            rotated_features = self.roi_rotate(fmaps, true_quads)


            # recognition branch
            pred_texts, true_texts, pred_txtlens, true_txtlens = [], [], [], []
            for b in range(len(rotated_features)):
                preds, ts, pred_lengths, t_lengths = self.recognizer(rotated_features[b], texts[b])

                pred_texts += [preds]
                true_texts += [ts]
                pred_txtlens += [pred_lengths]
                true_txtlens += [t_lengths]

            return (pos_indicator, pred_confs, pred_rboxes, true_rboxes), (pred_texts, true_texts, pred_txtlens, true_txtlens)

        else:
            pass




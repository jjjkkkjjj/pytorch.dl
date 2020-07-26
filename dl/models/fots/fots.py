import logging

from ..base import TextSpottingModelBase
from .detn import SharedConv, Detector
from .roi import RoIRotate
from .recog import CRNN
from .utils import matching_strategy


class FOTS(TextSpottingModelBase):
    def __init__(self, chars, input_shape):
        super().__init__(chars, input_shape)

        self.feature_extractor = SharedConv(out_channels=64)
        self.detector = Detector(in_channels=64, dist_scale=512)

        self.roi_rotate = RoIRotate()
        self.recognizer = CRNN(chars, (8, None, 64), 0)

    def forward(self, x, labels=None, texts=None):
        """
        :param x: img Tensor, shape = (b, c, h, w)
        :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
        :param texts: list(b) of list(text number) of Tensor, shape = (characters number,)
        :returns:
            detn:
                pos_indicator: bool Tensor, shape = (b, h/4, w/4)
                pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
                pred_locs: predicted Tensor, shape = (b, h/4, w/4, 5=(conf, t, l, b, r, angle))
                    distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, l, b, r)) for each pixel to target rectangle boundaries
                    angle: angle Tensor, shape = (b, h/4, w/4, 1)
                true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
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
        pred_confs, pred_locs = self.detector(fmaps)

        if self.training:
            pos_indicator, true_locs = matching_strategy(fmaps, labels)

            # RoI Rotate Branch
            # list(b) of Tensor, shape = (text nums, c, height=8, non-fixed width)
            rotated_features = self.roi_rotate(fmaps, pred_locs, true_locs)

            # recognition branch
            pred_texts, true_texts, pred_txtlens, true_txtlens = [], [], [], []
            for b in range(len(rotated_features)):
                preds, ts, pred_lengths, t_lengths = self.recognizer(rotated_features[b], texts[b])

                pred_texts += [preds]
                true_texts += [ts]
                pred_txtlens += [pred_lengths]
                true_txtlens += [t_lengths]

            return (pos_indicator, pred_confs, pred_locs, true_locs), (pred_texts, true_texts, pred_txtlens, true_txtlens)

        else:
            pass
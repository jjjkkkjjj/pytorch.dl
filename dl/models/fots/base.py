import logging, abc, torch
from torch import nn

from ..base.model import TextSpottingModelBase
from ...data.utils.inference import locally_aware_nms
from ...data.utils.quads import rboxes2quads_numpy, quads_iou
from ...data.utils.converter import toVisualizeQuadsTextRGBimg
from .modules.roi import RoIRotate
from .modules.recog import CRNNBase
from .modules.utils import matching_strategy
from ..._utils import _check_retval, _check_ins, _check_image, _get_normed_and_origin_img

class FeatureExtractorBase(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels

class DetectorBase(nn.Module):
    pass

class FOTSTrainConfig(object):
    def __init__(self, **kwargs):
        self.chars = _check_ins('chars', kwargs.get('chars'), (tuple, list))

        input_shape = kwargs.get('input_shape')
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape

        self.shrink_scale = _check_ins('shrink_scale', kwargs.get('shrink_scale', 0.3), (float, int))

        self.rgb_means = _check_ins('rgb_means', kwargs.get('rgb_means', (0.485, 0.456, 0.406)), (tuple, list, float, int))
        self.rgb_stds = _check_ins('rgb_stds', kwargs.get('rgb_stds', (0.229, 0.224, 0.225)), (tuple, list, float, int))


class FOTSValConfig(object):
    def __init__(self, **kwargs):
        self.conf_threshold = _check_ins('conf_threshold', kwargs.get('conf_threshold', 0.5), float)
        self.iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.1), float)
        self.topk = _check_ins('topk', kwargs.get('topk', 200), int)

class FOTSBase(TextSpottingModelBase):
    _train_config: FOTSTrainConfig
    _val_config: FOTSValConfig
    
    def __init__(self, train_config, val_config):
        self._train_config = _check_ins('train_config', train_config, FOTSTrainConfig)
        self._val_config = _check_ins('val_config', val_config, FOTSValConfig)

        super().__init__(train_config.chars, train_config.input_shape)

        self.feature_extractor = _check_retval('build_feature_extractor', self.build_feature_extractor(), FeatureExtractorBase)
        self.detector = _check_retval('build_detector', self.build_detector(), DetectorBase)

        self.roi_rotate = RoIRotate()
        self.recognizer = _check_retval('build_recognizer', self.build_recognizer(), CRNNBase)


    # train property
    @property
    def shrink_scale(self):
        return self._train_config.shrink_scale
    @property
    def rbg_means(self):
        return self._train_config.rgb_means
    @property
    def std_means(self):
        return self._train_config.rgb_stds
    @property
    def dist_scale(self):
        if self.input_width is None and self.input_height is None:
            logging.warning("Input width and height is set to None, so use 160 which default value as dist_scale.\n"
                            "160 is from 640/4")
            return 160
        else:
            if self.input_width and self.input_height:
                return max(self.input_width, self.input_height)
            elif self.input_width:
                return self.input_width
            else:
                return self.input_height

    # val property
    @property
    def conf_threshold(self):
        return self._val_config.conf_threshold
    @property
    def iou_threshold(self):
        return self._val_config.iou_threshold
    @property
    def topk(self):
        return self._val_config.topk
    
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
            if self.training is True:
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
            else:
                ret_quads: list(b) of Tensor, shape = (text nums, 8=(x1,y1,... clockwise from top-left))
                ret_raws: list(b) of list(text nums) of str
                ret_texts: list(b) of list(text nums) of str
        """
        if self.training and labels is None and texts is None:
            raise ValueError("pass \'labels\' and \'texts\' for training mode")

        elif not self.training and (labels is not None or texts is not None):
            logging.warning("forward as eval mode, but passed \'labels\' and \'texts\'")

        # detection branch
        fmaps = self.feature_extractor(x)
        # pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
        # pred_rboxes: predicted Tensor, shape = (b, h/4, w/4, 5=(conf, t, r, b, l, angle))
        #      distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, r, b, l)) for each pixel to target rectangle boundaries
        #      angle: angle Tensor, shape = (b, h/4, w/4, 1)
        pred_confs, pred_rboxes = self.detector(fmaps)

        if self.training:
            # create pos_indicator and rboxes from label
            _, _, h, w = fmaps.shape
            device = fmaps.device
            pos_indicator, true_rboxes = matching_strategy(labels, w, h, device, scale=self.shrink_scale)

            # RoI Rotate Branch
            # list(b) of Tensor, shape = (text nums, c, height=8, non-fixed width)
            true_quads = [l[:, 4:12] for l in labels]
            rotated_features = self.roi_rotate(fmaps, true_quads)


            # recognition branch
            pred_texts, true_texts, pred_txtlens, true_txtlens = [], [], [], []
            for b in range(len(rotated_features)):
                # ps: shape = (times, text_nums, class_labels)
                # ts: LongTensor, shape = (text_nums, max length of text)
                # pred_lengths: LongTensor, shape = (text_nums,)
                # t_lengths: LongTensor, shape = (text_nums,)
                preds, ts, pred_lengths, t_lengths = self.recognizer(rotated_features[b], texts[b])

                pred_texts += [preds]
                true_texts += [ts]
                pred_txtlens += [pred_lengths]
                true_txtlens += [t_lengths]

            return (pos_indicator, pred_confs, pred_rboxes, true_rboxes), (pred_texts, true_texts, pred_txtlens, true_txtlens)

        else:
            with torch.no_grad():
                batch_nums, _, h, w = fmaps.shape

                # convert rboxes into quads
                # shape = (b, h, w, 8=(x1, y1,... clockwise order from top-left))
                pred_quads = torch.from_numpy(rboxes2quads_numpy(pred_rboxes.cpu().numpy()))

                # reshape
                # pred_quads: shape = (b, h*w, 8)
                # pred_confs: shape = (b, h*w,)
                pred_quads = pred_quads.reshape((batch_nums, -1, 8))
                pred_confs = pred_confs.reshape((batch_nums, -1))

                # filter out by conf_threshold
                filtered_mask = pred_confs > self.conf_threshold

                # nms
                ret_quads = []
                for b in range(batch_nums):
                    # p_quads: shape = (h*w, 8)
                    # p_confs: shape = (h*w,)
                    p_quads = pred_quads[b][filtered_mask[b]]
                    p_confs = pred_confs[b][filtered_mask[b]]

                    indices = locally_aware_nms(p_confs, p_quads, self.topk, self.iou_threshold, quads_iou)

                    p_quads[:, ::2] /= w
                    p_quads[:, 1::2] /= h
                    ret_quads += [p_quads[indices]]

                # roi rotate
                rotated_features = self.roi_rotate(fmaps, ret_quads)

                # recognition
                ret_raws, ret_texts = [], []
                for b in range(batch_nums):
                    # preds: shape = (times, text_nums, class_labels)
                    # raw_txt: list(text_nums) of str, raw strings
                    # out_txt: list(text_nums) of str, decoded strings
                    ps, raw_txt, out_txt = self.recognizer(rotated_features[b])
                    ret_raws += [raw_txt]
                    ret_texts += [out_txt]

                return ret_quads, ret_raws, ret_texts

    def infer(self, image, conf_threshold=None, toNorm=False, visualize=False):
        if self.training:
            raise NotImplementedError("call \'eval()\' first")

        # img: Tensor, shape = (b, c, h, w)
        img, orig_imgs = _check_image(image, self.device, size=(self.input_width, self.input_height))

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_imgs, orig_imgs = _get_normed_and_origin_img(img, orig_imgs, self.rgb_means, self.rgb_stds, toNorm,
                                                            self.device)

        if list(img.shape[1:]) != [self.input_channel, self.input_height, self.input_width]:
            raise ValueError('image shape was not same as input shape: {}, but got {}'.format(
                [self.input_channel, self.input_height, self.input_width], list(img.shape[1:])))

        inf_quads, inf_raws, inf_texts = self(normed_imgs)

        img_num = normed_imgs.shape[0]
        if visualize:
            visualized_imgs = [toVisualizeQuadsTextRGBimg(orig_imgs[i], poly_pts=inf_quads[i], tensor2cvimg=False, verbose=False)
                               for i in range(img_num)]
            return (inf_quads, inf_raws, inf_texts), visualized_imgs, orig_imgs
        else:
            return (inf_quads, inf_raws, inf_texts), orig_imgs
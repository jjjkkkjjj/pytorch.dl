from ..layers import *
from ..base import ObjectRecognitionModelBase
from .codec import CTCCodec
from ..._utils import _check_retval, _check_image, _get_normed_and_origin_img

from torch.nn import functional as F
import logging, abc

class CRNNBase(ObjectRecognitionModelBase):
    def __init__(self, class_labels, input_shape, blankIndex):
        super().__init__(class_labels, input_shape)

        self.blankIndex = blankIndex
        self.codec = CTCCodec(class_labels, blankIndex)

        self.conv_layers = _check_retval('build_conv', self.build_conv(), nn.ModuleDict)
        self.rec_layers = _check_retval('build_rec', self.build_rec(), nn.ModuleDict)

    @property
    def encoder(self):
        return self.codec.encoder

    @property
    def decoder(self):
        return self.codec.decoder

    @abc.abstractmethod
    def build_conv(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def build_rec(self):
        raise NotImplementedError()

    def forward(self, x, targets=None):
        """
        :param x: input images tensor, shape = (b, c, h, w)
        :param targets: text numbers, list of tensor, represents number as text. tensor's shape = (length of text)
        :return:
            output: output tensor, shape = (times, b, class_nums)
        """
        if self.training and targets is None:
            raise ValueError("pass \'targets\' for training mode")

        elif not self.training and targets is not None:
            logging.warning("forward as eval mode, but passed \'targets\'")


        batch_num = x.shape[0]

        for name, layer in self.conv_layers.items():
            x = layer(x)

        b, c, h, w = x.shape
        assert h == 1, "the height of conv must be 1"
        # feature
        x = x.squeeze(2) # remove height due to 1
        x = x.permute(2, 0, 1)  # [w, b, c]

        for name, layer in self.rec_layers.items():
            x = layer(x)

        if self.training:
            # apply log softmax for ctc loss, shape = (times, b, class_labels)
            predicts = F.log_softmax(x, dim=2)
            targets, target_lengths = self.encoder(targets)
            predict_lengths = torch.LongTensor([x.shape[0]] * batch_num)

            return predicts, targets, predict_lengths, target_lengths

        else:
            # apply softmax for prediction, shape = (times, b, class_labels)
            predicts = F.softmax(x, dim=2)
            raw_texts, out_texts = self.decoder(predicts)
            return predicts, raw_texts, out_texts


    def infer(self, image, toNorm=False):
        if self.training:
            raise NotImplementedError("call \'eval()\' first")

        # img: Tensor, shape = (b, c, h, w)
        img, orig_imgs = _check_image(image, self.device, size=(self.input_width, self.input_height))

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_imgs, orig_imgs = _get_normed_and_origin_img(img, orig_imgs, (0.5,), (0.5,), toNorm, self.device)

        return self(normed_imgs)
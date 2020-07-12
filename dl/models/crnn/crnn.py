from ..layers import *
from ..base import ObjectRecognitionModelBase
from .codec import CTCCodec
from ..._utils import _check_image, _get_normed_and_origin_img

import string, logging
from torch.nn import functional as F

class CRNN(ObjectRecognitionModelBase):
    def __init__(self, class_labels=tuple(string.ascii_lowercase), input_shape=(32, None, 1), leakyReLu=False, blankIndex=0):
        super().__init__(class_labels, input_shape)
        self.leakyReLu = leakyReLu

        conv_layers = [
            *Conv2d.block_relumpool(1, 1, self.input_channel, 64, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 2)),

            *Conv2d.block_relumpool(2, 1, 64, 128, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 2)),

            *Conv2d.block_relumpool(3, 2, 128, 256, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(1, 2), pool_stride=(2, 2)),

            *Conv2d.block_relumpool(4, 2, 256, 512, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(1, 2), pool_stride=(2, 2)),

            *Conv2d.relu_one(5, 512, 512, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0),
                             batch_norm=False, relu_inplace=True)
        ]

        self.conv_layers = nn.ModuleDict(conv_layers)

        rec_layers = [
            ('BiLSTM1', BidirectionalLSTM(512, 256, 256)),
            ('BiLSTM2', BidirectionalLSTM(256, 256, self.class_nums)),
        ]

        self.rec_layers = nn.ModuleDict(rec_layers)

        self.blankIndex = blankIndex
        self.codec = CTCCodec(class_labels, blankIndex)

    @property
    def encoder(self):
        return self.codec.encoder
    @property
    def decoder(self):
        return self.codec.decoder


    def forward(self, x, targets=None):
        """
        :param x: input images tensor, shape = (b, c, h, w)
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
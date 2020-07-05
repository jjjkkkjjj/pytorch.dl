from ..layers import *
from ..base import ImageRecognitionBase

import string


class CRNN(ImageRecognitionBase):
    def __init__(self, class_labels=tuple(string.ascii_lowercase), input_shape=(32, None, 1), leakyReLu=False):
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

    def forward(self, x):
        features = self.conv_layers(x)

        b, c, h, w = features.shape
        assert h == 1, "the height of conv must be 1"
        features = features.squeeze(2) # remove height due to 1
        features = features.permute(2, 0, 1)  # [w, b, c]

        output = self.rec_layers(features)
        return output


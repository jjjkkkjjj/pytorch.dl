from ..layers import *
from .base import CRNNBase

import string, logging

class CRNN(CRNNBase):
    def __init__(self, class_labels=tuple(string.ascii_lowercase), input_shape=(32, None, 1), leakyReLu=False, blankIndex=0):
        super().__init__(class_labels, input_shape, blankIndex)
        self.leakyReLu = leakyReLu

    def build_conv(self):
        conv_layers = [
            *Conv2d.block_relumpool(1, 1, self.input_channel, 64, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 2)),

            *Conv2d.block_relumpool(2, 1, 64, 128, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 2)),

            *Conv2d.block_relumpool(3, 2, 128, 256, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=False, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 1), pool_padding=(0, 1)),

            *Conv2d.block_relumpool(4, 2, 256, 512, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 2), pool_stride=(2, 1), pool_padding=(0, 1)),

            *Conv2d.relu_one(5, 512, 512, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0),
                             batch_norm=False, relu_inplace=True)
        ]

        return nn.ModuleDict(conv_layers)

    def build_rec(self):
        rec_layers = [
            ('BiLSTM1', BidirectionalLSTM(512, 256, 256)),
            ('BiLSTM2', BidirectionalLSTM(256, 256, self.class_nums)),
        ]

        return nn.ModuleDict(rec_layers)

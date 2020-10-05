from ...layers import *
from ...crnn.base import CRNNBase

class CRNN(CRNNBase):
    def __init__(self, class_labels, input_shape, blankIndex):
        super().__init__(class_labels, input_shape, blankIndex)
        #assert self.input_height == 8, 'height must be 8'

    def build_conv(self):
        conv_layers = [
            *Conv2d.block_relumpool(1, 2, self.input_channel, 64, conv_k_size=(3, 3), conv_stride=(1, 1),
                                    conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1)),

            *Conv2d.block_relumpool(2, 2, 64, 128, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1)),

            *Conv2d.block_relumpool(3, 2, 128, 256, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1))
        ]

        return nn.ModuleDict(conv_layers)

    def build_rec(self):
        rec_layers = [
            ('BiLSTM', BidirectionalLSTM(256, 256, self.class_nums)),
        ]

        return nn.ModuleDict(rec_layers)

class CRNNin64(CRNNBase):
    def __init__(self, class_labels, input_shape, blankIndex):
        super().__init__(class_labels, input_shape, blankIndex)
        #assert self.input_height == 8, 'height must be 8'

    def build_conv(self):
        conv_layers = [
            *Conv2d.block_relumpool(1, 2, self.input_channel, 128, conv_k_size=(3, 3), conv_stride=(1, 1),
                                    conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1)),

            *Conv2d.block_relumpool(2, 2, 128, 256, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1)),

            *Conv2d.block_relumpool(3, 2, 256, 512, conv_k_size=(3, 3), conv_stride=(1, 1), conv_padding=(1, 1),
                                    batch_norm=True, relu_inplace=True, pool_k_size=(2, 1), pool_stride=(2, 1))
        ]

        return nn.ModuleDict(conv_layers)

    def build_rec(self):
        rec_layers = [
            ('BiLSTM1', BidirectionalLSTM(512, 256, 256)),
            ('BiLSTM2', BidirectionalLSTM(256, 256, self.class_nums)),
        ]

        return nn.ModuleDict(rec_layers)
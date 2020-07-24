from ..base import TextSpottingModelBase

from .detn import SharedConv


class FOTS(TextSpottingModelBase):
    def __init__(self, input_shape):
        super().__init__(input_shape)

        self.sharedConv = SharedConv()


    def forward(self, x, targets=None):
        self.sharedConv(x)
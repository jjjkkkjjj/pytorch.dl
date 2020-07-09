from torch import nn
import abc


class CodecBase(nn.Module):

    @abc.abstractmethod
    def encoder(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def decoder(self, *args, **kwargs):
        pass

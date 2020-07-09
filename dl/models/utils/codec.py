from ..._utils import _check_ins

from torch import nn

class EncoderBase(nn.Module):
    pass

class DecoderBase(nn.Module):
    pass


class CodecBase(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = _check_ins('encoder', encoder, EncoderBase)
        self.decoder = _check_ins('decoder', decoder, DecoderBase)

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.encoder = self.encoder.cuda(device)
        self.decoder = self.decoder.cuda(device)

        return super().cuda(device)
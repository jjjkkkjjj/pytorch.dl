import torch, re
from torch.nn.utils.rnn import pad_sequence

from ...base.codec import CodecBase

class CTCCodec(CodecBase):
    def __init__(self, class_labels, blankIndex):
        """
        :param class_labels:
        :param blank_index: None or int, if None is passed, append blank
        """
        super().__init__()

        self.class_labels = class_labels
        self.blankIndex = blankIndex

    @property
    def blank(self):
        return self.class_labels[self.blankIndex]

    def encoder(self, text_numbers):
        """
        :param text_numbers: list of tensor, represents number as text. tensor's shape = (length of text)
                        Note that the length of each tensor is unsame.
        :return:
            targets: LongTensor, shape = (b, max length of text)
            target_lengths: LongTensor, shape = (b,)
        """
        # shape = (b, max length of text)
        targets = pad_sequence(text_numbers, batch_first=True, padding_value=self.blankIndex).long()
        target_lengths = torch.LongTensor([text.size().numel() for text in text_numbers])

        return targets, target_lengths

    def decoder(self, predicts):
        """
        :param predicts: Tensor, shape = (times, b, class_nums)
        :return:
            raw_strings: list of str, raw strings
            decoded_strings: list of str, decoded strings
        """
        _, inds = predicts.max(dim=2)
        # inds' shape = (b, times)
        inds = inds.permute((1, 0)).contiguous().cpu().tolist()
        batch_num = len(inds)

        raw_strings = []
        decoded_strings = []
        for b in range(batch_num):
            # convert raw string
            raw = ''.join(self.class_labels[ind] for ind in inds[b])
            # re.sub(r'(.)\1{1,}', r'\1', 'bbb---bbas00aacc')
            # b-bas0ac

            # gather multiple characters
            decoded = re.sub(r'(.)\1{1,}', r'\1', raw)
            # remove blank
            decoded = str(decoded).replace(self.blank, '')

            raw_strings += [raw]
            decoded_strings += [decoded]

        return raw_strings, decoded_strings

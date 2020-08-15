import os

from .._utils import DATA_ROOT

SynthText_class_labels = ['text']
SynthText_class_nums = len(SynthText_class_labels)

SynthText_char_labels_with_upper = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
SynthText_char_nums_with_upper = len(SynthText_char_labels_with_upper)

SynthText_char_labels_with_upper_blank = [' '] + SynthText_char_labels_with_upper
SynthText_char_nums_with_upper_blank = len(SynthText_char_labels_with_upper_blank)

SynthText_char_labels_without_upper = list('0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
SynthText_char_nums_without_upper = len(SynthText_char_labels_without_upper)

SynthText_char_labels_without_upper_blank = [' '] + SynthText_char_labels_without_upper
SynthText_char_nums_without_upper_blank = len(SynthText_char_labels_without_upper_blank)

SynthText_ROOT = os.path.join(DATA_ROOT, 'text', 'SynthText')
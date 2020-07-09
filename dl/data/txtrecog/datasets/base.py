import string

from ...objrecog.datasets.base import ObjectRecognitionDatasetBase

ALPHABET_LABELS = list(string.ascii_lowercase)
ALPHABET_NUMBERS = len(ALPHABET_LABELS)

NUMBER_LABELS = [str(i) for i in range(10)]
NUMBER_NUMBERS = len(NUMBER_LABELS)

ALPHANUMERIC_LABELS = ALPHABET_LABELS + NUMBER_LABELS
ALPHANUMERIC_NUMBERS = ALPHABET_NUMBERS + NUMBER_NUMBERS

ALPHANUMERIC_WITH_BLANK_LABELS = ['-'] + ALPHABET_LABELS + NUMBER_LABELS
ALPHANUMERIC_WITH_BLANK_NUMBERS = 1 + ALPHABET_NUMBERS + NUMBER_NUMBERS

class TextRecognitionDatasetBase(ObjectRecognitionDatasetBase):
    def __getitem__(self, index):
        img, targets = super().__getitem__(index)
        texts = targets[0]
        return img, texts
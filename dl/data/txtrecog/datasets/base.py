import string

from ...objrecog.datasets.base import ObjectRecognitionDatasetBase

ALPHABET_LABELS = list(string.ascii_lowercase)
ALPHABET_NUMBERS = len(ALPHABET_LABELS)

NUMBER_LABELS = [str(i) for i in range(10)]
NUMBER_NUMBERS = len(NUMBER_LABELS)

ALPHANUMERIC_LABELS = ALPHABET_LABELS + NUMBER_LABELS
ALPHANUMERIC_NUMBERS = ALPHABET_NUMBERS + NUMBER_NUMBERS

class TextRecognitionDatasetBase(ObjectRecognitionDatasetBase):
    pass
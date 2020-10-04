import string

from ...objrecog.datasets.base import ObjectRecognitionDatasetBase

Alphabet_labels = list(string.ascii_lowercase)
Alphabet_with_upper_labels = Alphabet_labels + list(string.ascii_uppercase)
Alphabet_numbers = len(Alphabet_labels)
Alphabet_with_upper_numbers = len(Alphabet_with_upper_labels)

Number_labels = [str(i) for i in range(10)]
Number_numbers = len(Number_labels)

Alphanumeric_labels = Alphabet_labels + Number_labels
Alphanumeric_numbers = Alphabet_numbers + Number_numbers
Alphanumeric_with_upper_labels = Alphabet_with_upper_labels + Number_labels
Alphanumeric_with_upper_numbers = Alphabet_with_upper_numbers + Number_numbers

Alphanumeric_with_blank_labels = ['-'] + Alphabet_labels + Number_labels
Alphanumeric_with_blank_numbers = 1 + Alphabet_numbers + Number_numbers
Alphanumeric_with_upper_and_blank_labels = ['-'] + Alphabet_with_upper_labels + Number_labels
Alphanumeric_with_upper_and_blank_numbers = 1 + Alphabet_with_upper_numbers + Number_numbers

class TextRecognitionDatasetBase(ObjectRecognitionDatasetBase):
    def __getitem__(self, index):
        img, targets = self.get_imgtarget(index)
        texts = targets[0]
        return img, texts
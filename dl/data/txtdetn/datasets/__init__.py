from .coco import *
from .synthtext import *
from .icdar import *
from ...objdetn.datasets.base import Compose
from ...txtrecog.datasets import (
    Alphanumeric_labels,
    Alphanumeric_numbers,
    Alphabet_labels,
    Alphabet_numbers,
    Alphanumeric_with_blank_labels,
    Alphanumeric_with_blank_numbers,
    Number_labels,
    Number_numbers
)
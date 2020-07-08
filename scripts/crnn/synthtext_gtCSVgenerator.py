import sys
sys.path.append('../../')
from dl.data.utils.synthtext import TextRecogCSVGenerator, TextRecogOnlyAlphabetNumberCSVGenerator

import argparse

parser = argparse.ArgumentParser(description='Generate Synthtext\'s annotation xml file')
parser.add_argument('path', help='directory path under \'SynthText\'(, \'licence.txt\')',
                    type=str)
parser.add_argument('-id', '--image_dirname', help='image directory name including \'gt.mat\'',
                    type=str, default='SynthText')
parser.add_argument('-sm', '--skip_missing', help='Wheter to skip missing image',
                    action='store_true')
parser.add_argument('-e', '--encoding', help='encoding',
                    type=str, default='utf-8')
parser.add_argument('-an', '--alphanumeric', help='Wheter to non-alphanumeric string',
                    action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    if args.alphanumeric:
        TextRecogOnlyAlphabetNumberCSVGenerator(args.path, imagedirname=args.image_dirname, skip_missing=args.skip_missing)
    else:
        TextRecogCSVGenerator(args.path, imagedirname=args.image_dirname, skip_missing=args.skip_missing)
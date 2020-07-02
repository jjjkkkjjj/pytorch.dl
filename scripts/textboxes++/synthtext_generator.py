from text_data.datasets.synthtext import annotationGenerator

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

args = parser.parse_args()

if __name__ == '__main__':
    annotationGenerator(args.path, imagedirname=args.image_dirname, skip_missing=args.skip_missing)
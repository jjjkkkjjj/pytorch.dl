from dl.data.txtdetn.downloader import *
from dl.data.txtdetn.downloader import choices

import argparse

parser = argparse.ArgumentParser(description='Download datasets')
parser.add_argument('--datasets', help='select datasets from {}'.format(choices),
                    choices=choices, nargs='*', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    if 'coco2014_all' in args.datasets:
        coco2014_all()
import cv2, os, sys
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Convert png to jpg')
parser.add_argument('path', help='directory path', type=str)
parser.add_argument('-d', '--delete', help='delete original png image', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.isdir(args.path):
        raise IsADirectoryError('{} was not directory'.format(args.path))

    paths = list(Path(args.path).rglob('*.png'))
    for i, path in enumerate(paths):
        img = cv2.imread(str(path.absolute()), cv2.IMREAD_UNCHANGED)

        ret_path = str(path.absolute())[:-3] + 'jpg'
        cv2.imwrite(ret_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if args.delete:
            os.remove(str(path.absolute()))

        sys.stdout.write('\rConverting...\t{}%\t[{}/{}]'.format(100*(i + 1.0)/len(paths), i+1, len(paths)))
        sys.stdout.flush()

    print()
    print('finished')
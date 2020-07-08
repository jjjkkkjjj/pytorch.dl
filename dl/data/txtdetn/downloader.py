choices = ['coco2014_all']
__all__ = choices
import logging, glob, os, shutil
from ..base.downloader import _Downloader
from .._utils import DATA_ROOT

logging.basicConfig(level=logging.INFO)

def coco2014_all():
    logging.info('Downloading coco2014_all')

    # get train2014 image
    if not _isImgExist(DATA_ROOT + '/coco/coco2014/trainval', 'train2014'):
        train_downloader = _Downloader('http://images.cocodataset.org/zips/train2014.zip', 'zip')
        train_downloader.run(DATA_ROOT + '/coco/coco2014/trainval/images', 'train2014')

    # annotations
    if not os.path.exists(DATA_ROOT + '/coco/coco2014/trainval/annotations/COCO_Text.json'):
        annotation_downloader = _Downloader('https://vision.cornell.edu/se3/wp-content/uploads/2019/05/COCO_Text.zip', 'zip')
        annotation_downloader.run(DATA_ROOT + '/coco/coco2014', 'tmp', remove_comp_file=True)
        _move(DATA_ROOT + '/coco/coco2014/tmp', DATA_ROOT + '/coco/coco2014/trainval/annotations')
    else:
        logging.info('you have already downloaded \"{}\"'.format('COCO_Text.json'))

    logging.info('Downloaded coco2014_all')


def _isImgExist(dirpath, focus):
    if len(glob.glob(os.path.join(dirpath, 'images', focus, '*'))) > 0:
        logging.info('you have already downloaded \"{}\"'.format(focus))
        return True
    else:
        return False

def _move(srcdir, dstdir):
    os.makedirs(dstdir, exist_ok=True)

    srcpaths = glob.glob(os.path.join(srcdir, '*'))

    for srcpath in srcpaths:
        shutil.move(srcpath, dstdir)

    # remove srcdir
    shutil.rmtree(srcdir)
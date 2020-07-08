choices = ['voc2007_trainval', 'voc2007_test', 'voc2012_trainval', 'voc2012_test', 'coco2014_trainval', 'coco2017_trainval']
__all__ = choices

import logging, os, shutil, glob

from .._utils import DATA_ROOT
from ..base.downloader import _Downloader

def voc2007_trainval():
    logging.info('Downloading voc2007_trainval')

    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar')
    trainval_downloader.run(DATA_ROOT + '/voc/voc2007', 'trainval')

    logging.info('Downloaded voc2007_trainval')

def voc2007_test():
    logging.info('Downloading voc2007_test')

    test_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar')
    test_downloader.run(DATA_ROOT + '/voc/voc2007', 'test')

    logging.info('Downloaded voc2007_test')

def voc2012_trainval():
    logging.info('Downloading voc2012_trainval')

    trainval_downloader = _Downloader('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
    trainval_downloader.run(DATA_ROOT + '/voc/voc2012', 'trainval')

    logging.info('Downloaded voc2012_trainval')

def voc2012_test():
    logging.info('Downloading voc2012_test')

    test_downloader = _Downloader('http://pjreddie.com/media/files/VOC2012test.tar')
    test_downloader.run(DATA_ROOT + '/voc/voc2012', 'test')

    logging.info('Downloaded voc2012_test')

def coco2014_trainval():
    logging.info('Downloading coco2014_trainval')

    # annotations
    trainval_downloader = _Downloader('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', 'zip')
    trainval_downloader.run(DATA_ROOT + '/coco/coco2014', 'trainval', remove_comp_file=True)

    # get images
    train_downloader = _Downloader('http://images.cocodataset.org/zips/train2014.zip', 'zip')
    train_downloader.run(DATA_ROOT + '/coco/coco2014/train', 'images', remove_comp_file=True)

    val_downloader = _Downloader('http://images.cocodataset.org/zips/val2014.zip', 'zip')
    val_downloader.run(DATA_ROOT + '/coco/coco2014/val', 'images', remove_comp_file=True)

    _concat_trainval_images('/coco/coco2014', srcdirs=('train', 'val'), dstdir='trainval')

    logging.info('Downloaded coco2014_trainval')

def coco2017_trainval():
    logging.info('Downloading coco2017_trainval')

    # annotations
    trainval_downloader = _Downloader('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'zip')
    trainval_downloader.run(DATA_ROOT + '/coco/coco2017', 'trainval', remove_comp_file=True)

    # get images
    train_downloader = _Downloader('http://images.cocodataset.org/zips/train2017.zip', 'zip')
    train_downloader.run(DATA_ROOT + '/coco/coco2017/train', 'images', remove_comp_file=True)

    val_downloader = _Downloader('http://images.cocodataset.org/zips/val2017.zip', 'zip')
    val_downloader.run(DATA_ROOT + '/coco/coco2017/val', 'images', remove_comp_file=True)

    _concat_trainval_images('/coco/coco2017', srcdirs=('train', 'val'), dstdir='trainval')

    logging.info('Downloaded coco2017_trainval')



def _concat_trainval_images(basedir, srcdirs=('train', 'val'), dstdir='trainval'):
    srcpaths = []
    for srcname in srcdirs:
        srcpaths.extend(glob.glob(os.path.join(DATA_ROOT + basedir, srcname, 'images', '*')))

    dstpath = os.path.join(DATA_ROOT + basedir, dstdir, 'images')

    os.makedirs(dstpath, exist_ok=True)

    for srcpath in srcpaths:
        shutil.move(srcpath, dstpath)

    if len(glob.glob(os.path.join(dstpath, '*'))) > 0:
        # remove source
        for srcname in srcdirs:
            shutil.rmtree(os.path.join(DATA_ROOT + basedir, srcname))
    else:
        raise AssertionError('could not move files!!')
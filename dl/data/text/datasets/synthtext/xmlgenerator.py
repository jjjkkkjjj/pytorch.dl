import os, logging, re, shutil, sys
from scipy import io as sio
from lxml import etree
import cv2
import numpy as np

def annotationGenerator(basedir, imagedirname='SynthText', skip_missing=False, encoding='utf-8'):
    """
    convert gt.mat to https://github.com/MhLiao/TextBoxes_plusplus/blob/master/data/example.xml

    <annotation>
        <folder>train_images</folder>
        <filename>img_10.jpg</filename>
        <size>
            <width>1280</width>
            <height>720</height>
            <depth>3</depth>
        </size>
        <object>
            <difficult>1</difficult>
            <content>###</content>
            <name>text</name>
            <bndbox>
                <x1>1011</x1>
                <y1>157</y1>
                <x2>1079</x2>
                <y2>160</y2>
                <x3>1076</x3>
                <y3>173</y3>
                <x4>1011</x4>
                <y4>170</y4>
                <xmin>1011</xmin>
                <ymin>157</ymin>
                <xmax>1079</xmax>
                <ymax>173</ymax>
            </bndbox>
        </object>
        .
        .
        .

    </annotation>

    :param basedir: str, directory path under \'SynthText\'(, \'licence.txt\')
    :param imagedirname: (Optional) str, image directory name including \'gt.mat\
    :return:
    """
    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(basedir, imagedirname)
    gtpath = os.path.join(imagedir, 'gt.mat')

    annodir = os.path.join(basedir, 'Annotations')

    if not os.path.exists(gtpath):
        raise FileNotFoundError('{} was not found'.format(gtpath))

    if os.path.exists(annodir):
        logging.warning('{} has already existed. Would you remove it?[N/y]')
        key = input()
        if re.match(r'y|yes', key, flags=re.IGNORECASE):
            shutil.rmtree(annodir)
            logging.warning('Removed {}'.format(annodir))
        else:
            logging.warning('Please remove or rename it')
            exit()

    # create Annotations directory
    os.mkdir(annodir)

    """
    ref: http://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt
    gts = dict;
        __header__: bytes
        __version__: str
        __globals__: list
        charBB: object ndarray, shape = (1, image num). 
                Character level bounding box. shape = (2=(x,y), 4=(top left,...: clockwise), BBox word num)
        wordBB: object ndarray, shape = (1, image num). 
                Word level bounding box. shape = (2=(x,y), 4=(top left,...: clockwise), BBox char num)
        imnames: object ndarray, shape = (1, image num, 1).
        txt: object ndarray, shape = (i, image num).
             Text. shape = (word num)
    """
    logging.info('Loading {} now.\nIt may take a while.'.format(gtpath))
    gts = sio.loadmat(gtpath)
    logging.info('Loaded\n'.format(gtpath))

    charBB = gts['charBB'][0]
    wordBB = gts['wordBB'][0]
    imnames = gts['imnames'][0]
    texts = gts['txt'][0]

    image_num = imnames.size

    for i, (cbb, wBB, imname, txts) in enumerate(zip(charBB, wordBB, imnames, texts)):
        imname = imname[0]

        imgpath = os.path.join(imagedir, imname)
        if not os.path.exists(imgpath):
            if not skip_missing:
                raise FileNotFoundError('{} was not found'.format(imgpath))
            else:
                logging.warning('Missing image: {}'.format(imgpath))
                continue

        root = etree.Element('annotation')

        # folder
        folderET = etree.SubElement(root, 'folder')
        folder = os.path.dirname(imname)
        folderET.text = folder
        # filename
        filenameET = etree.SubElement(root, 'filename')
        filename = os.path.basename(imname)
        filenameET.text = filename

        # read image to get height, width, channel
        img = cv2.imread(imgpath)
        h, w, c = img.shape

        # size
        sizeET = etree.SubElement(root, 'size')

        # width
        widthET = etree.SubElement(sizeET, 'width')
        widthET.text = str(w)
        # height
        heightET = etree.SubElement(sizeET, 'height')
        heightET.text = str(h)
        # depth
        depthET = etree.SubElement(sizeET, 'depth')
        depthET.text = str(c)

        # convert txts to list of str
        # I don't know why txts is
        # ['Lines:\nI lost\nKevin ', 'will                ', 'line\nand            ',
        # 'and\nthe             ', '(and                ', 'the\nout             ',
        # 'you                 ', "don't\n pkg          "]
        # there is strange blank and the length of txts is different from the one of wBB
        txts = ' '.join(txts.tolist()).split()
        text_num = len(txts)

        if wBB.ndim == 2:
            # convert shape=(2, 4,) to (2, 4, 1)
            wBB = np.expand_dims(wBB, 2)

        assert text_num == wBB.shape[2], 'The length of text and wordBB must be same, but got {} and {}'.format(text_num, wBB.shape[2])
        for b in range(text_num):
            # object
            objectET = etree.SubElement(root, 'object')

            # difficult
            difficultET = etree.SubElement(objectET, 'difficult')
            difficultET.text = '0'
            # content
            contentET = etree.SubElement(objectET, 'content')
            contentET.text = '###'
            # name
            nameET = etree.SubElement(objectET, 'name')
            nameET.text = txts[b]
            # bndbox
            bndboxET = etree.SubElement(objectET, 'bndbox')

            # quad
            for q in range(4):
                xET = etree.SubElement(bndboxET, 'x{}'.format(q + 1))
                xET.text = str(wBB[0, q, b])
                yET = etree.SubElement(bndboxET, 'y{}'.format(q + 1))
                yET.text = str(wBB[1, q, b])

            # corner
            xminET = etree.SubElement(bndboxET, 'xmin')
            xminET.text = str(np.min(wBB[0, :, b]))
            yminET = etree.SubElement(bndboxET, 'ymin')
            yminET.text = str(np.min(wBB[1, :, b]))
            xmaxET = etree.SubElement(bndboxET, 'xmax')
            xmaxET.text = str(np.max(wBB[0, :, b]))
            ymaxET = etree.SubElement(bndboxET, 'ymax')
            ymaxET.text = str(np.max(wBB[1, :, b]))

        xmlstr = etree.tostring(root, pretty_print=True, encoding=encoding)
        dstpath = os.path.join(annodir, folder, os.path.splitext(filename)[0] + '.xml')

        if not os.path.isdir(os.path.dirname(dstpath)):
            os.mkdir(os.path.dirname(dstpath))

        with open(dstpath, 'wb') as f:
            f.write(xmlstr)


        sys.stdout.write('\rGenerating... {:0.1f}% ({}/{})'.format(100*(float(i + 1)/image_num), i+1, image_num))
        sys.stdout.flush()

    print()
    logging.info('Finished!!!')
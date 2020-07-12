import cv2

from dl.data.txtrecog import datasets
from dl.models import CRNN

if __name__ == '__main__':
    model = model = CRNN(class_labels=datasets.ALPHANUMERIC_WITH_BLANK_LABELS, input_shape=(32, 100, 1)).cuda()
    model.eval()
    model.load_weights('../../weights/crnn-synthtext/i-3000000_checkpoints20200710.pth')
    print(model)

    image = cv2.cvtColor(cv2.imread('../../scripts/crnn/assets/test.jpg'), cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(cv2.imread('../../scripts/crnn/assets/demo.png'), cv2.COLOR_BGR2GRAY)
    #cv2.imshow('test image', image)
    #cv2.waitKey()
    _, raws, outs = model.infer(image, toNorm=True)
    for raw, out in zip(raws, outs):
        print('{} -> {}'.format(raw, out))
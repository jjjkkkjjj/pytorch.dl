import cv2

from dl.data.txtrecog import datasets
from dl.models import CRNN

if __name__ == '__main__':
    model = model = CRNN(class_labels=datasets.ALPHANUMERIC_WITH_BLANK_LABELS, input_shape=(32, None, 1)).cuda()
    model.eval()
    model.load_weights('../../weights/crnn-synthtext/i-3000000_checkpoints20200710.pth')
    print(model)

    image = cv2.cvtColor(cv2.imread('../../scripts/crnn/assets/test.png'), cv2.COLOR_BGR2GRAY)
    model.infer(image, toNorm=True)
import cv2, torch

from dl.data.txtrecog import datasets
from dl.models import FOTSRes50

if __name__ == '__main__':
    model = FOTSRes50(chars=datasets.SynthText_char_labels_without_upper_blank, input_shape=(None, None, 3)).cuda()
    model.eval()
    model.load_weights('../../weights/fots-res50/e2_noaug.pth')
    print(model)

    image = cv2.cvtColor(cv2.imread('../../scripts/fots/assets/test.jpg'), cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(cv2.imread('../../scripts/crnn/assets/demo.png'), cv2.COLOR_BGR2RGB)
    cv2.imshow('test image', cv2.resize(image, (640, 640)))
    cv2.waitKey()
    img = torch.from_numpy(cv2.resize(image, (640, 640)).transpose((2,0,1))).unsqueeze(0)
    img = (img.float()/255. - torch.tensor((0.485, 0.456, 0.406)).reshape((-1, 1, 1)))/torch.tensor((0.229, 0.224, 0.225)).reshape((-1, 1, 1))
    print(model(img.cuda()))

    _, raws, outs = model.infer(image, toNorm=True)
    for raw, out in zip(raws, outs):
        print('{} -> {}'.format(raw, out))
# CRNN

![demo](./assets/demo.png?raw=true "demo")

Result >

```bash
av------a--i-la--b--l----e -> available
```

![test](./assets/test.jpg?raw=true "test")

Result >

```bash
go----------o---g---l----e -> google
```

# Pre-train

- First, download SynthText dataset from official.

- Second, convert `gt.mat` into csv file with alphanumeric text only.

  ```bash
  python synthtext_gtCSVgenerator.py {path} -id SynthText -sm -an
  ```

  ```python
  usage: synthtext_gtCSVgenerator.py [-h] [-id IMAGE_DIRNAME] [-sm]
                                     [-e ENCODING] [-an]
                                     path
  
  Generate Synthtext's annotation xml file
  
  positional arguments:
    path                  directory path under 'SynthText'(, 'licence.txt')
  
  optional arguments:
    -h, --help            show this help message and exit
    -id IMAGE_DIRNAME, --image_dirname IMAGE_DIRNAME
                          image directory name including 'gt.mat'
    -sm, --skip_missing   Wheter to skip missing image
    -e ENCODING, --encoding ENCODING
                          encoding
    -an, --alphanumeric   Wheter to non-alphanumeric string
  ```

  

- Train. See [demo/train-SynthText.ipynb](../../demo/crnn/train-synthtext.ipynb)

- You can download model with epoch 7 from [here](https://drive.google.com/file/d/1Ct4G7H-hQ9yCDhE4W2G3dK-612Xcqajf/view?usp=sharing)

# Test Script Example

- First, create model and load weight

```python
import cv2

from dl.data.txtrecog import datasets
from dl.models import CRNN

model = CRNN(class_labels=datasets.ALPHANUMERIC_WITH_BLANK_LABELS, input_shape=(32, 100, 1)).cuda()
model.eval()
model.load_weights('../../weights/crnn-synthtext/e-0000007_checkpoints20200814.pth')
print(model)
```

- Second, infer

```python
image = cv2.cvtColor(cv2.imread('./scripts/crnn/assets/test.jpg'), cv2.COLOR_BGR2GRAY)
#image = cv2.cvtColor(cv2.imread('./scripts/crnn/assets/demo.png'), cv2.COLOR_BGR2GRAY)

_, raws, outs = model.infer(image, toNorm=True)
for raw, out in zip(raws, outs):
    print('{} -> {}'.format(raw, out))
```


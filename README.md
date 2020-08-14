# pytorch.dl

## Installation
```bash
pip install torch==1.4.0+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install --upgrade git+https://github.com/jjjkkkjjj/pytorch.dl #--ignore-installed pycurl # <- maybe needed
```

## Model List

### CRNN

- Detail

  See [script](https://github.com/jjjkkkjjj/pytorch.dl/tree/master/scripts/crnn) in detail.

- Example

  ![demo](./scripts/crnn/assets/demo.png?raw=true "demo")

  Result >

  ```bash
  av------a--i-la--b--l----e -> available
  ```

  ![test](./scripts/crnn/assets/test.jpg?raw=true "test")

  Result >

  ```bash
  go----------o---g---l----e -> google
  ```

  

### SSD

- Detail

  See [script](https://github.com/jjjkkkjjj/pytorch.dl/tree/master/scripts/ssd) in detail.

- Example

  ![result img](./scripts/ssd/assets/coco_testimg-result.jpg?raw=true "result img")

### TextBox++

- Detail

  See [script](https://github.com/jjjkkkjjj/pytorch.dl/tree/master/scripts/textboxes%2B%2B) in detail.

- Example

  Note that this result are from model trained with SynthText and ICDAR2015. So, if you train printed text dataset, the precision will be much higher than below.

  ![icdar-trained img](./scripts/textboxes++/assets/train-icdar-result.png?raw=true "icdar-trained img")
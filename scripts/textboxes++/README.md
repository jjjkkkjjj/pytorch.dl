# TextBoxes++ with PyTorch

The implementation of [TextBoxes++](https://arxiv.org/abs/1801.02765) with PyTorch.

# Requirement

```bash
pip install --upgrade git+https://github.com/jjjkkkjjj/pytorch_SSD.git
conda install lxml
conda install -c conda-forge shapely
```

# Pre-train

- First, download SynthText dataset from [official](https://www.robots.ox.ac.uk/~vgg/data/scenetext/).

- Second, convert `gt.mat` into annotation xml files using `synthtext_generator.py`.

  ```bash
  python synthtext_generator.py {path} -id SynthText
  ```

  ```bash
  usage: synthtext_generator.py [-h] [-in IMAGE_DIRNAME] [-sm] [-e ENCODING]
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
  ```

- Train. See [demo/pre-train-SynthText.ipynb](demo/pre-train-SynthText.ipynb).

- You can download pre-trained model from [here](https://drive.google.com/file/d/1unqLYGhbORYHHWy7UtZkHA-C-MrN19mf/view?usp=sharing).

- Pre-trained model's output example;

![pre-trained img](assets/pre-train-result.png?raw=true "pre-trained img")

# Train ICDAR2015

- First, download dataset from [official](https://rrc.cvc.uab.es/?ch=4&com=downloads).

- Second, place annotation `.txt` and `.jpg` like this;

  ```bash
  ├── Annotations (place .txt)
  └── Images (place .jpg)
  ```

- Train. See [demo/train-ICDAR2015.ipynb](demo/train-ICDAR2015.ipynb).

- You can download pre-trained model from [here](https://drive.google.com/file/d/1vb7xnqClTy612qay7On1K37ZyAoulxd1/view?usp=sharing).

- ICDAR's model output example;

![icdar-trained img](assets/train-icdar-result.png?raw=true "icdar-trained img")

# Convert png to jpg for Born Digital Images

```bash
$ python png2jpg ~/data/text/Born-Digital-Images/Images/ -d
Converting...	100.0%	[307/307]
finished
```

# Reference

[SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

[COCO-text](https://vision.cornell.edu/se3/coco-text-2/#terms-of-use)

[COCO-text api](https://github.com/bgshih/coco-text)

[DDI-100](https://arxiv.org/pdf/1912.11658.pdf)

[DDI-100 api](https://github.com/machine-intelligence-laboratory/DDI-100)


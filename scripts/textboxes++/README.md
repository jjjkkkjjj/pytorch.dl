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

# Easy training

You can train (**your**) synthtext, coco or indar style dataset easily when you use `easy_train.py`!

Example;

```bash
python easy_train.py SynthText -r {your-synthtext-style-dataset-path} -lr 0.001
```

or

```bash
python easy_train.py COCO -r {your-coco-style-dataset-path} --focus COCO_Text --image_dir train2014 -lr 0.0005
```

```bash
usage: easy_train.py [-h] [-r DATASET_ROOTDIR [DATASET_ROOTDIR ...]]
                     [--focus FOCUS [FOCUS ...]] [--image_dir IMAGE_DIR]
                     [-ig [{difficult,strange} [{difficult,strange} ...]]]
                     [-is IMAGE_SIZE] [-n MODEL_NAME] [-w WEIGHTS_PATH]
                     [-bs BATCH_SIZE] [-nw NUM_WORKERS] [-d {cpu,cuda}]
                     [-si START_ITERATION] [-na] [-optimizer {SGD,Adam}]
                     [-lr LEARNING_RATE] [--momentum MOMENTUM]
                     [-wd WEIGHT_DECAY] [--steplr_gamma STEPLR_GAMMA]
                     [--steplr_milestones STEPLR_MILESTONES [STEPLR_MILESTONES ...]]
                     [-mi MAX_ITERATION] [-ci CHECKPOINTS_INTERVAL]
                     [--loss_alpha LOSS_ALPHA] [--neg_factor NEG_FACTOR]
                     {SynthText,COCO,ICDAR}

Easy training script for SynthText, COCO or ICDAR style dataset

positional arguments:
  {SynthText,COCO,ICDAR}
                        Dataset type

optional arguments:
  -h, --help            show this help message and exit
  -r DATASET_ROOTDIR [DATASET_ROOTDIR ...], --dataset_rootdir DATASET_ROOTDIR [DATASET_ROOTDIR ...]
                        Dataset root directory path. If dataset type is
                        'SynthText', Default is;
                        '['/home/kado/data/text/SynthText']' If dataset type
                        is 'COCO', Default is;
                        '['/home/kado/data/coco/coco2014/trainval']' If
                        dataset type is 'ICDAR', Default is;
                        '['/home/kado/data/text/ICDAR2015']'
  --focus FOCUS [FOCUS ...]
                        Image set name. if dataset type is 'COCO', Default is;
                        '['COCO_Text']'
  --image_dir IMAGE_DIR
                        Image set name. if dataset type is 'COCO', Default is;
                        'train2014'
  -ig [{difficult,strange} [{difficult,strange} ...]], --ignore [{difficult,strange} [{difficult,strange} ...]]
                        Whether to ignore object
  -is IMAGE_SIZE, --image_size IMAGE_SIZE
                        Trained model
  -n MODEL_NAME, --model_name MODEL_NAME
                        Model name, which will be used as save name
  -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        Pre-trained weights path. Default is pytorch's pre-
                        trained one for vgg
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers used in DataLoader
  -d {cpu,cuda}, --device {cpu,cuda}
                        Device for Tensor
  -si START_ITERATION, --start_iteration START_ITERATION
                        Resume training at this iteration
  -na, --no_augmentation
                        Whether to do augmentation to your dataset
  -optimizer {SGD,Adam}
                        Optimizer for training
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate
  --momentum MOMENTUM   Momentum value for Optimizer
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Weight decay for SGD
  --steplr_gamma STEPLR_GAMMA
                        Gamma for stepLR
  --steplr_milestones STEPLR_MILESTONES [STEPLR_MILESTONES ...]
                        Milestones for stepLR
  -mi MAX_ITERATION, --max_iteration MAX_ITERATION
  -ci CHECKPOINTS_INTERVAL, --checkpoints_interval CHECKPOINTS_INTERVAL
                        Checkpoints interval
  --loss_alpha LOSS_ALPHA
                        Loss's alpha
  --neg_factor NEG_FACTOR
                        Negative's factor for hard mining
```



# Reference

[SynthText](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

[COCO-text](https://vision.cornell.edu/se3/coco-text-2/#terms-of-use)

[COCO-text api](https://github.com/bgshih/coco-text)

[DDI-100](https://arxiv.org/pdf/1912.11658.pdf)

[DDI-100 api](https://github.com/machine-intelligence-laboratory/DDI-100)


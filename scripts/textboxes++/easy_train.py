import argparse
import os

from dl.data.text.datasets import SynthText_ROOT, COCO2014Text_ROOT, ICDARText_ROOT

synthtext_rootdir_default = [SynthText_ROOT]
coco_rootdir_default = [os.path.join(COCO2014Text_ROOT, 'trainval')]
icdar_rootdir_default = [ICDARText_ROOT]

coco_focus_default = ['COCO_Text']
coco_image_dir_default = 'train2014'

parser = argparse.ArgumentParser(description='Easy training script for SynthText, COCO or ICDAR style dataset')

# dataset type
# required
parser.add_argument('dataset_type', choices=['SynthText', 'COCO', 'ICDAR'],
                    type=str, help='Dataset type')
# root directory
parser.add_argument('-r', '--dataset_rootdir', default=None, nargs='+',
                    type=str, help='Dataset root directory path.\n'
                                   'If dataset type is \'SynthText\', Default is;\n\'{}\'\n\n'
                                   'If dataset type is \'COCO\', Default is;\n\'{}\'\n\n'
                                   'If dataset type is \'ICDAR\', Default is;\n\'{}\''.format(synthtext_rootdir_default, coco_rootdir_default, icdar_rootdir_default))
# focus
parser.add_argument('--focus', default=None, nargs='+',
                    type=str, help='Image set name.\n'
                                   'if dataset type is \'COCO\', Default is;\n\'{}\''.format(coco_focus_default))

# image dir for COCO
parser.add_argument('--image_dir', default=None,
                    type=str, help='Image set name.\n'
                                   'if dataset type is \'COCO\', Default is;\n\'{}\''.format(coco_image_dir_default))


# ignore difficult
parser.add_argument('-ig', '--ignore', choices=['difficult', 'strange'], nargs='*',
                    type=str, help='Whether to ignore object')
# model
parser.add_argument('-is', '--image_size', default=384, type=int, help='Trained model')
# model name
parser.add_argument('-n', '--model_name', default='TextBoxesPP', type=str,
                    help='Model name, which will be used as save name')
# batch normalization
#parser.add_argument('-bn', '--batch_norm', action='store_true',
#                    help='Whether to construct model with batch normalization')
# pretrained weight
parser.add_argument('-w', '--weights_path', type=str,
                    help='Pre-trained weights path. Default is pytorch\'s pre-trained one for vgg')
# batch size
parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='Batch size')
# num_workers in DataLoader
parser.add_argument('-nw', '--num_workers', default=4, type=int,
                    help='Number of workers used in DataLoader')
# device
parser.add_argument('-d', '--device', default='cuda', choices=['cpu', 'cuda'], type=str,
                    help='Device for Tensor')
#parser.add_argument('--resume', default=None, type=str,
#                    help='Checkpoint state_dict file to resume training from')
# start iteration
parser.add_argument('-si', '--start_iteration', default=0, type=int,
                    help='Resume training at this iteration')
# augmentation
parser.add_argument('-na', '--no_augmentation', action='store_false', default=False,
                    help='Whether to do augmentation to your dataset')
# optimizer
parser.add_argument('-optimizer', default='SGD', choices=['SGD', 'Adam'],
                    type=str, help='Optimizer for training')
# learning rate
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='Initial learning rate')
# momentum
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for Optimizer')
# weight decay
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# MultiStepLR gamma
parser.add_argument('--steplr_gamma', default=0.1, type=float,
                    help='Gamma for stepLR')
# MultiStepLR milestones
parser.add_argument('--steplr_milestones', default=[60000], type=int, nargs='+',
                    help='Milestones for stepLR')
# attr = list
# final iteration
parser.add_argument('-mi', '--max_iteration', default=60000, type=int,
                    help='')
# Checkpoints interval
parser.add_argument('-ci', '--checkpoints_interval', default=5000, type=int,
                    help='Checkpoints interval')
# loss alpha
parser.add_argument('--loss_alpha', default=0.2, type=float,
                    help='Loss\'s alpha')
# negative factor for hard mining
parser.add_argument('--neg_factor', default=3, type=int,
                    help='Negative\'s factor for hard mining')

args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO)
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from dl.data.text import datasets, utils, target_transforms, transforms, augmentations
from dl.models.ssd import *
from dl.loss.textboxespp import TextBoxLoss, ConfidenceLoss
from dl.optim.scheduler import IterMultiStepLR
from dl.log import *


rootdir = args.dataset_rootdir
if rootdir is None:
    if args.dataset_type == 'SynthText':
        rootdir = synthtext_rootdir_default
    elif args.dataset_type == 'COCO':
        rootdir = coco_rootdir_default
    else:
        rootdir = icdar_rootdir_default

class_labels = ['text']

focus = args.focus
if focus is None:
    if args.dataset_type == 'COCO':
        focus = coco_focus_default

image_dir = args.image_dir
if image_dir is None:
    if args.dataset_type == 'COCO':
        image_dir = coco_image_dir_default

if torch.cuda.is_available():
    if args.device != 'cuda':
        logging.warning('You can use CUDA device but you didn\'t set CUDA device.'
                        ' Run with \'-d cuda\' or \'--device cuda\'')
device = torch.device(args.device)

#### dataset ####
augmentation = None if args.no_augmentation else augmentations.RandomSampled()

size = (args.image_size, args.image_size)

transform = transforms.Compose(
    [transforms.Resize(size),
     transforms.ToTensor(),
     transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
)
target_transform = target_transforms.Compose(
    [target_transforms.Corners2Centroids(),
     target_transforms.OneHot(class_nums=len(class_labels), add_background=True),
     target_transforms.ToTensor()]
)

if args.ignore:
    kwargs = {key: True for key in args.ignore}
    ignore = target_transforms.Ignore(**kwargs)
else:
    ignore = None

if args.dataset_type == 'SynthText':
    train_dataset = datasets.SynthTextDetectionMultiDatasetBase(synthtext_dir=rootdir, ignore=ignore,
                                                                transform=transform, target_transform=target_transform, augmentation=augmentation)
elif args.dataset_type == 'COCO':
    train_dataset = datasets.COCOTextMultiDatasetBase(coco_dir=rootdir, focus=focus, image_dir=image_dir, datasetTypes=('train', 'val',), ignore=ignore,
                                                      transform=transform, target_transform=target_transform, augmentation=augmentation)
else:
    train_dataset = datasets.ICDARTextMultiDatasetBase(icdar_dir=rootdir, ignore=ignore,
                                                       transform=transform, target_transform=target_transform, augmentation=augmentation)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=utils.batch_ind_fn_droptexts, num_workers=args.num_workers, pin_memory=True)

logging.info('Dataset info:'
             '\nroot dir: {},'
             '\nfocus: {},'
             '\nlabels:{}'
             '\nignore object: {}'
             '\naugmentation: {}'
             '\nbatch size: {}'
             '\nnum_workers: {}\n'.format(rootdir, focus, class_labels,
                                          args.ignore, not args.no_augmentation,
                                          args.batch_size, args.num_workers))


#### model ####
model = TextBoxesPP(input_shape=(args.image_size, args.image_size, 3)).to(device)

if args.weights_path is None:
    model.load_vgg_weights()
else:
    model.load_weights(args.weights_path)

if args.device == 'cuda':
    model = nn.DataParallel(model)

logging.info(model)

### train info ###
if args.optimizer == 'SGD':
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    logging.info('Optimizer Info:'
                 '\nOptimizer: {}'
                 '\nlearning rate: {}, Momentum: {}, Weight decay: {}\n'.format(args.optimizer, args.learning_rate, args.momentum, args.weight_decay))
elif args.optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    logging.info('Optimizer Info:'
                 '\nOptimizer: {}'
                 '\nlearning rate: {}, Weight decay: {}\n'.format(args.optimizer, args.learning_rate, args.weight_decay))
else:
    assert False, "Invalid optimizer"

iter_scheduler = IterMultiStepLR(optimizer, milestones=args.steplr_milestones, gamma=args.steplr_gamma)
logging.info('Multi Step Info:'
             '\nmilestones: {}'
             '\ngamma: {}\n'.format(args.steplr_milestones, args.steplr_gamma))

save_manager = SaveManager(modelname=args.model_name, interval=args.checkpoints_interval, max_checkpoints=15, plot_interval=100)
trainer = TrainObjectDetectionConsoleLogger(loss_module=TextBoxLoss(alpha=args.loss_alpha, conf_loss=ConfidenceLoss(neg_factor=args.neg_factor)), model=model, optimizer=optimizer, scheduler=iter_scheduler)

logging.info('Save Info:'
             '\nfilename: {}'
             '\ncheckpoints interval: {}\n'.format(args.model_name, args.checkpoints_interval))

logging.info('Start Training\n\n')

trainer.train_iter(save_manager, args.max_iteration, train_loader, start_iteration=args.start_iteration)
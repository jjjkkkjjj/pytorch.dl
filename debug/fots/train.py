from dl.data.txtdetn import datasets, utils, target_transforms, augmentations
from dl.data import transforms
from dl.models.fots import FOTSRes50, FOTSRes34
from dl.loss.fots import FOTSLoss

from dl.optim.scheduler import IterMultiStepLR
from dl.log import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

import numpy as np

if __name__ == '__main__':
    """
    augmentation = augmentations.Compose(
        []
    )"""

    #augmentation = None
    augmentation = augmentations.Compose([
        augmentations.RandomLongerResize(smin=640, smax=2560),
        augmentations.RandomRotate(fill_rgb=(103.939, 116,779, 123.68), amin=-10, amax=10, same=True),
        augmentations.RandomScaleV(smin=0.8, smax=1.2, keep_aspect=True),
        augmentations.RandomSimpleCrop()
    ])

    ignore = target_transforms.Ignore(strange=True)

    transform = transforms.Compose(
        [transforms.Resize((640, 640)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Text2Number(class_labels=datasets.SynthText_char_labels_without_upper_blank, ignore_nolabel=False),
         target_transforms.ToTensor(textTensor=True)]
    )

    train_dataset = datasets.SynthTextDetectionDataset(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation,
                                                       onlyAlphaNumeric=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              collate_fn=utils.batch_ind_fn,
                              num_workers=4,
                              pin_memory=True)

    model = FOTSRes50(chars=datasets.SynthText_char_labels_without_upper_blank, input_shape=(None, None, 3), feature_height=8).cuda()
    print(model)
    """
    train_iter = iter(train_loader)
    img, targets, texts = next(train_iter)
    targets = [t.cuda() for t in targets]
    model(img.cuda(), targets, texts)
    """
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    iter_sheduler = IterMultiStepLR(optimizer, milestones=np.arange(10000, 107344*10, 10000), gamma=0.94, verbose=True)

    save_manager = SaveManager(modelname='fots', interval=1, max_checkpoints=3, plot_interval=10)

    trainer = TrainTextSpottingConsoleLogger(FOTSLoss(), model, optimizer, iter_sheduler)
    trainer.train_epoch(save_manager, 10, train_loader)

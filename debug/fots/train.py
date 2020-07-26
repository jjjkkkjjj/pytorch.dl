from dl.data.txtdetn import datasets, utils, target_transforms, augmentations
from dl.data import transforms
from dl.models.fots import FOTS

from dl.optim.scheduler import IterStepLR
from dl.log import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD

if __name__ == '__main__':
    """
    augmentation = augmentations.Compose(
        []
    )"""

    #augmentation = augmentations.AugmentationOriginal()
    augmentation = None

    ignore = target_transforms.Ignore(strange=True)

    transform = transforms.Compose(
        [transforms.Resize((600, 800)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Text2Number(class_labels=datasets.ALPHANUMERIC_WITH_BLANK_LABELS),
         target_transforms.ToTensor(textTensor=True)]
    )

    train_dataset = datasets.SynthTextDetectionDataset(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation,
                                                       onlyAlphaNumeric=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              collate_fn=utils.batch_ind_fn,
                              num_workers=4,
                              pin_memory=True)

    model = FOTS(input_shape=(None, None, 3)).cuda()
    print(model)
    train_iter = iter(train_loader)
    img, targets, texts = next(train_iter)
    model(img.cuda(), targets, texts)
    """
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    iter_sheduler = IterStepLR(optimizer, step_size=60000, gamma=0.1, verbose=True)

    save_manager = SaveManager(modelname='MK', interval=1, max_checkpoints=3, plot_interval=10)

    trainer = TrainObjectDetectionConsoleLogger(SSDLoss(), model, optimizer, iter_sheduler)
    trainer.train_epoch(save_manager, 2, train_loader)
    """
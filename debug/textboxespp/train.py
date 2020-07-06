from dl.data.text import datasets, target_transforms, transforms
from dl.data.text.utils import batch_ind_fn_droptexts

from dl.models import TextBoxesPP
from dl.loss.textboxespp import TextBoxLoss
from dl.optim.scheduler import IterStepLR
from dl.log import *

#from torchvision import transforms > not import!!
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

if __name__ == '__main__':

    #augmentation = augmentations.RandomSampled()
    augmentation = None

    ignore = target_transforms.Ignore(strange=True)

    transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Corners2Centroids(),
         #target_transforms.ToQuadrilateral(),
         target_transforms.OneHot(class_nums=datasets.COCOText_class_nums, add_background=True),
         target_transforms.ToTensor()]
    )

    #train_dataset = datasets.COCO2014Text_Dataset(ignore=target_transforms.Ignore(illegible=True), transform=transform, target_transform=target_transform, augmentation=None)
    #train_dataset = datasets.SynthTextDetectionDataset(ignore=None, transform=transform, target_transform=target_transform, augmentation=augmentation)
    #train_dataset = datasets.SynthTextDetectionDataset(ignore=None, transform=transform, target_transform=target_transform, augmentation=augmentation)
    #train_dataset = datasets.ICDAR2015TextDataset(ignore=None, transform=transform, target_transform=target_transform, augmentation=augmentation)
    #train_dataset = datasets.ICDARFocusedSceneTextDataset(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation)
    #train_dataset = datasets.ICDARBornDigitalTextDataset(ignore=ignore, transform=transform, target_transform=target_transform, augmentation=augmentation)

    train_dataset = datasets.Compose((datasets.ICDARFocusedSceneTextDataset, datasets.ICDARBornDigitalTextDataset),
                                     ignore=ignore, transform=transform, target_transform=target_transform,
                                     augmentation=augmentation)

    train_loader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              collate_fn=batch_ind_fn_droptexts,
                              num_workers=4,
                              pin_memory=True)

    model = TextBoxesPP(input_shape=(384, 384, 3)).cuda()
    model.load_vgg_weights()
    print(model)
    #model.load_weights('./weights/model_icdar15.pth')

    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    # iter_sheduler = IterMultiStepLR(optimizer, milestones=(10, 20, 30), gamma=0.1, verbose=True)
    iter_sheduler = IterStepLR(optimizer, step_size=60000, gamma=0.1, verbose=True)

    save_manager = SaveManager(modelname='test', interval=5000, max_checkpoints=3, plot_interval=100)

    trainer = TrainObjectDetectionConsoleLogger(TextBoxLoss(alpha=0.2), model, optimizer, iter_sheduler)
    trainer.train_iter(save_manager, 80000, train_loader)  # , evaluator=VOC2007Evaluator(val_dataset, iteration_interval=10))


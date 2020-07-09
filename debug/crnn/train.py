from dl.data.txtrecog import datasets, target_transforms, transforms
#from dl.data.text.utils import batch_ind_fn_droptexts

from dl.models.crnn import CRNN
from torch.nn import CTCLoss
from dl.optim.scheduler import IterStepLR
from dl.log import *

from torch.utils.data import DataLoader
from torch.optim.adadelta import Adadelta

if __name__ == '__main__':
    augmentation = None

    #ignore = target_transforms.TextDetectionIgnore(difficult=True)

    transform = transforms.Compose(
        [transforms.Resize((100, 32)),
         transforms.Grayscale(last_dims=1),
         transforms.ToTensor(),
         #transforms.Normalize(rgb_means=(0.485, 0.456, 0.406), rgb_stds=(0.229, 0.224, 0.225))
         # normalize 0.5, 0.5?: https://github.com/pytorch/vision/issues/288
         ]
    )
    target_transform = target_transforms.Compose(
        [target_transforms.Text2Number(class_labels=datasets.ALPHANUMERIC_LABELS),
         target_transforms.OneHot(class_nums=datasets.ALPHANUMERIC_NUMBERS, add_nolabel=False),
         target_transforms.ToTensor()
         ]
    )

    train_dataset = datasets.SynthTextRecognitionDataset(transform=transform, target_transform=target_transform, augmentation=augmentation)
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    """
    img, text = train_dataset[0]
    import cv2
    cv2.imshow(''.join(text), img)
    cv2.waitKey()
    """
    model = CRNN(class_labels=datasets.ALPHANUMERIC_LABELS, input_shape=(32, None, 1)).cuda()
    print(model)
    img, text = train_dataset[0]
    p = model(img.unsqueeze(0).cuda())

    optimizer = Adadelta(model.parameters())

    save_manager = SaveManager(modelname='test', interval=5000, max_checkpoints=3, plot_interval=100)

    trainer = TrainObjectDetectionConsoleLogger(CTCLoss(zero_infinity=True), model, optimizer)
    trainer.train_iter(save_manager, 80000, train_loader)  # , evaluator=VOC2007Evaluator(val_dataset, iteration_interval=10))

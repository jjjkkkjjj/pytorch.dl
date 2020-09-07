import torch
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
from ..layers import Flatten
from ..base.model import ModelBase
from ..._utils import _get_model_url
from collections import OrderedDict


class VGGBase(ModelBase):
    """
    :param
        load_model  : path, Bool or None

    """
    def __init__(self, model_name, conv_layers, class_nums=1000):
        super().__init__()
        self.model_name = model_name

        self.conv_layers = conv_layers
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = Flatten()
        self.classifier_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_nums)
        )

        self._init_weights()



    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 1e-2)
                nn.init.constant_(module.bias, 0)

    def load_weights(self, path=None):
        if path is None:
            model_url = _get_model_url(self.model_name)

            model_dir = './weights'
            pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)
            model_state_dict = self.state_dict()

            if len(model_state_dict) != len(pretrained_state_dict):
                raise ValueError('cannot load model due to unsame model architecture')

            # rename
            renamed = []
            # note that model_state_dict and pretrained_state_dict are ordereddict
            for (pre_key, mod_key) in zip(pretrained_state_dict.keys(), model_state_dict.keys()):
                renamed +=[(mod_key, pretrained_state_dict[pre_key])]

            pretrained_state_dict = OrderedDict(renamed)
            self.load_state_dict(pretrained_state_dict)
            # not save because using downloaded weights in dl
            #torch.save(self.state_dict(), os.path.join(model_dir, '{}'.format(model_url.split('/')[-1])))

        else:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.classifier_layers(x)


import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

from ..layers import Conv2d

class SharedConv(nn.Module):
    def __init__(self):
        super().__init__()

        resnet50 = models.resnet50(pretrained=True, progress=True)
        print(resnet50)
        self.conv1 = nn.Sequential(
            resnet50.conv1, resnet50.bn1, resnet50.relu
        )
        self.pool1 = resnet50.maxpool
        self.res2 = resnet50.layer1

        self.res3 = resnet50.layer2
        self.res4 = resnet50.layer3
        self.res5 = resnet50.layer4

        #self.deconv_res5 = Deconv(1024, 512)
        self.deconv_res4 = Deconv(2048, 1024 + 512, shared_channels=1024)
        self.deconv_res3 = Deconv(1024 + 512, 512 + 256, shared_channels=512)
        self.deconv_res2 = Deconv(512 + 256, 256 + 128, shared_channels=256)

        self.convlast = nn.Sequential(
            *Conv2d.relu_one('1', 256 + 128, 64, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )

    def forward(self, x):
        """
        :param x: input img Tensor, shape = (b, c, h, w)
        :return:
        """
        x = self.conv1(x)
        x = self.pool1(x)
        # shape = (b, 256, h/4, w/4)
        x = self.res2(x)
        shared_via_res2 = x.clone()

        # shape = (b, 512, h/8, w/8)
        x = self.res3(x)
        shared_via_res3 = x.clone()

        # shape = (b, 1024, h/16, w/16)
        x = self.res4(x)
        shared_via_res4 = x.clone()

        # shape = (b, 2048, h/32, w/32)
        x = self.res5(x)

        # shape = (b, 1024 + 512, h/16, w/16)
        x = self.deconv_res4(x, shared_via_res4)
        # shape = (b, 512 + 256, h/8, w/8)
        x = self.deconv_res3(x, shared_via_res3)
        # shape = (b, 256 + 128, h/4, w/4)
        x = self.deconv_res2(x, shared_via_res2)

        # shape = (b, 64, h/4, w/4)
        outputs = self.convlast(x)

        return outputs

class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, shared_channels):
        super().__init__()
        conv_out_channels = out_channels - shared_channels
        assert conv_out_channels > 0, "out_channels must be greater than shared_channels"

        self.conv = nn.Sequential(
            *Conv2d.relu_one('1', in_channels, conv_out_channels, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True), # reduce feature channels
        )
        self.shared_channels = shared_channels

    def forward(self, x, shared_x):
        _, c, h, w = shared_x.shape
        assert c == self.shared_channels, "shared_x\'s channels must be {}, but got {}".format(self.shared_channels, shared_x.shape[1])

        x = self.conv(x)
        # bilinear upsampling
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return torch.cat((x, shared_x), dim=1)

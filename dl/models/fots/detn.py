import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F
import math

from ..layers import Conv2d

class SharedConv(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_chennels = out_channels

        resnet50 = models.resnet50(pretrained=True, progress=True)
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
            *Conv2d.relu_one('1', 256 + 128, out_channels, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )

    def forward(self, x):
        """
        :param x: input img Tensor, shape = (b, c, h, w)
        :return: fmaps: output feature maps Tensor, shape = (b, out_channels, h/4, w/4)
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

        # shape = (b, out_channels, h/4, w/4)
        fmaps = self.convlast(x)

        return fmaps

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

class Detector(nn.Module):
    def __init__(self, in_channels, dist_scale=512):
        super().__init__()
        self.in_channels = in_channels
        self.dist_scale = dist_scale

        self.conf_layer = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.distances_layer = nn.Conv2d(in_channels, 4, kernel_size=(1, 1))
        self.angle_layer = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))

    def forward(self, features):
        """
        :param features: feature Tensor from shared conv, shape = (b, in_channels, h/4, w/4)
        :return:
            conf: confidence Tensor, shape = (b, 1, h/4, w/4)
            distances: distances Tensor, shape = (b, 4=(t, l, b, r), h/4, w/4) for each pixel to target rectangle boundaries
            angle: angle Tensor, shape = (b, 1, h/4, w/4)
        """
        conf = self.conf_layer(features)
        conf = torch.sigmoid(conf)

        distances = self.distances_layer(features)
        distances = torch.sigmoid(distances) * self.dist_scale

        angle = self.angle_layer(features)
        # angle range is (-pi/2, pi/2)
        angle = torch.sigmoid(angle) * math.pi / 2

        return conf, distances, angle
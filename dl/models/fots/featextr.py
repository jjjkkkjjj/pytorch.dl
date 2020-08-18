import torch
from torch import nn
from torchvision import models
from torch.nn import functional as F

from ..layers import Conv2d
from .base import FeatureExtractorBase

class Deconv(nn.Module):
    def __init__(self, prev_channels, out_channels, shared_channels):
        super().__init__()
        in_channels = prev_channels + shared_channels

        self.conv = nn.Sequential(
            *Conv2d.relu_one('1', in_channels, out_channels, kernel_size=(1, 1), batch_norm=True, sequential=True), # reduce feature channels by 1x1 kernel
            *Conv2d.relu_one('1', out_channels, out_channels, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )
        self.prev_channels = prev_channels
        self.shared_channels = shared_channels

    def forward(self, x, shared_x):
        _, c, h, w = x.shape
        assert c == self.prev_channels, "previous out_channels must be {}, but got {}".format(self.prev_channels, c)

        _, c, h_shared, w_shared = shared_x.shape
        assert c == self.shared_channels, "shared_x\'s channels must be {}, but got {}".format(self.shared_channels, c)

        # bilinear upsampling
        x = F.interpolate(x, size=(h_shared, w_shared), mode='bilinear', align_corners=True)
        assert shared_x.shape[2:] == x.shape[2:], "height and width must be same, but got shared conv: {} and previous conv: {}".format(shared_x.shape[2:], x.shape[2:])

        # share conv
        x = torch.cat((x, shared_x), dim=1)

        x = self.conv(x)

        return x


class SharedConvRes50(FeatureExtractorBase):
    def __init__(self, out_channels):
        super().__init__(out_channels)

        resnet50 = models.resnet50(pretrained=True, progress=True)
        self.conv1 = nn.Sequential(
            resnet50.conv1, resnet50.bn1, resnet50.relu
        )
        self.pool1 = resnet50.maxpool
        self.res2 = resnet50.layer1

        self.res3 = resnet50.layer2
        self.res4 = resnet50.layer3
        self.res5 = resnet50.layer4

        # Note that deconv args' formula is following;
        # prev_channel = previous out_channels
        # shared_channels = shared_conv's out_channels
        self.deconv_res4 = Deconv(2048, 128, shared_channels=1024)
        self.deconv_res3 = Deconv(128, 64, shared_channels=512)
        self.deconv_res2 = Deconv(64, 32, shared_channels=256)

        self.convlast = nn.Sequential(
            *Conv2d.relu_one('1', 32, out_channels, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )

    def forward(self, x):
        """
        :param x: input img Tensor, shape = (b, c, h, w)
        :return: fmaps: output feature maps Tensor, shape = (b, out_channels, h/4, w/4)
        """
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.res2(x)
        shared_via_res2 = x.clone() # shape = (b, 256, h/4, w/4)

        x = self.res3(x)
        shared_via_res3 = x.clone() # shape = (b, 512, h/8, w/8)

        x = self.res4(x)
        shared_via_res4 = x.clone() # shape = (b, 1024, h/16, w/16)

        x = self.res5(x) # shape = (b, 2048, h/32, w/32)
        x = self.deconv_res4(x, shared_via_res4) # shape = (b, 128, h/16, w/16)
        x = self.deconv_res3(x, shared_via_res3) # shape = (b, 64, h/8, w/8)
        x = self.deconv_res2(x, shared_via_res2) # shape = (b, 32, h/4, w/4)

        fmaps = self.convlast(x) # shape = (b, out_channels, h/4, w/4)

        return fmaps


class SharedConvRes34(FeatureExtractorBase):
    def __init__(self, out_channels):
        super().__init__(out_channels)

        resnet34 = models.resnet34(pretrained=True, progress=True)
        self.conv1 = nn.Sequential(
            resnet34.conv1, resnet34.bn1, resnet34.relu
        )
        #self.pool1 = resnet34.maxpool
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.res2 = resnet34.layer1

        self.res3 = resnet34.layer2
        self.res4 = resnet34.layer3
        self.res5 = resnet34.layer4

        self.center = nn.Sequential(
            *Conv2d.relu_one('1', 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=1, batch_norm=True, sequential=True),
            *Conv2d.relu_one('2', 512, 1024, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )

        # Note that deconv args' formula is following;
        # prev_channel = previous out_channels
        # shared_channels = shared_conv's out_channels
        self.deconv_res5 = Deconv(1024, 1024, shared_channels=512)
        self.deconv_res4 = Deconv(1024, 512, shared_channels=256)
        self.deconv_res3 = Deconv(512, 256, shared_channels=128)
        self.deconv_res2 = Deconv(256, 128, shared_channels=64)

        self.convlast = nn.Sequential(
            *Conv2d.relu_one('1', 128, out_channels, kernel_size=(3, 3), padding=1, batch_norm=True, sequential=True)
        )

    def forward(self, x):
        """
        :param x: input img Tensor, shape = (b, c, h, w)
        :return: fmaps: output feature maps Tensor, shape = (b, out_channels, h/4, w/4)
        """
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.res2(x)
        shared_via_res2 = x.clone() # shape = (b, 64, h/4, w/4)

        x = self.res3(x)
        shared_via_res3 = x.clone() # shape = (b, 128, h/8, w/8)

        x = self.res4(x)
        shared_via_res4 = x.clone() # shape = (b, 256, h/16, w/16)

        x = self.res5(x)
        shared_via_res5 = x.clone()  # shape = (b, 512, h/32, w/32)

        x = self.center(x) # shape = (b, 1024, h/64, w/64)

        x = self.deconv_res5(x, shared_via_res5) # shape = (b, 1024, h/32, w/32)
        x = self.deconv_res4(x, shared_via_res4) # shape = (b, 512, h/16, w/16)
        x = self.deconv_res3(x, shared_via_res3) # shape = (b, 256, h/8, w/8)
        x = self.deconv_res2(x, shared_via_res2)  # shape = (b, 128, h/4, w/4)

        fmaps = self.convlast(x) # shape = (b, out_channels, h/4, w/4)

        return fmaps

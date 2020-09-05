import torch
from torch import nn
import math

from .base import DetectorBase

class Detector(DetectorBase):
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
        :returns:
            pred_confs: confidence Tensor, shape = (b, h/4, w/4, 1)
            pred_rboxes: predicted Tensor, shape = (b, h/4, w/4, 5=(conf, t, r, b, l, angle))
                distances: distances Tensor, shape = (b, h/4, w/4, 4=(t, r, b, l)) for each pixel to target rectangle boundaries
                angle: angle Tensor, shape = (b, h/4, w/4, 1)
        """
        # shape = (b, 1, h/4, w/4)
        conf = self.conf_layer(features)
        conf = torch.sigmoid(conf)

        # shape = (b, 4=(t, r, b, l), h/4, w/4)
        distances = self.distances_layer(features)
        distances = torch.sigmoid(distances) * self.dist_scale

        # shape = (b, 1, h/4, w/4)
        angle = self.angle_layer(features)
        # angle range is (-pi/4, pi/4)
        angle = (torch.sigmoid(angle) - 0.5) * math.pi / 2

        return conf.permute((0, 2, 3, 1)).contiguous(), torch.cat((distances, angle), dim=1).permute((0, 2, 3, 1)).contiguous()
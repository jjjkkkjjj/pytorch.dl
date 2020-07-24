from torch.nn import Module

class RoIRotate(Module):
    def __init__(self):
        super().__init__()

    def forward(self, fmaps, gt_boxes):
        pass
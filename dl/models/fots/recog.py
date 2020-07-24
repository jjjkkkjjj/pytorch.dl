from torch.nn import Module

class RoIRotate(Module):
    def __init__(self, height=8):
        super().__init__()

        self.height = height

    def forward(self, x):
        pass
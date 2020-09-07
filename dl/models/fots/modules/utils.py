import torch

from ....data.utils.quads import quads2rboxes_numpy

def matching_strategy(labels, w, h, device, scale=0.3):
    """
    :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
    :param w: int
    :param h: int
    :param device: device
    :param scale: int
    :return:
        pos_indicator: bool Tensor, shape = (b, h, w)
        true_rboxes: Tensor, shape = (b, h, w, 5=(4=(t,r,b,l)+1=angle)))
    """
    pos_indicator = []
    true_rboxes = []

    for b, label in enumerate(labels):
        quads = label[:, 4:12].cpu().numpy()

        pos, rbox = quads2rboxes_numpy(quads, w, h, shrink_scale=scale)

        pos_indicator += [torch.from_numpy(pos).unsqueeze(0)]
        true_rboxes += [torch.from_numpy(rbox).unsqueeze(0)]

    return torch.cat(pos_indicator, dim=0).bool().to(device), torch.cat(true_rboxes, dim=0).to(device)
import torch

def batch_ind_fn(batch):
    """
    :param batch:
    :return:
        imgs: Tensor, shape = (b, c, h, w)
        targets: list of Tensor, whose shape = (text length)
    """
    imgs, texts = list(zip(*batch))

    return torch.stack(imgs), texts
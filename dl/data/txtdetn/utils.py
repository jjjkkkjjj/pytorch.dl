import torch

def batch_ind_fn_droptexts(batch):
    """
    :param batch:
    :return:
        imgs: Tensor, shape = (b, c, h, w)
        targets: list of Tensor, whose shape = (object box num, 4 + class num) including background
    """
    imgs, gts, texts = list(zip(*batch))

    return torch.stack(imgs), gts

def batch_ind_fn(batch):
    """
    :param batch:
    :return:
        imgs: Tensor, shape = (b, c, h, w)
        targets: list of Tensor, Tensor or ndarray of bboxes, quads and labels [box, quads, label]
                = [xmin, ymin, xmamx, ymax, x1, y1, x2, y2,..., label index(or one-hotted label)]
                or
                = [cx, cy, w, h, x1, y1, x2, y2,..., label index(or relu_one-hotted label)]
        texts: list of str, if it's illegal, str = ''
    """
    imgs, targets, texts = list(zip(*batch))

    return torch.stack(imgs), targets, texts
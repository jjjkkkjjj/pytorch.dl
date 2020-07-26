from PIL import Image, ImageDraw
import numpy as np
import torch, cv2

def matching_strategy(fmaps, labels):
    """
    :param fmaps: feature maps Tensor, shape = (b, out_channels, h/4, w/4)
    :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
    :return:
        pos_indicator: bool Tensor, shape = (b, h/4, w/4)
        true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
    """
    pos_indicator = []
    true_locs = []

    _, _, h, w = fmaps.shape
    device = fmaps.device

    for b, label in enumerate(labels):
        mask = np.zeros((h, w), dtype=np.bool)
        t_angles = []

        for t in range(label.shape[0]):
            quads = label[t, 4:12].clone()
            quads[::2] *= w
            quads[1::2] *= h

            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(quads.cpu().numpy(), outline=255, fill=255)
            #img.show()
            mask = np.logical_or(mask, np.array(img, dtype=np.bool))

            _, _, t_angle = cv2.minAreaRect(quads.cpu().numpy().reshape(4, 2).astype(np.float32))
            t_angles += [t_angle]

        pos_indicator += [mask]
        true_locs += [torch.cat((label[:, :12], torch.tensor(t_angles, dtype=torch.float, device=device).unsqueeze(-1)), dim=-1)]

    return torch.tensor(pos_indicator, device=device).bool(), true_locs
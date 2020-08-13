import torch
from torch.nn import Module
from torch.nn import functional as F
import numpy as np
import cv2

class RoIRotate(Module):
    def __init__(self, height=8):
        super().__init__()
        self.height = height

    def forward(self, fmaps, pred_locs, true_locs):
        """
        :param fmaps: feature maps Tensor, shape = (b, c, h/4, w/4)
        :param pred_locs: predicted Tensor, shape = (b, h/4, w/4, 5=(conf, t, l, b, r, angle))
        :param true_locs: list(b) of tensor, shape = (text number, 4=(xmin, ymin, xmax, ymax)+8=(x1, y1,...)+1=angle))
        :return:
            ret_rotated_features: list(b) of Tensor, shape = (text nums, c, height=8, non-fixed width)
            ret_true_angles: list(b) of Tensor, shape = (text nums,)
        """
        device = fmaps.device
        batch_nums, c, h, w = fmaps.shape

        distances, angles = pred_locs[:, 1:5], pred_locs[:, 5:]

        ret_rotated_features = []
        for b in range(batch_nums):
            images = []
            widths = []
            matrices = []

            bboxes, quads = true_locs[b][:, :4].cpu().numpy(), true_locs[b][:, 4:12].cpu().numpy()
            bboxes[:, ::2] *= w
            bboxes[:, 1::2] *= h
            quads[:, ::2] *= w
            quads[:, 1::2] *= h

            textnums = bboxes.shape[0]
            for t in range(textnums):
                xmin, ymin, xmax, ymax = bboxes[t]

                aspect_ratio = (xmax - xmin) / float(ymax - ymin)
                width = np.clip(int(aspect_ratio * self.height), 1, w)

                src = np.float32([[xmin, ymin], [xmin, ymax], [xmax, ymin]])
                dst = np.float32([[0, 0], [0, self.height], [width, 0]])

                # https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/17
                affine_matrix = cv2.getAffineTransform(dst, src)

                theta = _affine2theta(affine_matrix, width, self.height, device)

                images += [fmaps[b]]
                widths += [width]
                matrices += [theta]

            images = torch.stack(images) # shape = (text num, c, h/4, w/4)
            matrices = torch.stack(matrices) # shape = (text num, 2, 3)
            grid = F.affine_grid(matrices, images.size(), align_corners=True)
            rotated_features = F.grid_sample(images, grid, mode='bilinear', align_corners=True) # shape = (text num, c, h/4, w/4)

            max_width = np.max(widths)

            pad_images = torch.zeros((textnums, c, self.height, max_width), device=device)
            for t in range(textnums):
                pad_images[t, :, :self.height, :widths[t]] = rotated_features[t, :, :self.height, :widths[t]]

            ret_rotated_features += [pad_images]

        return ret_rotated_features

        """
        # shape = (b, h/4, w/4, 1)
        top, left = distances[:, 0].unsqueeze(-1), distances[:, 1].unsqueeze(-1)
        bottom, right = distances[:, 2].unsqueeze(-1), distances[:, 3].unsqueeze(-1)
        theta = angle[:, 0].unsqueeze(-1)
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
        # shape = (1, h/4, w/4, 1)
        x, y = x.unsqueeze(0).unsqueeze(-1), y.unsqueeze(0).unsqueeze(-1)

        tx = left * cos - top * sin - x
        ty = top * cos + left * sin - y

        # shape = (b, h/4, w/4, 1)
        scale = self.height / (top + bottom)
        width = scale * (left + right)

        # shape = (b, h/4, w/4, 1, 3)
        line1 = torch.cat([cos, -sin, tx*cos-ty*sin], dim=-1).unsqueeze(-2)
        line2 = torch.cat([sin, cos, tx*sin+ty*cos], dim=-1).unsqueeze(-2)
        line3 = torch.cat([torch.zeros((b, h, w, 1), device=device), torch.zeros((b, h, w, 1), device=device), torch.ones((b, h, w, 1), device=device) / scale], dim=-1).unsqueeze(-2)

        affine_matrix = scale.unsqueeze(-1) * torch.cat((line1, line2, line3), dim=-2)
        """

def _affine2theta(M, w, h, device):
    # convert affine_matrix into theta
    # formula is;
    # (x'_s, y'_s) = ((x_s - w/2)/(w/2), (y_s - h/2)/(h/2)) # to normalize
    #              = (x_s/(w/2)-1, y_s/(h/2))
    # where (x'_s, y'_s) is normalized source points
    # Therefore, affine matrix is;
    # M' = ( 2/w_d,     0,   -1)^-1        ( 2/w_s,     0,   -1)
    #      (     0, 2/h_d,   -1)    * M *  (     0, 2/h_s,   -1)
    #      (     0,     0,    1)           (     0,     0,    1)

    return torch.tensor(((M[0, 0], M[0, 1]*h/w, M[0, 2]*2/w + M[0, 0] + M[0, 1] - 1),
                         (M[1, 0]*w/h, M[1, 1], M[1, 2]*2/h + M[1, 0] + M[1, 1] - 1)), device=device)
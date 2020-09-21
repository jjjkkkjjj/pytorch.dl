import torch
from torch.nn import Module
from torch.nn import functional as F
import numpy as np
import cv2, math

class RoIRotate(Module):
    def __init__(self, height=8, _debug=False):
        super().__init__()
        self.height = height
        self._debug = _debug

    def forward(self, fmaps, quads):
        """
        :param fmaps: feature maps Tensor, shape = (b, c, h/4, w/4)
        :param quads: list(b) of Tensor, shape = (text number, 8=(x1, y1,...)))
        :return:
            ret_rotated_features: list(b) of Tensor, shape = (text nums, c, height=8, non-fixed width)
            ret_true_angles: list(b) of Tensor, shape = (text nums,)
        """
        device = fmaps.device
        batch_nums, c, h, w = fmaps.shape

        ret_rotated_features = []
        for b in range(batch_nums):
            images = []
            widths = []
            matrices = []

            _quads = quads[b].cpu().numpy().copy()
            _quads[:, ::2] *= w
            _quads[:, 1::2] *= h

            textnums = _quads.shape[0]
            for t in range(textnums):
                img = fmaps[b]
                quad = _quads[t].reshape((4, 2))
                tl, tr, br, bl = quad

                # minAreaRect returns center_point, size, angle(deg)
                _, size, _ = cv2.minAreaRect(quad)
                box_w, box_h = size
                """
            bboxes, quads = true_locs[b][:, :4].cpu().numpy(), true_locs[b][:, 4:12].cpu().numpy()
            bboxes[:, ::2] *= w
            bboxes[:, 1::2] *= h
            quads[:, ::2] *= w
            quads[:, 1::2] *= h

            textnums = bboxes.shape[0]
            for t in range(textnums):
                img = fmaps[b]
                quad = quads[t].reshape((4, 2))
                tl, tr, bl, br = quad

                # minAreaRect returns center_point, size, angle(deg)
                _, size, _ = cv2.minAreaRect(quad)
                box_w, box_h = size
                """
                # minAreaRect calculates angle for longer side
                # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned/21427814#21427814
                if box_w <= box_h:
                    box_w, box_h = box_h, box_w
                if box_w == 0 or box_h == 0:
                    print(quad)
                # ceil is for avoiding to box_w = zero
                box_w = math.ceil(self.height * box_w / box_h)
                box_w = min(w, box_w)

                src = np.float32([tl, tr, bl])
                dst = np.float32([[0, 0], [box_w, 0], [0, self.height]])

                # https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/17
                affine_matrix = cv2.getAffineTransform(src, dst)

                # for debug
                if self._debug:
                    img = cv2.warpAffine((fmaps[b]*255).permute(1,2,0).cpu().numpy().astype(np.uint8), affine_matrix, (int(box_w), int(self.height)))

                theta = _affine2theta(affine_matrix, w, h, device)

                images += [img]
                widths += [box_w]
                matrices += [theta]
            # for debug
            if self._debug:
                ret_rotated_features += [images]
                continue

            images = torch.stack(images) # shape = (text num, c, h/4, w/4)
            matrices = torch.stack(matrices) # shape = (text num, 2, 3)
            grid = F.affine_grid(matrices, images.size(), align_corners=True)
            rotated_features = F.grid_sample(images, grid, mode='bilinear', align_corners=True) # shape = (text num, c, h/4, w/4)
            max_width = np.max(widths)

            pad_images = torch.zeros((textnums, c, self.height, max_width), device=device)
            for t in range(textnums):
                pad_images[t, :, :self.height, :widths[t]] = rotated_features[t, :, :self.height, :widths[t]]

            ret_rotated_features += [pad_images]

        if self._debug:
            return ret_rotated_features

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

    """
    M = np.vstack([M, [0, 0, 1]])
    M = np.linalg.inv(M)

    theta00 = M[0, 0]
    theta01 = M[0, 1]*h/w
    theta02 = M[0, 2]*2/w + theta00 + theta01 - 1

    theta10 = M[1, 0]*w/h
    theta11 = M[1, 1]
    theta12 = M[1, 2]*2/h + theta10 + theta11 - 1

    return torch.tensor(((theta00, theta01, theta02),
                         (theta10, theta11, theta12)), device=device)
    """
    def norm_mat(W, H):
        return np.array(((2.0/W,     0,   -1),
                         (    0, 2.0/H,   -1),
                         (    0,     0,    1)))

    M = np.vstack([M, [0, 0, 1]])
    M = np.linalg.inv(M)

    theta = norm_mat(w, h) @ M @ np.linalg.inv(norm_mat(w, h))
    return torch.from_numpy(theta[:2, :]).to(dtype=torch.float, device=device)

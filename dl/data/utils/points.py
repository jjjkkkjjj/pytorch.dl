import numpy as np
import torch

def pts2homogeneousPts_numpy(pts):
    """
    :param pts: ndarray, shape = (points set num, points num, 2)
    :return pts_homogeneous: ndarray, shape = (points set num, 3=(x,y,1), points num)
    """
    assert pts.shape[-1] == 2, "must be 2d-points"
    assert pts.ndim == 3, "must have 3d"

    set_num, points_num = pts.shape[:2]
    return np.concatenate((np.swapaxes(pts.reshape((set_num, points_num, 2)), -2, -1),
                           np.ones((set_num, 1, points_num))), axis=1)

def apply_affine(affine, src_size, dst_size, *pts):
    """
    :param affine: ndarray, shape = (2, 3)
    :param src_size: tuple = (w, h)
    :param dst_size: tuple = (w, h)
    :param pts: tuple of ndarray, shape = (points set num, points num, 2)
    :return pts_affined: tuple of ndarray, shape = (points set num, points num, 2=(x,y))
    """
    assert len(pts) > 0, "must contain more than one source points"
    R = np.concatenate((affine, np.array([[0, 0, 1]])), axis=0)
    ret_pts = []
    for _pts in pts:
        # _pts: shape = (points set num, points num, 2)
        # reconstruct original coordinates
        _pts[..., 0] *= src_size[0]
        _pts[..., 1] *= src_size[1]

        # shape = (points set num, 3=(x,y,1), points num)
        pts_hom = pts2homogeneousPts_numpy(_pts)
        affined_pts = R @ pts_hom
        # shape = (points set num, points num, 2=(x,y))
        affined_pts = np.swapaxes(affined_pts[..., :2, :], -2, -1).astype(np.float32)

        # to percent
        affined_pts[..., 0] /= dst_size[0]
        affined_pts[..., 1] /= dst_size[1]
        ret_pts += [affined_pts]

    if len(pts) >= 2:
        return tuple(ret_pts)
    else:
        return ret_pts[0]
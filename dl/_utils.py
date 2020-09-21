import os, cv2
import torch
from torch import nn
import numpy as np

def weights_path(_file_, _root_num, dirname):
    basepath = os.path.dirname(_file_)
    backs = [".."]*_root_num
    model_dir = os.path.abspath(os.path.join(basepath, *backs, dirname))
    return model_dir


def _check_ins(name, val, cls, allow_none=False, default=None):
    if allow_none and val is None:
        return default

    if not isinstance(val, cls):
        err = 'Argument \'{}\' must be {}, but got {}'
        if isinstance(cls, (tuple, list)):
            types = [c.__name__ for c in cls]
            err = err.format(name, types, type(val).__name__)
            raise ValueError(err)
        else:
            err = err.format(name, cls.__name__, type(val).__name__)
            raise ValueError(err)
    return val

def _check_retval(funcname, val, cls):
    if not isinstance(val, cls):
        err = '\'{}\' must return {}, but got {}'
        if isinstance(cls, (tuple, list)):
            types = [c.__name__ for c in cls]
            err = err.format(funcname, types, type(val).__name__)
            raise ValueError(err)
        else:
            err = err.format(funcname, cls.__name__, type(val).__name__)
            raise ValueError(err)
    return val


def _check_norm(name, val):
    if isinstance(val, (float, int)):
        val = torch.tensor([float(val)], requires_grad=False)
    elif isinstance(val, (list, tuple)):
        val = torch.tensor(val, requires_grad=False).float()
    elif not isinstance(val, torch.Tensor):
        raise ValueError('{} must be int, float, list, tuple, Tensor, but got {}'.format(name, type(val).__name__))

    return val

def _initialize_xavier_uniform(layers):
    from .models.layers import ConvRelu

    for module in layers.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, ConvRelu):
            nn.init.xavier_uniform_(module.conv.weight)
            if module.conv.bias is not None:
                nn.init.constant_(module.conv.bias, 0)

def _get_model_url(name):
    model_urls = {
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }

    return model_urls[name]

def _check_image(image, device, size=None):
    """
    :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
    :param device: torch.device
    :param size: None or tuple, if None is passed, check will not be done
                 Note that size = (w, h)
    :return:
        img: Tensor, shape = (b, c, h, w)
        orig_imgs: list of Tensor, shape = (c, h, w) these images may be used for visualization
    """
    orig_imgs = []

    def __check(_tim, _cim, cfirst):
        """
        Note that 2d or 3d image is resizable
        :param _tim: tensor, shape = (h, w, ?) or (?, h, w)
        :param _cim: ndarray, shape = (h, w, ?) or (?, h, w)
        :return:
            tims: tensor, shape = (c, h, w)
            cims: ndarray, shape = (h, w, c)
        """

        #### check size of tensor ####
        if size:
            h, w = _tim.shape[-2:] if cfirst else _tim.shape[:2]
            wcond = size[0] if size[0] is not None else w
            hcond = size[1] if size[1] is not None else h
            if not (h == hcond and w == wcond):
                # do resize
                if cfirst and _cim.ndim == 3:
                    # note that _cim's shape must be (c, h, w)
                    _cim = _cim.transpose((1, 2, 0))
                # _cim's shape = (h, w, ?)
                resized_cim = cv2.resize(_cim, (wcond, hcond))
                return __check(torch.tensor(resized_cim, requires_grad=False), _cim, cfirst=False)

        #### check tensor ####
        assert isinstance(_tim, torch.Tensor)
        if _tim.ndim == 2:
            tim = _tim.unsqueeze(2)
        elif _tim.ndim == 3:
            tim = _tim
        else:
            raise ValueError('Invalid image found. image must be 2d or 3d, but got {}'.format(_tim.ndim))

        if not cfirst:
            # note that tim's shape must be (h, w, c)
            tim = tim.permute((2, 0, 1))

        #### check cvimg ####
        assert isinstance(_cim, np.ndarray)
        if _cim.ndim == 2:
            cim = np.broadcast_to(np.expand_dims(_cim, 2), (_cim.shape[0], _cim.shape[1], 3)).copy()
        elif _cim.ndim == 3:
            cim = _cim
        else:
            raise ValueError('Invalid image found. image must be 2d or 3d, but got {}'.format(_cim.ndim))

        if cfirst:
            # note that cim's shape must be (c, h, w)
            cim = cim.transpose((1, 2, 0))

        return tim, cim


    if isinstance(image, (list, tuple)):
        img = []
        for im in image:
            if isinstance(im, np.ndarray):
                tim = torch.tensor(im, requires_grad=False)
                # im and tim's shape = (h, w, ?)
                tim, cim = __check(tim, im, cfirst=False)

            elif isinstance(im, torch.Tensor):
                cim = im.cpu().numpy()
                # im and tim's shape = (?, h, w)
                tim, cim = __check(im, cim, cfirst=True)
            else:
                raise ValueError('Invalid image type. list or tuple\'s element must be ndarray, but got \'{}\''.format(type(im).__name__))

            img += [tim]
            orig_imgs += [cim]

        # (b, c, h, w)
        img = torch.stack(img)

    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            tim, cim = __check(torch.tensor(image, requires_grad=False), image, cfirst=False)
            img = tim.unsqueeze(0)
            orig_imgs += [cim]
        elif image.ndim == 3:
            tim, cim = __check(torch.tensor(image, requires_grad=False), image, cfirst=False)
            img = tim.unsqueeze(0)
            orig_imgs += [cim]
        elif image.ndim == 4:
            img = []
            for i in range(image.shape[0]):
                tim, cim = __check(torch.tensor(image[i], requires_grad=False), image[i], cfirst=False)
                img += [tim]
                orig_imgs += [cim]
            img = torch.stack(img)
        else:
            raise ValueError('Invalid image found. image must be from 2d to 4d, but got {}'.format(image.ndim))

    elif isinstance(image, torch.Tensor):
        if image.ndim == 2:
            tim, cim = __check(image, image.cpu().numpy(), cfirst=True)
            img = tim.unsqueeze(0)
            orig_imgs += [cim]
        elif image.ndim == 3:
            tim, cim = __check(image, image.cpu().numpy(), cfirst=True)
            img = tim.unsqueeze(0)
            orig_imgs += [cim]
        elif image.ndim == 4:
            img = []
            for i in range(image.shape[0]):
                tim, cim = __check(image[i], image[i].cpu().numpy(), cfirst=True)
                img += [tim]
                orig_imgs += [cim]
            img = torch.stack(img)
        else:
            raise ValueError('Invalid image found. image must be from 2d to 4d, but got {}'.format(image.ndim))

    else:
        raise ValueError('Invalid image type. list or tuple\'s element must be'
                         '\'list\', \'tuple\', \'ndarray\' or \'Tensor\', but got \'{}\''.format(type(image).__name__))

    assert img.ndim == 4, "may forget checking..."

    return img.to(device), orig_imgs


def _check_shape(desired_shape, input_shape):
    """
    Note that desired_shape is allowed to have None, which means whatever input size is ok
    :param desired_shape: array-like
    :param input_shape: array-like
    :return:
    """
    if len(desired_shape) != len(input_shape):
        raise ValueError("shape dim was not same, got {} and {}".format(len(desired_shape), len(input_shape)))

    for i, (des_d, inp_d) in enumerate(zip(desired_shape, input_shape)):
        if des_d is None:
            continue
        if des_d != inp_d:
            raise ValueError('dim:{} is invalid size, desired one: {}, but got {}'.format(i, des_d, inp_d))

def _get_normed_and_origin_img(img, orig_imgs, rgb_means, rgb_stds, toNorm, device):
    """
    :param img: Tensor, shape = (b, c, h, w)
    :param orig_imgs: list of ndarray, shape = (h, w, c)
    :param rgb_means: tuple or float
    :param rgb_stds: tuple or float
    :param toNorm: Bool
    :param device: torch.device
    :return:
        normed_img: Tensor, shape = (b, c, h, w)
        orig_img: Tensor, shape = (b, c, h, w). Order is rgb
    """
    rgb_means = _check_norm('rgb_means', rgb_means)
    rgb_stds = _check_norm('rgb_stds', rgb_stds)

    img = img.to(device)

    if toNorm:
        # shape = (1, 3, 1, 1)
        rgb_means = rgb_means.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        rgb_stds = rgb_stds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        normed_img = (img / 255. - rgb_means) / rgb_stds
        orig_imgs = orig_imgs
    else:
        normed_img = img

        # shape = (1, 1, 3)
        rgb_means = rgb_means.unsqueeze(0).unsqueeze(0).cpu().numpy()
        rgb_stds = rgb_stds.unsqueeze(0).unsqueeze(0).cpu().numpy()
        orig_imgs = [oim * rgb_stds + rgb_means for oim in orig_imgs]

    return normed_img, orig_imgs
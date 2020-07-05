from torch import nn
import torch
import abc

class ModelBase(nn.Module):
    @property
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))
        return next(iter(devices))

    def load_weights(self, path):
        """
        :param path: str
        :return:
        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def init_weights(self):
        raise NotImplementedError()

class ImageRecognitionBase(ModelBase):
    def __init__(self, class_labels, input_shape):
        """
        :param class_labels: int, class number
        :param input_shape: tuple, 3d and (height, width, channel)
        """
        super().__init__()

        self._class_labels = class_labels
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self._input_shape = input_shape

    @property
    def input_height(self):
        return self._input_shape[0]
    @property
    def input_width(self):
        return self._input_shape[1]
    @property
    def input_channel(self):
        return self._input_shape[2]

    @property
    def class_labels(self):
        return self._class_labels
    @property
    def class_nums(self):
        return len(self._class_labels)


    # @abc.abstractmethod
    def learn(self, x, targets):
        pass

    # @abc.abstractmethod
    def infer(self, image, visualize=False, **kwargs):
        """
        :param image:
        :param visualize:
        :param kwargs:
        :return:
            infers: Tensor, shape = (box_num, 5=(conf, cx, cy, w, h))
            Note: if there is no boxes, all of infers' elements are -1, which means (-1, -1, -1, -1, -1)
            visualized_images: list of ndarray, if visualize=True
        """
        pass

class ObjectDetectionModelBase(ImageRecognitionBase):
    @property
    def class_nums_with_background(self):
        return self.class_nums + 1





from torchvision.models.utils import load_state_dict_from_url
import logging, os

from ..._utils import (
    _check_ins, _initialize_xavier_uniform, _check_shape,
    _check_image, _get_normed_and_origin_img, _get_model_url
)
from dl.models.ssd.modules.codec import *
from dl.models.ssd.modules.dbox import *
from .modules.predict import *
from ..layers import *
from .modules.inference import *
from ..base.model import ObjectDetectionModelBase
from ...data.utils.converter import toVisualizeRectLabelRGBimg

class SSDTrainConfig(object):
    def __init__(self, **kwargs):
        self._class_labels = _check_ins('class_labels', kwargs.get('class_labels'), (tuple, list))

        input_shape = kwargs.get('input_shape')
        assert len(input_shape) == 3, "input dimension must be 3"
        assert input_shape[0] == input_shape[1], "input must be square size"
        self.input_shape = input_shape

        self.batch_norm = _check_ins('batch_norm', kwargs.get('batch_norm'), bool)

        self.aspect_ratios = _check_ins('aspect_ratios', kwargs.get('aspect_ratios'), (tuple, list))
        self.classifier_source_names = _check_ins('classifier_source_names', kwargs.get('classifier_source_names'), (tuple, list))
        self.addon_source_names = _check_ins('addon_source_names', kwargs.get('addon_source_names'), (tuple, list))

        self.codec_means = _check_ins('codec_means', kwargs.get('codec_means'), (tuple, list, float, int))
        self.codec_stds = _check_ins('codec_stds', kwargs.get('codec_stds'), (tuple, list, float, int))

        self.rgb_means = _check_ins('rgb_means', kwargs.get('rgb_means', (0.485, 0.456, 0.406)), (tuple, list, float, int))
        self.rgb_stds = _check_ins('rgb_stds', kwargs.get('rgb_stds', (0.229, 0.224, 0.225)), (tuple, list, float, int))

    @property
    def class_labels(self):
        return self._class_labels
    @property
    def class_nums(self):
        return len(self._class_labels)

    @property
    def input_height(self):
        return self.input_shape[0]
    @property
    def input_width(self):
        return self.input_shape[1]
    @property
    def input_channel(self):
        return self.input_shape[2]

class SSDValConfig(object):
    def __init__(self, **kwargs):
        self.val_conf_threshold = _check_ins('val_conf_threshold', kwargs.get('val_conf_threshold', 0.01), float)
        self.vis_conf_threshold = _check_ins('vis_conf_threshold', kwargs.get('vis_conf_threshold', 0.6), float)
        self.iou_threshold = _check_ins('iou_threshold', kwargs.get('iou_threshold', 0.45), float)
        self.topk = _check_ins('topk', kwargs.get('topk', 200), int)

class SSDBase(ObjectDetectionModelBase):
    defaultBox: DefaultBoxBase
    inferenceBox: InferenceBox
    _train_config: SSDTrainConfig
    _val_config: SSDValConfig


    feature_layers: nn.ModuleDict
    localization_layers: nn.ModuleDict
    confidence_layers: nn.ModuleDict
    addon_layers: nn.ModuleDict

    def __init__(self, train_config, val_config, defaultBox,
                 codec=None, predictor=None, inferenceBox=None, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValConfig
        :param defaultBox: instance inheriting DefaultBoxBase
        :param codec: SSDCodec, if it's None, use default SSDCodec
        :param predictor: Predictor, if it's None, use default Predictor
        :param inferenceBox: InferenceBox, if it's None, use default InferenceBox
        """
        self._train_config = _check_ins('train_config', train_config, SSDTrainConfig)
        self._val_config = _check_ins('val_config', val_config, SSDValConfig)
        super().__init__(train_config.class_labels, train_config.input_shape)

        self.codec = _check_ins('codec', codec, CodecBase, allow_none=True,
                                default=SSDCodec(norm_means=self.codec_means, norm_stds=self.codec_stds))
        self.defaultBox = _check_ins('defaultBox', defaultBox, DefaultBoxBase)

        self.predictor = _check_ins('predictor', predictor, PredictorBase, allow_none=True,
                                    default=Predictor(self.class_nums_with_background))

        self.inferenceBox = _check_ins('inferenceBox', inferenceBox, InferenceBoxBase, allow_none=True,
                                       default=InferenceBox(class_nums_with_background=self.class_nums_with_background,
                                                            filter_func=non_maximum_suppression, val_config=val_config))

        self.build(**build_kwargs)

    @property
    def isBuilt(self):
        return hasattr(self, 'feature_layers') and\
               hasattr(self, 'localization_layers') and\
               hasattr(self, 'confidence_layers')

    ### build ###
    @abc.abstractmethod
    def build_feature(self, **kwargs):
        pass
    @abc.abstractmethod
    def build_addon(self, **kwargs):
        pass
    @abc.abstractmethod
    def build_classifier(self, **kwargs):
        pass

    ### codec ###
    @property
    def encoder(self):
        return self.codec.encoder
    @property
    def decoder(self):
        return self.codec.decoder

    ### default box ###
    @property
    def dboxes(self):
        return self.defaultBox.dboxes
    @property
    def total_dboxes_num(self):
        return self.defaultBox.total_dboxes_nums

    ### train_config ###
    @property
    def class_labels(self):
        return self._train_config.class_labels
    @property
    def class_nums(self):
        return self._train_config.class_nums

    @property
    def batch_norm(self):
        return self._train_config.batch_norm

    @property
    def aspect_ratios(self):
        return self._train_config.aspect_ratios
    @property
    def classifier_source_names(self):
        return self._train_config.classifier_source_names
    @property
    def addon_source_names(self):
        return self._train_config.addon_source_names
    @property
    def codec_means(self):
        return self._train_config.codec_means
    @property
    def codec_stds(self):
        return self._train_config.codec_stds
    @property
    def rgb_means(self):
        return self._train_config.rgb_means
    @property
    def rgb_stds(self):
        return self._train_config.rgb_stds
    @property
    def val_conf_threshold(self):
        return self._val_config.val_conf_threshold
    @property
    def vis_conf_threshold(self):
        return self._val_config.vis_conf_threshold
    @property
    def iou_threshold(self):
        return self._val_config.iou_threshold
    @property
    def topk(self):
        return self._val_config.topk

    # device management
    def to(self, *args, **kwargs):
        self.defaultBox.dboxes = self.dboxes.to(*args, **kwargs)

        self.codec = self.codec.to(*args, **kwargs)

        self.inferenceBox.device = self.dboxes.device

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        self.defaultBox.dboxes = self.dboxes.cuda(device)

        self.codec = self.codec.cuda(device)

        self.inferenceBox.device = self.dboxes.device

        return super().cuda(device)


    def build(self, **kwargs):

        ### feature layers ###
        self.build_feature(**kwargs)

        ### addon layers ###
        self.build_addon(**kwargs)

        ### classifier layers ###
        self.build_classifier(**kwargs)

        ### default box ###
        self.defaultBox = self.defaultBox.build(self.feature_layers, self.classifier_source_names,
                                                self.localization_layers)

        self.init_weights()

        return self

    def forward(self, x, targets=None):
        """
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :param targets: list of Tensor, represents ground truth. if it's None, calculate as inference mode.
        :return:
            if training:
                pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
                predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_labels)
                targets: Tensor, matched targets. shape = (batch num, dbox num, 4 + class num)
            else:
                predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_labels)
        """
        if not self.isBuilt:
            raise NotImplementedError(
                "Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")

        if self.training and targets is None:
            raise ValueError("pass \'targets\' for training mode")

        elif not self.training and targets is not None:
            logging.warning("forward as eval mode, but passed \'targets\'")


        batch_num = x.shape[0]

        # feature
        sources = []
        addon_i = 1
        for name, layer in self.feature_layers.items():
            x = layer(x)

            source = x
            if name in self.addon_source_names:
                if name not in self.classifier_source_names:
                    logging.warning("No meaning addon: {}".format(name))
                source = self.addon_layers['addon_{}'.format(addon_i)](source)
                addon_i += 1

            # get features by feature map convolution
            if name in self.classifier_source_names:
                sources += [source]

        # classifier
        locs, confs = [], []
        for source, loc_name, conf_name in zip(sources, self.localization_layers, self.confidence_layers):
            locs += [self.localization_layers[loc_name](source)]
            confs += [self.confidence_layers[conf_name](source)]

        predicts = self.predictor(locs, confs)

        if self.training:
            pos_indicator, targets = self.encoder(targets, self.dboxes, batch_num)
            return pos_indicator, predicts, targets
        else:
            predicts = self.decoder(predicts, self.dboxes)
            return predicts

    def learn(self, x, targets):
        """
        Alias as self(x, targets)
        :param x: Tensor, input Tensor whose shape is (batch, c, h, w)
        :param targets: Tensor, list of Tensor, whose shape = (object num, 4 + class num) including background
        :return:
            pos_indicator: Bool Tensor, shape = (batch, default box num). this represents whether each default box is object or background.
            predicts: localization and confidence Tensor, shape is (batch, total_dbox_num, 4+class_labels)
            targets: Tensor, matched targets. shape = (batch num, dbox num, 4 + class num)
        """
        return self(x, targets)

    def infer(self, image, conf_threshold=None, toNorm=False, visualize=False):
        """
        Caution: this function use function checking image type which is slightly slow compared to __call__
                 if the performance is important for you, call __call__ directly
        :param image: ndarray or Tensor of list or tuple, or ndarray, or Tensor. Note that each type will be handled as;
            ndarray of list or tuple, ndarray: (?, h, w, c). channel order will be handled as RGB
            Tensor of list or tuple, Tensor: (?, c, h, w). channel order will be handled as RGB
        :param conf_threshold: float or None, if it's None, default value will be passed
        :param toNorm: bool, whether to normalize passed image
        :param visualize: bool,
        :return:
            if visualize:
                infers: list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
                visualized_imgs: list of ndarray, whose order is rgb
                orig_imgs: list of ndarray, whose order is rgb
            else:
                infers: list of tensor, shape = (box num, 5=(class index, cx, cy, w, h))
                orig_imgs: list of ndarray, whose order is rgb
        """
        if not self.isBuilt:
            raise NotImplementedError("Not initialized, implement \'build_feature\', \'build_classifier\', \'build_addon\'")
        if self.training:
            raise NotImplementedError("call \'eval()\' first")

        # img: Tensor, shape = (b, c, h, w)
        img, orig_imgs = _check_image(image, self.device, size=(self.input_width, self.input_height))

        # normed_img, orig_img: Tensor, shape = (b, c, h, w)
        normed_imgs, orig_imgs = _get_normed_and_origin_img(img, orig_imgs, self.rgb_means, self.rgb_stds, toNorm, self.device)

        _check_shape((self.input_channel, self.input_height, self.input_width), img.shape[1:])

        if conf_threshold is None:
            conf_threshold = self.vis_conf_threshold if visualize else self.val_conf_threshold

        with torch.no_grad():

            # predict
            predicts = self(normed_imgs)

            # list of tensor, shape = (box num, 6=(class index, confidence, cx, cy, w, h))
            infers = self.inferenceBox(predicts, conf_threshold)

            img_num = normed_imgs.shape[0]
            if visualize:
                visualized_imgs = [toVisualizeRectLabelRGBimg(orig_imgs[i], locs=infers[i][:, 2:], inf_labels=infers[i][:, 0], tensor2cvimg=False,
                                                              inf_confs=infers[i][:, 1], classe_labels=self.class_labels, verbose=False) for i in range(img_num)]
                return infers, visualized_imgs, orig_imgs
            else:
                return infers, orig_imgs

    def load_for_finetune(self, path):
        """
        load weights from input to extra features weights for fine tuning
        :param path: str
        :return: self
        """
        pretrained_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict = self.state_dict()

        # rename
        pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
        renamed = [(pre_key, pretrained_state_dict[pre_key]) for pre_key in pre_keys if not ('conf' in pre_key or 'loc' in pre_key)]

        # set vgg layer's parameters
        model_state_dict.update(OrderedDict(renamed))
        self.load_state_dict(model_state_dict, strict=False)

        logging.info("model loaded")

class SSDvggBase(SSDBase):

    def __init__(self, train_config, val_config, defaultBox, codec=None, predictor=None, inferenceBox=None, **build_kwargs):
        """
        :param train_config: SSDTrainConfig
        :param val_config: SSDValonfig
        :param defaultBox: instance inheriting DefaultBoxBase
        :param codec: SSDCodec, if it's None, use default SSDCodec
        :param predictor: Predictor, if it's None, use default Predictor
        :param inferenceBox: InferenceBox, if it's None, use default InferenceBox
        """
        self._vgg_index = -1

        super().__init__(train_config, val_config, defaultBox,
                         codec=codec, predictor=predictor, inferenceBox=inferenceBox, **build_kwargs)

    def build_feature(self, **kwargs):
        """
        :param vgg_layers: nn.ModuleDict
        :param extra_layers: nn.ModuleDict
        :return:
        """
        vgg_layers = kwargs.get('vgg_layers')
        extra_layers = kwargs.get('extra_layers')

        feature_layers = []
        vgg_layers = _check_ins('vgg_layers', vgg_layers, nn.ModuleDict)
        for name, module in vgg_layers.items():
            feature_layers += [(name, module)]
        self._vgg_index = len(feature_layers)

        extra_layers = _check_ins('extra_layers', extra_layers, nn.ModuleDict)
        for name, module in extra_layers.items():
            feature_layers += [(name, module)]

        self.feature_layers = nn.ModuleDict(OrderedDict(feature_layers))

    def build_addon(self, **kwargs):
        addon_layers = []
        for i, name in enumerate(self.addon_source_names):
            addon_layers += [
                ('addon_{}'.format(i + 1), L2Normalization(self.feature_layers[name].out_channels, gamma=20))
            ]
        self.addon_layers = nn.ModuleDict(addon_layers)

    def build_classifier(self, **kwargs):
        # loc and conf layers
        in_channels = tuple(self.feature_layers[name].out_channels for name in self.classifier_source_names)

        _dbox_num_per_fpixel = [len(aspect_ratio) * 2 for aspect_ratio in self.aspect_ratios]
        # loc
        out_channels = tuple(dbox_num * 4 for dbox_num in _dbox_num_per_fpixel)
        localization_layers = [
            *Conv2d.block('_loc', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3), padding=1,
                          batch_norm=False)
        ]
        self.localization_layers = nn.ModuleDict(OrderedDict(localization_layers))

        # conf
        out_channels = tuple(dbox_num * self.class_nums_with_background for dbox_num in _dbox_num_per_fpixel)
        confidence_layers = [
            *Conv2d.block('_conf', len(_dbox_num_per_fpixel), in_channels, out_channels, kernel_size=(3, 3),
                          padding=1, batch_norm=False)
        ]
        self.confidence_layers = nn.ModuleDict(OrderedDict(confidence_layers))


    def init_weights(self):
        _initialize_xavier_uniform(self.feature_layers)
        _initialize_xavier_uniform(self.localization_layers)
        _initialize_xavier_uniform(self.confidence_layers)

# weights management
def load_vgg_weights(model, name):
    assert isinstance(model, SSDvggBase), "must be inherited SSDvggBase"

    #model_dir = weights_path(__file__, _root_num=2, dirname='weights')
    model_dir = os.path.join(os.path.expanduser("~"), 'weights')

    model_url = _get_model_url(name)
    pretrained_state_dict = load_state_dict_from_url(model_url, model_dir=model_dir)
    #pretrained_state_dict = torch.load('/home/kado/Desktop/program/machile-learning/dl.pytorch/weights/vgg16_reducedfc.pth')
    model_state_dict = model.state_dict()

    # rename
    renamed = []
    pre_keys, mod_keys = list(pretrained_state_dict.keys()), list(model_state_dict.keys())
    # to avoid Error regarding num_batches_tracked
    pre_ind = 0
    for mod_ind in range(model._vgg_index):
        pre_key, mod_key = pre_keys[pre_ind], mod_keys[mod_ind]
        if 'num_batches_tracked' in mod_key:
            continue
        renamed += [(mod_key, pretrained_state_dict[pre_key])]
        pre_ind += 1
    """
    for (pre_key, mod_key) in zip(pre_keys[:model._vgg_index], mod_keys[:model._vgg_index]):
        renamed += [(mod_key, pretrained_state_dict[pre_key])]
    """


    # set vgg layer's parameters
    model_state_dict.update(OrderedDict(renamed))
    model.load_state_dict(model_state_dict, strict=False)

    logging.info("model loaded")



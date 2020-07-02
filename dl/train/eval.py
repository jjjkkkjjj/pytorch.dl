import abc
import sys
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('...')
from ..data.object.datasets import _DatasetBase
from ..models.base import ObjectDetectionModelBase
from .._utils import _check_ins

# mAP: https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge
# https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
# https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522
class EvaluatorBase(object):
    def __init__(self, dataloader, iteration_interval=5000, verbose=True, **eval_kwargs):
        self.dataloader = _check_ins('dataloader', dataloader, DataLoader)
        self.dataset = _check_ins('dataloader.dataset', dataloader.dataset, _DatasetBase)
        self.iteration_interval = _check_ins('iteration_interval', iteration_interval, int)
        self.verbose = verbose
        self.device = None
        self.eval_kwargs = eval_kwargs

        self._result = {}

    @property
    def class_labels(self):
        return self.dataset.class_labels
    @property
    def class_nums(self):
        return self.dataset.class_nums

    def __call__(self, model):
        model = _check_ins('model', model, ObjectDetectionModelBase)
        self.device = model.device

        # targets_loc: list of ndarray, whose shape = (targets box num, 4)
        # targets_label: list of ndarray, whose shape = (targets box num, 1)
        targets_loc, targets_label = [], []
        # infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        # infers_label: list of ndarray, whose shape = (inferred box num, 1)
        # infers_conf: list of ndarray, whose shape = (inferred box num, 1)
        infers_loc, infers_label, infers_conf = [], [], []


        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # predict
        #dataloader = iter(self.dataloader)
        #for i in range(10): # debug
        #    images, targets = next(dataloader)
        for i, (images, targets) in enumerate(self.dataloader):
            images = images.to(self.device)

            # infer is list of Tensor, shape = (box num, 6=(class index, confidence, cx, cy, w, h))
            infer = model.infer(images, visualize=False)

            targets_loc += [target.cpu().numpy()[:, :4] for target in targets]
            targets_label += [np.argmax(target.cpu().numpy()[:, 4:], axis=1) for target in targets]

            infers_loc += [inf.cpu().numpy()[:, 2:] for inf in infer]
            infers_label += [inf.cpu().numpy()[:, 0] for inf in infer]
            infers_conf += [inf.cpu().numpy()[:, 1] for inf in infer]
            """slower
            for target in targets:
                target = target.cpu().numpy()
                targets_loc += [target[:, :4]]
                targets_label += [np.argmax(target[:, 4:], axis=1)]

            for inf in infer:
                inf = inf.cpu().numpy()
                infers_loc += [inf[:, 2:]]
                infers_label += [inf[:, 0]]
                infers_conf += [inf[:, 1]]
            """

            if self.verbose:
                sys.stdout.write('\rinferring...\t{}/{}:\t{}%'.format(i+1, len(self.dataloader), int(100.*(i+1.)/len(self.dataloader))))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write('\n')
            sys.stdout.flush()

        # debug
        #_save_debugfile(targets_loc, targets_label, infers_loc, infers_label, infers_conf)

        return self.eval(targets_loc, targets_label, infers_loc, infers_label, infers_conf, **self.eval_kwargs)

    @abc.abstractmethod
    def eval(self, targets_loc, targets_label, infers_loc, infers_label, infers_conf, **kwargs):
        """
        :param targets_loc: list of ndarray, whose shape = (targets box num, 4)
        :param targets_label: list of ndarray, whose shape = (targets box num,)
        :param infers_loc: list of ndarray, whose shape = (inferred box num, 4)
        :param infers_label: list of ndarray, whose shape = (inferred box num,)
        :param infers_conf: list of ndarray, whose shape = (inferred box num,)
        Note that above len(list) == image number
        :param kwargs:
        :return:
        """
        raise NotImplementedError()




import math
import torch
from torch import nn
import time, logging, abc, sys, os

from .._utils import _check_ins
from ..models.base import ModelBase
from .graph import LiveGraph
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from .save import SaveManager

"""
    ref: https://nextjournal.com/gkoehler/pytorch-mnist
"""
class TrainLoggerBase(object):
    model: ModelBase
    optimizer: Optimizer
    scheduler: _LRScheduler

    def __init__(self, loss_module, model, optimizer, scheduler=None):
        self.loss_module = loss_module
        self.model = _check_ins('model', model, (ModelBase, nn.DataParallel))
        self.optimizer = _check_ins('optimizer', optimizer, Optimizer)
        self.scheduler = _check_ins('scheduler', scheduler, _LRScheduler, allow_none=True)

        self._now_epoch = -1
        self._now_iteration = -1

        # losses_dict is dict of list
        self._losses = {}
        self._x = []
        self._mode = None

    @property
    def now_epoch(self):
        return self._now_epoch
    @property
    def now_iteration(self):
        return self._now_iteration

    @property
    def device(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.device
        else:
            return self.model.device

    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, val):
        if not val in ['epoch', 'iteration']:
            raise ValueError()
        self._mode = val

    @property
    def isEpochMode(self):
        if self._mode is None:
            raise NotImplementedError('Must set mode')
        return self._mode == 'epoch'
    @property
    def isIterationMode(self):
        if self._mode is None:
            raise NotImplementedError('Must set mode')
        return self._mode == 'iteration'


    def __getattr__(self, item):
        """
        return losses_dict value too as attribution
        :param item:
        :return:
        """
        if item in self._losses.keys():
            return self._losses[item]
        raise AttributeError('\'{}\' object has no attribute \'{}\''.format(type(self).__name__, item))

    def set_losses(self, x, names, losses):
        for lossname, lossval in zip(names, losses):
            assert isinstance(lossval, (int, float)), "loss value must be number"
            if not lossname in self._losses.keys():
                self._losses[lossname] = []
            self._losses[lossname] += [lossval]
        self._x += [x]

    def train_epoch(self, savemanager, max_epochs, train_loader, start_epoch=0):
        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(train_loader.batch_size))
        max_iterations = max_epochs * iter_per_epoch
        start_iteration = start_epoch * len(train_loader.dataset)

        self._mode = 'epoch'

        self._train(savemanager, max_iterations, train_loader, start_iteration)

    def train_iter(self, savemanager, max_iterations, train_loader, start_iteration=0):
        self._mode = 'iteration'

        self._train(savemanager, max_iterations, train_loader, start_iteration)

    def _train(self, savemanager, max_iterations, train_loader, start_iteration):
        _ = _check_ins('savemanager', savemanager, SaveManager)

        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(train_loader.batch_size))
        max_epochs = math.ceil(max_iterations / len(train_loader.dataset))

        self._now_epoch = int(start_iteration / len(train_loader.dataset))
        self._now_iteration = start_iteration
        start_epoch = self._now_epoch

        train_iterator = iter(train_loader)
        for now_iteration in range(start_iteration, max_iterations):
            try:
                item = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                item = next(train_iterator)

            iter_starttime = time.time()

            self.optimizer.zero_grad()
            names, losses = self.learn(*item)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            iter_time = time.time() - iter_starttime

            _ = _check_ins('names', names, (list, tuple))
            _ = _check_ins('losses_dict', losses, (list, tuple))

            self._now_iteration += 1

            percent = 100. * (self.now_iteration % iter_per_epoch) / iter_per_epoch
            percentage = '{:.0f}%[{}/{}]'.format(percent, self.now_iteration % iter_per_epoch, iter_per_epoch)
            self.update_log(self.now_epoch, self.now_iteration, percentage, iter_time, names, losses)
            if self.isIterationMode:
                if self.now_iteration % savemanager.plot_interval == 0 or \
                        self.now_iteration == start_iteration + 1 or \
                             self.now_iteration == max_iterations:
                    self.set_losses(self.now_iteration, names, losses)

                saved_path, removed_path = savemanager.update_iteration(self.model, self.now_iteration, max_iterations)
                self.update_checkpointslog(saved_path, removed_path)

            if self.now_iteration % iter_per_epoch == 0:
                self._now_epoch += 1

                if self.isEpochMode:
                    if self.now_epoch % savemanager.plot_interval == 0 or \
                            self.now_epoch == start_epoch + 1 or \
                                 self.now_epoch == max_epochs:
                        self.set_losses(self.now_epoch, names, losses)

                    saved_path, removed_path = savemanager.update_epoch(self.model, self.now_epoch, max_epochs)
                    self.update_checkpointslog(saved_path, removed_path)

        if self.isIterationMode:
            savemanager.finish(self.model, self.optimizer, self.scheduler, 'iteration', self._x, names, self._losses)
        elif self.isEpochMode:
            savemanager.finish(self.model, self.optimizer, self.scheduler, 'epoch', self._x, names, self._losses)
        else:
            raise ValueError()


    def update_log(self, epoch, iteration, percentage, calc_time, names, losses):
        """
        this function will be called by train_iter only
        :param epoch
        :param iteration:
        :param percentage:
        :param calc_time:
        :param names: list of str
        :param losses: list of float
        :return:
        """
        iter_template = '\rTraining... Epoch: {}, Iter: {},\t {}\t{}\tIter time: {:.4f}'
        loss_template = ''
        for name, loss in zip(names, losses):
            loss_template += '{}: {:.6f} '.format(name, loss)

        sys.stdout.write(iter_template.format(epoch, iteration, percentage, loss_template, calc_time))
        sys.stdout.flush()

    def update_checkpointslog(self, saved_path, removed_path):
        if saved_path == '':
            return

        # append information for verbose
        saved_info = '\nSaved model to {}\n'.format(saved_path)
        if removed_path != '':
            removed_info = ' and removed {}'.format(removed_path)
            saved_info = '\n' + 'Saved model as {}{}\n'.format(os.path.basename(saved_path), removed_info)
            print(saved_info)
        else:
            print(saved_info)

    @abc.abstractmethod
    def learn(self, *items):
        """
        :param items: the value returned by train_loader
        :return:
            names: list of str
            losses: list of float
        """
        raise NotImplementedError()


class TrainConsoleLoggerBase(TrainLoggerBase):
    pass

class TrainJupyterLoggerBase(TrainLoggerBase):
    live_graph: LiveGraph
    def __init__(self, live_graph, loss_module, model, optimizer, scheduler=None):
        self.live_graph = _check_ins('live_graph', live_graph, LiveGraph)
        super().__init__(loss_module, model, optimizer, scheduler)


    def set_losses(self, x, names, losses):
        super().set_losses(x, names, losses)
        self.live_graph.update(self.now_epoch, self.now_iteration, self._x, names, self._losses)

    def update_checkpointslog(self, saved_path, removed_path):
        if saved_path == '':
            return

        # append information for verbose
        saved_info = '\nSaved model to {}\n'.format(saved_path)
        if removed_path != '':
            removed_info = '\nRemoved {}'.format(removed_path)
            saved_info = '\n' + 'Saved model as {}{}'.format(os.path.basename(saved_path), removed_info)
            self.live_graph.update_info(saved_info)
        else:
            print(saved_info)

from ..models.base import ObjectRecognitionModelBase
def _learn_objrecog(self, images, targets):
    images = images.to(self.device)
    targets = [target.to(self.device) for target in targets]

    predicts = self.model(images, targets)
    if not isinstance(predicts, (tuple, list)):
        raise ValueError('model\'s output must be tuple or list, but got {}'.format(predicts.__name__))

    loss = self.loss_module(*predicts)
    loss.backward()

    return ['loss'], [loss.item()]

class TrainObjectRecognitionConsoleLogger(TrainConsoleLoggerBase):
    model: ObjectRecognitionModelBase

    def __init__(self, loss_module, model, optimizer, scheduler=None):
        super().__init__(loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (ObjectRecognitionModelBase, nn.DataParallel))

    def learn(self, images, targets):
        return _learn_objrecog(self, images, targets)

class TrainObjectRecognitionJupyterLogger(TrainJupyterLoggerBase):
    model: ObjectRecognitionModelBase

    def __init__(self, live_graph, loss_module, model, optimizer, scheduler=None):
        super().__init__(live_graph, loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (ObjectRecognitionModelBase, nn.DataParallel))

    def learn(self, images, targets):
        return _learn_objrecog(self, images, targets)


from ..models.base import ObjectDetectionModelBase
def _learn_objdetn(self, images, targets):
    images = images.to(self.device)
    targets = [target.to(self.device) for target in targets]

    pos_indicator, predicts, gts = self.model(images, targets)

    confloss, locloss = self.loss_module(pos_indicator, predicts, gts)
    loss = confloss + self.loss_module.alpha * locloss
    loss.backward()

    return ['total', 'loc', 'conf'], [loss.item(), locloss.item(), confloss.item()]

class TrainObjectDetectionConsoleLogger(TrainConsoleLoggerBase):
    model: ObjectDetectionModelBase

    def __init__(self, loss_module, model, optimizer, scheduler=None):
        super().__init__(loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (ObjectDetectionModelBase, nn.DataParallel))

    def learn(self, images, targets):
        return _learn_objdetn(self, images, targets)

class TrainObjectDetectionJupyterLogger(TrainJupyterLoggerBase):
    model: ObjectDetectionModelBase

    def __init__(self, live_graph, loss_module, model, optimizer, scheduler=None):
        super().__init__(live_graph, loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (ObjectDetectionModelBase, nn.DataParallel))

    def learn(self, images, targets):
        return _learn_objdetn(self, images, targets)


from ..models.base import TextSpottingModelBase
def _learn_txtspotting(self, images, targets, texts):
    """
    :param self:
    :param images: Tensor, shape = (b, c, h, w)
    :param labels: list(b) of Tensor, shape = (text number in image, 4=(rect)+8=(quads)+...)
    :param texts: list(b) of list(text number) of Tensor, shape = (characters number,)
    :return:
    """
    images = images.to(self.device)
    targets = [target.to(self.device) for target in targets]
    texts = [[t.to(self.device) for t in _txts] for _txts in texts]

    detn, recog = self.model(images, targets, texts)

    confloss, locloss = self.loss_module(detn, recog)
    loss = confloss + self.loss_module.alpha * locloss
    loss.backward()

    return ['total', 'loc', 'conf'], [loss.item(), locloss.item(), confloss.item()]

class TrainTextSpottingConsoleLogger(TrainConsoleLoggerBase):
    model: TextSpottingModelBase

    def __init__(self, loss_module, model, optimizer, scheduler=None):
        super().__init__(loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (TextSpottingModelBase, nn.DataParallel))

    def learn(self, images, targets, texts):
        return _learn_txtspotting(self, images, targets, texts)

class TrainTextSpottingJupyterLogger(TrainJupyterLoggerBase):
    model: TextSpottingModelBase

    def __init__(self, live_graph, loss_module, model, optimizer, scheduler=None):
        super().__init__(live_graph, loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (TextSpottingModelBase, nn.DataParallel))

    def learn(self, images, targets, texts):
        return _learn_txtspotting(self, images, targets, texts)


import math
import torch
from torch import nn
import time, logging, abc, sys, os

from .._utils import _check_ins
from ..models.base import ModelBase
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
        max_iterations = max_epochs * len(train_loader.dataset)
        start_iteration = start_epoch * len(train_loader.dataset)

        self._mode = 'epoch'

        self._train(savemanager, max_iterations, train_loader, start_iteration)

    def train_iter(self, savemanager, max_iterations, train_loader, start_iteration=0):
        self._mode = 'iteration'

        self._train(savemanager, max_iterations, train_loader, start_iteration)

    def _train(self, savemanager, max_iterations, train_loader, start_iteration):
        _ = _check_ins('savemanager', savemanager, SaveManager)

        iter_per_epoch = math.ceil(len(train_loader.dataset) / float(train_loader.batch_size))
        max_epochs = math.ceil(max_iterations / float(iter_per_epoch))

        self._now_epoch = int(start_iteration / len(train_loader.dataset))
        self._now_iteration = start_iteration
        start_epoch = self._now_epoch

        train_iterator = iter(train_loader)
        for now_iteration in range(start_iteration, max_iterations):
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

            percentage = 100. * (self.now_iteration % iter_per_epoch) / iter_per_epoch
            self.update_log(self.now_epoch, self.now_iteration, percentage, iter_time, names, losses)
            if self.isIterationMode:
                if self.now_iteration % savemanager.plot_interval == 0 or \
                        self.now_iteration == start_iteration + 1 or \
                             self.now_iteration == max_iterations:
                    self.set_losses(self.now_iteration, names, losses)

                saved_path, removed_path = savemanager.update_iteration(self.model, self.now_iteration, max_iterations)
                self.update_checkpointslog(saved_path, removed_path)

            if self.now_iteration % len(train_loader.dataset) == 0:
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
        iter_template = '\rTraining... Epoch: {}, Iter: {},\t {:.0f}%\t{}\tIter time: {:.4f}'
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


from ..models.base import ObjectDetectionModelBase
class TrainObjectDetectionConsoleLogger(TrainConsoleLoggerBase):
    model: ObjectDetectionModelBase

    def __init__(self, loss_module, model, optimizer, scheduler=None):
        super().__init__(loss_module, model, optimizer, scheduler)
        _ = _check_ins('model', model, (ObjectDetectionModelBase, nn.DataParallel))

    def learn(self, images, targets):
        images = images.to(self.device)
        targets = [target.to(self.device) for target in targets]

        pos_indicator, predicts, gts = self.model(images, targets)

        confloss, locloss = self.loss_module(pos_indicator, predicts, gts)
        loss = confloss + self.loss_module.alpha * locloss
        loss.backward()

        return ['total', 'loc', 'conf'], [loss.item(), locloss.item(), confloss.item()]
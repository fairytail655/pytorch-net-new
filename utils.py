import os
import shutil
import json
import time
import torch
import numpy as np
import pandas as pd

class MyLogging(object):

    def __init__(self, log_file='./log.txt'):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write("")

    def info(self, message):
        date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
        message = "[{}] - INFO: {}\n".format(date, message)
        print(message, end='')
        with open(self.log_file, 'a') as f:
            f.write(message)

    def debug(self, message):
        date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) 
        message = "[{}] - DEBUG: {}\n".format(date, message)
        with open(self.log_file, 'a') as f:
            f.write(message)

class ResultsLog(object):

    def __init__(self, path='results.csv'):
        self.path = path
        self.results = pd.DataFrame()

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            # logging.debug('OPTIMIZER - setting method = %s' %
            #               setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    # logging.debug('OPTIMIZER - setting %s = %s' %
                    #               (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    # if callable(config):
    #     optimizer = modify_optimizer(optimizer, config(epoch))
    # else:
    for e in range(epoch + 1):  # run over all epochs - sticky setting
        if e in config:
            optimizer = modify_optimizer(optimizer, config[e])

    return optimizer

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.float().topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0)
    res = correct.mul_(100.0 / batch_size)

    return res

    # kernel_img = model.features[0][0].kernel.data.clone()
    # kernel_img.add_(-kernel_img.min())
    # kernel_img.mul_(255 / kernel_img.max())
    # save_image(kernel_img, 'kernel%s.jpg' % epoch)

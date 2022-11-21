import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

def update_meter(dict_meter, dict_content, batch_size):
    for key, value in dict_meter.items():
        if isinstance(dict_content[key], torch.Tensor):
            value.update(dict_content[key].item(), batch_size)
        else:
            value.update(dict_content[key], batch_size)

class RunningMean:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        # if self.count:
        #     return float(self.total_value) / self.count
        # else:
        #     return 0 
        return self.avg

    def __str__(self):
        return str(self.value)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
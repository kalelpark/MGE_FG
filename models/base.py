import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.models as models
import math
from importlib import import_module

class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        basenet = getattr(import_module('torchvision.models'), "resnet50")
        basenet = basenet(pretrained=True)

        self.conv5 = nn.Sequential(*list(basenet.children())[:-2])

        self.pool = nn.AƒdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, args.num_classes)
    
    def forward(self, x, y = None):
        conv5 = self.conv5(x)
        conv5_pool = self.pool(conv5)
        fea = conv5_pool.view(conv5_pool.size(0), -1)
        logits = self.classifier(fea)

        outputs = {'logits':[logits]}
        return outputs        
    
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params

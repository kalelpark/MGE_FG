import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
from utils import *

def update_dest(acc_meter, key, content, target):
    if not key in acc_meter.keys():
        acc_meter[key] = RunningMean()
    acc_tmp = accuracy(content, target, topk=(1,))
    acc_meter[key].update(acc_tmp[0], len(target))

def valid(args, valid_loader, model, loss_fn):
    acc_meter = {}
    p_acc = {}

    model.eval()
    accuracy, loder_len = 0, 0
    with torch.no_grad():
        for input, target in valid_loader:
            loder_len += 1
            input, target = input.float().to(args.device), target.float().to(args.device)
            output_dict = model(input)

            logits_list = output_dict["logits"]

            if len(args.acc_keys) > 1:
                update_dest(acc_meter, "Accuracy", logits_list[-1], target)
            else:
                acc_keys = args.test_acc_keys
                for key, value in zip(acc_keys, logits_list):
                    update_dest(acc_meter, key, value.to(args.device), target)
            
            best_score = 0
            for k, v in acc_meter.items():
                if best_score < v.value:
                    best_score = v.value

            accuracy += best_score
            
    accuracy /= loder_len
    return accuracy

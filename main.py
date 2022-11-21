import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from utils import *
from models import *
from dataset import *
from trainer import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, required = True)         
    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--imgsize", type = int, default = 512, required = False)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--epochs", type = int, default = 100, required = False)
    parser.add_argument("--batchsize", type = int, default = 32, required = False)
    parser.add_argument("--lr", type = float, default = 1e-2, required = False)
    parser.add_argument("--momentum", type = float, default = 0.9, required = False)
    parser.add_argument("--weight_decay", type = float, default = 1e-4, required = False)
    parser.add_argument('--box_thred', default=0.2, type=float, required = False)
    parser.add_argument("--gpu_ids", type = str)

    args = parser.parse_args()
    
    # Change 가능!
    args.loss_keys = ['lg', 'lg1', 'lg2', 'lb1', 'lb11', 'lb12', 'lb2', 'lb21', 'lb22', 'l_gate', 'kl1', 'kl2', 'l_a']
    args.acc_keys = ['ag', 'ag1', 'agc', 'ab1', 'ab11', 'ab1c', 'ab2', 'ab21', 'ab2c', 'a_gate']
    args.test_acc_keys = ['g', 'gm', 'gc', 'b', 'bm', 'bc', 'b2', 'b2m', 'b2c', 'gate']

    if args.dataset in ['cub', "cubbird"]:
        args.num_classes = 200
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(args.seed)

    model = get_model(args)
    model = nn.DataParallel(model).to(args.device)

    train_loader, valid_loader = get_loader(args)

    # optimizer
    extractor_params = model.module.get_params(prefix = "extractor")
    classifier_params = model.module.get_params(prefix = "classifier")
    lr_cls = args.lr
    lr_extractor = 0.1 * lr_cls

    params = [
              {'params': classifier_params, 'lr': lr_cls},
              {'params': extractor_params, 'lr': lr_extractor}
              ]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss().to(args.device)
    best_loss = 1e5

    wandb.init( name = args.dataset + "_ResNet50",
                project = "Mixture of Granularity-Specific Experts for Fine-Grained_1", reinit = True)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = run(args, train_loader, model, loss_fn, optimizer, scheduler)
        valid_acc = valid(args, valid_loader, model, loss_fn)
        scheduler.step()

        wandb.log({
            "train_loss" : train_loss,
            "valid Accuracy" : valid_acc
        })
        

    print("Done..")

    
    







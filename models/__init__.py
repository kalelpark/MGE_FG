from .base import * 
from .local_cam import *

def get_model(args):
    if args.model == "base":
        return BaseNet(args)
    if args.model == "mgecnn":
        return LocalCamNet(args)
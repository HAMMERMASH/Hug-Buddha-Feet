import os, sys
sys.path.insert(0, './lib')
from utils.load_config import load_config

config = load_config('./configs/resnet_v1_101_imagenet_vid_rfcn_end2end_ohem.yaml')
print config.network

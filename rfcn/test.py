# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import cv2
import argparse
import os
import sys
import time
import logging

import mxnet as mx
sys.path.insert(0, '../')
sys.path.insert(0, '../lib')
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger
from utils.load_config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Test a R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    config = load_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()

    return args, config
 
def main():
    args, config = parse_args()
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    test_rcnn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, config.pretrained_model.path, 
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path)

if __name__ == '__main__':
    main()

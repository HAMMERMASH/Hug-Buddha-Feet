# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import argparse
import pprint
import logging
import time
import os
import numpy as np
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import Sequence_TestLoader
from core.tester import Predictor, pred_eval, pred_eval_multiprocess
from utils.load_model import load_param_from_path

def get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_data, ctx):
    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_image_shape = list((1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))
    max_hidden_shape = [max_image_shape[0], cfg.network.CONV_FEAT_CHANNEL, np.ceil(max_image_shape[2]*1.0/16).astype(int), np.ceil(max_image_shape[3]*1.0/16).astype(int)]
    max_data_shape = [[('data', tuple(max_image_shape)),
                       ('hidden', tuple(max_hidden_shape)),]]
    
    print ctx
    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, model_path,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    sym_instance = eval(cfg.symbol + '.' + cfg.symbol)(cfg)
    sym = sym_instance.get_test_symbol(cfg)
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
    roidb = imdb.gt_roidb()

    # get test data iter
    # split roidbs
    gpu_num = len(ctx)
    roidbs = [[] for x in range(gpu_num)]
    gpu_id = 0
    cur_vid = roidb[0]['image'].split('/')[-2]
    for x in roidb:
        if gpu_id >= gpu_num:
            gpu_id =  gpu_id%gpu_num
        roidbs[gpu_id].append(x)
        if x['image'].split('/')[-2] != cur_vid:
            cur_vid = x['image'].split('/')[-2]
            gpu_id += 1

    # get test data iter
    test_datas = [Sequence_TestLoader(x, cfg, batch_size=1, shuffle=shuffle, has_rpn=has_rpn) for x in roidbs]

    # load model
    arg_params, aux_params = load_param_from_path(model_path, process=True)

    # create predictor
    predictors = [get_predictor(sym, sym_instance, cfg, arg_params, aux_params, test_datas[i], [ctx[i]]) for i in range(gpu_num)]

    # start detection
    #pred_eval(0, key_predictors[0], cur_predictors[0], test_datas[0], imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)
    pred_eval_multiprocess(gpu_num, predictors, test_datas, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

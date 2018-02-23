# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import sys
import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
sys.path.insert(0, '../')
from backbones.resnet_v1 import residual_layer

class resnet_v1_101_simple_rrfcn(Symbol):
    
    def __init__(self, cfg):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]
        self.sequence_len = cfg.TRAIN.SEQUENCE_LEN

        self.num_classes = cfg.dataset.NUM_CLASSES
        self.num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        self.num_anchors = cfg.network.NUM_ANCHORS
        
        # detection work variables
        self.rpn_cls_score_weight = mx.sym.var(name='rpn_cls_score_weight', shape=(2*self.num_anchors,512,1,1), lr_mult=1, dtype='float32')
        self.rpn_cls_score_bias = mx.sym.var(name='rpn_cls_score_bias', shape=(2*self.num_anchors,), lr_mult=1, dtype='float32')
        
        self.rpn_bbox_pred_weight = mx.sym.var(name='rpn_bbox_pred_weight', shape=(4*self.num_anchors,512,1,1), lr_mult=1, dtype='float32')
        self.rpn_bbox_pred_bias = mx.sym.var(name='rpn_bbox_pred_bias', shape=(4*self.num_anchors,), lr_mult=1, dtype='float32')
        
        self.rfcn_cls_weight = mx.sym.var(name='rfcn_cls_weight', shape=(7*7*self.num_classes,512,1,1), lr_mult=1, dtype='float32')
        self.rfcn_cls_bias = mx.sym.var(name='rfcn_cls_bias', shape=(7*7*self.num_classes,), lr_mult=1, dtype='float32')

        self.rfcn_bbox_weight = mx.sym.var(name='rfcn_bbox_weight', shape=(7*7*4*self.num_reg_classes,512,1,1), lr_mult=1, dtype='float32')
        self.rfcn_bbox_bias = mx.sym.var(name='rfcn_bbox_bias', shape=(7*7*4*self.num_reg_classes,), lr_mult=1, dtype='float32')

    def get_resnet_v1(self, data):
      
        res5c_relu = residual_layer(data, self.units, self.filter_list, ['2','3','4','5'], last_stride=True)
        feat_conv_3x3 = mx.sym.Convolution(
            data=res5c_relu, kernel=(3, 3), pad=(6, 6), dilate=(6, 6), num_filter=1024, name="feat_conv_3x3")
        feat_conv_3x3_relu = mx.sym.Activation(data=feat_conv_3x3, act_type="relu", name="feat_conv_3x3_relu")
        return feat_conv_3x3_relu
    
    def get_simple(self, data, hidden):
        
        hidden = mx.sym.Pooling(data=hidden, pool_type='avg', kernel=(3,3), pad=(1,1), stride=(1,1), name='hidden_pool')

        hidden_new = mx.sym.broadcast_add(data, hidden, name='hidden_new')

        return hidden_new

    def get_detection_train_symbol(self, data, im_info, gt_boxes, rpn_label, rpn_bbox_target, rpn_bbox_weight, attr, cfg):
        conv_feats = mx.sym.SliceChannel(data, axis=1, num_outputs=2)
    
        # RPN layers
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, weight=self.rpn_cls_score_weight, bias=self.rpn_cls_score_bias, kernel=(1, 1), pad=(0, 0), num_filter=2 * self.num_anchors, name="rpn_cls_score_"+attr)
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, weight=self.rpn_bbox_pred_weight, bias=self.rpn_bbox_pred_bias, kernel=(1, 1), pad=(0, 0), num_filter=4 * self.num_anchors, name="rpn_bbox_pred_"+attr)
    
        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_"+attr)
    
        # classification
        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob_"+attr)
        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_'+attr+'_', scalar=1.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=self.num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
        else:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_'+attr+'_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss_'+attr, data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
        
        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act_"+attr)
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * self.num_anchors, -1, 0), name='rpn_cls_act_reshape_'+attr)

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_'+attr,
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_'+attr,
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
    
         # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape_'+attr)
        rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=self.num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
    
        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, weight=self.rfcn_cls_weight, bias=self.rfcn_cls_bias, kernel=(1, 1), num_filter=7*7*self.num_classes, name="rfcn_cls_"+attr)
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, weight=self.rfcn_bbox_weight, bias=self.rfcn_bbox_bias, kernel=(1, 1), num_filter=7*7*4*self.num_reg_classes, name="rfcn_bbox_"+attr)
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois_'+attr, data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=self.num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois_'+attr, data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois_'+attr, data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois_'+attr, data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape_'+attr, data=cls_score, shape=(-1, self.num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape_'+attr, data=bbox_pred, shape=(-1, 4 * self.num_reg_classes))
    
        # classification
        if cfg.TRAIN.ENABLE_OHEM:
            print 'use ohem!'
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=self.num_classes,
                                                           num_reg_classes=self.num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob_'+attr, data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_'+attr+'_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss_'+attr, data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob_'+attr, data=cls_score, label=label, normalization='valid')
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_'+attr+'_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss_'+attr, data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = label
    
        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape_'+attr)
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, self.num_classes), name='cls_prob_reshape_'+attr)
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * self.num_reg_classes), name='bbox_loss_reshape_'+attr)
    
        group = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
        return group

    def get_train_symbol(self, cfg):

        data = mx.sym.Variable(name="data")
        im_infos = mx.sym.Variable(name="im_info")
        gt_boxeses = mx.sym.Variable(name="gt_boxes")
        rpn_labels = mx.sym.Variable(name='label')
        rpn_bbox_targets = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weights = mx.sym.Variable(name='bbox_weight')
    
        # shared convolutional layers
        conv_feats = self.get_resnet_v1(data)
        conv_feats = mx.sym.SliceChannel(conv_feats, axis=0, num_outputs=self.sequence_len)

        im_infos = mx.sym.SliceChannel(im_infos, axis=0, num_outputs=self.sequence_len) 
        gt_boxeses = mx.sym.SliceChannel(gt_boxeses, axis=0, num_outputs=self.sequence_len)
        rpn_labels = mx.sym.SliceChannel(rpn_labels, axis=0, num_outputs=self.sequence_len)
        rpn_bbox_targets = mx.sym.SliceChannel(rpn_bbox_targets, axis=0, num_outputs=self.sequence_len)
        rpn_bbox_weights = mx.sym.SliceChannel(rpn_bbox_weights, axis=0, num_outputs=self.sequence_len)


        hidden_list = []
        hidden = self.get_simple(conv_feats[0], conv_feats[0])
        hidden_list.append(hidden) 
        for i in range(self.sequence_len-1):
            hidden = self.get_simple(conv_feats[i+1], hidden) 
            hidden_list.append(hidden)
          
        # output lists
        group_list = [[] for _ in range(5)]
        for i in range(self.sequence_len):
            net = self.get_detection_train_symbol(data=hidden_list[i], im_info=im_infos[i], gt_boxes=gt_boxeses[i], rpn_label=rpn_labels[i],
                                             rpn_bbox_target=rpn_bbox_targets[i], rpn_bbox_weight=rpn_bbox_weights[i], attr='{}'.format(i+1), cfg=cfg) 

            for j in range(5):
                group_list[j].append(net[j])
        
        
        for i in range(5):
            group_list[i] = mx.sym.concat(*group_list[i],dim=0)
        
        group = mx.sym.Group(group_list)
        self.sym=group

        return group

    def get_test_symbol(self, cfg):
    
        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS # num anchors per cell

        data = mx.sym.Variable(name="data")
        hidden = mx.sym.Variable(name="hidden")
        im_info = mx.sym.Variable(name="im_info")
    
        # shared convolutional layers
        conv_feat = self.get_resnet_v1(data)
        hidden_new = self.get_simple(conv_feat, hidden)
        conv_feats = mx.sym.SliceChannel(hidden_new, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, weight=self.rpn_cls_score_weight, bias=self.rpn_cls_score_bias, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, weight=self.rpn_bbox_pred_weight, bias=self.rpn_bbox_pred_bias, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
    
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
    
        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            NotImplemented
    
        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, weight=self.rfcn_cls_weight, bias=self.rfcn_cls_bias, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, weight=self.rfcn_bbox_weight, bias=self.rfcn_bbox_bias, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
    
        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))
    
        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')
    
        # group output
        group = mx.sym.Group([rois, cls_prob, bbox_pred, hidden_new, mx.sym.BlockGrad(conv_feat, name='conv_feat')])
        self.sym = group
        # arg_shape, data_shape, axus_shape = psroipooled_loc_rois.infer_shape(data=(1,3,1000,600))
        # print data_shape
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        """
        arg_params['feat_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['feat_conv_3x3_weight'])
        arg_params['feat_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['feat_conv_3x3_bias'])

        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])
        """

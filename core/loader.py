# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from utils.image import tensor_vstack
from rpn.rpn import get_rpn_testbatch, get_rpn_batch, get_rpn_batch_, assign_anchor
from rcnn import get_rcnn_testbatch, get_rcnn_batch

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            self.data_name = ['data', 'rois']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb, self.cfg)
        self.data = [[mx.nd.array(idata[name]) for name in self.data_name] for idata in data]
        self.im_info = im_info

class AnchorLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, normalize_target=False, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.4, 0.4)):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.normalize_target = normalize_target
        self.bbox_mean = bbox_mean
        self.bbox_std = bbox_std

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]

        new_label_list = []
        for data, label in zip(data_list, label_list):
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

            # assign anchor for label
            label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border,
                                  self.normalize_target, self.bbox_mean, self.bbox_std)
            new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_batch(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

class Sequence_AnchorLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, normalize_target=False, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.4, 0.4)):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(Sequence_AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.normalize_target = normalize_target
        self.bbox_mean = bbox_mean
        self.bbox_std = bbox_std
        
        # max interval between frames
        self.frame_interval = cfg.TRAIN.MAX_INTERVAL

        # reduce epoch length
        self.sparse_factor = cfg.TRAIN.SPARSE_FACTOR

        # infer properties from roidb
        self.size = len(roidb)
        self.vid_start_ind = None 
        self.vid_len = None
        self.find_vid_start_ind()
        self.num_video = len(self.vid_start_ind)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size * self.sparse_factor
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape


    def get_batch_individual(self):
        index_list = self.get_index_list()
        roidb = [self.roidb[i] for i in index_list]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_batch_(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        data['gt_boxes'] = label['gt_boxes']
        for i in range(len(iroidb)):

            # assign anchor for label
            im_info = np.expand_dims(data['im_info'][i], axis=0)
            gt_boxes = label['gt_boxes'][i][0:label['num_gt'][i]]
            anchor_dict = assign_anchor(feat_shape, gt_boxes, im_info, self.cfg, self.feat_stride, 
                                        self.anchor_scales, self.anchor_ratios, self.allowed_border,
                                        self.normalize_target, self.bbox_mean, self.bbox_std)
            if i == 0:
                anchor_label = anchor_dict
            else:
                anchor_label['bbox_weight'] = np.concatenate((anchor_label['bbox_weight'], anchor_dict['bbox_weight']))
                anchor_label['bbox_target'] = np.concatenate((anchor_label['bbox_target'], anchor_dict['bbox_target']))
                anchor_label['label'] = np.concatenate((anchor_label['label'], anchor_dict['label']))
        return {'data': data, 'label': anchor_label}
    def get_index_list(self):
        """
          choose sequence of frames randomly
        """
        index_list = []
        sequence_batch = self.batch_size / len(self.ctx)
        for i in range(len(self.ctx)):
            stop_flag = False
            while not stop_flag:
                cur_vid = np.random.randint(0, self.num_video)
                cur_start = self.vid_start_ind[cur_vid] + np.random.randint(0, self.vid_len[cur_vid])
                curs = []
                curs.append(cur_start)
                for j in range(sequence_batch - 1):
                    cur_current = cur_start + np.random.randint(1, self.frame_interval)
                    if cur_current - self.vid_start_ind[cur_vid] + 1 >= self.vid_len[cur_vid]:
                        break
                    else:
                        curs.append(cur_current)
                        cur_start = cur_current
                if len(curs) == sequence_batch:
                    stop_flag = True
            for ind in curs:
                index_list.append(ind)
        return index_list

    def find_vid_start_ind(self):
        """
          find start index of each video in roidb
        """    
        self.vid_start_ind = []
        self.vid_len = []
        self.vid_start_ind.append(0)
        cur_vid_name = self.roidb[0]['image'].split('/')[-2]
        for i in range(self.size - 1):
            vid_name = self.roidb[i+1]['image'].split('/')[-2]
            if vid_name != cur_vid_name:
                self.vid_start_ind.append(i+1)
                cur_vid_name = vid_name
                self.vid_len.append(i+1-self.vid_start_ind[-2])

        self.vid_len.append(self.size - self.vid_start_ind[-1])

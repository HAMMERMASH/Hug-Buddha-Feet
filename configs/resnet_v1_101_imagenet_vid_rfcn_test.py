---
MXNET_VERSION: "mxnet"
output_path: "./output/rfcn/imagenet_vid"
symbol: resnet_v1_101_rfcn
gpus: '0,1'
CLASS_AGNOSTIC: true
SCALES:
- - 600
  - 1000
default:
  frequent: 100
  kvstore: device
pretrained_model:
  path: "/slwork/HAMMERMASH/Detections/rfcn/output/rfcn/imagenet_vid/resnet_v1_101_imagenet_vid_rfcn_end2end_ohem/DET_train_30classes_VID_train_15frames/rfcn_vid-0002.params" 
  pretrained_epoch: 4
network:
  PIXEL_MEANS:
  - 103.06
  - 115.90
  - 123.15
  IMAGE_STRIDE: 0
  RCNN_FEAT_STRIDE: 16
  RPN_FEAT_STRIDE: 16
  FIXED_PARAMS:
  - conv1
  - bn_conv1
  - res2
  - bn2
  - gamma
  - beta
  ANCHOR_RATIOS:
  - 0.5
  - 1
  - 2
  ANCHOR_SCALES:
  - 8
  - 16
  - 32
  ANCHOR_MEANS:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  ANCHOR_STDS:
  - 0.1
  - 0.1
  - 0.4
  - 0.4
  NORMALIZE_RPN: TRUE
  NUM_ANCHORS: 9
dataset:
  NUM_CLASSES: 31
  dataset: ImageNetVID
  dataset_path: "/slwork/HAMMERMASH/Detections/data/ILSVRC2015"
  image_set: DET_train_30classes+VID_train_15frames
  root_path: "/slwork/HAMMERMASH/Detections/data/"
  test_image_set: VID_val_frames
  proposal: rpn
TRAIN:
  lr: 0.00025
  lr_step: '1.333'
  lr_factor: 0.1 
  warmup: false
  warmup_lr: 0
  warmup_step: 0
  momentum: 0.9
  wd: 0.0005
  begin_epoch: 0
  end_epoch: 4
  model_prefix: 'rfcn_vid'
  # whether resume training
  RESUME: false
  # whether flip image
  FLIP: true
  # whether shuffle image
  SHUFFLE: true
  # whether use OHEM
  ENABLE_OHEM: true
  # size of images for each device, 1 for e2e
  BATCH_IMAGES: 1
  # e2e changes behavior of anchor loader and metric
  END2END: true
  # group images with similar aspect ratio
  ASPECT_GROUPING: true
  # R-CNN
  # rcnn rois batch size
  BATCH_ROIS: -1
  BATCH_ROIS_OHEM: 128
  # rcnn rois sampling params
  FG_FRACTION: 0.25
  FG_THRESH: 0.5
  BG_THRESH_HI: 0.5
  BG_THRESH_LO: 0.0
  # rcnn bounding box regression params
  BBOX_REGRESSION_THRESH: 0.5
  BBOX_WEIGHTS:
  - 1.0
  - 1.0
  - 1.0
  - 1.0

  # RPN anchor loader
  # rpn anchors batch size
  RPN_BATCH_SIZE: 256
  # rpn anchors sampling params
  RPN_FG_FRACTION: 0.5
  RPN_POSITIVE_OVERLAP: 0.7
  RPN_NEGATIVE_OVERLAP: 0.3
  RPN_CLOBBER_POSITIVES: false
  # rpn bounding box regression params
  RPN_BBOX_WEIGHTS:
  - 1.0
  - 1.0
  - 1.0
  - 1.0
  RPN_POSITIVE_WEIGHT: -1.0
  # used for end2end training
  # RPN proposal
  CXX_PROPOSAL: true
  RPN_NMS_THRESH: 0.7
  RPN_PRE_NMS_TOP_N: 6000
  RPN_POST_NMS_TOP_N: 300
  RPN_MIN_SIZE: 0
  # approximate bounding box regression
  BBOX_NORMALIZATION_PRECOMPUTED: true
  BBOX_MEANS:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  BBOX_STDS:
  - 0.1
  - 0.1
  - 0.2
  - 0.2
TEST:
  # use rpn to generate proposal
  HAS_RPN: true
  # size of images for each device
  BATCH_IMAGES: 1
  # RPN proposal
  CXX_PROPOSAL: true
  RPN_NMS_THRESH: 0.7
  RPN_PRE_NMS_TOP_N: 6000
  RPN_POST_NMS_TOP_N: 300
  RPN_MIN_SIZE: 0
  # RCNN nms
  NMS: 0.3
  max_per_image: 300
  test_epoch: 2

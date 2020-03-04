# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Pengfei Xian. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "run/"
_C.MULTI_RUN = False


# ---------------------------------------------------------------------------- #
# DATALOADER
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_CLASSES = 19
_C.DATALOADER.BASE_SIZE = 2048
_C.DATALOADER.CROP_SIZE = 513
_C.DATALOADER.BATCH_SIZE_TRAIN = 2
_C.DATALOADER.BATCH_SIZE_TEST = 2
_C.DATALOADER.BATCH_SIZE_VAL = 2
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.DATASET = "cityscapes"
_C.DATALOADER.DATASET_PATH = ""
_C.DATALOADER.IGNORE_INDEX = -1


# ---------------------------------------------------------------------------- #
# MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.WEIGHT = ""
_C.MODEL.DEVICE = "cuda"

_C.MODEL.RESNETS = CN()
# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1
# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True
# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1
_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHES = 200
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 10000
_C.SOLVER.TEST_PERIOD = 0
_C.SOLVER.SCHEDULE_TYPE = "poly"
_C.SOLVER.SHOW_IMAGE = False
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
# _C.SOLVER.IMS_PER_BATCH = 16




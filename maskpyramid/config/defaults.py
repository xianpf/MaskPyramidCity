# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Pengfei Xian. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "run/"
_C.MULTI_RUN = False
_C.TEST_ONLY = False


# ---------------------------------------------------------------------------- #
# DATALOADER
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_CLASSES = 19
_C.DATALOADER.BASE_SIZE = 2048
_C.DATALOADER.CROP_SIZE = 513
_C.DATALOADER.BATCH_SIZE_TRAIN = 4
_C.DATALOADER.BATCH_SIZE_TEST = 4
_C.DATALOADER.BATCH_SIZE_VAL = 4
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.DATASET = "cityscapes"
_C.DATALOADER.DATASET_PATH = ""
_C.DATALOADER.IGNORE_INDEX = -1


# ---------------------------------------------------------------------------- #
# Cudnn related params
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# ---------------------------------------------------------------------------- #
# MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.PRETRAINED = ''
_C.MODEL.WEIGHT = ""
_C.MODEL.DEVICE = "cuda"

_C.MODEL.RESNETS = CN()
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHES = 200
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.SCHEDULE_TYPE = "poly"
_C.SOLVER.SHOW_IMAGE = False
_C.SOLVER.SEMATIC_ONLY = False




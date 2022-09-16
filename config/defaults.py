
"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.DATASET = "kinetics"
_C.DATA.CATEGORY = 400
_C.DATA.PATH_TO_DATA_DIR = ""
_C.DATA.VFORMAT = "BTCHW"

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.5, 0.5, 0.5]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.5, 0.5, 0.5]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 224

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"
#_C.DATA.DECODING_BACKEND = "torchvision"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# List of input frame channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# ---------------------------------------------------------------------------- #
# MViTv2
# ---------------------------------------------------------------------------- #
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = False

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [3, 3]

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# Kernel size for Q, K, V pooling.
_C.MVIT.POOL_KVQ_KERNEL = (3, 3)

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = False

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = False

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = True

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = True


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# Using Label-Smooth
_C.TRAIN.LABEL_SMOOTH = 0.01

# Use Precised BN or Not
_C.TRAIN.PRECISE_BN = 0
_C.TRAIN.PBN_EPOCH = 1

# Batch Per GPU
_C.TRAIN.TRN_BATCH = 1
_C.TRAIN.VAL_BATCH = 1

# Learning Schedule
_C.TRAIN.LR_TYPE = "sgd"
_C.TRAIN.SCHEME = "step"
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0 
_C.TRAIN.GRADIENT_ACCUMULATION_STEPS = 10
_C.TRAIN.CLIP_GD = 1.0

# Learning Rate
_C.TRAIN.LR = 0.03 
_C.TRAIN.LR_STEPS = [10, 15, 18]


# ---------------------------------------------------------------------------- #
# Model options.
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.NET = "Swin"
_C.MODEL.BACKBONE = "swin_base_patch4_window7_224"

_C.MODEL.FOLD_DIV=0
_C.MODEL.S2_CLIP_CNT=[1, 1, 1, 1]
_C.MODEL.S3_CLIP_CNT=[1, 1, 1, 1]
_C.MODEL.S2_SKIP_LEVEL=[None, None, None, None]
_C.MODEL.S3_SKIP_LEVEL=[None, None, None, None]

_C.MODEL.S2_SHIFT_SELECT=[0,0,0,0]
_C.MODEL.SELECT_SE=[]
_C.MODEL.CycleMLP_SHIFT=""

_C.MODEL.ARCH = "i3d"
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow", "x3d"]

# Config for Split SE
_C.MODEL.SE_FOLD=1
_C.MODEL.SELECT_S1_MLP=[0,0,0,0,0,0,0]
_C.MODEL.SELECT_S2_MLP=[0,0,0,0]
_C.MODEL.SELECT_S3_MLP=[0,0,0,0]
_C.MODEL.SELECT_S2_ATTEN=[0,0,0,0]
_C.MODEL.SELECT_S3_ATTEN=[0,0,0,0]

# Config for MViT only
# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False
# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.0
# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Config for CvT only
_C.MODEL.NUM_CLASSES=1000
_C.MODEL.INIT_WEIGHTS=True
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.SPEC=CfgNode()
_C.MODEL.SPEC.INIT="trunc_norm"
_C.MODEL.SPEC.NUM_STAGES=3
_C.MODEL.SPEC.PATCH_SIZE=[7,3,3]
_C.MODEL.SPEC.PATCH_STRIDE=[4,2,2]
_C.MODEL.SPEC.PATCH_PADDING=[2,1,1]
_C.MODEL.SPEC.DIM_EMBED=[64,192,384]
_C.MODEL.SPEC.NUM_HEADS=[1,3,6]
_C.MODEL.SPEC.DEPTH=[1,2,10]
_C.MODEL.SPEC.MLP_RATIO=[4.0,4.0,4.0]
_C.MODEL.SPEC.ATTN_DROP_RATE=[0.0,0.0,0.0]
_C.MODEL.SPEC.DROP_RATE=[0.0,0.0,0.0]
_C.MODEL.SPEC.DROP_PATH_RATE=[0.0,0.0,0.1]
_C.MODEL.SPEC.QKV_BIAS=[True,True,True]
_C.MODEL.SPEC.CLS_TOKEN=[False,False,False]
_C.MODEL.SPEC.POS_EMBED=[False,False,False]
_C.MODEL.SPEC.QKV_PROJ_METHOD=['dw_bn','dw_bn','dw_bn']
_C.MODEL.SPEC.KERNEL_QKV=[3,3,3]
_C.MODEL.SPEC.PADDING_KV=[1,1,1]
_C.MODEL.SPEC.STRIDE_KV=[2,2,2]
_C.MODEL.SPEC.PADDING_Q=[1,1,1]
_C.MODEL.SPEC.STRIDE_Q=[1,1,1]

# ---------------------------------------------------------------------------- #
# Model options.
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


def get_cfg():
	"""
	Get a copy of the default config.
	"""
	return _C.clone()



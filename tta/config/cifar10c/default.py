from yacs.config import CfgNode as CN

# Default config
_C = CN()

# ---------- Model ---------- #
_C.MODEL = CN()
_C.MODEL.NAME = "wide_resnet28_10"
_C.MODEL.BACKEND = "robustbench"
_C.MODEL.PRETRAIN = "cifar10"

# ---------- Method ---------- #
_C.METHOD = CN()
_C.METHOD.NAME = "Source"

"""
<----- CfgNode for specific TTA methods.
"""
# ---------- TENT ---------- #

_C.METHOD.OPTIM_METHOD = "adam"
_C.METHOD.OPTIM_LR = 1e-4
_C.METHOD.OPTIM_BETA = 0.9  # Adam beta
_C.METHOD.OPTIM_MOMENTUM = 0.9  # Adam momentum
_C.METHOD.OPTIM_WD = 0.0  # L2 regularization (weight decay)

# ---------- MEMO ---------- #
# _C.METHOD.GROUP_NORM = 8
_C.METHOD.NORM_MEAN = [0.5, 0.5, 0.5]
_C.METHOD.NORM_STD = [0.5, 0.5, 0.5]
_C.METHOD.OPTIM_STEPS = 1
_C.METHOD.AUG_BATCH_SIZE = 4


# ---------- LAME ---------- #

# ---------- SHOT ---------- #


# ---------- Dataset ---------- #
_C.DATA = CN()

_C.DATA.DATASET_NAME = "Cifar10CDataset"  # Benchmark name
_C.DATA.NUM_SAMPLES = 10000
_C.DATA.BATCH_SIZE = 200
_C.DATA.NUM_WORKERS = 4
_C.DATA.SHIFT_TYPE = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
_C.DATA.SHIFT_SEVERITY = [5]


cfg = _C.clone()

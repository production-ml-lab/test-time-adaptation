from yacs.config import CfgNode as CN

# Default config
_C = CN()

# ---------- misc ---------- #
_C.MISC = CN()

# ---------- model ---------- #
_C.MODEL = CN()
_C.MODEL.NAME = "resnet26"
_C.MODEL.BACKEND = "custom"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.ADAPTATION = "Source"
_C.MODEL.PRETRAIN = "cifar10"

# ---------- Benchmark ---------- #
_C.DATA = CN()

_C.DATA.NAME = "CifarDataset"    # Benchmark name
_C.DATA.ROOT = "/data/cifar10c"
_C.DATA.BATCH_SIZE = 200
_C.DATA.NUM_WORKERS = 4

# ---------- Data shift ---------- #
_C.SHIFT = CN()

_C.SHIFT.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                'snow', 'frost', 'fog', 'brightness', 'contrast',
                'elastic_transform', 'pixelate', 'jpeg_compression']
_C.SHIFT.SEVERITY = [5, 4, 3, 2, 1]

# ---------- optimizer ---------- #
_C.OPTIM = CN()

_C.OPTIM.METHOD = 'adam'
_C.OPTIM.LR = 1e-4
_C.OPTIM.BETA = 0.9     # Adam beta
_C.OPTIM.MOMENTUM = 0.9     # Adam momentum
_C.OPTIM.WD = 0.0       # L2 regularization (weight decay)



"""
<----- CfgNode for specific TTA methods.
"""

# ---------- Tent ---------- #

# ---------- MEMO ---------- #

# ---------- LAME ---------- #

# ---------- SHOT ---------- #


"""
-----> CfgNode for specific TTA methods.
"""

cfg =  _C.clone()
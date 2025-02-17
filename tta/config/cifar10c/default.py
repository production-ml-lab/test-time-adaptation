from yacs.config import CfgNode as CN

# Default config
_C = CN()

# ---------- misc ---------- #
_C.MISC = CN()

# ---------- model ---------- #
_C.MODEL = CN()
_C.MODEL.ARCH = "ResNet18"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.ADAPTATION = "Source"

# ---------- Benchmark ---------- #
_C.DATA = CN()

_C.DATA.NAME = "cifar10"    # Benchmark name
_C.DATA.BATCH_SIZE = 200
_C.DATA.NUM_WORKERS = 4
_C.DATA.ROOT = "/data/cifar10-c"

# ---------- Data shift ---------- #
_C.SHIFT = CN()

_C.SHIFT.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                'snow', 'frost', 'fog', 'brightness', 'contrast',
                'elastic_transform', 'pixelate', 'jpeg_compression']

_C.SHIFT.SEVERITY = [5, 4, 3, 2, 1]


# ---------- optimizer ---------- #
_C.OPTIM = CN()

_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9     # Adam beta
_C.OPTIM.MOMENTUM = 0.9     # Adam momentum
_C.OPTIM.WD = 0.0       # L2 regularization (weight decay)



"""
<----- CfgNode for specific TTA methods.
"""

# ---------- Source ---------- #

# ---------- Tent ---------- #


"""
-----> CfgNode for specific TTA methods.
"""

# Config 객체 반환

cfg =  _C.clone()
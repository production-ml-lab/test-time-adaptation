import torch
import torchvision

from torch import optim
from yacs.config import CfgNode

from tta.config import cifar10c
from tta.model.resnet import build_resnet26


AVAILABLE_BACKEND = ['torchvision', 'custom']
AVAILABLE_OPTIM = ['sgd', 'adam']

def get_backbone(config: CfgNode) -> None:
    model_backend = config.MODEL.BACKEND
    assert model_backend in AVAILABLE_BACKEND

    if model_backend == 'custom':
        backbone = build_resnet26()
        return backbone

    if model_backend == 'torchvision':
        available_models = torchvision.models.list_models(module=torchvision.models)
        print(available_models)
        raise NotImplementedError

def get_optimizer(config: CfgNode, params) -> None:
    optim_method = config.OPTIM.METHOD
    assert optim_method in AVAILABLE_OPTIM

    if optim_method == 'sgd':
        optimizer = torch.optim.SGD()
    

if __name__ == "__main__":
    cfg = cifar10c
    get_backbone(cfg)
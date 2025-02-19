import torch

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class Tent(BaseMethod):

    def collect_params(self):
        return super().collect_params()
    
    def set_loss(self):
        return super().set_loss()

    def forward_and_adapt(self, x):
        return super().forward_and_adapt(x)

    @torch.no_grad()
    def predict(self, x):
        return super().predict(x)
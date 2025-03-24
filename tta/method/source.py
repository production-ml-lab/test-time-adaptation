import torch

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class Source(BaseMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def collect_params(self):
        # No params to update
        params = []
        names = []
        return params, names

    def set_loss_fn(self):
        return super().set_loss_fn()

    def forward_and_adapt(self, x):
        return super().forward_and_adapt(x)

    @torch.no_grad()
    def predict(self, x):
        return self.model(x.to(self.device)).argmax(1)

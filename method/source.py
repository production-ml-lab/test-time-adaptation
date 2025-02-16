from method import BaseMethod
from misc.registry import ADAPTATION_REGISTRY

@ADAPTATION_REGISTRY.register()
class Source(BaseMethod):
    def __init__(self, config):
        super().__init__(config=config)

    def collect_params(self):
        return super().collect_params()

    def set_optimizer(self):
        return super().set_optimizer()
    
    def set_loss(self):
        return super().set_loss()
    
    def forward_and_adapt(self, x):
        return super().forward_and_adapt(x)
    
    def predict(self, x):
        return super().predict(x)
    
    def reset(self):
        return super().reset()

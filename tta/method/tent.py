import torch
import torch.nn as nn

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Tent 논문의 엔트로피 함수 정의"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@ADAPTATION_REGISTRY.register()
class Tent(BaseMethod):
    def __init__(self, config):
        super().__init__(config)
        self.model.train()
        self._configure_model()
        self.params, _ = self.collect_params()
        self.optimizer = self.set_optimizer()

    def _configure_model(self):
        """
        Tent에서 제안한 바와 같이
        1) 전체 모델 파라미터를 먼저 전부 requires_grad=False
        2) BN의 scale, bias만 requires_grad=True
        3) 러닝 스탯을 안 쓰도록 설정(track_running_stats = False)
        """
        # 1) 전체 파라미터 고정
        for param in self.model.parameters():
            param.requires_grad = False

        # 2) BN만 업데이트할 수 있게 설정
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        # 3) train 모드
        self.model.train()

    def collect_params(self):
        """
        BN의 weight, bias만 모아서 optimizer에 등록
        """
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def set_loss(self):
        return self._entropy_loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        x = x.to(self.device)
        for _ in range(self.config.OPTIM.STEPS):
            outputs = self.model(x)
            loss = self.loss(outputs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _entropy_loss(self, logits):
        return softmax_entropy(logits).mean()

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        return self.model(x).argmax(1)
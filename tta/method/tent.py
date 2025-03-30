import torch
import torch.nn as nn
from copy import deepcopy

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Tent 논문의 엔트로피 함수 정의"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@ADAPTATION_REGISTRY.register()
class Tent(BaseMethod):
    def __init__(self, optim_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.optim_steps = optim_steps
        # self._configure_model()
        self.params, _ = self.collect_params()
        self.optimizer = self.set_optimizer()
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        self.episodic = True

    def forward(self, x):
        if self.model.training:
            outputs = self.forward_and_adapt(x)
        else:
            outputs = self.model(x)
        return outputs.argmax(1)

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

    def collect_params(self):
        """
        BN의 weight, bias만 모아서 optimizer에 등록
        """
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def set_loss_fn(self):
        return self._entropy_loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.model.train()
        x = x.to(self.device)
        outputs = self.model(x)
        for _ in range(self.optim_steps):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss(outputs)
            loss.backward()
            self.optimizer.step()
        return outputs

    def _entropy_loss(self, logits):
        return softmax_entropy(logits).mean()

    def reset(self):
        """Corruption 변경 시 초기 상태 복원"""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, x):
        return self.forward_and_adapt(x).argmax(1)

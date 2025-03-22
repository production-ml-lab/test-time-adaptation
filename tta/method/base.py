import logging
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from tta.model import load_resnet26, load_wide_resnet28_10


AVAILABLE_BACKEND = ["robustbench", "huggingface"]
AVAILABLE_MODEL = ["resnet26", "wide_resnet28_10"]
AVAILABLE_OPTIM = ["adam", "sgd"]


logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


class BaseMethod(ABC):
    def __init__(
        self,
        model_name: str = "wide_resnet28_10",
        model_backend: str = "robustbench",
        model_pretrain: str = "cifar10",
        optim_method: str = "adam",
        optim_lr: float = 1e-4,
        optim_beta: float = 0.9,
        optim_wd: float = 0.0,
        device: str = DEVICE,
        **kwargs,
    ) -> None:
        self.device = device
        # model
        self.model_backend = model_backend
        self.model_pretrain = model_pretrain
        self.model_name = model_name
        self.model = self.get_model()
        # optimizer
        self.params, param_names = self.collect_params()
        self.optim_method = optim_method
        self.optim_lr = optim_lr
        self.optim_beta = optim_beta
        self.optim_wd = optim_wd
        self.optimizer = self.set_optimizer()
        # loss
        self.loss = self.set_loss_fn()

    @abstractmethod
    def collect_params(self) -> List[nn.Parameter]:
        """Collect parameters that require gradients."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    @abstractmethod
    def set_loss_fn(self) -> None:
        """Define the loss function for the method."""
        return torch.nn.CrossEntropyLoss()

    @abstractmethod
    def forward_and_adapt(self, x: torch.Tensor) -> None:
        """Update model parameters using the current input."""
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Make a prediction for the current input without adaptation."""
        pass

    def get_model(self) -> None:
        assert self.model_backend in AVAILABLE_BACKEND
        assert self.model_name in AVAILABLE_MODEL

        if self.model_name == "resnet26":
            model = load_resnet26(
                pretrain=self.model_pretrain,
                device=self.device,
            )
        elif self.model_name == "wide_resnet28_10":
            model = load_wide_resnet28_10(
                backend=self.model_backend,
                pretrain=self.model_pretrain,
                device=self.device,
            )
        return model.to(self.device)

    def set_optimizer(self) -> torch.optim:
        """Define the optimizer for the method."""
        if len(self.params) == 0:
            return None

        optim_method = self.optim_method
        assert optim_method in AVAILABLE_OPTIM

        if optim_method == "adam":
            return torch.optim.Adam(
                self.params,
                lr=self.optim_lr,
                betas=(self.optim_beta, 0.999),
                weight_decay=self.optim_wd,
            )
        elif optim_method == "sgd":
            return torch.optim.SGD(self.params, lr=self.optim_lr)
        else:
            raise NotImplementedError

    def get_number_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"#Trainable/total parameters: {trainable:,}/{total:,} \t Ratio: {trainable / total * 100:.3f}% "
        )
        return trainable, total

    def reset(self) -> None:
        """Reset the model and optimizer state to the initial source state."""
        self.model = self.get_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.set_optimizer()

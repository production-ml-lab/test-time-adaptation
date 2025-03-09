import logging
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from tta.model import load_resnet26, load_wide_resnet28_10

AVAILABLE_BACKEND = ["robustbench", "huggingface"]
AVAILABLE_MODEL = ["resnet26", "wide_resnet28_10"]
AVAILABLE_OPTIM = ["adam"]


logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    def __init__(self, config) -> None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
        else:
            DEVICE = "cpu"
        self.device = DEVICE
        self.config = config
        self.model = self.get_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.set_optimizer()
        self.loss = self.set_loss()

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
    def set_loss(self) -> None:
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
        model_backend = self.config.MODEL.BACKEND
        model_pretrain = self.config.MODEL.PRETRAIN
        model_name = self.config.MODEL.NAME

        assert model_backend in AVAILABLE_BACKEND
        assert model_name in AVAILABLE_MODEL

        if model_name == "resnet26":
            model = load_resnet26(
                pretrain=model_pretrain,
                device=self.device,
            )
        elif model_name == "wide_resnet28_10":
            model = load_wide_resnet28_10(
                backend=model_backend,
                pretrain=model_pretrain,
                device=self.device,
            )
        return model.to(self.device)

    def set_optimizer(self) -> torch.optim:
        """Define the optimizer for the method."""
        if len(self.params) == 0:
            return None

        optim_method = self.config.OPTIM.METHOD
        assert optim_method in AVAILABLE_OPTIM

        if optim_method == "adam":
            return torch.optim.Adam(
                self.params,
                lr=self.config.OPTIM.LR,
                betas=(self.config.OPTIM.BETA, 0.999),
                weight_decay=self.config.OPTIM.WD,
            )
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
        self.optimizer = self.set_optimizer()  # <-- self.optmizer를 self.optimizer로 오타 수정

    # def reset(self) -> None:
    #     """Reset the model and optimizer state to the initial source state."""
    #     self.model = self.get_model()
    #     self.params, param_names = self.collect_params()
    #     self.optimzer = self.set_optimizer()

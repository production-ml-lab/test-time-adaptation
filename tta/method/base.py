import os
import torch
import torch.nn as nn
import torchvision
import logging
from abc import ABC, abstractmethod
from typing import List
from yacs.config import CfgNode

from tta.model.resnet import build_resnet26

AVAILABLE_BACKEND = ["torchvision", "custom"]
AVAILABLE_OPTIM = ["adam"]

logger = logging.getLogger(__name__)


class BaseMethod(ABC):
    def __init__(self, config) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        assert model_backend in AVAILABLE_BACKEND

        if model_backend == "custom":
            model = build_resnet26()

            if model_pretrain is not None:
                # Load pretrained model
                state_dict = torch.load(
                    os.path.abspath("tta/asset/resnet26_cifar10.pth"),
                    map_location=self.device,
                )
                model.load_state_dict(state_dict=state_dict, strict=True)

            return model

        elif model_backend == "torchvision":
            available_models = torchvision.models.list_models(module=torchvision.models)
            raise NotImplementedError

        else:
            raise NotImplementedError

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
        self.optimzer = self.set_optimizer()

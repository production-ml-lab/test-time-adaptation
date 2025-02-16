from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn

class BaseMethod(ABC):
    @abstractmethod
    def __init__(self, config) -> None:
        self.config = config
        self.model = config.MODEL.ARCH

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def collect_params(self) -> List[nn.Parameter]:
        """Collect parameters that require gradients."""
        pass

    @abstractmethod
    def set_optimizer(self) -> None:
        """Define the optimizer for the method."""
        pass

    @abstractmethod
    def set_loss(self) -> None:
        """Define the loss function for the method."""
        pass

    @abstractmethod
    def forward_and_adapt(self, x: torch.Tensor) -> None:
        """Update model parameters using the current input."""
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make a prediction for the current input without adaptation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the model and optimizer state to the initial source state."""
        pass
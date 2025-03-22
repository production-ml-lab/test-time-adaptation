import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def loss_shot(outputs):
    # (1) entropy
    ent_loss = softmax_entropy(outputs).mean(0)

    # (2) diversity
    softmax_out = F.softmax(outputs, dim=-1)
    msoftmax = softmax_out.mean(dim=0)
    ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

    # (3) pseudo label
    # adapt
    py, y_prime = F.softmax(outputs, dim=-1).max(1)
    flag = py > 0.9
    clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

    loss = ent_loss + 0.1 * clf_loss
    return loss


@ADAPTATION_REGISTRY.register()
class SHOT(BaseMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # collect params
    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms./
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names./
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    # set loss
    def set_loss_fn(self):
        return loss_shot

    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        optimizer = self.set_optimizer()
        x = x.to(self.device)
        # forward
        outputs = self.model(x)
        # adapt
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        self.model.eval()
        outputs = self.model(x)
        predictions = outputs.max(1)[1]  # [B]
        return predictions

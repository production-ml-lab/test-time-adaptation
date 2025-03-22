import torch
import torch.nn.functional as F

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY


class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int = 3, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float("inf")
    Y = (-unary).softmax(-1)  # [N, K]
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        if i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE)):
            # print(f"Converged in {i} iterations")
            break
        else:
            oldE = E
    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E


@ADAPTATION_REGISTRY.register()
class Lame(BaseMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.affinity = kNN_affinity()

    def collect_params(self):
        return super().collect_params()

    def set_loss_fn(self):
        return super().set_loss_fn()

    def forward_and_adapt(self, x):
        return super().forward_and_adapt(x)

    @torch.no_grad()
    def predict(self, x):
        x = x.to(self.device)
        feats = self.model.forward_feature(x)
        logits = self.model.forward(x)
        probas = F.softmax(logits, dim=1)
        feats = F.normalize(feats, p=2, dim=-1)  # [N, d]
        unary = -torch.log(probas + 1e-10)  # [N, K]
        kernel = self.affinity(feats)  # [N, N]
        Y = laplacian_optimization(unary, kernel)
        return Y.argmax(1)

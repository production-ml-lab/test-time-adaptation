from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.data.memo import memo_aug


@ADAPTATION_REGISTRY.register()
class MEMO(BaseMethod):
    """MEMO (Marginal Entropy Minimization with Online Adaptation) method."""

    def __init__(
        self,
        aug_batch_size: int,
        optim_steps: int,
        norm_mean: List[float],
        norm_std: List[float],
        **kwargs,
    ):
        """MEMO 초기화 및 모델 빌드"""
        super().__init__(**kwargs)
        self.optim_steps = optim_steps
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.aug_batch_size = aug_batch_size

        # source model의 state dict 저장
        self.source_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # 데이터 변환 정의
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

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

    def set_loss_fn(self):
        """Marginal entropy loss를 정의합니다."""

        def marginal_entropy(outputs):
            logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
            avg_logits = logits.logsumexp(dim=0) - torch.log(
                torch.tensor(logits.shape[0], dtype=torch.float)
            )
            min_real = torch.finfo(avg_logits.dtype).min
            avg_logits = torch.clamp(avg_logits, min=min_real)
            return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

        return marginal_entropy

    def forward_and_adapt(self, batch):
        """각 이미지마다 독립적으로 adaptation을 수행합니다."""
        if len(batch) == 3:  # Runner에서 (x, y, domain) 형태로 전달
            images = batch[0].to(self.device)
        else:  # 단일 텐서로 전달
            images = batch.to(self.device)

        predictions = []

        # 각 이미지에 대해 독립적으로 adaptation
        for img in images:
            # 매 이미지마다 source model로 초기화
            self.model.load_state_dict(self.source_state)
            optimizer = self.set_optimizer()

            # 단일 이미지를 PIL로 변환
            pil_img = self.tensor_to_pil(img.unsqueeze(0))

            # 단일 이미지에 대한 adaptation 수행
            self.model.train()
            for _ in range(self.optim_steps):
                # augmentation으로 batch 생성
                aug_imgs = [memo_aug(pil_img) for _ in range(self.aug_batch_size)]
                aug_imgs = torch.stack(aug_imgs).to(self.device)

                # adaptation step
                optimizer.zero_grad()
                outputs = self.model(aug_imgs)
                loss, _ = self.loss(outputs)
                loss.backward()
                optimizer.step()

            # 적응된 모델로 원본 이미지 예측
            self.model.eval()
            with torch.no_grad():
                pred = self.model(img.unsqueeze(0))
                predictions.append(pred)

        return torch.cat(predictions, dim=0)

    def tensor_to_pil(self, tensor):
        """텐서를 PIL 이미지로 변환"""
        # 차원 확인 및 처리
        if len(tensor.shape) == 3:
            # [C, H, W] -> [H, W, C]
            img = tensor.cpu().numpy().transpose(1, 2, 0)
        elif len(tensor.shape) == 4:
            # [B, C, H, W] -> [H, W, C]
            img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

        # 정규화 해제
        img = (img * self.norm_std) + self.norm_mean
        img = (img * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(img)

    @torch.no_grad()
    def predict(self, x):
        """adaptation된 모델로 배치 데이터에 대한 예측을 수행합니다.

        Args:
            x: torch.Tensor 형태의 입력 배치 데이터 [B, C, H, W]
        Returns:
            predictions: 예측된 클래스 레이블 [B]
        """
        self.model.eval()
        x = x.to(self.device)
        outputs = self.model(x)
        predictions = outputs.max(1)[1]  # [B]
        return predictions

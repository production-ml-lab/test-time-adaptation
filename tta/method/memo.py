import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from tta.method import BaseMethod
from tta.misc.registry import ADAPTATION_REGISTRY
from tta.utils.data.memo import memo_aug
from tta.utils.data.cifar import CifarDataset


@ADAPTATION_REGISTRY.register()
class MEMO(BaseMethod):
    """MEMO (Marginal Entropy Minimization with Online Adaptation) method."""

    def __init__(self, config):
        """MEMO 초기화 및 모델 빌드"""
        super().__init__(config)
        self.device = next(self.model.parameters()).device
        self.loss = self.set_loss()
        
        # source model의 state dict 저장
        self.source_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # 데이터 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.DATA.NORM.MEAN,
                std=self.config.DATA.NORM.STD
            )
        ])

    def set_loss(self):
        """Marginal entropy loss를 정의합니다."""
        def marginal_entropy(outputs):
            logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
            avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=torch.float))
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
            for _ in range(self.config.OPTIM.STEPS):
                # augmentation으로 batch 생성
                aug_imgs = [memo_aug(pil_img) for _ in range(self.config.DATA.BATCH_SIZE)]
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
        img = ((img * self.config.DATA.NORM.STD) + self.config.DATA.NORM.MEAN)
        img = (img * 255).clip(0, 255).astype('uint8')
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

    def set_optimizer(self):
        """MEMO는 SGD optimizer를 사용합니다."""
        return optim.SGD(self.params, lr=self.config.OPTIM.LR)
    
    def collect_params(self):
        """모든 모델 파라미터를 수집합니다."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    # def get_model(self):
    #     """ResNet26 with GroupNorm 모델을 생성합니다."""
    #     def gn_helper(planes):
    #         return nn.GroupNorm(self.config.MODEL.GROUP_NORM, planes)
            
    #     class BasicBlock(nn.Module):
    #         def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
    #             super(BasicBlock, self).__init__()
    #             self.downsample = downsample
    #             self.stride = stride
                
    #             self.bn1 = norm_layer(inplanes)
    #             self.relu1 = nn.ReLU(inplace=True)
    #             self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                
    #             self.bn2 = norm_layer(planes)
    #             self.relu2 = nn.ReLU(inplace=True)
    #             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

    #         def forward(self, x):
    #             residual = x
    #             residual = self.bn1(residual)
    #             residual = self.relu1(residual)
    #             residual = self.conv1(residual)

    #             residual = self.bn2(residual)
    #             residual = self.relu2(residual)
    #             residual = self.conv2(residual)

    #             if self.downsample is not None:
    #                 x = self.downsample(x)
    #             return x + residual

    #     class Downsample(nn.Module):
    #         def __init__(self, nIn, nOut, stride):
    #             super(Downsample, self).__init__()
    #             self.avg = nn.AvgPool2d(stride)
    #             assert nOut % nIn == 0
    #             self.expand_ratio = nOut // nIn

    #         def forward(self, x):
    #             x = self.avg(x)
    #             return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

    #     class ResNetCifar(nn.Module):
    #         def __init__(self, depth, width=1, channels=3, classes=10, norm_layer=nn.BatchNorm2d):
    #             super(ResNetCifar, self).__init__()
    #             assert (depth - 2) % 6 == 0  # depth is 6N+2
    #             self.N = (depth - 2) // 6
                
    #             # 첫 번째 convolution 레이어
    #             self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
    #             self.inplanes = 16
                
    #             # 레이어 구성
    #             self.layer1 = self._make_layer(norm_layer, 16 * width)
    #             self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
    #             self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
                
    #             # 출력 레이어
    #             self.bn = norm_layer(64 * width)
    #             self.relu = nn.ReLU(inplace=True)
    #             self.avgpool = nn.AvgPool2d(8)
    #             self.fc = nn.Linear(64 * width, classes)

    #         def _make_layer(self, norm_layer, planes, stride=1):
    #             downsample = None
    #             if stride != 1 or self.inplanes != planes:
    #                 downsample = Downsample(self.inplanes, planes, stride)
                
    #             layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
    #             self.inplanes = planes
    #             for _ in range(self.N - 1):
    #                 layers.append(BasicBlock(self.inplanes, planes, norm_layer))
    #             return nn.Sequential(*layers)

    #         def forward(self, x):
    #             x = self.conv1(x)
    #             x = self.layer1(x)
    #             x = self.layer2(x)
    #             x = self.layer3(x)
    #             x = self.bn(x)
    #             x = self.relu(x)
    #             x = self.avgpool(x)
    #             x = x.view(x.size(0), -1)
    #             x = self.fc(x)
    #             return x

    #     # ResNet26 모델 생성 및 반환
    #     model = ResNetCifar(
    #         depth=26,
    #         width=1, 
    #         channels=3,
    #         classes=self.config.MODEL.NUM_CLASSES,
    #         norm_layer=gn_helper
    #     ).to(self.device)
        
    #     return model

    # def collect_params(self):
    #     """모든 모델 파라미터를 수집합니다."""
    #     params = []
    #     names = []
    #     for nm, m in self.model.named_modules():
    #         for np, p in m.named_parameters():
    #             if np in ['weight', 'bias'] and p.requires_grad:
    #                 params.append(p)
    #                 names.append(f"{nm}.{np}")
    #     return params, names
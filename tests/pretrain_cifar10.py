import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 모델 정의된 파일 import
from tta.model.resnet import build_resnet26 

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 하이퍼파라미터 설정
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "resnet_cifar10.pth"

# CIFAR-10 데이터셋 변환 (Normalization 적용)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 데이터 증강 (좌우 반전)
    transforms.RandomCrop(32, padding=4),  # 데이터 증강 (랜덤 크롭)
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # CIFAR-10 정규화
])

# 데이터 로드
train_dataset = CIFAR10(root="test-time-adaptation/data", train=True, transform=transform, download=True)
test_dataset = CIFAR10(root="test-time-adaptation/data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 모델 초기화 및 설정
model = build_resnet26(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 모델 학습 함수
def train():
    logger.info("Starting Training...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate()
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # 베스트 모델 저장
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Best model saved at epoch {epoch+1} with accuracy {best_acc:.2f}%")

# 평가 함수
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    train()
    logger.info(f"Final model saved to {MODEL_SAVE_PATH}")
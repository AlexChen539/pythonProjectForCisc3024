import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


# SVHN 数据模块类，增加数据增强
class SVHNDataModule:
    def __init__(self, max_rotation=15, min_crop_size=24, max_aspect_ratio_change=0.1):
        # 定义训练集增强
        self.train_transform = A.Compose([
            A.Rotate(limit=max_rotation, p=0.5),
            A.RandomResizedCrop(height=32, width=32, scale=(min_crop_size / 32, 1.0),
                                ratio=(1 - max_aspect_ratio_change, 1 + max_aspect_ratio_change), p=0.5),
            A.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
            ToTensorV2()
        ])

        # 测试集仅进行归一化
        self.test_transform = A.Compose([
            A.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
            ToTensorV2()
        ])

    def load_data(self):

        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self._train_transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self._test_transform)


        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        return train_loader, val_loader, test_loader

    # 自定义转换函数，用于在 SVHN 数据集上应用 Albumentations
    def _train_transform(self, img):
        img = np.array(img)  # 将 PIL 图像转换为 NumPy 数组以适应 Albumentations
        return self.train_transform(image=img)['image']

    def _test_transform(self, img):
        img = np.array(img)
        return self.test_transform(image=img)['image']


# 定义小型 VGG 模型
class SmallVGG(nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 训练模型的函数，加入早停功能
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10, device='cuda'):
    model.train()
    device = torch.device(device)
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0  # 初始化早停计数器

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 训练过程
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.6f}')

        # 验证过程
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.6f}')

        # 早停检查
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), "small_vgg_svhn.pth")
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 绘制训练损失图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs with Early Stopping')
    plt.legend()
    plt.show()


# 评估模型的函数
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    num_classes = 10
    all_labels_onehot = np.eye(num_classes)[all_labels]
    all_probs = np.array(all_probs)

    micro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='micro')
    macro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro')

    print(f'Micro ROC AUC: {micro_roc_auc:.4f}')
    print(f'Macro ROC AUC: {macro_roc_auc:.4f}')


# 主函数
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据模块，传入自定义的增强参数
    data_module = SVHNDataModule(max_rotation=15, min_crop_size=24, max_aspect_ratio_change=0.1)
    train_loader, val_loader, test_loader = data_module.load_data()

    # 初始化模型、损失函数和优化器
    model = SmallVGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10, device=device)

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("small_vgg_svhn.pth"))
    evaluate_model(model, test_loader, device=device)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# SVHN 数据模块类
class SVHNDataModule:
    def __init__(self):
        # 数据预处理：将图像转换为Tensor并进行归一化，同时加入数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),     # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])

    def load_data(self):
        # 使用SVHN数据集并应用相应的变换
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self.train_transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self.test_transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        return train_loader, test_loader

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

# 训练模型的函数（加入早停法）
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, patience=5, device='cuda'):
    model.train()
    device = torch.device(device)
    epoch_losses = []
    best_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

        # 早停法逻辑
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # 绘制训练损失图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
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
    train_model(model, train_loader, criterion, optimizer, num_epochs=300, patience=10, device=device)
    # 初始化数据模块
    data_module = SVHNDataModule()
    train_loader, test_loader = data_module.load_data()

    # 初始化模型、损失函数和优化器
    model = SmallVGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=300, patience=10, device=device)

    # 评估模型
    evaluate_model(model, test_loader, device=device)

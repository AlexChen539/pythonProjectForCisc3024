import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import precision_score, recall_score
# 创建目录，如果不存在则自动创建
os.makedirs("ex4", exist_ok=True)

# SVHN 数据模块类
class SVHNDataModule:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])

    def load_data(self, batch_size):
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self.transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self.transform)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

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


# 修改训练模型函数，移除早停法的相关代码
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.train()
    device = torch.device(device)
    train_losses = []
    val_losses = []

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

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 评估模型在验证集上的损失
        val_loss = evaluate_model(model, val_loader, device, return_loss=True)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses
def evaluate_model(model, data_loader, device='cuda', return_loss=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))

    num_classes = 10
    all_labels_onehot = np.eye(num_classes)[all_labels]
    all_probs = np.array(all_probs)

    micro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='micro')
    macro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro')

    print(f'Accuracy: {accuracy * 100:.2f}%, Micro ROC AUC: {micro_roc_auc:.4f}, Macro ROC AUC: {macro_roc_auc:.4f}')

    if return_loss:
        return avg_loss
    else:
        return avg_loss, accuracy, micro_roc_auc, macro_roc_auc, all_labels, all_preds, all_probs

# 主函数，增加保存 Precision、Recall 和 Accuracy 柱状图的代码
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 30

    # 数据加载
    data_module = SVHNDataModule()
    train_loader, val_loader, test_loader = data_module.load_data(batch_size)

    # 定义优化器列表
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'AdamW': optim.AdamW
    }

    # 存储每个优化器的评估指标
    results = {}

    # 遍历不同的优化器
    for opt_name, opt_class in optimizers.items():
        print(f"\nUsing optimizer: {opt_name}")

        # 初始化模型、损失函数和优化器
        model = SmallVGG().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = opt_class(model.parameters(), lr=learning_rate)

        # 训练模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer,
                                               num_epochs=num_epochs, device=device)

        # 评估模型在测试集上的表现
        test_loss, accuracy, micro_roc_auc, macro_roc_auc, all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device)

        # 计算 Precision 和 Recall
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        precision = precision_score(all_labels, all_preds, average='weighted', labels=np.arange(10))
        recall = recall_score(all_labels, all_preds, average='weighted', labels=np.arange(10))

        # 保存当前优化器的指标
        results[opt_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

        # 绘制并保存 Train Loss 和 Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and Validation Loss with {opt_name}')
        plt.legend()
        plt.savefig(f"ex4/train_val_loss_{opt_name}.png")
        plt.close()

        # 绘制并保存 ROC 曲线
        fpr, tpr, _ = roc_curve(np.eye(10)[all_labels].ravel(), np.array(all_probs).ravel())
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {micro_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve with {opt_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"ex4/roc_curve_{opt_name}.png")
        plt.close()

        # 绘制并保存混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix with {opt_name}')
        plt.savefig(f"ex4/confusion_matrix_{opt_name}.png")
        plt.close()

    # 绘制并保存 Precision、Recall 和 Accuracy 的柱状对比图
    metrics = ['accuracy', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        values = [results[opt][metric] for opt in optimizers.keys()]
        plt.bar(np.arange(len(values)) + i*0.25, values, width=0.25, label=metric)

    plt.xticks(np.arange(len(optimizers)) + 0.25, optimizers.keys(), rotation=45)
    plt.ylabel('Scores')
    plt.title('Comparison of Precision, Recall, and Accuracy using Different Optimizers')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex4/comparison_bar_chart.png")
    plt.close()

if __name__ == "__main__":
    main()

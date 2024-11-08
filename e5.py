import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


os.makedirs("ex5", exist_ok=True)


class SVHNDataModule:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))
        ])

    def load_data(self):
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self.transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return train_loader, test_loader

# 定义小型 VGG 模型，支持不同的 Dropout 率
class SmallVGG(nn.Module):
    def __init__(self, dropout_rate_conv=0.25, dropout_rate_fc=0.5):
        super(SmallVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate_conv),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate_conv),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate_conv)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# 训练模型的函数，返回训练中的所有损失值
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()
    device = torch.device(device)
    train_losses = []

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
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}')

    return train_losses

# 评估模型的函数
def evaluate_model_metrics(model, test_loader, criterion, device='cuda'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return test_loss, accuracy, precision, recall, cm, all_labels, all_probs

# 生成 ROC 曲线的函数
def plot_roc_curve(all_labels, all_probs, dropout):
    plt.figure()
    for i in range(10):
        fpr, tpr, _ = roc_curve(np.array(all_labels) == i, np.array(all_probs)[:, i])
        auc_score = roc_auc_score(np.array(all_labels) == i, np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Dropout {dropout})')
    plt.legend()
    plt.savefig(f'ex5/roc_curve_dropout_{dropout}.png')
    plt.close()

# 主函数，测试不同 Dropout 率并保存图表
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_module = SVHNDataModule()
    train_loader, test_loader = data_module.load_data()
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    results = {}

    for dropout in dropout_rates:
        print(f"\nUsing Dropout Rate - Conv & FC: {dropout}")
        model = SmallVGG(dropout_rate_conv=dropout, dropout_rate_fc=dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        # 训练模型并记录损失
        train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)

        # 评估模型并获得所有预测的概率
        test_loss, accuracy, precision, recall, cm, all_labels, all_probs = evaluate_model_metrics(model, test_loader, criterion, device)

        results[dropout] = {
            "train_loss": train_losses[-1],
            "test_loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "cm": cm
        }

        # 绘制并保存训练过程的 Loss 曲线
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve (Dropout {dropout})")
        plt.legend()
        plt.savefig(f"ex5/loss_curve_dropout_{dropout}.png")
        plt.close()

        # 绘制并保存 ROC 曲线
        plot_roc_curve(all_labels, all_probs, dropout)

        # 保存混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Dropout {dropout})")
        plt.savefig(f"ex5/confusion_matrix_dropout_{dropout}.png")
        plt.close()

    # 绘制并保存不同 Dropout 率的指标柱状图
    metrics = ['train_loss', 'test_loss', 'accuracy', 'precision', 'recall']
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        values = [results[dropout][metric] for dropout in dropout_rates]
        plt.bar(np.arange(len(dropout_rates)) + i * 0.15, values, width=0.15, label=metric)

    plt.xticks(np.arange(len(dropout_rates)) + 0.3, [f"Dropout {d}" for d in dropout_rates], rotation=45)
    plt.ylabel('Scores')
    plt.title('Comparison of Metrics with Different Dropout Rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex5/metrics_comparison_bar_chart.png")
    plt.close()

if __name__ == "__main__":
    main()

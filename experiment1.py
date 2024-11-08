import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = 'ex1'
os.makedirs(output_dir, exist_ok=True)
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

# 训练模型的函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
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
        val_loss = evaluate_model(model, val_loader, device, return_loss=True)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses

# 评估模型的函数
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

def grid_search_hyperparameters(device='cuda'):
    learning_rates = [0.001, 0.0001]
    batch_sizes = [64, 128]
    num_epochs_list = [10, 50, 200]

    results = []

    # 遍历所有的超参数组合
    for lr, batch_size, num_epochs in itertools.product(learning_rates, batch_sizes, num_epochs_list):
        experiment_details = f"lr={lr}, batch_size={batch_size}, epochs={num_epochs}"
        print(f"Training with {experiment_details}")

        # 数据加载
        data_module = SVHNDataModule()
        train_loader, val_loader, test_loader = data_module.load_data(batch_size)

        # 初始化模型、损失函数和优化器
        model = SmallVGG().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

        # 评估模型在测试集上的表现
        test_loss, accuracy, micro_roc_auc, macro_roc_auc, all_labels, all_preds, all_probs = evaluate_model(model, test_loader, device)

        # 绘制 Train Loss 和 Validation Loss 并保存
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and Validation Loss for {experiment_details}')
        plt.legend()
        loss_plot_path = os.path.join(output_dir, f'loss_{lr}_{batch_size}_{num_epochs}.png')
        plt.savefig(loss_plot_path)
        plt.close()

        # 绘制 ROC 曲线并保存
        fpr, tpr, _ = roc_curve(np.eye(10)[all_labels].ravel(), np.array(all_probs).ravel())
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {micro_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {experiment_details}')
        plt.legend(loc="lower right")
        roc_plot_path = os.path.join(output_dir, f'roc_{lr}_{batch_size}_{num_epochs}.png')
        plt.savefig(roc_plot_path)
        plt.close()

        # 绘制混淆矩阵并保存
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {experiment_details}')
        cm_plot_path = os.path.join(output_dir, f'confusion_matrix_{lr}_{batch_size}_{num_epochs}.png')
        plt.savefig(cm_plot_path)
        plt.close()

        # 计算精确率和召回率
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)

        # 记录实验结果
        results.append({
            'Learning Rate': lr,
            'Batch Size': batch_size,
            'Epochs': num_epochs,
            'Test Loss': test_loss,
            'Accuracy': accuracy,
            'Micro ROC AUC': micro_roc_auc,
            'Macro ROC AUC': macro_roc_auc,
            'Precision': np.nanmean(precision),
            'Recall': np.nanmean(recall)
        })

    # 绘制精确率-召回率柱状图
    metrics_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    metrics_df[['Precision', 'Recall']].plot(kind='bar', figsize=(14, 7))
    plt.xticks(
        ticks=range(len(metrics_df)),
        labels=metrics_df[['Learning Rate', 'Batch Size', 'Epochs']].apply(
            lambda row: f"lr={row['Learning Rate']}, bs={row['Batch Size']}, ep={row['Epochs']}", axis=1
        ),
        rotation=45,
        ha='right'
    )
    plt.title('Precision and Recall by Hyperparameter Settings')
    plt.ylabel('Score')
    precision_recall_plot_path = os.path.join(output_dir, 'precision_recall_comparison.png')
    plt.savefig(precision_recall_plot_path)
    plt.close()

    # 将结果保存到 ex1 目录中的 CSV 文件
    results_csv_path = os.path.join(output_dir, 'hyperparameter_tuning_results.csv')
    metrics_df.to_csv(results_csv_path, index=False)
    print(f"结果已保存到 {results_csv_path}")
# 主函数
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_search_hyperparameters(device=device)

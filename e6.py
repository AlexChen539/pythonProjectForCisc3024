import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 创建保存图表的文件夹
output_dir = 'ex6'
os.makedirs(output_dir, exist_ok=True)

# SVHN 数据模块类，增加数据增强
class SVHNDataModule:
    def __init__(self, max_rotation=15, min_crop_size=24, max_aspect_ratio_change=0.1):
        self.train_transform = A.Compose([
            A.Rotate(limit=max_rotation, p=0.5),
            A.RandomResizedCrop(height=32, width=32, scale=(min_crop_size / 32, 1.0),
                                ratio=(1 - max_aspect_ratio_change, 1 + max_aspect_ratio_change), p=0.5),
            A.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
            ToTensorV2()
        ])

    def load_data(self):
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=self._train_transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=self._test_transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return train_loader, test_loader

    def _train_transform(self, img):
        img = np.array(img)
        return self.train_transform(image=img)['image']

    def _test_transform(self, img):
        img = np.array(img)
        return self.test_transform(image=img)['image']

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

class DeepVGG(nn.Module):
    def __init__(self):
        super(DeepVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.train()
    epoch_losses = []
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
    return epoch_losses

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    roc_auc = roc_auc_score(np.eye(10)[all_labels], np.array(all_probs), average='macro')
    return all_labels, all_preds, roc_auc, accuracy, all_probs

def plot_results(train_losses, labels, title, filename):
    plt.figure()
    for losses, label in zip(train_losses, labels):
        plt.plot(range(1, len(losses) + 1), losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_roc(all_labels, all_probs, labels, filename):
    plt.figure()
    for label, probs in zip(labels, all_probs):
        fpr, tpr, _ = roc_curve(np.eye(10)[all_labels[0]].ravel(), probs.ravel())
        plt.plot(fpr, tpr, label=f'{label} ROC')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve Comparison')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, labels, filename):
    for label, preds in zip(labels, all_preds):
        cm = confusion_matrix(all_labels[0], preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f'Confusion Matrix - {label}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, f'{filename}_{label}.png'))
        plt.close()

def plot_bar_chart(metrics, labels, metric_names, filename):
    x = np.arange(len(labels))
    width = 0.2
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, metric, width, label=metric_names[i])
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Comparison of Precision, Recall, and Accuracy')
    plt.xticks(x + width, labels)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def compare_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_module = SVHNDataModule()
    train_loader, test_loader = data_module.load_data()

    model_small, model_deep = SmallVGG().to(device), DeepVGG().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_small = optim.Adam(model_small.parameters(), lr=0.0001)
    optimizer_deep = optim.Adam(model_deep.parameters(), lr=0.0001)

    # Reduced number of epochs for testing
    small_losses = train_model(model_small, train_loader, criterion, optimizer_small, num_epochs=5, device=device)
    deep_losses = train_model(model_deep, train_loader, criterion, optimizer_deep, num_epochs=5, device=device)

    plot_results([small_losses, deep_losses], ['SmallVGG', 'DeepVGG'], 'Training Loss Curve Comparison', 'train_loss_comparison.png')

    small_labels, small_preds, small_roc, small_acc, small_probs = evaluate_model(model_small, test_loader, device=device)
    deep_labels, deep_preds, deep_roc, deep_acc, deep_probs = evaluate_model(model_deep, test_loader, device=device)

    plot_roc([small_labels, deep_labels], [small_probs, deep_probs], ['SmallVGG', 'DeepVGG'], 'roc_comparison.png')
    plot_confusion_matrix([small_labels], [small_preds, deep_preds], ['SmallVGG', 'DeepVGG'], 'confusion_matrix')

    plot_bar_chart([[small_acc, deep_acc], [small_roc, deep_roc]], ['SmallVGG', 'DeepVGG'], ['Accuracy', 'ROC AUC'], 'metric_comparison.png')

if __name__ == "__main__":
    compare_models()
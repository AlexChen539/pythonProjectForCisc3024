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
import os
from sklearn.metrics import precision_score, recall_score
# SVHN 数据模块类，接受数据增强参数
class SVHNDataModule:
    def __init__(self, augmentation=None):
        if augmentation is None:
            augmentation = []

        # 基础数据处理
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970))]

        # 应用数据增强
        transform_list = augmentation + transform_list
        self.transform = transforms.Compose(transform_list)

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
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda'):
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

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print(f'Accuracy: {accuracy * 100:.2f}%, Micro ROC AUC: {micro_roc_auc:.4f}, Macro ROC AUC: {macro_roc_auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    if return_loss:
        return avg_loss
    else:
        return avg_loss, accuracy, micro_roc_auc, macro_roc_auc, precision, recall, all_labels, all_probs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 20

    augmentations = {
        'No Augmentation': [],
        'Horizontal Flip': [transforms.RandomHorizontalFlip(p=0.5)],
        'Random Rotation': [transforms.RandomRotation(15)],
        'Random Crop and Flip': [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
        'Color Jitter': [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)],
    }

    results = {}  # Store metrics for each augmentation

    for aug_name, aug_transforms in augmentations.items():
        print(f"\nUsing augmentation: {aug_name}")

        data_module = SVHNDataModule(augmentation=aug_transforms)
        train_loader, val_loader, test_loader = data_module.load_data(batch_size)

        model = SmallVGG().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer,
                                               num_epochs=num_epochs, device=device)

        test_loss, accuracy, micro_roc_auc, macro_roc_auc, precision, recall, all_labels, all_probs = evaluate_model(model, test_loader, device)

        results[aug_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Train and Validation Loss with {aug_name}')
        plt.legend()
        plt.savefig(f"ex3/train_val_loss_{aug_name.replace(' ', '_')}.png")
        plt.close()

        fpr, tpr, _ = roc_curve(np.eye(10)[all_labels].ravel(), np.array(all_probs).ravel())
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {micro_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve with {aug_name}')
        plt.legend(loc="lower right")
        plt.savefig(f"ex3/roc_curve_{aug_name.replace(' ', '_')}.png")
        plt.close()

    # Plot and save the comparison bar chart
    metrics = ['accuracy', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        values = [results[aug][metric] for aug in augmentations.keys()]
        plt.bar(np.arange(len(values)) + i*0.25, values, width=0.25, label=metric)

    plt.xticks(np.arange(len(augmentations)) + 0.25, augmentations.keys(), rotation=45)
    plt.ylabel('Scores')
    plt.title('Comparison of Precision, Recall, and Accuracy using Different Augmentations')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ex3/comparison_bar_chart.png")
    plt.close()


if __name__ == "__main__":
    main()

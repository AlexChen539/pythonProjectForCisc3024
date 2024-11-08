import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the smaller version of VGG (8 layers, max channel size 32) for the SVHN dataset
class SmallVGG(nn.Module):
    def __init__(self):
        super(SmallVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # 3 input channels (RGB), 8 output channels
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Increase to 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 input channels, 32 output channels
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Keep 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8

            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Keep 32 output channels
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Keep 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),  # Flatten: 32 channels, 4x4 image
            nn.ReLU(),
            nn.Linear(256, 10)  # Output layer: 10 classes (digits 0-9)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers
        x = self.fc_layers(x)
        return x

# Example usage
if __name__ == "__main__":
    model = SmallVGG()
    print(model)
import torch
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move data to the device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Evaluation function
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # ROC and AUC metrics (micro and macro)
    num_classes = 10
    all_labels_onehot = np.eye(num_classes)[all_labels]  # One-hot encode labels
    all_probs = np.array(all_probs)

    # Compute ROC and AUC for each class
    micro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='micro')
    macro_roc_auc = roc_auc_score(all_labels_onehot, all_probs, average='macro')

    print(f'Micro ROC AUC: {micro_roc_auc:.4f}')
    print(f'Macro ROC AUC: {macro_roc_auc:.4f}')

# Example usage
if __name__ == "__main__":
    # Assume train_loader and test_loader are already defined
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate model, loss function, and optimizer
    model = SmallVGG().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10, device=device)

    # Evaluate the model
    evaluate_model(model, test_loader, device=device)

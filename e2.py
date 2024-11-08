import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = 'ex2'
os.makedirs(output_dir, exist_ok=True)

# SVHN Data Module
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

# Define Small VGG Model
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

# Training function with optional early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=None, device='cuda', early_stop_start_epoch=30):
    model.train()
    device = torch.device(device)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

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

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, device, return_loss=True)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping check (enabled if `patience` is set)
        if patience is not None and epoch >= early_stop_start_epoch:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        elif patience is None:
            torch.save(model.state_dict(), 'best_model_no_early_stopping.pth')

    return train_losses, val_losses

# Evaluation function
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

    if return_loss:
        return avg_loss
    else:
        return avg_loss, accuracy, all_labels, all_preds, all_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 50

    # Load data
    data_module = SVHNDataModule()
    train_loader, val_loader, test_loader = data_module.load_data(batch_size)

    # Model, criterion, and optimizer
    model = SmallVGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model with early stopping
    print("Training with early stopping:")
    patience = 5
    train_losses_es, val_losses_es = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience, device=device
    )

    # Test the model trained with early stopping
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss_es, accuracy_es, all_labels_es, all_preds_es, _ = evaluate_model(model, test_loader, device)

    # Train model without early stopping
    print("\nTraining without early stopping:")
    model = SmallVGG().to(device)  # Reinitialize model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses_no_es, val_losses_no_es = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=None, device=device
    )

    # Test the model trained without early stopping
    model.load_state_dict(torch.load('best_model_no_early_stopping.pth'))
    test_loss_no_es, accuracy_no_es, all_labels_no_es, all_preds_no_es, _ = evaluate_model(model, test_loader, device)

    # Plot training and validation losses for both cases
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_es, label='Train Loss (Early Stopping)')
    plt.plot(val_losses_es, label='Validation Loss (Early Stopping)')
    plt.plot(train_losses_no_es, label='Train Loss (No Early Stopping)')
    plt.plot(val_losses_no_es, label='Validation Loss (No Early Stopping)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'train_val_loss_comparison.png'))
    plt.close()

    # Print results for both cases
    print(f"Test Accuracy with Early Stopping: {accuracy_es * 100:.2f}%")
    print(f"Test Accuracy without Early Stopping: {accuracy_no_es * 100:.2f}%")

if __name__ == "__main__":
    main()

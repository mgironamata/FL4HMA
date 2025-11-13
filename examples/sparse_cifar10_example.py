"""
Minimal CIFAR-10 sparse masking example with vanilla CNN.

This example demonstrates:
1. Loading CIFAR-10 dataset
2. Creating sparse labels by masking
3. Training a simple CNN with sparse supervision
4. Evaluating performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SparseCIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with sparse labeling."""
    
    def __init__(self, root: str = './data', train: bool = True, 
                 sparsity: float = 0.1, transform=None, download: bool = True):
        """
        Args:
            root: Root directory for data
            train: Whether to load train or test split
            sparsity: Fraction of labels to keep (0.1 = 10% labeled)
            transform: Data transformations
            download: Whether to download dataset
        """
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.sparsity = sparsity
        self.sparse_mask = self._create_sparse_mask()
        
    def _create_sparse_mask(self) -> np.ndarray:
        """Create random mask for sparse labeling."""
        np.random.seed(42)  # For reproducibility
        mask = np.random.random(len(self.cifar10)) < self.sparsity
        print(f"Created sparse mask: {mask.sum()}/{len(mask)} samples labeled ({self.sparsity*100:.1f}%)")
        return mask
    
    def __len__(self) -> int:
        return len(self.cifar10)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        image, label = self.cifar10[idx]
        is_labeled = self.sparse_mask[idx]
        
        # Return -1 for unlabeled samples
        sparse_label = label if is_labeled else -1
        
        return image, sparse_label, is_labeled


class VanillaCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes: int = 10):
        super(VanillaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Conv block 2  
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(model: nn.Module, train_loader: DataLoader, 
                device: torch.device, epochs: int = 10) -> list:
    """Train the CNN with sparse labels."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        labeled_samples = 0
        
        for batch_idx, (data, targets, is_labeled) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            is_labeled = is_labeled.to(device)
            
            # Only use labeled samples for training
            labeled_mask = is_labeled & (targets != -1)
            
            if labeled_mask.sum() == 0:
                continue  # Skip batch if no labeled samples
                
            labeled_data = data[labeled_mask]
            labeled_targets = targets[labeled_mask]
            
            optimizer.zero_grad()
            outputs = model(labeled_data)
            loss = criterion(outputs, labeled_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            labeled_samples += labeled_mask.sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Labeled samples: {labeled_mask.sum()}')
        
        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.4f}, '
              f'Total labeled samples used: {labeled_samples}')
    
    return train_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, 
                  device: torch.device) -> Tuple[float, float]:
    """Evaluate the model on test set."""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets, _ in test_loader:  # Ignore is_labeled for test
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total


def plot_training_curve(losses: list):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_sparse_samples(dataset: SparseCIFAR10Dataset, num_samples: int = 8):
    """Visualize some sparse labeled samples."""
    
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    sample_count = 0
    for idx in range(len(dataset)):
        if sample_count >= num_samples:
            break
            
        image, label, is_labeled = dataset[idx]
        
        # Convert tensor to numpy for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np + 1) / 2  # Denormalize if using standard normalization
        
        axes[sample_count].imshow(image_np)
        title = f"{'Labeled' if is_labeled else 'Unlabeled'}: "
        title += f"{classes[label] if label != -1 else 'Unknown'}"
        axes[sample_count].set_title(title, fontsize=10)
        axes[sample_count].axis('off')
        
        sample_count += 1
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training and evaluation loop."""
    
    print("=" * 60)
    print("CIFAR-10 Sparse Labeling with Vanilla CNN")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets with different sparsity levels
    sparsity = 0.1  # Use only 10% of labels
    print(f"\nCreating sparse datasets with {sparsity*100}% labeled data...")
    
    train_dataset = SparseCIFAR10Dataset(
        root='./data', train=True, sparsity=sparsity, 
        transform=transform, download=True
    )
    
    test_dataset = SparseCIFAR10Dataset(
        root='./data', train=False, sparsity=1.0,  # Test set fully labeled
        transform=transform, download=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Visualize some samples
    print("\nVisualizing sparse labeled samples...")
    visualize_sparse_samples(train_dataset)
    
    # Create model
    model = VanillaCNN(num_classes=10).to(device)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses = train_model(model, train_loader, device, epochs=5)
    
    # Plot training curve
    plot_training_curve(train_losses)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, correct, total = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\nTraining completed!")
    
    # Save model
    torch.save(model.state_dict(), 'sparse_cifar10_cnn.pth')
    print("Model saved as 'sparse_cifar10_cnn.pth'")


if __name__ == "__main__":
    main()
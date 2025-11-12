"""
Sparse Pixel Image-to-Image CNN Example.

This example demonstrates:
1. Image-to-image tasks with sparse input pixels (masked images)
2. Sparse pixel-level supervision (only some output pixels have labels)
3. U-Net-like CNN for pixel-level reconstruction/prediction
4. Realistic federated learning scenario for remote sensing/climate data

This is relevant for:
- Satellite imagery with cloud cover (sparse inputs)
- Limited ground truth measurements (sparse pixel labels)
- Climate data reconstruction
- Remote sensing applications
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SparsePixelDataset(Dataset):
    """
    Dataset that creates sparse pixel labels for image-to-image tasks.
    
    Simulates scenarios where:
    - Input images have missing/masked pixels (clouds, sensor failures)
    - Target labels are only available for sparse pixel locations (field measurements)
    """
    
    def __init__(self, root: str = './data', train: bool = True, 
                 input_sparsity: float = 0.3, output_sparsity: float = 0.1, 
                 transform=None, download: bool = True):
        """
        Args:
            root: Data directory
            train: Whether to use training or test set
            input_sparsity: Fraction of input pixels to keep (rest are masked to 0)
            output_sparsity: Fraction of output pixels that have target labels
            transform: Data transforms
            download: Whether to download CIFAR-10
        """
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
        self.input_sparsity = input_sparsity
        self.output_sparsity = output_sparsity
        
        print(f"Created SparsePixelDataset:")
        print(f"  Input sparsity: {input_sparsity*100:.1f}% pixels visible")
        print(f"  Output sparsity: {output_sparsity*100:.1f}% pixels labeled")
        print(f"  Dataset size: {len(self.cifar10)} samples")
        
    def __len__(self) -> int:
        return len(self.cifar10)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            sparse_input: Image with masked pixels [C, H, W]
            sparse_target: Target with only some pixels labeled [C, H, W] 
            input_mask: Binary mask showing which input pixels are visible [H, W]
            output_mask: Binary mask showing which output pixels are labeled [H, W]
        """
        image, _ = self.cifar10[idx]  # Ignore class label for image-to-image task
        
        # Create random masks (different for each sample)
        np.random.seed(idx)  # For reproducible masks per sample
        
        # Input mask: which pixels are visible in the input
        input_mask = torch.rand(image.shape[1], image.shape[2]) < self.input_sparsity
        
        # Output mask: which pixels have target labels  
        output_mask = torch.rand(image.shape[1], image.shape[2]) < self.output_sparsity
        
        # Create sparse input (mask out some pixels)
        sparse_input = image.clone()
        sparse_input[:, ~input_mask] = 0.0  # Set masked pixels to 0
        
        # Create sparse target (only some pixels have labels)
        sparse_target = image.clone()
        sparse_target[:, ~output_mask] = -1.0  # Mark unlabeled pixels as -1
        
        return sparse_input, sparse_target, input_mask.float(), output_mask.float()


class UNetCNN(nn.Module):
    """
    U-Net-like CNN for sparse pixel-to-pixel prediction.
    
    Suitable for image reconstruction, completion, and sparse supervision tasks.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 32):
        super(UNetCNN, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)  # 8 = 4 + 4 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)  # 4 = 2 + 2 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)  # 2 = 1 + 1 (skip)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        enc1 = self.enc1(x)      # [B, 32, 32, 32]
        enc2 = self.enc2(self.pool(enc1))  # [B, 64, 16, 16]
        enc3 = self.enc3(self.pool(enc2))  # [B, 128, 8, 8]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))  # [B, 256, 4, 4]
        
        # Decoder path with skip connections
        up3 = self.upconv3(bottleneck)  # [B, 128, 8, 8]
        if up3.shape != enc3.shape:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))  # [B, 128, 8, 8]
        
        up2 = self.upconv2(dec3)  # [B, 64, 16, 16]
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # [B, 64, 16, 16]
        
        up1 = self.upconv1(dec2)  # [B, 32, 32, 32]
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))  # [B, 32, 32, 32]
        
        # Final output
        output = self.final_conv(dec1)  # [B, 3, 32, 32]
        return torch.sigmoid(output)  # Ensure output is in [0, 1] range


def sparse_pixel_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                     output_mask: torch.Tensor) -> torch.Tensor:
    """
    Loss function that only considers labeled pixels.
    
    Args:
        predictions: Model predictions [B, C, H, W]
        targets: Target images [B, C, H, W] (unlabeled pixels = -1)
        output_mask: Binary mask [B, H, W] indicating which pixels are labeled
        
    Returns:
        Average loss over labeled pixels only
    """
    batch_size = predictions.size(0)
    total_loss = 0.0
    total_pixels = 0
    
    for b in range(batch_size):
        # Get labeled pixel mask for this sample
        labeled_mask = output_mask[b].bool()  # [H, W]
        
        if labeled_mask.sum() > 0:
            # Extract predictions and targets for labeled pixels only
            pred_labeled = predictions[b][:, labeled_mask]  # [C, N_labeled]
            target_labeled = targets[b][:, labeled_mask]    # [C, N_labeled]
            
            # Compute MSE loss for labeled pixels
            loss = F.mse_loss(pred_labeled, target_labeled, reduction='sum')
            total_loss += loss
            total_pixels += labeled_mask.sum().item() * predictions.size(1)  # N_pixels * N_channels
    
    # Return average loss per labeled pixel
    return total_loss / max(1, total_pixels)


def train_sparse_pixel_model(model: nn.Module, train_loader: DataLoader, 
                            device: torch.device, epochs: int = 5) -> list:
    """Train the sparse pixel model."""
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_losses = []
    
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_labeled_pixels = 0
        
        for batch_idx, (sparse_input, sparse_target, input_mask, output_mask) in enumerate(train_loader):
            sparse_input = sparse_input.to(device)
            sparse_target = sparse_target.to(device) 
            output_mask = output_mask.to(device)
            
            optimizer.zero_grad()
            predictions = model(sparse_input)
            loss = sparse_pixel_loss(predictions, sparse_target, output_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_labeled_pixels = output_mask.sum().item()
            total_labeled_pixels += batch_labeled_pixels
            
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx:3d}: Loss={loss.item():.6f}, '
                      f'Labeled pixels in batch={int(batch_labeled_pixels)}')
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Total labeled pixels used: {total_labeled_pixels}")
        print("-" * 40)
    
    return train_losses


def visualize_sparse_reconstruction(sparse_input: torch.Tensor, sparse_target: torch.Tensor, 
                                  predictions: torch.Tensor, input_mask: torch.Tensor,
                                  output_mask: torch.Tensor, num_samples: int = 4):
    """Visualize sparse pixel reconstruction results."""
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, num_samples*4))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, sparse_input.size(0))):
        # Convert tensors to numpy and handle different formats
        sparse_inp = sparse_input[i].permute(1, 2, 0).cpu().numpy()
        sparse_tgt = sparse_target[i].permute(1, 2, 0).cpu().numpy()
        prediction = predictions[i].permute(1, 2, 0).cpu().detach().numpy()
        inp_mask = input_mask[i].cpu().numpy()
        out_mask = output_mask[i].cpu().numpy()
        
        # Create visualizations
        # 1. Original sparse input
        axes[i, 0].imshow(np.clip(sparse_inp, 0, 1))
        axes[i, 0].set_title(f'Sparse Input\n({inp_mask.mean()*100:.1f}% visible)', fontsize=11)
        axes[i, 0].axis('off')
        
        # 2. Input mask
        axes[i, 1].imshow(inp_mask, cmap='Blues', alpha=0.8)
        axes[i, 1].set_title('Input Mask\n(Blue = Visible)', fontsize=11)
        axes[i, 1].axis('off')
        
        # 3. Sparse target (only show labeled pixels)
        target_vis = sparse_tgt.copy()
        target_vis[target_vis == -1] = 0  # Set unlabeled pixels to black
        axes[i, 2].imshow(np.clip(target_vis, 0, 1))
        axes[i, 2].set_title(f'Sparse Target\n({out_mask.mean()*100:.1f}% labeled)', fontsize=11)
        axes[i, 2].axis('off')
        
        # 4. Output mask
        axes[i, 3].imshow(out_mask, cmap='Reds', alpha=0.8)
        axes[i, 3].set_title('Target Mask\n(Red = Labeled)', fontsize=11)
        axes[i, 3].axis('off')
        
        # 5. Model prediction
        axes[i, 4].imshow(np.clip(prediction, 0, 1))
        axes[i, 4].set_title('Full Prediction', fontsize=11)
        axes[i, 4].axis('off')
        
        # 6. Prediction on labeled pixels only (for comparison with target)
        pred_masked = prediction.copy()
        pred_masked[~np.stack([out_mask]*3, axis=2).astype(bool)] = 0
        axes[i, 5].imshow(np.clip(pred_masked, 0, 1))
        axes[i, 5].set_title('Pred on Labeled\nPixels Only', fontsize=11)
        axes[i, 5].axis('off')
    
    plt.suptitle('Sparse Pixel Image-to-Image Reconstruction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def evaluate_sparse_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluate model on test set."""
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_labeled_pixels = 0
    
    with torch.no_grad():
        for sparse_input, sparse_target, input_mask, output_mask in test_loader:
            sparse_input = sparse_input.to(device)
            sparse_target = sparse_target.to(device)
            output_mask = output_mask.to(device)
            
            predictions = model(sparse_input)
            loss = sparse_pixel_loss(predictions, sparse_target, output_mask)
            
            total_loss += loss.item()
            total_samples += 1
            total_labeled_pixels += output_mask.sum().item()
    
    avg_loss = total_loss / total_samples
    avg_labeled_per_sample = total_labeled_pixels / total_samples
    
    return avg_loss, avg_labeled_per_sample


def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 60)
    print("Sparse Pixel Image-to-Image CNN")
    print("=" * 60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 8
    input_sparsity = 0.4   # 40% of input pixels visible
    output_sparsity = 0.15  # 15% of output pixels labeled
    
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Input sparsity: {input_sparsity*100}% pixels visible")
    print(f"  Output sparsity: {output_sparsity*100}% pixels labeled")
    print()
    
    # Data transforms (keep values in [0,1] range)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization - keep in [0,1] for easier visualization
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SparsePixelDataset(
        root='./data', train=True,
        input_sparsity=input_sparsity,
        output_sparsity=output_sparsity,
        transform=transform, download=True
    )
    
    test_dataset = SparsePixelDataset(
        root='./data', train=False,
        input_sparsity=input_sparsity,
        output_sparsity=output_sparsity,
        transform=transform, download=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    # Create model
    model = UNetCNN(in_channels=3, out_channels=3, base_filters=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    print()
    
    # Train model
    train_losses = train_sparse_pixel_model(model, train_loader, device, num_epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, avg_labeled_pixels = evaluate_sparse_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Average labeled pixels per test sample: {avg_labeled_pixels:.1f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', linewidth=2, markersize=6)
    plt.title('Training Loss - Sparse Pixel Reconstruction', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss (MSE on labeled pixels)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Visualize results on test data
    print("\nGenerating visualizations...")
    model.eval()
    with torch.no_grad():
        # Get a batch from test set
        test_batch = next(iter(test_loader))
        sparse_input, sparse_target, input_mask, output_mask = test_batch
        sparse_input = sparse_input.to(device)
        
        predictions = model(sparse_input)
        
        # Visualize results
        visualize_sparse_reconstruction(
            sparse_input, sparse_target, predictions,
            input_mask, output_mask, num_samples=4
        )
    
    # Save model
    model_path = 'sparse_pixel_unet.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("This model demonstrates sparse pixel supervision relevant for:")
    print("  • Satellite imagery with cloud cover")
    print("  • Climate data with sparse measurements") 
    print("  • Remote sensing applications")
    print("  • Federated learning with heterogeneous data availability")
    print("=" * 60)


if __name__ == "__main__":
    main()
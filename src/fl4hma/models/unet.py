import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder path with skip connections
        up3 = self.upconv3(bottleneck)
        if up3.shape != enc3.shape:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        output = self.final_conv(dec1)
        return torch.sigmoid(output)


def sparse_pixel_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_mask: torch.Tensor,
) -> torch.Tensor:
    """Loss function that only considers labelled pixels.

    Parameters
    ----------
    predictions : Tensor [B, C, H, W]
    targets : Tensor [B, C, H, W]  (unlabelled pixels = -1)
    output_mask : Tensor [B, H, W]  binary mask of labelled pixels

    Returns
    -------
    Average MSE loss over labelled pixels only.
    """
    batch_size = predictions.size(0)
    total_loss = predictions.new_zeros(())  # scalar tensor on correct device/dtype
    total_pixels = 0

    for b in range(batch_size):
        labeled_mask = output_mask[b].bool()
        if labeled_mask.sum() > 0:
            pred_labeled = predictions[b][:, labeled_mask]
            target_labeled = targets[b][:, labeled_mask]
            total_loss = total_loss + F.mse_loss(pred_labeled, target_labeled, reduction='sum')
            total_pixels += labeled_mask.sum().item() * predictions.size(1)

    return total_loss / max(1, total_pixels)

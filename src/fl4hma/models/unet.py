import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Schlemper et al., 2019).

    Learns a spatial attention map from the gating signal (decoder) and the
    skip connection (encoder), then uses it to suppress irrelevant encoder
    features before concatenation.

    Args:
        F_g: Number of channels in the gating signal (upsampled decoder path).
        F_l: Number of channels in the skip connection (encoder path).
        F_int: Number of intermediate channels for the attention computation.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Args:
            g: Gating signal from the decoder (upsampled), shape (B, F_g, H, W).
            x: Skip connection from the encoder, shape (B, F_l, H, W).

        Returns:
            Attention-weighted skip connection, same shape as ``x``.
        """
        att = F.relu(self.W_g(g) + self.W_x(x), inplace=True)
        att = self.psi(att)
        return x * att


class UNetCNN(nn.Module):
    """
    U-Net-like CNN for sparse pixel-to-pixel prediction.

    Suitable for image reconstruction, completion, and sparse supervision tasks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 32,
        output_activation: Optional[str] = None,
        use_attention: bool = True,
    ):
        """Args:
            output_activation: None (default) for linear output, appropriate when targets
                are z-score normalised (both temperature and log-precipitation).
                'softplus' or 'relu' only if targets are guaranteed non-negative
                (e.g. min-max scaled precipitation).
            use_attention: If True, add attention gates on every skip connection
                (Attention U-Net). Recommended for sparse supervision tasks.
        """
        super(UNetCNN, self).__init__()
        if output_activation not in (None, 'relu', 'softplus'):
            raise ValueError(f"output_activation must be None, 'relu', or 'softplus', got {output_activation!r}")
        self.output_activation = output_activation
        self.use_attention = use_attention

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

        # Attention gates (one per skip connection, coarsest to finest)
        if use_attention:
            self.att3 = AttentionGate(
                F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2
            )
            self.att2 = AttentionGate(
                F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters
            )
            self.att1 = AttentionGate(
                F_g=base_filters, F_l=base_filters, F_int=max(base_filters // 2, 1)
            )

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

        # Decoder path with (optionally attended) skip connections
        up3 = self.upconv3(bottleneck)
        if up3.shape != enc3.shape:
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        skip3 = self.att3(g=up3, x=enc3) if self.use_attention else enc3
        dec3 = self.dec3(torch.cat([up3, skip3], dim=1))

        up2 = self.upconv2(dec3)
        if up2.shape != enc2.shape:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        skip2 = self.att2(g=up2, x=enc2) if self.use_attention else enc2
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))

        up1 = self.upconv1(dec2)
        if up1.shape != enc1.shape:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        skip1 = self.att1(g=up1, x=enc1) if self.use_attention else enc1
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))

        output = self.final_conv(dec1)
        if self.output_activation == 'relu':
            return F.relu(output)
        if self.output_activation == 'softplus':
            return F.softplus(output)
        return output


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

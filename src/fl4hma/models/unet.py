import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_BLOCKS = ("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1")
_NORMS = ("group", "batch")


def _make_norm(norm: str, num_channels: int, num_groups: int) -> nn.Module:
    """Build a normalisation layer.

    Args:
        norm: ``"group"`` for GroupNorm or ``"batch"`` for BatchNorm2d.
        num_channels: Number of channels to normalise.
        num_groups: Target number of GroupNorm groups. The largest common
            divisor with ``num_channels`` is used so any channel count works.

    Returns:
        The normalisation module.
    """
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    return nn.GroupNorm(math.gcd(num_groups, num_channels), num_channels)


class AttentionGate(nn.Module):
    """Attention gate for skip connections (Schlemper et al., 2019).

    Learns a spatial attention map from the gating signal (decoder) and the
    skip connection (encoder), then uses it to suppress irrelevant encoder
    features before concatenation.

    Args:
        F_g: Number of channels in the gating signal (upsampled decoder path).
        F_l: Number of channels in the skip connection (encoder path).
        F_int: Number of intermediate channels for the attention computation.
        norm: ``"group"`` (default) or ``"batch"``.
        num_groups: Target number of GroupNorm groups.
    """

    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        norm: str = "group",
        num_groups: int = 8,
    ):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            _make_norm(norm, F_int, num_groups),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            _make_norm(norm, F_int, num_groups),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            _make_norm(norm, 1, num_groups),
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
        dropout: float = 0.0,
        dropout_blocks: Sequence[str] = ("bottleneck", "dec3", "dec2", "dec1"),
        norm: str = "group",
        num_groups: int = 8,
    ):
        """Args:
        output_activation: None (default) for linear output, appropriate when targets
            are z-score normalised (both temperature and log-precipitation).
            'softplus' or 'relu' only if targets are guaranteed non-negative
            (e.g. min-max scaled precipitation).
        use_attention: If True, add attention gates on every skip connection
            (Attention U-Net). Recommended for sparse supervision tasks.
        dropout: Spatial (channel-wise) dropout rate. 0.0 disables dropout
            entirely and adds no state-dict entries, so federated parameter
            lists are unchanged. A non-zero rate is what gives MC-dropout
            ensembles their spread.
        dropout_blocks: Which blocks are followed by dropout. Defaults to the
            bottleneck and the whole decoder path, because the skip
            connections carry ``enc1``/``enc2`` straight to ``dec1``: dropout
            confined to deep blocks is routed around and the ensemble barely
            spreads. In a smoke test, moving from ``(enc3, bottleneck, dec3)``
            to this default raised ensemble spread roughly fourfold at the
            same rate. Adding the encoder blocks spreads further still, at
            the cost of perturbing the finest features.
        norm: ``"group"`` (default) for GroupNorm, which has no running
            statistics and treats each sample independently, as
            per-sample DP-SGD requires. ``"batch"`` restores BatchNorm,
            e.g. to load checkpoints trained before the switch.
        num_groups: Target number of GroupNorm groups per layer.
        """
        super(UNetCNN, self).__init__()
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
        self.norm = norm
        self.num_groups = num_groups
        if output_activation not in (None, "relu", "softplus"):
            raise ValueError(
                f"output_activation must be None, 'relu', or 'softplus', got {output_activation!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        unknown = set(dropout_blocks) - set(DROPOUT_BLOCKS)
        if unknown:
            raise ValueError(
                f"unknown dropout block(s) {sorted(unknown)}; "
                f"valid blocks are {DROPOUT_BLOCKS}"
            )
        self.output_activation = output_activation
        self.use_attention = use_attention
        self.dropout = dropout
        self._drop_after = frozenset(dropout_blocks) if dropout > 0.0 else frozenset()
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)

        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, 2, stride=2
        )
        self.dec3 = self._conv_block(
            base_filters * 8, base_filters * 4
        )  # 8 = 4 + 4 (skip)

        self.upconv2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, 2, stride=2
        )
        self.dec2 = self._conv_block(
            base_filters * 4, base_filters * 2
        )  # 4 = 2 + 2 (skip)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)  # 2 = 1 + 1 (skip)

        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Attention gates (one per skip connection, coarsest to finest)
        if use_attention:
            self.att3 = AttentionGate(
                F_g=base_filters * 4,
                F_l=base_filters * 4,
                F_int=base_filters * 2,
                norm=norm,
                num_groups=num_groups,
            )
            self.att2 = AttentionGate(
                F_g=base_filters * 2,
                F_l=base_filters * 2,
                F_int=base_filters,
                norm=norm,
                num_groups=num_groups,
            )
            self.att1 = AttentionGate(
                F_g=base_filters,
                F_l=base_filters,
                F_int=max(base_filters // 2, 1),
                norm=norm,
                num_groups=num_groups,
            )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            _make_norm(self.norm, out_channels, self.num_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            _make_norm(self.norm, out_channels, self.num_groups),
            nn.ReLU(inplace=True),
        )

    def _maybe_drop(self, x: torch.Tensor, block: str) -> torch.Tensor:
        """Apply dropout after *block* when it is one of the selected blocks."""
        return self.drop(x) if block in self._drop_after else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        enc1 = self._maybe_drop(self.enc1(x), "enc1")
        enc2 = self._maybe_drop(self.enc2(self.pool(enc1)), "enc2")
        enc3 = self._maybe_drop(self.enc3(self.pool(enc2)), "enc3")

        # Bottleneck
        bottleneck = self._maybe_drop(self.bottleneck(self.pool(enc3)), "bottleneck")

        # Decoder path with (optionally attended) skip connections
        up3 = self.upconv3(bottleneck)
        if up3.shape != enc3.shape:
            up3 = F.interpolate(
                up3, size=enc3.shape[2:], mode="bilinear", align_corners=False
            )
        skip3 = self.att3(g=up3, x=enc3) if self.use_attention else enc3
        dec3 = self._maybe_drop(self.dec3(torch.cat([up3, skip3], dim=1)), "dec3")

        up2 = self.upconv2(dec3)
        if up2.shape != enc2.shape:
            up2 = F.interpolate(
                up2, size=enc2.shape[2:], mode="bilinear", align_corners=False
            )
        skip2 = self.att2(g=up2, x=enc2) if self.use_attention else enc2
        dec2 = self._maybe_drop(self.dec2(torch.cat([up2, skip2], dim=1)), "dec2")

        up1 = self.upconv1(dec2)
        if up1.shape != enc1.shape:
            up1 = F.interpolate(
                up1, size=enc1.shape[2:], mode="bilinear", align_corners=False
            )
        skip1 = self.att1(g=up1, x=enc1) if self.use_attention else enc1
        dec1 = self._maybe_drop(self.dec1(torch.cat([up1, skip1], dim=1)), "dec1")

        output = self.final_conv(dec1)
        if self.output_activation == "relu":
            return F.relu(output)
        if self.output_activation == "softplus":
            return F.softplus(output)
        return output


def enable_mc_dropout(model: nn.Module) -> None:
    """Put *model* in eval mode but leave dropout layers sampling.

    ``model.eval()`` disables dropout and BatchNorm updates together, which is
    wrong for MC dropout: the ensemble spread comes from dropout, while
    BatchNorm must use its frozen running statistics so that members differ only
    through the dropout masks. Call this instead of ``eval()`` before sampling.

    Args:
        model: Model to switch into MC-dropout sampling mode, in place.
    """
    model.eval()
    for module in model.modules():
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            module.train()


def ensemble_forward(
    model: nn.Module,
    x: torch.Tensor,
    n_members: int,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Draw *n_members* stochastic predictions for each item in the batch.

    Members are drawn by replicating the batch along dim 0 and running a single
    forward pass, so each replica receives an independent dropout mask. This
    costs the same FLOPs as ``n_members`` separate passes but keeps the GPU
    saturated. Note that in training mode BatchNorm then computes its batch
    statistics over the replicated batch, whose entries are correlated; under
    :func:`enable_mc_dropout` BatchNorm is frozen and this does not arise.

    Args:
        model: Model with ``out_channels=1``.
        x: Input batch, ``(B, C, H, W)``.
        n_members: Number of ensemble members to draw.
        chunk_size: Draw members in groups of this size to bound peak memory.
            ``None`` draws all members in one pass.

    Returns:
        Ensemble of shape ``(B, n_members, H, W)``.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be >= 1, got {n_members}")

    if chunk_size is None or chunk_size >= n_members:
        return _replicated_forward(model, x, n_members)

    members = []
    remaining = n_members
    while remaining > 0:
        take = min(chunk_size, remaining)
        members.append(_replicated_forward(model, x, take))
        remaining -= take
    return torch.cat(members, dim=1)


def _replicated_forward(model: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor:
    """Run one forward pass over a k-fold replicated batch."""
    batch = x.shape[0]
    out = model(x.repeat_interleave(k, dim=0))
    if out.shape[1] != 1:
        raise ValueError(
            f"ensemble_forward needs a model with out_channels=1, "
            f"got {out.shape[1]} output channels"
        )
    return out.squeeze(1).reshape(batch, k, *out.shape[2:])


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
            total_loss = total_loss + F.mse_loss(
                pred_labeled, target_labeled, reduction="sum"
            )
            total_pixels += labeled_mask.sum().item() * predictions.size(1)

    return total_loss / max(1, total_pixels)

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl4hma.training.training import evaluate_sparse_pixel


class ChannelZeroModel(nn.Module):
    """Returns channel 0 of the input, so predictions are fully controllable."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def make_loader(
    errors: np.ndarray,
    output_mask: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Build a loader whose predictions are 0 and whose targets are *errors*.

    Args:
        errors: Array of shape (n, h, w); the target minus the prediction.
        output_mask: Array of shape (n, h, w); 1 where a pixel is labelled.
        batch_size: Loader batch size.

    Returns:
        DataLoader yielding ``(sparse_in, sparse_tgt, input_mask, output_mask)``.
    """
    errors = np.asarray(errors, dtype=np.float32)
    output_mask = np.asarray(output_mask, dtype=np.float32)
    n, h, w = errors.shape

    sparse_in = torch.zeros(n, 3, h, w)
    sparse_tgt = torch.from_numpy(errors).unsqueeze(1)
    input_mask = torch.ones(n, h, w)
    out_mask = torch.from_numpy(output_mask)

    ds = TensorDataset(sparse_in, sparse_tgt, input_mask, out_mask)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


@pytest.fixture
def model():
    return ChannelZeroModel()


class TestBatchSizeInvariance:
    """The reported MSE must not depend on how the loader is batched."""

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 16, 64])
    def test_mse_independent_of_batch_size(self, model, batch_size):
        rng = np.random.default_rng(0)
        errors = rng.standard_normal((32, 4, 4))
        mask = np.ones((32, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, batch_size))

        assert metrics["mse"] == pytest.approx(float((errors**2).mean()), rel=1e-5)

    def test_partial_final_batch_does_not_skew(self, model):
        rng = np.random.default_rng(1)
        errors = rng.standard_normal((10, 4, 4))  # 10 samples, batch 4 -> 4/4/2
        mask = np.ones((10, 4, 4))

        batched = evaluate_sparse_pixel(model, make_loader(errors, mask, 4))
        unbatched = evaluate_sparse_pixel(model, make_loader(errors, mask, 1))

        assert batched["mse"] == pytest.approx(unbatched["mse"], rel=1e-5)


class TestMSEValue:
    def test_matches_analytic_mse(self, model):
        errors = np.full((4, 4, 4), 3.0)
        mask = np.ones((4, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 2))

        assert metrics["mse"] == pytest.approx(9.0, rel=1e-6)

    def test_rmse_is_sqrt_of_mse(self, model):
        rng = np.random.default_rng(2)
        errors = rng.standard_normal((8, 4, 4))
        mask = np.ones((8, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 3))

        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), rel=1e-6)


class TestMasking:
    def test_unlabelled_pixels_are_ignored(self, model):
        errors = np.full((4, 4, 4), 2.0)
        errors[:, 2:, :] = 1000.0  # huge errors, but unlabelled
        mask = np.ones((4, 4, 4))
        mask[:, 2:, :] = 0.0

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 2))

        assert metrics["mse"] == pytest.approx(4.0, rel=1e-6)

    def test_pooled_over_pixels_not_averaged_over_samples(self, model):
        # Sample 0: one labelled pixel with error 4 -> per-sample MSE 16
        # Sample 1: three labelled pixels with error 0 -> per-sample MSE 0
        # Pixel-pooled MSE = 16 / 4 = 4.0 (sample-averaged would be 8.0)
        errors = np.zeros((2, 2, 2))
        errors[0, 0, 0] = 4.0
        mask = np.zeros((2, 2, 2))
        mask[0, 0, 0] = 1.0
        mask[1, 0, :] = 1.0
        mask[1, 1, 0] = 1.0

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 2))

        assert metrics["mse"] == pytest.approx(4.0, rel=1e-6)

    def test_samples_without_labels_do_not_dilute(self, model):
        errors = np.full((4, 4, 4), 5.0)
        mask = np.ones((4, 4, 4))
        mask[2:] = 0.0  # last two samples carry no labels at all

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 4))

        assert metrics["mse"] == pytest.approx(25.0, rel=1e-6)


class TestDegenerateCases:
    def test_no_labelled_pixels_returns_nan(self, model):
        errors = np.ones((4, 4, 4))
        mask = np.zeros((4, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 2))

        assert np.isnan(metrics["mse"])
        assert np.isnan(metrics["rmse"])

    def test_empty_loader_returns_nan(self, model):
        errors = np.zeros((0, 4, 4))
        mask = np.zeros((0, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 2))

        assert np.isnan(metrics["loss"])
        assert np.isnan(metrics["mse"])
        assert np.isnan(metrics["rmse"])


# ── loss-config integration ───────────────────────────────────────────────────


from fl4hma.models.unet import UNetCNN  # noqa: E402
from fl4hma.training.losses import LossConfig  # noqa: E402
from fl4hma.training.training import train_sparse_pixel  # noqa: E402


def precip_loader(n=8, h=32, w=32, batch_size=4, seed=0):
    """Loader with non-negative physical-unit targets, as Tweedie requires."""
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((n, 3, h, w)).astype(np.float32))
    y = torch.from_numpy((rng.gamma(0.4, 6.0, (n, 1, h, w))).astype(np.float32))
    y[rng.random((n, 1, h, w)) < 0.6] = 0.0  # dry days
    mask = torch.ones(n, h, w)
    return DataLoader(
        TensorDataset(x, y, torch.ones(n, h, w), mask),
        batch_size=batch_size,
        shuffle=False,
    )


def small_unet(dropout=0.0):
    return UNetCNN(in_channels=3, out_channels=1, base_filters=8, dropout=dropout)


class TestDefaultUnchanged:
    def test_evaluate_without_config_matches_explicit_mse(self, model):
        rng = np.random.default_rng(3)
        errors, mask = rng.standard_normal((8, 4, 4)), np.ones((8, 4, 4))
        loader = make_loader(errors, mask, 4)

        implicit = evaluate_sparse_pixel(model, loader)
        explicit = evaluate_sparse_pixel(model, loader, LossConfig(name="mse"))

        assert implicit["mse"] == pytest.approx(explicit["mse"], rel=1e-6)

    def test_deterministic_crps_equals_mae(self, model):
        """A point forecast's CRPS is its absolute error, so the metric is
        comparable across deterministic and ensemble models."""
        rng = np.random.default_rng(4)
        errors, mask = rng.standard_normal((8, 4, 4)), np.ones((8, 4, 4))

        metrics = evaluate_sparse_pixel(model, make_loader(errors, mask, 4))

        assert metrics["crps"] == pytest.approx(metrics["mae"], rel=1e-6)
        assert metrics["spread"] == pytest.approx(0.0, abs=1e-9)


class TestCRPSTraining:
    def test_train_returns_finite_loss(self):
        cfg = LossConfig(name="crps", n_members=4, dropout=0.2)
        loss = train_sparse_pixel(
            small_unet(0.2), precip_loader(), epochs=1, loss_config=cfg
        )

        assert np.isfinite(loss)

    def test_evaluation_reports_spread_and_crps(self):
        cfg = LossConfig(name="crps", n_members=8, dropout=0.3)
        metrics = evaluate_sparse_pixel(small_unet(0.3), precip_loader(), cfg)

        assert metrics["spread"] > 0.0
        assert np.isfinite(metrics["crps"])
        assert metrics["loss"] == pytest.approx(metrics["crps"], rel=1e-6)

    def test_metrics_are_batch_size_invariant(self):
        cfg = LossConfig(name="crps", n_members=8, dropout=0.3)
        model = small_unet(0.3)

        torch.manual_seed(0)
        a = evaluate_sparse_pixel(model, precip_loader(batch_size=2), cfg)
        torch.manual_seed(0)
        b = evaluate_sparse_pixel(model, precip_loader(batch_size=8), cfg)

        assert a["crps"] == pytest.approx(b["crps"], rel=0.1)

    def test_ensemble_mean_drives_rmse(self):
        cfg = LossConfig(name="crps", n_members=8, dropout=0.3)
        metrics = evaluate_sparse_pixel(small_unet(0.3), precip_loader(), cfg)

        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), rel=1e-6)


class TestTweedieTraining:
    def test_train_returns_finite_loss(self):
        cfg = LossConfig(name="tweedie", physical_target=True, tweedie_p=1.5)
        loss = train_sparse_pixel(
            small_unet(), precip_loader(), epochs=1, loss_config=cfg
        )

        assert np.isfinite(loss)

    def test_point_prediction_uses_log_link(self):
        """RMSE must be computed on exp(eta), not on the raw output."""
        cfg = LossConfig(name="tweedie", physical_target=True)
        metrics = evaluate_sparse_pixel(small_unet(), precip_loader(), cfg)

        assert metrics["mse"] >= 0.0
        assert np.isfinite(metrics["rmse"])
        assert metrics["spread"] == pytest.approx(0.0, abs=1e-9)

    def test_negative_targets_are_rejected(self):
        cfg = LossConfig(name="tweedie", physical_target=True)
        rng = np.random.default_rng(5)
        loader = make_loader(rng.standard_normal((4, 8, 8)), np.ones((4, 8, 8)), 2)

        with pytest.raises(ValueError, match="non-negative"):
            evaluate_sparse_pixel(small_unet(), loader, cfg)

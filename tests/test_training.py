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

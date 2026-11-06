import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl4hma.federation.differential_privacy import (
    DPAccountant,
    DPConfig,
    clip_model_update,
    dp_train_sparse_pixel,
    local_dp_epsilon,
    per_sample_gradients,
    privatise_gradients,
)
from fl4hma.models.unet import UNetCNN, sparse_pixel_loss
from fl4hma.training.training import train_sparse_pixel


def _batch(n=3, size=16, seed=0, full_mask=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, size, size, generator=g)
    y = torch.randn(n, 1, size, size, generator=g)
    if full_mask:
        mask = torch.ones(n, size, size)
    else:
        mask = (torch.rand(n, size, size, generator=g) > 0.5).float()
    return x, y, mask


def _model(seed=0, **kwargs):
    torch.manual_seed(seed)
    return UNetCNN(in_channels=4, out_channels=1, base_filters=4, **kwargs)


def _loader(x, y, mask, batch_size):
    ds = TensorDataset(x, y, torch.ones_like(mask), mask)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TestClipModelUpdate:
    def test_clips_to_norm(self):
        original = [np.zeros(4)]
        updated = [np.array([3.0, 4.0, 0.0, 0.0])]
        clipped = clip_model_update(original, updated, clip_norm=1.0)
        np.testing.assert_allclose(np.linalg.norm(clipped[0]), 1.0)

    def test_small_update_unchanged(self):
        original = [np.zeros(2)]
        updated = [np.array([0.3, 0.4])]
        clipped = clip_model_update(original, updated, clip_norm=1.0)
        np.testing.assert_allclose(clipped[0], updated[0])

    def test_mask_excludes_entries_from_norm(self):
        original = [np.zeros(2), np.zeros(1)]
        updated = [np.array([0.3, 0.4]), np.array([1e5])]
        clipped = clip_model_update(
            original, updated, clip_norm=1.0, mask=[True, False]
        )
        np.testing.assert_allclose(clipped[0], updated[0])

    def test_masked_out_entries_passed_through(self):
        original = [np.zeros(2), np.array([7.0])]
        updated = [np.array([30.0, 40.0]), np.array([9.0])]
        clipped = clip_model_update(
            original, updated, clip_norm=1.0, mask=[True, False]
        )
        np.testing.assert_allclose(clipped[1], updated[1])
        np.testing.assert_allclose(np.linalg.norm(clipped[0]), 1.0)


class TestPerSampleGradients:
    def test_matches_autograd_per_sample(self):
        model = _model()
        x, y, mask = _batch()
        grads, _, _ = per_sample_gradients(model, x, y, mask)
        for i in range(len(x)):
            model.zero_grad()
            sparse_pixel_loss(
                model(x[i : i + 1]), y[i : i + 1], mask[i : i + 1]
            ).backward()
            for name, p in model.named_parameters():
                torch.testing.assert_close(grads[name][i], p.grad, atol=1e-5, rtol=1e-4)

    def test_pooled_loss_matches_batch_loss(self):
        model = _model()
        x, y, mask = _batch()
        _, losses, counts = per_sample_gradients(model, x, y, mask)
        pooled = (losses * counts).sum() / counts.sum()
        with torch.no_grad():
            expected = sparse_pixel_loss(model(x), y, mask)
        torch.testing.assert_close(pooled, expected)

    def test_unlabelled_sample_has_zero_gradient(self):
        model = _model()
        x, y, mask = _batch()
        mask[1] = 0
        grads, losses, counts = per_sample_gradients(model, x, y, mask)
        assert counts[1] == 0
        assert losses[1] == 0
        for g in grads.values():
            assert torch.isfinite(g).all()
            assert g[1].abs().max() == 0

    def test_rejects_batch_norm(self):
        model = _model(norm="batch")
        x, y, mask = _batch()
        with pytest.raises(ValueError, match="BatchNorm"):
            per_sample_gradients(model, x, y, mask)


def _norm(grads, i=None):
    parts = [g if i is None else g[i] for g in grads.values()]
    return torch.sqrt(sum((p**2).sum() for p in parts))


class TestPrivatiseGradients:
    def test_no_clip_no_noise_is_mean(self):
        grads = {"w": torch.tensor([[1.0, 0.0], [0.0, 3.0]])}
        out = privatise_gradients(grads, max_norm=1e9, noise_std=0.0)
        torch.testing.assert_close(out["w"], torch.tensor([0.5, 1.5]))

    def test_each_sample_clipped_to_max_norm(self):
        grads = {"a": torch.tensor([[3.0], [0.3]]), "b": torch.tensor([[4.0], [0.4]])}
        out = privatise_gradients(grads, max_norm=1.0, noise_std=0.0)
        # sample 0 (norm 5) scaled to norm 1; sample 1 (norm 0.5) unchanged
        torch.testing.assert_close(out["a"], torch.tensor([(0.6 + 0.3) / 2]))
        torch.testing.assert_close(out["b"], torch.tensor([(0.8 + 0.4) / 2]))

    def test_clip_uses_norm_across_all_parameters(self):
        grads = {"a": torch.tensor([[0.8]]), "b": torch.tensor([[0.6]])}
        out = privatise_gradients(grads, max_norm=0.5, noise_std=0.0)
        torch.testing.assert_close(_norm(out), torch.tensor(0.5))

    def test_noise_std_scaled_by_batch_size(self):
        batch_size, dim = 4, 200_000
        grads = {"w": torch.zeros(batch_size, dim)}
        gen = torch.Generator().manual_seed(0)
        out = privatise_gradients(grads, max_norm=1.0, noise_std=2.0, generator=gen)
        assert out["w"].std().item() == pytest.approx(2.0 / batch_size, rel=0.02)

    def test_zero_gradient_sample_is_safe(self):
        grads = {"w": torch.zeros(2, 3)}
        out = privatise_gradients(grads, max_norm=1.0, noise_std=0.0)
        assert torch.isfinite(out["w"]).all()


class TestDPTrainSparsePixel:
    def test_private_gradient_matches_batch_gradient_without_clip_or_noise(self):
        model = _model()
        x, y, mask = _batch(n=4, full_mask=True)
        grads, _, _ = per_sample_gradients(model, x, y, mask)
        private = privatise_gradients(grads, max_norm=1e9, noise_std=0.0)
        model.zero_grad()
        sparse_pixel_loss(model(x), y, mask).backward()
        for name, p in model.named_parameters():
            torch.testing.assert_close(private[name], p.grad, atol=1e-6, rtol=1e-4)

    def test_loss_matches_standard_training(self):
        x, y, mask = _batch(n=4)
        cfg = DPConfig(noise_multiplier=0.0, clip_norm=1e9)
        dp_loss, _ = dp_train_sparse_pixel(_model(), _loader(x, y, mask, 4), cfg)
        ref_loss = train_sparse_pixel(_model(), _loader(x, y, mask, 4))
        assert dp_loss == pytest.approx(ref_loss, rel=1e-5)

    def test_returns_pooled_loss_and_steps_accountant(self):
        x, y, mask = _batch(n=4)
        cfg = DPConfig(noise_multiplier=1.0, clip_norm=1.0)
        model = _model()
        loss, accountant = dp_train_sparse_pixel(model, _loader(x, y, mask, 2), cfg)
        assert np.isfinite(loss)
        assert accountant.steps == 2

    def test_noise_changes_the_update(self):
        x, y, mask = _batch(n=4)
        a, b = _model(), copy.deepcopy(_model())
        dp_train_sparse_pixel(a, _loader(x, y, mask, 2), DPConfig(noise_multiplier=0.0))
        dp_train_sparse_pixel(b, _loader(x, y, mask, 2), DPConfig(noise_multiplier=1.0))
        diff = sum((p - q).abs().sum() for p, q in zip(a.parameters(), b.parameters()))
        assert diff > 0


class TestDPAccountantNumSteps:
    def test_num_steps_equals_repeated_steps(self):
        many, once = DPAccountant(), DPAccountant()
        for _ in range(5):
            many.step(1.0, sample_rate=0.01)
        once.step(1.0, sample_rate=0.01, num_steps=5)
        assert once.steps == 5
        assert once.epsilon == pytest.approx(many.epsilon)

    def test_default_is_single_step(self):
        acc = DPAccountant()
        acc.step(1.0)
        assert acc.steps == 1


class TestLocalDPEpsilon:
    def test_matches_manual_accounting(self):
        cfg = DPConfig(noise_multiplier=1.0, target_delta=1e-5)
        eps = local_dp_epsilon(
            cfg, dataset_size=100, batch_size=16, local_epochs=2, num_rounds=3
        )
        acc = DPAccountant(target_delta=1e-5)
        for _ in range(3 * 2 * 7):  # ceil(100 / 16) = 7 batches per epoch
            acc.step(1.0, sample_rate=16 / 100)
        assert eps == pytest.approx(acc.epsilon)

    def test_more_rounds_cost_more_privacy(self):
        cfg = DPConfig(noise_multiplier=1.0)
        kwargs = dict(dataset_size=1000, batch_size=16, local_epochs=1)
        assert local_dp_epsilon(cfg, num_rounds=10, **kwargs) > local_dp_epsilon(
            cfg, num_rounds=1, **kwargs
        )

    def test_zero_noise_is_infinite(self):
        cfg = DPConfig(noise_multiplier=0.0)
        eps = local_dp_epsilon(
            cfg, dataset_size=100, batch_size=16, local_epochs=1, num_rounds=1
        )
        assert eps == float("inf")

import pytest
import torch
import torch.nn as nn

from fl4hma.models.unet import UNetCNN, enable_mc_dropout, ensemble_forward


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


def build(dropout: float = 0.0, **kwargs) -> UNetCNN:
    return UNetCNN(
        in_channels=3, out_channels=1, base_filters=8, dropout=dropout, **kwargs
    )


class TestBackwardCompatibility:
    def test_default_has_no_dropout(self):
        assert build().dropout == 0.0

    def test_default_state_dict_keys_unchanged(self):
        """Dropout must not add state, or FedAvg parameter lists would shift."""
        assert list(build().state_dict()) == list(build(dropout=0.3).state_dict())

    def test_zero_dropout_is_deterministic_in_train_mode(self, batch):
        model = build().train()
        assert torch.allclose(model(batch), model(batch))


class TestDropoutPlacement:
    def test_rejects_unknown_block(self):
        with pytest.raises(ValueError, match="unknown dropout block"):
            build(dropout=0.2, dropout_blocks=("enc9",))

    def test_rejects_rate_outside_unit_interval(self):
        with pytest.raises(ValueError, match="dropout must be in"):
            build(dropout=1.0)

    def test_blocks_ignored_when_rate_is_zero(self, batch):
        model = build(dropout=0.0, dropout_blocks=("bottleneck",)).train()
        assert torch.allclose(model(batch), model(batch))


class TestMCDropout:
    def test_eval_alone_disables_dropout(self, batch):
        model = build(dropout=0.3).eval()
        assert torch.allclose(model(batch), model(batch))

    def test_enable_mc_dropout_makes_forward_stochastic(self, batch):
        model = build(dropout=0.3)
        enable_mc_dropout(model)
        assert not torch.allclose(model(batch), model(batch))

    def test_batchnorm_stays_in_eval(self):
        model = build(dropout=0.3, norm="batch")
        enable_mc_dropout(model)

        bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        drops = [m for m in model.modules() if isinstance(m, nn.Dropout2d)]

        assert bn and drops
        assert all(not m.training for m in bn)
        assert all(m.training for m in drops)

    def test_running_stats_not_updated(self, batch):
        """BatchNorm must not drift while drawing MC samples."""
        model = build(dropout=0.3, norm="batch")
        enable_mc_dropout(model)
        bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
        before = bn.running_mean.clone()

        for _ in range(5):
            model(batch)

        assert torch.allclose(bn.running_mean, before)


class TestEnsembleForward:
    def test_shape(self, batch):
        model = build(dropout=0.2)
        enable_mc_dropout(model)
        assert ensemble_forward(model, batch, n_members=6).shape == (2, 6, 32, 32)

    def test_members_differ_under_mc_dropout(self, batch):
        model = build(dropout=0.3)
        enable_mc_dropout(model)
        ens = ensemble_forward(model, batch, n_members=8)

        assert ens.std(dim=1).mean() > 0

    def test_members_identical_without_dropout(self, batch):
        model = build(dropout=0.0).eval()
        ens = ensemble_forward(model, batch, n_members=4)

        assert ens.std(dim=1).max().item() == pytest.approx(0.0, abs=1e-6)

    def test_chunking_preserves_shape_and_distribution(self, batch):
        """Chunking draws masks for differently shaped batches, so the RNG
        stream differs; only the ensemble distribution is preserved. The mean
        is compared in units of the ensemble std because it sits near zero."""
        model = build(dropout=0.3)
        enable_mc_dropout(model)

        torch.manual_seed(7)
        whole = ensemble_forward(model, batch, n_members=64)
        torch.manual_seed(7)
        chunked = ensemble_forward(model, batch, n_members=64, chunk_size=8)

        assert whole.shape == chunked.shape
        assert chunked.mean().item() == pytest.approx(
            whole.mean().item(), abs=0.15 * whole.std().item()
        )
        assert chunked.std().item() == pytest.approx(whole.std().item(), rel=5e-2)

    def test_gradient_flows_through_all_members(self, batch):
        model = build(dropout=0.2).train()
        ens = ensemble_forward(model, batch, n_members=4)
        ens.sum().backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_rejects_multi_channel_output(self, batch):
        model = UNetCNN(in_channels=3, out_channels=2, base_filters=8)
        with pytest.raises(ValueError, match="out_channels=1"):
            ensemble_forward(model, batch, n_members=3)

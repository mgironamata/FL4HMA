import math

import numpy as np
import pytest
import torch

from fl4hma.training.losses import (
    LossConfig,
    sparse_crps_loss,
    sparse_tweedie_loss,
    tweedie_deviance,
)


def analytic_gaussian_crps(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of N(mu, sigma) evaluated at y (Gneiting & Raftery 2007)."""
    w = (y - mu) / sigma
    cdf = 0.5 * (1.0 + math.erf(w / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
    return sigma * (w * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def full_mask(b: int, h: int, w: int) -> torch.Tensor:
    return torch.ones(b, h, w)


class TestCRPSPointForecast:
    """A degenerate ensemble is a point forecast, whose CRPS is the MAE."""

    def test_identical_members_give_mae(self):
        torch.manual_seed(0)
        y = torch.randn(4, 1, 8, 8)
        point = torch.randn(4, 1, 8, 8)
        ens = point.expand(-1, 12, -1, -1).contiguous()

        crps = sparse_crps_loss(ens, y, full_mask(4, 8, 8))

        assert crps.item() == pytest.approx((point - y).abs().mean().item(), rel=1e-5)

    def test_perfect_point_forecast_is_zero(self):
        y = torch.full((2, 1, 4, 4), 3.0)
        ens = torch.full((2, 8, 4, 4), 3.0)

        assert sparse_crps_loss(ens, y, full_mask(2, 4, 4)).item() == pytest.approx(0.0)


class TestCRPSAgainstAnalytic:
    def test_converges_to_gaussian_crps(self):
        torch.manual_seed(0)
        mu, sigma, y = 1.0, 2.0, 1.75
        ens = (torch.randn(1, 20000, 1, 1) * sigma + mu).clamp(-1e6, 1e6)
        target = torch.full((1, 1, 1, 1), y)

        crps = sparse_crps_loss(ens, target, full_mask(1, 1, 1)).item()

        assert crps == pytest.approx(analytic_gaussian_crps(mu, sigma, y), rel=0.02)

    def test_fair_estimator_is_unbiased_at_small_m(self):
        """Averaged over draws, the fair estimator recovers the true CRPS."""
        torch.manual_seed(0)
        mu, sigma, y, m = 0.0, 1.0, 0.5, 4
        truth = analytic_gaussian_crps(mu, sigma, y)
        target = torch.full((4000, 1, 1, 1), y)
        ens = torch.randn(4000, m, 1, 1) * sigma + mu

        fair = sparse_crps_loss(ens, target, full_mask(4000, 1, 1), fair=True).item()

        assert fair == pytest.approx(truth, rel=0.03)

    def test_nrg_estimator_is_biased_high_by_spread_over_2m(self):
        """NRG over-reports by E|X - X'| / 2m, a spread-proportional penalty."""
        torch.manual_seed(0)
        mu, sigma, y, m = 0.0, 1.0, 0.5, 4
        truth = analytic_gaussian_crps(mu, sigma, y)
        target = torch.full((40000, 1, 1, 1), y)
        ens = torch.randn(40000, m, 1, 1) * sigma + mu

        nrg = sparse_crps_loss(ens, target, full_mask(40000, 1, 1), fair=False).item()

        # E|X - X'| = 2 sigma / sqrt(pi) for X, X' iid normal
        expected_bias = (2.0 * sigma / math.sqrt(math.pi)) / (2.0 * m)
        assert nrg > truth
        assert nrg - truth == pytest.approx(expected_bias, rel=0.05)


class TestCRPSPropriety:
    """CRPS must be minimised by a correctly dispersed ensemble."""

    def test_penalises_under_and_over_dispersion(self):
        torch.manual_seed(0)
        n, m = 6000, 32
        y = torch.randn(n, 1, 1, 1)  # truth ~ N(0, 1)
        target_ens = torch.randn(n, m, 1, 1)
        under = torch.randn(n, m, 1, 1) * 0.3
        over = torch.randn(n, m, 1, 1) * 3.0
        mask = full_mask(n, 1, 1)

        calibrated = sparse_crps_loss(target_ens, y, mask).item()

        assert calibrated < sparse_crps_loss(under, y, mask).item()
        assert calibrated < sparse_crps_loss(over, y, mask).item()


class TestCRPSMasking:
    def test_unlabelled_pixels_ignored(self):
        torch.manual_seed(0)
        y = torch.zeros(2, 1, 4, 4)
        y[:, :, 2:, :] = 1e6  # huge errors, but unlabelled
        ens = torch.zeros(2, 8, 4, 4)
        mask = torch.ones(2, 4, 4)
        mask[:, 2:, :] = 0.0

        assert sparse_crps_loss(ens, y, mask).item() == pytest.approx(0.0)

    def test_pooled_over_pixels_not_samples(self):
        y = torch.zeros(2, 1, 2, 2)
        y[0, 0, 0, 0] = 4.0
        ens = torch.zeros(2, 4, 2, 2)
        mask = torch.zeros(2, 2, 2)
        mask[0, 0, 0] = 1.0
        mask[1, 0, :] = 1.0
        mask[1, 1, 0] = 1.0

        # one pixel with |error| 4, three with 0 -> pooled MAE 1.0
        assert sparse_crps_loss(ens, y, mask).item() == pytest.approx(1.0, rel=1e-5)

    def test_no_labelled_pixels_returns_nan(self):
        ens = torch.zeros(2, 4, 4, 4)
        y = torch.zeros(2, 1, 4, 4)

        assert torch.isnan(sparse_crps_loss(ens, y, torch.zeros(2, 4, 4)))


class TestCRPSShapesAndGuards:
    def test_accepts_unsqueezed_target(self):
        torch.manual_seed(0)
        ens = torch.randn(2, 6, 4, 4)
        mask = full_mask(2, 4, 4)
        y4 = torch.randn(2, 1, 4, 4)

        a = sparse_crps_loss(ens, y4, mask)
        b = sparse_crps_loss(ens, y4.squeeze(1), mask)

        assert a.item() == pytest.approx(b.item(), rel=1e-6)

    def test_fair_requires_two_members(self):
        with pytest.raises(ValueError, match="at least 2 members"):
            sparse_crps_loss(
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 2, 2),
                full_mask(1, 2, 2),
                fair=True,
            )

    def test_gradient_flows_to_ensemble(self):
        torch.manual_seed(0)
        ens = torch.randn(2, 8, 4, 4, requires_grad=True)
        loss = sparse_crps_loss(ens, torch.randn(2, 1, 4, 4), full_mask(2, 4, 4))
        loss.backward()

        assert ens.grad is not None
        assert torch.isfinite(ens.grad).all()
        assert ens.grad.abs().sum() > 0


class TestTweedieDeviance:
    @pytest.mark.parametrize("p", [1.1, 1.5, 1.9])
    @pytest.mark.parametrize("y", [0.5, 5.0, 100.0])
    def test_zero_at_perfect_prediction(self, p, y):
        yt = torch.tensor([y])
        dev = tweedie_deviance(yt, yt, p=p)

        assert dev.item() == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("p", [1.1, 1.5, 1.9])
    def test_dry_day_deviance_vanishes_only_in_the_limit(self, p):
        """At y = 0 the deviance is 2 mu^(2-p)/(2-p): zero only as mu -> 0."""
        y = torch.zeros(1)
        coarse = tweedie_deviance(torch.tensor([1e-2]), y, p=p).item()
        finer = tweedie_deviance(torch.tensor([1e-6]), y, p=p).item()

        assert finer < coarse
        assert finer == pytest.approx(2.0 * (1e-6) ** (2.0 - p) / (2.0 - p), rel=1e-4)

    def test_larger_p_punishes_false_drizzle_harder(self):
        """p is a dial on dry-day false positives, so it moves wet-day frequency."""
        y, mu = torch.zeros(1), torch.tensor([0.01])
        at_low_p = tweedie_deviance(mu, y, p=1.1).item()
        at_high_p = tweedie_deviance(mu, y, p=1.9).item()

        assert at_high_p > 100.0 * at_low_p

    @pytest.mark.parametrize("p", [1.2, 1.5, 1.8])
    def test_non_negative(self, p):
        torch.manual_seed(0)
        y = torch.rand(500) * 50.0
        mu = torch.rand(500) * 50.0 + 1e-3

        assert (tweedie_deviance(mu, y, p=p) >= -1e-5).all()

    def test_minimised_at_mu_equals_y(self):
        y = torch.tensor([4.0])
        mu = torch.tensor([4.0], requires_grad=True)
        tweedie_deviance(mu, y, p=1.5).sum().backward()

        assert mu.grad.abs().item() == pytest.approx(0.0, abs=1e-5)

    def test_handles_dry_days(self):
        """y = 0 is the whole point of Tweedie; it must stay finite."""
        y = torch.zeros(10)
        mu = torch.linspace(0.01, 10.0, 10)
        dev = tweedie_deviance(mu, y, p=1.5)

        assert torch.isfinite(dev).all()
        assert (dev > 0).all()
        assert (dev.diff() > 0).all()  # penalty grows as mu moves from 0


class TestSparseTweedieLoss:
    def test_log_link_optimum_at_log_y(self):
        y = torch.full((1, 1, 4, 4), 3.0)
        eta = torch.full((1, 1, 4, 4), math.log(3.0), requires_grad=True)
        sparse_tweedie_loss(eta, y, full_mask(1, 4, 4), p=1.5).backward()

        assert eta.grad.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_rejects_negative_targets(self):
        y = torch.full((1, 1, 4, 4), -1.0)
        with pytest.raises(ValueError, match="non-negative"):
            sparse_tweedie_loss(torch.zeros(1, 1, 4, 4), y, full_mask(1, 4, 4))

    def test_sentinel_pixels_do_not_trigger_guard(self):
        """Unlabelled pixels carry the -1 sentinel and must be masked out first."""
        y = torch.full((1, 1, 4, 4), -1.0)
        y[:, :, :2, :] = 2.0
        mask = torch.zeros(1, 4, 4)
        mask[:, :2, :] = 1.0

        loss = sparse_tweedie_loss(torch.zeros(1, 1, 4, 4), y, mask, p=1.5)

        assert torch.isfinite(loss)

    @pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 2.5])
    def test_rejects_power_outside_open_interval(self, p):
        with pytest.raises(ValueError, match="strictly between 1 and 2"):
            sparse_tweedie_loss(
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 2, 2),
                full_mask(1, 2, 2),
                p=p,
            )

    def test_gradient_is_finite_for_extreme_outputs(self):
        y = torch.full((1, 1, 4, 4), 5.0)
        eta = torch.full((1, 1, 4, 4), 50.0, requires_grad=True)
        sparse_tweedie_loss(eta, y, full_mask(1, 4, 4), p=1.5).backward()

        assert torch.isfinite(eta.grad).all()

    def test_no_labelled_pixels_returns_nan(self):
        loss = sparse_tweedie_loss(
            torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4)
        )
        assert torch.isnan(loss)


class TestLossConfig:
    def test_defaults_to_mse(self):
        assert LossConfig().name == "mse"

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="name must be one of"):
            LossConfig(name="huber")

    def test_tweedie_requires_physical_targets(self):
        with pytest.raises(ValueError, match="physical_target"):
            LossConfig(name="tweedie", physical_target=False)

    def test_tweedie_validates_power(self):
        with pytest.raises(ValueError, match="strictly between 1 and 2"):
            LossConfig(name="tweedie", physical_target=True, tweedie_p=2.0)

    def test_crps_requires_two_members(self):
        with pytest.raises(ValueError, match="n_members"):
            LossConfig(name="crps", n_members=1)

    def test_crps_requires_dropout_for_spread(self):
        with pytest.raises(ValueError, match="dropout"):
            LossConfig(name="crps", dropout=0.0)

    def test_valid_crps_config(self):
        cfg = LossConfig(name="crps", n_members=8, dropout=0.15)
        assert cfg.n_members == 8 and cfg.fair is True

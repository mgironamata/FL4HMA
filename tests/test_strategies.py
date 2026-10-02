import numpy as np
import pytest
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg, FedProx, FedYogi
from pydantic import ValidationError

from fl4hma.federation.strategies import AggregationConfig, build_strategy


def _initial_parameters():
    return ndarrays_to_parameters([np.zeros(3, dtype=np.float32)])


def _fit_result(values, num_examples=1):
    res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([np.asarray(values, dtype=np.float32)]),
        num_examples=num_examples,
        metrics={},
    )
    return (None, res)


class TestAggregationConfig:
    def test_default_is_fedavg(self):
        assert AggregationConfig().method == "fedavg"

    def test_fedprox_requires_positive_mu(self):
        with pytest.raises(ValidationError, match="proximal_mu"):
            AggregationConfig(method="fedprox")

    @pytest.mark.parametrize("method", ["fedavg", "fedadam", "fedyogi", "fedadagrad"])
    def test_mu_rejected_without_fedprox(self, method):
        with pytest.raises(ValidationError, match="proximal_mu"):
            AggregationConfig(method=method, proximal_mu=0.1)

    @pytest.mark.parametrize("method", ["fedavg", "fedprox"])
    @pytest.mark.parametrize("field", ["server_lr", "beta_1", "beta_2", "tau"])
    def test_server_optimiser_fields_rejected_for_non_adaptive(self, method, field):
        mu = 0.1 if method == "fedprox" else 0.0
        with pytest.raises(ValidationError, match=field):
            AggregationConfig(method=method, proximal_mu=mu, **{field: 0.5})

    @pytest.mark.parametrize("field", ["beta_1", "beta_2"])
    def test_betas_rejected_for_fedadagrad(self, field):
        with pytest.raises(ValidationError, match=field):
            AggregationConfig(method="fedadagrad", **{field: 0.5})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"proximal_mu": -1.0, "method": "fedprox"},
            {"server_lr": 0.0, "method": "fedadam"},
            {"beta_1": 1.0, "method": "fedadam"},
            {"beta_2": -0.1, "method": "fedyogi"},
            {"tau": 0.0, "method": "fedyogi"},
        ],
    )
    def test_out_of_range_values_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            AggregationConfig(**kwargs)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError):
            AggregationConfig(method="fedproc")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            AggregationConfig(mu=0.1)


class TestBuildStrategy:
    @pytest.mark.parametrize(
        "config, cls",
        [
            (AggregationConfig(), FedAvg),
            (AggregationConfig(method="fedprox", proximal_mu=0.1), FedProx),
            (AggregationConfig(method="fedadam"), FedAdam),
            (AggregationConfig(method="fedyogi"), FedYogi),
            (AggregationConfig(method="fedadagrad"), FedAdagrad),
        ],
    )
    def test_returns_matching_strategy(self, config, cls):
        strategy = build_strategy(config, _initial_parameters(), min_fit_clients=2)
        assert type(strategy) is cls
        assert strategy.min_fit_clients == 2

    def test_fedprox_passes_mu(self):
        strategy = build_strategy(
            AggregationConfig(method="fedprox", proximal_mu=0.3),
            _initial_parameters(),
        )
        assert strategy.proximal_mu == 0.3

    def test_adaptive_passes_server_optimiser_settings(self):
        config = AggregationConfig(
            method="fedadam", server_lr=0.05, beta_1=0.8, beta_2=0.9, tau=1e-4
        )
        strategy = build_strategy(config, _initial_parameters())
        assert (strategy.eta, strategy.beta_1, strategy.beta_2, strategy.tau) == (
            0.05,
            0.8,
            0.9,
            1e-4,
        )

    def test_fedadam_first_round_step_is_server_lr_times_sign(self):
        config = AggregationConfig(method="fedadam", server_lr=0.5, tau=1e-9)
        strategy = build_strategy(config, _initial_parameters())
        params, _ = strategy.aggregate_fit(
            1, [_fit_result([1.0, -2.0, 0.0]), _fit_result([3.0, -2.0, 0.0])], []
        )
        (new,) = parameters_to_ndarrays(params)
        beta_1, beta_2 = config.beta_1, config.beta_2
        eta_norm = 0.5 * np.sqrt(1 - beta_2**2) / (1 - beta_1**2)
        m = (1 - beta_1) * np.array([2.0, -2.0, 0.0])
        v = (1 - beta_2) * np.array([4.0, 4.0, 0.0])
        np.testing.assert_allclose(new, eta_norm * m / (np.sqrt(v) + 1e-9), rtol=1e-5)

    def test_fedavg_is_weighted_mean(self):
        strategy = build_strategy(AggregationConfig(), _initial_parameters())
        params, _ = strategy.aggregate_fit(
            1, [_fit_result([1.0, 1.0, 1.0], 1), _fit_result([4.0, 4.0, 4.0], 2)], []
        )
        np.testing.assert_allclose(parameters_to_ndarrays(params)[0], [3.0, 3.0, 3.0])

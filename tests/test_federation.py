import numpy as np
import pytest
import torch
from flwr.server.history import History
from flwr.server.strategy import FedAvg, FedYogi
from torch.utils.data import TensorDataset

from fl4hma.federation import federation
from fl4hma.federation.federation import run_federated
from fl4hma.federation.strategies import AggregationConfig
from tests.test_datasets import make_dataarray


def _tiny_federated_inputs(tmp_path):
    rng = np.random.default_rng(0)
    masks = {}
    for name in ("a", "b"):
        path = tmp_path / f"{name}.npy"
        np.save(path, rng.random((32, 32)) > 0.5)
        masks[name] = str(path)
    out_mask = tmp_path / "out.npy"
    np.save(out_mask, np.ones((32, 32), dtype=bool))
    da = make_dataarray(n_time=4, n_lat=32, n_lon=32)
    return dict(
        da_train=da,
        da_test=da,
        country_masks=masks,
        output_mask_path=str(out_mask),
        centralised_mask_path=masks["a"],
        num_rounds=2,
        batch_size=2,
        base_filters=4,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "aggregation",
    [
        None,
        AggregationConfig(method="fedprox", proximal_mu=0.1),
        AggregationConfig(method="fedadam"),
        AggregationConfig(method="fedyogi"),
        AggregationConfig(method="fedadagrad"),
    ],
    ids=["fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad"],
)
def test_run_federated_updates_global_model(tmp_path, monkeypatch, aggregation):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")

    result = run_federated(**_tiny_federated_inputs(tmp_path), aggregation=aggregation)

    losses = result["losses"]
    assert len(losses) == 3
    assert losses[1] != losses[0]
    expected = (aggregation or AggregationConfig()).model_dump()
    assert result["config"]["aggregation"] == expected


class TestRunFederatedStrategy:
    @pytest.fixture
    def captured(self, monkeypatch):
        seen = {}

        def fake_simulation(client_list, build_client, num_rounds, strategy):
            seen["strategy"] = strategy
            return History()

        monkeypatch.setattr(federation, "run_ray_simulation", fake_simulation)
        return seen

    def test_defaults_to_fedavg(self, tmp_path, captured):
        run_federated(**_tiny_federated_inputs(tmp_path))
        assert type(captured["strategy"]) is FedAvg

    def test_uses_configured_strategy(self, tmp_path, captured):
        run_federated(
            **_tiny_federated_inputs(tmp_path),
            aggregation=AggregationConfig(method="fedyogi", server_lr=0.02),
        )
        strategy = captured["strategy"]
        assert type(strategy) is FedYogi
        assert strategy.eta == 0.02
        assert strategy.min_fit_clients == 2
        assert strategy.fraction_evaluate == 0.0


class TestClientProximalMu:
    @pytest.fixture
    def client(self, monkeypatch):
        calls = []

        def fake_train(model, loader, epochs, lr, proximal_mu=0.0):
            calls.append(proximal_mu)
            return 0.0

        monkeypatch.setattr(federation, "train_sparse_pixel", fake_train)
        ds = TensorDataset(torch.zeros(2, 3, 8, 8))
        client = federation.AphroFlowerClient(ds, base_filters=4)
        return client, calls

    def test_fit_uses_proximal_mu_from_config(self, client):
        client, calls = client
        client.fit(client.get_parameters({}), {"proximal_mu": 0.25})
        assert calls == [0.25]

    def test_fit_defaults_to_no_proximal_term(self, client):
        client, calls = client
        client.fit(client.get_parameters({}), {})
        assert calls == [0.0]

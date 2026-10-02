import os
from types import SimpleNamespace

import cloudpickle
import flwr as fl
import numpy as np
import pytest
from flwr.client import NumPyClient
from flwr.server.strategy import FedAvg

from fl4hma.data.torch_dataset import build_country_datasets
from fl4hma.federation.simulation import (
    make_client_fn,
    ray_init_args_from_env,
    run_ray_simulation,
)
from tests.test_datasets import make_dataarray


class TestRayInitArgsFromEnv:
    def test_uses_slurm_cpus_per_task(self):
        env = {"SLURM_CPUS_PER_TASK": "10", "SLURM_CPUS_ON_NODE": "48"}
        args = ray_init_args_from_env(env, num_gpus=2)
        assert args["num_cpus"] == 10

    def test_falls_back_to_slurm_cpus_on_node(self):
        args = ray_init_args_from_env({"SLURM_CPUS_ON_NODE": "2"}, num_gpus=0)
        assert args["num_cpus"] == 2

    def test_falls_back_to_os_cpu_count_outside_slurm(self):
        args = ray_init_args_from_env({}, num_gpus=0)
        assert args["num_cpus"] == os.cpu_count()

    def test_passes_num_gpus(self):
        args = ray_init_args_from_env({}, num_gpus=2)
        assert args["num_gpus"] == 2

    def test_object_store_sized_from_slurm_mem_per_node(self):
        env = {"SLURM_MEM_PER_NODE": "131072"}
        args = ray_init_args_from_env(env, num_gpus=0, object_store_fraction=0.25)
        assert args["object_store_memory"] == int(0.25 * 131072 * 1024**2)

    def test_object_store_sized_from_slurm_mem_per_cpu(self):
        env = {"SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_CPU": "8192"}
        args = ray_init_args_from_env(env, num_gpus=0, object_store_fraction=0.5)
        assert args["object_store_memory"] == int(0.5 * 4 * 8192 * 1024**2)

    def test_no_object_store_limit_outside_slurm(self):
        args = ray_init_args_from_env({}, num_gpus=0)
        assert "object_store_memory" not in args

    def test_keeps_flower_defaults(self):
        args = ray_init_args_from_env({}, num_gpus=0)
        assert args["ignore_reinit_error"] is True
        assert args["include_dashboard"] is False

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
    def test_rejects_invalid_fraction(self, fraction):
        with pytest.raises(ValueError):
            ray_init_args_from_env({}, num_gpus=0, object_store_fraction=fraction)


class _EchoClient(NumPyClient):
    def __init__(self, dataset, tag):
        self.dataset = dataset
        self.tag = tag


def _context(partition_id):
    return SimpleNamespace(node_config={"partition-id": partition_id})


class _ListLoader:
    def __init__(self, datasets):
        self.datasets = datasets
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.datasets


class TestMakeClientFn:
    def test_builds_client_for_partition(self):
        datasets = ["ds0", "ds1", "ds2"]
        client_fn = make_client_fn(
            _ListLoader(datasets), lambda ds: _EchoClient(ds, tag="x")
        )
        client = client_fn(_context(2))
        assert client.numpy_client.dataset == "ds2"
        assert client.numpy_client.tag == "x"

    def test_loads_datasets_lazily(self):
        loader = _ListLoader(["ds0"])
        client_fn = make_client_fn(loader, lambda ds: _EchoClient(ds, tag="x"))
        assert loader.calls == 0
        client_fn(_context(0))
        assert loader.calls == 1

    def test_pickled_client_fn_only_holds_loader_and_builder(self):
        client_fn = make_client_fn(
            _HandleLoader("ref"), lambda ds: _EchoClient(ds, tag="x")
        )
        assert len(cloudpickle.dumps(client_fn)) < 10_000


class _HandleLoader:
    def __init__(self, handle):
        self.handle = handle

    def __call__(self):
        return [self.handle]


@pytest.mark.slow
class TestRunRaySimulation:
    def test_clients_train_through_object_store(self, tmp_path):
        class LenClient(NumPyClient):
            def __init__(self, n):
                self.n = n

            def fit(self, parameters, config):
                return parameters, self.n, {"n": self.n}

        mask = tmp_path / "mask.npy"
        np.save(mask, np.ones((32, 32), dtype=bool))
        datasets = build_country_datasets(
            make_dataarray(n_time=2, n_lat=32, n_lon=32),
            {"a": str(mask), "b": str(mask)},
            output_mask_path=str(mask),
        )
        seen = []

        def build_client(ds):
            return LenClient(len(ds))

        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=fl.common.ndarrays_to_parameters([np.zeros(1)]),
            fit_metrics_aggregation_fn=lambda ms: seen.extend(ms) or {},
        )
        history = run_ray_simulation(
            list(datasets.values()),
            build_client,
            num_rounds=1,
            strategy=strategy,
            ray_init_args=ray_init_args_from_env({}, num_gpus=0) | {"num_cpus": 2},
        )
        assert sorted(m["n"] for _, m in seen) == [2, 2]
        assert history is not None

import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import flwr as fl
import ray
import torch
from flwr.client import Client, NumPyClient
from flwr.common import Context
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.simulation import start_simulation

_MB = 1024**2


def _slurm_cpus(env: Mapping[str, str]) -> Optional[int]:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if env.get(key):
            return int(env[key])
    return None


def _slurm_mem_bytes(env: Mapping[str, str], num_cpus: int) -> Optional[int]:
    if env.get("SLURM_MEM_PER_NODE"):
        return int(env["SLURM_MEM_PER_NODE"]) * _MB
    if env.get("SLURM_MEM_PER_CPU"):
        return int(env["SLURM_MEM_PER_CPU"]) * num_cpus * _MB
    return None


def ray_init_args_from_env(
    env: Optional[Mapping[str, str]] = None,
    num_gpus: Optional[int] = None,
    object_store_fraction: float = 0.3,
) -> Dict[str, Any]:
    """Build ``ray.init`` kwargs that respect the SLURM allocation.

    Without these, Ray sizes itself to the whole node rather than the job's
    cgroup, creates more client actors than the job can hold, and gets
    OOM-killed.

    Args:
        env: Environment to read SLURM variables from. Defaults to
            ``os.environ``.
        num_gpus: GPUs to expose to Ray. Defaults to
            ``torch.cuda.device_count()``.
        object_store_fraction: Fraction of the SLURM memory allocation given
            to Ray's shared-memory object store. Must be in (0, 1).

    Returns:
        Keyword arguments for ``ray.init`` (and Flower's
        ``start_simulation(ray_init_args=...)``).

    Raises:
        ValueError: If ``object_store_fraction`` is not in (0, 1).
    """
    if not 0.0 < object_store_fraction < 1.0:
        raise ValueError(
            f"object_store_fraction must be in (0, 1), got {object_store_fraction}"
        )
    env = os.environ if env is None else env
    num_cpus = _slurm_cpus(env) or os.cpu_count() or 1
    args: Dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "num_cpus": num_cpus,
        "num_gpus": torch.cuda.device_count() if num_gpus is None else num_gpus,
    }
    mem_bytes = _slurm_mem_bytes(env, num_cpus)
    if mem_bytes is not None:
        args["object_store_memory"] = int(object_store_fraction * mem_bytes)
    return args


class ObjectRefLoader:
    """Callable that fetches an object from the Ray object store.

    Pickles to just the ``ObjectRef``, so it can be captured by a Flower
    ``client_fn`` without shipping the underlying data to every actor.
    Numpy arrays inside the object come back as zero-copy, read-only views of
    shared memory.

    Args:
        ref: Reference returned by ``ray.put``.
    """

    def __init__(self, ref: "ray.ObjectRef") -> None:
        self.ref = ref

    def __call__(self) -> Any:
        return ray.get(self.ref)


def make_client_fn(
    load_datasets: Callable[[], Sequence[Any]],
    build_client: Callable[[Any], NumPyClient],
) -> Callable[[Context], Client]:
    """Build a Flower ``client_fn`` that loads client data lazily.

    Args:
        load_datasets: Returns the per-client datasets, indexed by
            partition id. Called each time a client is created, inside the
            Ray actor, so it should hold only a small handle (e.g.
            ``ObjectRefLoader``) rather than the data itself.
        build_client: Builds a ``NumPyClient`` from one client's dataset.

    Returns:
        A ``client_fn`` for ``flwr.simulation.start_simulation``.
    """

    def client_fn(context: Context) -> Client:
        cid = int(context.node_config["partition-id"])
        return build_client(load_datasets()[cid]).to_client()

    return client_fn


def run_ray_simulation(
    client_datasets: Sequence[Any],
    build_client: Callable[[Any], NumPyClient],
    num_rounds: int,
    strategy: Strategy,
    ray_init_args: Optional[Dict[str, Any]] = None,
) -> History:
    """Run a Flower simulation with client data shared via the Ray object store.

    Ray is initialised with resources capped to the SLURM allocation, the
    client datasets are put in the object store once, and Ray is shut down
    afterwards so repeated calls (e.g. HP-tuning trials) don't accumulate
    object-store memory.

    Args:
        client_datasets: One dataset per client, indexed by partition id.
        build_client: Builds a ``NumPyClient`` from one client's dataset.
            Must not capture large objects, since it is pickled into every
            actor.
        num_rounds: Number of federated rounds.
        strategy: Flower server strategy.
        ray_init_args: Overrides for ``ray.init``. Defaults to
            ``ray_init_args_from_env()``.

    Returns:
        The Flower ``History`` of the simulation.
    """
    if ray_init_args is None:
        ray_init_args = ray_init_args_from_env()
    num_clients = len(client_datasets)
    ray.init(**ray_init_args)
    try:
        datasets_ref = ray.put(list(client_datasets))
        client_fn = make_client_fn(ObjectRefLoader(datasets_ref), build_client)
        return start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": 1,
                "num_gpus": (
                    (1.0 / num_clients) if ray_init_args.get("num_gpus") else 0.0
                ),
            },
            ray_init_args=ray_init_args,
            keep_initialised=True,
        )
    finally:
        ray.shutdown()

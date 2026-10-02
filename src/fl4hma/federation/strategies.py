from typing import Any, Literal

from flwr.common import Parameters
from flwr.server.strategy import FedAdagrad, FedAdam, FedAvg, FedProx, FedYogi, Strategy
from pydantic import BaseModel, ConfigDict, Field, model_validator

AggregationMethod = Literal["fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad"]

ADAPTIVE_METHODS = ("fedadam", "fedyogi", "fedadagrad")
_SERVER_OPTIMISER_FIELDS = ("server_lr", "beta_1", "beta_2", "tau")


class AggregationConfig(BaseModel):
    """How the server aggregates client updates.

    ``fedavg`` takes the example-weighted mean of client models. ``fedprox``
    does the same but clients add ``proximal_mu / 2 * ||w - w_global||^2`` to
    their loss (Li et al., 2020). ``fedadam``, ``fedyogi`` and ``fedadagrad``
    treat the mean client update as a pseudo-gradient and apply an adaptive
    server optimiser to it (Reddi et al., 2021).

    Attributes:
        method: Aggregation algorithm.
        proximal_mu: FedProx proximal-term weight. Must be > 0 for ``fedprox``
            and is not allowed for other methods.
        server_lr: Server learning rate (``eta``). Adaptive methods only.
        beta_1: First-moment decay. ``fedadam`` / ``fedyogi`` only.
        beta_2: Second-moment decay. ``fedadam`` / ``fedyogi`` only.
        tau: Adaptivity constant added to the denominator. Adaptive methods only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: AggregationMethod = "fedavg"
    proximal_mu: float = Field(default=0.0, ge=0.0)
    server_lr: float = Field(default=1e-2, gt=0.0)
    beta_1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta_2: float = Field(default=0.99, ge=0.0, lt=1.0)
    tau: float = Field(default=1e-3, gt=0.0)

    @model_validator(mode="after")
    def _check_fields_match_method(self) -> "AggregationConfig":
        if self.method == "fedprox" and self.proximal_mu <= 0.0:
            raise ValueError("fedprox requires proximal_mu > 0")
        if self.method != "fedprox" and self.proximal_mu != 0.0:
            raise ValueError(f"proximal_mu is only used by fedprox, not {self.method}")
        if self.method not in ADAPTIVE_METHODS:
            unused = sorted(set(_SERVER_OPTIMISER_FIELDS) & self.model_fields_set)
            if unused:
                raise ValueError(
                    f"{', '.join(unused)} only apply to adaptive methods "
                    f"{ADAPTIVE_METHODS}, not {self.method}"
                )
        if self.method == "fedadagrad":
            unused = sorted({"beta_1", "beta_2"} & self.model_fields_set)
            if unused:
                raise ValueError(f"{', '.join(unused)} not used by fedadagrad")
        return self


def build_strategy(
    config: AggregationConfig,
    initial_parameters: Parameters,
    **strategy_kwargs: Any,
) -> Strategy:
    """Build the Flower strategy described by ``config``.

    Args:
        config: Aggregation settings.
        initial_parameters: Initial global model. Adaptive methods need it to
            compute the first pseudo-gradient.
        **strategy_kwargs: Arguments shared by all ``FedAvg``-based
            strategies, e.g. ``fraction_fit``, ``min_fit_clients``,
            ``evaluate_fn``.

    Returns:
        A configured Flower strategy.
    """
    common = {"initial_parameters": initial_parameters, **strategy_kwargs}
    if config.method == "fedavg":
        return FedAvg(**common)
    if config.method == "fedprox":
        return FedProx(proximal_mu=config.proximal_mu, **common)
    if config.method == "fedadagrad":
        return FedAdagrad(eta=config.server_lr, tau=config.tau, **common)
    cls = FedAdam if config.method == "fedadam" else FedYogi
    return cls(
        eta=config.server_lr,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        tau=config.tau,
        **common,
    )

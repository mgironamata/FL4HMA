from dataclasses import dataclass
from typing import Optional, Tuple

import torch

LOSS_NAMES = ("mse", "crps", "tweedie")


@dataclass
class LossConfig:
    """Selects and parameterises the training objective.

    Attributes:
        name: One of ``"mse"``, ``"crps"``, ``"tweedie"``.
        n_members: Ensemble members drawn per forward pass for CRPS.
        fair: Use the unbiased (fair) CRPS estimator. The biased NRG estimator
            carries a spread-proportional bias that drives the ensemble
            under-dispersed, so it should not be used to train.
        dropout: Dropout rate used to generate ensemble spread under MC dropout.
            Must be non-zero for CRPS, otherwise every member is identical and
            the objective collapses to MAE. Defaults to 0.0 so that the default
            config leaves existing MSE runs bit-identical.
        tweedie_p: Tweedie variance power, strictly inside (1, 2).
        physical_target: Serve targets in physical units (mm) rather than
            z-scores. Required by Tweedie, which is undefined for negative y.
        eta_clamp: Bound on the raw network output before the Tweedie log link
            is applied, guarding ``exp`` against overflow.
        chunk_size: Draw CRPS members in groups of this size to bound peak
            memory. ``None`` draws all members in one pass.
        target_min: Lower clamp on physical targets, passed to the dataset.
            Defaults to 0.0 under Tweedie, which cannot accept negative y.
            Leave ``None`` for temperature.
    """

    name: str = "mse"
    n_members: int = 16
    fair: bool = True
    dropout: float = 0.0
    tweedie_p: float = 1.5
    physical_target: bool = False
    eta_clamp: float = 20.0
    chunk_size: Optional[int] = None
    target_min: Optional[float] = None

    def __post_init__(self) -> None:
        if self.name not in LOSS_NAMES:
            raise ValueError(f"name must be one of {LOSS_NAMES}, got {self.name!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        if self.name == "crps":
            if self.n_members < 2:
                raise ValueError(
                    f"n_members must be >= 2 for CRPS, got {self.n_members}"
                )
            if self.dropout == 0.0:
                raise ValueError(
                    "dropout must be > 0 for CRPS: with no dropout every MC "
                    "member is identical and CRPS degenerates to MAE"
                )
        if self.name == "tweedie":
            if not self.physical_target:
                raise ValueError(
                    "physical_target must be True for the tweedie loss: it is "
                    "undefined for the negative targets z-scoring produces"
                )
            if self.target_min is None:
                self.target_min = 0.0
            if not 1.0 < self.tweedie_p < 2.0:
                raise ValueError(
                    f"tweedie_p must be strictly between 1 and 2, "
                    f"got {self.tweedie_p}"
                )

    @property
    def is_ensemble(self) -> bool:
        """Whether this objective needs multiple forward passes per batch."""
        return self.name == "crps"


def _select_labelled(
    targets: torch.Tensor,
    output_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten targets to labelled pixels only.

    Args:
        targets: ``(B, 1, H, W)`` or ``(B, H, W)``.
        output_mask: ``(B, H, W)`` binary mask of labelled pixels.

    Returns:
        ``(labelled_targets, boolean_mask)`` where the targets are 1-D.
    """
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    labelled = output_mask.bool()
    return targets[labelled], labelled


def sparse_crps_loss(
    ensemble: torch.Tensor,
    targets: torch.Tensor,
    output_mask: torch.Tensor,
    fair: bool = True,
) -> torch.Tensor:
    """Sample-based CRPS over labelled pixels only.

    Uses the ensemble estimator

    ``CRPS = mean_i |x_i - y| - (1 / 2 m (m-1)) sum_i sum_j |x_i - x_j|``

    where the second denominator is ``2 m^2`` for the biased NRG form. The fair
    form is an unbiased estimator of the CRPS of the underlying predictive
    distribution. The NRG form carries a bias of ``+E|X - X'| / 2m`` — biased
    high, and proportional to the ensemble spread — so minimising it adds an
    extra penalty on spread and drives the ensemble under-dispersed. Use the
    fair form to train; NRG is kept only for comparison with published scores.

    The pairwise term is evaluated in ``O(m log m)`` via the sorted identity
    ``sum_i sum_j |x_i - x_j| = 2 sum_i (2i - m - 1) x_(i)``, which keeps memory
    at ``O(n m)`` rather than ``O(n m^2)``.

    Args:
        ensemble: Ensemble members, shape ``(B, M, H, W)``.
        targets: Observations, shape ``(B, 1, H, W)`` or ``(B, H, W)``.
        output_mask: Binary mask of labelled pixels, shape ``(B, H, W)``.
        fair: Use the unbiased estimator. Requires ``M >= 2``.

    Returns:
        Scalar CRPS pooled over every labelled pixel, in the units of
        ``targets``. ``nan`` when no pixel is labelled.
    """
    n_members = ensemble.shape[1]
    if fair and n_members < 2:
        raise ValueError(
            f"the fair CRPS estimator needs at least 2 members, got {n_members}"
        )

    y, labelled = _select_labelled(targets, output_mask)
    if y.numel() == 0:
        return ensemble.new_tensor(float("nan"))

    members = ensemble.permute(0, 2, 3, 1)[labelled]

    spread_from_obs = (members - y.unsqueeze(1)).abs().mean(dim=1)

    ordered, _ = torch.sort(members, dim=1)
    rank = torch.arange(1, n_members + 1, device=members.device, dtype=members.dtype)
    pairwise_sum = 2.0 * (ordered * (2.0 * rank - n_members - 1.0)).sum(dim=1)
    denominator = (
        2.0 * n_members * (n_members - 1) if fair else 2.0 * n_members * n_members
    )

    return (spread_from_obs - pairwise_sum / denominator).mean()


def tweedie_deviance(
    mu: torch.Tensor,
    y: torch.Tensor,
    p: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Elementwise unit Tweedie deviance, zero exactly when ``mu == y``.

    For ``1 < p < 2`` the Tweedie family is compound Poisson-Gamma, which places
    an exact point mass at zero and a continuous right tail — the structure of
    daily precipitation.

    On dry days (``y = 0``) the deviance reduces to ``2 mu^(2-p) / (2-p)``, which
    reaches zero only as ``mu -> 0`` and decays far more slowly as ``p``
    approaches 2. Predicting 0.01 mm on a dry day costs 0.035 deviance units at
    ``p = 1.1`` but 12.6 at ``p = 1.9``, so ``p`` is effectively a dial on how
    hard false-positive drizzle is punished, and it will move the wet-day
    frequency of the fitted model.

    Args:
        mu: Predicted mean, strictly positive (clamped at ``eps``).
        y: Observations, non-negative.
        p: Variance power in (1, 2).
        eps: Lower clamp on ``mu``.

    Returns:
        Deviance with the same shape as the inputs.
    """
    mu = mu.clamp(min=eps)
    y = y.clamp(min=0.0)
    observed = torch.pow(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
    cross = y * torch.pow(mu, 1.0 - p) / (1.0 - p)
    predicted = torch.pow(mu, 2.0 - p) / (2.0 - p)
    return 2.0 * (observed - cross + predicted)


def sparse_tweedie_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_mask: torch.Tensor,
    p: float = 1.5,
    eta_clamp: float = 20.0,
) -> torch.Tensor:
    """Tweedie deviance over labelled pixels, with a log link.

    ``predictions`` are the raw network outputs and are mapped to a strictly
    positive mean by ``mu = exp(eta)``, so no output activation is needed. Only
    labelled pixels are read, so the ``-1`` sentinel on unlabelled pixels never
    reaches the deviance.

    Args:
        predictions: Raw network output ``eta``, ``(B, 1, H, W)``.
        targets: Observations in physical units, non-negative on labelled
            pixels, ``(B, 1, H, W)`` or ``(B, H, W)``.
        output_mask: Binary mask of labelled pixels, ``(B, H, W)``.
        p: Variance power in (1, 2).
        eta_clamp: Bound applied to ``eta`` before the exponential.

    Returns:
        Scalar deviance pooled over every labelled pixel. ``nan`` when no pixel
        is labelled.
    """
    if not 1.0 < p < 2.0:
        raise ValueError(f"p must be strictly between 1 and 2, got {p}")

    y, labelled = _select_labelled(targets, output_mask)
    if y.numel() == 0:
        return predictions.new_tensor(float("nan"))
    if (y < 0).any():
        raise ValueError(
            "tweedie loss requires non-negative targets on labelled pixels; "
            "z-scored targets are not valid — set physical_target=True"
        )

    eta = predictions.squeeze(1)[labelled].clamp(-eta_clamp, eta_clamp)
    return tweedie_deviance(torch.exp(eta), y, p=p).mean()

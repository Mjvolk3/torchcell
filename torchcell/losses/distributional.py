# torchcell/losses/distributional
# [[torchcell.losses.distributional]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/losses/distributional
# Test file: tests/torchcell/losses/test_distributional.py
"""Distributional decoder heads + proper-scoring-rule losses for the multitask CGT.

The DECODER factorizes into two orthogonal axes (see
``notes/experiments.019-simb-multimodal.decoder-distributional-plan.md``):

* the STRUCTURAL form (S0 per-token / S1 pool / S3 cross-attention) -- *where* each
  output reads its information; lives in the model;
* the DISTRIBUTIONAL form (this module) -- a point estimate vs a predicted distribution;
  sets the mean-collapse incentive. It applies on TOP of any structural form, so it is a
  pluggable head, not a fourth structural row.

Three modes, all avoiding the binning + KDE-bandwidth tuning of past attempts (they model
the conditional distribution in continuous space, per instance):

* ``point``    -- one value per feature; MSE. Baseline; *rewards* mean-collapse (the optimal
                  point prediction under MSE is the conditional mean).
* ``gaussian`` -- ``(mu, sigma)`` per feature; closed-form Gaussian CRPS -- a PROPER scoring
                  rule (minimized in expectation only by the true distribution), in y-units,
                  with no ``1/sigma^2`` blow-up. Selected by the config value ``dist=crps``.
* ``quantile`` -- ``K`` predicted quantiles per feature; pinball / quantile loss
                  (distribution-free; no kernel/bandwidth). Mean pinball ~= empirical CRPS.
                  Selected by ``dist=quantile``.

Layout convention: a structural head emits ``[..., F]`` for ``point`` (unchanged from the
pre-distributional model) or ``[..., F, P]`` for ``P>1`` (``P=2`` gaussian, ``P=K``
quantile), where the LAST axis holds one feature's distributional parameters. ``DistHead``
reads that last axis, so the rest of the pipeline -- ``[..., F]`` targets, per-feature
Pearson computed on ``.point()`` -- is untouched and ranking stays loss-agnostic.
"""

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1/sqrt(pi): the constant term in the closed-form Gaussian CRPS.
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
# 1/sqrt(2): argument scale for Phi(z) = 0.5(1 + erf(z/sqrt(2))).
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
# 1/sqrt(2*pi): normalization of the standard-normal pdf phi(z).
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# The config `dist` name -> the DistHead output mode. `crps` names the LOSS; the head it
# needs is a parametric Gaussian, hence `gaussian`. Kept here so the model (which only needs
# the parameter WIDTH) and the training harness (which builds the head) agree on one mapping.
DIST_TO_MODE: dict[str, str] = {
    "point": "point",
    "crps": "gaussian",
    "quantile": "quantile",
}

# Default quantile grid: K=19 evenly spaced tau in [0.05, 0.95] (step 0.05, 0.5 included).
# Evenly spaced is the *robust* choice -- no kernel, no bandwidth to tune.
DEFAULT_NUM_QUANTILES = 19


def gaussian_crps(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Closed-form CRPS of a Gaussian :math:`\mathcal N(\mu,\sigma)` vs observations.

    .. math::
        \mathrm{CRPS}(\mathcal N(\mu,\sigma), y)
        = \sigma\Big[z\,(2\Phi(z)-1) + 2\varphi(z) - \tfrac{1}{\sqrt\pi}\Big],
        \quad z=\frac{y-\mu}{\sigma},

    with :math:`\Phi,\varphi` the standard-normal CDF/PDF (Gneiting & Raftery 2007). CRPS is
    a PROPER scoring rule -- minimized in expectation only by the true predictive
    distribution -- and, unlike Gaussian NLL, carries no :math:`1/\sigma^2` term, so
    :math:`\sigma` cannot drive the loss to :math:`-\infty` (the collapse that NLL/beta-NLL
    patch around). Value is in the units of ``y``.

    Args:
        mu: Predicted mean, any shape.
        sigma: Predicted standard deviation (STRICTLY positive), broadcastable to ``mu``.
        target: Observations, broadcastable to ``mu``.

    Returns:
        Elementwise CRPS with the broadcast shape of the inputs (callers reduce).
    """
    z = (target - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))
    pdf = _INV_SQRT_2PI * torch.exp(-0.5 * z * z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)


def pinball(
    pred_q: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    r"""Pinball (quantile / check) loss for a set of predicted quantiles.

    :math:`\rho_\tau(u)=\max(\tau u,(\tau-1)u)` with :math:`u=y-\hat q_\tau`. Training a full
    grid of ``tau`` fits the conditional quantile function directly (distribution-free); the
    mean pinball over an even ``tau`` grid approximates the empirical CRPS.

    Args:
        pred_q: Predicted quantiles ``[..., K]`` (last axis indexes ``taus``).
        target: Observations ``[...]`` (broadcast against the quantile axis).
        taus: Quantile levels ``[K]`` in ``(0, 1)``.

    Returns:
        Elementwise pinball ``[..., K]`` (callers reduce over ``K`` and features).
    """
    u = target.unsqueeze(-1) - pred_q
    return torch.maximum(taus * u, (taus - 1.0) * u)


class DistHead(nn.Module):
    """Pluggable distributional readout: interpret a head's raw params, score them, pool.

    Stateless except for the quantile grid (``quantile`` mode). Given a structural head's
    output ``params`` -- ``[..., F]`` (point) or ``[..., F, P]`` (``P=2`` gaussian, ``P=K``
    quantile) -- it exposes:

    * :meth:`point` -- the point estimate ``[..., F]`` used by the (loss-agnostic) metric
      and any downstream reporting: identity (point), ``mu`` (gaussian), or the median
      quantile (quantile);
    * :meth:`loss` -- a masked scalar training loss: MSE (point), mean Gaussian CRPS
      (gaussian), or mean pinball (quantile).
    """

    VALID_MODES = ("point", "gaussian", "quantile")

    def __init__(
        self,
        mode: str,
        quantiles: torch.Tensor | list[float] | None = None,
        sigma_floor: float = 1e-3,
    ) -> None:
        """Build the head.

        Args:
            mode: One of ``point`` / ``gaussian`` / ``quantile``.
            quantiles: ``quantile`` mode only -- the ``tau`` grid (defaults to K=19 evenly
                spaced in ``[0.05, 0.95]``). Ignored otherwise.
            sigma_floor: ``gaussian`` mode only -- additive floor on ``softplus(raw)`` so
                ``sigma`` stays strictly positive.
        """
        super().__init__()
        assert mode in self.VALID_MODES, f"unknown DistHead mode {mode!r}"
        self.mode = mode
        self.sigma_floor = float(sigma_floor)
        if mode == "quantile":
            if quantiles is None:
                q = torch.linspace(0.05, 0.95, DEFAULT_NUM_QUANTILES)
            else:
                q = torch.as_tensor(quantiles, dtype=torch.float32)
            self.register_buffer("_taus", q)
            # Median index = tau nearest 0.5 (== 0.5 for the default even grid): the point
            # estimate for the (loss-agnostic) per-feature Pearson metric.
            self.median_index = int(torch.argmin((q - 0.5).abs()).item())
        else:
            # Register an empty buffer so `.to(device)` / state_dict stay uniform across modes.
            self.register_buffer("_taus", torch.empty(0))
            self.median_index = 0

    @property
    def taus(self) -> torch.Tensor:
        """The quantile grid (empty for non-quantile modes).

        The buffer is stored as ``_taus`` and surfaced through this property because
        ``nn.Module.__getattr__`` is typed ``Tensor | Module``; casting once here keeps every
        caller (and mypy --strict) working with a plain ``Tensor``.
        """
        return cast(torch.Tensor, self._taus)

    @property
    def param_dim(self) -> int:
        """Params per feature: 1 (point), 2 (gaussian), or K (quantile)."""
        if self.mode == "point":
            return 1
        if self.mode == "gaussian":
            return 2
        return int(self.taus.numel())

    def _sigma(self, raw: torch.Tensor) -> torch.Tensor:
        """Map the raw scale param to a strictly-positive sigma via softplus + floor."""
        return F.softplus(raw) + self.sigma_floor

    def point(self, params: torch.Tensor) -> torch.Tensor:
        """Reduce raw params to a point estimate ``[..., F]`` (metric / reporting)."""
        if self.mode == "point":
            return params
        if self.mode == "gaussian":
            return params[..., 0]
        return params[..., self.median_index]

    def loss(
        self,
        params: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked scalar training loss for this head.

        Args:
            params: Head params ``[B, F]`` (point) or ``[B, F, P]`` (gaussian/quantile).
            target: Point targets ``[B, F]`` (same F as ``params``).
            mask: Optional bool ``[B]`` selecting supervised rows. When no row is
                supervised, a graph-connected zero is returned (grads flow as zero).

        Returns:
            Scalar loss.
        """
        if mask is not None:
            params = params[mask]
            target = target[mask]
        if target.shape[0] == 0:
            # Keep the head connected to the graph so DDP find-unused stays happy.
            return params.sum() * 0.0
        if self.mode == "point":
            return F.mse_loss(params, target)
        if self.mode == "gaussian":
            mu = params[..., 0]
            sigma = self._sigma(params[..., 1])
            return gaussian_crps(mu, sigma, target).mean()
        return pinball(params, target, self.taus.to(params.device)).mean()


def dist_param_dim(dist: str, num_quantiles: int = DEFAULT_NUM_QUANTILES) -> int:
    """Params-per-feature for a config ``dist`` value (the head Linear-width multiplier)."""
    return {"point": 1, "crps": 2, "quantile": num_quantiles}[dist]


def make_dist_head(dist: str, num_quantiles: int = DEFAULT_NUM_QUANTILES) -> DistHead:
    """Build a :class:`DistHead` from a config ``dist`` value (``point``/``crps``/``quantile``)."""
    mode = DIST_TO_MODE[dist]
    if mode == "quantile":
        return DistHead(mode, quantiles=torch.linspace(0.05, 0.95, num_quantiles))
    return DistHead(mode)

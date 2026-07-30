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

Six modes, all avoiding the binning + KDE-bandwidth tuning of past attempts (they model
the conditional distribution in continuous space, per instance):

* ``point``        -- one value per feature; MSE. Baseline; *rewards* mean-collapse (the
                      optimal point prediction under MSE is the conditional mean).
* ``gaussian``     -- ``(mu, sigma)`` per feature; closed-form Gaussian CRPS -- a PROPER
                      scoring rule (minimized in expectation only by the true distribution),
                      in y-units, with no ``1/sigma^2`` blow-up. Config ``dist=crps``.
* ``quantile``     -- ``K`` predicted quantiles per feature; pinball / quantile loss
                      (distribution-free; no kernel/bandwidth). Mean pinball ~= empirical
                      CRPS. Config ``dist=quantile``.
* ``laplace``      -- ``(mu, b)`` per feature; closed-form Laplace CRPS. The PARAMETRIC TWIN
                      of the quantile head (a Laplace point estimate is the MEDIAN, not the
                      mean), so comparing it against ``gaussian`` isolates whether a quantile
                      head's advantage comes from the distributional loss or merely from a
                      robust point estimate. Config ``dist=laplace_crps``.
* ``nll_gaussian`` -- ``(mu, sigma)`` per feature; Gaussian negative log-likelihood. Included
                      deliberately as the NEGATIVE control: it carries the ``1/sigma^2``
                      gradient-scaling pathology ("sigma-collapse"), which the PIT/coverage
                      diagnostic below can MEASURE rather than assume. Config ``dist=nll``.
* ``energy``       -- ``(mu, sigma)`` per feature PLUS a global low-rank factor ``V [F, k]``
                      owned by the head; the multivariate energy score, i.e. the
                      generalization of CRPS to the JOINT distribution over features. The
                      only mode whose score couples features. Config ``dist=energy``.

Layout convention: a structural head emits ``[..., F]`` for ``point`` (unchanged from the
pre-distributional model) or ``[..., F, P]`` for ``P>1`` (``P=2`` gaussian / laplace /
nll_gaussian / energy, ``P=K`` quantile), where the LAST axis holds one feature's
distributional parameters. ``DistHead`` reads that last axis, so the rest of the pipeline --
``[..., F]`` targets, per-feature Pearson computed on ``.point()`` -- is untouched and
ranking stays loss-agnostic.

Calibration is assessed for EVERY probabilistic mode through one common diagnostic, the
probability integral transform (:func:`pit_values`) and its interval companion
(:func:`coverage`).
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
    "laplace_crps": "laplace",
    "nll": "nll_gaussian",
    "energy": "energy",
}

# Default quantile grid: K=19 evenly spaced tau in [0.05, 0.95] (step 0.05, 0.5 included).
# Evenly spaced is the *robust* choice -- no kernel, no bandwidth to tune.
DEFAULT_NUM_QUANTILES = 19

# `energy` mode defaults -- all three MEASURED, not guessed, by
# `experiments/019-simb-multimodal/scripts/residual_covariance_diagnostic.py` on the 019
# expression panel (S=1,482 strains, F=6,169 reporter genes).
#
# RANK. The diagnostic SVDs the standardized kNN residuals and reports the
# participation-ratio effective rank  (sum lambda_i)^2 / sum lambda_i^2 = 32.8, with
# cumulative variance
#     k=8 -> 37.6%    k=16 -> 48.4%    k=32 -> 59.1%    k=64 -> 68.3%
# so the original k=8 could represent barely a third of the reproducible residual structure
# -- it would have understated the joint arm by construction. k=32 sits at the measured
# effective rank and still costs only F*k = 197k parameters (one column block; the sampler
# never forms the F x F Sigma). rank=0 remains the diagonal-only ablation.
DEFAULT_ENERGY_RANK = 32

# SAMPLES. `energy_score` is unbiased in m (the pairwise term divides by m(m-1), not m^2),
# so m buys variance reduction only -- never bias. Both values are MEASURED on the 019 shapes
# (B=32, F=6169, k=32) rather than guessed.
#
# TRAIN m=128. What matters is not MC noise in isolation but its share of the gradient noise
# the optimizer ALREADY absorbs from minibatch sampling. Measuring both on the same SHARED
# parameters (the global V plus a readout -- per-instance params are orthogonal across
# disjoint batch halves by construction and would report a meaningless 100%):
#
#     sigma_batch = 99.8%   (B=32 equivalent)
#     sigma_MC(m) = 150.0 / sqrt(m)     [fit exact: m=8 -> 52.5%, m=256 -> 9.4%]
#
# so the inflation of total gradient noise over minibatch-only is
#
#     rho(m) = sqrt(1 + sigma_MC^2 / sigma_batch^2) = sqrt(1 + 2.25/m)
#
# giving rho = 1.035 (m=32), 1.017 (m=64), 1.009 (m=128) -- matching measurement to three
# decimals. Since noise ~ 1/sqrt(B_eff), rho is equivalent to shrinking the batch by rho^2:
# m=32 trains at an effective batch of 29.9 out of 32.
#
# 128 rather than 32 because sigma_batch was measured on a synthetic readout, and a real
# model near convergence may have SMALLER minibatch noise, which raises MC's share. If the
# true sigma_batch were half what was measured, rho(32) would be 1.13 while rho(128) stays
# 1.035. m=128 is robust across that range and costs 0.04% of a training step -- the encoder
# (4 layers of attention over 6,607 gene tokens, ~4.9 s/step) dominates completely and does
# not amortize with batch size.
#
# EVAL m=256. The empirical CDF has PIT resolution 1/m, so a small m cannot populate a
# histogram or resolve `coverage_50` finely; 256 gives 0.4% resolution, and eval needs no
# gradients so the samples are nearly free.
#
# NOTE the joint ablation does NOT depend on this choice: the measured k=32 vs k=0 energy
# gap is ~3.6 against an MC sd of 0.16 at m=32 (SNR 22.6, and still 8.6 at m=8), so MC noise
# was never what limited the ability to detect whether modelling the covariance pays.
DEFAULT_ENERGY_SAMPLES = 128
DEFAULT_ENERGY_EVAL_SAMPLES = 256
# Init scale of V: per-feature low-rank variance starts at ~V_INIT^2, i.e. small relative to
# the diagonal, so an `energy` head begins near-diagonal and *learns* correlation.
DEFAULT_ENERGY_V_INIT = 0.1


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


def laplace_crps(
    mu: torch.Tensor, b: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Closed-form CRPS of a Laplace :math:`\mathrm{Lap}(\mu,b)` vs observations.

    .. math::
        \mathrm{CRPS}(\mathrm{Lap}(\mu,b), y)
        = |y-\mu| + b\,e^{-|y-\mu|/b} - \tfrac{3}{4}b .

    Like the Gaussian form this is a PROPER scoring rule in the units of ``y`` with no
    :math:`1/b^2` term, so ``b`` cannot be driven to a degenerate optimum.

    Why this mode exists: the Laplace point estimate is the **median**, whereas the
    Gaussian's is the **mean**. It is therefore the parametric twin of the ``quantile``
    head, and the ``laplace`` vs ``gaussian`` contrast is the controlled experiment that
    separates two confounded explanations for a quantile head's advantage -- the
    distributional loss itself, versus simply using a robust (median) point estimate. That
    matters here because expression log2-ratios are sparse and heavy-tailed: most features
    sit at zero and the informative ones are outliers, exactly the regime where a
    mean-seeking Gaussian and a median-seeking Laplace disagree.

    Args:
        mu: Predicted location (the MEDIAN of the predictive distribution), any shape.
        b: Predicted scale (STRICTLY positive), broadcastable to ``mu``. Note ``b`` is the
            Laplace scale, not a standard deviation (:math:`\sigma = \sqrt{2}\,b`).
        target: Observations, broadcastable to ``mu``.

    Returns:
        Elementwise CRPS with the broadcast shape of the inputs (callers reduce).
    """
    d = (target - mu).abs()
    return d + b * torch.exp(-d / b) - 0.75 * b


def gaussian_nll(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Gaussian negative log-likelihood, WITHOUT the constant term.

    .. math::
        \mathrm{NLL}(\mu,\sigma;y)
        = \tfrac{1}{2}\Big(\log\sigma^2 + \frac{(y-\mu)^2}{\sigma^2}\Big) .

    The dropped additive constant is :math:`\tfrac12\log(2\pi)`; it is omitted because it
    shifts every value by the same amount and so changes neither gradients nor the ranking
    of models. Values are therefore NOT comparable to a full log-likelihood in absolute
    terms, and unlike CRPS they can be negative.

    KNOWN PATHOLOGY -- read before using. The :math:`1/\sigma^2` factor multiplies the
    gradient on :math:`\mu`:

    .. math::
        \frac{\partial\,\mathrm{NLL}}{\partial \mu} = -\frac{y-\mu}{\sigma^2} .

    A model can therefore lower its loss on points it fits badly by INFLATING
    :math:`\sigma` there, and doing so simultaneously SHRINKS those same points' gradient
    on :math:`\mu` -- so the hard points stop teaching the mean, the fit stalls, and
    :math:`\sigma` keeps growing to cover the residual. This positive feedback is
    "sigma-collapse" (Seitzer et al. 2022, *On the Pitfalls of Heteroscedastic Uncertainty
    Estimation with Probabilistic Neural Networks*; the beta-NLL fix reweights by
    :math:`\sigma^{2\beta}`). CRPS has no such factor, which is why it is the default.

    It is implemented here UNPATCHED and on purpose: it is the negative control of the
    distributional sweep. Rather than assuming the pathology occurs, we MEASURE it with
    :func:`pit_values` / :func:`coverage` -- sigma-collapse shows up as an n-shaped
    (hump-at-0.5) PIT histogram and over-coverage.

    Args:
        mu: Predicted mean, any shape.
        sigma: Predicted standard deviation (STRICTLY positive), broadcastable to ``mu``.
        target: Observations, broadcastable to ``mu``.

    Returns:
        Elementwise NLL with the broadcast shape of the inputs (callers reduce).
    """
    var = sigma * sigma
    return 0.5 * (torch.log(var) + (target - mu) ** 2 / var)


def energy_score(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    v: torch.Tensor | None,
    target: torch.Tensor,
    num_samples: int = DEFAULT_ENERGY_SAMPLES,
) -> torch.Tensor:
    r"""Monte-Carlo energy score of a low-rank-plus-diagonal Gaussian, per row.

    The energy score is the multivariate proper scoring rule that GENERALIZES CRPS to a
    joint distribution (Gneiting & Raftery 2007):

    .. math::
        \mathrm{ES}(F, y) = \mathbb E\|X-y\|_2 - \tfrac12\,\mathbb E\|X-X'\|_2,
        \qquad X, X' \stackrel{iid}{\sim} F .

    At :math:`F=1` the Euclidean norm is the absolute value and this is exactly the energy
    form of CRPS, so :func:`gaussian_crps` is its ORACLE in one dimension.

    The predictive family is Gaussian with a low-rank-plus-diagonal covariance,

    .. math:: \Sigma = \mathrm{diag}(\sigma^2) + V V^\top,\quad V\in\mathbb R^{F\times k},

    sampled WITHOUT ever forming :math:`\Sigma` (which would be ``F x F``):

    .. math::
        x = \mu + \sigma\odot\varepsilon + V z,\quad
        \varepsilon\sim\mathcal N(0, I_F),\ z\sim\mathcal N(0, I_k).

    The estimator from ``m`` samples is

    .. math::
        \widehat{\mathrm{ES}} = \frac1m\sum_i \|x_i-y\|
        - \frac{1}{2m(m-1)}\sum_{i\neq j}\|x_i-x_j\| ,

    which is UNBIASED for every ``m >= 2``: the pairwise term divides by ``m(m-1)``, not
    ``m^2``. Excluding the ``m`` zero self-distances is exactly what removes the bias.
    Dividing by ``m^2`` instead keeps those zeros in the average, shrinking the SUBTRACTED
    term by a factor ``(m-1)/m`` and so inflating the score by
    :math:`\tfrac{1}{2m}\mathbb E\|X-X'\|`. That surcharge is proportional to the
    predictive SPREAD, so it is not a harmless offset: it penalizes dispersion and moves
    the optimum to an over-sharp (overconfident) distribution, the more so the smaller
    ``m`` is. With the correct denominator ``m`` trades variance only. The pairwise
    distances go through :func:`torch.cdist`, which materializes
    ``[B, m, m]``; the naive ``x[:, :, None] - x[:, None]`` would materialize
    ``[B, m, m, F]`` (~790 MB at ``B=32, m=10, F=6169``) and is never formed.

    Sampling is reparameterized -- ``eps`` and ``z`` are noise, ``mu``/``sigma``/``V`` carry
    the gradient -- so the score is differentiable end to end. Nothing is detached.

    ``k = 0`` (``v is None`` or ``v.shape[-1] == 0``) is the ABLATION arm and the
    distinction is worth stating precisely: the SCORE still couples coordinates (the
    Euclidean norm is taken over all ``F`` features jointly), but the predictive FAMILY is
    diagonal, so it cannot represent any correlation between features. Comparing ``k = 0``
    against ``k > 0`` therefore separates "scoring jointly" from "modelling jointly".

    Args:
        mu: Predicted means ``[B, F]``.
        sigma: Predicted marginal scales ``[B, F]`` (STRICTLY positive); the diagonal part.
        v: Global low-rank factor ``V`` ``[F, k]``, SHARED across the batch (a learned
            feature-correlation basis, not a per-row output), or ``None`` for ``k = 0``.
        target: Observations ``[B, F]``.
        num_samples: ``m``, the number of predictive samples. Must be ``>= 2``.

    Returns:
        Per-row energy score ``[B]`` (callers reduce over the batch).
    """
    assert mu.dim() == 2, f"energy_score expects mu [B, F]; got {tuple(mu.shape)}"
    assert num_samples >= 2, "the unbiased pairwise term requires num_samples >= 2"
    b_rows, n_feat = mu.shape
    m = num_samples
    eps = torch.randn(m, b_rows, n_feat, device=mu.device, dtype=mu.dtype)
    x = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    if v is not None and v.shape[-1] > 0:
        assert v.shape[0] == n_feat, (
            f"V must be [F, k]; got {tuple(v.shape)}, F={n_feat}"
        )
        z = torch.randn(m, b_rows, v.shape[-1], device=mu.device, dtype=mu.dtype)
        x = x + z @ v.transpose(0, 1)
    x = x.permute(1, 0, 2)  # [B, m, F]
    dist: torch.Tensor = torch.linalg.vector_norm(x - target.unsqueeze(1), dim=-1)
    term1 = dist.mean(dim=1)
    # cdist's diagonal is exactly zero, so summing the FULL [m, m] matrix equals the
    # sum over i != j; the m(m-1) denominator is what makes the estimator unbiased.
    pair = torch.cdist(x, x)
    term2 = pair.sum(dim=(1, 2)) / (2.0 * m * (m - 1))
    return term1 - term2


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


def masked_mean(elem: torch.Tensor, feature_mask: torch.Tensor | None) -> torch.Tensor:
    r"""Mean of ``elem`` over the entries ``feature_mask`` selects.

    WHY THIS EXISTS. The masked-label objective scores a prediction ONLY on the genes that
    are still hidden at unmasking step :math:`k`. Genes whose true value has been revealed
    are model INPUT at that step, so including them in the loss lets the network copy the
    input to the output -- train loss collapses, train Pearson inflates, and none of it
    transfers to validation, where nothing is revealed. Restricting the reduction is the
    whole correctness requirement of that objective.

    ``feature_mask`` is ``[B, F]`` (True = score this entry) while ``elem`` may be
    ``[B, F]`` (point) or ``[B, F, K]`` (quantile / distributional), so the mask is
    broadcast along any trailing axes -- a quantile knot is scored exactly when its gene is.

    Args:
        elem: Elementwise loss ``[B, F]`` or ``[B, F, ...]``.
        feature_mask: Optional bool ``[B, F]``; ``None`` reduces over everything.

    Returns:
        Scalar mean over selected entries. Returns a graph-connected zero when the mask
        selects nothing, so DDP find-unused stays happy.
    """
    if feature_mask is None:
        return elem.mean()
    w = feature_mask.to(elem.dtype)
    while w.dim() < elem.dim():
        w = w.unsqueeze(-1)
    w = w.expand_as(elem)
    denom = w.sum()
    if float(denom) == 0.0:
        return elem.sum() * 0.0
    return (elem * w).sum() / denom


def _quantile_pit(
    quantiles: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    r"""PIT from a quantile head: the empirical CDF interpolated through the knots.

    The head gives ``K`` points :math:`(\hat q_{\tau_k}, \tau_k)` of the predictive CDF.
    :math:`\hat F(y)` is the piecewise-linear interpolant through them. Outside the knot
    range the grid carries NO information, so values below the lowest knot map to ``0`` and
    values above the highest to ``1`` -- no tail shape is invented. With the default
    ``tau in [0.05, 0.95]`` grid a perfectly calibrated head therefore puts ~5% of its PIT
    mass exactly at each endpoint; that is a property of the grid, not miscalibration.

    Args:
        quantiles: Predicted quantiles ``[..., F, K]`` (last axis indexes ``taus``).
        target: Observations ``[..., F]``.
        taus: Quantile levels ``[K]``, ascending, in ``(0, 1)``.

    Returns:
        PIT values ``[..., F]`` in ``[0, 1]``.
    """
    # Sorting repairs quantile crossing (nothing constrains the head's K outputs to be
    # monotone), which the interpolation below requires.
    q, _ = torch.sort(quantiles, dim=-1)
    k = q.shape[-1]
    y = target.unsqueeze(-1)
    idx = torch.searchsorted(q.contiguous(), y.contiguous())  # [..., F, 1] in [0, K]
    lo = (idx - 1).clamp(0, k - 1)
    hi = idx.clamp(0, k - 1)
    q_lo = q.gather(-1, lo)
    q_hi = q.gather(-1, hi)
    t = taus.to(q.dtype).to(q.device).expand_as(q)
    t_lo = t.gather(-1, lo)
    t_hi = t.gather(-1, hi)
    gap = q_hi - q_lo
    # gap == 0 only at the two boundaries (lo == hi), which are overwritten below.
    safe = torch.where(gap > 0, gap, torch.ones_like(gap))
    w = ((y - q_lo) / safe).clamp(0.0, 1.0)
    pit = t_lo + w * (t_hi - t_lo)
    # Only STRICTLY outside the knot range is clamped. Landing exactly ON the lowest or
    # highest knot collapses lo == hi, which the w = 0 / w = 1 branches already resolve to
    # that knot's tau -- the interpolant stays continuous at the ends, as a CDF must.
    pit = torch.where(y < q[..., :1], torch.zeros_like(pit), pit)
    pit = torch.where(y > q[..., -1:], torch.ones_like(pit), pit)
    return pit.squeeze(-1)


def pit_values(
    mode: str,
    target: torch.Tensor,
    mu: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    quantiles: torch.Tensor | None = None,
    taus: torch.Tensor | None = None,
    samples: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Probability integral transform -- ONE calibration diagnostic for every mode.

    .. math:: \mathrm{PIT}_{s,f} = \hat F_{s,f}(y_{s,f}) \in [0,1],

    the predictive CDF of instance ``s``, feature ``f``, evaluated at its own observation.
    Its whole point is mode-independence: a Gaussian CRPS head, a quantile head and a
    sample-based energy head all produce PIT values on the same ``[0, 1]`` scale, so
    calibration can be compared ACROSS the distributional sweep even though the losses are
    not comparable to each other.

    Per mode:

    * ``gaussian`` / ``nll_gaussian``: :math:`\Phi\!\big((y-\mu)/\sigma\big)`;
    * ``laplace``: the Laplace CDF at ``y``;
    * ``quantile``: the empirical CDF interpolated through the ``K`` knots
      (see :func:`_quantile_pit`);
    * ``energy``: :math:`\frac1m\sum_i \mathbf 1[x_{i,f}\le y_f]` from predictive samples
      (per feature -- the MARGINAL calibration of a joint model).

    If the predictive distribution is correct then ``PIT ~ Uniform(0, 1)``. Read the
    histogram shape as a diagnosis:

    * **flat** -- calibrated;
    * **U-shaped** (mass piled at 0 and 1) -- too many observations land in the tails, so
      the intervals are too NARROW: OVERCONFIDENT / under-dispersed;
    * **n-shaped** (hump at 0.5) -- observations cluster near the predictive median, so the
      intervals are too WIDE: UNDERCONFIDENT / over-dispersed. This is the signature of
      NLL sigma-collapse (see :func:`gaussian_nll`);
    * **skewed / shifted** (mass toward 0 or toward 1) -- the location is biased, i.e. the
      predicted mean is systematically high or low.

    Args:
        mode: A :class:`DistHead` mode. ``point`` has no predictive distribution and is
            rejected.
        target: Observations ``[..., F]``.
        mu: ``gaussian`` / ``nll_gaussian`` / ``laplace`` -- predicted location ``[..., F]``.
        scale: ``gaussian`` / ``nll_gaussian`` -- ``sigma``; ``laplace`` -- ``b``. Already
            positive (this function does NOT apply the softplus; pass ``DistHead``-
            transformed values, e.g. via :meth:`DistHead.pit`).
        quantiles: ``quantile`` -- predicted quantiles ``[..., F, K]``.
        taus: ``quantile`` -- the levels ``[K]``.
        samples: ``energy`` -- predictive samples ``[..., m, F]``.

    Returns:
        PIT values ``[..., F]`` in ``[0, 1]``.
    """
    if mode in ("gaussian", "nll_gaussian"):
        assert mu is not None and scale is not None, f"{mode} PIT needs mu and scale"
        return 0.5 * (1.0 + torch.erf((target - mu) / scale * _INV_SQRT_2))
    if mode == "laplace":
        assert mu is not None and scale is not None, "laplace PIT needs mu and scale"
        z = (target - mu) / scale
        # exp(-|z|) form: algebraically identical to the two-branch CDF, overflow-free.
        half = 0.5 * torch.exp(-z.abs())
        return torch.where(z < 0, half, 1.0 - half)
    if mode == "quantile":
        assert quantiles is not None and taus is not None, (
            "quantile PIT needs quantiles and taus"
        )
        return _quantile_pit(quantiles, target, taus)
    if mode == "energy":
        assert samples is not None, "energy PIT needs samples [..., m, F]"
        le = (samples <= target.unsqueeze(-2)).to(target.dtype)
        return le.mean(dim=-2)
    raise ValueError(f"mode {mode!r} has no predictive CDF, so no PIT")


def coverage(pit: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""Empirical coverage of the central ``alpha`` predictive interval, from PIT values.

    An observation lies inside the central ``alpha`` interval
    :math:`[\hat F^{-1}(\tfrac{1-\alpha}{2}),\ \hat F^{-1}(\tfrac{1+\alpha}{2})]` exactly
    when its PIT lies in :math:`[\tfrac{1-\alpha}{2}, \tfrac{1+\alpha}{2}]`, so coverage is
    a scalar read-off of the same transform rather than a second computation:

    .. math::
        \mathrm{cov}_\alpha
        = \frac1N\sum \mathbf 1\!\left[\tfrac{1-\alpha}{2}\le \mathrm{PIT}\le
          \tfrac{1+\alpha}{2}\right].

    Calibrated means :math:`\mathrm{cov}_\alpha \approx \alpha`. UNDER-coverage
    (:math:`<\alpha`) is the U-shaped-PIT / overconfident case; OVER-coverage
    (:math:`>\alpha`) is the n-shaped / sigma-collapse case.

    Args:
        pit: PIT values, any shape (all entries are pooled).
        alpha: Nominal central interval mass in ``(0, 1)``, e.g. ``0.8``.

    Returns:
        Scalar coverage fraction.
    """
    assert 0.0 < alpha < 1.0, f"alpha must be in (0, 1); got {alpha}"
    lo = 0.5 * (1.0 - alpha)
    hi = 0.5 * (1.0 + alpha)
    inside = (pit >= lo) & (pit <= hi)
    return inside.to(pit.dtype).mean()


def pit_ks(pit: torch.Tensor) -> torch.Tensor:
    r"""Kolmogorov-Smirnov distance between the PIT distribution and ``Uniform(0,1)``.

    One scalar for the whole PIT histogram -- the sup-norm gap between the empirical CDF of
    the pooled PIT values and the uniform CDF that perfect calibration implies:

    .. math::
        D = \sup_{u\in[0,1]} \big| F_N(u) - u \big|
          = \max_{i=1..N} \max\!\Big(\tfrac{i}{N} - p_{(i)},\; p_{(i)} - \tfrac{i-1}{N}\Big),

    with :math:`p_{(1)}\le\dots\le p_{(N)}` the sorted PIT values. ``0`` is perfect; larger
    is worse, in either direction (it is a distance, so it does not say WHICH way -- read
    :func:`coverage` or the histogram shape for that).

    **Comparable WITHIN a head, not ACROSS distributional modes.** A calibrated ``quantile``
    head on the default ``tau in [0.05, 0.95]`` grid parks ~5% of its PIT mass exactly at
    ``0`` and ~5% exactly at ``1`` (see :func:`_quantile_pit`), which alone forces
    :math:`D \approx 0.05` even when the head is perfect. A parametric head has no such
    floor. Comparing ``pit_ks`` between a quantile arm and a Gaussian arm therefore measures
    the grid, not the model. Use :func:`coverage` for cross-mode comparison -- it evaluates
    the CDF at interior points where no mode has a structural artifact -- and use
    ``pit_ks`` to track one head across epochs or hyperparameters.

    Args:
        pit: PIT values, any shape (all entries are pooled and flattened).

    Returns:
        Scalar KS distance in ``[0, 1]``.
    """
    p, _ = torch.sort(pit.flatten())
    n = p.numel()
    assert n > 0, "pit_ks needs at least one PIT value"
    i = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
    above = i / n - p  # ECDF just AFTER each point minus uniform
    below = p - (i - 1) / n  # uniform minus ECDF just BEFORE it
    return torch.maximum(above.max(), below.max())


class DistHead(nn.Module):
    """Pluggable distributional readout: interpret a head's raw params, score them, pool.

    Stateless except for the quantile grid (``quantile`` mode) and the global low-rank
    factor ``V`` (``energy`` mode). Given a structural head's output ``params`` -- ``[..., F]``
    (point) or ``[..., F, P]`` (``P=2`` gaussian / laplace / nll_gaussian / energy, ``P=K``
    quantile) -- it exposes:

    * :meth:`point` -- the point estimate ``[..., F]`` used by the (loss-agnostic) metric
      and any downstream reporting: identity (point), ``mu`` (gaussian / nll_gaussian /
      energy), the predictive MEDIAN ``mu`` (laplace), or the median quantile (quantile);
    * :meth:`loss` -- a masked scalar training loss: MSE (point), mean Gaussian CRPS
      (gaussian), mean pinball (quantile), mean Laplace CRPS (laplace), mean Gaussian NLL
      (nll_gaussian), or mean energy score (energy);
    * :meth:`pit` -- per-feature PIT values ``[B, F]`` for the calibration diagnostic
      (every mode but ``point``).

    ``energy`` is the one mode with learnable state of its own: ``V`` is an
    ``nn.Parameter`` of shape ``[num_features, rank]`` that is GLOBAL (shared across the
    batch), because a correlation basis over features is a property of the phenotype space,
    not of an individual strain. It therefore does NOT widen ``param_dim`` -- the structural
    head still emits ``(mu, sigma)``, i.e. ``P=2``, and only the head owns ``V``.
    """

    VALID_MODES = ("point", "gaussian", "quantile", "laplace", "nll_gaussian", "energy")

    def __init__(
        self,
        mode: str,
        quantiles: torch.Tensor | list[float] | None = None,
        sigma_floor: float = 1e-3,
        num_features: int | None = None,
        rank: int = DEFAULT_ENERGY_RANK,
        num_samples: int = DEFAULT_ENERGY_SAMPLES,
        v_init: float = DEFAULT_ENERGY_V_INIT,
    ) -> None:
        """Build the head.

        Args:
            mode: One of ``point`` / ``gaussian`` / ``quantile`` / ``laplace`` /
                ``nll_gaussian`` / ``energy``.
            quantiles: ``quantile`` mode only -- the ``tau`` grid (defaults to K=19 evenly
                spaced in ``[0.05, 0.95]``). Ignored otherwise.
            sigma_floor: additive floor on ``softplus(raw)`` so the scale (``sigma``, or the
                Laplace ``b``) stays strictly positive. Ignored by ``point``/``quantile``.
            num_features: ``energy`` mode only -- ``F``, the number of features, needed to
                allocate the global factor ``V [F, rank]``. REQUIRED for ``energy``.
            rank: ``energy`` mode only -- ``k``, the rank of ``V`` (default 32, the
                measured effective rank of the 019 residuals). ``rank=0`` is the
                diagonal-only ABLATION: ``V`` is ``None``, so the score still couples
                features but the family cannot represent correlation.
            num_samples: ``energy`` mode only -- ``m``, TRAINING predictive samples per
                score. The estimator is unbiased in ``m``, so this trades variance for
                compute. :meth:`pit` uses the larger eval default instead.
            v_init: ``energy`` mode only -- std of the ``V`` init, scaled by
                ``1/sqrt(rank)`` so the initial per-feature low-rank variance is
                ``~v_init^2`` regardless of ``rank``.
        """
        super().__init__()
        assert mode in self.VALID_MODES, f"unknown DistHead mode {mode!r}"
        self.mode = mode
        self.sigma_floor = float(sigma_floor)
        self.num_samples = int(num_samples)
        # Declared once so every mode has the same attribute type; only `energy` with
        # rank > 0 fills it (register_parameter(None) keeps state_dict uniform otherwise).
        self._v: nn.Parameter | None
        self.register_parameter("_v", None)
        if mode == "energy":
            assert num_features is not None, "energy mode requires num_features (F)"
            assert rank >= 0, f"energy rank must be >= 0; got {rank}"
            if rank > 0:
                self._v = nn.Parameter(
                    torch.randn(num_features, rank) * (v_init / math.sqrt(rank))
                )
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
    def v(self) -> torch.Tensor | None:
        """The global low-rank factor ``V [F, k]`` (``energy`` mode) or ``None``.

        ``None`` for every non-``energy`` mode and for ``energy`` with ``rank=0`` (the
        diagonal-only ablation). Surfaced through a property for the same reason as
        :attr:`taus`: ``nn.Module.__getattr__`` is typed ``Tensor | Module``.
        """
        return self._v

    @property
    def param_dim(self) -> int:
        """Params per feature: 1 (point), K (quantile), or 2 (every parametric mode).

        ``energy`` is 2 as well -- its extra state is the GLOBAL ``V``, owned by this head,
        not per-feature output of the structural head.
        """
        if self.mode == "point":
            return 1
        if self.mode == "quantile":
            return int(self.taus.numel())
        return 2

    def _sigma(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a raw scale param to a strictly-positive scale via softplus + floor.

        Shared by every parametric mode: ``sigma`` for gaussian / nll_gaussian / energy and
        the Laplace ``b``, so positivity is enforced identically everywhere.
        """
        return F.softplus(raw) + self.sigma_floor

    def point(self, params: torch.Tensor) -> torch.Tensor:
        """Reduce raw params to a point estimate ``[..., F]`` (metric / reporting).

        Identity (point), the median knot (quantile), or ``mu`` for every parametric mode --
        which is the predictive MEAN for gaussian / nll_gaussian / energy and the predictive
        MEDIAN for laplace.
        """
        if self.mode == "point":
            return params
        if self.mode == "quantile":
            return params[..., self.median_index]
        return params[..., 0]

    def _location_scale(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ``[..., F, 2]`` raw params into ``(mu, positive scale)``."""
        return params[..., 0], self._sigma(params[..., 1])

    def samples(self, params: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Draw ``energy``-mode predictive samples ``[B, m, F]`` (reparameterized).

        Args:
            params: Head params ``[B, F, 2]``.
            num_samples: ``m``, samples per row.

        Returns:
            Samples ``[B, m, F]`` from ``N(mu, diag(sigma^2) + V V^T)``.
        """
        assert self.mode == "energy", "samples() is energy-mode only"
        mu, sigma = self._location_scale(params)
        eps = torch.randn(num_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        x = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        v = self.v
        if v is not None and v.shape[-1] > 0:
            z = torch.randn(
                num_samples, mu.shape[0], v.shape[-1], device=mu.device, dtype=mu.dtype
            )
            x = x + z @ v.to(mu.device).transpose(0, 1)
        return x.permute(1, 0, 2)

    def loss(
        self,
        params: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Masked scalar training loss for this head.

        Args:
            params: Head params ``[B, F]`` (point) or ``[B, F, P]`` (every other mode).
            target: Point targets ``[B, F]`` (same F as ``params``).
            mask: Optional bool ``[B]`` selecting supervised rows. When no row is
                supervised, a graph-connected zero is returned (grads flow as zero).
            feature_mask: Optional bool ``[B, F]`` selecting which FEATURES to score.
                Used by the masked-label objective to score only the genes still hidden
                at the current unmasking step -- scoring a revealed gene would let the
                model copy its own input. Row-masked alongside ``params``/``target`` so
                the two stay aligned.

        Returns:
            Scalar loss.
        """
        if mask is not None:
            params = params[mask]
            target = target[mask]
            if feature_mask is not None:
                feature_mask = feature_mask[mask]
        if target.shape[0] == 0:
            # Keep the head connected to the graph so DDP find-unused stays happy -- V too,
            # since it is a parameter of this head and would otherwise go untouched.
            zero = params.sum() * 0.0
            v = self.v
            if v is not None:
                zero = zero + v.sum() * 0.0
            return zero
        if self.mode == "point":
            return masked_mean((params - target) ** 2, feature_mask)
        if self.mode == "quantile":
            return masked_mean(
                pinball(params, target, self.taus.to(params.device)), feature_mask
            )
        mu, scale = self._location_scale(params)
        if self.mode == "gaussian":
            return masked_mean(gaussian_crps(mu, scale, target), feature_mask)
        if self.mode == "laplace":
            return masked_mean(laplace_crps(mu, scale, target), feature_mask)
        if self.mode == "nll_gaussian":
            return masked_mean(gaussian_nll(mu, scale, target), feature_mask)
        v = self.v
        return energy_score(
            mu,
            scale,
            None if v is None else v.to(params.device),
            target,
            self.num_samples,
        ).mean()

    def pit(
        self,
        params: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Per-feature PIT values ``[B, F]`` for this head's predictive distribution.

        The mode-agnostic calibration diagnostic -- see :func:`pit_values` for how to read
        the histogram and :func:`coverage` for the interval summary. Not available in
        ``point`` mode (no predictive distribution).

        Args:
            params: Head params ``[B, F]`` (point) or ``[B, F, P]``.
            target: Point targets ``[B, F]``.
            mask: Optional bool ``[B]`` selecting rows, as in :meth:`loss`.
            num_samples: ``energy`` mode only -- samples used to build the empirical CDF.
                Defaults to :data:`DEFAULT_ENERGY_EVAL_SAMPLES`, NOT the head's training
                ``m``: PIT resolution is ``1/m``, so the (deliberately cheap) training
                ``m`` would quantize the histogram far too coarsely to read. Eval needs no
                gradients, so the extra samples are nearly free.

        Returns:
            PIT values ``[B, F]`` in ``[0, 1]``.
        """
        if mask is not None:
            params = params[mask]
            target = target[mask]
        if self.mode == "quantile":
            return pit_values(
                self.mode, target, quantiles=params, taus=self.taus.to(params.device)
            )
        if self.mode == "energy":
            m = DEFAULT_ENERGY_EVAL_SAMPLES if num_samples is None else num_samples
            return pit_values(self.mode, target, samples=self.samples(params, m))
        mu, scale = self._location_scale(params)
        return pit_values(self.mode, target, mu=mu, scale=scale)


def dist_param_dim(dist: str, num_quantiles: int = DEFAULT_NUM_QUANTILES) -> int:
    """Params-per-feature for a config ``dist`` value (the head Linear-width multiplier).

    ``energy`` is 2 because its low-rank factor ``V`` is owned by the :class:`DistHead`
    (global, ``[F, k]``), not emitted per feature by the structural head.
    """
    return {
        "point": 1,
        "crps": 2,
        "quantile": num_quantiles,
        "laplace_crps": 2,
        "nll": 2,
        "energy": 2,
    }[dist]


def make_dist_head(
    dist: str,
    num_quantiles: int = DEFAULT_NUM_QUANTILES,
    num_features: int | None = None,
    rank: int = DEFAULT_ENERGY_RANK,
    num_samples: int = DEFAULT_ENERGY_SAMPLES,
) -> DistHead:
    """Build a :class:`DistHead` from a config ``dist`` value.

    Args:
        dist: One of ``point`` / ``crps`` / ``quantile`` / ``laplace_crps`` / ``nll`` /
            ``energy`` (the keys of :data:`DIST_TO_MODE`).
        num_quantiles: ``quantile`` only -- size of the ``tau`` grid.
        num_features: ``energy`` only -- ``F``; REQUIRED there to allocate ``V [F, rank]``.
        rank: ``energy`` only -- ``k`` (``0`` = the diagonal-only ablation).
        num_samples: ``energy`` only -- ``m`` predictive samples per score.

    Returns:
        The configured head.
    """
    mode = DIST_TO_MODE[dist]
    if mode == "quantile":
        return DistHead(mode, quantiles=torch.linspace(0.05, 0.95, num_quantiles))
    if mode == "energy":
        return DistHead(
            mode, num_features=num_features, rank=rank, num_samples=num_samples
        )
    return DistHead(mode)

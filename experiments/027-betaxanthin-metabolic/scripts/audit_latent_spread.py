# experiments/027-betaxanthin-metabolic/scripts/audit_latent_spread.py
# [[experiments.027-betaxanthin-metabolic.wave8-sampling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_latent_spread.py

r"""How much of its own polytope does the flux layer's latent actually cover?

THE QUESTION THIS ANSWERS, and why it comes before any GPU is spent. Flux Cone Learning
(Merzbacher 2025) reads a deletion's phenotype off the GEOMETRY of its flux cone, sampled
with OptGPSampler at 124 samples per cone. Our layer draws ONE latent per genotype per
forward pass. The obvious next experiment is K draws per pass, but K draws are K copies of
one point unless the posterior spread :math:`\sigma` is large compared with the polytope
the draws are supposed to explore. This script measures that ratio, on trained weights,
before the experiment is run.

Four measurements per audited cell, each on real genotypes from the pinned dataset:

1. **What the arm actually ran.** ``latent_stochastic``, the layer's ``kl_weight``, the
   constraint scale that multiplies it (the KL is inside the ramped sum, so the effective
   weight is ``kl_weight * constraint_scale``), and ``c_kl`` at the first and last epoch.
   Because :math:`c_{\mathrm{KL}}=\tfrac12\overline{(\mu^2+\sigma^2-1-2\ln\sigma)}`, its
   trajectory already says whether :math:`\sigma` grew or shrank during training.

2. **The posterior itself.** :math:`\mu` and :math:`\sigma` on one real batch, with the KL
   split into its :math:`\mu^2` and its :math:`\sigma` halves so a rising KL is attributed
   to the term that caused it rather than assumed.

3. **The polytope, along the directions the latent can move in.** For one genotype, the
   CHORD the box allows along random ADMISSIBLE latent directions: directions that leave
   every zero-width reaction where its bounds pin it. Per-coordinate chords are useless
   here and the reason is measured rather than assumed -- every basis direction touches
   some zero-width reaction, so every per-coordinate chord is exactly zero. The ratio
   ``sigma_rms / chord`` is the share of the available run one draw covers; at 1e-3 the
   sampler is moving inside a ball a thousand times smaller than the polytope it sits in.

4. **What the draws do downstream**, which is the only part the phenotype can see: 64
   draws through :meth:`FluxLayer.sample`, then the per-reaction sd across draws against
   the box width, and the scalar head's prediction sd ACROSS DRAWS against its sd ACROSS
   GENOTYPES. If the second ratio is ~0, averaging over K draws returns the K = 1
   prediction and no phenotype signal can come from sampling depth.

Run from the worktree root (CPU is enough and is the default, so a running training job
keeps the GPU)::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/audit_latent_spread.py

Writes ``results/wave8_latent_spread.json`` and a figure under
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    FluxMetaboliteHead,
    FluxScalarHead,
    perturbed_gene_pool,
)
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg  # noqa: E402

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

#: The cells audited by default, and what each one is here to answer.
#:
#: * ``wave6_proj_flux_projection_s2700`` -- the plain projection head at constraint scale
#:   zero, the family the wave-8 K-draw cells are paired against.
#: * ``wave6_proj_flux_projection_anchored_tdg_stoch_ue_s2700`` -- the ONLY 027 cell trained
#:   with the latent draw on, so the only one whose sigma was ever given a gradient.
#: * ``wave7_aa_turnover_e200_s2700`` -- the amino-acid TURNOVER readout, which reads the
#:   flux vector and so could in principle respond to a draw.
#: * ``wave7_aa_conc_e200_s2700`` -- the amino-acid CONCENTRATION readout. Its prediction is
#:   a slice of the layer's ``log c``, a function of the genotype context alone, so no draw
#:   can move it. It is audited to record that rather than to leave it inferred.
DEFAULT_CELLS: list[str] = [
    "wave6_proj_flux_projection_s2700.json",
    "wave6_proj_flux_projection_anchored_tdg_stoch_ue_s2700.json",
    "wave7_aa_turnover_e200_s2700.json",
    "wave7_aa_conc_e200_s2700.json",
]


class PosteriorSpread(BaseModel):
    """The latent posterior on one real batch, and the KL split into its two halves."""

    model_config = ConfigDict(extra="forbid")

    n_genotypes: int
    n_latent: int
    sigma_min: float
    sigma_p10: float
    sigma_p50: float
    sigma_p90: float
    sigma_max: float
    sigma_across_genotype_sd_median: float = Field(
        description=(
            "Median over latent dimensions of the sd of sigma ACROSS genotypes. The spread "
            "head is initialized with zero weights, so this reads exactly 0 for a cell "
            "whose sigma never received a gradient."
        )
    )
    mu_abs_p50: float
    mu_abs_p90: float
    sigma_over_abs_mu_p50: float
    kl_total: float
    kl_mu_term: float = Field(description="0.5 * mean(mu^2).")
    kl_sigma_term: float = Field(description="0.5 * mean(sigma^2 - 1 - 2 ln sigma).")


class RayExtent(BaseModel):
    r"""How far the box lets a latent step travel, along random ADMISSIBLE directions.

    WHY NOT PER BASIS DIRECTION. The obvious measure, the chord along each basis direction
    :math:`n_i`, is identically zero here and says nothing: the null basis is built from the
    full stoichiometric matrix and knows nothing about the bounds, so every one of the 1,538
    directions moves at least one reaction whose box has zero width (``lb = ub``, a blocked
    or deleted reaction), and a single-coordinate step is therefore blocked by definition.
    Measured on all four audited cells: ``frac_directions_blocked`` is exactly 1.000, with
    as few as 10 zero-width reactions. The polytope is not degenerate; the coordinates are
    the wrong ones to read it in.

    WHAT IS MEASURED INSTEAD. The admissible subspace :math:`\{d:(\mathcal N d)_j=0\ \forall
    j\in Z\}`, with :math:`Z` the zero-width reactions, is the set of latent moves that keep
    every blocked reaction at its fixed value. Its dimension is reported. Random unit
    directions are drawn inside it FROM THE BOX CENTER, and the chord is the distance to the
    first face in either direction. A single draw displaces the latent by
    :math:`\sqrt{\sum_i\sigma_i^2d_i^2}\approx\sigma_{\mathrm{rms}}` along such a
    direction, so ``sigma_rms / chord`` is the share of the available run one draw covers.

    THE BASE POINT IS THE BOX CENTER AND NOT THE MODEL'S OWN FLUX, which is a measured
    necessity rather than a convenience. The head's flux sits ON a bound for a large
    fraction of reactions (reported as ``frac_reactions_at_bound``), so a ray from it has a
    chord of exactly zero in whichever direction touches any one of them, and every random
    direction touches some. The center chord measures the BOX's extent along the direction,
    which is an upper bound on the polytope's, and is a property of the genotype's bounds
    alone.
    """

    model_config = ConfigDict(extra="forbid")

    genotype_row: int
    n_latent: int
    n_reactions_zero_width: int
    frac_basis_directions_blocked: float
    admissible_dimension: int
    n_rays: int
    sigma_rms: float
    frac_reactions_at_bound: float
    chord_p10: float
    chord_p50: float
    chord_p90: float
    sigma_rms_over_chord_p50: float
    frac_rays_sigma_exceeds_chord: float


class DrawSpread(BaseModel):
    """What ``n_draws`` draws do to the flux and to the phenotype the head reads off it."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int
    n_genotypes: int
    flux_sd_over_box_width_p50: float
    flux_sd_over_box_width_p90: float
    uniform_coverage_p50: float = Field(
        description=(
            "The p50 across-draw flux sd as a MULTIPLE of the sd a uniform sample of that "
            "reaction's own bound interval would have, which is 1/sqrt(12) = 0.2887 of the "
            "width. 1.0 is marginal coverage of the box; 0.0 is a point mass. It calibrates "
            "the raw fraction, which on its own does not say whether 0.15 is a lot."
        )
    )
    frac_reactions_moved: float = Field(
        description=(
            "Fraction of (genotype, reaction) pairs whose across-draw sd exceeds 1e-6 of "
            "the layer's flux scale, i.e. reactions the sampler moves at all."
        )
    )
    readout_reads_flux: bool = Field(
        description=(
            "Whether the arm's supervised readout is a function of the flux vector at all. "
            "The concentration readout emits a slice of the layer's learned log c, which "
            "depends on the genotype context and not on the drawn flux, so no number of "
            "draws can move its prediction."
        )
    )
    prediction_sd_across_draws_median: float | None
    prediction_sd_across_genotypes: float | None
    prediction_draw_to_genotype_ratio: float | None = Field(
        default=None,
        description=(
            "The decision number. Median across-draw sd of the head's prediction divided "
            "by the across-genotype sd of the same prediction. At 0 the K-draw average IS "
            "the K = 1 prediction. Null when the readout does not read the flux."
        ),
    )


class CellSpread(BaseModel):
    """One audited cell: what it ran, and how much of its polytope its latent covers."""

    model_config = ConfigDict(extra="forbid")

    result_file: str
    arm: str
    seed: int
    epochs: int
    parameterization_raw: str
    projection_steps: int
    projection_end: str
    flux_scale: float
    latent_log_std_init: float
    latent_stochastic_trained: bool
    kl_weight: float
    constraint_final_scale: float
    effective_kl_weight: float
    c_kl_first_epoch: float | None
    c_kl_last_epoch: float | None
    test_spearman_pinned: float | None
    posterior: PosteriorSpread
    rays: RayExtent
    draws: DrawSpread


def _load_train_bx() -> Any:
    """Import ``train_bx.py`` by path, as the other 027 audits do."""
    spec = importlib.util.spec_from_file_location(
        "train_bx_for_latent_audit", osp.join(EXP_DIR, "scripts", "train_bx.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def ray_extent(
    null_basis: torch.Tensor,
    v: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    sigma: torch.Tensor,
    flux_scale: float,
    row: int,
    n_rays: int,
    seed: int,
) -> RayExtent:
    r"""Chord lengths along random admissible latent directions, and sigma against them.

    The admissible subspace is the kernel of :math:`\mathcal N_{Z,:}` with :math:`Z` the
    zero-width reactions, obtained from an SVD of that (few hundred by 1,538) block, so a
    move inside it leaves every blocked reaction exactly where the bounds pin it. Rays start
    at the model's own flux CLIPPED into the box, because the projection loop of a
    ``balance``-ended cell returns a point that is off the box and every chord from there
    would read zero for a reason that is about the head rather than about the polytope.
    """
    basis = null_basis.to(torch.float64)
    upper = ub[row].to(torch.float64)
    lower = lb[row].to(torch.float64)
    width = (upper - lower).clamp(min=0.0)
    v_model = v[row].to(torch.float64)
    at_bound = (
        (v_model >= upper - 1.0e-9 * width.clamp(min=1.0))
        | (v_model <= lower + 1.0e-9 * width.clamp(min=1.0))
    ) & (width > 1.0e-12)
    # THE BOX CENTER, strictly interior wherever the box has width, so a ray from it has a
    # positive chord. See the class note for why the model's own flux cannot serve.
    v0 = 0.5 * (lower + upper)
    zero_width = width <= 1.0e-12
    step_basis = basis * flux_scale
    blocked = ((step_basis.abs() > 1.0e-8) & zero_width.unsqueeze(1)).any(dim=0)

    n_latent = int(basis.shape[1])
    if bool(zero_width.any()):
        block = basis[zero_width]
        _, singular, vh = torch.linalg.svd(block, full_matrices=True)
        rank = (
            int((singular > 1.0e-8 * float(singular.max())).sum())
            if singular.numel()
            else 0
        )
        admissible = vh[rank:].transpose(0, 1)
    else:
        admissible = torch.eye(n_latent, dtype=torch.float64)
    dim = int(admissible.shape[1])
    assert dim > 0, (
        "the zero-width reactions leave no admissible latent direction at all."
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.randn(dim, n_rays, generator=generator, dtype=torch.float64)
    directions = admissible @ coefficients
    directions = directions / directions.norm(dim=0, keepdim=True).clamp(min=1.0e-30)
    step = (basis @ directions) * flux_scale
    slack_up = (upper - v0).clamp(min=0.0).unsqueeze(1)
    slack_down = (v0 - lower).clamp(min=0.0).unsqueeze(1)
    big = torch.tensor(float("inf"), dtype=torch.float64)
    pos = step > 1.0e-8
    neg = step < -1.0e-8
    t_plus = torch.minimum(
        torch.where(pos, slack_up / step.clamp(min=1.0e-8), big).min(dim=0).values,
        torch.where(neg, slack_down / (-step).clamp(min=1.0e-8), big).min(dim=0).values,
    )
    t_minus = torch.minimum(
        torch.where(pos, slack_down / step.clamp(min=1.0e-8), big).min(dim=0).values,
        torch.where(neg, slack_up / (-step).clamp(min=1.0e-8), big).min(dim=0).values,
    )
    chord = torch.minimum(t_plus, t_minus)
    finite = torch.isfinite(chord)
    chord_np = chord[finite].numpy()
    sigma_rms = float(torch.sqrt((sigma[row].to(torch.float64) ** 2).mean()))
    ratio = sigma_rms / np.clip(chord_np, 1.0e-30, None)
    return RayExtent(
        genotype_row=row,
        n_latent=n_latent,
        n_reactions_zero_width=int(zero_width.sum()),
        frac_basis_directions_blocked=float(blocked.to(torch.float64).mean()),
        admissible_dimension=dim,
        n_rays=int(finite.sum()),
        sigma_rms=sigma_rms,
        frac_reactions_at_bound=float(
            at_bound.to(torch.float64).sum() / (width > 1.0e-12).to(torch.float64).sum()
        ),
        chord_p10=_percentile(chord_np, 10.0),
        chord_p50=_percentile(chord_np, 50.0),
        chord_p90=_percentile(chord_np, 90.0),
        sigma_rms_over_chord_p50=float(np.median(ratio)),
        frac_rays_sigma_exceeds_chord=float((ratio > 1.0).mean()),
    )


def audit_cell(
    result_file: str,
    n_draws: int,
    n_rays: int,
    device: torch.device,
    tbx: Any,
    dataset: Any,
    cfg: Any,
) -> tuple[CellSpread, np.ndarray]:
    """Rebuild one cell from its checkpoint and measure its latent spread.

    Returns the record plus the array the figure draws: the per-(genotype, reaction)
    across-draw sd as a fraction of the box width.
    """
    stem = osp.splitext(result_file)[0]
    ckpt_path = osp.join(
        os.getenv("TC_CHECKPOINT_DIR", osp.join(DATA_ROOT, "checkpoints-027")),
        f"{stem}.pt",
    )
    assert osp.isfile(ckpt_path), f"no checkpoint at {ckpt_path}"
    with open(osp.join(RESULTS, result_file)) as handle:
        result = json.load(handle)
    coverage = result["flux_meta"]["coverage"]

    cell = tbx.BxCell(**result["cell"])
    # THE CHANNEL, from the result file rather than from the environment. Wave-7 cells did
    # not serialize `flux_parameterization`, so a rebuild that trusted the cell alone would
    # build the base arm's default channel and load the wrong parameter names. The layer
    # recorded what it ran under `parameterization_raw`, which is the same string the
    # training job passed through TC_FLUX_PARAMETERIZATION.
    cell.flux_parameterization = str(coverage["parameterization_raw"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint seed {checkpoint['cell']['seed']} against result seed {cell.seed}"
    )

    model, _, main_head, _ = tbx.build_model_for_cell(cell, cfg, dataset, device)
    layer = model.flux_layer
    assert layer is not None, f"arm {cell.arm!r} has no flux layer."
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    model.eval()

    pinned, _ = tbx.resolve_pinned_test(dataset, cfg)
    dm = tbx.CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    cell_graph = dataset.cell_graph.to(device)
    batch = next(iter(dm.all_dataloader())).to(device)

    # WHICH READOUT, and whether it reads the flux at all. `FluxScalarHead` takes the flux
    # and its turnover, `FluxMetaboliteHead` the turnover alone, and the concentration arm
    # has no head module: its prediction is a slice of the layer's learned log c, so it is
    # a function of the genotype context and not of the drawn flux.
    head = getattr(model, f"{main_head}_head", None)
    with torch.no_grad():
        _, reps = model(cell_graph, batch)
        h_genes = reps["H_genes_pert"]
        pert = batch["gene"].perturbation_indices
        pert_batch = batch["gene"].perturbation_indices_batch
        context = perturbed_gene_pool(h_genes, pert, pert_batch)
        mu = layer.latent_mlp(context)
        sigma = torch.exp(layer.latent_log_std(context))
        v = reps["flux"]["v"]
        lb = reps["flux"]["lb"]
        ub = reps["flux"]["ub"]

        flux_scale = float(layer.config.flux_scale)
        kl_mu = 0.5 * float((mu**2).mean())
        kl_sigma = 0.5 * float(
            (sigma**2 - 1.0 - 2.0 * torch.log(sigma.clamp(min=1e-30))).mean()
        )
        sigma_np = sigma.cpu().numpy()
        posterior = PosteriorSpread(
            n_genotypes=int(mu.shape[0]),
            n_latent=int(mu.shape[1]),
            sigma_min=float(sigma_np.min()),
            sigma_p10=_percentile(sigma_np, 10.0),
            sigma_p50=_percentile(sigma_np, 50.0),
            sigma_p90=_percentile(sigma_np, 90.0),
            sigma_max=float(sigma_np.max()),
            sigma_across_genotype_sd_median=float(np.median(sigma_np.std(axis=0))),
            mu_abs_p50=_percentile(np.abs(mu.cpu().numpy()), 50.0),
            mu_abs_p90=_percentile(np.abs(mu.cpu().numpy()), 90.0),
            sigma_over_abs_mu_p50=_percentile(
                sigma_np / np.abs(mu.cpu().numpy()).clip(min=1e-30), 50.0
            ),
            kl_total=kl_mu + kl_sigma,
            kl_mu_term=kl_mu,
            kl_sigma_term=kl_sigma,
        )

        rays = ray_extent(
            layer.null_basis.cpu(),
            v.cpu(),
            lb.cpu(),
            ub.cpu(),
            sigma.cpu(),
            flux_scale,
            row=0,
            n_rays=n_rays,
            seed=int(cell.seed),
        )

        # THE DRAWS. `config.stochastic` is the runtime switch the projection kind honors;
        # a cell trained deterministically has a spread head that was built but never given
        # a gradient, which is itself part of the finding rather than a reason not to draw.
        layer.config.stochastic = True
        assert layer.latent_stochastic, (
            "the layer did not become a sampler; only the projection and fluxforce kinds "
            "honor the runtime flip."
        )
        drawn = layer.sample(h_genes, context, pert, pert_batch, n_draws)
        v_draws = drawn.v  # [S, B, r]
        width = (ub - lb).clamp(min=1e-12)
        flux_sd = v_draws.std(dim=0)
        ratio = (flux_sd / width).cpu().numpy()
        moved = float((flux_sd > 1.0e-6 * flux_scale).float().mean().cpu().numpy())
        reads_flux = isinstance(head, (FluxScalarHead, FluxMetaboliteHead))
        draw_sd_median: float | None = None
        genotype_sd: float | None = None
        pred_ratio: float | None = None
        if isinstance(head, (FluxScalarHead, FluxMetaboliteHead)):
            preds: list[np.ndarray] = []
            for k in range(n_draws):
                turnover_k = layer.turnover(v_draws[k])
                out_k = (
                    head(turnover_k)
                    if isinstance(head, FluxMetaboliteHead)
                    else head(v_draws[k], turnover_k)
                )
                preds.append(out_k.reshape(int(v.shape[0]), -1).cpu().numpy())
            pred_stack = np.stack(preds)  # [S, B, P]
            draw_sd_median = float(np.median(pred_stack.std(axis=0)))
            genotype_sd = float(np.mean(pred_stack.mean(axis=0).std(axis=0)))
            pred_ratio = draw_sd_median / max(genotype_sd, 1e-30)
        draws = DrawSpread(
            n_draws=n_draws,
            n_genotypes=int(v.shape[0]),
            flux_sd_over_box_width_p50=_percentile(ratio, 50.0),
            flux_sd_over_box_width_p90=_percentile(ratio, 90.0),
            uniform_coverage_p50=_percentile(ratio, 50.0) * math.sqrt(12.0),
            frac_reactions_moved=moved,
            readout_reads_flux=reads_flux,
            prediction_sd_across_draws_median=draw_sd_median,
            prediction_sd_across_genotypes=genotype_sd,
            prediction_draw_to_genotype_ratio=pred_ratio,
        )

    history = result.get("history") or []
    record = CellSpread(
        result_file=result_file,
        arm=str(cell.arm),
        seed=int(cell.seed),
        epochs=int(cell.epochs),
        parameterization_raw=str(coverage["parameterization_raw"]),
        projection_steps=int(coverage["projection_steps"]),
        projection_end=str(coverage["projection_end"]),
        flux_scale=flux_scale,
        latent_log_std_init=float(layer.config.latent_log_std_init),
        latent_stochastic_trained=bool(coverage["latent_stochastic"]),
        kl_weight=float(coverage["kl_weight"]),
        constraint_final_scale=float(coverage["constraint_final_scale"]),
        effective_kl_weight=float(coverage["kl_weight"])
        * float(coverage["constraint_final_scale"]),
        c_kl_first_epoch=float(history[0]["c_kl"]) if history else None,
        c_kl_last_epoch=float(history[-1]["c_kl"]) if history else None,
        # The betaxanthin head reports one Spearman; the 19-column amino-acid head reports
        # the mean over columns. Both are the pinned Merzbacher test set at the
        # validation-selected epoch, which is the number every 027 wave is scored on.
        test_spearman_pinned=float(
            result["test"]["pinned"].get(
                "spearman", result["test"]["pinned"].get("mean_spearman", float("nan"))
            )
        ),
        posterior=posterior,
        rays=rays,
        draws=draws,
    )
    return record, ratio.reshape(-1)


def _short(record: CellSpread) -> str:
    """A short tick label: which readout the cell runs, and whether sigma was trained."""
    if "aa_turnover" in record.result_file:
        return "aa\nturnover"
    if "aa_conc" in record.result_file:
        return "aa\nconc."
    if "stoch" in record.result_file:
        return "betaxanthin\nsigma trained"
    return "betaxanthin"


def make_figure(records: list[CellSpread], flux_ratios: list[np.ndarray]) -> str:
    """Three panels: the posterior, what it moves, and what the readout sees.

    Panel a is the posterior sd per latent dimension against the value the spread head is
    initialized at, which separates a sigma that was never trained from one the objective
    shrank. Panel b is the across-draw flux sd as a fraction of each reaction's own box
    width, against the 1/sqrt(12) a uniform sample of that interval would give, so a draw
    cloud can be read as under- or over-covering the box rather than as a bare number.
    Panel c is the head's prediction sd across draws against its sd across genotypes, which
    is the only channel by which sampling depth can reach the phenotype loss.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(179.0), mm_to_in(62.0)), constrained_layout=True
    )
    labels = [_short(r) for r in records]
    positions = np.arange(len(records), dtype=float)

    ax = axes[0]
    for i, record in enumerate(records):
        posterior = record.posterior
        ax.plot(
            [positions[i]] * 3,
            [posterior.sigma_p10, posterior.sigma_p50, posterior.sigma_p90],
            marker="_",
            markersize=7,
            linestyle="-",
            linewidth=0.8,
            color=PLOT_PALETTE[i],
        )
    init = float(np.exp(records[0].latent_log_std_init))
    ax.axhline(init, color="black", linewidth=0.5, linestyle="--")
    ax.text(
        len(records) - 0.5,
        init * 1.15,
        f"init {init:.3f}",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(records) - 0.4)
    ax.set_ylabel("posterior sd, p10 to p90")
    ax.set_title("a  the latent posterior", loc="left", fontsize=6)

    ax = axes[1]
    ax.boxplot(
        [np.clip(f, 1.0e-9, None) for f in flux_ratios],
        positions=positions,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.6},
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
    )
    uniform = 1.0 / float(np.sqrt(12.0))
    ax.axhline(uniform, color="black", linewidth=0.5, linestyle="--")
    ax.text(
        len(records) - 0.5,
        uniform * 1.15,
        "uniform over the box",
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(records) - 0.4)
    ax.set_ylabel("across-draw flux sd / box width")
    ax.set_title("b  what the draws move", loc="left", fontsize=6)

    ax = axes[2]
    # A readout that does not read the flux has no bar rather than a bar at zero: zero
    # would read as "measured and tiny" when the truth is that the quantity is not defined.
    for i, record in enumerate(records):
        value = record.draws.prediction_draw_to_genotype_ratio
        if value is None:
            ax.text(
                positions[i],
                1.5e-3,
                "readout does\nnot read the flux",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
            continue
        ax.bar(
            positions[i],
            max(value, 1.0e-9),
            width=0.5,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-3, 3.0e1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(records) - 0.4)
    ax.set_ylabel("prediction sd, draws / genotypes")
    ax.set_title("c  what the readout sees", loc="left", fontsize=6)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(width=0.5)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, f"wave8_latent_spread_{timestamp()}")
    savefig_true_size_svg(fig, f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    return f"{stem}.svg"


def main() -> None:
    """Audit every named cell and write the JSON and the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=DEFAULT_CELLS)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--rays", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="wave8_latent_spread.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    tbx = _load_train_bx()
    cfg = tbx.load_config()
    dataset = tbx.build_dataset()

    records: list[CellSpread] = []
    flux_ratios: list[np.ndarray] = []
    for name in args.cells:
        record, flux_ratio = audit_cell(
            name, args.draws, args.rays, device, tbx, dataset, cfg
        )
        records.append(record)
        flux_ratios.append(flux_ratio)
        pred_ratio = record.draws.prediction_draw_to_genotype_ratio
        print(
            f"{name}: trained_stochastic={record.latent_stochastic_trained} "
            f"sigma_p50={record.posterior.sigma_p50:.3e} "
            f"admissible_dim={record.rays.admissible_dimension} "
            f"sigma_rms/chord_p50={record.rays.sigma_rms_over_chord_p50:.3e} "
            f"pred draw/genotype="
            + (
                "none (readout does not read flux)"
                if pred_ratio is None
                else f"{pred_ratio:.3e}"
            )
        )

    figure = make_figure(records, flux_ratios)
    payload = {
        "generated_at": timestamp(),
        "n_draws": args.draws,
        "figure": figure,
        "cells": [r.model_dump() for r in records],
    }
    out_path = osp.join(RESULTS, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path}")
    print(f"wrote {figure}")


if __name__ == "__main__":
    main()

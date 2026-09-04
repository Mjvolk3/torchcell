# experiments/027-betaxanthin-metabolic/scripts/train_bx.py
# [[experiments.027-betaxanthin-metabolic.scripts.train_bx]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/train_bx.py

r"""Betaxanthin-only training cell for the metabolic-module decision.

WHAT THIS DECIDES
-----------------
Whether the enzyme-constrained thermodynamic flux layer, specialized to the betaxanthin
route, predicts the Cachera screen better than (a) the same CGT with no metabolic module and
(b) the published Flux Cone Learning baseline (Merzbacher 2025, RandomForest_Resampled,
Spearman +0.0391 on the 639 pinned test genes).

WHY THIS FILE EXISTS RATHER THAN A 026 OR A 019 RUN
----------------------------------------------------
026's ``train_flux.py`` has the flux layer but the wrong protocol: two heads, a re-rolled
split per seed, and a headline number that is a MAXIMUM over epochs of the same validation
curve used to pick the epoch. 019's ``train_cgt_multitask.py`` has the right protocol --
pinned Merzbacher test split, pinned partition, checkpoint-by-validation, test pass, per-gene
dump -- but it never passes ``flux_layer=`` to ``CellGraphTransformerMetabolism``, so it
cannot build the module at all.

So this file takes 026's flux construction VERBATIM by import (``ARMS``, ``build_dataset``,
``build_flux_layer``, ``extract_targets``, ``BETAXANTHIN_PRECURSORS``) and reimplements only
the loop, where the protocol lives. Nothing about the flux layer is redefined here; if 026's
arm registry changes, this experiment follows it.

THE THREE PROTOCOL CHANGES, each fixing a measured defect
----------------------------------------------------------
1. **One head.** 026 trained betaxanthin and the Mulleder 19 amino acids jointly, so a
   betaxanthin gain could have come from the co-head. The FCL comparison is betaxanthin-only,
   so the training must be too. Mulleder-only genotypes still arrive in the batch; their
   betaxanthin mask is False and they contribute no loss.
2. **The split is pinned; the seed varies weight initialization only.** 026 let ``seed`` set
   both, which is why its paired arm gap carried an across-seed SD of 0.0303 with real labels
   and 0.0722 with permuted ones. 019 measured the same nuisance directly: between-seed SD
   0.0444 against a within-seed across-arm SD of 0.0058.
3. **Selection and reporting use DISJOINT data.** The epoch is chosen by the argmax of a
   centered 5-epoch moving average of validation Pearson; the number reported is the Spearman
   on the pinned test genes at that epoch. A max over epochs of validation is an upward-biased
   order statistic whose bias grows with the epoch budget -- with 353 validation measurements
   its null width is 0.0535, and every arm 026 reported sat inside one null width of every
   other and of its own permuted-label control.

Run one cell by hand from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
        --arm flux_anchored --seed 0 --epochs 2 --smoke
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import pearsonr, spearmanr

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
CONF_DIR = osp.join(EXP_DIR, "conf")
RESULTS_DIR = osp.join(EXP_DIR, "results")
#: Per-gene test predictions land beside 020's, so the existing FCL head-to-head scripts
#: (`plot_merzbacher_comparison.py`, `evaluate_merzbacher_head_to_head.py`) read them without
#: modification. Re-deriving the FCL labels here would fork their provenance.
TEST_PRED_DIR = os.getenv(
    "TC_TEST_PRED_DIR", osp.join(DATA_ROOT, "test-predictions-027")
)

# 026 owns the flux layer's construction; this experiment owns only the protocol.
#
# `TC_FLUX_SCRIPTS_DIR` exists for one situation and it is worth naming: 027 was authored in a
# worktree whose branch predates the 026 landing, so `$EXPERIMENT_ROOT/026-metabolism-flux/`
# there has no `train_flux.py` and `torchcell/metabolism/` has no `flux_layer.py`. The
# override lets a smoke test run against the primary checkout. On Delta and after the rebase
# it is unset and the default path is the right one.
FLUX_SCRIPTS_DIR = os.getenv(
    "TC_FLUX_SCRIPTS_DIR", osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
)
assert osp.isfile(osp.join(FLUX_SCRIPTS_DIR, "train_flux.py")), (
    f"026's train_flux.py is not at {FLUX_SCRIPTS_DIR}. This branch predates the 026 "
    "landing; rebase onto main, or set TC_FLUX_SCRIPTS_DIR for a local smoke test."
)
sys.path.insert(0, FLUX_SCRIPTS_DIR)
from train_flux import (  # noqa: E402
    ARMS,
    BETAXANTHIN_PRECURSORS,
    build_dataset,
    build_flux_layer,
    extract_targets,
)

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    CellGraphTransformerMetabolism,
)

HEAD = "betaxanthin"


class ScoringConfig(BaseModel):
    """The scoring rule, fixed before the first launch."""

    model_config = ConfigDict(extra="forbid")

    primary: str
    select_metric: str
    select_smoothing_window: int
    aggregate: str


class BaselineConfig(BaseModel):
    """The published numbers the decision is made against, read from 020's results JSON."""

    model_config = ConfigDict(extra="forbid")

    fcl_rf_test_spearman: float
    fcl_rf_test_auc_high: float
    cgt_020_v4_test_spearman_mean: float
    cgt_020_v4_test_spearman_sd: float


class DataConfig(BaseModel):
    """Dataset location, the external test split to pin, and the partition seed."""

    model_config = ConfigDict(extra="forbid")

    dataset_subdir: str
    query_path: str
    pinned_test_split_file: str
    split_seed: int


class ModelConfig(BaseModel):
    """The 026 incumbent hyperparameters, identical across all six arms."""

    model_config = ConfigDict(extra="forbid")

    hidden: int
    layers: int
    batch_size: int
    lr: float
    weight_decay: float


class TrainConfig(BaseModel):
    """Epoch ceiling and the two knobs the launcher may override."""

    model_config = ConfigDict(extra="forbid")

    epochs: int
    early_stopping: bool
    num_workers: int


class FluxConfig(BaseModel):
    """The kcat-coverage floor that stops a flux arm running on default parameters."""

    model_config = ConfigDict(extra="forbid")

    min_kcat_experimental_fraction: float


class BxConfig(BaseModel):
    """``conf/base.yaml``, validated.

    ``extra="forbid"`` throughout, so a typo in a YAML key is an error at load rather than a
    silently ignored hyperparameter -- the failure mode that makes two arms differ by
    something nobody chose.
    """

    model_config = ConfigDict(extra="forbid")

    scoring: ScoringConfig
    baselines: BaselineConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    flux: FluxConfig


class ArmConfig(BaseModel):
    """``conf/arm_<name>.yaml``: the only two things an arm is allowed to change."""

    model_config = ConfigDict(extra="forbid")

    base_arm: str
    permute_train_targets: bool


class BxCell(BaseModel):
    """One training run, fully specified. Serialized into the results file verbatim."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    base_arm: str
    seed: int
    permute_train_targets: bool
    epochs: int
    hidden: int
    layers: int
    batch_size: int
    num_workers: int
    lr: float
    weight_decay: float
    split_seed: int
    select_smoothing_window: int = Field(ge=1)


def arm_names() -> list[str]:
    """Every arm this experiment defines, in a FIXED order.

    The order is load-bearing: the shard rule assigns worker ``w`` the setting at index
    ``w % n_settings``, so a reordering silently reassigns work between workers mid-experiment.
    Sorted, so it cannot drift with filesystem order.
    """
    return sorted(
        f[len("arm_") : -len(".yaml")]
        for f in os.listdir(CONF_DIR)
        if f.startswith("arm_") and f.endswith(".yaml")
    )


def load_config() -> BxConfig:
    with open(osp.join(CONF_DIR, "base.yaml")) as fh:
        return BxConfig(**yaml.safe_load(fh))


def build_cell(arm: str, seed: int, cfg: BxConfig, epochs: int | None = None) -> BxCell:
    """Merge ``base.yaml`` with one arm file into a fully specified cell."""
    with open(osp.join(CONF_DIR, f"arm_{arm}.yaml")) as fh:
        armcfg = ArmConfig(**yaml.safe_load(fh))
    assert armcfg.base_arm in ARMS, (
        f"arm {arm!r} names base_arm={armcfg.base_arm!r}, which is not in 026's ARMS "
        f"registry ({sorted(ARMS)}). The registry is the single source of truth for what "
        "the flux layer does; this file never redefines it."
    )
    return BxCell(
        arm=arm,
        base_arm=armcfg.base_arm,
        seed=seed,
        permute_train_targets=armcfg.permute_train_targets,
        epochs=epochs if epochs is not None else cfg.train.epochs,
        hidden=cfg.model.hidden,
        layers=cfg.model.layers,
        batch_size=cfg.model.batch_size,
        num_workers=int(os.getenv("NUM_WORKERS", str(cfg.train.num_workers))),
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        split_seed=cfg.data.split_seed,
        select_smoothing_window=cfg.scoring.select_smoothing_window,
    )


def resolve_pinned_test(dataset: Any, cfg: BxConfig) -> tuple[set[int], dict[str, Any]]:
    """Merzbacher 2025's 639 test ORFs -> record indices in our build.

    Lifted from ``train_cgt_multitask._pinned_test_indices`` including its two guards, both
    of which were found the hard way there:

    * the mapping goes through ``is_any_deletion_gene_index``, NOT
      ``is_any_perturbed_gene_index``. The pigment cassettes are emitted as perturbations on
      every strain and three of their members are native ORFs (ARO4/YBR249C and ARO7/YPR060C
      in Cachera's betaxanthin cassette), two of which are in Merzbacher's test list. Under
      the perturbed-gene index the pin resolved 639 genes to 4,885 of 4,930 records and left
      28 training examples -- and trained anyway.
    * a requested gene absent from the built LMDB is COUNTED AND REPORTED. The Cachera build
      is stale w.r.t. the shared name resolver (issue #195) and the split file names the 10
      it loses; a comparison that quietly shrank its own test set would be reporting on a
      different gene set than it claims.
    """
    path = osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)
    with open(path) as fh:
        payload = json.load(fh)
    genes = payload["split"]["test"]
    gene_index = dataset.is_any_deletion_gene_index
    indices: set[int] = set()
    found: list[str] = []
    absent: list[str] = []
    for gene in genes:
        hits = gene_index.get(gene)
        if hits:
            found.append(gene)
            indices.update(int(h) for h in hits)
        else:
            absent.append(gene)
    frac = len(indices) / len(dataset)
    assert frac < 0.5, (
        f"pinned test set is {frac:.1%} of the dataset ({len(indices)}/{len(dataset)}). "
        "639 genes should select a small minority of records; this means the gene index is "
        "resolving cassette membership rather than deletions."
    )
    report = {
        "path": path,
        "requested": len(genes),
        "found": len(found),
        "absent": absent,
        "records": len(indices),
        "fraction_of_dataset": frac,
    }
    print(
        f"[pinned-split] {len(found)}/{len(genes)} genes present -> {len(indices)} records "
        f"pinned to TEST ({frac:.1%} of {len(dataset)}); {len(absent)} absent (issue #195)",
        flush=True,
    )
    return indices, report


def _params_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the TRAINABLE parameters only.

    Not ``state_dict()``: the flux layer registers the stoichiometric matrix and the
    thermodynamic tables as buffers, and the selection ring holds five snapshots, so copying
    buffers would carry ~46 MB of constant S matrix five times per worker for nothing.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()}


def _restore(model: nn.Module, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(snap[name].to(param.device))


def _select_epoch(val_curve: list[float], window: int) -> tuple[int, str]:
    """Epoch chosen by argmax of a CENTERED moving average of validation Pearson.

    Returns ``(epoch, rule)``. The rule string is written into the result so a short run --
    a canary, or a cell the wall clock truncated -- is never silently scored by a different
    rule than the one the design fixed.
    """
    finite = [v if np.isfinite(v) else -np.inf for v in val_curve]
    if len(finite) < window:
        return int(np.argmax(finite)), "raw_argmax_short_run"
    half = window // 2
    smoothed = [
        float(np.mean(finite[i - half : i + half + 1]))
        for i in range(half, len(finite) - half)
    ]
    return int(np.argmax(smoothed)) + half, f"centered_ma_{window}"


def _masked_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson on a 1-D pair, with the RELATIVE variance guard 026 arrived at.

    An absolute floor is the wrong test: a head collapsed onto a constant emits a residual
    spread of order 1e-11, which clears any absolute floor and then makes ``corrcoef`` divide
    by a number indistinguishable from zero, yielding NaN -- reported as "metric unavailable"
    when what happened is the documented mean-collapse failure. Scaling by the target's spread
    makes "collapsed" scale-free, and a collapsed head scores 0.0, which is what it earned.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    r = float(pearsonr(p, t)[0])
    return r if np.isfinite(r) else 0.0


def _masked_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < 10 or t.std() < 1e-12 or p.std() < 1e-8 * t.std():
        return 0.0
    rho = float(spearmanr(p, t)[0])
    return rho if np.isfinite(rho) else 0.0


def _top_k_enrichment(
    pred: np.ndarray, true: np.ndarray, k: int, top_frac: float = 0.1
) -> dict[str, float]:
    """Of the k genes the model ranks highest, how many are truly in the top ``top_frac``.

    The imbalance-immune half of the 020 recipe, and the metric the task is actually about:
    accuracy on a 67%-majority problem actively selects against finding high producers.
    ``top_frac`` defines "high" on OUR measured values so the metric needs no external label
    file; the 3-class head-to-head against Merzbacher's released labels is a separate step,
    run by 020's own scripts on the per-gene dump this file writes.
    """
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    if p.size < k:
        return {"hits": float("nan"), "enrichment": float("nan")}
    n_high = max(1, int(round(top_frac * t.size)))
    high = set(np.argsort(-t)[:n_high].tolist())
    top = np.argsort(-p)[:k].tolist()
    hits = sum(1 for i in top if i in high)
    base = n_high / t.size
    return {
        "hits": float(hits),
        "enrichment": float((hits / k) / base) if base else float("nan"),
    }


def run_cell(
    cell: BxCell,
    cfg: BxConfig,
    dataset: Any,
    pinned_test: set[int],
    pinned_report: dict[str, Any],
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    dump_predictions: bool = True,
) -> dict[str, Any]:
    """Train one cell, select an epoch on validation, score it on the pinned test genes."""
    torch.manual_seed(cell.seed)
    np.random.seed(cell.seed)
    arm = ARMS[cell.base_arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    flux_layer, flux_meta = build_flux_layer(arm, gene_ids, cell.hidden)

    if arm["flux"]:
        frac = float(flux_meta["kcat_experimental_fraction"])
        # NO FALLBACK. `resolve_kcat_table` catches a missing Open Enzyme Database mirror and
        # returns an empty record list, which replaces every measured kcat with the organism
        # default -- an "enzyme-constrained" arm with no enzyme constraints, running to
        # completion and reporting a plausible number. The mirror is 512 KB and is NOT on
        # Delta as of 2026-09-03; see README.md, "Prerequisites".
        assert frac >= cfg.flux.min_kcat_experimental_fraction, (
            f"kcat experimental fraction is {frac:.4f}, below the floor "
            f"{cfg.flux.min_kcat_experimental_fraction}. The Open Enzyme Database mirror at "
            f"$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae is missing or "
            "empty, so every kcat has silently become the organism default."
        )

    head_specs: dict[str, dict[str, Any]] = {
        # Betaxanthin and the Mulleder metabolome both carry the label `metabolite_level`;
        # WIDTH is what separates them. Declaring only the width-1 spec supervises betaxanthin
        # alone, and a Mulleder-only genotype simply matches nothing and arrives masked False.
        HEAD: {"label": "metabolite_level", "width": 1}
    }
    if arm["flux"]:
        met_index = flux_meta["met_index"]
        heads_config = {
            HEAD: {
                "kind": "flux_scalar",
                "output_dim": 1,
                "precursor_indices": [
                    met_index[m] for m in BETAXANTHIN_PRECURSORS if m in met_index
                ],
            }
        }
    else:
        heads_config = {HEAD: {"kind": "scalar", "output_dim": 1}}

    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=len(gene_ids),
        hidden_channels=cell.hidden,
        num_transformer_layers=cell.layers,
        num_attention_heads=4,
        dropout=0.1,
        heads_config=heads_config,
        learnable_embedding_config={
            "enabled": True,
            "size": cell.hidden,
            "preprocessor": {"num_layers": 2, "dropout": 0.1},
        },
        flux_layer=flux_layer,
    ).to(device)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        # THE PINNED PARTITION. `random_seed` selects train/val/test, so it is held at
        # `split_seed` for every arm and every seed; `cell.seed` reached torch above and
        # varies weight INITIALIZATION only.
        random_seed=cell.split_seed,
        num_workers=cell.num_workers,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
        pinned_test_indices=pinned_test,
    )
    dm.setup()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cell.lr, weight_decay=cell.weight_decay
    )
    loss_fn = nn.MSELoss(reduction="none")

    # Target standardization from the TRAIN split only. A metric computed against
    # test-informed statistics is not a held-out metric.
    chunks: list[torch.Tensor] = []
    for batch in dm.train_dataloader():
        t, m = extract_targets(batch, head_specs)[HEAD]
        chunks.append(torch.where(m, t, torch.nan))
    cat = torch.cat(chunks, dim=0)
    mean = torch.nanmean(cat, dim=0)
    centered = cat - mean
    sd = torch.nanmean(centered * centered, dim=0).sqrt().clamp(min=1e-6)
    mean, sd = mean.to(device), sd.to(device)

    window = cell.select_smoothing_window
    half = window // 2
    #: ``(epoch, trainable-parameter snapshot)``, `window` deep. Keyed by epoch rather than
    #: indexed by position: a run whose validation curve is entirely non-finite selects epoch
    #: 0, which position arithmetic would resolve to a snapshot the ring no longer holds.
    ring: deque[tuple[int, dict[str, torch.Tensor]]] = deque(maxlen=window)
    val_curve: list[float] = []
    history: list[dict[str, Any]] = []
    best_smoothed = -np.inf
    best_snap: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(cell.epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        feas: dict[str, float] = {}
        for batch in dm.train_dataloader():
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = extract_targets(batch, head_specs)[HEAD]
            t, m = t.to(device), m.to(device)
            if cell.permute_train_targets:
                # THE NULL: target and mask permuted TOGETHER, so the genotype-to-phenotype
                # association is destroyed while the target distribution and the missingness
                # rate are untouched. Validation and test are never permuted.
                perm = torch.randperm(t.shape[0], device=device)
                t, m = t[perm], m[perm]
            z = (t - mean) / sd
            pred = reps["head_outputs"][HEAD]
            if pred.dim() == 3:
                pred = pred[..., 0]
            loss = (loss_fn(pred, z) * m).sum() / m.sum().clamp(min=1)
            loss = loss + reps["graph_reg_loss"]
            if "flux" in reps:
                # `reps` carries "flux" only when the model was built WITH a layer, so this
                # cannot be None here. Asserted rather than type-ignored: if it ever were
                # None the constraint terms would silently vanish and the arm would still
                # report a plausible number.
                assert model.flux_layer is not None
                loss = loss + model.flux_layer.constraint_loss(reps["flux"])
                for k, v in reps["flux"].items():
                    if v.dim() == 0 and k.startswith(
                        ("c_", "feas_", "g_diss", "protein_used")
                    ):
                        feas[k] = feas.get(k, 0.0) + float(v.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        preds, trues = _predict(
            model, dm.val_dataloader(), dataset, head_specs, mean, sd
        )
        val_r = _masked_pearson(preds, trues)
        val_curve.append(val_r)
        ring.append((epoch, _params_snapshot(model)))

        # The centered window can only score epoch `t - half` once epoch `t` exists, which is
        # exactly why the ring is `window` deep: the snapshot for that epoch is still in it.
        if len(val_curve) >= window:
            centered_epoch = epoch - half
            score = float(np.mean(val_curve[-window:]))
            if score > best_smoothed:
                best_smoothed = score
                best_snap = ring[-(half + 1)][1]
                best_epoch = centered_epoch

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "val_pearson": val_r,
            "n_val": int(np.isfinite(trues).sum()),
        }
        for k, v in feas.items():
            row[k] = v / max(n_batches, 1)
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        print(
            f"[{cell.arm} s{cell.seed}] ep {epoch:3d} loss {row['train_loss']:.4f} "
            f"val_r {val_r:+.4f} best_ep {best_epoch}",
            flush=True,
        )

    sel_epoch, sel_rule = _select_epoch(val_curve, window)
    snap_by_epoch = dict(ring)
    if best_snap is not None and sel_rule.startswith("centered_ma"):
        # The ring selects online (it must -- a centered window needs a snapshot that is
        # about to fall out of memory) and `_select_epoch` recomputes offline from the saved
        # curve. They are two implementations of ONE rule, so they must agree, and a silent
        # divergence would mean the reported epoch is not the epoch that was scored.
        assert sel_epoch == best_epoch, (
            f"selection disagreement: online ring picked {best_epoch}, offline recompute "
            f"picked {sel_epoch}. The ring and `_select_epoch` must implement one rule."
        )
        _restore(model, best_snap)
    else:
        # A run shorter than the smoothing window (a canary, or a wall-clock truncation)
        # falls to the raw argmax, and `select_rule` in the result says so.
        assert sel_epoch in snap_by_epoch, (
            f"selected epoch {sel_epoch} has no snapshot; the ring holds "
            f"{sorted(snap_by_epoch)}. This means every validation epoch was non-finite."
        )
        _restore(model, snap_by_epoch[sel_epoch])

    test = _score_test(model, dm, dataset, head_specs, mean, sd, cell, dump_predictions)
    return {
        "cell": cell.model_dump(),
        "select_epoch": sel_epoch,
        "select_rule": sel_rule,
        "select_val_pearson": val_curve[sel_epoch] if val_curve else float("nan"),
        "test": test,
        "history": history,
        "flux_meta": {k: v for k, v in flux_meta.items() if k != "met_index"},
        "pinned_split": pinned_report,
        "n_parameters": model.num_parameters,
        "wall_time_s": time.time() - t0,
    }


def _predict(
    model: nn.Module,
    loader: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalized predictions and targets over a loader; unobserved rows become NaN."""
    device = next(model.parameters()).device
    model.eval()
    ps: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            t, m = extract_targets(batch, head_specs)[HEAD]
            pred = reps["head_outputs"][HEAD]
            if pred.dim() == 3:
                pred = pred[..., 0]
            ps.append((pred * sd + mean).cpu().numpy().reshape(-1))
            ts.append(torch.where(m, t, torch.nan).cpu().numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(ts)


def _score_test(
    model: nn.Module,
    dm: Any,
    dataset: Any,
    head_specs: dict[str, dict[str, Any]],
    mean: torch.Tensor,
    sd: torch.Tensor,
    cell: BxCell,
    dump_predictions: bool,
) -> dict[str, Any]:
    """The prespecified test metrics, on the pinned genes and on the whole test split.

    ``test_dataloader`` is built with ``shuffle=False``, so batches arrive in the order of
    ``test_dataset.indices`` -- that is what lets a running cursor map row -> record -> gene.
    The gene name comes from the DELETION index for the reason spelled out in
    :func:`resolve_pinned_test`.
    """
    order = [int(i) for i in dm.test_dataset.indices]
    preds, trues = _predict(model, dm.test_dataloader(), dataset, head_specs, mean, sd)
    assert len(preds) == len(order), (
        f"test pass covered {len(preds)}/{len(order)} records -- a short count is a silent "
        "truncation of the comparison set."
    )

    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))

    pin_path = osp.join(
        EXPERIMENT_ROOT, "020-cachera-betaxanthin/results/merzbacher_nested_split.json"
    )
    with open(pin_path) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])

    records = []
    is_pinned = np.zeros(len(order), dtype=bool)
    for i, rec in enumerate(order):
        genes = idx_to_genes.get(rec, [])
        is_pinned[i] = any(g in pinned_genes for g in genes)
        records.append(
            {
                "record_index": rec,
                "genes": genes,
                "pred": float(preds[i]),
                "target": None if not np.isfinite(trues[i]) else float(trues[i]),
                "pinned": bool(is_pinned[i]),
            }
        )

    def block(sel: np.ndarray) -> dict[str, Any]:
        p, t = preds[sel], trues[sel]
        obs = np.isfinite(t)
        return {
            "n": int(obs.sum()),
            "spearman": _masked_spearman(p, t),
            "pearson": _masked_pearson(p, t),
            **{f"top{k}": _top_k_enrichment(p[obs], t[obs], k) for k in (10, 25, 50)},
        }

    out: dict[str, Any] = {
        "pinned": block(is_pinned),
        "all": block(np.ones(len(order), dtype=bool)),
    }

    if dump_predictions:
        os.makedirs(TEST_PRED_DIR, exist_ok=True)
        path = osp.join(TEST_PRED_DIR, f"{cell.arm}_s{cell.seed}.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "experiment": "027-betaxanthin-metabolic",
                    "cell": cell.model_dump(),
                    "head": HEAD,
                    "n_test_records": len(order),
                    "metrics": out,
                    "predictions": records,
                },
                fh,
            )
        out["dump_path"] = path
        print(f"[test-dump] wrote {path} ({len(records)} records)", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=arm_names())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip the per-gene dump. For a local build/shape check, never for a result.",
    )
    args = parser.parse_args()

    cfg = load_config()
    cell = build_cell(args.arm, args.seed, cfg, epochs=args.epochs)
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)
    pinned, report = resolve_pinned_test(dataset, cfg)
    result = run_cell(
        cell, cfg, dataset, pinned, report, dump_predictions=not args.smoke
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or f"single_{cell.arm}_s{cell.seed}.json"
    path = osp.join(RESULTS_DIR, out)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(
        f"wrote {path}\n"
        f"  select_epoch {result['select_epoch']} ({result['select_rule']})\n"
        f"  pinned test spearman {result['test']['pinned']['spearman']:+.4f} "
        f"(n={result['test']['pinned']['n']})",
        flush=True,
    )


if __name__ == "__main__":
    main()

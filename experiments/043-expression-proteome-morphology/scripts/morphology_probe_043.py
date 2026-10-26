# experiments/043-expression-proteome-morphology/scripts/morphology_probe_043.py
# [[experiments.043-expression-proteome-morphology.scripts.morphology_probe_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/morphology_probe_043.py
"""Which representation of a deletion strain maps to its CalMorph morphology (Ohya 2005).

Owner, 2026-10-26: "see if gene embeddings can map to morphology, and if the cls or sum
over genes or something can map to morphology better." Ohya 2005 is the 043 probe
dataset: no model is ever trained on it (``global_split_043.Role.PROBE``).

Strains and targets. The Ohya 2005 single-deletion records of the store generation
(``STORE_043_VERSION``, v2 by default here) from the ``linear_evals_043`` cache, one row per
deleted gene (the mean over its records), restricted to genes in the 6,607-gene index.
Targets: the 116 "moving" CalMorph features of the 019 / ``linear_evals_043``
morphology probe (``MOVING_PATH``, sha256 recorded), and all 281 features as a second set.

Representations (one row per strain, each mapped to the feature vector by ridge):

* ``null``: the per-feature training mean (a constant prediction; its Pearson is 0 by
  the convention of ``linear_evals_043._corr_cols``, its held-out R^2 is reported).
* ``frozen_<block>``: the S288C frozen token of the deleted gene, the composite (3,328)
  and each of its four blocks alone (FUDT upstream, FUDT downstream, ProtT5-XL, CaLM),
  each block LayerNorm-standardized per gene (``linear_evals_043.load_tokens``).
* From a trained ``CellModel043`` checkpoint (built as ``train_043`` builds it:
  ``checkpoint_config`` -> ``PreparedStore043`` -> ``build_view`` -> ``h5_vector_source``
  -> ``build_model``, weights loaded strictly; batches by ``train_043.make_batch``; marks
  by ``train_043.record_marks`` from the record's ``RecordRepresentation``, the same
  function the prepare stage uses), one forward pass per strain on CPU (the S288C
  context plus the strain's one deletion mark), read through forward hooks on the trunk
  and the operator (the forward is not changed):

  - ``trunk_gene``: the deleted gene's trunk token (trunk output; the operator's input
    under ``inject = head``; under ``inject = operator`` the operator's input adds the
    record's ontology shift to every token, which the trunk output does not carry);
  - ``pert_gene``: the deleted gene's row of ``H_pert`` (after the operator);
  - ``cls``: the CLS row of ``H_pert``;
  - ``mean_genes``: the mean of ``H_pert`` over the 6,607 gene rows;
  - ``update_sum``: ``sum_g (H_pert - H)`` over the gene rows, ``H`` the operator input;
  - ``z_s``: the mark key (``CellModel043Output.z_s``, one mark per strain).

  ``mean_genes`` and ``update_sum`` differ by a strain-independent offset and a scale
  (``mean H`` is the same S288C context for every strain), so under per-column
  standardization they are the same ridge input; both are kept as asked.

Split and fit. Primary: the global fold file (``global_split_v2``) gene folds. Held-out
strains = Ohya strains whose deleted gene is in the test fold (fold 0; never a training
deletion of the trained models); the penalty is chosen on the validation fold (fold 1)
with a fit on folds 2-9, then refit on folds 1-9 (``linear_evals_043.ridge_fold``:
inputs and targets z-scored on the fitting rows, kernel ridge, ``lambda = (trace K / n)
* 10^g``, g in -3..3 by halves, selected by the median per-feature Pearson). Targets are
z-scored with the training strains' mean and SD and clipped at |z| 8 before the fit.
Statistic: Pearson per feature across the held-out strains; median, mean and the count
above 0.3; 95% bootstrap intervals (1,000 resamples of held-out strains, seed 0) for the
median, and paired against ``frozen_composite``. Secondary: the Table morph construction
of ``linear_evals_043.run_morphology`` (5-fold out-of-fold over every strain,
``morph_oof``), so the frozen composite row can be checked against that earlier run.

    STORE_043_VERSION=v2 python experiments/043-expression-proteome-morphology/scripts/morphology_probe_043.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import os.path as osp
import subprocess
import sys
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from global_split_043 import load_split  # noqa: E402
from linear_evals_043 import (  # noqa: E402
    BLOCKS,
    EMBEDDINGS,
    M_ZCLIP,
    MOVING_PATH,
    LinearEvalConfig,
    _colr,
    _corr_cols,
    _mz,
    load_rows,
    load_tokens,
    morph_oof,
    paired_boot,
    ridge_fold,
)
from store_version_043 import active as active_store  # noqa: E402
from store_version_043 import store_paths  # noqa: E402
from train_043 import (  # noqa: E402
    PreparedStore043,
    TrainingView,
    build_model,
    build_view,
    checkpoint_config,
    h5_vector_source,
    make_batch,
    record_marks,
    record_ontology_key,
)

from torchcell.data.experiment_ontology import AnnotationRecipe  # noqa: E402
from torchcell.models.cell_model_043 import CellModel043  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
OHYA = "ScmdOhya2005Dataset"
CKPT_ROOT = "experiments/043-expression-proteome-morphology/checkpoints"
#: label -> checkpoint run directory under $DATA_ROOT/<CKPT_ROOT>
CHECKPOINTS: dict[str, str] = {
    "bs16": "all_onto_hlgauss_equal_s0/gs1024_w4_bs16",
    "onto_inject_op": "all_onto_injop_hlgauss_equal_s0/gs1024_w5_onto_inject_op",
}
TRAINED_REPS = ("trunk_gene", "pert_gene", "cls", "mean_genes", "update_sum", "z_s")
REFERENCE = "frozen_composite"
R_THRESHOLD = 0.3


class ProbeConfig(BaseModel):
    """Run configuration."""

    data_root: str
    experiment_dir: str
    checkpoints: dict[str, str] = Field(default_factory=lambda: dict(CHECKPOINTS))
    batch_size: int = Field(default=128, gt=0)
    n_boot: int = 1000
    boot_seed: int = 0
    ridge_log10_grid: list[float] = Field(
        default_factory=lambda: [x / 2 for x in range(-6, 7)]
    )

    @property
    def json_path(self) -> str:
        """The committed result file."""
        return osp.join(self.experiment_dir, "results", "morphology_probe_043.json")


# ============================================================================ targets


def ohya_targets(
    lcfg: LinearEvalConfig, genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(116 moving features, all 281) per deleted gene in the index; and the sources."""
    gidx = set(genes)
    with open(MOVING_PATH, "rb") as fh:
        raw = fh.read()
    moving = json.loads(raw)["moving"]
    rows = load_rows(lcfg, "ohya")
    feats = rows[0]["features"]
    by_g: dict[str, list[np.ndarray]] = {}
    n_multi = 0
    for r in rows:
        if r["features"] != feats:
            raise ValueError("CalMorph feature order differs between records")
        if len(r["genes"]) != 1:
            n_multi += 1
            continue
        by_g.setdefault(r["genes"][0], []).append(r["y"].astype(np.float64))
    full = pd.DataFrame(
        {g: np.nanmean(np.stack(v), 0) for g, v in by_g.items()}, index=feats
    ).T
    n_outside = int(sum(g not in gidx for g in full.index))
    full = full.loc[sorted(g for g in full.index if g in gidx)]
    full = full.loc[:, full.std(ddof=0) > 0]
    mov = full[[m for m in moving if m in full.columns]]
    return (
        mov,
        full,
        {
            "n_records": len(rows),
            "n_multi_gene_records_skipped": n_multi,
            "n_genes_outside_index_skipped": n_outside,
            "moving_features_source": {
                "path": MOVING_PATH,
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        },
    )


# ============================================================================ trained model


def ohya_marks(data_root: str, route: str, strains: list[str]) -> dict[str, Any]:
    """Marks (vocabulary ids), raw index and ontology key of each strain's record, from
    the record's representation (``train_043.record_marks``, ``record_ontology_key``).
    """
    import dataset_043 as d043
    import representation_043 as rep043

    from torchcell.data.representation_043 import RepresentationConfig

    ds = d043.Raw043Dataset(
        d043.Dataset043Config(
            data_root=data_root,
            datasets=[OHYA],
            representation=RepresentationConfig(route=route),
        )
    )
    want = set(strains)
    out: dict[str, dict[str, Any]] = {}
    for i, (_, raw_index) in enumerate(ds.items):
        record = ds.record(raw_index)
        perts = record["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        idx = ds._gene_idx(perts[0]["systematic_gene_name"])
        if idx is None or ds.genes[idx] not in want:
            continue
        gene = ds.genes[idx]
        if gene in out:
            raise ValueError(f"{gene}: two single-deletion records")
        r = ds[i]["representation"]
        if r.context_id != rep043.S288C_CONTEXT:
            raise ValueError(f"data_{raw_index}: context {r.context_id}, not S288C")
        m = record_marks(r)
        if len(m["gene"]) != 1 or not m["on_graph"][0] or int(m["gene"][0]) != idx:
            raise ValueError(f"data_{raw_index}: not one on-graph mark at {gene}")
        out[gene] = {
            "raw_index": raw_index,
            "marks": m,
            "onto": record_ontology_key(record),
        }
    missing = want - set(out)
    if missing:
        raise KeyError(f"{len(missing)} strains have no record: {sorted(missing)[:5]}")
    return out


def _check_marks_against_store(
    store: PreparedStore043, marks: dict[str, Any]
) -> dict[str, Any]:
    """Compare each strain's mark with the prepared store's single-deletion marks of
    the same gene (any dataset); report how many agree field by field.
    """
    n = store.mark_ptr[1:] - store.mark_ptr[:-1]
    single = np.nonzero(n == 1)[0]
    lo = store.mark_ptr[single]
    by_gene: dict[int, int] = {}
    for r, m in zip(single, lo, strict=True):
        if store.mark_on_graph[m] and store.rec_context[r] == 0:
            by_gene.setdefault(int(store.mark_gene[m]), int(m))
    fields = {
        "change": "mark_change",
        "parent": "mark_parent",
        "keys": "mark_keys",
        "gaps": "mark_gaps",
        "replicon": "mark_replicon",
        "relmid": "mark_relmid",
        "chrom_copy": "mark_chrom_copy",
        "gene_copy": "mark_gene_copy",
        "pgap": "mark_pgap",
    }
    n_cmp = 0
    agree = dict.fromkeys(fields, 0)
    for v in marks.values():
        g = int(v["marks"]["gene"][0])
        if g not in by_gene:
            continue
        n_cmp += 1
        m = by_gene[g]
        for f, sf in fields.items():
            a = np.asarray(v["marks"][f][0])
            b = np.asarray(getattr(store, sf)[m])
            agree[f] += int(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    return {"n_strains_compared": n_cmp, "n_agree_by_field": agree}


def ohya_view(
    view: TrainingView, marks: dict[str, Any], strains: list[str]
) -> tuple[TrainingView, dict[str, Any]]:
    """The arm's view with the strains' records appended (S288C context, their marks and
    ontology profile), so ``make_batch`` collates them; positions = the strains in order.
    """
    store = view.store
    if store.meta.context_ids[0] != "S288C":
        raise ValueError("context 0 of the prepared store is not S288C")
    R0, n = len(store.rec_raw_index), len(strains)
    cat = {
        k: np.concatenate([marks[s]["marks"][k] for s in strains])
        for k in marks[strains[0]]["marks"]
    }
    n_marks = np.asarray([len(marks[s]["marks"]["gene"]) for s in strains], np.int64)
    ptr = np.concatenate([store.mark_ptr, store.mark_ptr[-1] + np.cumsum(n_marks)])
    onto_info: dict[str, Any] = {}
    rec_onto_new = np.zeros(n, np.int32)
    onto_profiles = store.onto_profiles
    if store.rec_onto.size:
        vocab = store.ontology_vocabulary
        assert vocab is not None
        keys = sorted({marks[s]["onto"] for s in strains})
        rows = np.stack(
            [
                vocab.vector_from_recipe(
                    AnnotationRecipe.model_validate_json(k)
                ).numpy()
                for k in keys
            ]
        ).astype(np.float32)
        same = [
            int(np.nonzero((store.onto_profiles == row).all(1))[0][0])
            if (store.onto_profiles == row).all(1).any()
            else -1
            for row in rows
        ]
        onto_profiles = np.concatenate([store.onto_profiles, rows])
        pos = {k: len(store.onto_profiles) + j for j, k in enumerate(keys)}
        rec_onto_new = np.asarray([pos[marks[s]["onto"]] for s in strains], np.int32)
        onto_info = {
            "n_ohya_profiles": len(keys),
            "ohya_profile_nodes": [
                {vocab.nodes[j].name: float(row[j]) for j in np.nonzero(row)[0]}
                for row in rows
            ],
            "equal_to_a_training_profile": same,
        }
    new_store = dataclasses.replace(
        store,
        rec_raw_index=np.concatenate(
            [store.rec_raw_index, [marks[s]["raw_index"] for s in strains]]
        ).astype(np.int64),
        rec_context=np.concatenate([store.rec_context, np.zeros(n, np.int32)]),
        rec_env=np.concatenate([store.rec_env, np.zeros(n, np.int32)]),
        rec_onto=np.concatenate([store.rec_onto, rec_onto_new])
        if store.rec_onto.size
        else store.rec_onto,
        onto_profiles=onto_profiles,
        mark_ptr=ptr,
        mark_gene=np.concatenate([store.mark_gene, cat["gene"]]),
        mark_on_graph=np.concatenate([store.mark_on_graph, cat["on_graph"]]),
        mark_change=np.concatenate([store.mark_change, cat["change"]]),
        mark_parent=np.concatenate([store.mark_parent, cat["parent"]]),
        mark_keys=np.concatenate([store.mark_keys, cat["keys"].reshape(-1, 4)]),
        mark_gaps=np.concatenate([store.mark_gaps, cat["gaps"].reshape(-1, 4)]),
        mark_replicon=np.concatenate([store.mark_replicon, cat["replicon"]]),
        mark_relmid=np.concatenate([store.mark_relmid, cat["relmid"]]),
        mark_chrom_copy=np.concatenate([store.mark_chrom_copy, cat["chrom_copy"]]),
        mark_gene_copy=np.concatenate([store.mark_gene_copy, cat["gene_copy"]]),
        mark_pgap=np.concatenate([store.mark_pgap, cat["pgap"]]),
    )
    # compact key rows into the model's tables (build_view's rule); a key with no row
    # in the tables raises
    keys = cat["keys"].reshape(-1, 4)
    new_rows = np.full(keys.shape, -1, np.int32)
    for e in range(4):
        ids = view.used_ids[e]
        v = keys[:, e]
        have = v >= 0
        pos_e = np.searchsorted(ids, v[have])
        if (pos_e >= len(ids)).any() or (
            ids[np.minimum(pos_e, len(ids) - 1)] != v[have]
        ).any():
            raise KeyError(
                f"encoder {e}: an Ohya mark key has no row in the model table"
            )
        new_rows[have, e] = pos_e
    zeros = np.zeros(n, np.int64)
    new_view = dataclasses.replace(
        view,
        store=new_store,
        rec=np.arange(R0, R0 + n, dtype=np.int64),
        local_dataset=zeros,
        head_dataset=zeros,
        level_gene=np.zeros(n, bool),
        role=np.zeros(n, np.int8),
        exp_type=zeros,
        z_scalar=np.concatenate([view.z_scalar, np.full(n, np.nan, np.float32)]),
        mark_rows=np.concatenate([view.mark_rows, new_rows]),
        mark_cap=None,
        train_mask=None,
        group=zeros,
        T=None,
        t_scalar=None,
    )
    return new_view, onto_info


@torch.no_grad()
def trained_representations(
    cfg: ProbeConfig, label: str, strains: list[str], marks: dict[str, Any] | None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """The six read-outs of every strain from one checkpoint (module docstring)."""
    run_dir = osp.join(cfg.data_root, CKPT_ROOT, cfg.checkpoints[label])
    path = osp.join(run_dir, "best.pt")
    if not osp.exists(path):
        path = osp.join(run_dir, "last.pt")
    with open(path, "rb") as fh:
        ck_sha = hashlib.sha256(fh.read()).hexdigest()
    state = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = checkpoint_config(state)
    generation = store_paths(tcfg.data.store_version)
    if generation.version != active_store().version:
        raise ValueError(
            f"{label}: trained on store {generation.version}, the probe reads "
            f"{active_store().version} (STORE_043_VERSION)"
        )
    store = PreparedStore043.load(
        tcfg.paths.resolved_prep_dir(
            tcfg.data.route, generation, tcfg.data.prep_version
        )
    )
    view = build_view(
        store,
        tcfg.arm,
        tcfg.data.mark_cap,
        tcfg.data.target_transform,
        tcfg.data.target_stats_by,
        tcfg.weighting.by,
    )
    source = h5_vector_source(tcfg.paths, generation)
    tables = [source(e, view.used_ids[e]) for e in range(len(view.used_ids))]
    model: CellModel043 = build_model(tcfg, view, tables)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    if marks is None:
        marks = ohya_marks(cfg.data_root, tcfg.data.route, strains)
    mark_check = _check_marks_against_store(store, marks)
    ov, onto_info = ohya_view(view, marks, strains)

    cap: dict[str, torch.Tensor] = {}

    def trunk_hook(_m: Any, _a: Any, o: torch.Tensor) -> None:
        cap["trunk"] = o

    def op_hook(_m: Any, a: tuple[torch.Tensor, ...], o: torch.Tensor) -> None:
        cap["h_in"], cap["h_out"] = a[0], o

    handles = [
        model.trunk.register_forward_hook(trunk_hook),
        model.operator.register_forward_hook(op_hook),
    ]
    out = {
        k: np.zeros((len(strains), model.config.hidden_dim), np.float32)
        for k in TRAINED_REPS
    }
    empty = torch.zeros(0, dtype=torch.long)
    trunk0: torch.Tensor | None = None
    trunk_drift = 0.0
    op_norms: dict[str, float] = {"attn": 0.0, "ffn": 0.0, "update": 0.0}
    t0 = time.time()
    for lo in range(0, len(strains), cfg.batch_size):
        pos = list(range(lo, min(lo + cfg.batch_size, len(strains))))
        batch, _ = make_batch(ov, pos)
        res = model(batch, empty, empty)
        trunk, h_in, h_out = cap["trunk"], cap["h_in"], cap["h_out"]
        if trunk.shape[0] != 1:
            raise ValueError(f"{trunk.shape[0]} contexts in a batch of S288C records")
        if trunk0 is None:
            trunk0 = trunk[0].clone()
        trunk_drift = max(trunk_drift, float((trunk[0] - trunk0).abs().max()))
        g = batch.marks.gene[:, 0] + 1  # one on-graph mark per strain
        b = torch.arange(len(pos))
        rows = slice(lo, lo + len(pos))
        out["trunk_gene"][rows] = trunk[0, g].numpy()
        out["pert_gene"][rows] = h_out[b, g].numpy()
        out["cls"][rows] = h_out[:, 0].numpy()
        out["mean_genes"][rows] = h_out[:, 1:].mean(1).numpy()
        out["update_sum"][rows] = (h_out[:, 1:] - h_in[:, 1:]).sum(1).numpy()
        out["z_s"][rows] = res.z_s.numpy()
        for k, v in model.operator.last_norms.items():
            op_norms[k] += float(v) * len(pos)
        print(
            f"[{label}] {pos[-1] + 1}/{len(strains)} strains, {time.time() - t0:.0f} s",
            flush=True,
        )
    for h in handles:
        h.remove()
    meta = {
        "checkpoint": path,
        "checkpoint_sha256": ck_sha,
        "epoch": int(state["epoch"]),
        "step": int(state["step"]),
        "best_val_loss": state.get("best_val_loss"),
        "best_val_epoch": state.get("best_val_epoch"),
        "arm": tcfg.arm.name,
        "prep_version": store.meta.prep_version,
        "conditioning": tcfg.model.conditioning.model_dump(),
        "hidden_dim": model.config.hidden_dim,
        "num_layers": model.config.num_layers,
        "forward_seconds": round(time.time() - t0, 1),
        "trunk_output_max_abs_drift_across_batches": trunk_drift,
        "operator_relative_norms_mean": {
            k: v / len(strains) for k, v in op_norms.items()
        },
        "marks_vs_prepared_store": mark_check,
        "ontology": onto_info,
    }
    reps = {f"{label}_{k}": pd.DataFrame(v, index=strains) for k, v in out.items()}
    return reps, meta, marks


# ============================================================================ scoring


def _target_z(T: pd.DataFrame, fit: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Targets as z with the fitting strains' mean and SD, clipped at |z| = M_ZCLIP (NaN kept)."""
    A = T.to_numpy(np.float64)
    m, s = np.nanmean(A[fit], 0), np.nanstd(A[fit], 0)
    return np.clip((A - m) / np.where(s > 0, s, 1.0), -M_ZCLIP, M_ZCLIP)


def _r2(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Held-out R^2 per column over finite entries (1 - SSE / SS around the held-out mean)."""
    ok = np.isfinite(Y)
    y = np.where(ok, Y, 0.0)
    n = ok.sum(0)
    ym = y.sum(0) / np.maximum(n, 1)
    sse = (np.where(ok, (Y - P) ** 2, 0.0)).sum(0)
    sst = (np.where(ok, (Y - ym) ** 2, 0.0)).sum(0)
    return np.where(sst > 0, 1.0 - sse / np.where(sst > 0, sst, 1.0), np.nan)


def _summ(r: np.ndarray) -> dict[str, float | int]:
    f = r[np.isfinite(r)]
    return {
        "median_pearson": float(np.median(f)),
        "mean_pearson": float(np.mean(f)),
        "n_features_pearson_gt_0.3": int((f > R_THRESHOLD).sum()),
        "n_features_scored": int(len(f)),
    }


def score_split(
    cfg: ProbeConfig,
    X: pd.DataFrame | None,
    T: pd.DataFrame,
    folds: npt.NDArray[np.int64],
    test_fold: int,
    val_fold: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold-file split (module docstring). ``X`` None = the training-mean null.
    Returns the summary, the held-out predictions, targets, the training mean and the
    held-out row indices.
    """
    tr = np.nonzero(folds != test_fold)[0]
    va = np.nonzero(folds == val_fold)[0]
    te = np.nonzero(folds == test_fold)[0]
    Y = _target_z(T, tr)
    mu = np.nanmean(Y[tr], 0)
    if X is None:
        P = np.tile(mu, (len(te), 1))
        best, scores = None, {}
    else:
        P, best, scores = ridge_fold(
            X.loc[T.index].to_numpy(np.float64),
            None,
            Y,
            tr,
            va,
            te,
            cfg.ridge_log10_grid,
        )
    r = _corr_cols(P, Y[te])
    out: dict[str, Any] = {
        **_summ(r),
        "mean_r2": float(np.nanmean(_r2(P, Y[te]))),
        "n_train_strains": int(len(tr)),
        "n_val_strains": int(len(va)),
        "n_test_strains": int(len(te)),
        "selected_log10_lambda_rel": best,
        "inner_median_pearson_by_log10_lambda_rel": scores,
        "per_feature_pearson": dict(zip(T.columns, [float(x) for x in r], strict=True)),
    }
    return out, P, Y[te], mu, te


def score_oof(X: pd.DataFrame, T: pd.DataFrame) -> dict[str, Any]:
    """The Table morph construction (``linear_evals_043.run_morphology``, raw_all)."""
    Xs = X.loc[T.index]
    Xs = Xs.loc[:, Xs.std(ddof=0) > 0]
    Xa = _mz(Xs).fillna(0.0).to_numpy()
    Ya = _mz(T).fillna(0.0).to_numpy()
    p, t = morph_oof(Xa, Ya, None, residualize=False)
    return _summ(_colr(p, t))


# ============================================================================ figure

FAMILY_LABEL = {
    "null": "null",
    "frozen": "frozen token",
    "bs16": "trained, bs16",
    "onto_inject_op": "trained, onto inject op",
}
REP_LABEL = {
    "null": "training mean",
    "frozen_composite": "composite (3,328)",
    "frozen_fudt_up": "FUDT upstream",
    "frozen_fudt_down": "FUDT downstream",
    "frozen_prott5": "ProtT5-XL",
    "frozen_calm": "CaLM",
    "trunk_gene": "deleted gene, trunk",
    "pert_gene": "deleted gene, after operator",
    "cls": "CLS, after operator",
    "mean_genes": "mean over genes",
    "update_sum": "sum of operator update",
    "z_s": "mark key z_S",
}


def _family(name: str) -> str:
    if name == "null":
        return "null"
    if name.startswith("frozen_"):
        return "frozen"
    for lab in CHECKPOINTS:
        if name.startswith(lab + "_"):
            return lab
    raise KeyError(name)


def _label(name: str) -> str:
    fam = _family(name)
    if fam in ("null", "frozen"):
        return REP_LABEL[name]
    return REP_LABEL[name[len(fam) + 1 :]]


def make_figure(res: dict[str, Any]) -> str:
    """Median per-feature Pearson (95% CI) and the count of features above 0.3, per
    representation, on the held-out strains of the fold file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    from torchcell.timestamp import timestamp
    from torchcell.utils.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    color = {
        "null": PLOT_PALETTE[5],
        "frozen": PLOT_PALETTE[0],
        "bs16": PLOT_PALETTE[1],
        "onto_inject_op": PLOT_PALETTE[2],
    }
    names = list(res["representations"])
    prim = res["representations"]
    n = len(names)
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(8 + 4.2 * n)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.6, 1.0]},
        constrained_layout=True,
    )
    y = np.arange(n)[::-1]
    n_feat = prim[names[0]]["moving_116"]["fold_split"]["n_features_scored"]
    for yi, nm in zip(y, names, strict=True):
        s = prim[nm]["moving_116"]["fold_split"]
        fam = _family(nm)
        med = s["median_pearson"]
        ci = s.get("ci_median_pearson")
        err = None if ci is None else [[max(0.0, med - ci[0])], [max(0.0, ci[1] - med)]]
        a1.barh(
            yi,
            med,
            color=color[fam],
            edgecolor="black",
            linewidth=0.4,
            height=0.7,
            zorder=2,
        )
        if err is not None:
            a1.errorbar(
                med, yi, xerr=err, color="black", linewidth=0.5, capsize=1.2, zorder=3
            )
        a1.plot(
            s["mean_pearson"],
            yi,
            marker="o",
            markersize=2.0,
            color="black",
            markerfacecolor="white",
            markeredgewidth=0.5,
            zorder=4,
            linestyle="none",
        )
        a2.barh(
            yi,
            s["n_features_pearson_gt_0.3"],
            color=color[fam],
            edgecolor="black",
            linewidth=0.4,
            height=0.7,
            zorder=2,
        )
    a1.set_yticks(y)
    a1.set_yticklabels([_label(nm) for nm in names])
    lo_ci = min(
        prim[nm]["moving_116"]["fold_split"].get("ci_median_pearson", [0.0])[0]
        for nm in names
    )
    hi_ci = max(
        prim[nm]["moving_116"]["fold_split"].get("ci_median_pearson", [0.0, 0.0])[-1]
        for nm in names
    )
    a1.set_xlim(np.floor(min(lo_ci, 0.0) * 10) / 10, np.ceil(max(hi_ci, 0.2) * 10) / 10)
    a1.xaxis.set_major_locator(MultipleLocator(0.2))
    a1.xaxis.set_minor_locator(MultipleLocator(0.1))
    a1.tick_params(which="minor", length=0)
    a1.grid(axis="x", which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
    a1.set_xlabel(
        f"Pearson r per CalMorph feature across held-out strains\n"
        f"bar = median over {n_feat} features (95% bootstrap CI), dot = mean"
    )
    a2.set_xlim(0, n_feat)
    a2.grid(axis="x", linewidth=0.3, color="#DDDDDD", zorder=0)
    a2.set_xlabel(f"features with Pearson r > {R_THRESHOLD} (of {n_feat})")
    for ax in (a1, a2):
        ax.axvline(0, color="black", linewidth=0.5, zorder=1)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
    from matplotlib.patches import Patch

    a2.legend(
        handles=[
            Patch(
                facecolor=color[k],
                edgecolor="black",
                linewidth=0.4,
                label=FAMILY_LABEL[k],
            )
            for k in color
        ],
        loc="lower right",
        frameon=False,
    )
    d = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
    os.makedirs(d, exist_ok=True)
    base = osp.join(d, f"morphology_probe_043_{timestamp()}")
    fig.savefig(base + ".png", dpi=300)
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    return base + ".svg"


# ============================================================================ main


def run(cfg: ProbeConfig, only_frozen: bool) -> dict[str, Any]:
    t0 = time.time()
    split_path = osp.join(
        cfg.experiment_dir, "conf", "splits", f"{active_store().split_version}.json"
    )
    split = load_split(split_path)
    lcfg = LinearEvalConfig(
        data_root=cfg.data_root,
        experiment_dir=cfg.experiment_dir,
        split_path=split_path,
    )
    genes = json.load(open(osp.join(lcfg.cache, "genes.json")))
    T116, T281, tsrc = ohya_targets(lcfg, genes)
    strains = list(T116.index)
    folds = np.asarray([split.gene_folds[g] for g in strains], np.int64)
    gidx = {g: i for i, g in enumerate(genes)}
    tokens = load_tokens(lcfg, genes)
    reps: dict[str, pd.DataFrame | None] = {"null": None}
    for e in EMBEDDINGS:
        reps[f"frozen_{e}"] = pd.DataFrame(
            tokens[e][[gidx[g] for g in strains]], index=strains
        )
    models: dict[str, Any] = {}
    if not only_frozen:
        marks = None
        for label in cfg.checkpoints:
            r, meta, marks = trained_representations(cfg, label, strains, marks)
            reps.update(r)
            models[label] = meta
    out: dict[str, Any] = {}
    held: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, X in reps.items():
        s116, P, Yte, mu, _ = score_split(
            cfg, X, T116, folds, split.test_fold, split.val_fold
        )
        s281, *_ = score_split(cfg, X, T281, folds, split.test_fold, split.val_fold)
        held[name] = (P, Yte, mu)
        out[name] = {
            "family": _family(name),
            "dim": None if X is None else int(X.shape[1]),
            "moving_116": {
                "fold_split": s116,
                "oof_5fold_table_morph": None if X is None else score_oof(X, T116),
            },
            "all_281": {
                "fold_split": {
                    k: v for k, v in s281.items() if k != "per_feature_pearson"
                }
            },
        }
        print(
            f"{name}: fold-split median r {s116['median_pearson']:.3f}, mean "
            f"{s116['mean_pearson']:.3f}, n>0.3 {s116['n_features_pearson_gt_0.3']}; "
            + (
                ""
                if X is None
                else f"OOF median {out[name]['moving_116']['oof_5fold_table_morph']['median_pearson']:.3f}"
            ),
            flush=True,
        )
    Pref = held[REFERENCE][0]
    for name, (P, Yte, mu) in held.items():
        if name == "null":
            continue
        b = paired_boot(
            P, None if name == REFERENCE else Pref, Yte, mu, cfg.n_boot, cfg.boot_seed
        )
        s = out[name]["moving_116"]["fold_split"]
        s["ci_median_pearson"] = b["ci_median_gene_pearson"]
        if name != REFERENCE:
            s["ci_diff_median_pearson_vs_frozen_composite"] = b[
                "ci_diff_median_gene_pearson"
            ]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=SCRIPTS,
    ).stdout.strip()
    res = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/morphology_probe_043.py",
        "torchcell_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "store_version": active_store().version,
        "split_version": split.version,
        "split_store_sha256": split.store_sha256,
        "test_fold": split.test_fold,
        "val_fold": split.val_fold,
        "config": cfg.model_dump(),
        "n_strains": len(strains),
        "n_features_moving": int(T116.shape[1]),
        "n_features_all": int(T281.shape[1]),
        "targets": tsrc,
        "frozen_blocks": BLOCKS,
        "models": models,
        "statistic": (
            "Pearson r per CalMorph feature across held-out strains (fold_split: the "
            "fold-0 strains of the fold file, penalty on fold 1; oof_5fold_table_morph: "
            "5-fold out-of-fold over every strain, the linear_evals_043 construction); "
            "median and mean over features, count above 0.3; a constant prediction "
            "scores 0"
        ),
        "representations": out,
        "wall_seconds": round(time.time() - t0, 1),
    }
    res["figure"] = make_figure(res)
    os.makedirs(osp.dirname(cfg.json_path), exist_ok=True)
    with open(cfg.json_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {cfg.json_path} and {res['figure']}")
    return res


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--only-frozen",
        action="store_true",
        help="skip the checkpoints (a quick check of the frozen rows)",
    )
    ap.add_argument(
        "--figure-from-json",
        action="store_true",
        help="redraw the figure from the written json (no refit)",
    )
    args = ap.parse_args()
    load_dotenv()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    cfg = ProbeConfig(
        data_root=os.environ["DATA_ROOT"],
        experiment_dir=osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT),
        batch_size=args.batch_size,
        n_boot=args.n_boot,
    )
    if args.figure_from_json:
        with open(cfg.json_path) as f:
            res = json.load(f)
        res["figure"] = make_figure(res)
        with open(cfg.json_path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {res['figure']}")
        return
    run(cfg, args.only_frozen)


if __name__ == "__main__":
    main()

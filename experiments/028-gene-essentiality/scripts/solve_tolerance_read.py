# experiments/028-gene-essentiality/scripts/solve_tolerance_read.py
# [[experiments.028-gene-essentiality.scripts.solve_tolerance_read]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/solve_tolerance_read

"""Does a loose-tolerance or first-order solve rank the genes as the exact one does?

The scalable module form (round 27) reads the growth-maximizing flux of the GEM-box
program, solved once per genotype by HiGHS's simplex at its default tolerances (1e-7),
through a linear head over log(1 + |v*|). A batched solve on the card (NVIDIA cuOpt) is a
first-order method, PDLP, run to a relative tolerance of 1e-4 in its own benchmark, and
it returns a point of the optimal face rather than a vertex. Before any card is involved,
the same question is answerable on the CPU with HiGHS 1.15, which carries the PDLP
algorithm (the cuPDLP-C port), an interior-point solver that can stop before crossover,
and the simplex at any tolerance.

For every labeled gene of the released split the program is the round-27 one: the
GEM's own box clamped to the layer's flux scale, the YPD medium on the exchanges, the
gene's reactions closed through the GPR (``cobra.manipulation.knock_out_model_genes``),
the split-variable form in micromoles (``lp_layer.UNIT_SCALE``), maximize growth. It is
solved by

* ``simplex_exact``: dual simplex at HiGHS's defaults (the reference);
* ``simplex_1e-4``: the same with primal and dual feasibility tolerances at 1e-4;
* ``ipm_nocross_1e-4``: interior point without crossover at KKT tolerance 1e-4, the
  optimal-face point a first-order method also approaches;
* ``pdlp_1e-4``: HiGHS's PDLP at KKT tolerance 1e-4, the cuOpt algorithm on the CPU.

Reads against the reference, per solver: the growth ratio's agreement (the maximum
absolute difference, and its AUROC on the 195 held-out genes and on all labeled genes),
the flux vector's agreement (relative L2 distance of log(1 + |v|), the fraction of
reactions moved by more than 0.01 in that encoding), the wall time per solve, and the
head read: a logistic regression over log(1 + |v|) fitted on the 629 training genes'
REFERENCE vectors and scored on the 195 held-out genes' vectors from each solver, which
is what a head trained on exact solves would see at inference over a loose one, beside
the same probe fitted and scored within each solver. The probe is not the transformer's
head; it is the linear read over the same 4,131 columns and it says whether the
columns still carry the ranking.

Writes ``results/solve_tolerance_read.json``. Needs ``highspy`` on the path (HiGHS
1.15; the torchcell env carries scipy's HiGHS 1.8, which has no PDLP and no crossover
switch), e.g. ``PYTHONPATH=<dir with highspy>:.``.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT_DIR = osp.join(RESULTS_DIR, "splits")

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.lp_layer import UNIT_SCALE, split_bounds  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

#: The layer's flux scale, as ess_reference_flux_table.py and the flux layer clamp.
FLUX_SCALE = 100.0
SPLIT_FILE = "ess_fcl_val0.2_s0.json.gz"
MEDIUM = "YPD"

#: Solver name -> HiGHS options. Everything not named is the HiGHS default.
SOLVERS: dict[str, dict[str, Any]] = {
    "simplex_exact": {"solver": "simplex"},
    # The same tolerances by the primal simplex: another exact vertex of the same
    # optimal face, so the distance between two exact solves is the non-uniqueness of
    # the argmax and not the tolerance (measured 2026-09-18 on three programs: the dual
    # simplex at 1e-4 moved 3 percent of the reactions with the growth unchanged to
    # 1e-6, which is what a different vertex looks like).
    "simplex_primal_exact": {"solver": "simplex", "simplex_strategy": 4},
    "simplex_1e-4": {
        "solver": "simplex",
        "primal_feasibility_tolerance": 1e-4,
        "dual_feasibility_tolerance": 1e-4,
    },
    "ipm_nocross_1e-4": {
        "solver": "ipm",
        "run_crossover": "off",
        "kkt_tolerance": 1e-4,
    },
    "pdlp_1e-4": {"solver": "pdlp", "run_crossover": "off", "kkt_tolerance": 1e-4},
}
REFERENCE = "simplex_exact"
#: Per-solve wall clock cap; a solver that hits it records the status and a zero flux.
TIME_LIMIT_S = 120.0

_LP: dict[str, Any] = {}


def _init_worker(a_csc: sp.csc_matrix, c: np.ndarray) -> None:
    """Each worker holds the program's matrix; a Highs object is built per solve."""
    import highspy

    m, n = a_csc.shape
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = -c  # minimize -growth
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = np.full(n, np.inf)
    lp.row_lower_ = np.zeros(m)
    lp.row_upper_ = np.zeros(m)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = a_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = a_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = a_csc.data.astype(np.float64)
    _LP["lp"] = lp
    _LP["c"] = c


def _fresh_highs(name: str) -> Any:
    """A NEW Highs object per solve, holding the program and one solver's options.

    HiGHS's ``time_limit`` is read against a clock the object keeps for its whole life,
    not per ``run()``: with one object per worker, PDLP at 11 to 22 s a solve stopped
    with an instant non-optimal status after about ten programs per worker, exactly
    where the cumulative run time crossed the 120 s limit (measured 2026-09-18 under
    both forked and spawned pools; 77 to 80 of 1,121 optimal each time, the simplex and
    interior-point solvers at 0.1 s a solve never reached it). A fresh object per solve
    is a few milliseconds of ``passModel`` and makes every solve independent of the
    object's history, which a comparison of solvers wants anyway.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", TIME_LIMIT_S)
    for key, value in SOLVERS[name].items():
        status = h.setOptionValue(key, value)
        assert status == highspy.HighsStatus.kOk, (name, key, value, status)
    h.passModel(_LP["lp"])
    return h


def _solve_gene(args: tuple[str, np.ndarray, np.ndarray]) -> dict[str, Any]:
    """One gene's program under every solver: growth, flux, seconds, status."""
    import highspy

    gene, lb, ub = args
    ub = np.maximum(ub, lb)
    lo, hi = split_bounds(lb * UNIT_SCALE, ub * UNIT_SCALE)
    n = lo.size
    r = n // 2
    out: dict[str, Any] = {"gene": gene}
    for name in SOLVERS:
        h = _fresh_highs(name)
        h.changeColsBounds(n, np.arange(n, dtype=np.int32), lo, hi)
        t0 = time.perf_counter()
        h.run()
        seconds = time.perf_counter() - t0
        status = h.getModelStatus()
        ok = status == highspy.HighsModelStatus.kOptimal
        if ok:
            x = np.asarray(h.getSolution().col_value, dtype=np.float64) / UNIT_SCALE
            v = x[:r] - x[r:]
            growth = float(v[np.flatnonzero(_LP["c"][:r])[0]])
        else:
            v = np.zeros(r)
            growth = 0.0
        out[name] = {
            "growth": growth,
            "v": v.astype(np.float32),
            "seconds": seconds,
            "status": h.modelStatusToString(status),
            "iterations": int(
                h.getInfo().simplex_iteration_count
                if name.startswith("simplex")
                else (
                    h.getInfo().ipm_iteration_count
                    if name.startswith("ipm")
                    else h.getInfo().pdlp_iteration_count
                )
            ),
        }
    return out


class SolverRead(BaseModel):
    """One solver's agreement with the reference and the probe's read of its vectors."""

    solver: str
    options: dict[str, Any]
    n_optimal: int
    n_genes: int
    seconds_median: float
    seconds_total: float
    iterations_median: float
    growth_ratio_max_abs_diff: float
    growth_ratio_auroc_test: float
    growth_ratio_auroc_all: float
    log1p_flux_relative_l2_median: float
    log1p_flux_relative_l2_max: float
    frac_reactions_moved_gt_0_01_mean: float
    probe_auroc_test_fit_on_reference: float
    probe_auroc_test_fit_on_reference_ci: list[float]
    probe_auroc_test_fit_within: float
    probe_auroc_test_fit_within_ci: list[float]
    probe_spearman_with_reference_scores: float


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            draws.append(float(roc_auc_score(y[idx], s[idx])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    import highspy
    from scipy.stats import spearmanr

    started = time.time()
    with gzip.open(osp.join(SPLIT_DIR, SPLIT_FILE), "rt") as fh:
        split = json.load(fh)
    labels: dict[str, int] = {g: int(v) for g, v in split["labels"].items()}
    ordered = sorted(labels)
    # The split's ``train``/``val``/``test`` blocks are dataset record indices; the gene
    # names of each block are under ``genes``.
    train_genes = list(split["genes"]["train"])
    test_genes = list(split["genes"]["test"])
    assert len(test_genes) == 195 and len(train_genes) == 629, (
        len(test_genes),
        len(train_genes),
    )

    source = YeastGEM()
    model = source.model
    media = media_report(model, MEDIUM)
    gem = build_gem_tensors(model, source.model_dir)
    lb0 = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    ub0 = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    biomass = int(gem.biomass_index)
    r = len(gem.rxn_ids)
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    a_csc = sp.hstack([sp.csr_matrix(s_dense), -sp.csr_matrix(s_dense)]).tocsc()
    c = np.zeros(2 * r)
    c[biomass] = 1.0
    c[r + biomass] = -1.0
    gem_genes = {g.id for g in model.genes}

    jobs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for gene in ordered:
        if gene not in gem_genes:
            jobs.append((gene, lb0, ub0))
            continue
        with model:
            knock_out_model_genes(model, [gene])
            k_lb = np.clip(
                np.array([rx.lower_bound for rx in model.reactions], dtype=np.float64),
                -FLUX_SCALE,
                FLUX_SCALE,
            )
            k_ub = np.clip(
                np.array([rx.upper_bound for rx in model.reactions], dtype=np.float64),
                -FLUX_SCALE,
                FLUX_SCALE,
            )
        jobs.append((gene, k_lb, k_ub))
    print(
        f"{len(jobs)} programs built, {time.time() - started:.0f}s; solving under {list(SOLVERS)}"
    )

    workers = int(os.environ.get("TC_LP_WORKERS", "8"))
    # Spawned workers rebuild the program from the pickled matrix; the PDLP failure
    # first blamed on the fork was the object-lifetime time limit (see _fresh_highs),
    # and spawning is kept so no worker inherits a solver's thread state.
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(a_csc, c)) as pool:
        rows = pool.map(_solve_gene, jobs, chunksize=4)
    print(f"solved, {time.time() - started:.0f}s")

    y_all = np.array([labels[g] for g in ordered])
    idx_test = np.array([ordered.index(g) for g in test_genes])
    idx_train = np.array([ordered.index(g) for g in train_genes])
    growth = {name: np.array([row[name]["growth"] for row in rows]) for name in SOLVERS}
    flux = {name: np.stack([row[name]["v"] for row in rows]) for name in SOLVERS}
    enc = {name: np.log1p(np.abs(flux[name])) for name in SOLVERS}
    _init_worker(a_csc, c)
    wild_row = _solve_gene(("wild_type", lb0, ub0))
    assert wild_row[REFERENCE]["status"] == "Optimal", wild_row[REFERENCE]["status"]
    wild_growth = float(wild_row[REFERENCE]["growth"])
    ratio = {name: growth[name] / wild_growth for name in SOLVERS}

    ref_enc = enc[REFERENCE]
    probe_ref = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")
    mu = ref_enc[idx_train].mean(axis=0)
    sd = ref_enc[idx_train].std(axis=0)
    sd[sd < 1e-8] = 1.0
    probe_ref.fit((ref_enc[idx_train] - mu) / sd, y_all[idx_train])
    ref_scores_test = probe_ref.decision_function((ref_enc[idx_test] - mu) / sd)

    reads: list[SolverRead] = []
    for name, options in SOLVERS.items():
        e = enc[name]
        d = np.linalg.norm(e - ref_enc, axis=1) / np.maximum(
            np.linalg.norm(ref_enc, axis=1), 1e-12
        )
        moved = (np.abs(e - ref_enc) > 0.01).mean(axis=1)
        scores_from_ref = probe_ref.decision_function((e[idx_test] - mu) / sd)
        probe_w = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")
        mu_w = e[idx_train].mean(axis=0)
        sd_w = e[idx_train].std(axis=0)
        sd_w[sd_w < 1e-8] = 1.0
        probe_w.fit((e[idx_train] - mu_w) / sd_w, y_all[idx_train])
        scores_within = probe_w.decision_function((e[idx_test] - mu_w) / sd_w)
        secs = np.array([row[name]["seconds"] for row in rows])
        its = np.array([row[name]["iterations"] for row in rows])
        # Lower growth ratio = more essential: score is 1 - ratio.
        reads.append(
            SolverRead(
                solver=name,
                options=options,
                n_optimal=int(sum(row[name]["status"] == "Optimal" for row in rows)),
                n_genes=len(rows),
                seconds_median=float(np.median(secs)),
                seconds_total=float(secs.sum()),
                iterations_median=float(np.median(its)),
                growth_ratio_max_abs_diff=float(
                    np.abs(ratio[name] - ratio[REFERENCE]).max()
                ),
                growth_ratio_auroc_test=float(
                    roc_auc_score(y_all[idx_test], 1 - ratio[name][idx_test])
                ),
                growth_ratio_auroc_all=float(roc_auc_score(y_all, 1 - ratio[name])),
                log1p_flux_relative_l2_median=float(np.median(d)),
                log1p_flux_relative_l2_max=float(d.max()),
                frac_reactions_moved_gt_0_01_mean=float(moved.mean()),
                probe_auroc_test_fit_on_reference=float(
                    roc_auc_score(y_all[idx_test], scores_from_ref)
                ),
                probe_auroc_test_fit_on_reference_ci=bootstrap_auroc(
                    y_all[idx_test], scores_from_ref
                ),
                probe_auroc_test_fit_within=float(
                    roc_auc_score(y_all[idx_test], scores_within)
                ),
                probe_auroc_test_fit_within_ci=bootstrap_auroc(
                    y_all[idx_test], scores_within
                ),
                probe_spearman_with_reference_scores=float(
                    spearmanr(scores_from_ref, ref_scores_test).correlation
                ),
            )
        )
        print(reads[-1].model_dump_json(indent=None))

    # THE PDLP STORE. The one store no other script builds: the cuOpt algorithm's own
    # point for every labeled gene, written in the shape ess_reference_flux_table.py
    # writes (genes, v_deletion, v_wild_type, rxn_ids, provenance; no statistics
    # block, so only the `pfba` feature kind reads it), for the training test of a
    # first-order store (round 38). Programs PDLP reported non-optimal keep the zero
    # flux and are counted.
    table_dir = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
    os.makedirs(table_dir, exist_ok=True)
    pdlp = "pdlp_1e-4"
    store_path = osp.join(table_dir, f"ess_solve_encodings_{MEDIUM}_pdlp.npz")
    store_provenance = {
        "gem": osp.basename(source.model_dir),
        "flux_scale": FLUX_SCALE,
        "method": (
            f"growth-maximizing LP (the round-27 program, split form in micromoles) by "
            f"HiGHS {highspy.Highs().version()} PDLP, no crossover, KKT 1e-4, with the "
            "gene deleted through cobra.manipulation.knock_out_model_genes; written by "
            "solve_tolerance_read.py"
        ),
        "solver": "pdlp",
        "options": SOLVERS[pdlp],
        "medium": MEDIUM,
        "split_file": SPLIT_FILE,
        "wild_type_growth": float(wild_row[pdlp]["growth"]),
        "wild_type_status": wild_row[pdlp]["status"],
        "n_nonoptimal_zero_flux": int(
            sum(row[pdlp]["status"] != "Optimal" for row in rows)
        ),
        "n_lethal_zero_flux": int(
            sum(
                row[pdlp]["status"] == "Optimal" and row[pdlp]["growth"] < 1e-6
                for row in rows
            )
        ),
    }
    np.savez(
        store_path,
        genes=np.array(ordered),
        v_deletion=flux[pdlp].astype(np.float32),
        v_wild_type=wild_row[pdlp]["v"].astype(np.float32),
        rxn_ids=np.array(list(gem.rxn_ids)),
        provenance=np.array(json.dumps(store_provenance)),
    )
    print(
        f"PDLP store {store_path}: {store_provenance['n_nonoptimal_zero_flux']} non-optimal rows"
    )

    out = {
        "script": "solve_tolerance_read.py",
        "pdlp_store": store_path,
        "pdlp_store_provenance": store_provenance,
        "program": (
            "GEM box clamped to +/-100 mmol/gDW/h, YPD on the exchanges, the gene's "
            "reactions closed through the GPR, split variables in micromoles, maximize "
            "growth (the round-27 program)"
        ),
        "gem": osp.basename(source.model_dir),
        "medium": {k: v for k, v in media.items() if k != "unresolved"},
        "highs_version": highspy.Highs().version(),
        "wild_type_growth": wild_growth,
        "n_genes": len(ordered),
        "n_test": len(test_genes),
        "n_train": len(train_genes),
        "n_genes_in_gem": sum(g in gem_genes for g in ordered),
        "reference": REFERENCE,
        "time_limit_s": TIME_LIMIT_S,
        "workers": workers,
        "probe": (
            "LogisticRegression(C=0.01, balanced) over log(1 + |v|) standardized on the "
            "training genes; 'fit_on_reference' is fitted on the reference solver's "
            "vectors and scored on each solver's held-out vectors, 'fit_within' fitted "
            "and scored on the same solver"
        ),
        "reads": [x.model_dump() for x in reads],
        "seconds": time.time() - started,
    }
    path = osp.join(RESULTS_DIR, "solve_tolerance_read.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{time.time() - started:.0f}s\n{path}")


if __name__ == "__main__":
    main()

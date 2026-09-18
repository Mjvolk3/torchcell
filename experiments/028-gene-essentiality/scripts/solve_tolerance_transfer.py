# experiments/028-gene-essentiality/scripts/solve_tolerance_transfer.py
# [[experiments.028-gene-essentiality.scripts.solve_tolerance_transfer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/solve_tolerance_transfer

"""Does a head trained on one tolerance of a first-order solve transfer to another?

The solve-tolerance read (``solve_tolerance_read.py``) found the program's argmax
degenerate: every solver picks its own point of the optimal face, a probe fitted on the
reference vertex collapses on any other solver's point, and a point inside the face
(interior point, PDLP) is as rankable as a vertex when the probe is fitted on it. What
that read did not separate is the change of point from the change of tolerance: PDLP at
KKT 1e-4 was compared with the exact vertex, never with PDLP at a tighter tolerance. A
store built on the card at one tolerance and a head served at another is the case the
module's scalable form would meet, so the question is whether the same solver's points
at 1e-4, 1e-6 and 1e-8 carry one ranking.

The program is the round-27 one for every labeled gene of the released split (the GEM
box clamped to the layer's flux scale, YPD on the exchanges, the gene's reactions closed
through the GPR, split variables in micromoles, maximize growth), solved by

* ``simplex_exact``: the dual simplex at HiGHS's defaults, the anchor;
* ``ipm_nocross_1e-4``, ``ipm_nocross_1e-6``, ``ipm_nocross_1e-8``: the interior point
  without crossover at three KKT tolerances;
* ``pdlp_1e-4``, ``pdlp_1e-6``, ``pdlp_1e-8``: HiGHS's PDLP (the cuOpt algorithm on the
  CPU) at the same three tolerances.

Reads, per ordered pair of solvers (fit, score): a logistic probe over log(1 + |v|)
fitted on the 629 training genes' vectors from the ``fit`` solver and scored on the 195
held-out genes' vectors from the ``score`` solver (AUROC with a bootstrap interval), and
the two solvers' vector agreement (median relative L2 of log(1 + |v|), the fraction of
reactions moved by more than 0.01 in that encoding). The diagonal is the within-solver
read; the off-diagonal within a family (PDLP 1e-4 fitted, PDLP 1e-6 scored) is the
transfer across tolerance; across families it is the transfer across point.

Writes ``results/solve_tolerance_transfer.json`` and the PDLP stores at 1e-6 and 1e-8
(``ess_solve_encodings_YPD_pdlp1e-6.npz``, ``..._pdlp1e-8.npz``) in the shape the
trainer's ``ESS_SOLVE_TABLE`` reads, for a training round over them beside the 1e-4
store of round 38. Programs a solver reports non-optimal (the 120 s cap included) keep
the zero flux and are counted. Needs ``highspy`` 1.15 on the path (``TC_HIGHSPY_PATH``
in the launcher); the solving helpers are those of ``solve_tolerance_read.py`` with this
script's solver table installed in every spawned worker.
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
SCRIPTS_DIR = osp.dirname(osp.abspath(__file__))

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)
import solve_tolerance_read as base  # noqa: E402
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

FLUX_SCALE = base.FLUX_SCALE
SPLIT_FILE = base.SPLIT_FILE
MEDIUM = base.MEDIUM

SOLVERS: dict[str, dict[str, Any]] = {
    "simplex_exact": {"solver": "simplex"},
    "ipm_nocross_1e-4": {"solver": "ipm", "run_crossover": "off", "kkt_tolerance": 1e-4},
    "ipm_nocross_1e-6": {"solver": "ipm", "run_crossover": "off", "kkt_tolerance": 1e-6},
    "ipm_nocross_1e-8": {"solver": "ipm", "run_crossover": "off", "kkt_tolerance": 1e-8},
    "pdlp_1e-4": {"solver": "pdlp", "run_crossover": "off", "kkt_tolerance": 1e-4},
    "pdlp_1e-6": {"solver": "pdlp", "run_crossover": "off", "kkt_tolerance": 1e-6},
    "pdlp_1e-8": {"solver": "pdlp", "run_crossover": "off", "kkt_tolerance": 1e-8},
}
#: The stores written, keyed by the file suffix after ``ess_solve_encodings_YPD_``.
STORES = {"pdlp1e-6": "pdlp_1e-6", "pdlp1e-8": "pdlp_1e-8"}


def _init_worker(a_csc: sp.csc_matrix, c: np.ndarray) -> None:
    """The base initializer, then this script's solver table in the worker's module."""
    base._init_worker(a_csc, c)
    base.SOLVERS = SOLVERS


class PairRead(BaseModel):
    """A probe fitted on one solver's training vectors, scored on another's held-out."""

    fit: str
    score: str
    probe_auroc_test: float
    probe_auroc_test_ci: list[float]
    log1p_flux_relative_l2_median: float
    frac_reactions_moved_gt_0_01_mean: float


class SolverSummary(BaseModel):
    solver: str
    options: dict[str, Any]
    n_optimal: int
    n_timed_out_or_other: int
    seconds_median: float
    seconds_total: float
    iterations_median: float
    growth_ratio_max_abs_diff_vs_reference: float
    growth_ratio_auroc_test: float


def main() -> None:
    import highspy

    started = time.time()
    with gzip.open(osp.join(SPLIT_DIR, SPLIT_FILE), "rt") as fh:
        split = json.load(fh)
    labels: dict[str, int] = {g: int(v) for g, v in split["labels"].items()}
    ordered = sorted(labels)
    train_genes = list(split["genes"]["train"])
    test_genes = list(split["genes"]["test"])
    assert len(test_genes) == 195 and len(train_genes) == 629

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
    print(f"{len(jobs)} programs built, {time.time() - started:.0f}s; solving under {list(SOLVERS)}")

    workers = int(os.environ.get("TC_LP_WORKERS", "16"))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(a_csc, c)) as pool:
        rows = pool.map(base._solve_gene, jobs, chunksize=2)
    print(f"solved, {time.time() - started:.0f}s")
    _init_worker(a_csc, c)
    wild_row = base._solve_gene(("wild_type", lb0, ub0))
    reference = "simplex_exact"
    assert wild_row[reference]["status"] == "Optimal", wild_row[reference]["status"]
    wild_growth = float(wild_row[reference]["growth"])

    y_all = np.array([labels[g] for g in ordered])
    idx_test = np.array([ordered.index(g) for g in test_genes])
    idx_train = np.array([ordered.index(g) for g in train_genes])
    growth = {name: np.array([row[name]["growth"] for row in rows]) for name in SOLVERS}
    flux = {name: np.stack([row[name]["v"] for row in rows]) for name in SOLVERS}
    enc = {name: np.log1p(np.abs(flux[name])) for name in SOLVERS}
    ratio = {name: growth[name] / wild_growth for name in SOLVERS}

    summaries: list[SolverSummary] = []
    probes: dict[str, tuple[LogisticRegression, np.ndarray, np.ndarray]] = {}
    for name, options in SOLVERS.items():
        e = enc[name]
        mu = e[idx_train].mean(axis=0)
        sd = e[idx_train].std(axis=0)
        sd[sd < 1e-8] = 1.0
        probe = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced")
        probe.fit((e[idx_train] - mu) / sd, y_all[idx_train])
        probes[name] = (probe, mu, sd)
        secs = np.array([row[name]["seconds"] for row in rows])
        its = np.array([row[name]["iterations"] for row in rows])
        n_opt = int(sum(row[name]["status"] == "Optimal" for row in rows))
        summaries.append(
            SolverSummary(
                solver=name,
                options=options,
                n_optimal=n_opt,
                n_timed_out_or_other=len(rows) - n_opt,
                seconds_median=float(np.median(secs)),
                seconds_total=float(secs.sum()),
                iterations_median=float(np.median(its)),
                growth_ratio_max_abs_diff_vs_reference=float(
                    np.abs(ratio[name] - ratio[reference]).max()
                ),
                growth_ratio_auroc_test=float(
                    roc_auc_score(y_all[idx_test], 1 - ratio[name][idx_test])
                ),
            )
        )
        print(summaries[-1].model_dump_json())

    pairs: list[PairRead] = []
    for fit_name, (probe, mu, sd) in probes.items():
        for score_name in SOLVERS:
            e = enc[score_name]
            scores = probe.decision_function((e[idx_test] - mu) / sd)
            d = np.linalg.norm(e - enc[fit_name], axis=1) / np.maximum(
                np.linalg.norm(enc[fit_name], axis=1), 1e-12
            )
            moved = (np.abs(e - enc[fit_name]) > 0.01).mean(axis=1)
            pairs.append(
                PairRead(
                    fit=fit_name,
                    score=score_name,
                    probe_auroc_test=float(roc_auc_score(y_all[idx_test], scores)),
                    probe_auroc_test_ci=base.bootstrap_auroc(y_all[idx_test], scores),
                    log1p_flux_relative_l2_median=float(np.median(d)),
                    frac_reactions_moved_gt_0_01_mean=float(moved.mean()),
                )
            )
            print(pairs[-1].model_dump_json())

    table_dir = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
    stores: dict[str, dict[str, Any]] = {}
    for suffix, name in STORES.items():
        store_path = osp.join(table_dir, f"ess_solve_encodings_{MEDIUM}_{suffix}.npz")
        provenance = {
            "gem": osp.basename(source.model_dir),
            "flux_scale": FLUX_SCALE,
            "method": (
                f"growth-maximizing LP (the round-27 program, split form in micromoles) by "
                f"HiGHS {highspy.Highs().version()} PDLP, no crossover, KKT "
                f"{SOLVERS[name]['kkt_tolerance']:g}, with the gene deleted through "
                "cobra.manipulation.knock_out_model_genes; written by "
                "solve_tolerance_transfer.py"
            ),
            "solver": "pdlp",
            "options": SOLVERS[name],
            "medium": MEDIUM,
            "split_file": SPLIT_FILE,
            "wild_type_growth": float(wild_row[name]["growth"]),
            "wild_type_status": wild_row[name]["status"],
            "n_nonoptimal_zero_flux": int(
                sum(row[name]["status"] != "Optimal" for row in rows)
            ),
            "n_lethal_zero_flux": int(
                sum(
                    row[name]["status"] == "Optimal" and row[name]["growth"] < 1e-6
                    for row in rows
                )
            ),
        }
        np.savez(
            store_path,
            genes=np.array(ordered),
            v_deletion=flux[name].astype(np.float32),
            v_wild_type=wild_row[name]["v"].astype(np.float32),
            rxn_ids=np.array(list(gem.rxn_ids)),
            provenance=np.array(json.dumps(provenance)),
        )
        stores[suffix] = {"path": store_path, "provenance": provenance}
        print(f"store {store_path}: {provenance['n_nonoptimal_zero_flux']} non-optimal rows")

    out = {
        "script": "solve_tolerance_transfer.py",
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
        "reference": reference,
        "time_limit_s": base.TIME_LIMIT_S,
        "workers": workers,
        "probe": (
            "LogisticRegression(C=0.01, balanced) over log(1 + |v|) standardized on the "
            "fit solver's training genes; each pair fits on one solver's training "
            "vectors and scores another's held-out vectors"
        ),
        "solvers": [x.model_dump() for x in summaries],
        "pairs": [x.model_dump() for x in pairs],
        "stores": stores,
        "seconds": time.time() - started,
    }
    path = osp.join(RESULTS_DIR, "solve_tolerance_transfer.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{time.time() - started:.0f}s\n{path}")


if __name__ == "__main__":
    main()

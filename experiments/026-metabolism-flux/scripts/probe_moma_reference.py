# experiments/026-metabolism-flux/scripts/probe_moma_reference.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_moma_reference
r"""Would a MOMA reference be a DIFFERENT reference, or the same one?

THE QUESTION AND WHY IT IS WORTH A CPU JOB BEFORE ANY TABLE IS BUILT. The per-genotype pFBA
table changes essentially nothing: 3,991 of 4,930 deleted genes are outside the GEM, and of
the 939 inside only **201** move the parsimonious optimum at all, so a random batch of 32
shows about 2 genotypes with a reference different from the wild type by more than 1 percent
(job 1973). The note's own suggestion is MOMA, on the argument that it is defined as a
minimal ADJUSTMENT from the wild-type flux and therefore moves whenever the knockout forces
any rerouting. That argument is a hypothesis until the distance is measured, and it can fail
in BOTH directions:

* if MOMA lands on the pFBA knockout solution, it is the same reference under another name
  and a MOMA table would train exactly as the pFBA table did;
* if MOMA lands on the wild type, it is the SINGLE-reference head under another name, which
  is what job 1946 already ran.

So three distances per gene, all relative to :math:`\lVert v_{wt}\rVert`, and MOMA is
informative only if the first is large while the second is not zero.

THE PROGRAM, and it is a QP rather than an LP:

.. math::
    \min_v \tfrac12\lVert v-v_{wt}\rVert_2^2 \quad\text{s.t.}\quad Sv=0,\;
    \ell^{KO}\le v\le u^{KO}.

NO QP SOLVER IS INSTALLED in this environment (optlang carries GLPK and SCIPY, both LP
only), so ``cobra.flux_analysis.moma`` cannot run its quadratic form here and the linear
form would answer a different question. :func:`solve_qp` is an OSQP-style ADMM with one
sparse LU factorization of the KKT matrix, reused across every iteration and every gene.
Its correctness is CHECKED rather than assumed: on the unperturbed bounds the program's
solution is :math:`v_{wt}` itself, so the wild-type round-trip distance is reported first
and a run whose round trip is not near zero is refused.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_moma_reference.py --genes 30

Writes ``experiments/026-metabolism-flux/results/probe_moma_reference.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from scipy.sparse import csc_matrix, csr_matrix, eye, hstack, vstack
from scipy.sparse.linalg import splu

load_dotenv()

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
SCRIPTS_026 = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
sys.path.insert(0, SCRIPTS_026)

from train_flux import media_report  # noqa: E402

TABLE_PATH = osp.join(
    DATA_ROOT, "data/torchcell/reference_flux/reference_flux_table_ypd.npz"
)
FLUX_SCALE = 100.0
#: A genotype counts as one whose pFBA reference MOVED at the threshold the table records.
MOVED_THRESHOLD = 0.01


def solve_qp(
    s: csr_matrix,
    lb: np.ndarray,
    ub: np.ndarray,
    target: np.ndarray,
    rho: float = 1.0e-1,
    sigma: float = 1.0e-6,
    relax: float = 1.6,
    max_iter: int = 100000,
    eps_abs: float = 1.0e-7,
    eps_rel: float = 1.0e-8,
    check_every: int = 25,
    adapt_every: int = 200,
) -> tuple[np.ndarray, dict[str, float]]:
    r"""Minimize :math:`\tfrac12\lVert v-\text{target}\rVert^2` over :math:`Sv=0`, box.

    OSQP's ADMM, written out because no QP package is installed here. It factorizes

    .. math::
        \begin{bmatrix}P+\sigma I & A^\top\\ A & -\rho^{-1}I\end{bmatrix},\qquad
        A=\begin{bmatrix}S\\ I\end{bmatrix},

    and reuses the LU across iterations. The equality rows carry a penalty
    :math:`10^3\rho`, which is OSQP's own rule and is what keeps :math:`Sv=0` tight while
    the box rows stay soft.

    **ADAPTIVE RHO, and it is not optional here.** At a fixed :math:`\rho` the first
    knockout ran the full 20,000 iterations and stopped with a primal residual of
    :math:`1.2\times10^{-2}`, which is the same order as the distances being measured, so
    every number would have been solver noise. The step size is re-estimated from the ratio
    of the normalized residuals and the KKT is refactorized whenever it moves by more than
    5x, which is OSQP's rule and costs a handful of factorizations per solve.

    Termination is OSQP's relative test on both residuals. Returns the solution and a record
    carrying the residuals and whether the test was MET, so a non-converged solve is visible
    to the caller rather than silently reported as a distance.
    """
    n = s.shape[1]
    m_eq = s.shape[0]
    a = vstack([s, eye(n, format="csr")], format="csr")
    a_t = csr_matrix(a.T)
    m = a.shape[0]
    eq_scale = np.ones(m)
    eq_scale[:m_eq] = 1.0e3

    def factorize(rho_value: float):
        rho_vector = rho_value * eq_scale
        block = csr_matrix(
            (np.reciprocal(rho_vector), (np.arange(m), np.arange(m))), shape=(m, m)
        )
        kkt = vstack(
            [
                hstack([eye(n, format="csr") * (1.0 + sigma), a_t], format="csr"),
                hstack([a, -block], format="csr"),
            ],
            format="csc",
        )
        return splu(csc_matrix(kkt)), rho_vector

    lu, rho_vec = factorize(rho)
    n_factorizations = 1
    q = -target
    low = np.concatenate([np.zeros(m_eq), lb])
    high = np.concatenate([np.zeros(m_eq), ub])
    x = np.clip(target, lb, ub)
    z = a @ x
    y = np.zeros(m)
    r_prim = r_dual = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        rhs = np.concatenate([sigma * x - q, z - y / rho_vec])
        sol = lu.solve(rhs)
        x_tilde = sol[:n]
        nu = sol[n:]
        z_tilde = z + (nu - y) / rho_vec
        x = relax * x_tilde + (1.0 - relax) * x
        z_relaxed = relax * z_tilde + (1.0 - relax) * z
        z_new = np.clip(z_relaxed + y / rho_vec, low, high)
        y = y + rho_vec * (z_relaxed - z_new)
        z = z_new
        if iteration % check_every and iteration != max_iter:
            continue
        ax = a @ x
        aty = a_t @ y
        r_prim = float(np.max(np.abs(ax - z)))
        r_dual = float(np.max(np.abs(x + q + aty)))
        prim_scale = max(float(np.max(np.abs(ax))), float(np.max(np.abs(z))), 1e-12)
        dual_scale = max(
            float(np.max(np.abs(x))),
            float(np.max(np.abs(aty))),
            float(np.max(np.abs(q))),
            1e-12,
        )
        if (
            r_prim <= eps_abs + eps_rel * prim_scale
            and r_dual <= eps_abs + eps_rel * dual_scale
        ):
            converged = True
            break
        if iteration % adapt_every == 0:
            ratio = (r_prim / prim_scale) / max(r_dual / dual_scale, 1e-30)
            new_rho = float(np.clip(rho * np.sqrt(ratio), 1e-6, 1e6))
            if new_rho / rho > 5.0 or rho / new_rho > 5.0:
                rho = new_rho
                lu, rho_vec = factorize(rho)
                n_factorizations += 1
    x = np.clip(x, lb, ub)
    return x, {
        "iterations": float(iteration),
        "converged": bool(converged),
        "final_rho": float(rho),
        "n_factorizations": float(n_factorizations),
        "primal_residual_inf": r_prim,
        "dual_residual_inf": r_dual,
        "balance_residual_inf": float(np.max(np.abs(s @ x))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=int, default=30)
    parser.add_argument("--table", default=TABLE_PATH)
    parser.add_argument("--medium", default="YPD")
    parser.add_argument("--out", default="probe_moma_reference.json")
    args = parser.parse_args()

    started = time.time()
    table = np.load(args.table, allow_pickle=True)
    keys = list(table["keys"])
    fluxes = table["v"]
    provenance = json.loads(str(table["provenance"]))
    rxn_ids = list(table["rxn_ids"])
    wild_row = keys.index("")
    v_wt = fluxes[wild_row].astype(float)
    wt_norm = float(np.linalg.norm(v_wt))

    moved = []
    for row, key in enumerate(keys):
        if row == wild_row or not key:
            continue
        rel = float(np.linalg.norm(fluxes[row].astype(float) - v_wt) / max(wt_norm, 1e-12))
        if rel > MOVED_THRESHOLD:
            moved.append((key, row, rel))
    moved.sort(key=lambda entry: entry[0])
    print(
        f"table {osp.basename(args.table)}: {len(keys)} sets, {len(moved)} whose pFBA "
        f"reference moved more than {MOVED_THRESHOLD:.0%}; medium "
        f"{provenance['medium']}, wild-type growth {provenance['wild_type_growth']:.4f}"
    )
    # A DETERMINISTIC SUBSET rather than the first N by table order, which is the dataset's
    # own record order and carries whatever the LMDB happened to hold first.
    rng = np.random.default_rng(0)
    take = moved if len(moved) <= args.genes else [
        moved[i] for i in sorted(rng.choice(len(moved), size=args.genes, replace=False))
    ]

    source = YeastGEM()
    model = source.model
    # THE SAME FUNCTION THE CELL AND THE TABLE BUILDER USE, so the three vectors compared
    # here live in one program rather than in three boundaries that happen to share a name.
    media = media_report(model, None if args.medium == "none" else args.medium)
    assert media.get("applied") == (args.medium != "none")
    gem = build_gem_tensors(model, source.model_dir)
    assert list(gem.rxn_ids) == rxn_ids, (
        "the GEM's reaction order differs from the table's, so a comparison by position "
        "would compare different reactions."
    )
    s = csr_matrix(gem.s.to_dense().double().numpy())
    lb_wt = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    ub_wt = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    gem_genes = {g.id for g in model.genes}

    # THE SOLVER CHECK, and it is a refusal rather than a warning. On the unperturbed bounds
    # the program's own optimum IS the wild-type flux, so a round trip that does not return
    # it means every distance below is a solver artifact.
    round_trip, round_stats = solve_qp(s, lb_wt, ub_wt, v_wt)
    round_rel = float(np.linalg.norm(round_trip - v_wt) / max(wt_norm, 1e-12))
    print(f"wild-type round trip: {round_rel:.3e}; {json.dumps(round_stats)}")
    if round_rel > 1.0e-3:
        raise SystemExit(
            f"the QP does not reproduce the wild-type flux on the wild-type bounds "
            f"({round_rel:.3e}); every distance below would be a solver artifact."
        )

    rows: list[dict[str, object]] = []
    for key, row, pfba_rel in take:
        names = [g for g in key.split("+") if g]
        inside = [g for g in names if g in gem_genes]
        with model:
            knock_out_model_genes(model, inside)
            lb = (
                torch.tensor([r.lower_bound for r in model.reactions])
                .clamp(-FLUX_SCALE, FLUX_SCALE)
                .double()
                .numpy()
            )
            ub = (
                torch.tensor([r.upper_bound for r in model.reactions])
                .clamp(-FLUX_SCALE, FLUX_SCALE)
                .double()
                .numpy()
            )
        v_ko = fluxes[row].astype(float)
        v_moma, stats = solve_qp(s, lb, ub, v_wt)
        entry = {
            "gene": key,
            "n_genes_in_gem": len(inside),
            "moma_to_pfba_knockout_rel": float(
                np.linalg.norm(v_moma - v_ko) / max(wt_norm, 1e-12)
            ),
            "moma_to_wild_type_rel": float(
                np.linalg.norm(v_moma - v_wt) / max(wt_norm, 1e-12)
            ),
            "pfba_knockout_to_wild_type_rel": pfba_rel,
            "moma_biomass": float(v_moma[int(gem.biomass_index)]),
            "pfba_knockout_biomass": float(v_ko[int(gem.biomass_index)]),
            "wild_type_biomass": float(v_wt[int(gem.biomass_index)]),
            "solver": stats,
        }
        rows.append(entry)
        print(
            f"{key:>10}  moma-ko {entry['moma_to_pfba_knockout_rel']:.4f}  "
            f"moma-wt {entry['moma_to_wild_type_rel']:.4f}  "
            f"ko-wt {entry['pfba_knockout_to_wild_type_rel']:.4f}  "
            f"growth {entry['moma_biomass']:.4f}/{entry['pfba_knockout_biomass']:.4f}  "
            f"iters {stats['iterations']:.0f} rp {stats['primal_residual_inf']:.1e}"
        )

    # CONVERGENCE IS REPORTED PER GENE AND SUMMARIZED, never assumed. A solve that stops at
    # the iteration cap has a residual of the same order as the distances being measured,
    # so its three numbers are solver noise and the summary below says how many there are.
    converged = [r for r in rows if r["solver"]["converged"]]
    if len(converged) != len(rows):
        print(
            f"WARNING: {len(rows) - len(converged)} of {len(rows)} solves hit the iteration "
            "cap; their distances are NOT a result. The summary is over the converged ones."
        )
    rows_all = rows
    rows = converged

    def summary(field: str) -> dict[str, float]:
        values = np.array([float(r[field]) for r in rows])
        return {
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "n_above_1pct": int((values > 0.01).sum()),
        }

    payload = {
        "question": (
            "is MOMA a DIFFERENT reference from the pFBA knockout solution, or the same "
            "one? Informative only if moma_to_pfba_knockout_rel is large while "
            "moma_to_wild_type_rel is not zero; the first small means a MOMA table would "
            "train as the pFBA table did, the second zero means it would train as the "
            "single-reference head did."
        ),
        "definitions": {
            "moma_to_pfba_knockout_rel": (
                "||v_moma - v_pfba_ko|| / ||v_wt||, where v_pfba_ko is the row this "
                "genotype already has in the shipped reference table."
            ),
            "moma_to_wild_type_rel": "||v_moma - v_wt|| / ||v_wt||",
            "pfba_knockout_to_wild_type_rel": (
                "||v_pfba_ko - v_wt|| / ||v_wt||, the quantity the table's "
                "'moved more than 1 percent' count is thresholded on."
            ),
        },
        "program": (
            "min 0.5 ||v - v_wt||^2 s.t. S v = 0 and the knockout's bounds, clamped to the "
            "layer's flux scale. Solved by the OSQP-style ADMM in this file, since no QP "
            "package is installed (optlang carries GLPK and SCIPY, both LP only)."
        ),
        "table": args.table,
        "table_provenance": provenance,
        "medium": args.medium,
        "media_report": {k: v for k, v in media.items() if k != "unresolved"},
        "n_sets_moved": len(moved),
        "moved_threshold": MOVED_THRESHOLD,
        "n_genes_solved": len(rows_all),
        "n_genes_converged": len(rows),
        "n_genes_measured": len(rows),
        "gene_selection": "seeded uniform sample (numpy default_rng(0)) of the moved sets",
        "wild_type_round_trip_rel": round_rel,
        "wild_type_round_trip_solver": round_stats,
        "summary": {
            field: summary(field)
            for field in (
                "moma_to_pfba_knockout_rel",
                "moma_to_wild_type_rel",
                "pfba_knockout_to_wild_type_rel",
            )
        },
        "genes": rows_all,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

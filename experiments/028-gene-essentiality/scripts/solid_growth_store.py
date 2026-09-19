# experiments/028-gene-essentiality/scripts/solid_growth_store.py
# [[experiments.028-gene-essentiality.scripts.solid_growth_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/solid_growth_store

"""The module's store over the solid-growth build: one solve per (GEM gene subset, medium).

The scalable module form (rounds 27 to 57) reads the deletion's solved flux vector
through the phenotype head's affine, solved once per distinct (genotype, medium) pair
offline. Over the 025 solid-growth build the distinct keys are the GEM gene subsets the
genotypes delete, by medium (``store_key_count.py``); a genotype deleting no GEM gene
reads the wild-type row. This solves every key's program and writes the store.

The program is the round-27 one, the GEM box clamped to the layer's flux scale, the
medium on the exchanges, the key's genes closed through the GPR
(``cobra.manipulation.knock_out_model_genes`` over the whole subset, so a reaction
whose rule is A or B closes only when both are deleted), the split form in micromoles,
maximize growth, solved by HiGHS 1.15's interior point without crossover at KKT 1e-4
(``--solver ipm``, the store the cell trains over to the pFBA store's number, round
36) or the dual simplex (``--solver simplex``). A program the solver does not certify
keeps the zero flux and is counted, as in the tolerance reads.

The SGA selection media of the build are SD-MSG with amino-acid dropouts and drugs;
no FBA recipe carries them, so the store is solved on synthetic complete medium
(``--medium SC``) for every key, the approximation round 42 measured on essentiality
(the SC store and the YPD store train to one number); the record's own medium name is
kept beside each row so a later recipe can replace the rows it covers.

Keys are solved in the order of records served (``store_keys.json.gz`` is sorted so),
and ``--limit N`` builds a partial store that covers the most records first. Writes
``$DATA_ROOT/data/torchcell/reference_flux/solid_growth_store_<medium>_<solver>.npz``
with ``keys`` (sorted GEM genes joined by ``+``, the empty string for the wild type),
``media``, ``n_records``, ``v_deletion``, ``growth``, ``optimal``, ``v_wild_type``,
``rxn_ids`` and ``provenance``, and ``results/solid_growth_store_<medium>_<solver>.json``.
Needs ``highspy`` 1.15 on the path (``TC_HIGHSPY_PATH`` in the launcher).
"""

from __future__ import annotations

import argparse
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
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
SCRIPTS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
sys.path.insert(0, SCRIPTS_DIR)
import solve_tolerance_read as base  # noqa: E402
from ess_reference_flux_table import FLUX_SCALE, SOLVER_KEYS, growth_solve  # noqa: E402
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

_W: dict[str, Any] = {}


def _init_worker(a_csc: sp.csc_matrix, c: np.ndarray, medium: str, key: str) -> None:
    """Each worker holds the program's matrix and its own GEM for the knockouts."""
    from cobra.manipulation import knock_out_model_genes  # noqa: F401

    base._init_worker(a_csc, c)
    source = YeastGEM()
    model = source.model
    media_report(model, medium)
    gem = build_gem_tensors(model, source.model_dir)
    _W["model"] = model
    _W["biomass"] = int(gem.biomass_index)
    _W["key"] = key


def _solve_key(genes: list[str]) -> tuple[np.ndarray, float, bool]:
    from cobra.manipulation import knock_out_model_genes

    model = _W["model"]
    with model:
        if genes:
            knock_out_model_genes(model, genes)
        lb = np.clip(
            np.array([rx.lower_bound for rx in model.reactions], dtype=np.float64),
            -FLUX_SCALE,
            FLUX_SCALE,
        )
        ub = np.clip(
            np.array([rx.upper_bound for rx in model.reactions], dtype=np.float64),
            -FLUX_SCALE,
            FLUX_SCALE,
        )
    solve = growth_solve(lb, ub, _W["key"], _W["biomass"])
    return solve.v.astype(np.float32), solve.growth, solve.optimal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium", default="SC")
    parser.add_argument("--solver", default="ipm", choices=["ipm", "simplex"])
    parser.add_argument("--keys", default=osp.join(RESULTS_DIR, "store_keys.json.gz"))
    parser.add_argument("--limit", type=int, default=0, help="Build the first N keys only.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("TC_LP_WORKERS", "20")))
    args = parser.parse_args()
    import highspy

    started = time.time()
    with gzip.open(args.keys, "rt") as fh:
        keys = json.load(fh)
    if args.limit:
        keys = keys[: args.limit]
    genes_per_key = [k["gem_genes"] for k in keys]
    print(f"{len(keys)} keys ({sum(k['n_records'] for k in keys)} records served)")

    source = YeastGEM()
    model = source.model
    media = media_report(model, args.medium)
    gem = build_gem_tensors(model, source.model_dir)
    r = len(gem.rxn_ids)
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    a_csc = sp.hstack([sp.csr_matrix(s_dense), -sp.csr_matrix(s_dense)]).tocsc()
    c = np.zeros(2 * r)
    c[gem.biomass_index] = 1.0
    c[r + gem.biomass_index] = -1.0
    solver_key = SOLVER_KEYS[args.solver]
    assert solver_key is not None

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(a_csc, c, args.medium, solver_key)
    ) as pool:
        wild_v, wild_growth, wild_ok = pool.apply(_solve_key, ([],))
        assert wild_ok
        print(f"wild-type growth {wild_growth:.4f}; solving {len(keys)} keys")
        v = np.zeros((len(keys), r), dtype=np.float32)
        growth = np.zeros(len(keys), dtype=np.float32)
        optimal = np.zeros(len(keys), dtype=bool)
        for i, (vi, gi, oki) in enumerate(pool.imap(_solve_key, genes_per_key, chunksize=64)):
            v[i], growth[i], optimal[i] = vi, gi, oki
            if (i + 1) % 20000 == 0:
                print(f"{i + 1} solved, {time.time() - started:.0f}s")
    print(f"solved, {time.time() - started:.0f}s; {int((~optimal).sum())} uncertified")

    provenance = {
        "gem": osp.basename(source.model_dir),
        "flux_scale": FLUX_SCALE,
        "method": (
            "growth-maximizing LP (the round-27 program, split form in micromoles) by "
            f"HiGHS {highspy.Highs().version()} {solver_key}, every GEM gene of the key "
            "deleted through cobra.manipulation.knock_out_model_genes; written by "
            "solid_growth_store.py"
        ),
        "solver": args.solver,
        "medium": args.medium,
        "medium_report": {k: val for k, val in media.items() if k != "unresolved"},
        "keys_file": args.keys,
        "n_keys": len(keys),
        "n_records_served": int(sum(k["n_records"] for k in keys)),
        "wild_type_growth": float(wild_growth),
        "n_uncertified_zero_flux": int((~optimal).sum()),
        "n_lethal": int((optimal & (growth < 1e-6)).sum()),
        "medium_note": (
            "the build's SGA selection media (SD-MSG with dropouts and drugs) have no FBA "
            "recipe; every key is solved on the --medium recipe and its own medium name is "
            "kept in `media`"
        ),
    }
    os.makedirs(TABLE_DIR, exist_ok=True)
    stem = f"solid_growth_store_{args.medium}_{args.solver}" + (
        f"_first{args.limit}" if args.limit else ""
    )
    path = osp.join(TABLE_DIR, f"{stem}.npz")
    np.savez(
        path,
        keys=np.array(["+".join(k["gem_genes"]) for k in keys]),
        media=np.array([k["medium"] for k in keys]),
        n_records=np.array([k["n_records"] for k in keys], dtype=np.int64),
        v_deletion=v,
        growth=growth,
        optimal=optimal,
        v_wild_type=wild_v.astype(np.float32),
        rxn_ids=np.array(list(gem.rxn_ids)),
        provenance=np.array(json.dumps(provenance)),
    )
    out = {"path": path, **provenance, "seconds": time.time() - started}
    with open(osp.join(RESULTS_DIR, f"{stem}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: val for k, val in out.items() if k != "medium_report"}, indent=1))


if __name__ == "__main__":
    main()

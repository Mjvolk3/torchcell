# experiments/028-gene-essentiality/scripts/ess_sampled_flux_store.py
# [[experiments.028-gene-essentiality.scripts.ess_sampled_flux_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/ess_sampled_flux_store

r"""A store sampled AROUND the deletion's optimum, instead of one point at it.

Every store in this document is a single point: the parsimonious solution, another
solver's vertex, an interior point, a first-order point. Those differ (the argmax is
degenerate, Section \texttt{sec:solve-tolerance}), but they are all one flux per
genotype. Flux Cone Learning instead samples the cone, 124 points per deletion with
artificial-centering hit-and-run at thinning 5,000, and scores a gene by the mean over
per-sample model calls; its own figure puts the gain from depth at one to two points
(1 sample per cone 0.927 to 0.950, 100 per cone 0.940 to 0.970 on E. coli). Their sampled
matrix is not in our mirror, which holds their splits, hyperparameter tables and figure
CSVs only, so this samples ourselves.

Per labeled gene: knock the gene out, solve, fix the growth reaction's lower bound at
``--growth-fraction`` of that deletion's own optimum, and draw ``--samples`` points with
``OptGPSampler`` at ``--thinning``. A lethal deletion (no growth) is stored as a zero row
and counted, exactly as the point stores treat it. Two stores are written in the solve
store's shape (``genes``, ``v_deletion``, ``v_wild_type``, ``rxn_ids``, ``provenance``),
both read through ``ESS_SOLVE_TABLE``:

* ``ess_solve_encodings_YPD_sampledmean.npz``: the mean over the draws, a like-for-like
  replacement for the parsimonious vector;
* ``ess_solve_encodings_YPD_sampledspread.npz``: that mean with the per-reaction standard
  deviation over the draws appended, which is the part a single point cannot carry.

MEASURED before writing this: five draws of one deletion at thinning 10 take 147 seconds
in one process, so the labeled genes cost about 45 process-hours; at their thinning it
would be far more. The defaults here are therefore shallow on purpose, and the question
is whether a shallow sample beats the point, not whether depth saturates.

    sbatch experiments/028-gene-essentiality/scripts/gh_sampled_flux_store.slurm
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import os.path as osp
import time
from multiprocessing import Pool

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PFBA_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD.npz")
SPLIT = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")

_MODEL = None
_ARGS: argparse.Namespace | None = None


def build_model():
    """The medium-applied model the point stores were built on, once per worker."""
    from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds
    from torchcell.metabolism.yeast_GEM import YeastGEM

    model = YeastGEM().model
    bounds = media_to_bounds(FBA_MEDIA["YPD"], model)
    assert not bounds.unresolved_names, bounds.unresolved_names
    bounds.apply(model)
    return model


def _init(args: argparse.Namespace) -> None:
    global _MODEL, _ARGS
    _ARGS = args
    _MODEL = build_model()


def sample_one(gene: str) -> tuple[str, np.ndarray, np.ndarray, float, int]:
    """``(gene, mean, sd, growth, n_draws)``; a lethal deletion returns zero rows."""
    from cobra.manipulation import knock_out_model_genes
    from cobra.sampling import OptGPSampler

    assert _MODEL is not None and _ARGS is not None
    n_rxn = len(_MODEL.reactions)
    zeros = np.zeros(n_rxn, dtype=np.float32)
    with _MODEL:
        if gene in {g.id for g in _MODEL.genes}:
            knock_out_model_genes(_MODEL, [gene])
        solution = _MODEL.optimize()
        growth = float(solution.objective_value or 0.0)
        if growth < _ARGS.lethal_growth:
            return gene, zeros, zeros, growth, 0
        objective = [r for r in _MODEL.reactions if r.objective_coefficient][0]
        objective.lower_bound = _ARGS.growth_fraction * growth
        draws = OptGPSampler(_MODEL, thinning=_ARGS.thinning, processes=1).sample(
            _ARGS.samples
        )
        values = np.asarray(draws, dtype=np.float32)
    return (
        gene,
        values.mean(axis=0),
        values.std(axis=0, ddof=1) if len(values) > 1 else zeros,
        growth,
        len(values),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--growth-fraction", type=float, default=0.99)
    parser.add_argument("--lethal-growth", type=float, default=1e-9)
    parser.add_argument("--processes", type=int, default=16)
    args = parser.parse_args()

    pfba = np.load(PFBA_STORE, allow_pickle=False)
    genes = [str(g) for g in pfba["genes"]]
    rxn_ids = [str(r) for r in pfba["rxn_ids"]]
    with gzip.open(SPLIT, "rt") as fh:
        assert set(json.load(fh)["labels"]) >= set(genes)

    started = time.time()
    with Pool(args.processes, initializer=_init, initargs=(args,)) as pool:
        rows = pool.map(sample_one, genes, chunksize=4)
    order = {gene: i for i, gene in enumerate(genes)}
    mean = np.zeros((len(genes), len(rxn_ids)), dtype=np.float32)
    sd = np.zeros_like(mean)
    growth = np.zeros(len(genes), dtype=np.float32)
    n_lethal = 0
    for gene, m, s, g, n in rows:
        mean[order[gene]] = m
        sd[order[gene]] = s
        growth[order[gene]] = g
        n_lethal += int(n == 0)
    seconds = time.time() - started

    common = {
        "method": (
            f"{args.samples} OptGPSampler draws at thinning {args.thinning} from the "
            f"deletion's own near-optimal space (growth fixed at "
            f"{args.growth_fraction} of its optimum) on YPD; written by "
            "ess_sampled_flux_store.py"
        ),
        "source_store": PFBA_STORE,
        "samples": args.samples,
        "thinning": args.thinning,
        "growth_fraction": args.growth_fraction,
        "n_lethal_zero_rows": n_lethal,
    }
    written = {}
    for suffix, block, names in (
        ("sampledmean", mean, rxn_ids),
        (
            "sampledspread",
            np.concatenate([mean, sd], axis=1),
            rxn_ids + [f"sd_{r}" for r in rxn_ids],
        ),
    ):
        path = osp.join(TABLE_DIR, f"ess_solve_encodings_YPD_{suffix}.npz")
        np.savez_compressed(
            path,
            genes=np.array(genes),
            v_deletion=block.astype(np.float32),
            v_wild_type=np.zeros(block.shape[1], dtype=np.float32),
            rxn_ids=np.array(names),
            provenance=json.dumps({**common, "columns": len(names)}),
        )
        written[suffix] = {"path": path, "shape": list(block.shape)}
        print(f"{path}: {block.shape}", flush=True)

    point = pfba["v_deletion"].astype(np.float32)
    viable = growth >= args.lethal_growth
    report = {
        **common,
        "n_genes": len(genes),
        "n_viable": int(viable.sum()),
        "seconds": seconds,
        "processes": args.processes,
        "stores": written,
        # How far the sampled mean sits from the parsimonious point, in the head's
        # encoding, over the viable deletions.
        "median_rel_l2_mean_vs_pfba": float(
            np.median(
                np.linalg.norm(
                    np.log1p(np.abs(mean[viable])) - np.log1p(np.abs(point[viable])),
                    axis=1,
                )
                / np.maximum(
                    np.linalg.norm(np.log1p(np.abs(point[viable])), axis=1), 1e-12
                )
            )
        ),
        "mean_frac_reactions_with_nonzero_sd": float(
            np.mean((sd[viable] > 1e-9).mean(axis=1))
        ),
    }
    out = osp.join(RESULTS, "ess_sampled_flux_store.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"{len(genes)} genes, {report['n_viable']} viable, {n_lethal} lethal, "
        f"{seconds / 60:.1f} min on {args.processes} processes; median relative distance "
        f"to the pFBA point {report['median_rel_l2_mean_vs_pfba']:.3f}",
        flush=True,
    )
    print(out)


if __name__ == "__main__":
    main()

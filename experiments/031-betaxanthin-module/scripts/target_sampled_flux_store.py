# experiments/031-betaxanthin-module/scripts/target_sampled_flux_store.py
# [[experiments.031-betaxanthin-module.scripts.target_sampled_flux_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_sampled_flux_store

r"""A flux store sampled from each deletion's near-optimal cone, on the target models.

Flux Cone Learning represents a deletion by samples of its feasible flux cone rather than
by one optimum, and scores a gene by the mean over per-sample model calls. The
essentiality study's sampler (``ess_sampled_flux_store.py``) never finished on the labeled
genes (job 2623 timed out), so the sampled representation was never measured anywhere.
This script is that sampler on the two TARGET models over the source store's gene list:
the Cachera strain (cassette, BY4741 markers deleted, YPD) for betaxanthin and the
prototroph on synthetic minimal medium for the amino acids. Per gene: knock the gene out,
maximize growth, fix the growth reaction's lower bound at ``--growth-fraction`` of that
optimum, and draw ``--samples`` points with cobra's ``OptGPSampler`` at ``--thinning``.
A lethal deletion (no growth) is the zero row, as every store has it. Two stores in the
solve store's shape, both read through ``ESS_SOLVE_TABLE``:

* ``ess_solve_encodings_<stem>_sampledmean.npz``: the mean over draws, a like-for-like
  replacement for the parsimonious vector;
* ``ess_solve_encodings_<stem>_sampledspread.npz``: that mean with the per-reaction
  standard deviation over draws appended, the part one point cannot carry.

MEASURED before writing the essentiality version: five draws of one deletion at thinning
10 took 147 s in one process, so the defaults here are shallow on purpose and the question
is whether a shallow sample beats the point, not whether depth saturates. Run from the
repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_sampled_flux_store.py \
        --target betaxanthin --samples 6 --thinning 10 --processes 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
for study in ("028-gene-essentiality", "027-betaxanthin-metabolic", "026-metabolism-flux"):
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, study, "scripts"))
from bx_reference_flux_table import build_strain_model  # noqa: E402
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
)
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.flux_layer import genotype_key  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

TARGETS: dict[str, dict[str, str]] = {
    "betaxanthin": {"source_stem": "YPD_bx", "medium": "YPD", "study": "031-betaxanthin-module"},
    "amino_acids": {"source_stem": "SM_gem", "medium": "SM", "study": "032-amino-acid-module"},
}


class SampledStoreProvenance(StoreProvenance):
    """The store record plus the sampler's settings."""

    target: str
    source_store: str
    samples: int
    thinning: int
    sampler_growth_fraction: float
    n_draws_wild_type: int
    column_layout: str


_MODEL: Any = None
_ARGS: argparse.Namespace | None = None
_MEDIA: dict[str, Any] | None = None


def build_target_model(target: str) -> tuple[Any, dict[str, Any]]:
    """The medium-applied target model with the store's bound clamp."""
    if target == "betaxanthin":
        model, _demands, info = build_strain_model()
        media = info["media"]
    else:
        model = YeastGEM().model
        media = media_report(model, TARGETS[target]["medium"])
    for r in model.reactions:
        r.lower_bound = max(r.lower_bound, -FLUX_SCALE)
        r.upper_bound = min(r.upper_bound, FLUX_SCALE)
    return model, media


def _init(args: argparse.Namespace) -> None:
    global _MODEL, _ARGS
    _ARGS = args
    _MODEL, _ = build_target_model(args.target)
    # OptGPSampler copies the model, and optlang re-optimizes the copy from a cold
    # standard basis; on the prototroph three of 1,161 deletions cycled in glp_simplex
    # there for two hours with no return (2026-09-23, py-spy). Presolve is cloned into
    # every copy's configuration and avoids the cycling; a bad status then raises. It
    # is switched off again on the copy before the flux-variability warm-up, whose
    # thousands of LPs per deletion need warm starts (under presolve each is a cold
    # solve and one deletion took over an hour, 2026-09-24).
    _MODEL.solver.configuration.presolve = True
    from cobra.sampling.hr_sampler import HRSampler

    original_warmup = HRSampler.generate_fva_warmup

    def warm_started_warmup(self: Any) -> None:
        self.model.solver.configuration.presolve = False
        original_warmup(self)

    HRSampler.generate_fva_warmup = warm_started_warmup  # type: ignore[method-assign]


def sample_model(model: Any, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, int]:
    """``(mean, sd, growth, n_draws)`` of ``model`` as bounded now; lethal is zeros."""
    from cobra.sampling import OptGPSampler

    n_rxn = len(model.reactions)
    zeros = np.zeros(n_rxn, dtype=np.float32)
    solution = model.optimize()
    growth = float(solution.objective_value or 0.0)
    if solution.status != "optimal" or growth < args.lethal_growth:
        return zeros, zeros, growth, 0
    objective = [r for r in model.reactions if r.objective_coefficient][0]
    objective.lower_bound = args.growth_fraction * growth
    draws = OptGPSampler(model, thinning=args.thinning, processes=1, seed=args.seed).sample(
        args.samples
    )
    values = np.asarray(draws, dtype=np.float32)
    return (
        values.mean(axis=0),
        values.std(axis=0, ddof=1) if len(values) > 1 else zeros,
        growth,
        len(values),
    )


def sample_one(gene: str) -> tuple[str, np.ndarray, np.ndarray, float, int]:
    from cobra.manipulation import knock_out_model_genes

    assert _MODEL is not None and _ARGS is not None
    with _MODEL:
        knock_out_model_genes(_MODEL, [gene])
        mean, sd, growth, n = sample_model(_MODEL, _ARGS)
    return gene, mean, sd, growth, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--growth-fraction", type=float, default=0.99)
    parser.add_argument("--lethal-growth", type=float, default=1e-6)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument(
        "--genes", default="", help="Comma-separated subset, in store order; a test."
    )
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default="")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for every store.")
    spec = TARGETS[args.target]
    results_dir = args.results_dir or osp.join(EXPERIMENT_ROOT, spec["study"], "results")
    started = time.time()

    source_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{spec['source_stem']}.npz")
    source = np.load(source_path, allow_pickle=False)
    source_provenance = json.loads(str(source["provenance"]))
    genes = [str(g) for g in source["genes"]]
    if args.limit_genes:
        genes = genes[: args.limit_genes]
    if args.genes:
        wanted = set(args.genes.split(","))
        genes = [g for g in genes if g in wanted]
        assert len(genes) == len(wanted), sorted(wanted - set(genes))

    model, media = build_target_model(args.target)
    rxn_ids = [r.id for r in model.reactions]
    n_rxn = len(rxn_ids)
    assert rxn_ids == [str(r) for r in source["rxn_ids"]][:n_rxn], (
        "the model's reaction order differs from the source store's"
    )
    assert all(g in {x.id for x in model.genes} for g in genes)
    with model:
        wt_mean, wt_sd, wt_growth, wt_n = sample_model(model, args)
    assert wt_n == args.samples, "the wild type did not sample"
    print(
        f"{args.target}: {n_rxn} reactions, {len(genes)} genes; wild-type growth "
        f"{wt_growth:.4f}, {wt_n} draws in {time.time() - started:.0f}s",
        flush=True,
    )

    with Pool(args.processes, initializer=_init, initargs=(args,)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(sample_one, genes, chunksize=2), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f"  {i}/{len(genes)} genes, {time.time() - started:.0f}s", flush=True)
    order = {g: i for i, g in enumerate(genes)}
    mean = np.zeros((len(genes), n_rxn), dtype=np.float32)
    sd = np.zeros_like(mean)
    growth = np.zeros(len(genes))
    n_draws = np.zeros(len(genes), dtype=int)
    for gene, m, s, g, n in rows:
        mean[order[gene]], sd[order[gene]], growth[order[gene]], n_draws[order[gene]] = m, s, g, n
    lethal = n_draws == 0
    seconds = time.time() - started

    source_v = source["v_deletion"].astype(np.float32)[: len(genes), :n_rxn]
    viable = ~lethal
    rel = np.linalg.norm(
        np.log1p(np.abs(mean[viable])) - np.log1p(np.abs(source_v[viable])), axis=1
    ) / np.maximum(np.linalg.norm(np.log1p(np.abs(source_v[viable])), axis=1), 1e-12)

    def provenance(layout: str, block: np.ndarray) -> SampledStoreProvenance:
        return SampledStoreProvenance(
            gem=source_provenance["gem"],
            growth_fraction=1.0,
            flux_scale=FLUX_SCALE,
            method=(
                f"{args.samples} OptGPSampler draws at thinning {args.thinning} from the "
                f"deletion's own near-optimal space (growth fixed at {args.growth_fraction} "
                "of its optimum) on the target model over the source store's genes; gene "
                "deleted through cobra.manipulation.knock_out_model_genes; written by "
                "target_sampled_flux_store.py"
            ),
            solver="cobra OptGPSampler (glpk)",
            n_nonoptimal_zero_flux=0,
            medium=spec["medium"],
            media_report=MediaApplied.model_validate(media),
            split_file=source_provenance["split_file"],
            wild_type_growth=wt_growth,
            wild_type_key=genotype_key([]),
            n_sets=len(genes) + 1,
            n_lethal_zero_flux=int(lethal.sum()),
            n_genes_outside_the_gem=0,
            background_genes=list(source_provenance["background_genes"]),
            python_hash_seed=os.environ.get("PYTHONHASHSEED"),
            flux_sha256=hashlib.sha256(block.tobytes()).hexdigest(),
            target=args.target,
            source_store=source_path,
            samples=args.samples,
            thinning=args.thinning,
            sampler_growth_fraction=args.growth_fraction,
            n_draws_wild_type=wt_n,
            column_layout=layout,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    written = {}
    for suffix, block, wild, names, layout in (
        ("sampledmean", mean, wt_mean, rxn_ids, "mean over draws"),
        (
            "sampledspread",
            np.concatenate([mean, sd], axis=1),
            np.concatenate([wt_mean, wt_sd]),
            rxn_ids + [f"sd_{r}" for r in rxn_ids],
            "mean over draws | sd over draws",
        ),
    ):
        path = osp.join(args.out_dir, f"ess_solve_encodings_{spec['source_stem']}_{suffix}.npz")
        assert not osp.exists(path), f"{path} exists; stores are never overwritten"
        np.savez_compressed(
            path,
            genes=np.array(genes),
            v_deletion=block.astype(np.float32),
            v_wild_type=wild.astype(np.float32),
            rxn_ids=np.array(names),
            provenance=np.array(provenance(layout, block).model_dump_json()),
            growth_ratio=(growth / wt_growth).astype(np.float32),
        )
        written[suffix] = {"path": path, "shape": list(block.shape)}
        print(f"{path}: {block.shape}", flush=True)

    report = {
        "target": args.target,
        "source_store": source_path,
        "samples": args.samples,
        "thinning": args.thinning,
        "growth_fraction": args.growth_fraction,
        "seed": args.seed,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_lethal_source": int(
            (source["gene_flux_stats"][: len(genes), list(source["stat_names"]).index("deletion_growth_ratio")] < 1e-6).sum()
        ),
        "wild_type_growth": wt_growth,
        "median_rel_l2_log1p_mean_vs_source": float(np.median(rel)),
        "mean_frac_reactions_with_nonzero_sd": float(np.mean((sd[viable] > 1e-9).mean(axis=1))),
        "wild_type_frac_reactions_with_nonzero_sd": float((wt_sd > 1e-9).mean()),
        "seconds": seconds,
        "processes": args.processes,
        "stores": written,
        "rows": [
            {"gene": g, "growth": float(growth[i]), "growth_ratio": float(growth[i] / wt_growth),
             "n_draws": int(n_draws[i])}
            for i, g in enumerate(genes)
        ],
    }
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"target_sampled_flux_store_{args.target}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        f"{len(genes)} genes, {int(lethal.sum())} lethal, {seconds / 60:.1f} min on "
        f"{args.processes} processes; median relative distance of the sampled mean to the "
        f"source point {report['median_rel_l2_log1p_mean_vs_source']:.3f}",
        flush=True,
    )
    print(out)


if __name__ == "__main__":
    main()

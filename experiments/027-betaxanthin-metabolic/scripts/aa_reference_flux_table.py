# experiments/027-betaxanthin-metabolic/scripts/aa_reference_flux_table.py
# [[experiments.027-betaxanthin-metabolic.scripts.aa_reference_flux_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/aa_reference_flux_table

r"""The layer-free module's store for the amino-acid screen: growth and 19 syntheses per gene.

Mulleder 2016 measured the intracellular concentrations of 19 amino acids in the
prototrophic BY4741 deletion collection grown on synthetic minimal medium. The 028 store
built for that medium (``ess_reference_flux_table.py --genes gem --medium SM``) carries
the growth vector alone, which is the strain's program under no product demand. This
store adds what the target is about: for each amino acid, how much of it the knockout can
synthesize while still growing.

THE STRAIN, AS THIS STORE REPRESENTS IT.

* yeast-GEM 9.0.2 with no cassette and no marker deletions: the collection is
  prototrophic (pHLUM restores HIS3, LEU2, URA3 and MET15), so the wild type is the
  unmodified model;
* the medium is the dataset's synthetic minimal medium (``FBA_MEDIA["SM"]``: YNB vitamins,
  ammonium sulfate, D-glucose);
* the 19 amino acids are the cytosolic metabolites of 026's
  ``AMINO_ACID_TO_GEM_METABOLITE``, each given a zero-lower-bound demand ``DM_aa_<key>``
  that forces no flux, so the growth optimum is unchanged (asserted).

PER GENE, one parsimonious growth solve (``cobra.flux_analysis.pfba``) and 19 linear
programs: with growth held at ``PRODUCTION_GROWTH_FRACTION`` of the knockout's own optimum,
maximize each amino acid's demand in turn. That is the ``aa_at_90pct_reference_growth``
reading of the constraint-based baseline, made relative to the knockout's own growth as
the betaxanthin store does. The 19 maxima are kept as summary columns, not as 19 flux
vectors, because 19 vectors of 4,131 columns each would put 78,000 columns under the
affine on a target the whole growth vector already does not carry.

The store is written in the shape the 028 trainer reads (``ESS_SOLVE_TABLE``): ``genes``,
``v_deletion`` ``[n, r]`` (the growth vector), ``v_wild_type``, ``rxn_ids``,
``gene_flux_stats`` with ``stat_names`` (growth, the growth ratio, the reactions knocked
out, the relative change of the vector, then ``aa_<key>_at_own_growth_floor`` and
``aa_<key>_ratio_to_wild_type`` for the 19 amino acids), and ``provenance``. Beside it a
CSV of the growth ratio per gene for the trainer's medium column (``ESS_FBA_CSV``). Run
from the worktree root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/aa_reference_flux_table.py \
        --processes 8
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
from cobra.flux_analysis import pfba
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
)
from train_flux import AMINO_ACID_TO_GEM_METABOLITE, media_report  # noqa: E402

from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

MEDIUM = "SM"
AA_KEYS = sorted(AMINO_ACID_TO_GEM_METABOLITE)
PRODUCTION_GROWTH_FRACTION = 0.9
LETHAL_GROWTH = 1e-6
STAT_NAMES = (
    [
        "growth",
        "deletion_growth_ratio",
        "n_reactions_knocked_out",
        "relative_change_from_wild_type",
    ]
    + [f"aa_{k}_at_own_growth_floor" for k in AA_KEYS]
    + [f"aa_{k}_ratio_to_wild_type" for k in AA_KEYS]
)


class AaStoreProvenance(StoreProvenance):
    """The essentiality store's record plus what makes this the amino-acid screen."""

    amino_acids: dict[str, str]
    demand_reactions: list[str]
    production_growth_fraction: float
    wild_type_aa_at_own_growth_floor: dict[str, float]
    n_reactions_total: int
    n_stat_columns: int
    stat_layout: str
    background_note: str


class GeneRow(BaseModel):
    """One gene's row of the store summary."""

    gene: str
    growth: float
    growth_ratio: float
    n_reactions_knocked_out: int
    relative_change_from_wild_type: float
    aa_ratio: dict[str, float]


_W: dict[str, Any] = {}


def build_strain_model() -> tuple[Any, dict[str, str], dict[str, Any]]:
    """yeast-GEM on the synthetic minimal medium with one zero-flux demand per amino acid."""
    model = YeastGEM().model
    # GLPK rebuilds its basis every solve under presolve. Without it the warm start
    # carried from one knockout's demand program to the next made the simplex cycle
    # without end on the COQ1 knockout's alanine program (job 2751, no progress in
    # thirty minutes), while the same program on a fresh copy solved in 0.3 s.
    model.solver.configuration.presolve = True
    media = media_report(model, MEDIUM)
    growth_before = float(model.slim_optimize())
    demands: dict[str, str] = {}
    for key in AA_KEYS:
        rxn = model.add_boundary(
            model.metabolites.get_by_id(AMINO_ACID_TO_GEM_METABOLITE[key]),
            type="demand",
            reaction_id=f"DM_aa_{key}",
        )
        demands[key] = rxn.id
    growth_after = float(model.slim_optimize())
    assert abs(growth_after - growth_before) < 1e-6, (growth_before, growth_after)
    info = {"media": media, "growth_unmodified": growth_before}
    return model, demands, info


def solve_gene(
    model: Any, demands: dict[str, str]
) -> tuple[np.ndarray, float, dict[str, float]]:
    """(growth vector, growth, each amino acid's maximum at the growth floor)."""
    n = len(model.reactions)
    growth_rxn = next(r for r in model.reactions if r.objective_coefficient != 0)
    zero = dict.fromkeys(demands, 0.0)
    try:
        grown = pfba(model)
    except Exception:
        # An infeasible growth program is a lethal knockout, the zero row, as in the
        # betaxanthin store.
        return np.zeros(n, np.float32), 0.0, zero
    mu = float(grown.fluxes[growth_rxn.id])
    if mu < LETHAL_GROWTH:
        return np.zeros(n, np.float32), 0.0, zero
    v_growth = grown.fluxes.to_numpy().astype(np.float32)
    maxima: dict[str, float] = {}
    with model:
        growth_rxn.lower_bound = PRODUCTION_GROWTH_FRACTION * mu
        for key, rid in demands.items():
            model.objective = {model.reactions.get_by_id(rid): 1.0}
            value = model.slim_optimize(error_value=float("nan"))
            if not np.isfinite(value):
                raise RuntimeError(
                    f"{key} demand program not optimal ({model.solver.status}) with "
                    f"growth floor {PRODUCTION_GROWTH_FRACTION * mu:.4f}"
                )
            maxima[key] = float(value)
    return v_growth, mu, maxima


def _init_worker() -> None:
    model, demands, _ = build_strain_model()
    for r in model.reactions:
        r.lower_bound = max(r.lower_bound, -FLUX_SCALE)
        r.upper_bound = min(r.upper_bound, FLUX_SCALE)
    _W["model"] = model
    _W["demands"] = demands


def _solve_gene(gene: str) -> tuple[str, int, np.ndarray, float, dict[str, float]]:
    model = _W["model"]
    with model:
        knocked = knock_out_model_genes(model, [gene])
        v_growth, mu, maxima = solve_gene(model, _W["demands"])
    return gene, len(knocked), v_growth, mu, maxima


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--out-tag", default="aa")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for the 028 stores.")
    started = time.time()

    model, demands, info = build_strain_model()
    rxn_ids = [r.id for r in model.reactions]
    n_rxn = len(rxn_ids)
    for r in model.reactions:
        r.lower_bound = max(r.lower_bound, -FLUX_SCALE)
        r.upper_bound = min(r.upper_bound, FLUX_SCALE)
    wt_v, wt_growth, wt_max = solve_gene(model, demands)
    assert wt_growth > LETHAL_GROWTH, "the wild type does not grow on the medium"
    print(
        f"strain: {n_rxn} reactions, medium {MEDIUM}; wild-type growth {wt_growth:.4f} "
        f"(unmodified {info['growth_unmodified']:.4f}); amino-acid maxima at "
        f"{PRODUCTION_GROWTH_FRACTION:.0%} own growth: "
        + ", ".join(f"{k} {wt_max[k]:.3f}" for k in AA_KEYS)
        + f"; {time.time() - started:.0f}s",
        flush=True,
    )
    genes = sorted(g.id for g in model.genes)
    if args.limit_genes:
        genes = genes[: args.limit_genes]

    rows: list[GeneRow] = []
    vectors: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    n_lethal = 0
    with Pool(args.processes, initializer=_init_worker) as pool:
        for i, (gene, n_knocked, v_g, mu, maxima) in enumerate(
            pool.imap(_solve_gene, genes, chunksize=4), start=1
        ):
            if mu < LETHAL_GROWTH:
                n_lethal += 1
            ratio = mu / wt_growth
            moved = float(
                np.linalg.norm(v_g - wt_v) / max(float(np.linalg.norm(wt_v)), 1e-12)
            )
            aa_ratio = {
                k: (maxima[k] / wt_max[k] if wt_max[k] > 0 else 0.0) for k in AA_KEYS
            }
            vectors.append(v_g)
            stats.append(
                np.array(
                    [mu, ratio, n_knocked, moved]
                    + [maxima[k] for k in AA_KEYS]
                    + [aa_ratio[k] for k in AA_KEYS],
                    dtype=np.float32,
                )
            )
            rows.append(
                GeneRow(
                    gene=gene,
                    growth=mu,
                    growth_ratio=ratio,
                    n_reactions_knocked_out=n_knocked,
                    relative_change_from_wild_type=moved,
                    aa_ratio=aa_ratio,
                )
            )
            if i % 100 == 0:
                print(
                    f"  {i}/{len(genes)} genes, {time.time() - started:.0f}s",
                    flush=True,
                )

    matrix = np.vstack(vectors).astype(np.float32)
    provenance = AaStoreProvenance(
        gem="yeast-GEM 9.0.2",
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=(
            "one cobra pfba growth solve per gene (max growth then least flux), then 19 "
            f"linear programs with growth held at {PRODUCTION_GROWTH_FRACTION} of the "
            "knockout's own optimum, each maximizing one amino acid's cytosolic demand; "
            "gene deleted through cobra.manipulation.knock_out_model_genes; written by "
            "aa_reference_flux_table.py"
        ),
        solver="cobra pfba and slim_optimize (glpk)",
        n_nonoptimal_zero_flux=0,
        medium=MEDIUM,
        media_report=MediaApplied.model_validate(info["media"]),
        split_file="every GEM gene",
        wild_type_growth=wt_growth,
        wild_type_key="",
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=n_lethal,
        n_genes_outside_the_gem=0,
        background_genes=[],
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        amino_acids=dict(AMINO_ACID_TO_GEM_METABOLITE),
        demand_reactions=[demands[k] for k in AA_KEYS],
        production_growth_fraction=PRODUCTION_GROWTH_FRACTION,
        wild_type_aa_at_own_growth_floor={k: wt_max[k] for k in AA_KEYS},
        n_reactions_total=n_rxn,
        n_stat_columns=len(STAT_NAMES),
        stat_layout=(
            "growth | deletion_growth_ratio | n_reactions_knocked_out | "
            "relative_change_from_wild_type | aa_<key>_at_own_growth_floor x19 | "
            "aa_<key>_ratio_to_wild_type x19"
        ),
        background_note=(
            "the Mulleder collection is prototrophic (pHLUM), so the wild type is the "
            "unmodified model on the synthetic minimal medium; no marker is deleted."
        ),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    enc_path = osp.join(
        args.out_dir, f"ess_solve_encodings_{MEDIUM}_{args.out_tag}.npz"
    )
    np.savez(
        enc_path,
        genes=np.array(genes),
        v_deletion=matrix,
        v_wild_type=wt_v.astype(np.float32),
        gene_flux_stats=np.vstack(stats).astype(np.float32),
        stat_names=np.array(STAT_NAMES),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(provenance.model_dump_json()),
    )
    ratio_csv = osp.join(args.results_dir, f"fba_ratio_{MEDIUM}_{args.out_tag}.csv")
    with open(ratio_csv, "w") as fh:
        fh.write(f"gene,{MEDIUM}\n")
        for row in rows:
            fh.write(f"{row.gene},{row.growth_ratio:.6f}\n")
    n_aa_moved = {
        k: int(sum(abs(r.aa_ratio[k] - 1.0) > 1e-3 for r in rows)) for k in AA_KEYS
    }
    summary = {
        "store": enc_path,
        "ratio_csv": ratio_csv,
        "n_genes": len(genes),
        "n_lethal": n_lethal,
        "n_growth_moved": int(sum(abs(r.growth_ratio - 1.0) > 1e-3 for r in rows)),
        "n_aa_moved": n_aa_moved,
        "n_any_aa_moved": int(
            sum(any(abs(r.aa_ratio[k] - 1.0) > 1e-3 for k in AA_KEYS) for r in rows)
        ),
        "provenance": json.loads(provenance.model_dump_json()),
        "seconds": time.time() - started,
    }
    out = osp.join(args.results_dir, f"aa_reference_flux_table_{args.out_tag}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"{len(genes)} genes: {n_lethal} lethal, {summary['n_growth_moved']} move growth, "
        f"{summary['n_any_aa_moved']} move at least one amino acid; "
        f"{time.time() - started:.0f}s"
    )
    print(enc_path)
    print(out)


if __name__ == "__main__":
    main()

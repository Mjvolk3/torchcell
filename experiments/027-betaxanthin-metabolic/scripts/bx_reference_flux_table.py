# experiments/027-betaxanthin-metabolic/scripts/bx_reference_flux_table.py
# [[experiments.027-betaxanthin-metabolic.scripts.bx_reference_flux_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/bx_reference_flux_table

r"""The layer-free module's store for the betaxanthin strain: one solve pair per GEM gene.

The 028 essentiality work settled the module's portable form: an offline parsimonious
solve per genotype and medium, its flux vector read as :math:`\log(1+|v|)` through a
zero-initialized affine on the output. That store was built on the wild-type model, which
is the strain the essentiality labels describe. The Cachera screen is not that strain.
Every genotype there is BY4741 (``his3D1 leu2D0 met15D0 ura3D0``) carrying the
betaxanthin cassette, CYP76AD1 and DOD integrated at XII-5, grown on YPD; so the store
must carry the cassette, the marker deletions and the medium, or it describes a strain
nobody screened.

THE STRAIN, AS THIS STORE REPRESENTS IT.

* yeast-GEM 9.0.2 plus ``build_betaxanthin_pathway`` (``torchcell.metabolism.betaxanthin``):
  tyrosine 3-hydroxylase, DOPA dioxygenase, the oxidase branch, and one spontaneous
  condensation per amine partner with a demand ``DM_btx_<partner>``; the cassette genes
  are never deleted;
* the four BY4741 markers knocked out through the GPR, as ``cbm_baseline_bx.py`` does, so
  the reference is the background the screen was made in, not a prototroph;
* the medium is the corrected YPD stand-in (``FBA_MEDIA["YPD"]``: YNB, twenty amino
  acids, uracil and adenine), under which the ``ura3`` background grows;
* ARO4-K229L and ARO7-G141S are feedback-insensitive alleles of native genes. A
  stoichiometric model carries no feedback inhibition, so there is nothing for the
  alleles to relax; they are recorded as background and change no bound.

TWO SOLVES PER GENE, both parsimonious (``cobra.flux_analysis.pfba``):

1. **growth**: maximize growth, then least total flux; the vector every 028 store carries;
2. **production**: with growth held at ``PRODUCTION_GROWTH_FRACTION`` of the knockout's own
   optimum, maximize the sum of the betaxanthin demands, then least total flux. This is
   the ``btx_at_90pct_own_growth`` reading of the constraint-based baseline, kept as a
   whole vector rather than one number, because the module reads vectors.

A gene outside the GEM reads the wild-type pair. A knockout with no growth is the zero
pair. The store is written in the shape the 028 trainer reads (``ESS_SOLVE_TABLE``):
``genes``, ``v_deletion`` ``[n, 2 r + 1]`` with the growth vector, the production vector
and the total betaxanthin demand as the last column, ``v_wild_type``, ``rxn_ids`` (the
production block suffixed ``__btx``, the last ``btx_total_demand``), ``gene_flux_stats``
with ``stat_names``, and ``provenance``. Beside it a CSV of the growth ratio per gene for
the trainer's medium column (``ESS_FBA_CSV``). Run from the worktree root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/bx_reference_flux_table.py \
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
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.betaxanthin import (  # noqa: E402
    betaxanthin_demand_ids,
    build_betaxanthin_pathway,
)
from torchcell.metabolism.pathway import apply_pathway  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

MEDIUM = "YPD"
#: BY4741's four marker deletions, as systematic ORFs (MET15 is the GEM's MET17).
MARKER_KNOCKOUTS: dict[str, str] = {
    "HIS3": "YOR202W",
    "LEU2": "YCL018W",
    "MET15/MET17": "YLR303W",
    "URA3": "YEL021W",
}
#: The cassette genes and the two deregulated native alleles: background, never deleted.
BACKGROUND_GENES = ["CYP76AD1", "DOD", "YBR249C", "YPR060C"]
PRODUCTION_GROWTH_FRACTION = 0.9
LETHAL_GROWTH = 1e-6
STAT_NAMES = [
    "growth",
    "btx_at_own_growth_floor",
    "btx_ratio_to_wild_type",
    "n_reactions_knocked_out",
    "relative_change_from_wild_type",
    "deletion_growth_ratio",
]


class BxStoreProvenance(StoreProvenance):
    """The essentiality store's record plus what makes this the Cachera strain."""

    cassette_reactions: int
    cassette_metabolites: int
    betaxanthin_demands: list[str]
    marker_knockouts: dict[str, str]
    production_growth_fraction: float
    wild_type_btx_at_own_growth_floor: float
    background_growth_unmodified: float
    n_reactions_total: int
    n_columns: int
    column_layout: str
    allele_note: str


class GeneRow(BaseModel):
    """One gene's row of the store summary."""

    gene: str
    in_gem: bool
    growth: float
    growth_ratio: float
    btx: float
    btx_ratio: float
    n_reactions_knocked_out: int
    relative_change_from_wild_type: float


_W: dict[str, Any] = {}


def build_strain_model() -> tuple[Any, list[str], dict[str, Any]]:
    """yeast-GEM plus the cassette on corrected YPD, the markers deleted for good."""
    base = YeastGEM().model
    pathway = build_betaxanthin_pathway(base)
    model = apply_pathway(base, pathway)
    demands = betaxanthin_demand_ids(pathway)
    media = media_report(model, MEDIUM)
    growth_unmodified = float(model.slim_optimize())
    knock_out_model_genes(model, list(MARKER_KNOCKOUTS.values()))
    info = {
        "cassette_reactions": len(pathway.reactions),
        "cassette_metabolites": len(pathway.metabolites),
        "demands": demands,
        "media": media,
        "growth_unmodified": growth_unmodified,
    }
    return model, demands, info


def solve_pair(
    model: Any, demands: list[str]
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """(growth vector, growth, production vector, total betaxanthin demand) of ``model``."""
    n = len(model.reactions)
    growth_rxn = next(r for r in model.reactions if r.objective_coefficient != 0)
    try:
        grown = pfba(model)
    except Exception:
        return np.zeros(n, np.float32), 0.0, np.zeros(n, np.float32), 0.0
    mu = float(grown.fluxes[growth_rxn.id])
    if mu < LETHAL_GROWTH:
        return np.zeros(n, np.float32), 0.0, np.zeros(n, np.float32), 0.0
    v_growth = grown.fluxes.to_numpy().astype(np.float32)
    with model:
        growth_rxn.lower_bound = PRODUCTION_GROWTH_FRACTION * mu
        model.objective = {model.reactions.get_by_id(d): 1.0 for d in demands}
        try:
            produced = pfba(model)
        except Exception:
            return v_growth, mu, np.zeros(n, np.float32), 0.0
        v_btx = produced.fluxes.to_numpy().astype(np.float32)
        total = float(sum(produced.fluxes[d] for d in demands))
    return v_growth, mu, v_btx, total


def _init_worker() -> None:
    model, demands, _ = build_strain_model()
    _W["model"] = model
    _W["demands"] = demands
    _W["gem_genes"] = {g.id for g in model.genes}


def _solve_gene(gene: str) -> tuple[str, int, np.ndarray, float, np.ndarray, float]:
    model = _W["model"]
    with model:
        knocked = knock_out_model_genes(model, [gene])
        v_growth, mu, v_btx, total = solve_pair(model, _W["demands"])
    return gene, len(knocked), v_growth, mu, v_btx, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--limit-genes", type=int, default=0, help="Smoke test.")
    parser.add_argument("--out-dir", default=TABLE_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--out-tag", default="bx")
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
    wt_growth_v, wt_growth, wt_btx_v, wt_btx = solve_pair(model, demands)
    assert wt_growth > LETHAL_GROWTH, (
        "the BY4741 background with the cassette does not grow"
    )
    print(
        f"strain: {n_rxn} reactions ({info['cassette_reactions']} cassette), medium {MEDIUM}; "
        f"growth unmodified {info['growth_unmodified']:.4f}, background {wt_growth:.4f}; "
        f"betaxanthin at {PRODUCTION_GROWTH_FRACTION:.0%} own growth {wt_btx:.4f} "
        f"mmol/gDW/h over {len(demands)} demands; {time.time() - started:.0f}s",
        flush=True,
    )
    background = set(MARKER_KNOCKOUTS.values()) | set(BACKGROUND_GENES)
    genes = sorted(g.id for g in model.genes if g.id not in background)
    if args.limit_genes:
        genes = genes[: args.limit_genes]

    rows: list[GeneRow] = []
    vectors: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    n_lethal = 0
    wt_pair = np.concatenate([wt_growth_v, wt_btx_v, [np.float32(wt_btx)]])
    with Pool(args.processes, initializer=_init_worker) as pool:
        for i, (gene, n_knocked, v_g, mu, v_b, total) in enumerate(
            pool.imap(_solve_gene, genes, chunksize=4), start=1
        ):
            if mu < LETHAL_GROWTH:
                n_lethal += 1
            ratio = mu / wt_growth
            btx_ratio = total / wt_btx if wt_btx > 0 else 0.0
            moved = float(
                np.linalg.norm(v_g - wt_growth_v)
                / max(float(np.linalg.norm(wt_growth_v)), 1e-12)
            )
            vectors.append(np.concatenate([v_g, v_b, [np.float32(total)]]))
            stats.append(
                np.array(
                    [mu, total, btx_ratio, n_knocked, moved, ratio], dtype=np.float32
                )
            )
            rows.append(
                GeneRow(
                    gene=gene,
                    in_gem=True,
                    growth=mu,
                    growth_ratio=ratio,
                    btx=total,
                    btx_ratio=btx_ratio,
                    n_reactions_knocked_out=n_knocked,
                    relative_change_from_wild_type=moved,
                )
            )
            if i % 100 == 0:
                print(
                    f"  {i}/{len(genes)} genes, {time.time() - started:.0f}s",
                    flush=True,
                )

    matrix = np.vstack(vectors).astype(np.float32)
    columns = rxn_ids + [f"{r}__btx" for r in rxn_ids] + ["btx_total_demand"]
    provenance = BxStoreProvenance(
        gem="yeast-GEM 9.0.2 + Cachera betaxanthin cassette (build_betaxanthin_pathway)",
        growth_fraction=1.0,
        flux_scale=FLUX_SCALE,
        method=(
            "two cobra pfba solves per gene on the cassette model with the BY4741 markers "
            "deleted: (1) max growth then least flux; (2) growth floor at "
            f"{PRODUCTION_GROWTH_FRACTION} of the knockout's own optimum, max the sum of the "
            "betaxanthin demands, then least flux; gene deleted through "
            "cobra.manipulation.knock_out_model_genes; written by bx_reference_flux_table.py"
        ),
        solver="cobra pfba (glpk)",
        n_nonoptimal_zero_flux=0,
        medium=MEDIUM,
        media_report=MediaApplied.model_validate(info["media"]),
        split_file="every GEM gene outside the background",
        wild_type_growth=wt_growth,
        wild_type_key="",
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=n_lethal,
        n_genes_outside_the_gem=0,
        background_genes=sorted(background),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        cassette_reactions=info["cassette_reactions"],
        cassette_metabolites=info["cassette_metabolites"],
        betaxanthin_demands=demands,
        marker_knockouts=MARKER_KNOCKOUTS,
        production_growth_fraction=PRODUCTION_GROWTH_FRACTION,
        wild_type_btx_at_own_growth_floor=wt_btx,
        background_growth_unmodified=info["growth_unmodified"],
        n_reactions_total=n_rxn,
        n_columns=len(columns),
        column_layout="growth vector | production vector (__btx) | btx_total_demand",
        allele_note=(
            "ARO4-K229L and ARO7-G141S relieve feedback inhibition the stoichiometric "
            "model does not carry; recorded as background, no bound changed."
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
        v_wild_type=wt_pair.astype(np.float32),
        gene_flux_stats=np.vstack(stats).astype(np.float32),
        stat_names=np.array(STAT_NAMES),
        rxn_ids=np.array(columns),
        provenance=np.array(provenance.model_dump_json()),
    )
    ratio_csv = osp.join(args.results_dir, f"fba_ratio_{MEDIUM}_{args.out_tag}.csv")
    with open(ratio_csv, "w") as fh:
        fh.write(f"gene,{MEDIUM}\n")
        for row in rows:
            fh.write(f"{row.gene},{row.growth_ratio:.6f}\n")
    summary = {
        "store": enc_path,
        "ratio_csv": ratio_csv,
        "n_genes": len(genes),
        "n_lethal": n_lethal,
        "n_btx_moved": int(sum(abs(r.btx_ratio - 1.0) > 1e-3 for r in rows)),
        "n_growth_moved": int(sum(abs(r.growth_ratio - 1.0) > 1e-3 for r in rows)),
        "provenance": json.loads(provenance.model_dump_json()),
        "seconds": time.time() - started,
    }
    out = osp.join(args.results_dir, f"bx_reference_flux_table_{args.out_tag}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"{len(genes)} genes: {n_lethal} lethal, {summary['n_growth_moved']} move growth, "
        f"{summary['n_btx_moved']} move betaxanthin; {time.time() - started:.0f}s"
    )
    print(enc_path)
    print(out)


if __name__ == "__main__":
    main()

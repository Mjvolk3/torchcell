# experiments/028-gene-essentiality/scripts/ess_reference_flux_table.py
# [[experiments.028-gene-essentiality.scripts.ess_reference_flux_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/ess_reference_flux_table

r"""One parsimonious reference flux per single-gene deletion, for the essentiality cells.

Two consumers, one solve.

1. **The flux layer's warm start.** Under ``--flux-parameterization warmstart`` the layer's
   proposal is ``v_ref(genotype) + alpha N z``, and ``--flux-reference-table`` supplies
   ``v_ref`` per deleted-gene set through ``FluxLayer.genotype_reference``. With this table
   the growth column starts at the deletion's own pFBA growth instead of at a free
   coordinate, which is what rounds 2, 4 and 7 and the growth probe found it to be. The
   table is built on the medium the essentiality cell runs on (``HEAD_MEDIUM``, YPD under
   ``--apply-media dataset``), through the same ``media_report`` the cell's own box is
   built with, so the reference sits inside the cell's box.

2. **The ladder's solve-derived encodings.** The same per-deletion flux vectors, and a
   handful of numbers about each gene's own reactions under the wild-type solve (flux
   carried, flux variability), are gene encodings that cost one linear program per gene.
   ``logistic_baselines.py`` reads them from the ``encodings`` file this script writes and
   scores them on the split beside the sequence embeddings.

The gene list is every labeled gene of the split file (the union of its train, validation
and held-out genes), so the layer's lookup misses none of them; a labeled gene outside the
GEM gets the wild-type flux, as the 026 table does, and is counted.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/ess_reference_flux_table.py --medium YPD

Writes ``$DATA_ROOT/data/torchcell/reference_flux/ess_reference_flux_table_<medium>.npz``
(the layer's contract: ``v``, ``keys``, ``rxn_ids``, ``provenance``),
``ess_solve_encodings_<medium>.npz`` (``genes``, ``v_deletion``, ``v_wild_type``,
``gene_flux_stats``, ``stat_names``) and a summary under ``results/``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import torch
from cobra.flux_analysis import flux_variability_analysis
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT_DIR = osp.join(RESULTS_DIR, "splits")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import (  # noqa: E402
    genotype_key,
    parsimonious_reference_flux,
)
from torchcell.metabolism.media import FBA_MEDIA  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

#: The layer's flux scale; the 026 table clamps to the same value.
FLUX_SCALE = 100.0
#: The FVA optimum fraction. 0.9 is cobrapy's documented convention for a variability
#: that is not pinned to the single optimal vertex.
FVA_FRACTION = 0.9
STAT_NAMES = (
    "n_reactions",
    "n_reactions_carrying_wild_type_flux",
    "max_abs_wild_type_flux",
    "sum_abs_wild_type_flux",
    "max_forced_abs_flux",
    "min_fva_range",
    "max_fva_range",
    "deletion_growth_ratio",
)


#: ``--solver`` -> the solver entry of ``solve_tolerance_read.SOLVERS`` (``pfba`` is the
#: scipy two-stage solve and needs no entry).
SOLVER_KEYS = {"pfba": None, "simplex": "simplex_exact", "ipm": "ipm_nocross_1e-4"}


class MediaApplied(BaseModel):
    """What ``media_report`` recorded when the medium was applied to the model.

    The bound vector itself is the artifact at ``bounds_artifact`` (a ``MediaBounds``
    record under DATA_ROOT, hash-named); ``bounds_sha256`` is its identity.
    """

    model_config = {"extra": "forbid"}

    applied: bool
    media_name: str | None = None
    recipe: str | None = None
    bounds_artifact: str | None = None
    bounds_sha256: str | None = None
    state: str | None = None
    n_components: int | None = None
    n_resolved: int | None = None
    n_exchanges_opened: int | None = None
    unresolved: list[str] = []
    excluded_by_role: list[str] = []
    wild_type_growth_before: float | None = None
    wild_type_growth_after: float | None = None
    provenance: str | None = None
    note: str | None = None


class StoreProvenance(BaseModel):
    """Where a reference flux store came from, written into the npz beside the fluxes.

    Read back with ``StoreProvenance.model_validate_json(str(payload["provenance"]))``.
    """

    model_config = {"extra": "forbid"}

    gem: str
    growth_fraction: float
    flux_scale: float
    method: str
    #: Absent from stores written before the round-36 solver arms (the pFBA stores of
    #: rounds 31 to 35 and the 026 table); ``None`` there means pFBA.
    solver: str | None = None
    n_nonoptimal_zero_flux: int | None = None
    medium: str
    media_report: MediaApplied
    split_file: str
    wild_type_growth: float
    wild_type_key: str
    n_sets: int
    n_lethal_zero_flux: int
    n_genes_outside_the_gem: int
    background_genes: list[str]
    python_hash_seed: str | None
    flux_sha256: str


class DerivedStoreProvenance(BaseModel):
    """A store derived from a solver store (``isozyme_affine_store.py`` and kin).

    ``source_store_provenance`` is the solver store's own record when the derived store
    carries its flux vector, and ``None`` for a store of derived columns alone.
    """

    model_config = {"extra": "forbid"}

    method: str
    gem_version: str
    columns: list[str]
    source_store: str | None = None
    source_store_provenance: StoreProvenance | None = None


class GrowthSolve(BaseModel):
    """One growth-maximizing solve by a named HiGHS solver: the flux and whether it solved."""

    model_config = {"arbitrary_types_allowed": True}

    v: np.ndarray
    growth: float
    optimal: bool
    status: str


def growth_solve(lb: np.ndarray, ub: np.ndarray, key: str, biomass: int) -> GrowthSolve:
    """The round-27 program under one solver, through ``solve_tolerance_read``'s LP.

    The split form in micromoles that ``lp_layer`` solves, with a fresh Highs object per
    solve (its time limit counts the object's lifetime). A non-optimal status returns the
    zero flux and is counted by the caller; the interior point at KKT 1e-4 reported 6 of
    1,121 programs non-optimal in the tolerance read.
    """
    import highspy
    from solve_tolerance_read import UNIT_SCALE, _fresh_highs, split_bounds

    ub = np.maximum(ub, lb)
    lo, hi = split_bounds(lb * UNIT_SCALE, ub * UNIT_SCALE)
    n = lo.size
    r = n // 2
    h = _fresh_highs(key)
    h.changeColsBounds(n, np.arange(n, dtype=np.int32), lo, hi)
    h.run()
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        return GrowthSolve(
            v=np.zeros(r),
            growth=0.0,
            optimal=False,
            status=h.modelStatusToString(status),
        )
    x = np.asarray(h.getSolution().col_value, dtype=np.float64) / UNIT_SCALE
    v = x[:r] - x[r:]
    return GrowthSolve(v=v, growth=float(v[biomass]), optimal=True, status="Optimal")


class DeletionRow(BaseModel):
    """One single-gene deletion's solve."""

    gene: str
    in_gem: bool
    growth: float
    growth_ratio: float
    n_reactions_knocked_out: int
    relative_change_from_wild_type: float


def labeled_genes(split_file: str) -> list[str]:
    with gzip.open(osp.join(SPLIT_DIR, split_file), "rt") as fh:
        split = json.load(fh)
    return sorted(split["labels"])


def gene_stats(
    model: Any, wild: np.ndarray, rxn_index: dict[str, int], fva: Any, gene: str
) -> np.ndarray:
    """The wild-type solve and its variability, read on one gene's own reactions."""
    if gene not in {g.id for g in model.genes}:
        return np.zeros(len(STAT_NAMES) - 1, dtype=np.float32)
    rxns = [r.id for r in model.genes.get_by_id(gene).reactions]
    idx = [rxn_index[r] for r in rxns]
    v = np.abs(wild[idx])
    lo = fva.loc[rxns, "minimum"].to_numpy()
    hi = fva.loc[rxns, "maximum"].to_numpy()
    forced = np.where(
        np.sign(lo) == np.sign(hi), np.minimum(np.abs(lo), np.abs(hi)), 0.0
    )
    rng = hi - lo
    return np.array(
        [
            len(rxns),
            int((v > 1e-6).sum()),
            float(v.max()),
            float(v.sum()),
            float(forced.max()),
            float(rng.min()),
            float(rng.max()),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium", default="YPD", choices=sorted(FBA_MEDIA) + ["none"])
    parser.add_argument("--split-file", default="ess_fcl_val0.2_s0.json.gz")
    parser.add_argument(
        "--genes",
        default="split",
        choices=("split", "gem"),
        help=(
            "split (default): the labeled genes of --split-file. gem: EVERY gene of the "
            "GEM, so the store covers any single-deletion dataset on this medium; a gene "
            "outside the GEM reads the wild-type row at training time. The portable form."
        ),
    )
    parser.add_argument("--growth-fraction", type=float, default=1.0)
    parser.add_argument(
        "--solver",
        default="pfba",
        choices=sorted(SOLVER_KEYS),
        help=(
            "pfba (default): the two-stage parsimonious solve through scipy's HiGHS, the "
            "table every round through 33 read. simplex / ipm: the plain growth-maximizing "
            "program (the round-27 flux head's, no parsimony stage) solved by HiGHS 1.15's "
            "dual simplex at default tolerances or its interior point without crossover "
            "at KKT 1e-4, through solve_tolerance_read.py's program; the outputs carry "
            "the solver's name. Round 36: a store built from the kind of point a "
            "first-order solve on the card returns."
        ),
    )
    parser.add_argument(
        "--out-tag",
        default="",
        help=(
            "Appended to the output stems, so a rebuilt store (a corrected medium, for "
            "one) lands beside the store it replaces instead of overwriting it."
        ),
    )
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for the 026 reference table.")
    suffix = ("" if args.solver == "pfba" else f"_{args.solver}") + (
        f"_{args.out_tag}" if args.out_tag else ""
    )

    started = time.time()
    os.makedirs(TABLE_DIR, exist_ok=True)
    source = YeastGEM()
    model = source.model
    media = media_report(model, None if args.medium == "none" else args.medium)
    print("medium:", json.dumps({k: v for k, v in media.items() if k != "unresolved"}))
    gem = build_gem_tensors(model, source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    rxn_index = {r: i for i, r in enumerate(rxn_ids)}
    gem_genes = {g.id for g in model.genes}
    lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE)
    ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE)
    biomass = int(gem.biomass_index)

    genes = sorted(gem_genes) if args.genes == "gem" else labeled_genes(args.split_file)
    print(
        f"{len(genes)} {'GEM' if args.genes == 'gem' else 'labeled'} genes, "
        f"{sum(g in gem_genes for g in genes)} in the GEM"
    )

    solver_key = SOLVER_KEYS[args.solver]
    if solver_key is None:
        wild = parsimonious_reference_flux(
            gem.s, lb, ub, biomass, growth_fraction=args.growth_fraction
        )
        wild_v = wild.v.numpy()
        wild_growth = wild.objective_value
    else:
        import scipy.sparse as sp
        from solve_tolerance_read import _init_worker

        s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
        a_csc = sp.hstack([sp.csr_matrix(s_dense), -sp.csr_matrix(s_dense)]).tocsc()
        c = np.zeros(2 * len(rxn_ids))
        c[biomass] = 1.0
        c[len(rxn_ids) + biomass] = -1.0
        _init_worker(a_csc, c)
        wild_solve = growth_solve(
            lb.double().numpy(), ub.double().numpy(), solver_key, biomass
        )
        assert wild_solve.optimal, wild_solve.status
        wild_v = wild_solve.v.astype(np.float32)
        wild_growth = wild_solve.growth
    print(f"wild-type growth {wild_growth:.4f} ({args.solver})")
    n_nonoptimal = 0
    fva = flux_variability_analysis(
        model,
        reaction_list=model.reactions,
        fraction_of_optimum=FVA_FRACTION,
        processes=8,
    )
    print(f"FVA on {len(fva)} reactions, {time.time() - started:.0f}s")

    keys = [genotype_key([])]
    fluxes = [wild_v]
    per_gene_v: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    rows: list[DeletionRow] = []
    n_lethal = 0
    for count, gene in enumerate(genes, start=1):
        if gene not in gem_genes:
            flux, growth, n_knocked = wild_v, wild_growth, 0
        else:
            with model:
                knocked = knock_out_model_genes(model, [gene])
                k_lb = torch.tensor(
                    [r.lower_bound for r in model.reactions], dtype=torch.float32
                ).clamp(-FLUX_SCALE, FLUX_SCALE)
                k_ub = torch.tensor(
                    [r.upper_bound for r in model.reactions], dtype=torch.float32
                ).clamp(-FLUX_SCALE, FLUX_SCALE)
                n_knocked = len(knocked)
            if solver_key is None:
                try:
                    record = parsimonious_reference_flux(
                        gem.s, k_lb, k_ub, biomass, growth_fraction=args.growth_fraction
                    )
                    flux, growth = record.v.numpy(), record.objective_value
                except ValueError:
                    # A lethal knockout is a result: the zero flux, as in the 026 table.
                    n_lethal += 1
                    flux, growth = np.zeros(len(rxn_ids), dtype=np.float32), 0.0
            else:
                solve = growth_solve(
                    k_lb.double().numpy(), k_ub.double().numpy(), solver_key, biomass
                )
                flux, growth = solve.v.astype(np.float32), solve.growth
                if not solve.optimal:
                    # The growth program is feasible for every deletion (the zero flux
                    # with the medium's exchanges is always admissible), so a non-optimal
                    # status here is the solver stopping short, not a lethal call; the
                    # row keeps the zero flux and the count is in the provenance.
                    n_nonoptimal += 1
                elif growth < 1e-6:
                    n_lethal += 1
        ratio = growth / wild_growth
        moved = float(
            np.linalg.norm(flux - wild_v) / max(float(np.linalg.norm(wild_v)), 1e-12)
        )
        keys.append(genotype_key([gene]))
        fluxes.append(flux)
        per_gene_v.append(flux)
        stats.append(
            np.concatenate(
                [gene_stats(model, wild_v, rxn_index, fva, gene), [np.float32(ratio)]]
            )
        )
        rows.append(
            DeletionRow(
                gene=gene,
                in_gem=gene in gem_genes,
                growth=float(growth),
                growth_ratio=float(ratio),
                n_reactions_knocked_out=n_knocked,
                relative_change_from_wild_type=moved,
            )
        )
        if count % 200 == 0:
            print(f"  {count}/{len(genes)} genes, {time.time() - started:.0f}s")

    matrix = np.vstack(fluxes).astype(np.float32)
    provenance = StoreProvenance(
        gem=osp.basename(source.model_dir),
        growth_fraction=args.growth_fraction,
        flux_scale=FLUX_SCALE,
        method=(
            "pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) with the gene "
            "deleted through cobra.manipulation.knock_out_model_genes"
            if solver_key is None
            else (
                f"growth-maximizing LP (the round-27 program, split form in micromoles) "
                f"by HiGHS {solver_key} through solve_tolerance_read.py, with the gene "
                "deleted through cobra.manipulation.knock_out_model_genes"
            )
        ),
        solver=args.solver,
        n_nonoptimal_zero_flux=n_nonoptimal,
        medium=args.medium,
        media_report=MediaApplied.model_validate(media),
        split_file=args.split_file
        if args.genes == "split"
        else "every GEM gene (--genes gem)",
        wild_type_growth=wild_growth,
        wild_type_key=keys[0],
        n_sets=len(keys),
        n_lethal_zero_flux=n_lethal,
        n_genes_outside_the_gem=sum(1 for r in rows if not r.in_gem),
        background_genes=[],
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
    )
    table_path = osp.join(
        TABLE_DIR, f"ess_reference_flux_table_{args.medium}{suffix}.npz"
    )
    np.savez(
        table_path,
        v=matrix,
        keys=np.array(keys),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(provenance.model_dump_json()),
    )
    # The per-gene growth ratio on this medium as the CSV the trainer's media-ratio
    # column reads (ESS_FBA_CSV, --ess-fba-media <medium>), so a store over every GEM gene
    # carries its own ratio table for the targets the 1,121-gene CSV does not cover.
    ratio_csv = osp.join(RESULTS_DIR, f"fba_ratio_{args.medium}{suffix}.csv")
    with open(ratio_csv, "w") as fh:
        fh.write(f"gene,{args.medium}\n")
        for row in rows:
            fh.write(f"{row.gene},{row.growth_ratio:.6f}\n")
    print(ratio_csv)
    enc_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{args.medium}{suffix}.npz")
    np.savez(
        enc_path,
        genes=np.array(genes),
        v_deletion=np.vstack(per_gene_v).astype(np.float32),
        v_wild_type=wild_v.astype(np.float32),
        gene_flux_stats=np.vstack(stats).astype(np.float32),
        stat_names=np.array(STAT_NAMES),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(provenance.model_dump_json()),
    )
    summary = {
        "table_path": table_path,
        "encodings_path": enc_path,
        "provenance": provenance.model_dump(),
        "seconds": time.time() - started,
        "rows": [r.model_dump() for r in rows],
    }
    out = osp.join(RESULTS_DIR, f"ess_reference_flux_table_{args.medium}{suffix}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"{len(keys)} rows, {n_lethal} lethal, "
        f"{provenance.n_genes_outside_the_gem} outside the GEM; "
        f"{time.time() - started:.0f}s\n{table_path}\n{enc_path}\n{out}"
    )


if __name__ == "__main__":
    main()

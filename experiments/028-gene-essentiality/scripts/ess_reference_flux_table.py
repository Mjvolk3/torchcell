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
    forced = np.where(np.sign(lo) == np.sign(hi), np.minimum(np.abs(lo), np.abs(hi)), 0.0)
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
    parser.add_argument("--growth-fraction", type=float, default=1.0)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for the 026 reference table.")

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

    genes = labeled_genes(args.split_file)
    print(f"{len(genes)} labeled genes, {sum(g in gem_genes for g in genes)} in the GEM")

    wild = parsimonious_reference_flux(
        gem.s, lb, ub, biomass, growth_fraction=args.growth_fraction
    )
    wild_v = wild.v.numpy()
    print(f"wild-type growth {wild.objective_value:.4f}")
    fva = flux_variability_analysis(
        model, reaction_list=model.reactions, fraction_of_optimum=FVA_FRACTION, processes=8
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
            flux, growth, n_knocked = wild_v, wild.objective_value, 0
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
            try:
                record = parsimonious_reference_flux(
                    gem.s, k_lb, k_ub, biomass, growth_fraction=args.growth_fraction
                )
                flux, growth = record.v.numpy(), record.objective_value
            except ValueError:
                # A lethal knockout is a result: the zero flux, as in the 026 table.
                n_lethal += 1
                flux, growth = np.zeros(len(rxn_ids), dtype=np.float32), 0.0
        ratio = growth / wild.objective_value
        moved = float(np.linalg.norm(flux - wild_v) / max(float(wild.v.norm()), 1e-12))
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
    provenance = {
        "gem": osp.basename(source.model_dir),
        "growth_fraction": args.growth_fraction,
        "flux_scale": FLUX_SCALE,
        "method": (
            "pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) with the gene "
            "deleted through cobra.manipulation.knock_out_model_genes"
        ),
        "medium": args.medium,
        "media_report": media,
        "split_file": args.split_file,
        "wild_type_growth": wild.objective_value,
        "wild_type_key": keys[0],
        "n_sets": len(keys),
        "n_lethal_zero_flux": n_lethal,
        "n_genes_outside_the_gem": sum(1 for r in rows if not r.in_gem),
        "background_genes": [],
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "flux_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    table_path = osp.join(TABLE_DIR, f"ess_reference_flux_table_{args.medium}.npz")
    np.savez(
        table_path,
        v=matrix,
        keys=np.array(keys),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(json.dumps(provenance)),
    )
    enc_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{args.medium}.npz")
    np.savez(
        enc_path,
        genes=np.array(genes),
        v_deletion=np.vstack(per_gene_v).astype(np.float32),
        v_wild_type=wild_v.astype(np.float32),
        gene_flux_stats=np.vstack(stats).astype(np.float32),
        stat_names=np.array(STAT_NAMES),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(json.dumps(provenance)),
    )
    summary = {
        "table_path": table_path,
        "encodings_path": enc_path,
        "provenance": provenance,
        "seconds": time.time() - started,
        "rows": [r.model_dump() for r in rows],
    }
    out = osp.join(RESULTS_DIR, f"ess_reference_flux_table_{args.medium}.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"{len(keys)} rows, {n_lethal} lethal, "
        f"{provenance['n_genes_outside_the_gem']} outside the GEM; "
        f"{time.time() - started:.0f}s\n{table_path}\n{enc_path}\n{out}"
    )


if __name__ == "__main__":
    main()

# experiments/026-metabolism-flux/scripts/probe_reference_flux_table.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_reference_flux_table
r"""One reference flux PER GENOTYPE, so the warm start carries the genotype.

WHY, MEASURED. The warm-started head's proposal is ``v_ref + alpha N z`` with ONE wild-type
reference, so the flux is very nearly the same vector for every genotype and the head reading
it is close to a constant predictor. Job 1946 (seed 2700, 200 epochs, read at epoch 143)
never left the betaxanthin mean plateau: validation prediction spread 0.0004 to 0.012,
validation Pearson flat at -0.052 after epoch 5, training data loss flat at 0.527 from epoch
0. The one channel the data term had for genotype information was ``alpha``, which grew from
0.0099 to 0.087 while the training-batch active-row balance max went from about 1.0 to 2.0
and ``c_balance_active`` from 0.001 to 0.17. Feasibility was the price of the only genotype
signal available.

WHAT THIS BUILDS. For every DISTINCT deleted-gene set in the dataset, a pFBA solution of the
GEM with those genes knocked out through the GPR, keyed by the sorted gene-set string. Keying
on the SET rather than on single genes removes the need for a multi-gene fallback entirely on
the genotypes the dataset contains: a double deletion is solved as a double deletion, not
composed from two singles, because a composition of two single-knockout fluxes is not a flux.

The knockout is applied with cobra's own GPR evaluation
(``cobra.manipulation.knock_out_model_genes``), so a reaction goes to zero exactly when its gene
protein reaction rule says it does, which is the same rule the flux layer's availability
chain applies. Solving directly on the bounds would have to re-implement that rule.

COST. The parameter review measured 0.06 s per knockout FBA over 1,161 genes. pFBA is two
LPs, so a table over a few thousand distinct gene sets is minutes of CPU, paid once.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_reference_flux_table.py

Writes the table to ``$DATA_ROOT/data/torchcell/reference_flux/`` and a summary to
``experiments/026-metabolism-flux/results/probe_reference_flux_table.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
BX_SCRIPTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")
sys.path.insert(0, BX_SCRIPTS)

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import (  # noqa: E402
    genotype_key,
    parsimonious_reference_flux,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

FLUX_SCALE = 100.0
#: A reference counts as moved by its deletion when it differs from the wild type by more
#: than this fraction of the wild type's norm.
DELETION_EFFECT_THRESHOLD = 0.01


def dataset_gene_sets() -> tuple[list[list[str]], dict[str, object]]:
    """Every distinct deleted-gene set in the 027 dataset, as GEM gene ids.

    Read from the dataset's own perturbation records rather than from a gene list, so a
    genotype the table is built for is a genotype the trainer will actually ask about.
    """
    from train_bx import build_dataset

    dataset = build_dataset()
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    seen: dict[str, list[str]] = {}
    counts = {"records": 0, "with_perturbation": 0, "sizes": {}}
    for index in range(len(dataset)):
        record = dataset[index]
        counts["records"] += 1
        indices = record["gene"].perturbation_indices.tolist()
        names = sorted({gene_ids[i] for i in indices})
        if names:
            counts["with_perturbation"] += 1
        size = str(len(names))
        counts["sizes"][size] = counts["sizes"].get(size, 0) + 1
        seen.setdefault(genotype_key(names), names)
    return list(seen.values()), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--growth-fraction", type=float, default=1.0)
    parser.add_argument("--max-sets", type=int, default=0, help="0 means every set.")
    parser.add_argument("--out", default="reference_flux_table.npz")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit(
            "PYTHONHASHSEED must be 0: this table is keyed by gene-set strings and is "
            "consumed beside a k_cat table whose resolution depends on the hash seed."
        )

    started = time.time()
    os.makedirs(TABLE_DIR, exist_ok=True)
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(model, source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    gem_genes = {g.id for g in model.genes}
    lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE)
    ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE)
    biomass = int(gem.biomass_index)

    sets, counts = dataset_gene_sets()
    print(f"distinct gene sets in the dataset: {len(sets)}; record counts {counts}")
    if args.max_sets:
        sets = sets[: args.max_sets]

    # THE WILD TYPE FIRST, and it is row 0 under the empty key, so a genotype with no
    # deletion and a genotype the table misses resolve to the same vector by construction.
    wild = parsimonious_reference_flux(
        gem.s, lb, ub, biomass, growth_fraction=args.growth_fraction
    )
    keys = [genotype_key([])]
    fluxes = [wild.v.numpy()]
    rows: list[dict[str, object]] = []
    n_moved = 0
    n_outside_gem = 0
    n_infeasible = 0
    for count, names in enumerate(sets, start=1):
        key = genotype_key(names)
        if key == keys[0]:
            continue
        inside = [g for g in names if g in gem_genes]
        if len(inside) != len(names):
            n_outside_gem += 1
        if not inside:
            # Every deleted gene is outside the GEM, so the knockout cannot reach the
            # stoichiometry and the wild-type flux IS this genotype's reference.
            keys.append(key)
            fluxes.append(wild.v.numpy())
            continue
        # `with model:` restores every bound on exit, so the knockout is scoped and the
        # next genotype starts from the shipped model rather than from the last one's
        # leftovers.
        with model:
            knocked = knock_out_model_genes(model, inside)
            knocked_lb = torch.tensor(
                [r.lower_bound for r in model.reactions], dtype=torch.float32
            ).clamp(-FLUX_SCALE, FLUX_SCALE)
            knocked_ub = torch.tensor(
                [r.upper_bound for r in model.reactions], dtype=torch.float32
            ).clamp(-FLUX_SCALE, FLUX_SCALE)
            n_knocked = len(knocked)
        try:
            record = parsimonious_reference_flux(
                gem.s,
                knocked_lb,
                knocked_ub,
                biomass,
                growth_fraction=args.growth_fraction,
            )
            flux = record.v.numpy()
            growth = record.objective_value
        except ValueError:
            # A LETHAL KNOCKOUT IS A RESULT, NOT AN ERROR. The growth LP has no solution at
            # the requested growth fraction, so the reference for that genotype is the zero
            # flux: the cell the reference describes does not grow.
            n_infeasible += 1
            flux = np.zeros(len(rxn_ids), dtype=np.float32)
            growth = 0.0
        moved = float(
            np.linalg.norm(flux - wild.v.numpy()) / max(float(wild.v.norm()), 1e-12)
        )
        if moved > DELETION_EFFECT_THRESHOLD:
            n_moved += 1
        keys.append(key)
        fluxes.append(flux)
        rows.append(
            {
                "key": key,
                "n_genes": len(names),
                "n_genes_in_gem": len(inside),
                "growth": growth,
                "n_reactions_knocked_out": n_knocked,
                "relative_change_from_wild_type": moved,
            }
        )
        if count % 200 == 0:
            print(f"  {count}/{len(sets)} sets, {time.time() - started:.0f}s")

    matrix = np.vstack(fluxes).astype(np.float32)
    provenance = {
        "gem": osp.basename(source.model_dir),
        "growth_fraction": args.growth_fraction,
        "flux_scale": FLUX_SCALE,
        "method": (
            "pFBA (parsimonious_reference_flux, two scipy HiGHS LPs) on the GEM with the "
            "genotype's genes deleted through cobra.manipulation.knock_out_model_genes, so "
            "the GPR decides which reactions close"
        ),
        "medium": (
            "the GEM's shipped exchange bounds, clamped to the layer's flux scale; no "
            "medium was applied on top"
        ),
        "wild_type_growth": wild.objective_value,
        "wild_type_key": keys[0],
        "n_sets": len(keys),
        "n_lethal_zero_flux": n_infeasible,
        "n_sets_with_genes_outside_the_gem": n_outside_gem,
        "n_sets_moved_more_than_1pct": n_moved,
        "deletion_effect_threshold": DELETION_EFFECT_THRESHOLD,
        "dataset_record_counts": counts,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "flux_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    table_path = osp.join(TABLE_DIR, args.out)
    np.savez(
        table_path,
        v=matrix,
        keys=np.array(keys),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(json.dumps(provenance)),
    )
    summary = {
        "table_path": table_path,
        "provenance": provenance,
        "fraction_moved_more_than_1pct": (
            n_moved / len(rows) if rows else 0.0
        ),
        "relative_change_quantiles": {
            q: float(
                np.quantile([r["relative_change_from_wild_type"] for r in rows], q)
            )
            for q in (0.1, 0.5, 0.9, 1.0)
        }
        if rows
        else {},
        "examples": rows[:25],
    }
    out_path = osp.join(RESULTS_DIR, "probe_reference_flux_table.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(provenance, indent=2))
    print(
        f"moved more than 1 percent: {n_moved} of {len(rows)} "
        f"({summary['fraction_moved_more_than_1pct']:.4f})"
    )
    print(f"wrote {table_path} and {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

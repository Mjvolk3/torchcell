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
from torchcell.metabolism.media import FBA_MEDIA  # noqa: E402
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
from train_flux import media_report  # noqa: E402

FLUX_SCALE = 100.0
#: A reference counts as moved by its deletion when it differs from the wild type by more
#: than this fraction of the wild type's norm.
DELETION_EFFECT_THRESHOLD = 0.01


def dataset_gene_sets() -> tuple[list[list[str]], dict[str, object]]:
    """Every distinct DELETED-gene set in the 027 dataset, as gene ids.

    DELETIONS ONLY, and that is a correction rather than a refinement. Every Cachera
    betaxanthin strain carries the same four-gene betaxanthin cassette as GENE ADDITIONS:
    the heterologous CYP76AD1 and DOD, plus the feedback-resistant native alleles
    ARO4^K229L (YBR249C) and ARO7^G141S (YPR060C) integrated ectopically at locus XII-5
    (``torchcell/datasets/scerevisiae/cachera2023.py``). They are an engineered BACKGROUND
    that every strain shares, not the experimental variable.

    ``GraphProcessor._process_gene_info`` collects ``systematic_gene_name`` from EVERY
    perturbation regardless of type, so those four reach the model as perturbation indices
    and the flux layer's availability gate reads them as knockouts. Keying a reference
    table the same way would solve the reference for a cell that has had its DAHP synthase
    and chorismate mutase deleted, which is the opposite of what these strains are: they
    overproduce THROUGH that pathway.

    The rule here is ``DeletionKeyedGenotypeAggregator``'s, reused rather than restated:
    a perturbation counts when its ``perturbation_type`` ends with ``DELETION_TYPE_SUFFIX``.
    """
    from train_bx import build_dataset

    from torchcell.data.genotype_aggregate import DELETION_TYPE_SUFFIX

    dataset = build_dataset()
    # LENGTH FIRST, THEN THE HANDLE. `Neo4jCellDataset.len` opens its own read handle and
    # closes it, so an environment opened before that call is closed underneath us and the
    # next read fails on a None env.
    n_records = len(dataset)
    dataset._init_lmdb_read()
    seen: dict[str, list[str]] = {}
    counts: dict[str, object] = {
        "records": 0,
        "with_deletion": 0,
        "deletion_sizes": {},
        "non_deletion_genes": {},
        "perturbation_types": {},
    }
    for index in range(n_records):
        raw = dataset._deserialize_json(dataset._read_from_lmdb(index))
        for item in raw:
            counts["records"] = int(counts["records"]) + 1
            perturbations = item["experiment"]["genotype"]["perturbations"]
            deletions = sorted(
                {
                    p["systematic_gene_name"]
                    for p in perturbations
                    if p["perturbation_type"].endswith(DELETION_TYPE_SUFFIX)
                }
            )
            for p in perturbations:
                kind = str(p["perturbation_type"])
                types = counts["perturbation_types"]
                types[kind] = types.get(kind, 0) + 1
                if not kind.endswith(DELETION_TYPE_SUFFIX):
                    other = counts["non_deletion_genes"]
                    name = str(p["systematic_gene_name"])
                    other[name] = other.get(name, 0) + 1
            if deletions:
                counts["with_deletion"] = int(counts["with_deletion"]) + 1
            sizes = counts["deletion_sizes"]
            sizes[str(len(deletions))] = sizes.get(str(len(deletions)), 0) + 1
            seen.setdefault(genotype_key(deletions), deletions)
    return list(seen.values()), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--growth-fraction", type=float, default=1.0)
    parser.add_argument(
        "--medium",
        default="YPD",
        choices=sorted(FBA_MEDIA) + ["none"],
        help=(
            "The medium to apply to the exchange bounds BEFORE the knockout. MEASURED "
            "reason for the default: on the GEM's shipped bounds every one of the "
            "dataset's genotypes has a growth optimum of exactly 0, because essentially "
            "all of them delete ARO4 (YBR249C) and ARO7 (YPR060C) and nothing supplies "
            "aromatic amino acids. The strains were grown and measured, so a reference "
            "that says they are dead is a statement about the medium, not about them. "
            "Cachera 2023 is solid YPD, and HEAD_MEDIUM maps the betaxanthin head to YPD."
        ),
    )
    parser.add_argument("--max-sets", type=int, default=0, help="0 means every set.")
    parser.add_argument("--out", default="reference_flux_table_ypd.npz")
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
    # THE MEDIUM FIRST, so the knockout is applied on top of it and the reference is solved
    # on the same exchange bounds the cell's own box will carry under --apply-media dataset.
    # A reference solved on one medium and clipped into a box built on another is two
    # different cells.
    # THE SAME FUNCTION THE CELL USES. `train_flux.media_report` closes every exchange and
    # opens exactly the named recipe, then records the wild-type growth before and after.
    # Calling it here rather than re-implementing it is what guarantees the reference and
    # the cell's own box are built on one boundary.
    media = media_report(model, None if args.medium == "none" else args.medium)
    print("medium:", json.dumps({k: v for k, v in media.items() if k != "unresolved"}))
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
        "medium": args.medium,
        "media_report": media,
        "medium_note": (
            f"{args.medium!r} applied through train_flux.media_report before the knockout, "
            "then clamped to the layer's flux scale. 'none' is the GEM's shipped boundary, "
            "on which EVERY genotype in this dataset has growth exactly 0 because almost "
            "all of them delete ARO4 (YBR249C) and ARO7 (YPR060C) and nothing supplies "
            "aromatic amino acids."
        ),
        "wild_type_growth": wild.objective_value,
        "wild_type_key": keys[0],
        "n_sets": len(keys),
        "n_lethal_zero_flux": n_infeasible,
        "n_sets_with_genes_outside_the_gem": n_outside_gem,
        "n_sets_moved_more_than_1pct": n_moved,
        "deletion_effect_threshold": DELETION_EFFECT_THRESHOLD,
        "dataset_record_counts": counts,
        # THE ENGINEERED BACKGROUND, carried by the table so the layer can drop it from the
        # lookup key without being told. These genes are perturbed in the dataset and are
        # NOT deleted: the Cachera betaxanthin cassette (CYP76AD1, DOD, and the
        # feedback-resistant native alleles ARO4^K229L / YBR249C and ARO7^G141S / YPR060C)
        # and the Ozaydin beta-carotene cassette (crtYB, crtI, BTS1 / YPL069C). The graph
        # processor puts every perturbation's gene name into perturbed_names regardless of
        # type, so without this list the layer's key would name them and miss every row.
        "background_genes": sorted(counts["non_deletion_genes"]),
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
    out_path = osp.join(
        RESULTS_DIR,
        f"probe_reference_flux_table_{osp.splitext(args.out)[0]}.json",
    )
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

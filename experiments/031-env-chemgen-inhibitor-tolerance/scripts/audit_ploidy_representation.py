# experiments/031-env-chemgen-inhibitor-tolerance/scripts/audit_ploidy_representation.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.audit_ploidy_representation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/audit_ploidy_representation
"""Audit how the five served chemogenomic datasets store ploidy and gene dosage.

Reads a few records from each built dev-tree LMDB through its loader (read-only) and
prints, per record, the stored ``ReferenceGenome`` on the experiment reference and every
perturbation in the genotype with its dosage-bearing fields. The point of the audit is to
answer one question from the STORED BYTES rather than from the loader source: is a
heterozygous (half-dose) arm distinguishable from a homozygous (absent) arm using only the
record, and does the record carry ploidy at all.

Writes a flat CSV of every perturbation seen to ``results/ploidy_audit.csv``.
"""

from __future__ import annotations

import csv
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    HetHillenmeyer2008Dataset,
    HomHillenmeyer2008Dataset,
)
from torchcell.datasets.scerevisiae.hoepfner2014 import EnvChemgenHoepfner2014Dataset
from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset
from torchcell.datasets.scerevisiae.wildenhain2015 import (
    EnvChemgenWildenhain2015Dataset,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)

# (label, class, dev-tree LMDB dir, indices to pull). Hoepfner holds BOTH arms in one
# store, so it needs indices far apart enough to land on HIP and on HOP.
SPECS: list[tuple[str, type, str, list[int]]] = [
    (
        "Vanacloig2022 (haploid MATalpha, 3DeltaAlpha host)",
        EnvChemgenVanacloig2022Dataset,
        "data/torchcell/env_chemgen_vanacloig2022",
        [0, 1],
    ),
    (
        "Hillenmeyer2008 HET (heterozygous diploid)",
        HetHillenmeyer2008Dataset,
        "data/torchcell/env_chemgen_hillenmeyer2008_het",
        [0, 1],
    ),
    (
        "Hillenmeyer2008 HOM (homozygous diploid)",
        HomHillenmeyer2008Dataset,
        "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        [0, 1],
    ),
    (
        "Hoepfner2014 (HIP heterozygous + HOP homozygous diploid)",
        EnvChemgenHoepfner2014Dataset,
        "data/torchcell/env_chemgen_hoepfner2014",
        [0, 1],
    ),
    (
        "Wildenhain2015 (haploid BY4741)",
        EnvChemgenWildenhain2015Dataset,
        "data/torchcell/env_chemgen_wildenhain2015",
        [0, 1],
    ),
]

# Every dosage- or state-bearing field a GenePerturbation leaf can carry.
DOSAGE_FIELDS = (
    "perturbation_type",
    "state",
    "copy_number",
    "reference_copy_number",
    "marker",
    "provenance",
    "mechanism_so_name",
)


def _hoepfner_arm_indices(ds: Any, n_probes: int = 64) -> list[int]:
    """Return one HIP index and one HOP index, found by probing the store at intervals.

    Hoepfner stores HIP and HOP in one LMDB in blocks, so index 0 and 1 are both the same
    arm while a linear scan to the first record of the other arm reads a large fraction of
    a multi-million-record store. Probing ``n_probes`` evenly spaced offsets finds both arms
    in a bounded number of reads. The arm is read off the perturbation class, which is the
    very thing under audit; the scan RAISES if a probe grid this coarse misses an arm rather
    than silently reporting one arm as the whole dataset.
    """
    total = len(ds)
    seen: dict[str, int] = {}
    for probe in range(n_probes):
        idx = min(total - 1, probe * total // n_probes)
        kinds = {
            p["perturbation_type"]
            for p in ds[idx]["experiment"]["genotype"]["perturbations"]
        }
        key = (
            "engineered_copy_number"
            if "engineered_copy_number" in kinds
            else "deletion"
        )
        seen.setdefault(key, idx)
        if len(seen) == 2:
            return sorted(seen.values())
    raise RuntimeError(
        f"{n_probes} probes over {total} records found only {sorted(seen)}; the HIP and "
        "HOP arms are meant to coexist in this store"
    )


def _print_record(label: str, idx: int, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Print one record's reference genome + genotype; return flat perturbation rows."""
    experiment = record["experiment"]
    reference = record["reference"]
    genome_ref = reference["genome_reference"]
    print(f"  [{idx}] record keys     = {', '.join(sorted(record.keys()))}")
    print(f"       experiment_type = {experiment['experiment_type']}")
    print("       experiment keys            = " + ", ".join(sorted(experiment.keys())))
    print(
        "       reference.genome_reference = "
        f"species={genome_ref['species']!r} strain={genome_ref['strain']!r} "
        f"ploidy={genome_ref.get('ploidy')!r}"
    )
    rows: list[dict[str, Any]] = []
    for pert in experiment["genotype"]["perturbations"]:
        fields = " ".join(
            f"{name}={pert.get(name)!r}" for name in DOSAGE_FIELDS if name in pert
        )
        print(
            f"       pert {pert['systematic_gene_name']} "
            f"({pert['perturbed_gene_name']}): {fields}"
        )
        row = {
            "dataset": label,
            "record_index": idx,
            "ploidy": genome_ref.get("ploidy"),
            "strain": genome_ref["strain"],
            "systematic_gene_name": pert["systematic_gene_name"],
            "perturbed_gene_name": pert["perturbed_gene_name"],
        }
        row.update({name: pert.get(name) for name in DOSAGE_FIELDS})
        rows.append(row)
    return rows


def main() -> None:
    """Print the genotype + reference genome for a couple of records per dataset."""
    all_rows: list[dict[str, Any]] = []
    for label, cls, rel_root, default_indices in SPECS:
        root = osp.join(DATA_ROOT, rel_root)
        print("=" * 88)
        print(f"{label}\n  root = {root}")
        ds = cls(root=root)
        print(f"  len  = {len(ds)}")
        indices = (
            _hoepfner_arm_indices(ds)
            if cls is EnvChemgenHoepfner2014Dataset
            else default_indices
        )
        for idx in indices:
            all_rows.extend(_print_record(label, idx, ds[idx]))
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = osp.join(RESULTS_DIR, "ploidy_audit.csv")
    fieldnames = list(all_rows[0].keys())
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} perturbation rows to {out}")


if __name__ == "__main__":
    main()

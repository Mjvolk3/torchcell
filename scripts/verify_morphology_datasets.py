# scripts/verify_morphology_datasets.py
# [[scripts.verify_morphology_datasets]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/verify_morphology_datasets
"""Run the WS6 L0-L4 record-level gate over the Ohya2005 CalMorph morphology dataset.

Loads the built LMDB for `scmd_ohya2005` from `$DATA_ROOT`, runs
:func:`torchcell.verification.morphology.verify_morphology_dataset`, adds an L4
cross-source check (Ohya's deletion-mutant gene set should contain the deletion genes
of the expression datasets -- same yeast deletion library, consistent naming), writes
a `verification_report.json` sibling of `experiment_reference_index.json`, and exits
non-zero if any level fails.

The Ohya dataset lives under `database/data/torchcell/scmd_ohya2005` (the KG-build
path), not `data/torchcell/...`.

Usage:
    ~/miniconda3/envs/torchcell/bin/python scripts/verify_morphology_datasets.py
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
import sys
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.verification.morphology import (
    perturbed_gene_set,
    verify_morphology_dataset,
)
from torchcell.verification.report import Level, LevelResult, Provenance

# Ohya morphology dataset (the abstract's r=0.619 phenotype). The built LMDB lives
# under the KG-build tree (`database/data/...`), which may be owned by the KG-build
# user and read-only; we READ it there but WRITE the report to the standard, writable
# `data/torchcell/...` tree where the other datasets' reports live.
OHYA_SOURCE_ROOT = "database/data/torchcell/scmd_ohya2005"
OHYA_REPORT_ROOT = "data/torchcell/scmd_ohya2005"
OHYA_EXPECTED_COUNT = 4718  # mutant records; 122 WT are aggregated into the reference
OHYA_PROVENANCE = Provenance(
    source_uri="http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/ (mt4718data.tsv, wt122data.tsv)",
    citation_key="ohyaHighdimensionalLargescalePhenotyping2005a",
    method="SCMD CalMorph TSV (mt4718 mutants + wt122 wildtype)",
    page="Ohya et al. 2005 PNAS; PMID 16365294",
)

# Expression deletion datasets whose perturbed genes should be covered by Ohya's.
DELETION_GENE_SOURCES = {
    "microarray_kemmeren2014": "data/torchcell/microarray_kemmeren2014",
    "sm_microarray_sameith2015": "data/torchcell/sm_microarray_sameith2015",
    "dm_microarray_sameith2015": "data/torchcell/dm_microarray_sameith2015",
}
# Fraction of each deletion dataset's genes that must appear in Ohya. Empirically
# Kemmeren=0.970, SM Sameith=0.988; 0.90 leaves headroom for genes legitimately
# profiled for expression but absent from the morphology screen, while still catching
# a gene-naming or format break (which would collapse the overlap).
MIN_OVERLAP = 0.90


def load_records(abs_root: str) -> list[dict[str, Any]]:
    """Read every LMDB entry under `<abs_root>/processed/lmdb`."""
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            records.append(pickle.loads(value))
    env.close()
    return records


def l4_gene_containment(
    ohya_genes: set[str], other_name: str, other_genes: set[str]
) -> LevelResult:
    """L4: fraction of `other_genes` present in Ohya's deletion set is >= MIN_OVERLAP."""
    if not other_genes:
        overlap = 0.0
    else:
        overlap = len(other_genes & ohya_genes) / len(other_genes)
    passed = overlap >= MIN_OVERLAP
    return LevelResult(
        level=Level.L4,
        name=f"gene_containment_{other_name}",
        passed=passed,
        message=(
            f"{overlap:.3f} of {other_name}'s {len(other_genes)} deletion genes "
            f"are in Ohya (>= {MIN_OVERLAP})"
        ),
        details={
            "other": other_name,
            "n_other": len(other_genes),
            "n_in_ohya": len(other_genes & ohya_genes),
            "overlap": overlap,
            "missing_examples": sorted(other_genes - ohya_genes)[:20],
        },
    )


def main() -> int:
    """Verify the Ohya morphology dataset, write its report, return an exit code."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    ohya_abs = osp.join(data_root, OHYA_SOURCE_ROOT)
    ohya_records = load_records(ohya_abs)
    report = verify_morphology_dataset(
        ohya_records,
        dataset_name="scmd_ohya2005",
        provenance=OHYA_PROVENANCE,
        expected_count=OHYA_EXPECTED_COUNT,
    )

    # L4: Ohya's deletion gene set should contain the expression datasets' genes.
    ohya_genes = perturbed_gene_set(ohya_records)
    for name, root in DELETION_GENE_SOURCES.items():
        other_abs = osp.join(data_root, root)
        if not osp.exists(osp.join(other_abs, "processed", "lmdb")):
            continue
        other_genes = perturbed_gene_set(load_records(other_abs))
        report.add(l4_gene_containment(ohya_genes, name, other_genes))

    report_dir = osp.join(data_root, OHYA_REPORT_ROOT, "preprocess")
    os.makedirs(report_dir, exist_ok=True)
    out = osp.join(report_dir, "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(report.summary())
    print(f"  -> verified LMDB: {osp.join(ohya_abs, 'processed', 'lmdb')}")
    print(f"  -> wrote report:  {out}")

    print("=" * 60)
    print("OHYA MORPHOLOGY PASS" if report.passed else "OHYA MORPHOLOGY FAILED")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

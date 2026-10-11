# experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_release_inventory.py
"""Measure the N'Guessan 2025 release before the loader builds: cells, the cell-to-
strain assignment, cells per segregant, the phenotype bytes per record, and, once the
walled Dryad files are deposited (#1014), the genotype cardinality.

Reads the four GitHub release files from the raw mirror (each sha256-checked against the
mirror's ``manifest.json``) and writes
``experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_release_inventory.json``:

- the release: barcodes, genes, median UMI and genes per cell (Table 1), the GEM-group
  suffixes, and the genes outside the S288C ORF set when the genomes tier is mounted;
- the assignment: cells with a significant match, the strains they reach, cells per
  strain (distribution and histogram), the unassigned cells and how many of them sit at
  or above the third UMI quartile (the paper's doublet definition starts there);
- the records: nonzero genes per pseudobulk record and the JSON bytes of its
  ``umi_count``, and the reference pseudobulk;
- the Dryad side: which of the four walled files are on disk, and when all are, the
  batch-1 strain count, the SNP count, and per assigned strain the blocks and pickled
  genotype bytes at a pure called-parent run and the 0.1, 0.05, 0.02 and 0.01 grids, the
  exact-0.5 and strictly-uncertain posterior fractions.

Run from the repo root: ``python experiments/036-dataset-fixes-before-kg-build/scripts/
nguessan2025_release_inventory.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.datamodels.schema import HaplotypeBlock
from torchcell.datasets.scerevisiae import nguessan2025 as ng
from torchcell.datasets.scerevisiae.bloom2019 import encode_blocks
from torchcell.verification.runners import _sgd_gene_set

STEPS: list[float | None] = [None, 0.1, 0.05, 0.02, 0.01]
OUT = "experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_release_inventory.json"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[int] | np.ndarray[Any, Any]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": round(float(array.mean()), 4),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def blocks_at(
    posterior: np.ndarray[Any, Any], markers: list[Any], step: float | None
) -> list[HaplotypeBlock]:
    """Runs of called parent (step None) or of called parent and grid level."""
    if step is None:
        coded = np.where(posterior > 0.5, 2, 1).astype(np.int8)
        return encode_blocks(coded, markers)
    parent = np.where(posterior > 0.5, 2, 1)
    q = np.where(parent == 2, posterior, 1 - posterior)
    level = np.rint(q / step).astype(np.int64)
    chrom = [m.chromosome for m in markers]
    out: list[HaplotypeBlock] = []
    start = 0
    for i in range(1, len(markers) + 1):
        if (
            i == len(markers)
            or parent[i] != parent[start]
            or level[i] != level[start]
            or chrom[i] != chrom[start]
        ):
            out.append(
                HaplotypeBlock(
                    chromosome=chrom[start],
                    start=markers[start].position,
                    end=markers[i - 1].position,
                    parent=int(parent[start]),  # type: ignore[arg-type]
                    posterior=min(1.0, round(int(level[start]) * step, 6)),
                    n_markers=i - start,
                )
            )
            start = i
    return out


def release_section(mirror: str) -> tuple[dict[str, Any], Any, Any]:
    """The GitHub release, verified, and the numbers measured on it."""
    manifest = ng.load_manifest(osp.dirname(osp.dirname(mirror)))
    rel = ng.raw_relpaths()
    for name in ng.RELEASE_FILES:
        path = osp.join(mirror, rel[name])
        if _sha256(path) != ng.manifest_sha256(manifest, rel[name]):
            raise RuntimeError(f"{path} does not match its manifest sha256")
    release = ng.read_expression(
        [osp.join(mirror, rel[n]) for n in ng.EXPRESSION_PARTS]
    )
    genes = ng.read_gene_table(osp.join(mirror, rel[ng.GENE_TABLE]))
    if genes != release.genes:
        raise RuntimeError("the gene table and the matrix columns differ")
    assignment = ng.read_assignment(osp.join(mirror, rel[ng.ASSIGNMENT_TABLE]))
    if [c.barcode for c in assignment] != release.barcodes:
        raise RuntimeError("the assignment and the matrix rows differ")
    totals = release.cell_totals
    genes_per_cell = np.diff(release.counts.indptr)
    section = {
        "barcodes": len(release.barcodes),
        "genes": len(release.genes),
        "median_umi_per_cell": float(np.median(totals)),
        "median_genes_per_cell": float(np.median(genes_per_cell)),
        "gem_groups": dict(Counter(ng.library_of(b) for b in release.barcodes)),
        "best_match_max": max(c.best_match for c in assignment),
        "best_match_strains": len({c.best_match for c in assignment}),
    }
    return section, release, assignment


def assignment_section(release: Any, assignment: Any) -> dict[str, Any]:
    """Cells per strain, the unassigned cells, and the per-record phenotype size."""
    cells = ng.assigned_cells(assignment)
    per = np.array([len(rows) for rows in cells.values()])
    n_assigned = int(per.sum())
    totals = release.cell_totals
    q3 = float(np.quantile(totals, 0.75))
    unassigned = [
        r for r, c in enumerate(assignment) if c.significant_best_match is None
    ]
    nonzero: list[int] = []
    json_bytes: list[int] = []
    for index in sorted(cells):
        summed = np.asarray(release.counts[cells[index]].sum(axis=0)).ravel()
        keyed = {release.genes[j]: int(summed[j]) for j in np.flatnonzero(summed)}
        nonzero.append(len(keyed))
        json_bytes.append(len(json.dumps(keyed)))
    all_rows = sorted(r for rows in cells.values() for r in rows)
    reference = np.asarray(release.counts[all_rows].sum(axis=0)).ravel()
    histogram = Counter(int(n) for n in per)
    return {
        "assigned_cells": n_assigned,
        "assigned_strains": len(cells),
        "mean_cells_per_strain": round(n_assigned / len(cells), 4),
        "cells_per_strain": _distribution(per),
        "cells_per_strain_histogram_1_to_10": {
            str(k): histogram[k] for k in range(1, 11)
        },
        "strains_with_more_than_ten_cells": int((per > 10).sum()),
        "unassigned_cells": len(unassigned),
        "unassigned_median_umi": float(np.median(totals[unassigned])),
        "assigned_median_umi": float(np.median(totals[all_rows])),
        "umi_third_quartile": q3,
        "unassigned_at_or_above_q3": int((totals[unassigned] >= q3).sum()),
        "record_nonzero_genes": _distribution(nonzero),
        "record_umi_count_json_bytes": _distribution(json_bytes),
        "record_umi_count_json_bytes_total": int(sum(json_bytes)),
        "reference_nonzero_genes": int((reference > 0).sum()),
        "reference_umis": int(reference.sum()),
    }


def dryad_section(mirror: str, cells: dict[int, list[int]]) -> dict[str, Any]:
    """The walled files' presence and, once deposited, the genotype cardinality."""
    missing = ng.missing_dryad_files(osp.dirname(osp.dirname(mirror)))
    section: dict[str, Any] = {
        "issue": ng.DRYAD_ISSUE,
        "missing": missing,
        "deposited": not missing,
    }
    if missing:
        return section
    rel = ng.raw_relpaths()
    manifest = ng.load_manifest(osp.dirname(osp.dirname(mirror)))
    for name in ng.DRYAD_FILES:
        path = osp.join(mirror, rel[name])
        if _sha256(path) != ng.manifest_sha256(manifest, rel[name]):
            raise RuntimeError(f"{path} does not match its manifest sha256")
    numbers = ng.read_batch_numbers(osp.join(mirror, rel[ng.SEGREGANT_INFO_NAME]))
    markers, order = ng.read_snp_markers(osp.join(mirror, rel[ng.SNP_LIST_NAME]))
    needed = {numbers[i] for i in cells}
    genotypes = ng.read_reference_genotypes(
        osp.join(mirror, rel[ng.GENO_NAME]), needed, len(markers)
    )
    by_step: dict[str, dict[str, Any]] = {}
    exact_half: list[float] = []
    uncertain: list[float] = []
    for number in sorted(genotypes):
        posterior = genotypes[number][order]
        exact_half.append(float(np.mean(posterior == 0.5)))
        uncertain.append(float(np.mean((posterior > 0.05) & (posterior < 0.95))))
    for step in STEPS:
        label = "called_parent_only" if step is None else str(step)
        counts: list[int] = []
        sizes: list[int] = []
        for number in sorted(genotypes):
            blocks = blocks_at(genotypes[number][order], markers, step)
            counts.append(len(blocks))
            sizes.append(len(pickle.dumps([b.model_dump() for b in blocks])))
        by_step[label] = {
            "blocks": _distribution(counts),
            "genotype_bytes": _distribution(sizes),
            "genotype_bytes_total": int(sum(sizes)),
        }
    section.update(
        {
            "batch1_strains": len(numbers),
            "snps": len(markers),
            "strains_read": len(genotypes),
            "fraction_exactly_half": _distribution(exact_half),
            "fraction_in_0.05_0.95": _distribution(uncertain),
            "encodings": by_step,
        }
    )
    return section


def main() -> None:
    """Measure and write the inventory JSON; print its headline numbers."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    mirror = str(ng.raw_mirror_dir(data_root))
    release_numbers, release, assignment = release_section(mirror)
    sgd = _sgd_gene_set(data_root)
    outside = sorted(set(release.genes) - sgd)
    release_numbers["genes_outside_sgd_s288c"] = len(outside)
    release_numbers["genes_outside_sgd_s288c_names"] = outside
    assigned = assignment_section(release, assignment)
    dryad = dryad_section(mirror, ng.assigned_cells(assignment))
    out = {
        "citation_key": ng.CITATION_KEY,
        "code_commit": ng.CODE_COMMIT,
        "release": release_numbers,
        "assignment": assigned,
        "dryad": dryad,
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

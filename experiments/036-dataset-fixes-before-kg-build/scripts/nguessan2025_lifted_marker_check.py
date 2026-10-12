# experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_lifted_marker_check.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_lifted_marker_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_lifted_marker_check.py
"""The two L4 genome-reference rules on the SNPs lifted to S288C R64-4-1 (#1050).

The stored N'Guessan 2025 records keep the released SNP positions (BY4742_fixed
coordinates), so the verifier's ``marker_reference`` and ``mat_fixation`` report a
finding on them. This script applies the same two rules to the DERIVED lifted table
(``nguessan2025_snp_liftover.py``), the coordinates modeling would use:

- ``marker_reference`` over every lifted SNP (not a sample): the fraction whose BY
  allele is the S288C R64-4-1 base at its lifted position;
- ``mat_fixation``: the mean, over the 2,951 assigned strains, of P(RM) at the lifted
  SNP nearest the verifier's S288C MAT coordinate (``MAT_POSITION``), with the
  verifier's tolerance.

Every input is checked against its pin (raw-mirror manifest, genomes registry). Writes
``experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_lifted_marker_check.json``.
Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), under slurm.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import nguessan2025 as ng
from torchcell.sequence.genome.registry import SGD_S288C_R64, resolve
from torchcell.verification import segregant_expression_reference_panel as v

OUT = "experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_lifted_marker_check.json"


def _pinned(mirror: Path, relpath: str, manifest: Any) -> Path:
    path = mirror / relpath
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    if digest.hexdigest() != ng.manifest_sha256(manifest, relpath):
        raise RuntimeError(f"{path} does not match its manifest sha256")
    return path


def main() -> None:
    """Apply the two rules to the lifted markers and write the result JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    mirror = ng.raw_mirror_dir(data_root)
    manifest = ng.load_manifest(data_root)
    rel = ng.raw_relpaths()
    snp_list = _pinned(mirror, rel[ng.SNP_LIST_NAME], manifest)
    table = _pinned(mirror, ng.LIFTED_TABLE_REL, manifest)
    markers, order, n_snps = ng.read_lifted_markers(snp_list, table)
    s288c = ng.s288c_chromosomes(
        resolve(SGD_S288C_R64, ng.S288C_FSA, data_root=data_root)
    )
    matches = sum(s288c[m.chromosome][m.position - 1] == m.ref for m in markers)
    assignment = ng.read_assignment(_pinned(mirror, rel[ng.ASSIGNMENT_TABLE], manifest))
    cells = ng.assigned_cells(assignment)
    numbers = ng.read_batch_numbers(
        _pinned(mirror, rel[ng.SEGREGANT_INFO_NAME], manifest)
    )
    geno = _pinned(mirror, rel[ng.GENO_NAME], manifest)
    genotypes = ng.read_reference_genotypes(geno, {numbers[i] for i in cells}, n_snps)
    mat = v._nearest_marker(markers, v.MAT_CHROMOSOME, v.MAT_POSITION)
    if mat is None:
        raise RuntimeError("no lifted SNP on the MAT chromosome")
    column = order[mat]
    mean_rm = float(np.mean([g[column] for g in genotypes.values()]))
    out = {
        "lifted_table_sha256": ng.LIFTED_TABLE_SHA256,
        "lifted_snps": len(markers),
        "released_snps": n_snps,
        "marker_reference": {
            "by_allele_is_s288c_base": matches,
            "fraction": round(matches / len(markers), 6),
            "bar": 0.95,
        },
        "mat_fixation": {
            "strains": len(genotypes),
            "nearest_lifted_snp": markers[mat].model_dump(),
            "mean_p_rm": round(mean_rm, 6),
            "tolerance": v.MAT_FIXATION_TOLERANCE,
            "fixed": min(mean_rm, 1 - mean_rm) <= v.MAT_FIXATION_TOLERANCE,
        },
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

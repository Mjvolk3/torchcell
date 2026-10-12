# experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_snp_liftover]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py
"""Lift the 41,594 Nguyen Ba 2022 SNPs from BY4742_fixed to S288C R64-4-1 (#1050).

The released positions are ``BY4742_fixed.fasta`` coordinates, not S288C (measured by
``nguessan2025_snp_coordinate_audit.py``). For every SNP of the deposited
``SNP_list.txt`` this takes the flanking sequence from ``BY4742_fixed`` at the released
position and finds its exact matches in S288C R64-4-1 on both strands
(:func:`torchcell.datasets.scerevisiae.nguessan2025.lift_snps`: flanks of 25, 50, 100
and 200 bp on each side, the SNP base excluded, a longer flank only while the shorter
one matches more than once). Every SNP is written, with its status (``lifted``,
``no_match``, ``multiple``, ``edge``), the flank length used and the hit count; none is
dropped.

Inputs, each checked against its pin before use: ``data/dryad/SNP_list.txt`` and
``data/bbqtl/BY4742_fixed.fasta`` against the raw-mirror manifest, and the S288C FASTA
through the genomes registry. Outputs:

- the lifted table, ``derived/SNP_list_lifted_S288C_R64-4-1.tsv`` in the raw mirror
  (``nguessan2025.LIFTED_TABLE_REL``), a derived artifact the loader pins and reads;
- ``experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json``:
  the input digests, the commit this ran at, the table's digest, and the counts.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), under slurm, after
committing, so the table's provenance names a clean commit::

    python experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import nguessan2025 as ng
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    load_genome_manifest,
    resolve,
)

OUT = "experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json"
SCRIPT = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py"
)
REPO = Path(__file__).resolve().parents[3]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _pinned(mirror: Path, relpath: str, manifest: Any) -> Path:
    path = mirror / relpath
    if _sha256(path) != ng.manifest_sha256(manifest, relpath):
        raise RuntimeError(f"{path} does not match its manifest sha256")
    return path


def summarize(rows: list[ng.LiftedSnp]) -> dict[str, Any]:
    """Counts by status and flank, strands, chromosome changes, base agreement."""
    lifted = [r for r in rows if r.status == ng.LiftStatus.lifted]
    per_chromosome: dict[str, dict[str, Any]] = {}
    for chrom in ng._CHROM_ORDER:
        on = [r for r in rows if r.chromosome == chrom]
        up = [r for r in on if r.status == ng.LiftStatus.lifted]
        offsets = np.array(
            [r.s288c_position - r.position for r in up if r.s288c_position]
        )
        per_chromosome[chrom] = {
            "snps": len(on),
            "lifted": len(up),
            "status": dict(Counter(r.status.value for r in on)),
            "by_allele_is_s288c_base": sum(r.s288c_base == r.by for r in up),
            "offset_min": int(offsets.min()) if len(offsets) else None,
            "offset_max": int(offsets.max()) if len(offsets) else None,
        }
    return {
        "snps": len(rows),
        "status": dict(Counter(r.status.value for r in rows)),
        "flank_bp_of_lifted": {
            str(k): v for k, v in sorted(Counter(r.flank_bp for r in lifted).items())
        },
        "hits_of_multiple": dict(
            Counter(
                r.hits for r in rows if r.status == ng.LiftStatus.multiple
            ).most_common(10)
        ),
        "strand_of_lifted": dict(Counter(r.s288c_strand for r in lifted)),
        "lifted_to_another_chromosome": sum(
            r.s288c_chromosome != r.chromosome for r in lifted
        ),
        "by_allele_is_s288c_base": sum(r.s288c_base == r.by for r in lifted),
        "lifted_unchanged_position": sum(
            r.s288c_position == r.position for r in lifted
        ),
        "per_chromosome": per_chromosome,
    }


def main() -> None:
    """Lift, write the table into the raw mirror, and write the summary JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    mirror = ng.raw_mirror_dir(data_root)
    manifest = ng.load_manifest(data_root)
    snp_rel = ng.raw_relpaths()[ng.SNP_LIST_NAME]
    snp_list = _pinned(mirror, snp_rel, manifest)
    by4742 = _pinned(mirror, ng.BY4742_REL, manifest)
    s288c = resolve(SGD_S288C_R64, ng.S288C_FSA, data_root=data_root)
    s288c_sha = (
        load_genome_manifest(SGD_S288C_R64, data_root).record(ng.S288C_FSA).sha256
    )
    markers, _ = ng.read_snp_markers(snp_list)
    snps = [
        (m.index, f"snp{m.column + 1}", m)
        for m in sorted(markers, key=lambda m: m.column)
    ]
    rows = ng.lift_snps(
        snps, ng.by4742_chromosomes(by4742), ng.s288c_chromosomes(s288c)
    )
    table = mirror / ng.LIFTED_TABLE_REL
    table.parent.mkdir(parents=True, exist_ok=True)
    staged = table.with_suffix(".tsv.new")
    ng.write_lifted_table(rows, staged)
    if table.exists() and _sha256(table) != _sha256(staged):
        raise RuntimeError(f"{table} exists with other content; refusing to overwrite")
    staged.replace(table)
    summary = summarize(rows)
    out: dict[str, Any] = {
        "script": SCRIPT,
        "commit": commit,
        "dirty_tracked_paths": dirty.splitlines(),
        "function": "torchcell.datasets.scerevisiae.nguessan2025.lift_snps",
        "params": {"flanks": list(ng.LIFT_FLANKS), "seed": ng.LIFT_SEED},
        "inputs": {
            snp_rel: ng.manifest_sha256(manifest, snp_rel),
            ng.BY4742_REL: ng.BY4742_SHA256,
            f"{SGD_S288C_R64}/{ng.S288C_FSA}": s288c_sha,
        },
        "table": {
            "path": ng.LIFTED_TABLE_REL,
            "bytes": table.stat().st_size,
            "sha256": _sha256(table),
        },
        "summary": summary,
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: out[k] for k in ("commit", "table")}, indent=2))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "per_chromosome"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

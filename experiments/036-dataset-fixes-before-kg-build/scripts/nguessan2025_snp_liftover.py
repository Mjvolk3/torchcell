# experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_snp_liftover]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py
"""A modeling-time lift of the Nguyen Ba 2022 SNPs from BY4742_fixed to S288C R64-4-1.

NOT part of the N'Guessan 2025 dataset: the loader stores the Dryad genotypes exactly
as released, positions included, and reads nothing written here (owner, 2026-10-12,
#1050). The released positions are ``BY4742_fixed.fasta`` coordinates (measured by
``nguessan2025_snp_coordinate_audit.py``); a model that needs S288C coordinates can
use this lift.

For every SNP of the deposited ``SNP_list.txt``, the 25, 50, 100 and 200 bp flanks on
each side of the released position (the SNP base excluded) are taken from
``BY4742_fixed.fasta`` and searched on both strands of every S288C R64-4-1 chromosome.
One exact match lifts the SNP; none is ``no_match``; several try the next flank and end
as ``multiple``; a flank off the chromosome is ``edge``. Every SNP is written with its
flank length and hit count. On the lifted SNPs the script then applies the verifier's
two L4 genome-reference rules (``marker_reference`` over every lifted SNP, and
``mat_fixation`` at the lifted SNP nearest the verifier's S288C MAT coordinate).

Inputs, each checked against its pin: ``SNP_list.txt``, ``segregant_info.txt``, the
assignment table and ``geno_data_1.txt.gz`` against the raw-mirror manifest; the S288C
FASTA through the genomes registry; ``BY4742_fixed.fasta`` against ``BY4742_SHA256``
(``--retrieve`` fetches it from ``arturrc/bbqtl_inference`` at ``BBQTL_COMMIT`` with
``torchcell.literature.retrieve.direct_url`` and writes its ``RetrievalRecord``
beside it). Outputs, outside the dataset's raw mirror:

- ``$DATA_ROOT/experiments/036-dataset-fixes-before-kg-build/nguessan2025_coordinates/``:
  ``BY4742_fixed.fasta`` and its retrieval record, the lifted table
  ``SNP_list_lifted_S288C_R64-4-1.tsv`` and its processing record;
- ``experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json``.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), under slurm, after
committing, so the processing record names a clean commit::

    python experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py --retrieve
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import nguessan2025 as ng
from torchcell.datasets.scerevisiae.bloom2019 import Marker
from torchcell.literature.manifest import (
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.retrieve import direct_url
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    load_genome_manifest,
    resolve,
)
from torchcell.verification import segregant_expression_reference_panel as v

OUT = "experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_liftover.json"
SCRIPT = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_liftover.py"
)
OUT_DIR_REL = "experiments/036-dataset-fixes-before-kg-build/nguessan2025_coordinates"
REPO = Path(__file__).resolve().parents[3]

BBQTL_REPO = "arturrc/bbqtl_inference"
BBQTL_COMMIT = "2b1c89d6a602a8001b3b18dd00e75a8c97950d9d"
BY4742_NAME = "BY4742_fixed.fasta"
BY4742_URL = (
    f"https://raw.githubusercontent.com/{BBQTL_REPO}/{BBQTL_COMMIT}/"
    f"1_genotype_inference/indexes/{BY4742_NAME}"
)
BY4742_SHA256 = "c2e18c61d0529cf44d5991702f7ee50a3d968b2fa8e97bbc65a35d90b83c6f4e"
S288C_FSA = "S288C_reference_sequence_R64-4-1_20230830.fsa"
TABLE_NAME = "SNP_list_lifted_S288C_R64-4-1.tsv"

#: Flank lengths tried, shortest first, on each side of the SNP. A longer flank is
#: tried only while the shorter one matches more than once: a window with no exact
#: match cannot gain one by growing. The seed is the ``LIFT_SEED`` bases next to the
#: SNP on the left of the window, so every exact match is among its seed's occurrences.
LIFT_FLANKS = (25, 50, 100, 200)
LIFT_SEED = 25
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
    "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
]  # fmt: skip
_CHROM_ORDER = {f"chr{r}": i for i, r in enumerate(_ROMAN)}


class LiftStatus(StrEnum):
    """How one SNP's flanking sequence matched S288C R64-4-1 (none is dropped)."""

    lifted = "lifted"
    no_match = "no_match"
    multiple = "multiple"
    edge = "edge"


class LiftedSnp(BaseModel):
    """One released SNP and where its BY4742_fixed flanks match S288C R64-4-1.

    ``chromosome``/``position``/``by``/``rm`` are the released row. ``flank_bp`` is the
    flank on each side at which the status was decided and ``hits`` the number of exact
    matches, both strands, at that flank. The SNP base is not part of the match, so the
    S288C base at the lifted position is an independent check. The ``s288c_*`` fields
    are set only for ``lifted``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    snp: str
    chromosome: str
    position: int
    by: str
    rm: str
    status: LiftStatus
    flank_bp: int
    hits: int
    s288c_chromosome: str | None
    s288c_position: int | None
    s288c_strand: Literal["+", "-"] | None
    s288c_base: str | None


TABLE_COLUMNS = tuple(LiftedSnp.model_fields)


def reverse_complement(sequence: str) -> str:
    """The reverse complement of an ACGTN sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


def _nuclear(path: str | Path, name_of: Any) -> dict[str, str]:
    """The 16 nuclear chromosomes of a FASTA, named by ``name_of(header)`` (None: skip)."""
    seqs: dict[str, list[str]] = {}
    current: list[str] | None = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                name = name_of(line)
                current = None
                if name is not None:
                    current = seqs[name] = []
            elif current is not None:
                current.append(line.strip())
    missing = sorted(set(_CHROM_ORDER) - set(seqs), key=_CHROM_ORDER.__getitem__)
    if missing:
        raise ValueError(f"{path}: no nuclear chromosome {missing}")
    return {name: "".join(parts).upper() for name, parts in seqs.items()}


def by4742_chromosomes(path: str | Path) -> dict[str, str]:
    """``BY4742_fixed.fasta``'s ``chr01``..``chr16`` as ``chrI``..``chrXVI``."""
    names = {f"chr{i:02d}": f"chr{r}" for i, r in enumerate(_ROMAN, 1)}
    return _nuclear(path, lambda line: names.get(line[1:].split()[0]))


def s288c_chromosomes(path: str | Path) -> dict[str, str]:
    """The SGD R64 FASTA's ``[chromosome=I]``.. records as ``chrI``.."""

    def name_of(line: str) -> str | None:
        match = re.search(r"\[chromosome=([IVX]+)\]", line)
        return None if match is None else f"chr{match.group(1)}"

    return _nuclear(path, name_of)


def _windows(
    source: str, center: int, flank: int
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """(left, right) flanks of ``center``, then those of its reverse complement."""
    if center - flank < 0 or center + 1 + flank > len(source):
        return None
    left = source[center - flank : center]
    right = source[center + 1 : center + 1 + flank]
    return (left, right), (reverse_complement(right), reverse_complement(left))


def lift_snps(
    snps: Sequence[tuple[int, str, Marker]],
    source: Mapping[str, str],
    target: Mapping[str, str],
    *,
    flanks: Sequence[int] = LIFT_FLANKS,
    seed: int = LIFT_SEED,
) -> list[LiftedSnp]:
    """Lift each ``(Index, SNP name, released Marker)`` by its flanks' unique match."""
    if seed > min(flanks) or list(flanks) != sorted(set(flanks)):
        raise ValueError(
            f"need seed <= min(flanks) and ascending flanks: {seed} {flanks}"
        )
    seeds: dict[str, list[tuple[str, int]]] = {}
    for _, _, marker in snps:
        windows = _windows(source[marker.chromosome], marker.position - 1, seed)
        for left, _ in windows or ():
            seeds.setdefault(left, [])
    for name, sequence in target.items():
        for start in range(len(sequence) - seed + 1):
            occurrences = seeds.get(sequence[start : start + seed])
            if occurrences is not None:
                occurrences.append((name, start + seed))
    out: list[LiftedSnp] = []
    for index, snp, marker in snps:
        center = marker.position - 1
        status, flank_bp, found = LiftStatus.edge, flanks[0], []
        for flank in flanks:
            windows = _windows(source[marker.chromosome], center, flank)
            if windows is None:
                break
            found = [
                (chrom, at, strand)
                for (left, right), strand in zip(windows, ("+", "-"))
                for chrom, at in seeds[left[-seed:]]
                if target[chrom][at - flank : at] == left
                and target[chrom][at + 1 : at + 1 + flank] == right
            ]
            flank_bp = flank
            status = (
                LiftStatus.lifted
                if len(found) == 1
                else LiftStatus.no_match
                if not found
                else LiftStatus.multiple
            )
            if status != LiftStatus.multiple:
                break
        hit = found[0] if status == LiftStatus.lifted else None
        base = None
        if hit is not None:
            forward = target[hit[0]][hit[1]]
            base = forward if hit[2] == "+" else reverse_complement(forward)
        out.append(
            LiftedSnp(
                index=index,
                snp=snp,
                chromosome=marker.chromosome,
                position=marker.position,
                by=marker.ref,
                rm=marker.alt,
                status=status,
                flank_bp=flank_bp,
                hits=len(found),
                s288c_chromosome=None if hit is None else hit[0],
                s288c_position=None if hit is None else hit[1] + 1,
                s288c_strand=None if hit is None else hit[2],  # type: ignore[arg-type]
                s288c_base=base,
            )
        )
    return out


def write_table(rows: Sequence[LiftedSnp], path: str | Path) -> None:
    """Tab-separated ``TABLE_COLUMNS``, empty for None."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        for row in rows:
            values = row.model_dump(mode="json")
            writer.writerow(
                "" if values[c] is None else values[c] for c in TABLE_COLUMNS
            )


def lifted_markers(rows: Sequence[LiftedSnp]) -> list[Marker]:
    """The lifted rows as S288C markers (alleles on the forward strand, geno column)."""
    return sorted(
        (
            Marker(
                chromosome=r.s288c_chromosome,
                position=r.s288c_position,
                ref=r.by if r.s288c_strand == "+" else reverse_complement(r.by),
                alt=r.rm if r.s288c_strand == "+" else reverse_complement(r.rm),
                index=r.index,
                column=int(r.snp.removeprefix("snp")) - 1,
            )
            for r in rows
            if r.status == LiftStatus.lifted
            and r.s288c_chromosome is not None
            and r.s288c_position is not None
        ),
        key=lambda m: (_CHROM_ORDER[m.chromosome], m.position),
    )


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


def retrieve_by4742(out_dir: Path) -> Path:
    """Fetch ``BY4742_fixed.fasta`` at the pinned commit, check it, record how."""
    path = out_dir / BY4742_NAME
    data = direct_url(BY4742_URL)
    digest = hashlib.sha256(data).hexdigest()
    if digest != BY4742_SHA256:
        raise RuntimeError(f"{BY4742_URL} served sha256 {digest}, not {BY4742_SHA256}")
    path.write_bytes(data)
    record = RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=BY4742_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": BY4742_URL, "repo": BBQTL_REPO, "commit": BBQTL_COMMIT},
        sha256=digest,
        retrieved_at=datetime.now(UTC).date().isoformat(),
    )
    (out_dir / f"{BY4742_NAME}.retrieval.json").write_text(
        record.model_dump_json(indent=2) + "\n"
    )
    return path


def summarize(rows: list[LiftedSnp]) -> dict[str, Any]:
    """Counts by status and flank, strands, chromosome changes, base agreement."""
    lifted = [r for r in rows if r.status == LiftStatus.lifted]
    per_chromosome: dict[str, dict[str, Any]] = {}
    for chrom in _CHROM_ORDER:
        on = [r for r in rows if r.chromosome == chrom]
        up = [r for r in on if r.status == LiftStatus.lifted]
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
            str(k): n for k, n in sorted(Counter(r.flank_bp for r in lifted).items())
        },
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


def l4_on_lifted(
    markers: list[Marker],
    s288c: Mapping[str, str],
    genotypes: Mapping[int, np.ndarray[Any, Any]],
) -> dict[str, Any]:
    """The verifier's two L4 genome-reference rules applied to the lifted SNPs."""
    matches = sum(s288c[m.chromosome][m.position - 1] == m.ref for m in markers)
    mat = v._nearest_marker(markers, v.MAT_CHROMOSOME, v.MAT_POSITION)
    if mat is None:
        raise RuntimeError("no lifted SNP on the MAT chromosome")
    mean_rm = float(np.mean([g[markers[mat].column] for g in genotypes.values()]))
    return {
        "marker_reference": {
            "lifted_snps": len(markers),
            "by_allele_is_s288c_base": matches,
            "fraction": round(matches / len(markers), 6),
        },
        "mat_fixation": {
            "strains": len(genotypes),
            "nearest_lifted_snp": markers[mat].model_dump(),
            "mean_p_rm": round(mean_rm, 6),
            "tolerance": v.MAT_FIXATION_TOLERANCE,
            "fixed": min(mean_rm, 1 - mean_rm) <= v.MAT_FIXATION_TOLERANCE,
        },
    }


def main() -> None:
    """Lift, check the two rules on the lift, write the table and the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--retrieve", action="store_true", help="fetch BY4742_fixed")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no").splitlines()
    out_dir = Path(data_root) / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    by4742 = retrieve_by4742(out_dir) if args.retrieve else out_dir / BY4742_NAME
    if _sha256(by4742) != BY4742_SHA256:
        raise RuntimeError(f"{by4742} does not hash to {BY4742_SHA256}")
    mirror = ng.raw_mirror_dir(data_root)
    manifest = ng.load_manifest(data_root)
    rel = ng.raw_relpaths()
    snp_list = _pinned(mirror, rel[ng.SNP_LIST_NAME], manifest)
    s288c_path = resolve(SGD_S288C_R64, S288C_FSA, data_root=data_root)
    s288c_sha = load_genome_manifest(SGD_S288C_R64, data_root).record(S288C_FSA).sha256
    released, _ = ng.read_snp_markers(snp_list)
    snps = [
        (m.index, f"snp{m.column + 1}", m)
        for m in sorted(released, key=lambda m: m.column)
    ]
    s288c = s288c_chromosomes(s288c_path)
    rows = lift_snps(snps, by4742_chromosomes(by4742), s288c)
    table = out_dir / TABLE_NAME
    write_table(rows, table)
    inputs = {
        rel[ng.SNP_LIST_NAME]: ng.manifest_sha256(manifest, rel[ng.SNP_LIST_NAME]),
        BY4742_NAME: BY4742_SHA256,
        f"{SGD_S288C_R64}/{S288C_FSA}": s288c_sha,
    }
    processing = ProcessingRecord(
        processor=f"{SCRIPT} (lift_snps)",
        tool="exact flank match, both strands",
        version=commit,
        params={"flanks": list(LIFT_FLANKS), "seed": LIFT_SEED},
        input_sha256=list(inputs.values()),
    )
    (out_dir / f"{TABLE_NAME}.processing.json").write_text(
        processing.model_dump_json(indent=2) + "\n"
    )
    assignment = ng.read_assignment(_pinned(mirror, rel[ng.ASSIGNMENT_TABLE], manifest))
    cells = ng.assigned_cells(assignment)
    numbers = ng.read_batch_numbers(
        _pinned(mirror, rel[ng.SEGREGANT_INFO_NAME], manifest)
    )
    genotypes = ng.read_reference_genotypes(
        _pinned(mirror, rel[ng.GENO_NAME], manifest),
        {numbers[i] for i in cells},
        len(released),
    )
    summary = summarize(rows)
    out: dict[str, Any] = {
        "script": SCRIPT,
        "commit": commit,
        "dirty_tracked_paths": dirty,
        "params": {"flanks": list(LIFT_FLANKS), "seed": LIFT_SEED},
        "inputs": inputs,
        "by4742_url": BY4742_URL,
        "table": {
            "path": f"$DATA_ROOT/{OUT_DIR_REL}/{TABLE_NAME}",
            "bytes": table.stat().st_size,
            "sha256": _sha256(table),
        },
        "summary": summary,
        "l4_on_lifted_snps": l4_on_lifted(lifted_markers(rows), s288c, genotypes),
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    shown = {k: out[k] for k in ("commit", "table", "l4_on_lifted_snps")}
    print(json.dumps(shown, indent=2))
    print(
        json.dumps(
            {k: n for k, n in summary.items() if k != "per_chromosome"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

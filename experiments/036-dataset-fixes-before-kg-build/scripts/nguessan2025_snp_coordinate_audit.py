# experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_coordinate_audit.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_snp_coordinate_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_coordinate_audit.py
"""Which reference the Nguyen Ba 2022 SNP positions sit on, and where MAT is in them.

The N'Guessan 2025 store (slurm job 4199) failed L4 ``marker_reference`` (0.6740 of 500
sampled SNPs carry the S288C base as their BY allele) and ``mat_fixation`` (mean P(RM)
0.1097 at the SNP nearest chrIII:200,000), although the Dryad README says the SNPs are
on ``S288C_reference_genome_R64-3-1_20210421``. This script measures, over all 41,594
SNPs of the deposited ``SNP_list.txt`` (sha256-checked against the raw-mirror
manifest):

- per chromosome, the fraction whose ``BY allele`` equals the base at the stated
  1-based position of S288C R64-4-1 (genomes tier ``sgd_S288C_R64-4-1_20230830``) and
  of ``BY4742_fixed.fasta``, the reference the BB-QTL genotyping code indexes
  (``arturrc/bbqtl_inference`` @ ``2b1c89d6``, ``1_genotype_inference/indexes/``; #1015);
- per chromosome, the piecewise offset that best aligns consecutive windows of 15 SNPs
  to S288C (search range +-6,000 bp; a window whose best score is under 13 of 15 is
  reported as unresolved);
- along chrIII, the mean P(RM) of the built store's 2,951 records (blocks expanded to
  the SNPs, so within the 0.025 grid error) per 20 kb window, the SNPs where it is
  lowest, and the SNP nearest 200,000, the S288C MAT coordinate the verifier reads.

``BY4742_fixed.fasta`` is not in a mirror. It is read from ``--by4742`` and must hash to
``BY4742_SHA256``; retrieve it with::

    curl -sSfL -o BY4742_fixed.fasta https://raw.githubusercontent.com/arturrc/bbqtl_inference/2b1c89d6a602a8001b3b18dd00e75a8c97950d9d/1_genotype_inference/indexes/BY4742_fixed.fasta

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``), under slurm::

    python experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_snp_coordinate_audit.py --by4742 <path>

Writes ``experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_coordinate_audit.json``.
"""

from __future__ import annotations

import argparse
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
from torchcell.verification import runners
from torchcell.verification.segregant_perturbation import read_fasta_by_chromosome

OUT = "experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_snp_coordinate_audit.json"
BY4742_URL = (
    "https://raw.githubusercontent.com/arturrc/bbqtl_inference/"
    "2b1c89d6a602a8001b3b18dd00e75a8c97950d9d/1_genotype_inference/indexes/"
    "BY4742_fixed.fasta"
)
BY4742_SHA256 = "c2e18c61d0529cf44d5991702f7ee50a3d968b2fa8e97bbc65a35d90b83c6f4e"
S288C_FSA = "S288C_reference_sequence_R64-4-1_20230830.fsa"
WINDOW = 15
MIN_WINDOW_SCORE = 13
OFFSET_RANGE = 6000
MAT_POSITION = 200_000


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_by4742(path: str) -> dict[int, str]:
    """``BY4742_fixed.fasta`` keyed by chromosome number (headers ``chr01``..``chr16``)."""
    seqs: dict[int, list[str]] = {}
    current: int | None = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                name = line[1:].split()[0]
                current = int(name[3:]) if name[3:].isdigit() else None
                if current is not None:
                    seqs[current] = []
            elif current is not None:
                seqs[current].append(line.strip())
    return {k: "".join(v).upper() for k, v in seqs.items()}


def agreement(markers: list[Any], seqs: dict[int, str]) -> dict[str, dict[str, Any]]:
    """Per chromosome: SNPs and the fraction whose BY allele is the stated base."""
    out: dict[str, dict[str, Any]] = {}
    for chrom, number in ng.CHROMOSOME_TOKEN.items():
        seq = seqs[int(chrom)].upper()
        on = [m for m in markers if m.chromosome == number]
        hits = sum(seq[m.position - 1] == m.ref for m in on)
        out[number] = {"snps": len(on), "by_allele_matches": hits}
        out[number]["fraction"] = round(hits / len(on), 4)
    return out


def offsets(
    markers: list[Any], seqs: dict[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Runs of windows sharing the offset that best aligns their BY alleles to S288C."""
    shifts = np.arange(-OFFSET_RANGE, OFFSET_RANGE + 1)
    out: dict[str, list[dict[str, Any]]] = {}
    for chrom, number in ng.CHROMOSOME_TOKEN.items():
        seq = np.frombuffer(seqs[int(chrom)].upper().encode(), dtype=np.uint8)
        on = [m for m in markers if m.chromosome == number]
        pos = np.array([m.position - 1 for m in on])
        base = np.frombuffer("".join(m.ref for m in on).encode(), dtype=np.uint8)
        runs: list[dict[str, Any]] = []
        for i in range(0, len(on) - WINDOW + 1, WINDOW):
            idx = pos[i : i + WINDOW, None] + shifts[None, :]
            valid = (idx >= 0) & (idx < len(seq))
            hit = valid & (
                seq[np.clip(idx, 0, len(seq) - 1)] == base[i : i + WINDOW, None]
            )
            score = hit.sum(axis=0)
            best = int(score.max())
            shift: int | None = (
                int(shifts[score.argmax()]) if best >= MIN_WINDOW_SCORE else None
            )
            start, end = int(pos[i]) + 1, int(pos[i + WINDOW - 1]) + 1
            if runs and runs[-1]["offset"] == shift:
                runs[-1]["end"] = end
                runs[-1]["snps"] += WINDOW
            else:
                runs.append(
                    {"start": start, "end": end, "offset": shift, "snps": WINDOW}
                )
        out[number] = runs
    return out


def mat_profile(store: str, markers: list[Any]) -> dict[str, Any]:
    """Mean P(RM) along chrIII over the store's records, from their expanded blocks."""
    chr3 = [m for m in markers if m.chromosome == "chrIII"]
    pos = np.array([m.position for m in chr3])
    total = np.zeros(len(chr3))
    records = 0
    for record in runners.stream_records(store):
        p_rm = np.full(len(chr3), np.nan)
        for block in record["experiment"]["genotype"]["blocks"]:
            if block["chromosome"] != "chrIII":
                continue
            sel = (pos >= block["start"]) & (pos <= block["end"])
            level = block["posterior"]
            p_rm[sel] = level if block["parent"] == 2 else 1 - level
        if np.isnan(p_rm).any():
            raise ValueError("a chrIII SNP is outside every block of a record")
        total += p_rm
        records += 1
    mean = total / records
    lowest = sorted(np.argsort(mean)[:5].tolist())
    nearest = int(np.argmin(np.abs(pos - MAT_POSITION)))
    windows = []
    for lo in range(0, int(pos.max()) + 20_000, 20_000):
        sel = (pos >= lo) & (pos < lo + 20_000)
        if sel.any():
            windows.append(
                {
                    "from": lo,
                    "to": lo + 20_000,
                    "snps": int(sel.sum()),
                    "min": round(float(mean[sel].min()), 4),
                    "mean": round(float(mean[sel].mean()), 4),
                }
            )
    return {
        "records": records,
        "chrIII_snps": len(chr3),
        "lowest": [
            {"position": int(pos[i]), "mean_p_rm": round(float(mean[i]), 4)}
            for i in lowest
        ],
        "nearest_to_200000": {
            "position": int(pos[nearest]),
            "mean_p_rm": round(float(mean[nearest]), 4),
        },
        "windows_20kb": windows,
    }


def main() -> None:
    """Measure and write the audit JSON; print the per-chromosome agreement."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--by4742", required=True, help="BY4742_fixed.fasta path")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if _sha256(args.by4742) != BY4742_SHA256:
        raise RuntimeError(f"{args.by4742} does not hash to {BY4742_SHA256}")
    mirror = ng.raw_mirror_dir(data_root)
    rel = ng.raw_relpaths()[ng.SNP_LIST_NAME]
    snp_path = mirror / rel
    if _sha256(snp_path) != ng.manifest_sha256(ng.load_manifest(data_root), rel):
        raise RuntimeError(f"{snp_path} does not match its manifest sha256")
    markers, _ = ng.read_snp_markers(snp_path)
    s288c = read_fasta_by_chromosome(
        resolve(SGD_S288C_R64, S288C_FSA, data_root=data_root)
    )
    by4742 = read_by4742(args.by4742)
    on_s288c = agreement(markers, s288c)
    on_by4742 = agreement(markers, by4742)
    out = {
        "snp_list_sha256": ng.DRYAD_FILES[ng.SNP_LIST_NAME].sha256,
        "s288c": {"assembly_set": SGD_S288C_R64, "file": S288C_FSA},
        "by4742_fixed": {"url": BY4742_URL, "sha256": BY4742_SHA256},
        "snps": len(markers),
        "by_allele_on_s288c": {
            "total": sum(v["by_allele_matches"] for v in on_s288c.values()),
            "per_chromosome": on_s288c,
        },
        "by_allele_on_by4742_fixed": {
            "total": sum(v["by_allele_matches"] for v in on_by4742.values()),
            "per_chromosome": on_by4742,
        },
        "offsets_to_s288c": {
            "window_snps": WINDOW,
            "min_window_score": MIN_WINDOW_SCORE,
            "search_range_bp": OFFSET_RANGE,
            "runs": offsets(markers, s288c),
        },
        "chrIII_mean_p_rm": mat_profile(
            osp.join(data_root, ng.VERIFICATION_SPEC["root"]), markers
        ),
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    for number in on_s288c:
        print(
            f"{number:8s} snps {on_s288c[number]['snps']:5d}  S288C "
            f"{on_s288c[number]['fraction']:.4f}  BY4742_fixed "
            f"{on_by4742[number]['fraction']:.4f}"
        )


if __name__ == "__main__":
    main()

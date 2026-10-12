# tests/torchcell/datasets/scerevisiae/_nguessan2025_release.py
"""The synthetic N'Guessan 2025 release the loader and verifier tests build.

Every file is written in the released format the loader reads, so nothing is patched
but the release-size constants. The GitHub side:

    matrix part 1 (AAAC-1, AAAG-1) and part 2 (AAAT-1, AACA-1, AACC-1), genes
    YAL001C YAL002W YCL001W YFL026W:

    cell    counts    best  significant
    AAAC-1  2 0 1 1   1     1
    AAAG-1  0 3 0 2   0     0
    AAAT-1  1 1 0 1   1     1
    AACA-1  5 0 0 0   2     NA   <- refused, no significant match
    AACC-1  0 0 4 1   2     2

The Dryad side, in the deposited layouts: ``segregant_info.txt`` rows Number 10 (batch
1), 20 (batch 2), 11, 12 (batch 1), so batch-1 index 0, 1, 2 is strain 10, 11, 12.
``SNP_list.txt`` lists four SNPs out of order (snp1 chromosome 1:300, snp2 1:100, snp3
3:150, snp4 1:200), then the ``END`` row of chromosomes 1 and 3; sorted order [1, 3, 0,
2]. ``geno_data_1.txt.gz`` posteriors P(RM) by geno column 0..3 (snp1..snp4):

    10: .20 .99 .98 .97 -> sorted .99 .97 .20 .98 -> parents 2 2 1 2, levels
        1.0 .95 .8 1.0 -> 4 blocks
    11: .50 .01 .99 .02 -> sorted .01 .02 .50 .99 -> parents 1 1 1(tie) 2, levels
        1.0 1.0 .5 1.0 -> 3 blocks (chrI 100..200 one run)
    12: 0 0 1 0         -> sorted 0 0 0 1 -> 2 blocks
    20: batch 2, never read

Records (batch-1 index order): 10 = AAAG-1 {YAL002W: 3, YFL026W: 2} total 5; 11 =
AAAC-1 + AAAT-1 {YAL001C: 3, YAL002W: 1, YCL001W: 1, YFL026W: 2} total 7; 12 = AACC-1
{YCL001W: 4, YFL026W: 1} total 5. Reference: the four assigned cells, {YAL001C: 3,
YAL002W: 4, YCL001W: 5, YFL026W: 5} total 17. Gene set: YAL001C (chrI 120-180) and
YCL001W (chrIII 140-160) lie in the marker span; YAL002W (chrI 400-500) and YFL026W
(chrVI) do not.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import pickle
import tarfile
from pathlib import Path
from typing import Any, cast

import lmdb
import pytest

from tests.torchcell.datasets.scerevisiae._boocock2025_release import (
    _Feature,
    _write_genomes_tier,
)
from torchcell.datasets.scerevisiae import nguessan2025 as m
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

GENES = ["YAL001C", "YAL002W", "YCL001W", "YFL026W"]
PART_1 = [("AAAC-1", [2, 0, 1, 1]), ("AAAG-1", [0, 3, 0, 2])]
PART_2 = [("AAAT-1", [1, 1, 0, 1]), ("AACA-1", [5, 0, 0, 0]), ("AACC-1", [0, 0, 4, 1])]
ASSIGNMENT = [
    ("AAAC-1", 1, "1"),
    ("AAAG-1", 0, "0"),
    ("AAAT-1", 1, "1"),
    ("AACA-1", 2, "NA"),
    ("AACC-1", 2, "2"),
]
SEGREGANT_INFO = [(10, 1), (20, 2), (11, 1), (12, 1)]
SNPS = [
    (0, "snp1", 1, 300, "G", "C"),
    (1, "snp2", 1, 100, "A", "T"),
    (2, "snp3", 3, 150, "A", "G"),
    (3, "snp4", 1, 200, "C", "G"),
    (4, "END", 1, 0, "0", "0"),
    (5, "END", 3, 0, "0", "0"),
]
GENO = {
    10: [0.20, 0.99, 0.98, 0.97],
    20: [0.40, 0.40, 0.40, 0.40],
    11: [0.50, 0.01, 0.99, 0.02],
    12: [0.0, 0.0, 1.0, 0.0],
}


def matrix_csv(rows: list[tuple[str, list[int]]], genes: list[str] = GENES) -> bytes:
    """A released matrix CSV: the header names the genes, rows are barcode + counts."""
    lines = [",".join(f'"{g}"' for g in genes)]
    lines += [f'"{b}",' + ",".join(str(c) for c in counts) for b, counts in rows]
    return ("\n".join(lines) + "\n").encode()


def write_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    """A ``.tar.gz`` holding ``members`` (name -> bytes)."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def gene_table(genes: list[str] = GENES) -> str:
    """``Table_expressed_genes.csv`` (tab-separated, ``_CDS`` names)."""
    original = [0, 1, 3, 5][: len(genes)]
    rows = ["Expressed_gene_original_index\tsc_eQTL_gene_id\tGene_name"]
    rows += [f"{o}\t{i}\t{g}_CDS" for i, (o, g) in enumerate(zip(original, genes))]
    return "\n".join(rows) + "\n"


def assignment_csv(rows: list[tuple[str, int, str]] = ASSIGNMENT) -> str:
    """The barcode-to-strain table (quoted barcode, NA for no significant match)."""
    lines = ['"single_cell_barcode","best_match","significant_best_match"']
    lines += [f'"{b}",{best},{sig}' for b, best, sig in rows]
    return "\n".join(lines) + "\n"


def segregant_info(rows: list[tuple[int, int]] = SEGREGANT_INFO) -> str:
    """``segregant_info.txt`` with the deposited header (Number ... Barcode2)."""
    lines = ["Number\tIndex\tBatch\tSet\tPlate\tWell\tBarcode1\tBarcode2"]
    lines += [
        f"{n}\t{i + 1}\t{b}\t1\t1\tA{i + 1}\tACGT\tTGCA"
        for i, (n, b) in enumerate(rows)
    ]
    return "\n".join(lines) + "\n"


def snp_list(rows: list[tuple[int, str, int, int, str, str]] = SNPS) -> str:
    """``SNP_list.txt`` as deposited: a quoted header, tab-separated rows."""
    lines = ["\t".join(f'"{c}"' for c in m.SNP_LIST_COLUMNS)]
    lines += ["\t".join(str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def lifted_table(
    path: Path, rows: list[tuple[int, str, int, int, str, str]] = SNPS
) -> None:
    """A lifted SNP table of ``rows``: every SNP lifted to its own position, + strand.

    The derived artifact the deposit records (#1050); the records do not read it, and
    the lift itself is tested on its own in ``test_nguessan2025.py``.
    """
    m.write_lifted_table(
        [
            m.LiftedSnp(
                index=index,
                snp=snp,
                chromosome=m.CHROMOSOME_TOKEN[str(chrom)],
                position=position,
                by=by,
                rm=rm,
                status=m.LiftStatus.lifted,
                flank_bp=25,
                hits=1,
                s288c_chromosome=m.CHROMOSOME_TOKEN[str(chrom)],
                s288c_position=position,
                s288c_strand="+",
                s288c_base=by,
            )
            for index, snp, chrom, position, by, rm in rows
            if snp != m.SNP_END
        ],
        path,
    )


def geno_gz(path: Path, rows: dict[int, list[float]] = GENO) -> None:
    """``geno_data_1.txt.gz``: Number then one posterior per SNP index, tab-separated."""
    with gzip.open(path, "wt") as handle:
        for number, values in rows.items():
            handle.write("\t".join([str(number), *(repr(v) for v in values)]) + "\n")


def write_release(raw: Path) -> None:
    """Every file the loader reads, under ``raw/`` with its loader name."""
    raw.mkdir(parents=True, exist_ok=True)
    write_tar_gz(raw / m.EXPRESSION_PARTS[0], {"part1.csv": matrix_csv(PART_1)})
    write_tar_gz(raw / m.EXPRESSION_PARTS[1], {"part2.csv": matrix_csv(PART_2)})
    (raw / m.GENE_TABLE).write_text(gene_table())
    (raw / m.ASSIGNMENT_TABLE).write_text(assignment_csv())
    (raw / m.README_NAME).write_text("Dryad README placeholder\n")
    (raw / m.SNP_LIST_NAME).write_text(snp_list())
    (raw / m.SEGREGANT_INFO_NAME).write_text(segregant_info())
    geno_gz(raw / m.GENO_NAME)


def sha256(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_mirror_manifest(data_root: Path, raw: Path) -> None:
    """A raw-mirror manifest pinning every ``raw/`` file at its mirror path."""
    rel = m.raw_relpaths()
    mirror = data_root / "torchcell-raw" / m.CITATION_KEY
    mirror.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        citation_key=m.CITATION_KEY,
        files=[
            ArtifactRecord(
                path=rel[name],
                role=ROLE_RAW_DATA,
                bytes=(raw / name).stat().st_size,
                sha256=sha256(raw / name),
            )
            for name in rel
        ],
        provenance_complete=True,
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json(indent=2))


class StubGenome:
    """``db``/``gene_set`` for the gene set and cis loci; ``fasta_dna``/``chr_to_nc``
    for the marker reference check (chrI 100 A, 200 C, 300 G; chrIII 150 A).
    """

    gene_set = ["YAL001C", "YAL002W", "YCL001W", "YFL026W"]
    db = {
        "YAL001C": _Feature(chrom="chrI", start=120, end=180),
        "YAL002W": _Feature(chrom="chrI", start=400, end=500),
        "YCL001W": _Feature(chrom="chrIII", start=140, end=160),
        "YFL026W": _Feature(chrom="chrVI", start=10, end=20),
    }
    chr_to_nc = {1: "chrI", 3: "chrIII"}

    class _Seq:
        def __init__(self, seq: str) -> None:
            self.seq = seq

    fasta_dna = {
        "chrI": _Seq("N" * 99 + "A" + "N" * 99 + "C" + "N" * 99 + "G"),
        "chrIII": _Seq("N" * 149 + "A"),
    }


def genome() -> SCerevisiaeGenome:
    """The stub genome, typed as the loader expects."""
    return cast(SCerevisiaeGenome, StubGenome())


def patch_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the release-size constants at the synthetic release."""
    monkeypatch.setattr(m, "N_BARCODES", 5)
    monkeypatch.setattr(m, "N_GENES", 4)
    monkeypatch.setattr(m, "N_ASSIGNED_CELLS", 4)
    monkeypatch.setattr(m, "N_SEGREGANTS", 3)
    monkeypatch.setattr(m, "BATCH1_STRAINS", 3)


def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> m.Nguessan2025Dataset:
    """Write the release, its manifest and the genomes tier; build the store."""
    patch_sizes(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "nguessan2025"
    write_release(root / "raw")
    write_mirror_manifest(data_root, root / "raw")
    _write_genomes_tier(data_root)
    return m.Nguessan2025Dataset(root=str(root), genome=genome())


def raw_records(root: str | Path) -> list[dict[str, Any]]:
    """Every stored record, unpickled and unresolved, in key order."""
    env = lmdb.open(str(Path(root) / "processed" / "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        records = [pickle.loads(v) for _, v in txn.cursor()]
    env.close()
    return records

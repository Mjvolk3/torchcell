# tests/torchcell/datasets/scerevisiae/_boocock2025_release.py
"""The synthetic two-panel Boocock 2025 release the loader and verifier tests build.

The release is R ``.RDS`` / ``.RData``, read through ``rdata``; ``rdata.read_rds`` and
``rdata.read_rda`` are repointed with ``monkeypatch`` to return, per file name, the
objects ``rdata`` itself returns for the real files (a ``SimpleNamespace`` dgCMatrix with
``i``/``p``/``x``/``Dim``/``Dimnames``, an ``xarray.DataArray`` for ``Gsub`` and
``gdata``, pandas frames for the tables). The raw files on disk are placeholder bytes the
manifest pins, so ``verify_raw_files`` runs unchanged. ``PANELS``,
``N_POOLED_SEGREGANTS`` and ``EXPECTED_RECORDS`` are repointed at the fixture.

Panel ``P`` (pooled, cross A, BY x RM, libraries LP1 and LP2), transcripts YAL001C,
YAL002W, YBL001C:

    cell  library barcode  counts   nUMI stage  matched
    c1    LP1     AAAC-1   2 0 1    5    G1     S2
    c2    LP1     AAAG-1   0 3 0    4    S      S1
    c3    LP2     AAAC-1   1 1 0    3    G1     S2

so S1 = {YAL002W: 3}, total 4, 1 cell (S); S2 = {YAL001C: 3, YAL002W: 1, YBL001C: 1},
total 8, 2 cells (G1, G1; LP1, LP2). Lookup genotypes at chrI:100, chrI:300, chrIII:150:
S1 = -1 -1 -1 (one BY block per chromosome, 2 blocks), S2 = +1 -1 -1 (3 blocks).

Panel ``D`` (de novo, cross B, YJM145 x YPS163, library LD), transcripts YAL001C, YCL001W,
posterior P(parent 2) at chrI 100, 200, 300 and chrIII 150:

    d1  TTTA-1  counts 4 0  nUMI 6  G2/M  posterior .99 .97 .20 .90
        -> parents 2 2 1 2, levels 1.0 0.95 0.8 0.9 -> 4 blocks
    d2  TTTC-1  counts 1 2  nUMI 3  M/G1  posterior .01 .02 .60 .30
        -> parents 1 1 2 1, levels 1.0 1.0 0.6 0.7 -> 3 blocks (100..200 one run)

Records: S1, S2, LD:TTTA-1, LD:TTTC-1. Gene set: stub genes YAL001C (chrI 120-180) and
YCL001W (chrIII 140-160) lie in a marker span, YAL002W (chrI 400-500) and YDL001W
(chrIV) do not.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lmdb
import numpy as np
import openpyxl
import pandas as pd
import pytest
import scipy.sparse as sp
import xarray as xr

from torchcell.datasets.scerevisiae import boocock2025 as m
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import GenomeManifest
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_TAR_SHA = "53540d095958ae8c32509c04485f0d2d0948069c7647f828698d611899a9b4da"
_S288C_SHA = "dbf065ffc3f5bbaa7554ef53c6861399eecc597fe058df5f81ba557944e3f86b"
_INDEX_LINES = (
    "AAA\tGENOMES_ASSEMBLED/AAA_6.re.fa\nAAS\tGENOMES_ASSEMBLED/AAS_3.re.fa\n"
)
_INDEX_LINES += "ACD\tGENOMES_ASSEMBLED/ACD_7.re.fa\n"
_BY = "BY MatA (YLK1879)"
_RM = "RM MatAlpha AMN1-BY ho∆::HphMX flo8∆::NatMX (YLK1950)"
_YPS = "YPS163 MatA ho∆::HphMX flo8∆::NatMX (YLK2438)"
_YJM = "YJM145 MatAlpha ho∆::HphMX flo8∆::NatMX (YLK2436)"

_PANELS = {
    "P": m.PanelSpec(
        panel="P",
        cross="A",
        s4_label="A*",
        strain="YLKP",
        pooled=True,
        parent_1="BY",
        parent_2="RM",
        mata_parent=1,
        libraries=("LP1", "LP2"),
        expected_cells=3,
        expected_transcripts=3,
        median_umi=4.0,
    ),
    "D": m.PanelSpec(
        panel="D",
        cross="B",
        s4_label="B",
        strain="YLKD",
        pooled=False,
        parent_1="YJM145",
        parent_2="YPS163",
        mata_parent=2,
        libraries=("LD",),
        expected_cells=2,
        expected_transcripts=2,
        median_umi=4.5,
    ),
}


def _dgc(dense: list[list[int]], cells: list[str], genes: list[str]) -> Any:
    csc = sp.csc_matrix(np.asarray(dense, dtype=np.float64))
    return SimpleNamespace(
        i=csc.indices.astype(np.int32),
        p=csc.indptr.astype(np.int32),
        x=csc.data,
        Dim=np.asarray([len(cells), len(genes)], dtype=np.int32),
        Dimnames=[np.asarray(cells), np.asarray(genes)],
        factors=[],
    )


def _gsub(values: list[list[float]], cells: list[str], markers: list[str]) -> Any:
    return xr.DataArray(
        np.asarray(values),
        dims=("dim_0", "dim_1"),
        coords={"dim_0": cells, "dim_1": markers},
    )


def _seg_data(
    cells: list[str],
    libraries: list[str],
    genes: list[str],
    dense: list[list[int]],
    n_umi: list[float],
    stages: list[str],
    markers: list[str],
    posterior: list[list[float]],
) -> dict[str, Any]:
    return {
        "Yr": _dgc(dense, cells, genes),
        "Gsub": _gsub(posterior, cells, markers),
        "barcode.features": pd.DataFrame({"barcode": cells, "nUMI": n_umi}),
        "cell.cycle.df": pd.DataFrame(
            {
                "named_dataset": libraries,
                "X": [float(i + 1) for i in range(len(cells))],
                "cell_name": cells,
                "cell_cycle": stages,
                "seurat_clusters": ["0"] * len(cells),
            }
        ),
    }


_P_MARKERS = ["chrI_100_A_T_1", "chrI_200_C_G_2", "chrIII_150_A_G_3"]
_D_MARKERS = ["chrI_100_A_T_1", "chrI_200_C_G_2", "chrI_300_G_C_3", "chrIII_150_A_G_4"]
_RELEASE: dict[str, Any] = {
    "segData_P.RDS": _seg_data(
        ["AAAC-1", "AAAG-1", "AAAC-1"],
        ["LP1", "LP1", "LP2"],
        ["YAL001C", "YAL002W", "YBL001C"],
        [[2, 0, 1], [0, 3, 0], [1, 1, 0]],
        [5.0, 4.0, 3.0],
        ["G1", "S", "G1"],
        _P_MARKERS,
        [[0.9, 0.8, 0.1], [0.1, 0.2, 0.3], [0.7, 0.4, 0.2]],
    ),
    "segData_D.RDS": _seg_data(
        ["TTTA-1", "TTTC-1"],
        ["LD", "LD"],
        ["YAL001C", "YCL001W"],
        [[4, 0], [1, 2]],
        [6.0, 3.0],
        ["G2/M", "M/G1"],
        _D_MARKERS,
        [[0.99, 0.97, 0.20, 0.90], [0.01, 0.02, 0.60, 0.30]],
    ),
    m.SEGMATCH_NAME: pd.DataFrame(
        {
            "X.Intercept.": [1.0, 1.0, 1.0],
            "log.barcode.features.nUMI.": list(np.log([5.0, 4.0, 3.0])),
            "experiment": [0.0, 0.0, 1.0],
            "best_match_seg": ["S2", "S1", "S2"],
            "best_match_seg.r": [0.9, 0.8, 0.7],
        }
    ),
}
_LOOKUP = {
    "gdata": xr.DataArray(
        np.asarray([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]]),
        dims=("dim_0", "dim_1"),
        coords={
            "dim_0": ["S0", "S1", "S2"],
            "dim_1": ["chrI:100_A/T", "chrI:300_G/C", "chrIII:150_A/G"],
        },
    )
}


class _Feature(SimpleNamespace):
    chrom: str
    start: int
    end: int


class _StubGenome:
    """``db``/``gene_set`` for the gene set; ``fasta_dna``/``chr_to_nc`` for L4."""

    gene_set = ["YAL001C", "YAL002W", "YCL001W", "YDL001W"]
    db = {
        "YAL001C": _Feature(chrom="chrI", start=120, end=180),
        "YAL002W": _Feature(chrom="chrI", start=400, end=500),
        "YCL001W": _Feature(chrom="chrIII", start=140, end=160),
        "YDL001W": _Feature(chrom="chrIV", start=490, end=510),
    }
    chr_to_nc = {1: "chrI", 3: "chrIII"}
    fasta_dna = {
        "chrI": SimpleNamespace(seq="N" * 99 + "A" + "N" * 99 + "C" + "N" * 99 + "G"),
        "chrIII": SimpleNamespace(seq="N" * 149 + "A"),
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_si1(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "TableS1. Yeast Strains"
    sheet.append(["Table S1: Yeast strains used in this study"])
    sheet.append(["Strains used in this study."])
    sheet.append(["Strain", "Parent 1", "Parent 2", "Name of cross"])
    for row in rows:
        sheet.append(list(row))
    workbook.save(path)


_SI1_ROWS = [("YLKD", _YPS, _YJM, "B"), ("YLKP", _BY, _RM, "A")]


def _write_raw(raw: Path, si1_rows: list[tuple[str, str, str, str]]) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name in [*_RELEASE, m.GDATA_NAME]:
        (raw / name).write_bytes(f"placeholder {name}".encode())
    _write_si1(raw / m.SI1_NAME, si1_rows)


def _write_mirror_manifest(data_root: Path, raw: Path) -> None:
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
                sha256=_sha256(raw / name),
            )
            for name in rel
        ],
        provenance_complete=True,
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _write_genomes_tier(data_root: Path) -> None:
    tier = data_root / "torchcell-genomes" / "peter2018_1011_assemblies"
    tier.mkdir(parents=True, exist_ok=True)
    index = tier / "1011Assemblies.tar.gz.member_index.tsv"
    index.write_text(_INDEX_LINES)
    (tier / "manifest.json").write_text(
        GenomeManifest(
            assembly_set="peter2018_1011_assemblies",
            organism="Saccharomyces cerevisiae",
            strain_or_population="1,011 isolates",
            source="Peter et al. 2018",
            release="2018",
            files=[
                ArtifactRecord(
                    path="1011Assemblies.tar.gz",
                    role="container",
                    bytes=1,
                    sha256=_TAR_SHA,
                ),
                ArtifactRecord(
                    path=index.name,
                    role="index",
                    bytes=index.stat().st_size,
                    sha256=_sha256(index),
                ),
            ],
            provenance_complete=True,
            created_at="2026-10-11T00:00:00+00:00",
        ).model_dump_json(indent=2)
    )
    sgd = data_root / "torchcell-genomes" / "sgd_S288C_R64-4-1_20230830"
    sgd.mkdir(parents=True, exist_ok=True)
    (sgd / "manifest.json").write_text(
        GenomeManifest(
            assembly_set="sgd_S288C_R64-4-1_20230830",
            organism="Saccharomyces cerevisiae",
            strain_or_population="S288C",
            source="SGD",
            release="R64-4-1_20230830",
            files=[
                ArtifactRecord(
                    path="S288C_reference_sequence_R64-4-1_20230830.fsa",
                    role="sequence",
                    bytes=2,
                    sha256=_S288C_SHA,
                )
            ],
            provenance_complete=True,
            created_at="2026-10-11T00:00:00+00:00",
        ).model_dump_json(indent=2)
    )


def _patch_release(monkeypatch: pytest.MonkeyPatch) -> None:
    import rdata

    monkeypatch.setattr(m, "PANELS", dict(_PANELS))
    monkeypatch.setattr(m, "N_POOLED_SEGREGANTS", 2)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 4)
    monkeypatch.setattr(rdata, "read_rds", lambda path: _RELEASE[Path(path).name])
    monkeypatch.setattr(rdata, "read_rda", lambda path: _LOOKUP)


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    si1_rows: list[tuple[str, str, str, str]] = _SI1_ROWS,
) -> m.Boocock2025Dataset:
    _patch_release(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "boocock2025"
    _write_raw(root / "raw", si1_rows)
    _write_mirror_manifest(data_root, root / "raw")
    _write_genomes_tier(data_root)
    return m.Boocock2025Dataset(root=str(root), genome=_genome())


def _raw_records(root: str) -> list[dict[str, Any]]:
    env = lmdb.open(str(Path(root) / "processed" / "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        records = [pickle.loads(v) for _, v in txn.cursor()]
    env.close()
    return records

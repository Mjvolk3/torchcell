# tests/torchcell/datasets/scerevisiae/test_lopez2024.py
# [[tests.torchcell.datasets.scerevisiae.test_lopez2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_lopez2024.py
"""Montaño López 2024 loaders: Table S2 row records, Table S3 two-block read, released names.

The supplementary workbook is written synthetically with openpyxl into ``<root>/raw/`` so
PyG never calls ``download()``; ``process()`` runs for real. No genome is consulted
(#1045): every row is a record under its released ORF (upper-cased), which is also its
``perturbed_gene_name``; a name that is not ORF-shaped (``ABC1``) is counted under the
leaf deferral rule in ``preprocess/dropped_records.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import subprocess
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import lopez2024 as m
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

# Table S2 rows (Gene-knockout strain, Fold change). Two rows for YAL001C (one lowercase:
# two plate measurements, two records), six other ORF-shaped names each stored as released
# whatever the annotation says of them, and one token that is not ORF-shaped (ABC1).
_SCREEN_ROWS: list[tuple[str, float]] = [
    ("yal001c", 1.5),
    ("YAL001C", 2.5),
    ("YBR001C", 0.5),
    ("YCR001W", 4.0),
    ("YBR002C", 0.8),
    ("YDR002W", 0.25),
    ("YER001W", 9.0),
    ("YFR001W", 1.1),
    ("ABC1", 1.2),
]

# Table S3 UP block (FC>=2) and DOWN block (FC<=0.5): (gene, FC average, STD). YBL071W-A is
# in both blocks and both rows are stored; the DOWN block is one row longer so the UP
# block's trailing gene cell is blank (exercises the dropna on the gene column).
_UP_ROWS: list[tuple[str, float, float]] = [
    ("YAL001C", 3.0, 0.3),
    ("YBL071W-A", 3.469, 0.1),
    ("YDR002W", 2.2, 0.6),
]
_DOWN_ROWS: list[tuple[str, float, float]] = [
    ("ybr001c", 0.25, 0.06),
    ("YBL071W-A", 0.0757, 0.01),
    ("YCR001W", 0.4, 0.09),
    ("YER001W", 0.3, 0.05),
]


def _write_workbook(
    raw: Path,
    screen_rows: list[tuple[str, float]],
    up_rows: list[tuple[str, float, float]],
    down_rows: list[tuple[str, float, float]],
) -> None:
    """Write ``supplementary_tables.xlsx`` with the loaders' header offsets.

    ``Table S2``: title row, then the header on 0-based row 1. ``Table S3``: title row,
    block-label row, then the header on 0-based row 2 with the UP block in columns A-C and
    the DOWN block in columns E-G (pandas suffixes the repeated names with ``.1``).
    """
    workbook = openpyxl.Workbook()
    s2 = workbook.active
    s2.title = "Table S2"
    s2.append(["Table S2. First genome-wide biosensor screen"])
    s2.append(["Gene-knockout strain", "Fold change"])
    for gene, fc in screen_rows:
        s2.append([gene, fc])
    s3 = workbook.create_sheet("Table S3")
    s3.append(["Table S3. Validated hits"])
    s3.append(["FC >= 2", None, None, None, "FC <= 0.5"])
    s3.append(
        [
            "Gene knockout-strain",
            "FC (average)",
            "STD",
            None,
            "Gene knockout-strain",
            "FC (average)",
            "STD",
        ]
    )
    for i in range(max(len(up_rows), len(down_rows))):
        up = list(up_rows[i]) if i < len(up_rows) else [None, None, None]
        down = list(down_rows[i]) if i < len(down_rows) else [None, None, None]
        s3.append([*up, None, *down])
    workbook.save(raw / m._XLSX_FILENAME)


def _root(
    tmp_path: Path,
    slug: str,
    screen_rows: list[tuple[str, float]] = _SCREEN_ROWS,
    up_rows: list[tuple[str, float, float]] = _UP_ROWS,
    down_rows: list[tuple[str, float, float]] = _DOWN_ROWS,
) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw", screen_rows, up_rows, down_rows)
    return root


@pytest.fixture
def screen(tmp_path: Path) -> m.IsobutanolScreenLopez2024Dataset:
    root = _root(tmp_path, "isobutanol_screen_lopez2024")
    return m.IsobutanolScreenLopez2024Dataset(root=str(root))


@pytest.fixture
def validated(tmp_path: Path) -> m.IsobutanolValidatedLopez2024Dataset:
    root = _root(tmp_path, "isobutanol_validated_lopez2024")
    return m.IsobutanolValidatedLopez2024Dataset(root=str(root))


_ENVIRONMENT = Environment(
    media=Media(name="SC", state="liquid", is_synthetic=True),
    temperature=Temperature(value=30.0),
    aerobicity="aerobic",
)
_PUBLICATION = Publication(
    pubmed_id="35022416",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/35022416/",
    doi="10.1038/s41467-021-27852-x",
    doi_url="https://doi.org/10.1038/s41467-021-27852-x",
)


def _phenotype(fc: float, se: float | None, n: int) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level={"isobutanol": fc},
        metabolite_level_se=None if se is None else {"isobutanol": se},
        n_replicates={"isobutanol": n},
        measurement_type="biosensor_gfp_fluorescence_fold_change",
        target_metabolite_ids=None,
    )


def _experiment(
    dataset_name: str, orf: str, gene: str, fc: float, se: float | None, n: int
) -> dict[str, object]:
    return MetaboliteExperiment(
        dataset_name=dataset_name,
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=gene
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=_phenotype(fc, se, n),
    ).model_dump()


def _reference(dataset_name: str, n: int) -> dict[str, object]:
    return MetaboliteExperimentReference(
        dataset_name=dataset_name,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=_phenotype(1.0, None, n),
    ).model_dump()


def _records(
    dataset: m._IsobutanolLopez2024Base,
) -> list[tuple[str, str, float, float | None, int]]:
    """(orf, perturbed name, fold change, se, n) per stored record, in LMDB order."""
    out = []
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        phenotype = experiment["phenotype"]
        se = phenotype["metabolite_level_se"]
        out.append(
            (
                perturbation["systematic_gene_name"],
                perturbation["perturbed_gene_name"],
                phenotype["metabolite_level"]["isobutanol"],
                None if se is None else se["isobutanol"],
                phenotype["n_replicates"]["isobutanol"],
            )
        )
    return out


def test_screen_stores_one_record_per_released_row(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """Nine Table S2 rows become nine records in row order, each n = 1 with no SE.

    YAL001C appears twice (rows 1.5 and 2.5, one typed lowercase): two records of one
    deletion, not an average. YBR002C, YDR002W, YER001W and YFR001W are stored under
    their released names whatever the annotation makes of them. ABC1 is not ORF-shaped
    and is stored on the released-gene-name leaf verbatim (#1045).
    """
    assert len(screen) == 9
    assert _records(screen) == [
        ("YAL001C", "YAL001C", 1.5, None, 1),
        ("YAL001C", "YAL001C", 2.5, None, 1),
        ("YBR001C", "YBR001C", 0.5, None, 1),
        ("YCR001W", "YCR001W", 4.0, None, 1),
        ("YBR002C", "YBR002C", 0.8, None, 1),
        ("YDR002W", "YDR002W", 0.25, None, 1),
        ("YER001W", "YER001W", 9.0, None, 1),
        ("YFR001W", "YFR001W", 1.1, None, 1),
        ("ABC1", "ABC1", 1.2, None, 1),
    ]
    (released,) = screen[8]["experiment"]["genotype"]["perturbations"]
    assert (
        released["perturbation_type"],
        released["systematic_gene_name"],
        released["released_kind"],
        released["cassette"],
    ) == ("released_identifier", "ABC1", "deletion", "KanMX")
    assert screen[0]["experiment"] == _experiment(
        "IsobutanolScreenLopez2024Dataset", "YAL001C", "YAL001C", 1.5, None, 1
    )
    assert screen[0]["publication"] == _PUBLICATION.model_dump()
    log = json.loads(
        (Path(screen.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert log == {
        "dataset": "IsobutanolScreenLopez2024Dataset",
        "source_records": 9,
        "kept_records": 9,
        "dropped_records": 0,
        "rules": [
            {
                "rule": m.RETENTION_RULE,
                "description": m.RETENTION_DESCRIPTION,
                "n_records": 0,
                "items": ["ABC1"],
            }
        ],
    }


def test_screen_reference_is_one_shared_wt_control_with_n_one(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """Every record is one measurement, so the WT reference (FC 1.0, n = 1, SE None,
    BY4741, SC liquid 30 C aerobic) is one shared entry in
    ``experiment_reference_index.json`` over all nine records.
    """
    assert screen[0]["reference"] == _reference("IsobutanolScreenLopez2024Dataset", 1)
    assert screen[7]["reference"] == _reference("IsobutanolScreenLopez2024Dataset", 1)
    index = json.loads(
        (
            Path(screen.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [list(range(9))]
    assert index[0]["reference"]["phenotype_reference"]["n_replicates"] == {
        "isobutanol": 1
    }
    assert index[0]["reference"]["genome_reference"] == {
        "provenance_gaps": [],
        "species": "Saccharomyces cerevisiae",
        "strain": "BY4741",
        "ploidy": "haploid",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "assembly_accession": "GCA_000146045.2",
    }


def test_screen_writes_gene_set_and_build_manifest(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """``preprocess/gene_set.json`` is the eight released names sorted; the build manifest
    names the root slug, the loader class/module, this host, and the worktree HEAD.

    The HEAD comparison shells out to ``git rev-parse HEAD`` in the loader's directory,
    so this test fails in a non-git checkout, where ``torchcell_commit`` is None by design.
    """
    preprocess = Path(screen.root) / "preprocess"
    released = [
        "ABC1",
        "YAL001C",
        "YBR001C",
        "YBR002C",
        "YCR001W",
        "YDR002W",
        "YER001W",
        "YFR001W",
    ]
    assert json.loads((preprocess / "gene_set.json").read_text()) == released
    assert screen.gene_set == set(released)
    assert screen.experiment_class is MetaboliteExperiment
    assert screen.reference_class is MetaboliteExperimentReference
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(m.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["manifest_schema_version"] == 3
    assert manifest["dataset_name"] == "isobutanol_screen_lopez2024"
    assert manifest["loader_class"] == "IsobutanolScreenLopez2024Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.lopez2024"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {"MetaboliteExperiment", "KanMxDeletionPerturbation"} <= set(
        manifest["closure"]
    )


def test_validated_reads_both_blocks_and_stores_every_row(
    validated: m.IsobutanolValidatedLopez2024Dataset,
) -> None:
    """UP rows then DOWN rows, each a triplicate record with SE = STD / sqrt(3), under
    the released ORF: YAL001C 3.0, YBL071W-A 3.469, YDR002W 2.2, ybr001c -> YBR001C
    (upper-cased) 0.25, YBL071W-A 0.0757, YCR001W 0.4, YER001W 0.3. Both YBL071W-A rows
    are stored; 7 source rows give 7 records. The reference is one shared object with
    n = 3, so the index has one entry over all seven.
    """
    assert len(validated) == 7
    assert _records(validated) == [
        ("YAL001C", "YAL001C", 3.0, 0.3 / math.sqrt(3), 3),
        ("YBL071W-A", "YBL071W-A", 3.469, 0.1 / math.sqrt(3), 3),
        ("YDR002W", "YDR002W", 2.2, 0.6 / math.sqrt(3), 3),
        ("YBR001C", "YBR001C", 0.25, 0.06 / math.sqrt(3), 3),
        ("YBL071W-A", "YBL071W-A", 0.0757, 0.01 / math.sqrt(3), 3),
        ("YCR001W", "YCR001W", 0.4, 0.09 / math.sqrt(3), 3),
        ("YER001W", "YER001W", 0.3, 0.05 / math.sqrt(3), 3),
    ]
    assert validated[2]["experiment"] == _experiment(
        "IsobutanolValidatedLopez2024Dataset",
        "YDR002W",
        "YDR002W",
        2.2,
        0.6 / math.sqrt(3),
        3,
    )
    assert validated[3]["reference"] == _reference(
        "IsobutanolValidatedLopez2024Dataset", 3
    )
    assert validated[3]["publication"] == _PUBLICATION.model_dump()
    index = json.loads(
        (
            Path(validated.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [list(range(7))]
    assert json.loads(
        (Path(validated.root) / "preprocess" / "gene_set.json").read_text()
    ) == ["YAL001C", "YBL071W-A", "YBR001C", "YCR001W", "YDR002W", "YER001W"]
    manifest = json.loads(
        (Path(validated.root) / "preprocess" / "build_manifest.json").read_text()
    )
    assert manifest["dataset_name"] == "isobutanol_validated_lopez2024"
    assert manifest["loader_class"] == "IsobutanolValidatedLopez2024Dataset"
    log = json.loads(
        (Path(validated.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert (log["source_records"], log["kept_records"], log["dropped_records"]) == (
        7,
        7,
        0,
    )


def test_validated_stores_a_name_that_is_not_orf_shaped_as_released(
    tmp_path: Path,
) -> None:
    """A Table S3 name with no ORF form is stored on the released-gene-name leaf, never
    resolved and never dropped; the ledger lists it under the retention rule.
    """
    root = _root(
        tmp_path,
        "released",
        up_rows=[("YAL001C", 3.0, 0.3), ("YLR287-A", 2.5, 0.2)],
        down_rows=[("YCR001W", 0.4, 0.09)],
    )
    dataset = m.IsobutanolValidatedLopez2024Dataset(root=str(root))
    assert len(dataset) == 3
    (released,) = dataset[1]["experiment"]["genotype"]["perturbations"]
    assert released["perturbation_type"] == "released_identifier"
    assert released["systematic_gene_name"] == "YLR287-A"
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert (log["kept_records"], log["dropped_records"]) == (3, 0)
    assert log["rules"] == [
        {
            "rule": m.RETENTION_RULE,
            "description": m.RETENTION_DESCRIPTION,
            "n_records": 0,
            "items": ["YLR287-A"],
        }
    ]


def test_download_reads_the_library_mirror_and_verifies_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw file, ``download()`` copies from ``$DATA_ROOT/torchcell-library/
    lopezSystemsMetabolicEngineering2024/data/`` and checks the pinned digest. A missing
    mirror file raises naming the path; a mirror file holding ``b"not the real workbook"``
    is rejected with ``RawSha256MismatchError`` naming it, the pinned sha256 and its
    actual one (55b4751d...) BEFORE the copy, so ``raw/`` stays empty.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / "data"
    with pytest.raises(
        RuntimeError, match=f"library mirror data file not found: {mirror}"
    ):
        m.IsobutanolScreenLopez2024Dataset(root=str(tmp_path / "a"))
    mirror.mkdir(parents=True)
    (mirror / m._XLSX_FILENAME).write_bytes(b"not the real workbook")
    digest = hashlib.sha256(b"not the real workbook").hexdigest()
    assert digest.startswith("55b4751d")
    with pytest.raises(RawSha256MismatchError) as err:
        m.IsobutanolScreenLopez2024Dataset(root=str(tmp_path / "b"))
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / m._XLSX_FILENAME}: expected "
        f"{m._XLSX_SHA256}, observed {digest}"
    )
    assert list((tmp_path / "b" / "raw").iterdir()) == []


@pytest.mark.parametrize(
    "dataset_cls",
    [m.IsobutanolScreenLopez2024Dataset, m.IsobutanolValidatedLopez2024Dataset],
)
def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    dataset_cls: Any, off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with the workbook already in ``raw/`` PyG skips
    ``download()``, so each screen's ``process()`` verifies it against ``_XLSX_SHA256``
    first and raises ``RawSha256MismatchError`` naming it and both digests before a row
    is read; no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [m._XLSX_FILENAME])
    raw = staged.root / "raw" / m._XLSX_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        dataset_cls(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected {m._XLSX_SHA256}, "
        f"observed {staged.observed}"
    )
    assert m._XLSX_SHA256 == (
        "f97cf13c2d40c2a0a374475001dcf4dc8493429314661c6960d5f019ce23889b"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


# Phase 24: the verified-copy log line and the inert generic hooks


def test_download_copies_a_matching_mirror_file_and_logs_the_verified_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A mirror file whose digest matches the pin is copied byte for byte and logged.

    ``_XLSX_SHA256`` is patched to ``sha256(b"pinned workbook")`` so a hand-built file
    passes; ``download`` is called unbound on a namespace with only ``raw_dir`` (the
    one attribute it reads), so no build runs. The log line is
    ``Verified <raw_dir>/<file> (sha256 <digest>)`` at INFO.
    """
    import logging
    from types import SimpleNamespace

    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / "data"
    mirror.mkdir(parents=True)
    (mirror / m._XLSX_FILENAME).write_bytes(b"pinned workbook")
    digest = hashlib.sha256(b"pinned workbook").hexdigest()
    monkeypatch.setattr(m, "_XLSX_SHA256", digest)
    raw_dir = tmp_path / "root" / "raw"
    with caplog.at_level(logging.INFO, logger=m.__name__):
        m.IsobutanolScreenLopez2024Dataset.download(
            cast(Any, SimpleNamespace(raw_dir=str(raw_dir)))
        )
    dest = raw_dir / m._XLSX_FILENAME
    assert dest.read_bytes() == b"pinned workbook"
    assert [r.getMessage() for r in caplog.records if r.name == m.__name__] == [
        f"Verified {dest} (sha256 {digest})"
    ]


def test_generic_hooks_pass_the_frame_through_and_refuse_create_experiment() -> None:
    """``preprocess_raw`` returns the very frame it was given (``is``), ``preprocess``
    ignored; ``create_experiment`` raises a bare ``NotImplementedError`` (empty message),
    since both screens build their experiments inline in ``process()``.
    """
    frame = pd.DataFrame({"a": [1]})
    for cls in (
        m.IsobutanolScreenLopez2024Dataset,
        m.IsobutanolValidatedLopez2024Dataset,
    ):
        assert cls.preprocess_raw(cast(Any, None), frame, {"k": 1}) is frame
        with pytest.raises(NotImplementedError) as err:
            cls.create_experiment(cast(Any, None))
        assert str(err.value) == ""

# tests/torchcell/datasets/scerevisiae/test_dasilveira2014.py
# [[tests.torchcell.datasets.scerevisiae.test_dasilveira2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_dasilveira2014.py
"""Hermetic build of the da Silveira dos Santos 2014 lipidome loader on synthetic tables.

Both workbooks are written with openpyxl into ``<root>/raw/`` so PyG never calls
``download()``. The genome stub carries the two attributes the loader reads: ``gene_set``
(YAL001C, YBR001C) and ``alias_to_systematic`` (YBR002C -> YBR001C, YAL003W -> YAL001C).

Table S4 ``Quant`` sheet (Systematic Name, Standard Name, PC 32:1, PE 34:2, Erg):

    Y7092    WT1    10    20    (blank)
    Y7220    WT2    12    (blank)  5
    BY4741   WT3    14    22    7
    YAL001C  TFC3   1.5   2.5   3.5
    YBR002C  (blank) 4.0  (blank) 6.0      alias -> YBR001C, gene name falls back to the ORF
    YZZ999W  GHOST  1     1     1          unresolved -> dropped
    YAL003W  OLD    9     9     9          alias -> YAL001C already seen -> dropped

WT reference: PC 32:1 mean(10, 12, 14) = 12.0 over n = 3; PE 34:2 mean(20, 22) = 21.0,
n = 2; Erg mean(5, 7) = 6.0, n = 2. Each record's reference is restricted to the lipids
that record measured, and ``n_replicates`` per lipid is the fixed biological-replicate
count 2 with no SE.

Table S10 (header on the third row: ID, Name, Formula, ChEBI): PC 32:1 -> CHEBI:64998;
PE 34:2 has no ChEBI cell; a third row has no Name. Only the first survives the map.

2026.09.30 (Phase 16): the same seven rows, plus

- the ledger (lines 254, 282, 289 and 322): "1/3 lipids have a Table S10 ChEBI id (2
  missing)"; a warning "duplicate ORF YAL001C after resolution; keeping first" for
  YAL003W; "2 mutant records, 3 WT control rows -> measured reference (3 lipids), 1
  unresolved ORFs (['YZZ999W'])"; "Wrote 2 da Silveira lipidome experiments to LMDB".
- ``transform_item`` round trips through ``MetaboliteExperiment`` and
  ``MetaboliteExperimentReference``.
- an edge matrix: the three WT rows never measure Erg (10/20/-, 12/-/-, 14/22/-), a
  padded `` YAL001C `` / `` TFC3 `` row measures all three, and a lowercase ``ybr001c``
  row. The WT reference covers two lipids (PC 32:1 12.0 over n = 3, PE 34:2 21.0 over
  n = 2); the padded row is stripped and kept with Erg 3.5 but its reference has no Erg;
  ``ybr001c`` is unresolved (``gene_set`` membership is case-sensitive, line 201, and
  the alias table has no entry for a systematic name), so the summary ends "1 unresolved
  ORFs (['ybr001c'])"; with nothing unresolved the suffix is empty.
- a mutant row with every lipid blank: its level is ``{}`` and the phenotype validator
  refuses "metabolite_level cannot be empty".
- ``download`` with both pins replaced by the digests of the synthetic workbooks: both
  are copied from the mirror and each logs "Verified <dest> (sha256 <digest>)".
- ``main`` with ``load_dotenv`` stubbed and the genome and dataset classes as recorders.

Findings pinned here: a lipid no WT control measured is stored on the mutant but has no
reference entry (line 246 keeps only non-NaN WT means, line 361 restricts to those), so
the reference is a strict subset even when the strain measured everything; a mutant row
with no measured lipid raises pydantic's ``ValidationError`` after the write env is open
(line 305), leaving ``data.csv`` with ``n_lipids`` 0 and an empty ``processed/lmdb``
that a retry serves as 0 records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pydantic
import pytest

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
from torchcell.datasets.scerevisiae import dasilveira2014 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


@pytest.fixture(autouse=True)
def _no_tc_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build here reads ``raw/``; an inherited ``TC_DATA_URL`` would send it to
    the tc-data endpoint instead (``ExperimentDataset._download``).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)


_LIPIDS = ["PC 32:1", "PE 34:2", "Erg"]
_QUANT_ROWS: list[list[Any]] = [
    ["Y7092", "WT1", 10, 20, None],
    ["Y7220", "WT2", 12, None, 5],
    ["BY4741", "WT3", 14, 22, 7],
    ["YAL001C", "TFC3", 1.5, 2.5, 3.5],
    ["YBR002C", None, 4.0, None, 6.0],
    ["YZZ999W", "GHOST", 1, 1, 1],
    ["YAL003W", "OLD", 9, 9, 9],
]
_S10_ROWS: list[list[Any]] = [
    ["Table S10. LipidX identifiers"],
    ["ChEBI ids per lipid species"],
    ["ID", "Name", "Formula", "ChEBI"],
    ["LX1", "PC 32:1", "C40H78NO8P", "CHEBI:64998"],
    ["LX2", "PE 34:2", "C39H74NO8P", None],
    ["LX3", None, "C28H44O", "CHEBI:16933"],
]


class _StubGenome:
    gene_set = {"YAL001C", "YBR001C"}
    alias_to_systematic: dict[str, list[str]] = {
        "YBR002C": ["YBR001C"],
        "YAL003W": ["YAL001C"],
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _write_raw(raw: Path, quant_rows: list[list[Any]] = _QUANT_ROWS) -> None:
    s4 = openpyxl.Workbook()
    quant = s4.active
    quant.title = m._QUANT_SHEET
    quant.append([m._SYS_COL, m._STD_COL, *_LIPIDS])
    for row in quant_rows:
        quant.append(row)
    s4.save(raw / m.DATA_FILENAME)
    s10 = openpyxl.Workbook()
    sheet = s10.active
    for row in _S10_ROWS:
        sheet.append(row)
    s10.save(raw / m.CHEBI_FILENAME)


def _root(tmp_path: Path, quant_rows: list[list[Any]] = _QUANT_ROWS) -> Path:
    root = tmp_path / "metabolite_dasilveira2014"
    (root / "raw").mkdir(parents=True)
    _write_raw(root / "raw", quant_rows)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.MetaboliteDaSilveira2014Dataset:
    return m.MetaboliteDaSilveira2014Dataset(
        root=str(_root(tmp_path)), genome=_genome()
    )


_ENVIRONMENT = Environment(
    media=Media(name="YPD", state="liquid", is_synthetic=False),
    temperature=Temperature(value=30),
)
_PUBLICATION = Publication(
    pubmed_id="25143408",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/25143408/",
    doi="10.1091/mbc.E14-03-0851",
    doi_url="https://doi.org/10.1091/mbc.E14-03-0851",
)


def _phenotype(level: dict[str, float], n: dict[str, int]) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=None,
        n_replicates=n,
        measurement_type="lipidomics_ms_relative_abundance_au",
        target_metabolite_ids=None,
    )


def _experiment(orf: str, gene: str, level: dict[str, float]) -> dict[str, Any]:
    return MetaboliteExperiment(
        dataset_name="MetaboliteDaSilveira2014Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=gene
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=_phenotype(level, dict.fromkeys(level, 2)),
    ).model_dump()


def _reference(level: dict[str, float], n: dict[str, int]) -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="MetaboliteDaSilveira2014Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=_phenotype(level, n),
    ).model_dump()


def test_two_mutant_records_with_wt_reference_restricted_per_record(
    dataset: m.MetaboliteDaSilveira2014Dataset,
) -> None:
    """Seven Quant rows give two records. Record 0 (YAL001C/TFC3) measures all three
    lipids, n = 2 each, SE None; its reference is the three WT means with n 3/2/2.
    Record 1 (YBR002C -> YBR001C, blank standard name -> the ORF) measured PC 32:1 and
    Erg only, so its reference drops PE 34:2. The index has one entry per reference.
    """
    assert len(dataset) == 2
    assert dataset[0]["experiment"] == _experiment(
        "YAL001C", "TFC3", {"PC 32:1": 1.5, "PE 34:2": 2.5, "Erg": 3.5}
    )
    assert dataset[0]["reference"] == _reference(
        {"PC 32:1": 12.0, "PE 34:2": 21.0, "Erg": 6.0},
        {"PC 32:1": 3, "PE 34:2": 2, "Erg": 2},
    )
    assert dataset[1]["experiment"] == _experiment(
        "YBR001C", "YBR001C", {"PC 32:1": 4.0, "Erg": 6.0}
    )
    assert dataset[1]["reference"] == _reference(
        {"PC 32:1": 12.0, "Erg": 6.0}, {"PC 32:1": 3, "Erg": 2}
    )
    assert dataset[1]["publication"] == _PUBLICATION.model_dump()
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0], [1]]


def test_side_files(dataset: m.MetaboliteDaSilveira2014Dataset) -> None:
    """``lipid_chebi.csv`` lists every Quant lipid with the Table S10 ChEBI id or blank;
    ``data.csv`` has the resolved ORF, the stored gene name and the measured-lipid count.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "lipid_chebi.csv").read_text() == (
        "lipid,chebi\nPC 32:1,CHEBI:64998\nPE 34:2,\nErg,\n"
    )
    assert (preprocess / "data.csv").read_text() == (
        "orf,gene,n_lipids\nYAL001C,TFC3,3\nYBR001C,YBR001C,2\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "metabolite_dasilveira2014"
    assert manifest["loader_class"] == "MetaboliteDaSilveira2014Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.dasilveira2014"


def test_wrong_wt_row_count_raises(tmp_path: Path) -> None:
    """Dropping the BY4741 control row leaves two of the three expected WT ids."""
    rows = [row for row in _QUANT_ROWS if row[0] != "BY4741"]
    with pytest.raises(RuntimeError, match="expected 3 WT control rows, found 2"):
        m.MetaboliteDaSilveira2014Dataset(
            root=str(_root(tmp_path, rows)), genome=_genome()
        )


def test_requires_an_injected_genome(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match="MetaboliteDaSilveira2014Dataset requires an injected SCerevisiaeGenome "
        "to validate systematic ORF ids against R64",
    ):
        m.MetaboliteDaSilveira2014Dataset(root=str(_root(tmp_path)), genome=None)


def test_download_verifies_present_files_and_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``download()`` on a root holding the synthetic Table S4 rejects it with its actual
    digest against the pinned one; on an empty root it looks for the file under
    ``$DATA_ROOT/torchcell-library/daSystematicLipidomicAnalysis2014/data/`` and names
    that path when it is absent.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.MetaboliteDaSilveira2014Dataset(
        root=str(_root(tmp_path)), genome=_genome()
    )
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{m.DATA_FILENAME} sha256 mismatch: got {digest}, expected {m.DATA_SHA256}"
        ),
    ):
        dataset.download()
    src = (
        tmp_path
        / "data_root"
        / "torchcell-library"
        / m._LIBRARY_CITATION_KEY
        / "data"
        / m.DATA_FILENAME
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"library mirror data file not found: {src}. This dataset's source is the "
            f"sha256-pinned {m.DATA_FILENAME} in the torchcell-library mirror."
        ),
    ):
        m.MetaboliteDaSilveira2014Dataset(
            root=str(tmp_path / "empty"), genome=_genome()
        )


def test_items_retype_through_the_metabolite_classes(
    dataset: m.MetaboliteDaSilveira2014Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    155 to 163), dumping back to exactly the stored dictionaries; ``preprocess_raw``
    returns its frame unchanged.
    """
    for index in range(2):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is MetaboliteExperiment
        assert type(typed["reference"]) is MetaboliteExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION.model_dump()
    frame = pd.DataFrame({m._SYS_COL: ["YAL001C"]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {m._SYS_COL: ["YAL001C"]}


def test_ledger_logs_chebi_coverage_the_duplicate_and_the_unresolved_orf(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One of three lipids has a ChEBI id; YAL003W resolves to the already seen YAL001C
    and is dropped with a warning; YZZ999W is the one unresolved ORF.
    """
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.MetaboliteDaSilveira2014Dataset(root=str(_root(tmp_path)), genome=_genome())
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == m.log.name
    ] == [
        ("INFO", "da Silveira: 1/3 lipids have a Table S10 ChEBI id (2 missing)"),
        ("WARNING", "duplicate ORF YAL001C after resolution; keeping first"),
        (
            "INFO",
            "da Silveira: 2 mutant records, 3 WT control rows -> measured reference "
            "(3 lipids), 1 unresolved ORFs (['YZZ999W'])",
        ),
        ("INFO", "Wrote 2 da Silveira lipidome experiments to LMDB"),
    ]


def test_lipid_no_wt_measured_has_no_reference_entry_and_names_are_stripped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding: Erg is blank in all three WT rows, so the reference covers two lipids
    and the padded YAL001C record keeps Erg 3.5 with no reference value for it (lines
    246 and 361). The padded ORF and standard name are stripped; the lowercase
    ``ybr001c`` does not resolve (``gene_set`` membership is case-sensitive, line 201).
    Pinned until the loader either refuses a lipid without a WT baseline or records it.
    """
    rows: list[list[Any]] = [
        ["Y7092", "WT1", 10, 20, None],
        ["Y7220", "WT2", 12, None, None],
        ["BY4741", "WT3", 14, 22, None],
        [" YAL001C ", " TFC3 ", 1.5, 2.5, 3.5],
        ["ybr001c", "NTH1", 1, 2, 3],
    ]
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.MetaboliteDaSilveira2014Dataset(
            root=str(_root(tmp_path, rows)), genome=_genome()
        )
    assert len(dataset) == 1
    assert dataset[0]["experiment"] == _experiment(
        "YAL001C", "TFC3", {"PC 32:1": 1.5, "PE 34:2": 2.5, "Erg": 3.5}
    )
    assert dataset[0]["reference"] == _reference(
        {"PC 32:1": 12.0, "PE 34:2": 21.0}, {"PC 32:1": 3, "PE 34:2": 2}
    )
    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and "mutant records" in r.getMessage()
    ]
    assert summaries == [
        "da Silveira: 1 mutant records, 3 WT control rows -> measured reference "
        "(2 lipids), 1 unresolved ORFs (['ybr001c'])"
    ]
    dataset.close_lmdb()


def test_clean_matrix_logs_no_unresolved_suffix(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With every mutant resolved the summary ends at "0 unresolved ORFs", no list."""
    rows = [row for row in _QUANT_ROWS if row[0] in {"Y7092", "Y7220", "BY4741"}]
    rows.append(["YAL001C", "TFC3", 1.5, 2.5, 3.5])
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.MetaboliteDaSilveira2014Dataset(
            root=str(_root(tmp_path, rows)), genome=_genome()
        )
    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and "mutant records" in r.getMessage()
    ]
    assert summaries == [
        "da Silveira: 1 mutant records, 3 WT control rows -> measured reference "
        "(3 lipids), 0 unresolved ORFs"
    ]
    dataset.close_lmdb()


def test_all_blank_mutant_row_fails_validation_and_leaves_an_empty_store(
    tmp_path: Path,
) -> None:
    """Finding: a mutant with every lipid blank gets ``metabolite_level={}`` and the
    phenotype validator raises pydantic's ``ValidationError`` ("metabolite_level cannot
    be empty") inside the open write transaction (line 305), so ``data.csv`` already
    records the strain with ``n_lipids`` 0 and a retry on the same root finds the empty
    ``processed/lmdb`` and serves 0 records. Pinned until the loader drops or refuses
    the row before opening the store.
    """
    rows = [row for row in _QUANT_ROWS if row[0] in {"Y7092", "Y7220", "BY4741"}]
    rows.append(["YAL001C", "TFC3", None, None, None])
    root = _root(tmp_path, rows)
    with pytest.raises(pydantic.ValidationError) as info:
        m.MetaboliteDaSilveira2014Dataset(root=str(root), genome=_genome())
    assert [e["msg"] for e in info.value.errors()] == [
        "Value error, metabolite_level cannot be empty"
    ]
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,gene,n_lipids\nYAL001C,TFC3,0\n"
    )
    # The failed constructor's write handle is unreachable and still open, and the CI
    # py-lmdb refuses a second open of the path in this process, so the half-built
    # state is pinned on disk: the store directory exists with its data file, and no
    # gene set was written, which is what a retry would read as zero records.
    assert (root / "processed" / "lmdb" / "data.mdb").exists()
    assert not (root / "preprocess" / "gene_set.json").exists()


def test_download_copies_both_verified_mirror_workbooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With both pins set to the synthetic workbooks' digests, an empty root copies
    Table S4 then Table S10 from ``<DATA_ROOT>/torchcell-library/<key>/data/``, logs one
    "Verified" line per file and builds the two records.
    """
    source = tmp_path / "source"
    source.mkdir()
    _write_raw(source)
    data_root = tmp_path / "data_root"
    mirror = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / "data"
    mirror.mkdir(parents=True)
    content = {}
    for name in (m.DATA_FILENAME, m.CHEBI_FILENAME):
        content[name] = (source / name).read_bytes()
        (mirror / name).write_bytes(content[name])
    pins = {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}
    monkeypatch.setattr(m, "DATA_SHA256", pins[m.DATA_FILENAME])
    monkeypatch.setattr(m, "CHEBI_SHA256", pins[m.CHEBI_FILENAME])
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.MetaboliteDaSilveira2014Dataset(
            root=str(tmp_path / "fresh"), genome=_genome()
        )
    raw = tmp_path / "fresh" / "raw"
    assert {name: (raw / name).read_bytes() for name in content} == content
    verified = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and r.getMessage().startswith("Verified")
    ]
    assert verified == [
        f"Verified {raw / name} (sha256 {pins[name]})"
        for name in (m.DATA_FILENAME, m.CHEBI_FILENAME)
    ]
    assert len(dataset) == 2


def test_main_builds_genome_and_dataset_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds the genome from ``$DATA_ROOT/data/sgd/genome`` and ``data/go`` with
    ``overwrite=False``, then the dataset at
    ``$DATA_ROOT/data/torchcell/metabolite_dasilveira2014`` with that genome, and prints
    its length and first item. Both classes are recorders.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Genome:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("genome", kwargs))

    class _Dataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("dataset", kwargs))

        def __len__(self) -> int:
            return 127

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(m, "SCerevisiaeGenome", _Genome)
    monkeypatch.setattr(m, "MetaboliteDaSilveira2014Dataset", _Dataset)
    m.main()
    assert [name for name, _ in calls] == ["genome", "dataset"]
    assert calls[0][1] == {
        "genome_root": f"{tmp_path}/data/sgd/genome",
        "go_root": f"{tmp_path}/data/go",
        "overwrite": False,
    }
    assert list(calls[1][1]) == ["root", "genome"]
    assert calls[1][1]["root"] == f"{tmp_path}/data/torchcell/metabolite_dasilveira2014"
    assert type(calls[1][1]["genome"]) is _Genome
    assert capsys.readouterr().out == "len = 127\nitem[0]\n"

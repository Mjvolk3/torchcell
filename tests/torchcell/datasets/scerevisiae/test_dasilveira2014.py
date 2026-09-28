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
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import openpyxl
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
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
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

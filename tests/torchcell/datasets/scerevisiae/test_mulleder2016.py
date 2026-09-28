# tests/torchcell/datasets/scerevisiae/test_mulleder2016.py
# [[tests.torchcell.datasets.scerevisiae.test_mulleder2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_mulleder2016.py
"""Hermetic build of the Mulleder 2016 amino-acid metabolome loader on a synthetic Table S3.

The workbook is written with openpyxl under the loader's ``.xls`` filename (pandas sniffs
the zip container and opens it with openpyxl) into ``<root>/raw/`` so PyG never calls
``download()``. No genome is involved: ORFs are validated with a regex.

``intracellular_concentration_mM`` sheet (ORF + the 19 amino acids in ``AMINO_ACIDS``
order):

    YAL001C     0.5, 1.0, ..., 9.5   (0.5 * (i + 1) for amino acid i)
    YBR001C     2.0 x 19
    YBR001C     3.0 x 19             duplicate ORF -> dropped, first row kept
    WT          1.0 x 19             not a systematic name -> skipped
    YLR287-A    1.0 x 19             malformed -> skipped
    " YCR001W " 1.0 x 19             stripped

``robust_summary_statistics`` sheet (amino acid, mean (mM), sd (mM)): mean 0.25 * (i + 1)
per amino acid; every record's reference is that table, n = 1 per amino acid, SE None.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from torchcell.datamodels.media import SM_AGAR
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import mulleder2016 as m

_AA = m.AMINO_ACIDS
_LEVELS_YAL001C = [0.5 * (i + 1) for i in range(19)]
_CONC_ROWS: list[list[Any]] = [
    ["YAL001C", *_LEVELS_YAL001C],
    ["YBR001C", *([2.0] * 19)],
    ["YBR001C", *([3.0] * 19)],
    ["WT", *([1.0] * 19)],
    ["YLR287-A", *([1.0] * 19)],
    [" YCR001W ", *([1.0] * 19)],
]
_SUMMARY_MEANS = {aa: 0.25 * (i + 1) for i, aa in enumerate(_AA)}


def _write_workbook(
    raw: Path,
    conc_columns: list[str] = _AA,
    summary_means: dict[str, float] = _SUMMARY_MEANS,
) -> None:
    workbook = openpyxl.Workbook()
    conc = workbook.active
    conc.title = m._CONC_SHEET
    conc.append(["ORF", *conc_columns])
    for row in _CONC_ROWS:
        conc.append(row[: 1 + len(conc_columns)])
    summary = workbook.create_sheet(m._SUMMARY_SHEET)
    summary.append(["amino acid", "mean (mM)", "sd (mM)"])
    for aa, mean in summary_means.items():
        summary.append([aa, mean, 0.1])
    workbook.save(raw / m.DATA_FILENAME)


def _root(tmp_path: Path, slug: str = "amino_acid_mulleder2016", **kwargs: Any) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw", **kwargs)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.AminoAcidMulleder2016Dataset:
    return m.AminoAcidMulleder2016Dataset(root=str(_root(tmp_path)))


_ENVIRONMENT = Environment(media=SM_AGAR, temperature=Temperature(value=30))
_PUBLICATION = Publication(
    pubmed_id="27693354",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/27693354/",
    doi="10.1016/j.cell.2016.09.007",
    doi_url="https://doi.org/10.1016/j.cell.2016.09.007",
).model_dump()


def _experiment(orf: str, levels: list[float]) -> dict[str, Any]:
    return MetaboliteExperiment(
        dataset_name="AminoAcidMulleder2016Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level=dict(zip(_AA, levels, strict=True)),
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(_AA, 1),
            measurement_type="intracellular_concentration_mM",
            target_metabolite_ids=None,
        ),
    ).model_dump()


def _reference(means: dict[str, float]) -> dict[str, Any]:
    return MetaboliteExperimentReference(
        dataset_name="AminoAcidMulleder2016Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level=means,
            metabolite_level_se=None,
            n_replicates=dict.fromkeys(means, 1),
            measurement_type="intracellular_concentration_mM",
        ),
    ).model_dump()


def test_three_records_n_one_with_the_robust_mean_reference(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Six rows give three records (YAL001C, the first YBR001C, the stripped YCR001W);
    the WT and malformed names are skipped and the second YBR001C is a collision. Each
    amino acid is n = 1 with no SE on solid SM at 30 C; the reference is the summary mean.
    """
    assert len(dataset) == 3
    assert dataset[0]["experiment"] == _experiment("YAL001C", _LEVELS_YAL001C)
    assert dataset[1]["experiment"] == _experiment("YBR001C", [2.0] * 19)
    assert dataset[2]["experiment"] == _experiment("YCR001W", [1.0] * 19)
    assert dataset[0]["reference"] == _reference(_SUMMARY_MEANS)
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["experiment"]["environment"]["media"]["state"] == "solid"


def test_side_files(dataset: m.AminoAcidMulleder2016Dataset) -> None:
    """``data.csv`` is one row per kept ORF with the 19 concentrations; one reference
    covers all three records; the gene set is the three ORFs.
    """
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    expected_rows = [
        ["YAL001C", *_LEVELS_YAL001C],
        ["YBR001C", *([2.0] * 19)],
        ["YCR001W", *([1.0] * 19)],
    ]
    assert (preprocess / "data.csv").read_text() == (
        ",".join(["orf", *_AA])
        + "\n"
        + "".join(",".join(str(v) for v in row) + "\n" for row in expected_rows)
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "amino_acid_mulleder2016"
    assert manifest["loader_class"] == "AminoAcidMulleder2016Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.mulleder2016"


def test_reference_takes_every_summary_row_not_only_the_19_amino_acids(
    tmp_path: Path,
) -> None:
    """Finding: ``_reference_levels`` is built from every row of the summary sheet, so an
    extra ``cysteine`` row (the amino acid the paper excludes) lands in the reference
    phenotype with n = 1 while no experiment measures it; the reference then has 20 keys
    against the experiment's 19.
    """
    means = {**_SUMMARY_MEANS, "cysteine": 0.125}
    root = _root(tmp_path, summary_means=means)
    dataset = m.AminoAcidMulleder2016Dataset(root=str(root))
    assert dataset[0]["reference"] == _reference(means)
    assert set(dataset[0]["experiment"]["phenotype"]["metabolite_level"]) == set(_AA)


def test_missing_concentration_column_raises(tmp_path: Path) -> None:
    root = _root(tmp_path, conc_columns=_AA[:-1])
    with pytest.raises(
        RuntimeError,
        match=re.escape("Table S3 concentration sheet missing columns: ['tyrosine']"),
    ):
        m.AminoAcidMulleder2016Dataset(root=str(root))


def test_missing_summary_amino_acid_raises(tmp_path: Path) -> None:
    means = {aa: v for aa, v in _SUMMARY_MEANS.items() if aa != "tyrosine"}
    root = _root(tmp_path, summary_means=means)
    with pytest.raises(
        RuntimeError, match=re.escape("summary sheet missing amino acids: ['tyrosine']")
    ):
        m.AminoAcidMulleder2016Dataset(root=str(root))


def test_download_trusts_a_present_file_without_hashing(
    dataset: m.AminoAcidMulleder2016Dataset,
) -> None:
    """Finding: the docstring says ``download()`` verifies the sha256, but an already
    present raw file returns early unverified (source line 135). The synthetic workbook
    does not match the pinned digest, yet ``download()`` returns, leaves the bytes
    untouched and never reaches ``urlopen`` (the conftest guard would raise if it did).
    """
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    before = dest.read_bytes()
    assert hashlib.sha256(before).hexdigest() != m.DATA_SHA256
    dataset.download()
    assert dest.read_bytes() == before

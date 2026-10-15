# tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
"""Hermetic build of the Cachera 2023 CRI-SPA betaxanthin loader on a synthetic CSV.

``GA1_2_4_6.csv`` is written into ``<root>/raw/`` so PyG never calls ``download()``. No
genome is read (#1045): every row that names a gene and carries a level is a record
under the token the ``gene`` cell releases, upper-cased; a systematic name goes on
``KanMxDeletionPerturbation`` and any other name on ``ReleasedIdentifierPerturbation``.

Rows (gene, 24 h mean, std, count):

    TFC3       1.5    0.3   4      -> released symbol, SE 0.3 / sqrt(4) = 0.15, n 4
    YBR001C    0.5    0.1   1      -> placed leaf; count 1 -> SE NaN, n 1
    ycr001w    -0.25  (blank) (blank) -> YCR001W on the placed leaf; n = 1, SE NaN
    OLDNAME    2.0    0.6   3      -> released symbol as written, SE 0.6 / sqrt(3), n 3
    FLO8X      0.75   0.2   2      -> released symbol, SE 0.2 / sqrt(2)
    WT         0.0    0.1   4      -> the control row, no record
    0          1.0    0.1   4      -> no gene named, no record
    (blank)    1.0    0.1   4      -> gene NaN, no record
    NTH2       (blank) 0.1  4      -> level NaN, no record
    TFC3ALIAS  9.0    0.1   4      -> released symbol, a record of its own beside TFC3

Six records; the reference is the population-centered 0.0 for every record.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.scerevisiae import cachera2023 as m
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

_CSV_ROWS = [
    ["TFC3", "1.5", "0.3", "4"],
    ["YBR001C", "0.5", "0.1", "1"],
    ["ycr001w", "-0.25", "", ""],
    ["OLDNAME", "2.0", "0.6", "3"],
    ["FLO8X", "0.75", "0.2", "2"],
    ["WT", "0.0", "0.1", "4"],
    ["0", "1.0", "0.1", "4"],
    ["", "1.0", "0.1", "4"],
    ["NTH2", "", "0.1", "4"],
    ["TFC3ALIAS", "9.0", "0.1", "4"],
]


def _root(tmp_path: Path, slug: str = "betaxanthin_cachera2023") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    header = ["gene", m._LEVEL, m._STD, m._COUNT]
    (root / "raw" / m.DATA_FILENAME).write_text(
        "\n".join(",".join(r) for r in [header, *_CSV_ROWS]) + "\n"
    )
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.BetaxanthinCachera2023Dataset:
    return m.BetaxanthinCachera2023Dataset(root=str(_root(tmp_path)))


_ENVIRONMENT = Environment(
    media=m.CACHERA_YPD_G418, temperature=None, provenance_gaps=[m.TEMPERATURE_GAP]
)
_REFERENCE = MetaboliteExperimentReference(
    dataset_name="BetaxanthinCachera2023Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain="BY4741",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=MetabolitePhenotype(
        metabolite_level={"betaxanthin": 0.0},
        metabolite_level_se=None,
        n_replicates={"betaxanthin": 1},
        measurement_type=m.MEASUREMENT_TYPE,
    ),
).model_dump()
_PUBLICATION = Publication(
    pubmed_id="37572348",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37572348/",
    doi="10.1093/nar/gkad656",
    doi_url="https://doi.org/10.1093/nar/gkad656",
).model_dump()


def _experiment(gene: str, level: float, se: float, n: int) -> dict[str, Any]:
    return MetaboliteExperiment(
        dataset_name="BetaxanthinCachera2023Dataset",
        genotype=Genotype(
            perturbations=[m._released_deletion(gene), *m._betaxanthin_cassette()]
        ),
        environment=_ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level={"betaxanthin": level},
            metabolite_level_se={"betaxanthin": se},
            n_replicates={"betaxanthin": n},
            measurement_type=m.MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        ),
    ).model_dump()


def _pop_se(experiment: dict[str, Any]) -> float:
    """Remove and return the single betaxanthin SE so a NaN record can be compared."""
    se = experiment["phenotype"].pop("metabolite_level_se")
    assert list(se) == ["betaxanthin"]
    return cast(float, se["betaxanthin"])


def test_six_records_under_released_names_with_se_from_std_over_sqrt_count(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Ten rows give six records in source order; the three rows naming no gene or
    carrying no level and the ``WT`` control have none. A symbol is stored as written
    (OLDNAME, TFC3ALIAS: no annotation is asked whether they alias another gene), a
    systematic name on the placed leaf.
    """
    assert len(dataset) == 6
    assert dataset[0]["experiment"] == _experiment("TFC3", 1.5, 0.15, 4)
    assert dataset[3]["experiment"] == _experiment(
        "OLDNAME", 2.0, 0.6 / math.sqrt(3), 3
    )
    assert dataset[4]["experiment"] == _experiment("FLO8X", 0.75, 0.2 / math.sqrt(2), 2)
    assert dataset[5]["experiment"] == _experiment("TFC3ALIAS", 9.0, 0.05, 4)
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    kinds = {
        p["systematic_gene_name"]: p["perturbation_type"]
        for i in range(6)
        for p in dataset[i]["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] != "gene_addition"
    }
    assert kinds == {
        "TFC3": "released_identifier",
        "YBR001C": "kanmx_deletion",
        "YCR001W": "kanmx_deletion",
        "OLDNAME": "released_identifier",
        "FLO8X": "released_identifier",
        "TFC3ALIAS": "released_identifier",
    }


def test_single_count_rows_store_nan_se_and_n_one(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: with count 1 (YBR001C) or a blank count (ycr001w -> count 1) the SE is
    stored as ``nan`` inside the dict rather than ``None``, so these records carry a
    non-comparable SE; every other field matches the hand-built record. The lower-case
    ``ycr001w`` is upper-cased to its systematic name.
    """
    for i, (gene, level) in enumerate([("YBR001C", 0.5), ("YCR001W", -0.25)], start=1):
        actual = dataset[i]["experiment"]
        assert math.isnan(_pop_se(actual))
        expected = _experiment(gene, level, 0.0, 1)
        _pop_se(expected)
        assert actual == expected


def test_side_files(dataset: m.BetaxanthinCachera2023Dataset) -> None:
    """``data.csv`` carries gene, level, se (blank for NaN) and n; one shared reference
    covers all six records; the ledger names the four rows without a record.

    Finding: the gene set includes the cassette's ``CYP76AD1`` and ``DOD`` symbols plus
    ``YBR249C``/``YPR060C`` because ``compute_gene_set`` reads every perturbation.
    """
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "gene,level,se,n\n"
        "TFC3,1.5,0.15,4\n"
        "YBR001C,0.5,,1\n"
        "YCR001W,-0.25,,1\n"
        f"OLDNAME,2.0,{0.6 / math.sqrt(3)},3\n"
        f"FLO8X,0.75,{0.2 / math.sqrt(2)},2\n"
        "TFC3ALIAS,9.0,0.05,4\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "CYP76AD1",
        "DOD",
        "FLO8X",
        "OLDNAME",
        "TFC3",
        "TFC3ALIAS",
        "YBR001C",
        "YBR249C",
        "YCR001W",
        "YPR060C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3, 4, 5]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "betaxanthin_cachera2023"
    assert manifest["loader_class"] == "BetaxanthinCachera2023Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.cachera2023"
    ledger = m.DropLog.model_validate_json(
        (preprocess / "dropped_records.json").read_text()
    )
    assert (ledger.kept_records, ledger.dropped_records) == (6, 4)
    assert [(r.rule, r.items) for r in ledger.rules] == [
        (m.RULE_NO_GENE_OR_LEVEL, ["0", "nan", "NTH2"]),
        (m.RULE_CONTROL_ROW, ["WT"]),
    ]


def test_process_rejects_a_present_file_with_the_wrong_sha256(
    dataset: m.BetaxanthinCachera2023Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild over the synthetic CSV under the real build-time check raises
    ``RawSha256MismatchError`` naming the file and both digests, before any row is read.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    dest = Path(dataset.root) / "raw" / m.DATA_FILENAME
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected {m.DATA_SHA256}, observed {digest}"
    )

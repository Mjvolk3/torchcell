# tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_cachera2023_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_cachera2023_synthetic.py
"""Hermetic build of the Cachera 2023 CRI-SPA betaxanthin loader on a synthetic CSV.

``GA1_2_4_6.csv`` is written into ``<root>/raw/`` so PyG never calls ``download()``. The
genome stub implements ``resolve_gene_name`` (real ``GeneNameResolution`` objects) and
``feature_index["standard_to_ids"]``, which ``canonical_common_names`` reads: current ORFs
YAL001C (TFC3), YBR001C (NTH2), YCR001W (no standard name), YDR001C (NTH1); aliases
OLDNAME -> YDR001C and TFC3ALIAS -> YAL001C (renamed); FLO8X -> YER109C as a
``blocked_reading_frame`` non-gene feature; anything else retired.

Rows (gene, 24 h mean, std, count):

    TFC3       1.5    0.3   4      -> YAL001C, TFC3, SE 0.3 / sqrt(4) = 0.15, n 4
    YBR001C    0.5    0.1   1      -> NTH2 from the genome; count 1 -> SE NaN, n 1
    ycr001w    -0.25  (blank) (blank) -> no standard name -> stores the ORF; n = 1, SE NaN
    OLDNAME    2.0    0.6   3      -> YDR001C, NTH1, SE 0.6 / sqrt(3), n 3
    FLO8X      0.75   0.2   2      -> YER109C kept (non-gene feature), SE 0.2 / sqrt(2)
    WT         0.0    0.1   4      -> retired -> unresolved, dropped
    0          1.0    0.1   4      -> control row
    (blank)    1.0    0.1   4      -> gene NaN -> control row
    NTH2       (blank) 0.1  4      -> level NaN -> control row
    TFC3ALIAS  9.0    0.1   4      -> resolves to YAL001C, already seen -> collision

Five records; the reference is the population-centered 0.0 for every record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, cast

import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
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
from torchcell.datasets.scerevisiae import cachera2023 as m
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
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
_CURRENT = {"YAL001C", "YBR001C", "YCR001W", "YDR001C"}
_STANDARD = {"TFC3": ["YAL001C"], "NTH2": ["YBR001C"], "NTH1": ["YDR001C"]}
_ALIASES = {"OLDNAME": "YDR001C", "TFC3ALIAS": "YAL001C"}


class _StubGenome:
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.strip().upper()
        if upper in _CURRENT:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
            )
        if upper in _STANDARD:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_STANDARD[upper][0],
            )
        if upper in _ALIASES:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_ALIASES[upper],
            )
        if upper == "FLO8X":
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name="YER109C",
                feature_type="blocked_reading_frame",
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=upper
        )


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


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
    return m.BetaxanthinCachera2023Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(
    media=Media(name="SC", state="solid", is_synthetic=True),
    temperature=Temperature(value=30),
)
_REFERENCE = MetaboliteExperimentReference(
    dataset_name="BetaxanthinCachera2023Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4741"
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


def _experiment(
    orf: str, common: str, level: float, se: float, n: int
) -> dict[str, Any]:
    return MetaboliteExperiment(
        dataset_name="BetaxanthinCachera2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=common
                ),
                *m._betaxanthin_cassette(),
            ]
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


def test_five_records_with_genome_standard_names_and_se_from_std_over_sqrt_count(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Ten rows give five records in source order; the three control/NaN rows, ``WT``
    (unresolved) and ``TFC3ALIAS`` (ORF collision) are dropped. The stored common name
    is the genome's standard name (TFC3, NTH2, NTH1), or the ORF when none exists.
    """
    assert len(dataset) == 5
    assert dataset[0]["experiment"] == _experiment("YAL001C", "TFC3", 1.5, 0.15, 4)
    assert dataset[3]["experiment"] == _experiment(
        "YDR001C", "NTH1", 2.0, 0.6 / math.sqrt(3), 3
    )
    assert dataset[4]["experiment"] == _experiment(
        "YER109C", "YER109C", 0.75, 0.2 / math.sqrt(2), 2
    )
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION


def test_single_count_rows_store_nan_se_and_n_one(
    dataset: m.BetaxanthinCachera2023Dataset,
) -> None:
    """Finding: with count 1 (YBR001C) or a blank count (ycr001w -> count 1) the SE is
    stored as ``nan`` inside the dict rather than ``None``, so these records carry a
    non-comparable SE; every other field matches the hand-built record.
    """
    for i, (orf, common, level) in enumerate(
        [("YBR001C", "NTH2", 0.5), ("YCR001W", "YCR001W", -0.25)], start=1
    ):
        actual = dataset[i]["experiment"]
        assert math.isnan(_pop_se(actual))
        expected = _experiment(orf, common, level, 0.0, 1)
        _pop_se(expected)
        assert actual == expected


def test_side_files(dataset: m.BetaxanthinCachera2023Dataset) -> None:
    """``data.csv`` carries orf, common, level, se (blank for NaN) and n; one shared
    reference covers all five records.

    Finding: the gene set includes the cassette's ``CYP76AD1`` and ``DOD`` symbols plus
    ``YBR249C``/``YPR060C`` because ``compute_gene_set`` reads every perturbation.
    """
    assert dataset.experiment_class is MetaboliteExperiment
    assert dataset.reference_class is MetaboliteExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "orf,common,level,se,n\n"
        "YAL001C,TFC3,1.5,0.15,4\n"
        "YBR001C,NTH2,0.5,,1\n"
        "YCR001W,YCR001W,-0.25,,1\n"
        f"YDR001C,NTH1,2.0,{0.6 / math.sqrt(3)},3\n"
        f"YER109C,YER109C,0.75,{0.2 / math.sqrt(2)},2\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "CYP76AD1",
        "DOD",
        "YAL001C",
        "YBR001C",
        "YBR249C",
        "YCR001W",
        "YDR001C",
        "YER109C",
        "YPR060C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3, 4]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "betaxanthin_cachera2023"
    assert manifest["loader_class"] == "BetaxanthinCachera2023Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.cachera2023"


def test_requires_an_injected_genome(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Cachera2023 requires an injected SCerevisiaeGenome to resolve common gene "
            "names to systematic ORF ids (source uses common names)."
        ),
    ):
        m.BetaxanthinCachera2023Dataset(root=str(_root(tmp_path)), genome=None)


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

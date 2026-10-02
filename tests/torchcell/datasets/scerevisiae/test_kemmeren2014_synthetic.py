# tests/torchcell/datasets/scerevisiae/test_kemmeren2014_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_kemmeren2014_synthetic.py
"""Kemmeren 2014 microarray loader built end to end on in-memory GEOparse objects.

``raw/`` holds the six ``*_family.soft.gz`` names (placeholder bytes: ``process()`` never
opens them, their presence only stops PyG calling ``download()``), Table S1 as
``kemmeren2014_table_s1.xlsx`` (openpyxl), and pickled REAL ``GEOparse.GEOTypes.GSE``
objects built in memory (``GSM``/``GPL`` with a ``description`` column frame) at the
``<accession>.pkl`` paths ``process()`` checks first. The two deletion GSEs both have
one, so ``GEOparse.get_GEO`` (the network branch) is never reached. The four wildtype
series have pickles too (GSE42215 excepted) but ``process()`` no longer opens them
(#484); a test removes them and gets the same records. The genome is a stub with the
three attributes ``resolve_gene_name_comprehensive`` reads.

Channels follow GEO's own metadata, as on the real series: ``label_ch1`` is Cy5 and
``label_ch2`` Cy3 on every array, and ``source_name_ch1`` / ``source_name_ch2`` say
which channel holds the reference pool ("refpool", or "ref1" on GSE42217). The title
suffix is not consulted.

Platform probes: 1 YAL001C, 2 YBR001C, 3 Q0010, 4 NaN (unmapped). Table S1: YPL177C
CUP9 MATa, YHR127W HSN1 MATalpha, TLC1 (-> YNCB0010W) MATa, CMS1 (-> YLR003C) MATa,
YXX001W with mating type ``diploid`` (skipped).

Deletion samples (responsive GSE42527 then non-responsive GSE42526), signals listed
as YAL001C, YBR001C, Q0010:

    GSM1 "[HS1991] cup9-del-a"   refpool Cy5 4, 4, 1   deletion Cy3 2, 8, 0
    GSM2 "cup9-del-b"            deletion Cy5 8, 2, 4  refpool Cy3 4, 4, 1  (dye swap)
    GSM3 "Sample X"              deletion Cy5 3, 6, 1  refpool Cy3 3, 3, 2; the gene is
                                 named only in ``characteristics_ch1``
    GSM4 "wt control"            wildtype, ignored
    GSM5 "cdk8-del-a"            resolves through the shared reconciler to YPL042C,
                                 which has no Table S1 strain: not written
    GSM6 "zzz9-del-a"            retired name: unresolved

CUP9 -> YPL177C (BY4741): the log2 ratio is taken within each array, log2(deletion /
refpool). YAL001C 2/4 -> -1 and 8/4 -> +1, mean 0, sample SD sqrt(2), SE sqrt(2)/sqrt(2)
= 1, variance 2 (stored as sqrt(2)**2 = 2.0000000000000004); YBR001C the mirror image,
same statistics; Q0010 drops the array with the 0 signal and keeps 4/1 -> +2 alone (n =
1, SE and variance NaN). The linear ``expression`` is the mean deletion signal over the
kept arrays, 5, 5, 4, and the reference ``expression`` the mean refpool signal over the
same arrays, 4, 4, 1. HSN1 -> YHR127W (BY4742): 3/3 -> 0, 6/3 -> +1, 1/2 -> -1, all n
= 1, linear 3, 6, 1, refpool 3, 3, 2.

The reference ``n_replicates`` of a gene is the number of arrays whose refpool value
entered its reference ``expression`` mean, the same arrays as the mutant's count (#484):
CUP9 YAL001C 2, YBR001C 2, Q0010 1; HSN1 1 each. The wildtype arrays (MATa W1, W2, W3;
MATalpha W4) once gave counts of 3, 2, 2 and 1, 1, 1 here; they enter no record.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import openpyxl
import pandas as pd
import pytest
from GEOparse.GEOTypes import GPL, GSE, GSM

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import kemmeren2014 as m
from torchcell.sequence.genome.scerevisiae import GeneNameStatus, SCerevisiaeGenome
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameResolution

_DATASET = "MicroarrayKemmeren2014Dataset"
_NAN = "NaN"
_PROBES = {"1": "YAL001C", "2": "YBR001C", "3": "Q0010"}


class _StubGenome:
    """The three attributes ``resolve_gene_name_comprehensive`` reads."""

    gene_attribute_table = pd.DataFrame(
        {"ID": ["YPL177C"], "gene": ["CUP9"], "Alias": ["NONE1"]}
    )
    alias_to_systematic: dict[str, list[str]] = {"CDK8": ["YPL042C"]}

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        if name.upper() == "CDK8":
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.CURRENT,
                systematic_name="YPL042C",
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=name
        )


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _describe(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"description": [f"{c} column" for c in table.columns]}, index=table.columns
    )


def _gsm(
    name: str,
    title: str,
    cy5: list[float],
    cy3: list[float],
    refpool_in: str = "Cy5",
    test: str = "wt",
    reference: str = "refpool",
    characteristics: list[str] | None = None,
) -> GSM:
    """A two-channel array with GEO-style channel metadata.

    ``label_ch1`` is Cy5 and ``label_ch2`` Cy3, as on every real array; ``refpool_in``
    says which dye carries the reference (named ``reference``), and ``test`` names the
    other channel. ``characteristics`` are attached to the test channel.
    """
    table = pd.DataFrame(
        {
            "ID_REF": [1, 2, 3, 4],
            "VALUE": [0.0, 0.0, 0.0, 0.0],
            "Signal Norm_Cy5": [*cy5, 9.0],
            "Signal Norm_Cy3": [*cy3, 9.0],
        }
    )
    reference_channel = 1 if refpool_in == "Cy5" else 2
    test_channel = 3 - reference_channel
    metadata: dict[str, list[str]] = {
        "title": [title],
        "geo_accession": [name],
        "label_ch1": ["Cy5"],
        "label_ch2": ["Cy3"],
        f"source_name_ch{reference_channel}": [reference],
        f"source_name_ch{test_channel}": [test],
        f"characteristics_ch{test_channel}": characteristics or [],
    }
    return GSM(name=name, metadata=metadata, table=table, columns=_describe(table))


def _gpl() -> GPL:
    table = pd.DataFrame(
        {"ID": [1, 2, 3, 4], "ORF": ["YAL001C", "YBR001C", "Q0010", np.nan]}
    )
    return GPL(name="GPL11232", metadata={}, table=table, columns=_describe(table))


def _gse(name: str, gsms: list[GSM], with_platform: bool = True) -> GSE:
    return GSE(
        name=name,
        metadata={"title": [name]},
        gpls={"GPL11232": _gpl()} if with_platform else {},
        gsms={gsm.name: gsm for gsm in gsms},
    )


def _write_table_s1(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["orf name", "gene", "mating type"])
    sheet.append(["YPL177C", "CUP9", "MATa"])
    sheet.append(["YHR127W", "HSN1", "MATalpha"])
    sheet.append(["TLC1", "TLC1", "MATa"])
    sheet.append(["CMS1", None, "MATa"])
    sheet.append(["YXX001W", "XXX1", "diploid"])
    workbook.save(path)


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True)
    for accession in (
        "GSE42527",
        "GSE42526",
        "GSE42241",
        "GSE42240",
        "GSE42217",
        "GSE42215",
    ):
        name = f"{accession}_family.soft.gz"
        (raw / name).write_bytes(b"placeholder: process() reads the .pkl")
    _write_table_s1(raw / "kemmeren2014_table_s1.xlsx")
    gses = {
        "GSE42527": _gse(
            "GSE42527",
            [
                _gsm(
                    "GSM1",
                    "[HS1991] cup9-del-a",
                    [4.0, 4.0, 1.0],
                    [2.0, 8.0, 0.0],
                    refpool_in="Cy5",
                    test="cup9-del",
                ),
                _gsm(
                    "GSM2",
                    "cup9-del-b",
                    [8.0, 2.0, 4.0],
                    [4.0, 4.0, 1.0],
                    refpool_in="Cy3",
                    test="cup9-del",
                ),
            ],
        ),
        "GSE42526": _gse(
            "GSE42526",
            [
                _gsm(
                    "GSM3",
                    "Sample X",
                    [3.0, 6.0, 1.0],
                    [3.0, 3.0, 2.0],
                    refpool_in="Cy3",
                    test="hsn1-del",
                    characteristics=["strain: BY4742", "genotype/variation: hsn1-del"],
                ),
                _gsm(
                    "GSM4",
                    "wt control",
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    refpool_in="Cy5",
                    test="wt",
                    characteristics=["genotype/variation: wt"],
                ),
                _gsm(
                    "GSM5",
                    "cdk8-del-a",
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    refpool_in="Cy5",
                    test="cdk8-del",
                ),
                _gsm(
                    "GSM6",
                    "zzz9-del-a",
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    refpool_in="Cy5",
                    test="zzz9-del",
                ),
            ],
            with_platform=False,
        ),
        "GSE42241": _gse(
            "GSE42241",
            [
                _gsm(
                    "W1",
                    "wt-matA-1-a",
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    refpool_in="Cy5",
                    test="wt-matA",
                ),
                _gsm(
                    "W2",
                    "wt-matA-1-b",
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    refpool_in="Cy3",
                    test="wt-matA",
                ),
            ],
        ),
        "GSE42240": _gse(
            "GSE42240",
            [
                _gsm(
                    "W3",
                    "wt-matA-THM012-a",
                    [3.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    refpool_in="Cy5",
                    test="wt-matA",
                )
            ],
        ),
        "GSE42217": _gse(
            "GSE42217",
            [
                _gsm(
                    "W4",
                    "wt-htp07-a",
                    [5.0, 5.0, 5.0],
                    [0.0, 0.0, 0.0],
                    refpool_in="Cy5",
                    test="wt-htp07-a",
                    reference="ref1",
                )
            ],
        ),
    }
    for accession, gse in gses.items():
        with open(raw / f"{accession}.pkl", "wb") as handle:
            pickle.dump(gse, handle)


@pytest.fixture
def dataset(tmp_path: Path) -> m.MicroarrayKemmeren2014Dataset:
    root = tmp_path / "kemmeren"
    _write_raw(root / "raw")
    return m.MicroarrayKemmeren2014Dataset(root=str(root), genome=_genome())


def _nan_safe(value: Any) -> Any:
    """Replace float NaN by a string so two dumps compare exactly (NaN != NaN)."""
    if isinstance(value, dict):
        return {k: _nan_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return _NAN
    return value


_ENVIRONMENT = Environment(
    media=Media(name="SC", state="liquid", is_synthetic=True),
    temperature=Temperature(value=30),
)
_PUBLICATION = Publication(
    pubmed_id="24766815",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/24766815/",
    doi="10.1016/j.cell.2014.02.054",
    doi_url="https://doi.org/10.1016/j.cell.2014.02.054",
)


def _experiment(
    orf: str,
    log2: dict[str, float],
    se: dict[str, float],
    var: dict[str, float],
    n: dict[str, int],
    linear: dict[str, float],
) -> MicroarrayExpressionExperiment:
    return MicroarrayExpressionExperiment(
        dataset_name=_DATASET,
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=MicroarrayExpressionPhenotype(
            expression=linear,
            expression_log2_ratio=log2,
            expression_log2_ratio_se=se,
            expression_log2_ratio_variance=var,
            n_replicates=n,
        ),
    )


def _reference(
    strain: str, refpool: dict[str, float], n: dict[str, int]
) -> MicroarrayExpressionExperimentReference:
    return MicroarrayExpressionExperimentReference(
        dataset_name=_DATASET,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MicroarrayExpressionPhenotype(
            expression=refpool,
            expression_log2_ratio=dict.fromkeys(refpool, 0.0),
            n_replicates=n,
        ),
    )


_NANF = float("nan")
# The variance is stored as sd**2 with sd = np.std(ddof=1) = sqrt(2), which squares back
# to 2.0000000000000004 in floating point, not 2.0.
_SD2 = math.sqrt(2.0)
_CUP9 = _experiment(
    "YPL177C",
    {"Q0010": 2.0, "YAL001C": 0.0, "YBR001C": 0.0},
    {"Q0010": _NANF, "YAL001C": 1.0, "YBR001C": 1.0},
    {"Q0010": _NANF, "YAL001C": _SD2**2, "YBR001C": _SD2**2},
    {"Q0010": 1, "YAL001C": 2, "YBR001C": 2},
    {"Q0010": 4.0, "YAL001C": 5.0, "YBR001C": 5.0},
)
_HSN1 = _experiment(
    "YHR127W",
    {"Q0010": -1.0, "YAL001C": 0.0, "YBR001C": 1.0},
    dict.fromkeys(("Q0010", "YAL001C", "YBR001C"), _NANF),
    dict.fromkeys(("Q0010", "YAL001C", "YBR001C"), _NANF),
    {"Q0010": 1, "YAL001C": 1, "YBR001C": 1},
    {"Q0010": 1.0, "YAL001C": 3.0, "YBR001C": 6.0},
)
_REF_CUP9 = _reference(
    "BY4741",
    {"Q0010": 1.0, "YAL001C": 4.0, "YBR001C": 4.0},
    {"Q0010": 1, "YAL001C": 2, "YBR001C": 2},
)
_REF_HSN1 = _reference(
    "BY4742",
    {"Q0010": 2.0, "YAL001C": 3.0, "YBR001C": 3.0},
    {"Q0010": 1, "YAL001C": 1, "YBR001C": 1},
)


def test_two_deletions_build_with_within_array_log2_statistics(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """Record 0 is CUP9 (two dye-swapped arrays, MATa), record 1 HSN1 (one array,
    MATalpha, named only in ``characteristics_ch1``); CDK8 resolves but has no Table S1
    strain and ZZZ9 does not resolve, so neither is written. The deleted gene is not
    on the platform, so the ratios are the docstring's within-array values.
    """
    assert len(dataset) == 2
    assert _nan_safe(dataset[0]["experiment"]) == _nan_safe(_CUP9.model_dump())
    assert _nan_safe(dataset[1]["experiment"]) == _nan_safe(_HSN1.model_dump())
    assert dataset[0]["reference"] == _REF_CUP9.model_dump()
    assert dataset[1]["reference"] == _REF_HSN1.model_dump()
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()


def test_side_files_gene_set_reference_index_and_sample_table(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """One reference per strain; ``data.csv`` lists all six deletion-GSE samples with
    the resolved ORF (empty for the wildtype and the unresolved title).
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YHR127W",
        "YPL177C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0], [1]]
    assert [entry["reference"] for entry in index] == [
        _REF_CUP9.model_dump(),
        _REF_HSN1.model_dump(),
    ]
    samples = pd.read_csv(preprocess / "data.csv", keep_default_na=False)
    assert samples.to_dict(orient="list") == {
        "geo_accession": ["GSM1", "GSM2", "GSM3", "GSM4", "GSM5", "GSM6"],
        "title": [
            "[HS1991] cup9-del-a",
            "cup9-del-b",
            "Sample X",
            "wt control",
            "cdk8-del-a",
            "zzz9-del-a",
        ],
        "systematic_gene_name": ["YPL177C", "YPL177C", "YHR127W", "", "YPL042C", ""],
        "is_deletion": [True, True, True, False, True, True],
        "is_wildtype": [False, False, False, True, False, False],
        "is_responsive_mutant": [True, True, False, False, False, False],
    }
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "kemmeren"
    assert manifest["loader_class"] == _DATASET
    assert dataset.experiment_class is MicroarrayExpressionExperiment
    assert dataset.reference_class is MicroarrayExpressionExperimentReference
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame


def test_mating_type_map_reads_table_s1_and_rewrites_common_names(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """TLC1 and CMS1 in the orf column become YNCB0010W and YLR003C; ``diploid`` is
    neither mating type and is skipped.
    """
    assert dataset._load_mating_type_map() == (
        {
            "YPL177C": "BY4741",
            "YHR127W": "BY4742",
            "YNCB0010W": "BY4741",
            "YLR003C": "BY4741",
        },
        {"CUP9": "YPL177C", "HSN1": "YHR127W", "TLC1": "YNCB0010W", "CMS1": "YLR003C"},
    )


def test_a_table_s1_without_an_orf_column_yields_empty_maps(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """Finding: ``_load_mating_type_map`` raises ``ValueError("Required 'orf name'
    column not found ...")`` inside a ``try`` whose ``except Exception`` only logs, so
    a Table S1 missing its required column returns two empty maps instead of failing.
    """
    workbook = openpyxl.Workbook()
    workbook.active.append(["gene", "mating type"])
    workbook.active.append(["CUP9", "MATa"])
    workbook.save(Path(dataset.raw_dir) / "kemmeren2014_table_s1.xlsx")
    assert dataset._load_mating_type_map() == ({}, {})
    (Path(dataset.raw_dir) / "kemmeren2014_table_s1.xlsx").unlink()
    with pytest.raises(FileNotFoundError, match="Supplementary Table S1 not found at"):
        dataset._load_mating_type_map()


def test_channels_come_from_geo_metadata_not_from_the_title() -> None:
    """``_channel_columns`` reads ``source_name_ch*`` and ``label_ch*``: a "-a" title
    with the refpool in Cy5 gives the test in Cy3, a "-b" title of a gene whose own
    name carries ``-a`` (``ycr087c-a-del-b``) with the refpool in Cy3 gives the test in
    Cy5, and "ref1" counts as the reference. Two reference channels, none, and a label
    pair other than Cy5 + Cy3 raise ``ValueError``; a deletion named ``ref2-del`` is
    not mistaken for a reference.
    """
    columns = m.MicroarrayKemmeren2014Dataset._channel_columns
    ones = [1.0, 1.0, 1.0]
    assert columns(_gsm("A", "cup9-del-a", ones, ones, "Cy5", "cup9-del")) == (
        "Signal Norm_Cy3",
        "Signal Norm_Cy5",
    )
    assert columns(
        _gsm("B", "ycr087c-a-del-b", ones, ones, "Cy3", "ycr087c-a-del")
    ) == ("Signal Norm_Cy5", "Signal Norm_Cy3")
    assert columns(_gsm("C", "wt-htp07-a", ones, ones, "Cy5", "wt", "ref1")) == (
        "Signal Norm_Cy3",
        "Signal Norm_Cy5",
    )
    assert columns(_gsm("D", "ref2-del-a", ones, ones, "Cy5", "ref2-del")) == (
        "Signal Norm_Cy3",
        "Signal Norm_Cy5",
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "E: expected the reference pool in exactly one channel, source names "
            "['refpool', 'refpool']"
        ),
    ):
        columns(_gsm("E", "x", ones, ones, "Cy5", "refpool"))
    with pytest.raises(ValueError, match="F: expected the reference pool"):
        columns(_gsm("F", "x", ones, ones, "Cy5", "cup9-del", reference="wt"))
    swapped = _gsm("G", "x", ones, ones, "Cy5", "cup9-del")
    swapped.metadata["label_ch2"] = ["Cy5"]
    with pytest.raises(
        ValueError,
        match=re.escape("G: channel labels ['Cy5', 'Cy5'] are not one Cy5 and one Cy3"),
    ):
        columns(swapped)


def test_extract_channels_pairs_the_two_signals_of_each_row() -> None:
    """``ycr087c-a-del-b`` with the refpool in Cy3: the deletion values are the Cy5
    column and the refpool values the Cy3 column, row by row, the unmapped probe 4
    skipped; a missing signal column raises.
    """
    extract = m.MicroarrayKemmeren2014Dataset._extract_channels_from_gsm_static
    gsm = _gsm(
        "G", "ycr087c-a-del-b", [8.0, 2.0, 4.0], [4.0, 4.0, 1.0], "Cy3", "ycr087c-a-del"
    )
    deletion, refpool = extract(gsm, _PROBES)
    assert dict(deletion) == {"Q0010": 4.0, "YAL001C": 8.0, "YBR001C": 2.0}
    assert dict(refpool) == {"Q0010": 1.0, "YAL001C": 4.0, "YBR001C": 4.0}
    gsm.table = gsm.table.drop(columns=["Signal Norm_Cy3"])
    with pytest.raises(
        ValueError, match=re.escape("G: column 'Signal Norm_Cy3' missing")
    ):
        extract(gsm, _PROBES)


def test_replicate_pairs_are_one_pair_per_array_in_array_order() -> None:
    """GSM1 (refpool Cy5 4, 4, 1; deletion Cy3 2, 8, 0) then GSM2 (deletion Cy5 8, 2,
    4; refpool Cy3 4, 4, 1) give per gene the (deletion, refpool) pairs in that order,
    the 0 signal kept here and dropped later by ``create_expression_experiment``.
    """
    collect = m.MicroarrayKemmeren2014Dataset._collect_replicate_pairs_static
    first = _gsm(
        "GSM1", "cup9-del-a", [4.0, 4.0, 1.0], [2.0, 8.0, 0.0], "Cy5", "cup9-del"
    )
    second = _gsm(
        "GSM2", "cup9-del-b", [8.0, 2.0, 4.0], [4.0, 4.0, 1.0], "Cy3", "cup9-del"
    )
    assert dict(collect([first, second], _PROBES)) == {
        "Q0010": [(0.0, 1.0), (4.0, 1.0)],
        "YAL001C": [(2.0, 4.0), (8.0, 4.0)],
        "YBR001C": [(8.0, 4.0), (2.0, 4.0)],
    }
    assert dict(collect([], _PROBES)) == {}


def test_channel_check_counts_arrays_with_the_deleted_gene_depleted() -> None:
    """For YAL001C (probe 1) on two arrays: 2/4 on the first is depleted, 8/4 on the
    second is not, so the fraction is 0.5; a gene with no probe on the platform, or an
    array with a 0 signal at that probe, is not counted (NaN when none is).
    """
    check = m.MicroarrayKemmeren2014Dataset._validate_channel_assignment
    depleted = _gsm("A", "yal001c-del-a", [4.0, 4.0, 1.0], [2.0, 8.0, 0.0], "Cy5", "x")
    raised = _gsm("B", "yal001c-del-b", [8.0, 2.0, 4.0], [4.0, 4.0, 1.0], "Cy3", "x")
    zero = _gsm("C", "yal001c-del-a", [4.0, 4.0, 1.0], [0.0, 8.0, 0.0], "Cy5", "x")
    assert check({"YAL001C": [depleted, raised, zero]}, _PROBES) == 0.5
    assert check({"YAL001C": [depleted]}, _PROBES) == 1.0
    assert math.isnan(check({"YPL177C": [depleted], "YAL001C": [zero]}, _PROBES))


def test_reference_n_replicates_count_the_arrays_in_the_refpool_mean(
    tmp_path: Path,
) -> None:
    """#484: the reference ``n_replicates`` describes the stored reference value, the
    refpool mean over the mutant's own arrays, so it equals the mutant's own count gene
    by gene (CUP9: Q0010 1 after the 0-signal array is dropped, YAL001C 2, YBR001C 2),
    never a wildtype-series count (the fixture's MATa series would give 3, 2, 2, and the
    real series 28 and 400). With the wildtype pickles removed the build is identical,
    because no stored value reads them.
    """
    root = tmp_path / "kemmeren"
    _write_raw(root / "raw")
    for accession in ("GSE42241", "GSE42240", "GSE42217"):
        (root / "raw" / f"{accession}.pkl").unlink()
    dataset = m.MicroarrayKemmeren2014Dataset(root=str(root), genome=_genome())
    assert len(dataset) == 2
    for i in range(2):
        experiment_n = dataset[i]["experiment"]["phenotype"]["n_replicates"]
        reference = dataset[i]["reference"]["phenotype_reference"]
        assert reference["n_replicates"] == experiment_n
        assert set(reference["n_replicates"]) == set(reference["expression"])
    assert dataset[0]["reference"]["phenotype_reference"]["n_replicates"] == {
        "Q0010": 1,
        "YAL001C": 2,
        "YBR001C": 2,
    }
    assert dataset[0]["reference"] == _REF_CUP9.model_dump()
    assert dataset[1]["reference"] == _REF_HSN1.model_dump()


def test_resolution_passes_in_priority_order(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """Special map (LUG1 -> YCR087C-A even without a strain), Table S1 common name,
    the gene-attribute table's ``ID`` and ``Alias`` columns, a direct Table S1 key, the
    shared reconciler, and ``None`` for a retired name.
    """
    strains = {"YPL177C": "BY4741", "YXX009W": "BY4742"}
    resolve = dataset.resolve_gene_name_comprehensive
    dataset.resolved_by_alias = dataset.resolved_by_excel = 0
    dataset.resolved_by_gene_table = dataset.resolved_by_shared_reconciler = 0
    dataset.unresolved_genes = 0
    assert resolve("lug1", {}, strains) == "YCR087C-A"
    assert resolve("cup9", {"CUP9": "YPL177C"}, strains) == "YPL177C"
    assert resolve("YPL177C", {}, strains) == "YPL177C"
    assert resolve("none1", {}, strains) == "YPL177C"
    assert resolve("yxx009w", {}, strains) == "YXX009W"
    assert resolve("cdk8", {}, strains) == "YPL042C"
    assert resolve("zzz9", {}, strains) is None
    assert (
        dataset.resolved_by_alias,
        dataset.resolved_by_excel,
        dataset.resolved_by_gene_table,
        dataset.resolved_by_shared_reconciler,
        dataset.unresolved_genes,
    ) == (1, 2, 2, 1, 1)


def test_create_expression_experiment_from_pairs() -> None:
    """No strain raises; no pair, or only pairs with a non-positive signal, returns the
    ``(None, None, None)`` skip; pairs (2, 4) and (8, 4) give log2 -1 and +1, mean 0,
    SE 1, variance 2.0000000000000004, n 2, linear 5, refpool 4, and the reference
    ``n_replicates`` is the 2 arrays of that refpool mean.
    """
    build = m.MicroarrayKemmeren2014Dataset.create_expression_experiment
    with pytest.raises(
        ValueError,
        match=re.escape("Strain (BY4741 or BY4742) must be specified in sample_info"),
    ):
        build("d", {"systematic_gene_name": "YAL001C"}, {})
    info = {"systematic_gene_name": "YAL001C", "strain": "BY4741"}
    assert build("d", info, {}) == (None, None, None)
    assert build("d", info, {"YAL001C": [(0.0, 1.0), (1.0, 0.0)]}) == (None, None, None)
    experiment, reference, publication = build(
        "d", info, {"YAL001C": [(2.0, 4.0), (8.0, 4.0)]}
    )
    assert experiment.phenotype.model_dump() == {
        "graph_level": "node",
        "label_name": "expression_log2_ratio",
        "label_statistic_name": "expression_log2_ratio_se",
        "expression": {"YAL001C": 5.0},
        "expression_log2_ratio": {"YAL001C": 0.0},
        "expression_log2_ratio_se": {"YAL001C": 1.0},
        "expression_log2_ratio_variance": {"YAL001C": _SD2**2},
        "n_replicates": {"YAL001C": 2},
        "provenance_gaps": [],
    }
    assert reference.phenotype_reference.model_dump() == {
        "graph_level": "node",
        "label_name": "expression_log2_ratio",
        "label_statistic_name": "expression_log2_ratio_se",
        "expression": {"YAL001C": 4.0},
        "expression_log2_ratio": {"YAL001C": 0.0},
        "expression_log2_ratio_se": None,
        "expression_log2_ratio_variance": None,
        "n_replicates": {"YAL001C": 2},
        "provenance_gaps": [],
    }
    assert reference.genome_reference.strain == "BY4741"
    assert publication == _PUBLICATION


def test_parallel_build_writes_the_same_records(tmp_path: Path) -> None:
    """``process_workers=1`` routes through ``_process_parallel`` and the static batch
    helpers (one worker process, batches of one gene); the two stored records, both
    references and the publication equal the sequential build's.
    """
    root = tmp_path / "kemmeren"
    _write_raw(root / "raw")
    dataset = m.MicroarrayKemmeren2014Dataset(
        root=str(root), genome=_genome(), process_workers=1, batch_size=1
    )
    assert len(dataset) == 2
    assert _nan_safe(dataset[0]["experiment"]) == _nan_safe(_CUP9.model_dump())
    assert _nan_safe(dataset[1]["experiment"]) == _nan_safe(_HSN1.model_dump())
    assert dataset[0]["reference"] == _REF_CUP9.model_dump()
    assert dataset[1]["reference"] == _REF_HSN1.model_dump()
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()

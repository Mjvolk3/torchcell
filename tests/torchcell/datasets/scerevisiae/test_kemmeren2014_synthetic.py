# tests/torchcell/datasets/scerevisiae/test_kemmeren2014_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_kemmeren2014_synthetic.py
"""Kemmeren 2014 microarray loader built end to end on in-memory GEOparse objects.

``raw/`` holds the six ``*_family.soft.gz`` names (placeholder bytes: ``process()`` never
opens them, their presence only stops PyG calling ``download()``), Table S1 as
``kemmeren2014_table_s1.xlsx`` (openpyxl), and pickled REAL ``GEOparse.GEOTypes.GSE``
objects built in memory (``GSM``/``GPL`` with a ``description`` column frame) at the
``<accession>.pkl`` paths ``process()`` checks first. GSE42215 (MATalpha flask) is left
without a pickle; the WT loop skips a missing pickle, and the two deletion GSEs both
have one, so ``GEOparse.get_GEO`` (the network branch) is never reached. The genome is a
stub with the three attributes ``resolve_gene_name_comprehensive`` reads.

Platform probes: 1 YAL001C, 2 YBR001C, 3 Q0010, 4 NaN (unmapped). Table S1: YPL177C
CUP9 MATa, YHR127W HSN1 MATalpha, TLC1 (-> YNCB0010W) MATa, CMS1 (-> YLR003C) MATa,
YXX001W with mating type ``diploid`` (skipped).

Deletion samples (responsive GSE42527 then non-responsive GSE42526):

    GSM1 "[HS1991] cup9-del-a"   deletion Cy5 2, 8, 0   refpool Cy3 4, 4, 1
    GSM2 "cup9-del-b"            deletion Cy3 8, 2, 4   refpool Cy5 4, 4, 1  (dye swap)
    GSM3 "Sample X", ch2 hsn1-del deletion Cy5 3, 6, 1  refpool Cy3 3, 3, 2  (default)
    GSM4 "wt control"            wildtype, ignored
    GSM5 "cdk8-del-a"            resolves through the shared reconciler to YPL042C,
                                 which has no Table S1 strain: not written
    GSM6 "zzz9-del-a"            retired name: unresolved

CUP9 -> YPL177C (BY4741): refpool averaged over GSM1 Cy3 and GSM2 Cy5 = 4, 4, 1. Per
replicate log2 ratio -log2(deletion / refpool): YAL001C 2/4 -> +1 and 8/4 -> -1, mean 0,
sample SD sqrt(2), SE sqrt(2)/sqrt(2) = 1, variance 2; YBR001C the mirror image, same
statistics (the variance is stored as sqrt(2)**2 = 2.0000000000000004);
Q0010 drops the 0 replicate and keeps 4/1 -> -2 alone (n = 1, SE and variance
NaN); linear means 5, 5, 2. HSN1 -> YHR127W (BY4742): 3/3 -> 0, 6/3 -> -1, 1/2 -> +1,
all n = 1.

WT refpool replicate counts (positive refpool channel per gene): MATa W1 "refpool vs wt"
(Cy5 1, 1, 1), W2 "wt vs refpool" (Cy3 2, 0, 2), W3 "wt_b" (Cy5 3, 3, 0) -> YAL001C 3,
YBR001C 2, Q0010 2; MATalpha W4 "plain" (Cy3 5, 5, 5) -> 1 each.
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
    characteristics: list[str] | None = None,
) -> GSM:
    table = pd.DataFrame(
        {
            "ID_REF": [1, 2, 3, 4],
            "VALUE": [0.0, 0.0, 0.0, 0.0],
            "Signal Norm_Cy5": [*cy5, 9.0],
            "Signal Norm_Cy3": [*cy3, 9.0],
        }
    )
    metadata = {
        "title": [title],
        "geo_accession": [name],
        "characteristics_ch2": characteristics or [],
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
                _gsm("GSM1", "[HS1991] cup9-del-a", [2.0, 8.0, 0.0], [4.0, 4.0, 1.0]),
                _gsm("GSM2", "cup9-del-b", [4.0, 4.0, 1.0], [8.0, 2.0, 4.0]),
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
                    ["strain: BY4742", "genotype/variation: hsn1-del"],
                ),
                _gsm(
                    "GSM4",
                    "wt control",
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    ["genotype/variation: refpool"],
                ),
                _gsm("GSM5", "cdk8-del-a", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
                _gsm("GSM6", "zzz9-del-a", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            ],
            with_platform=False,
        ),
        "GSE42241": _gse(
            "GSE42241",
            [
                _gsm("W1", "refpool vs wt", [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
                _gsm("W2", "wt vs refpool", [0.0, 0.0, 0.0], [2.0, 0.0, 2.0]),
            ],
        ),
        "GSE42240": _gse(
            "GSE42240", [_gsm("W3", "wt_b", [3.0, 3.0, 0.0], [0.0, 0.0, 0.0])]
        ),
        "GSE42217": _gse(
            "GSE42217", [_gsm("W4", "plain", [0.0, 0.0, 0.0], [5.0, 5.0, 5.0])]
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
    {"Q0010": -2.0, "YAL001C": 0.0, "YBR001C": 0.0},
    {"Q0010": _NANF, "YAL001C": 1.0, "YBR001C": 1.0},
    {"Q0010": _NANF, "YAL001C": _SD2**2, "YBR001C": _SD2**2},
    {"Q0010": 1, "YAL001C": 2, "YBR001C": 2},
    {"Q0010": 2.0, "YAL001C": 5.0, "YBR001C": 5.0},
)
_HSN1 = _experiment(
    "YHR127W",
    {"Q0010": 1.0, "YAL001C": 0.0, "YBR001C": -1.0},
    dict.fromkeys(("Q0010", "YAL001C", "YBR001C"), _NANF),
    dict.fromkeys(("Q0010", "YAL001C", "YBR001C"), _NANF),
    {"Q0010": 1, "YAL001C": 1, "YBR001C": 1},
    {"Q0010": 1.0, "YAL001C": 3.0, "YBR001C": 6.0},
)
_REF_CUP9 = _reference(
    "BY4741",
    {"Q0010": 1.0, "YAL001C": 4.0, "YBR001C": 4.0},
    {"Q0010": 2, "YAL001C": 3, "YBR001C": 2},
)
_REF_HSN1 = _reference(
    "BY4742",
    {"Q0010": 2.0, "YAL001C": 3.0, "YBR001C": 3.0},
    {"Q0010": 1, "YAL001C": 1, "YBR001C": 1},
)


def test_two_deletions_build_with_log2_statistics_per_replicate(
    dataset: m.MicroarrayKemmeren2014Dataset,
) -> None:
    """Record 0 is CUP9 (two dye-swapped arrays, MATa), record 1 HSN1 (one array,
    MATalpha, named only in ``characteristics_ch2``); CDK8 resolves but has no Table S1
    strain and ZZZ9 does not resolve, so neither is written.
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
    column not found ...")`` at kemmeren2014.py line 1591 inside a ``try`` whose
    ``except Exception`` (line 1700) only logs, so a Table S1 missing its required
    column returns two empty maps instead of failing.
    """
    workbook = openpyxl.Workbook()
    workbook.active.append(["gene", "mating type"])
    workbook.active.append(["CUP9", "MATa"])
    workbook.save(Path(dataset.raw_dir) / "kemmeren2014_table_s1.xlsx")
    assert dataset._load_mating_type_map() == ({}, {})
    (Path(dataset.raw_dir) / "kemmeren2014_table_s1.xlsx").unlink()
    with pytest.raises(FileNotFoundError, match="Supplementary Table S1 not found at"):
        dataset._load_mating_type_map()


def test_a_dye_swap_title_containing_dash_a_is_read_as_standard_orientation() -> None:
    """Finding: the channel rule at kemmeren2014.py line 902 tests ``"-a" in title``
    before ``"-b"``, so a dye-swapped ``-b`` array of a gene whose name itself carries
    ``-A`` (``ycr087c-a-del-b``) reads the DELETION from Cy5, which on a dye swap is the
    refpool channel.

    The channel rule itself is also contradicted by the source. The audit read GEO's
    own channel labels: on ``-a`` arrays GEO puts the reference pool in Cy5 and the
    deletion in Cy3 (the reverse on ``-b``), the opposite of the loader's rule
    (kemmeren2014.py lines 902-907: ``-a`` -> deletion in Cy5). On the real data the
    loader therefore reads the reference pool as the deletion on 2594 of 2633 arrays.
    The fixture here encodes the loader's belief (deletion Cy5 on ``-a``), so these
    tests pin code behavior, not biology.
    """
    gsm = _gsm("G", "ycr087c-a-del-b", [4.0, 4.0, 1.0], [8.0, 2.0, 4.0])
    probes = {"1": "YAL001C", "2": "YBR001C", "3": "Q0010"}
    extract = m.MicroarrayKemmeren2014Dataset._extract_expression_from_gsm_static
    assert dict(extract(gsm, probes)) == {"Q0010": 1.0, "YAL001C": 4.0, "YBR001C": 4.0}
    plain = _gsm("G", "cup9-del-b", [4.0, 4.0, 1.0], [8.0, 2.0, 4.0])
    assert dict(extract(plain, probes)) == {
        "Q0010": 4.0,
        "YAL001C": 8.0,
        "YBR001C": 2.0,
    }


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


def test_create_expression_experiment_requires_a_strain_and_a_refpool() -> None:
    build = m.MicroarrayKemmeren2014Dataset.create_expression_experiment
    with pytest.raises(
        ValueError,
        match=re.escape("Strain (BY4741 or BY4742) must be specified in sample_info"),
    ):
        build("d", {"systematic_gene_name": "YAL001C"}, {}, {}, {})
    info = {"systematic_gene_name": "YAL001C", "strain": "BY4741"}
    assert build("d", info, {"YAL001C": [1.0]}, {}, {}) == (None, None, None)
    assert build("d", info, {"YAL001C": [0.0]}, {"YAL001C": 1.0}, {}) == (
        None,
        None,
        None,
    )


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

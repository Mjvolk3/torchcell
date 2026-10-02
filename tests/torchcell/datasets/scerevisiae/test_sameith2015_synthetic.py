# tests/torchcell/datasets/scerevisiae/test_sameith2015_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_sameith2015_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_sameith2015_synthetic.py
"""Sameith 2015 single- and double-mutant loaders built on in-memory GEOparse objects.

Each root's ``raw/`` holds ``GSE42536_family.soft.gz`` (placeholder bytes, never
opened), the supplementary workbook ``12915_2015_222_MOESM1_ESM.xlsx`` (openpyxl, sheets
``Single mutants - info`` and ``Double mutants - info``) and ``GSE42536.pkl``, a pickled
REAL ``GEOparse.GEOTypes.GSE`` built in memory, so ``process()`` loads the pickle and the
``GEOparse.get_GEO`` network branch is never reached. The genome is a stub carrying the
``gene_attribute_table`` / ``alias_to_systematic`` pair ``_convert_to_systematic`` reads.

Platform probes: 1 YAL001C, 2 YBR001C, 3 YCR001W. The mutant channel is Cy5 unless
``source_name_ch1`` names the refpool, then Cy3; log2 ratio = log2(mutant / refpool),
dropped where either channel is 0 (which then fails validation, see the Finding test).

Single-mutant GSE (title: Cy5 / Cy3, source):

    S1 "yal001c-del-a"            2 4 8 / 1 4 2             -> log2 1 0 2
    S2 "yal001c-del-b"            2 2 2 / 8 2 2, refpool    -> mutant 8 2 2, log2 2 0 0
    S3 "nth2-del" (NTH2 -> YBR001C via the gene table)
                                  4 1 1 / 2 2 1             -> log2 1 -1 0
    S4 "wt-a"                     wildtype
    S5 "swt1-del-a"               wildtype by substring (Finding)
    S6 "ycr001w-del+ydr001c-del"  double-deletion title: dropped (#479)
    S7 "ycr001w-del+zzz9-del"     "+" with an unresolved partner: dropped (#479)
    S8 "yzz999w-del"              no gene name: ignored
    S9 "yer001w-del"              1 1 1 / 1 1 1             -> not in the SI list, kept

YAL001C statistics over S1, S2: mutant mean 5 3 5; refpool mean 1.5 3 2; log2 mean 1.5
0 1, sample SD sqrt(0.5), 0, sqrt(2), so variance sd**2 = 0.5000000000000001, 0.0,
2.0000000000000004 and SE sd/sqrt(2) = 0.5, 0.0, 1.0. Single-array groups store NaN SE
and variance.

Double-mutant GSE: S4 wildtype, S1 (one gene, ignored), D1 "ycr001w-del+ydr001c-del"
(2 2 2 / 1 1 1) and D2 "ycr001w-del+ydr001c-del-b" (refpool in ch1, mutant Cy3 4 4 4,
refpool 1 1 1) -> pair (YCR001W, YDR001C), log2 1.5 per gene with the same SD as above;
D3 "yal001c-del+nth2-del" (1 2 4 / 1 1 1, log2 0 1 2, n = 1) -> pair (YAL001C, YBR001C);
D4 "yel001c-del+yfl001w-del" is a pair that failed curation and is skipped.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, cast

import openpyxl
import pandas as pd
import pytest
from GEOparse.GEOTypes import GPL, GSE, GSM

from torchcell.datamodels.schema import (
    Environment,
    GenePerturbationType,
    Genotype,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import sameith2015 as m
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_SM = "SmMicroarraySameith2015Dataset"
_DM = "DmMicroarraySameith2015Dataset"
_NAN = "NaN"
_NANF = float("nan")
_SD_HALF = math.sqrt(0.5)
_SD_TWO = math.sqrt(2.0)
_GENES = ("YAL001C", "YBR001C", "YCR001W")
_GAP = " " * 11


class _StubGenome:
    """The two attributes ``_convert_to_systematic`` reads."""

    gene_attribute_table = pd.DataFrame(
        {
            "ID": ["YBR001C", "YOR166C", "YGL999W"],
            "gene": ["NTH2", "SWT1", None],
            "Alias": [None, None, "OLDNAME"],
        }
    )
    alias_to_systematic: dict[str, list[str]] = {"ALIAS9": ["YHR999W", "YHR998W"]}


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _describe(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"description": [f"{c} column" for c in table.columns]}, index=table.columns
    )


def _gsm(
    name: str, title: str, cy5: list[float], cy3: list[float], source: str = ""
) -> GSM:
    table = pd.DataFrame(
        {"ID_REF": [1, 2, 3], "Signal Norm_Cy5": cy5, "Signal Norm_Cy3": cy3}
    )
    metadata = {"title": [title], "source_name_ch1": [source]}
    return GSM(name=name, metadata=metadata, table=table, columns=_describe(table))


def _gse(gsms: list[GSM]) -> GSE:
    platform = pd.DataFrame({"ID": [1, 2, 3], "ORF": list(_GENES)})
    gpl = GPL(name="GPL", metadata={}, table=platform, columns=_describe(platform))
    return GSE(
        name="GSE42536",
        metadata={"title": ["GSE42536"]},
        gpls={"GPL": gpl},
        gsms={gsm.name: gsm for gsm in gsms},
    )


_ONES = [1.0, 1.0, 1.0]
_S1 = _gsm("S1", "yal001c-del-a", [2.0, 4.0, 8.0], [1.0, 4.0, 2.0])
_S2 = _gsm("S2", "yal001c-del-b", [2.0, 2.0, 2.0], [8.0, 2.0, 2.0], "refpool")
_S4 = _gsm("S4", "wt-a", [3.0, 3.0, 3.0], [5.0, 5.0, 5.0])
_SINGLE_GSMS = [
    _S1,
    _S2,
    _gsm("S3", "nth2-del", [4.0, 1.0, 1.0], [2.0, 2.0, 1.0]),
    _S4,
    _gsm("S5", "swt1-del-a", [9.0, 9.0, 9.0], [7.0, 7.0, 7.0]),
    _gsm("S6", "ycr001w-del+ydr001c-del", [2.0, 2.0, 2.0], _ONES),
    _gsm("S7", "ycr001w-del+zzz9-del", [2.0, 2.0, 2.0], _ONES),
    _gsm("S8", "yzz999w-del", _ONES, _ONES),
    _gsm("S9", "yer001w-del", _ONES, _ONES),
]
_DOUBLE_GSMS = [
    _S4,
    _S1,
    _gsm("D1", "ycr001w-del+ydr001c-del", [2.0, 2.0, 2.0], _ONES),
    _gsm("D2", "ycr001w-del+ydr001c-del-b", _ONES, [4.0, 4.0, 4.0], "refpool"),
    _gsm("D3", "yal001c-del+nth2-del", [1.0, 2.0, 4.0], _ONES),
    _gsm("D4", "yel001c-del+yfl001w-del", _ONES, _ONES),
]


def _write_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    single = workbook.active
    single.title = "Single mutants - info"
    single.append(["systematic name", "gene symbol"])
    single.append(["YAL001C", "TFC3"])
    single.append([" ybr001c ", "NTH2"])
    single.append([None, "blank"])
    double = workbook.create_sheet("Double mutants - info")
    double.append(
        [
            f"GSTF1,{_GAP}systematic name",
            f"GSTF2,{_GAP}systematic name",
            f"GSTF1,{_GAP}gene symbol",
            f"GSTF2,{_GAP}gene symbol",
            "selection",
            "curation",
            "comments",
        ]
    )
    double.append(["YCR001W", "ydr001c ", "A", "B", "sel1", "passed", "MATa strain"])
    double.append(["YBR001C", "YAL001C", "NTH2", "TFC3", "sel2", "passed", "matA"])
    double.append(["YEL001C", "YFL001W", "E", "F", "sel3", "failed", "MATa"])
    double.append(["YGL001C", "YHL001W", "G", "H", "sel4", "passed", None])
    double.append(["YIL001W", "YJL001W", "I", "J", "sel5", "passed", "MATα"])
    double.append(["YKL001C", "YLL001W", "K", "L", "sel6", "passed", "other note"])
    workbook.save(path)


def _write_raw(raw: Path, gsms: list[GSM]) -> None:
    raw.mkdir(parents=True)
    (raw / "GSE42536_family.soft.gz").write_bytes(b"placeholder: process reads .pkl")
    _write_workbook(raw / "12915_2015_222_MOESM1_ESM.xlsx")
    with open(raw / "GSE42536.pkl", "wb") as handle:
        pickle.dump(_gse(gsms), handle)


@pytest.fixture
def single(tmp_path: Path) -> m.SmMicroarraySameith2015Dataset:
    root = tmp_path / "sm"
    _write_raw(root / "raw", _SINGLE_GSMS)
    return m.SmMicroarraySameith2015Dataset(root=str(root), genome=_genome())


@pytest.fixture
def double(tmp_path: Path) -> m.DmMicroarraySameith2015Dataset:
    root = tmp_path / "dm"
    _write_raw(root / "raw", _DOUBLE_GSMS)
    return m.DmMicroarraySameith2015Dataset(root=str(root), genome=_genome())


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
# GSE42536's own "!Series_pubmed_id = 26700642" (#478; 26687005 is an eLife paper).
_PUBLICATION = Publication(
    pubmed_id="26700642",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/26700642/",
    doi="10.1186/s12915-015-0222-5",
    doi_url="https://doi.org/10.1186/s12915-015-0222-5",
)


def _three(a: float, b: float, c: float) -> dict[str, float]:
    return dict(zip(_GENES, (a, b, c), strict=True))


def _record(
    dataset: str,
    strain: str,
    perturbations: list[GenePerturbationType],
    mutant: dict[str, float],
    refpool: dict[str, float],
    log2: dict[str, float],
    se: dict[str, float],
    var: dict[str, float],
    n: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = MicroarrayExpressionExperiment(
        dataset_name=dataset,
        genotype=Genotype(perturbations=perturbations),
        environment=_ENVIRONMENT,
        phenotype=MicroarrayExpressionPhenotype(
            expression=mutant,
            expression_log2_ratio=log2,
            expression_log2_ratio_se=se,
            expression_log2_ratio_variance=var,
            n_replicates=n,
        ),
    )
    reference = MicroarrayExpressionExperimentReference(
        dataset_name=dataset,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=MicroarrayExpressionPhenotype(
            expression=refpool,
            expression_log2_ratio=dict.fromkeys(refpool, 0.0),
            expression_log2_ratio_se=None,
            expression_log2_ratio_variance=None,
            n_replicates=dict.fromkeys(refpool, 1),
        ),
    )
    return _nan_safe(experiment.model_dump()), reference.model_dump()


def _kan(orf: str) -> SgaKanMxDeletionPerturbation:
    return SgaKanMxDeletionPerturbation(
        systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=f"KanMX_{orf}"
    )


def _nat(orf: str) -> SgaNatMxDeletionPerturbation:
    return SgaNatMxDeletionPerturbation(
        systematic_gene_name=orf, perturbed_gene_name=orf, strain_id=f"NatMX_{orf}"
    )


_N1 = dict.fromkeys(_GENES, 1)
_N2 = dict.fromkeys(_GENES, 2)
_NANS = _three(_NANF, _NANF, _NANF)

_SINGLE_EXPECTED = [
    _record(
        _SM,
        "BY4742",
        [_kan("YAL001C")],
        _three(5.0, 3.0, 5.0),
        _three(1.5, 3.0, 2.0),
        _three(1.5, 0.0, 1.0),
        _three(_SD_HALF / math.sqrt(2), 0.0, _SD_TWO / math.sqrt(2)),
        _three(_SD_HALF**2, 0.0, _SD_TWO**2),
        _N2,
    ),
    _record(
        _SM,
        "BY4742",
        [_kan("YBR001C")],
        _three(4.0, 1.0, 1.0),
        _three(2.0, 2.0, 1.0),
        _three(1.0, -1.0, 0.0),
        _NANS,
        _NANS,
        _N1,
    ),
    _record(
        _SM,
        "BY4742",
        [_kan("YER001W")],
        _three(1.0, 1.0, 1.0),
        _three(1.0, 1.0, 1.0),
        _three(0.0, 0.0, 0.0),
        _NANS,
        _NANS,
        _N1,
    ),
]


def test_single_mutant_records_average_replicates_per_gene(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """Three groups in first-appearance order: YAL001C (S1 + S2), YBR001C (S3, named
    NTH2) and YER001W (S9, outside the SI list but still processed). The
    double-deletion arrays S6 and S7 make no record.
    """
    assert len(single) == 3
    for i, (experiment, reference) in enumerate(_SINGLE_EXPECTED):
        assert _nan_safe(single[i]["experiment"]) == experiment
        assert single[i]["reference"] == reference
    assert single[0]["publication"] == _PUBLICATION.model_dump()


def test_double_deletion_arrays_are_not_single_mutant_records(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """#479: the parser returns every gene a title names, systematic names first
    (``rpn4-del+ydr026c-del`` style: ``nth2-del+ycr001w-del`` -> YCR001W, YBR001C), and
    ``process()`` keeps an array as a single mutant only when its title names exactly
    one gene and has no ``+``. S6 (two genes) and S7 (``+`` with the unresolvable
    ZZZ9) are dropped under ``double_deletion_title``, so no record carries YCR001W.
    Before the fix S6 was stored as a YCR001W single deletion, and on GSE42536 all 143
    double-deletion arrays were folded into 45 of the 82 single-mutant records.
    """
    title_genes = single._extract_gene_names_from_title
    assert title_genes("ycr001w-del+ydr001c-del") == ["YCR001W", "YDR001C"]
    assert title_genes("nth2-del+ycr001w-del") == ["YCR001W", "YBR001C"]
    assert title_genes("ycr001w-del+zzz9-del") == ["YCR001W"]
    assert title_genes("nth2-del-1-a") == ["YBR001C"]
    assert title_genes("yal001c-del-a") == ["YAL001C"]
    assert single.dropped_double_deletion_titles == 2
    stored = [
        perturbation["systematic_gene_name"]
        for i in range(len(single))
        for perturbation in single[i]["experiment"]["genotype"]["perturbations"]
    ]
    assert stored == ["YAL001C", "YBR001C", "YER001W"]
    assert [
        single[i]["experiment"]["phenotype"]["n_replicates"]["YAL001C"]
        for i in range(len(single))
    ] == [2, 1, 1]


def test_both_loaders_store_the_geo_series_pubmed_id(
    single: m.SmMicroarraySameith2015Dataset, double: m.DmMicroarraySameith2015Dataset
) -> None:
    """#478: every record cites PubMed 26700642, GSE42536's ``!Series_pubmed_id``,
    and no longer 26687005, an unrelated eLife 2015 paper.
    """
    assert m.SAMEITH2015_PUBMED_ID == "26700642"
    publications = [single[i]["publication"] for i in range(len(single))] + [
        double[i]["publication"] for i in range(len(double))
    ]
    assert len(publications) == 5
    assert all(p == _PUBLICATION.model_dump() for p in publications)
    assert {p["pubmed_url"] for p in publications} == {
        "https://pubmed.ncbi.nlm.nih.gov/26700642/"
    }


def test_a_gene_named_with_wt_is_classified_as_wildtype(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """Finding: ``is_wildtype`` is ``"wt" in title.lower()`` (sameith2015.py line 239),
    so ``swt1-del-a`` (SWT1 = YOR166C) is a wildtype array and no YOR166C record is
    written, although its gene name resolves and ``is_single_mutant`` is true.
    """
    samples = pd.read_csv(Path(single.root) / "preprocess" / "data.csv")
    assert samples.to_dict(orient="list") == {
        "geo_accession": ["S1", "S2", "S3", "S5", "S9"],
        "title": [
            "yal001c-del-a",
            "yal001c-del-b",
            "nth2-del",
            "swt1-del-a",
            "yer001w-del",
        ],
        "is_wildtype": [False, False, False, True, False],
        "is_single_mutant": [True] * 5,
        "gene_names": [
            "['YAL001C']",
            "['YAL001C']",
            "['YBR001C']",
            "['YOR166C']",
            "['YER001W']",
        ],
    }
    assert all(
        single[i]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        != "YOR166C"
        for i in range(len(single))
    )


def test_single_side_files_and_wildtype_reference(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """Each of the three records has its own refpool and reference; the wildtype
    reference averages the refpool channel of S4 (Cy3 5) and S5 (Cy3 7), mean 6 with
    sample SD sqrt(2). ``_calculate_wt_reference_with_std``'s output is used by no
    record: ``process()`` stores it on ``wt_reference_expression`` /
    ``wt_std_expression`` (sameith2015.py line 264) and nothing reads either attribute,
    so the helper is pinned here in isolation.
    """
    preprocess = Path(single.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YER001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0], [1], [2]]
    assert single.experiment_class is MicroarrayExpressionExperiment
    assert single.reference_class is MicroarrayExpressionExperimentReference
    assert single.raw_file_names == ["GSE42536_family.soft.gz"]
    frame = pd.DataFrame({"a": [1]})
    assert single.preprocess_raw(frame) is frame
    mean, std = single._calculate_wt_reference_with_std(
        [_S4, _SINGLE_GSMS[4]], {"1": "YAL001C"}
    )
    assert (dict(mean), dict(std)) == ({"YAL001C": 6.0}, {"YAL001C": _SD_TWO})
    assert single._calculate_wt_reference_with_std([], {}) == ({}, {})
    assert single._load_authoritative_single_mutants() == {
        "YAL001C": {
            "systematic_name": "YAL001C",
            "gene_symbol": "TFC3",
            "strain": "BY4742",
        },
        "YBR001C": {
            "systematic_name": "YBR001C",
            "gene_symbol": "NTH2",
            "strain": "BY4742",
        },
    }


def test_convert_to_systematic_tries_table_gene_alias_then_alias_map(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    convert = single._convert_to_systematic
    assert [
        convert(name) for name in ("", "yal001c", "nth2", "oldname", "alias9", "none9")
    ] == [None, "YAL001C", "YBR001C", "YGL999W", "YHR999W", None]


_DOUBLE_EXPECTED = [
    _record(
        _DM,
        "BY4741",
        [_kan("YCR001W"), _nat("YDR001C")],
        _three(3.0, 3.0, 3.0),
        _three(1.0, 1.0, 1.0),
        _three(1.5, 1.5, 1.5),
        _three(*[_SD_HALF / math.sqrt(2)] * 3),
        _three(*[_SD_HALF**2] * 3),
        _N2,
    ),
    _record(
        _DM,
        "BY4741",
        [_kan("YAL001C"), _nat("YBR001C")],
        _three(1.0, 2.0, 4.0),
        _three(1.0, 1.0, 1.0),
        _three(0.0, 1.0, 2.0),
        _NANS,
        _NANS,
        _N1,
    ),
]


def test_double_mutant_records_pair_replicates_and_take_the_strain_from_comments(
    double: m.DmMicroarraySameith2015Dataset,
) -> None:
    """Pair (YCR001W, YDR001C) over D1 + D2 carries comment ``MATa strain`` and is built
    in BY4741; pair (YAL001C, YBR001C) from D3 carries ``matA`` and is built in BY4741;
    the failed-curation pair D4 and the one-gene S1 write nothing.
    """
    assert len(double) == 2
    for i, (experiment, reference) in enumerate(_DOUBLE_EXPECTED):
        assert _nan_safe(double[i]["experiment"]) == experiment
        assert double[i]["reference"] == reference
    assert double[1]["publication"] == _PUBLICATION.model_dump()


def test_gstf_pairs_map_mating_comments_onto_strains(
    double: m.DmMicroarraySameith2015Dataset,
) -> None:
    """``_load_authoritative_gstf_pairs`` maps a ``MATa`` or ``matA`` comment (any case)
    to BY4741, the MATa strain, and ``MATα`` or ``MATalpha`` to BY4742, the MATalpha
    strain; a blank or unrelated comment defaults to BY4742.

    The paper states: "All single mutants and most double mutants carry the mating type
    matα and are in the genetic background of BY4742. Few double mutants carry the
    mating type matA and are in the genetic background of BY4741" (verbatim). The SI
    has exactly four ``MATa`` comments, on the passed pairs HAC1+SNT1, SNT1+SPT2,
    CUP2+HAA1 and SIP4+YER184C, and GEO's sample characteristics carry ``strain:
    BY4741`` on exactly those four double mutants (GSE42536, 2026-09-28). Before this
    mapping the loader stored all 72 passed pairs as BY4742.
    """
    pairs = double._load_authoritative_gstf_pairs()
    assert {key: value["strain"] for key, value in pairs.items()} == {
        ("YCR001W", "YDR001C"): "BY4741",
        ("YAL001C", "YBR001C"): "BY4741",
        ("YGL001C", "YHL001W"): "BY4742",
        ("YIL001W", "YJL001W"): "BY4742",
        ("YKL001C", "YLL001W"): "BY4742",
    }
    assert {
        key: value
        for key, value in pairs[("YCR001W", "YDR001C")].items()
        if key != "strain"
    } == {
        "gstf1_systematic": "YCR001W",
        "gstf2_systematic": "YDR001C",
        "gstf1_symbol": "A",
        "gstf2_symbol": "B",
        "selection": "sel1",
        "comments": "MATa strain",
    }


def test_double_side_files(double: m.DmMicroarraySameith2015Dataset) -> None:
    """``data.csv`` keeps the four two-gene arrays (the unmatched D4 included)."""
    preprocess = Path(double.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
        "YDR001C",
    ]
    samples = pd.read_csv(preprocess / "data.csv")
    assert samples["geo_accession"].tolist() == ["D1", "D2", "D3", "D4"]
    assert samples["gene_names"].tolist() == [
        "['YCR001W', 'YDR001C']",
        "['YCR001W', 'YDR001C']",
        "['YAL001C', 'YBR001C']",
        "['YEL001C', 'YFL001W']",
    ]
    # Both pairs are now BY4741 with the same all-ones refpool, so they share one
    # reference.
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1]]
    assert double.raw_file_names == ["GSE42536_family.soft.gz"]


def test_double_parallel_build_writes_the_same_records(tmp_path: Path) -> None:
    """``process_workers=1`` routes through ``_process_parallel``, ``_process_batch``
    and the static extraction and statistics helpers; the records match the sequential
    build.
    """
    root = tmp_path / "dm_parallel"
    _write_raw(root / "raw", _DOUBLE_GSMS)
    dataset = m.DmMicroarraySameith2015Dataset(
        root=str(root), genome=_genome(), process_workers=1, batch_size=1
    )
    assert len(dataset) == 2
    for i, (experiment, reference) in enumerate(_DOUBLE_EXPECTED):
        assert _nan_safe(dataset[i]["experiment"]) == experiment
        assert dataset[i]["reference"] == reference


def test_a_zero_signal_probe_fails_phenotype_validation(tmp_path: Path) -> None:
    """Finding: ``_extract_expression_from_gsm`` keeps a 0 mutant signal in
    ``mutant_data`` but drops its log2 ratio (sameith2015.py lines 671-674), and
    ``create_single_mutant_expression_experiment`` (lines 782-788) then pairs the
    three-gene ``expression`` with two-gene ``n_replicates``, so one zero cell aborts
    the whole build with a validation error instead of skipping the probe.
    """
    root = tmp_path / "sm_zero"
    _write_raw(root / "raw", [_gsm("Z1", "yal001c-del-a", [4.0, 0.0, 1.0], _ONES)])
    with pytest.raises(
        ValueError, match="n_replicates must have the same keys as expression"
    ):
        m.SmMicroarraySameith2015Dataset(root=str(root), genome=_genome())

# tests/torchcell/datasets/scerevisiae/test_sameith2015_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_sameith2015_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_sameith2015_synthetic.py
"""Sameith 2015 single- and double-mutant loaders built on in-memory GEOparse objects.

Each root's ``raw/`` holds ``GSE42536_family.soft.gz`` (placeholder bytes, never
opened), the supplementary workbook ``12915_2015_222_MOESM1_ESM.xlsx`` (openpyxl, sheets
``List of 215 GSTFs``, ``Single mutants - info`` and ``Double mutants - info``) and ``GSE42536.pkl``, a pickled
REAL ``GEOparse.GEOTypes.GSE`` built in memory, so ``process()`` loads the pickle and the
``GEOparse.get_GEO`` network branch is never reached. No genome is involved: a title
token resolves only through the workbook's own (systematic name, gene symbol) pairs, read
from all three sheets (``List of 215 GSTFs`` too), or as an ORF-shaped token, and
``perturbed_gene_name`` is the workbook's symbol for the ORF (else the ORF).

Platform probes: 1 YAL001C, 2 YBR001C, 3 YCR001W (one spot per gene; the spot test
adds a second YAL001C probe). The mutant channel is Cy5 unless ``source_name_ch1``
names the refpool, then Cy3; log2 ratio = log2(mutant / refpool). A signal that is not
a positive number raises.

ONE RECORD PER ARRAY (2026-11-24, #1174, #1178): an array is one culture, so every
array is its own record with that array's values, ``n_replicates`` 1 per gene, no SE or
variance, ``replicate_id`` = the GSM name, and the reference = that array's refpool
channel (``n_replicates`` 1, ``replicate_id`` None). Records come in group order (first
appearance of a genotype), arrays in GSE order inside a group.

Single-mutant GSE (title: Cy5 / Cy3, source):

    S1 "yal001c-del-a"            2 4 8 / 1 4 2             -> log2 1 0 2
    S2 "yal001c-del-b"            2 2 2 / 8 2 2, refpool    -> mutant 8 2 2, log2 2 0 0
    S3 "nth2-del" (NTH2 -> YBR001C, the workbook's pair "ybr001c | Nth2")
                                  4 1 1 / 2 2 1             -> log2 1 -1 0
    S4 "wt-a"                     wildtype
    S5 "swt1-del-a"               wildtype by substring (Finding; SWT1 -> YOR166C
                                  through "List of 215 GSTFs" only)
    S6 "ycr001w-del+ydr001c-del"  double-deletion title: dropped (#479)
    S7 "ycr001w-del+zzz9-del"     "+" with an unresolved partner: dropped (#479)
    S8 "yzz999w-del"              no gene name: ignored
    S9 "yer001w-del"              1 1 1 / 1 1 1             -> not in the SI list, kept

So four records: S1, S2 (YAL001C), S3 (YBR001C), S9 (YER001W).

Double-mutant GSE: S4 wildtype, S1 (one gene, ignored), D1 "ycr001w-del+ydr001c-del"
(2 2 2 / 1 1 1, log2 1) and D2 "ycr001w-del+ydr001c-del-b" (refpool in ch1, mutant Cy3
4 4 4, refpool 1 1 1, log2 2) -> pair (YCR001W, YDR001C), two records; D3
"yal001c-del+nth2-del" (1 2 4 / 1 1 1, log2 0 1 2) -> pair (YAL001C, YBR001C); D4
"yel001c-del+yfl001w-del" is a pair that failed curation and is skipped.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import urllib.request
from pathlib import Path
from typing import Any

import GEOparse
import lmdb
import numpy as np
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
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

_SM = "SmMicroarraySameith2015Dataset"
_DM = "DmMicroarraySameith2015Dataset"
_NAN = "NaN"
_NANF = float("nan")
_SD_TWO = math.sqrt(2.0)
_GENES = ("YAL001C", "YBR001C", "YCR001W")
_GAP = " " * 11


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


def _gse(gsms: list[GSM], platform: pd.DataFrame | None = None) -> GSE:
    if platform is None:
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
    """The release's three sheets, spelled as the release spells them (``Ybr001C``,
    ``Nth2``). YCR001W carries the symbol ``Ccr1`` in the GSTF list and itself
    (``Ycr001W``) in the double sheet, as the real workbook does for YML081W / Tda9, so
    its symbol is CCR1. SWT1 is listed only in ``List of 215 GSTFs``.
    """
    workbook = openpyxl.Workbook()
    gstfs = workbook.active
    gstfs.title = "List of 215 GSTFs"
    gstfs.append(["systematic name", "gene symbol"])
    gstfs.append(["Yor166C", "Swt1"])
    gstfs.append(["Ycr001W", "Ccr1"])
    gstfs.append([None, None])
    single = workbook.create_sheet("Single mutants - info")
    single.append(["systematic name", "gene symbol"])
    single.append(["YAL001C", "Tfc3"])
    single.append([" ybr001c ", "Nth2"])
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
    double.append(
        ["YCR001W", "ydr001c ", "Ycr001W", "Dsy1", "sel1", "passed", "MATa strain"]
    )
    double.append(["YBR001C", "YAL001C", "Nth2", "Tfc3", "sel2", "passed", "matA"])
    double.append(["YEL001C", "YFL001W", "Eaa1", "Ffa1", "sel3", "failed", "MATa"])
    double.append(["YGL001C", "YHL001W", "Gga1", "Hha1", "sel4", "passed", None])
    double.append(["YIL001W", "YJL001W", "Iia1", "Jja1", "sel5", "passed", "MATα"])
    double.append(
        ["YKL001C", "YLL001W", "Kka1", "Lla1", "sel6", "passed", "other note"]
    )
    workbook.save(path)


def _write_raw(
    raw: Path, gsms: list[GSM], platform: pd.DataFrame | None = None
) -> None:
    raw.mkdir(parents=True)
    (raw / "GSE42536_family.soft.gz").write_bytes(b"placeholder: process reads .pkl")
    _write_workbook(raw / "12915_2015_222_MOESM1_ESM.xlsx")
    with open(raw / "GSE42536.pkl", "wb") as handle:
        pickle.dump(_gse(gsms, platform), handle)


@pytest.fixture
def single(tmp_path: Path) -> m.SmMicroarraySameith2015Dataset:
    root = tmp_path / "sm"
    _write_raw(root / "raw", _SINGLE_GSMS)
    return m.SmMicroarraySameith2015Dataset(root=str(root))


@pytest.fixture
def double(tmp_path: Path) -> m.DmMicroarraySameith2015Dataset:
    root = tmp_path / "dm"
    _write_raw(root / "raw", _DOUBLE_GSMS)
    return m.DmMicroarraySameith2015Dataset(root=str(root))


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
    replicate_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One array's record: its own values, n = 1 per gene, no SE or variance."""
    experiment = MicroarrayExpressionExperiment(
        dataset_name=dataset,
        genotype=Genotype(perturbations=perturbations),
        environment=_ENVIRONMENT,
        phenotype=MicroarrayExpressionPhenotype(
            expression=mutant,
            expression_log2_ratio=log2,
            expression_log2_ratio_se=None,
            expression_log2_ratio_variance=None,
            n_replicates=dict.fromkeys(mutant, 1),
            replicate_id=replicate_id,
        ),
    )
    reference = MicroarrayExpressionExperimentReference(
        dataset_name=dataset,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=strain,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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


def _kan(orf: str, name: str) -> SgaKanMxDeletionPerturbation:
    return SgaKanMxDeletionPerturbation(
        systematic_gene_name=orf, perturbed_gene_name=name, strain_id=f"KanMX_{orf}"
    )


def _nat(orf: str, name: str) -> SgaNatMxDeletionPerturbation:
    return SgaNatMxDeletionPerturbation(
        systematic_gene_name=orf, perturbed_gene_name=name, strain_id=f"NatMX_{orf}"
    )


_ALL_ONES = _three(1.0, 1.0, 1.0)

_SINGLE_EXPECTED = [
    _record(
        _SM,
        "BY4742",
        [_kan("YAL001C", "TFC3")],
        _three(2.0, 4.0, 8.0),
        _three(1.0, 4.0, 2.0),
        _three(1.0, 0.0, 2.0),
        "S1",
    ),
    _record(
        _SM,
        "BY4742",
        [_kan("YAL001C", "TFC3")],
        _three(8.0, 2.0, 2.0),
        _three(2.0, 2.0, 2.0),
        _three(2.0, 0.0, 0.0),
        "S2",
    ),
    _record(
        _SM,
        "BY4742",
        [_kan("YBR001C", "NTH2")],
        _three(4.0, 1.0, 1.0),
        _three(2.0, 2.0, 1.0),
        _three(1.0, -1.0, 0.0),
        "S3",
    ),
    _record(
        _SM,
        "BY4742",
        [_kan("YER001W", "YER001W")],
        _ALL_ONES,
        _ALL_ONES,
        _three(0.0, 0.0, 0.0),
        "S9",
    ),
]


def test_single_mutant_records_are_one_per_array(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """One record per array, groups in first-appearance order: YAL001C (S1, then S2,
    two cultures and two records, never their mean), YBR001C (S3, named NTH2) and
    YER001W (S9, outside the SI list but still processed). Each record is its array's
    values verbatim with n = 1 and no SE or variance, ``replicate_id`` the GSM name,
    and its reference is that array's own refpool channel. The double-deletion arrays
    S6 and S7 make no record.
    """
    assert len(single) == 4
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
    assert stored == ["YAL001C", "YAL001C", "YBR001C", "YER001W"]
    assert [
        single[i]["experiment"]["phenotype"]["replicate_id"] for i in range(len(single))
    ] == ["S1", "S2", "S3", "S9"]


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
    assert len(publications) == 7
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
    """Each of the four records has its own refpool and reference; the wildtype
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
    assert [entry["member_indices"] for entry in index] == [[0], [1], [2], [3]]
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
            "gene_symbol": "Tfc3",
            "strain": "BY4742",
        },
        "YBR001C": {
            "systematic_name": "YBR001C",
            "gene_symbol": "Nth2",
            "strain": "BY4742",
        },
    }


def test_title_tokens_resolve_only_through_the_release_workbook(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """A systematic-shaped token is taken verbatim (upper-cased; GEO titles are lower
    case); any other token resolves only through the workbook's symbol map, read from
    all three sheets (SWT1 is only in ``List of 215 GSTFs``); a token in neither is not a
    gene token (None). The ORF -> symbol inverse drops ORF-as-symbol entries, so
    YCR001W's symbol is CCR1 although the double sheet writes ``Ycr001W``.
    """
    convert = single._convert_to_systematic
    assert [
        convert(name)
        for name in ("", "yal001c", "ybr089c-a", "nth2", "swt1", "del", "mata", "x9")
    ] == [None, "YAL001C", "YBR089C-A", "YBR001C", "YOR166C", None, None, None]
    assert single.symbol_map == {
        "SWT1": "YOR166C",
        "CCR1": "YCR001W",
        "TFC3": "YAL001C",
        "NTH2": "YBR001C",
        "YCR001W": "YCR001W",
        "DSY1": "YDR001C",
        "EAA1": "YEL001C",
        "FFA1": "YFL001W",
        "GGA1": "YGL001C",
        "HHA1": "YHL001W",
        "IIA1": "YIL001W",
        "JJA1": "YJL001W",
        "KKA1": "YKL001C",
        "LLA1": "YLL001W",
    }
    assert single.gene_symbols == {
        orf: symbol for symbol, orf in single.symbol_map.items() if symbol != orf
    }


def test_release_symbol_map_refuses_a_symbol_paired_with_two_orfs(
    tmp_path: Path,
) -> None:
    """A symbol the workbook pairs with two different ORFs across sheets raises: the
    release would not say which gene a title means. Two different non-ORF symbols for
    one ORF also raise in the inverse.
    """
    path = tmp_path / "book.xlsx"
    _write_workbook(path)
    workbook = openpyxl.load_workbook(path)
    workbook["List of 215 GSTFs"].append(["Yzz001W", "Nth2"])
    workbook.save(path)
    with pytest.raises(ValueError, match="pairs the symbol NTH2 with both"):
        m.release_symbol_map(str(path))
    with pytest.raises(ValueError, match="names YAL001C both"):
        m.release_gene_symbols({"TFC3": "YAL001C", "TFC9": "YAL001C"})


_DOUBLE_EXPECTED = [
    _record(
        _DM,
        "BY4741",
        [_kan("YCR001W", "CCR1"), _nat("YDR001C", "DSY1")],
        _three(2.0, 2.0, 2.0),
        _ALL_ONES,
        _ALL_ONES,
        "D1",
    ),
    _record(
        _DM,
        "BY4741",
        [_kan("YCR001W", "CCR1"), _nat("YDR001C", "DSY1")],
        _three(4.0, 4.0, 4.0),
        _ALL_ONES,
        _three(2.0, 2.0, 2.0),
        "D2",
    ),
    _record(
        _DM,
        "BY4741",
        [_kan("YAL001C", "TFC3"), _nat("YBR001C", "NTH2")],
        _three(1.0, 2.0, 4.0),
        _ALL_ONES,
        _three(0.0, 1.0, 2.0),
        "D3",
    ),
]


def test_double_mutant_records_are_one_per_array_and_take_the_strain_from_comments(
    double: m.DmMicroarraySameith2015Dataset,
) -> None:
    """Pair (YCR001W, YDR001C) is two arrays, D1 and D2 (two cultures in opposite dye
    orientations), so two records; it carries comment ``MATa strain`` and is built in
    BY4741. Pair (YAL001C, YBR001C) from D3 carries ``matA`` and is built in BY4741.
    The failed-curation pair D4 and the one-gene S1 write nothing.
    """
    assert len(double) == 3
    for i, (experiment, reference) in enumerate(_DOUBLE_EXPECTED):
        assert _nan_safe(double[i]["experiment"]) == experiment
        assert double[i]["reference"] == reference
    assert double[2]["publication"] == _PUBLICATION.model_dump()


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
        "gstf1_symbol": "Ycr001W",
        "gstf2_symbol": "Dsy1",
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
    # The three arrays are BY4741 with the same all-ones refpool channel and
    # n_replicates 1, so their three references are one reference.
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2]]
    assert double.raw_file_names == ["GSE42536_family.soft.gz"]


def test_double_parallel_build_writes_the_same_records(tmp_path: Path) -> None:
    """``process_workers=1`` routes through ``_process_parallel``, ``_process_batch``
    and the static extraction and statistics helpers; the records match the sequential
    build.
    """
    root = tmp_path / "dm_parallel"
    _write_raw(root / "raw", _DOUBLE_GSMS)
    dataset = m.DmMicroarraySameith2015Dataset(
        root=str(root), process_workers=1, batch_size=1
    )
    assert len(dataset) == 3
    for i, (experiment, reference) in enumerate(_DOUBLE_EXPECTED):
        assert _nan_safe(dataset[i]["experiment"]) == experiment
        assert dataset[i]["reference"] == reference


def test_a_zero_signal_probe_refuses_the_build(tmp_path: Path) -> None:
    """A 0 signal has no log ratio, so ``combine_spots`` raises naming the array and
    the gene; GSE42536 has no non-positive signal. Before 2026-11-24 the zero was kept
    in ``expression`` and its log2 dropped, and the build died later on mismatched
    ``n_replicates`` keys.
    """
    root = tmp_path / "sm_zero"
    _write_raw(root / "raw", [_gsm("Z1", "yal001c-del-a", [4.0, 0.0, 1.0], _ONES)])
    with pytest.raises(
        ValueError, match=r"^Z1 YBR001C: spot signals mutant=0\.0, refpool=1\.0 are not"
    ):
        m.SmMicroarraySameith2015Dataset(root=str(root))


def _two_probe_gsm(name: str, title: str, source: str) -> GSM:
    """An array whose table has no row for probe 2 (YBR001C)."""
    table = pd.DataFrame(
        {"ID_REF": [1, 3], "Signal Norm_Cy5": [2.0, 2.0], "Signal Norm_Cy3": [8.0, 2.0]}
    )
    metadata = {"title": [title], "source_name_ch1": [source]}
    return GSM(name=name, metadata=metadata, table=table, columns=_describe(table))


def test_each_array_record_keeps_its_own_genes_and_refpool(tmp_path: Path) -> None:
    """#630 superseded: with one record per array the reference is that array's refpool
    channel, n = 1 per gene. M1 carries all three probes (refpool Cy3 1 4 2); M2 names
    the refpool in ch1 (refpool Cy5 2 _ 2) and has no row for YBR001C, so M2's record
    and reference hold YAL001C and YCR001W only. The same holds for the double-mutant
    loader.
    """
    single_root = tmp_path / "sm_missing"
    _write_raw(
        single_root / "raw",
        [
            _gsm("M1", "yal001c-del-a", [2.0, 4.0, 8.0], [1.0, 4.0, 2.0]),
            _two_probe_gsm("M2", "yal001c-del-b", "refpool"),
        ],
    )
    single = m.SmMicroarraySameith2015Dataset(root=str(single_root))
    assert len(single) == 2
    assert single[0]["reference"]["phenotype_reference"]["expression"] == {
        "YAL001C": 1.0,
        "YBR001C": 4.0,
        "YCR001W": 2.0,
    }
    reference = single[1]["reference"]["phenotype_reference"]
    assert reference["expression"] == {"YAL001C": 2.0, "YCR001W": 2.0}
    assert reference["n_replicates"] == {"YAL001C": 1, "YCR001W": 1}
    assert reference["replicate_id"] is None
    phenotype = single[1]["experiment"]["phenotype"]
    assert phenotype["expression"] == {"YAL001C": 8.0, "YCR001W": 2.0}
    assert phenotype["n_replicates"] == {"YAL001C": 1, "YCR001W": 1}
    assert phenotype["replicate_id"] == "M2"

    double_root = tmp_path / "dm_missing"
    _write_raw(
        double_root / "raw",
        [
            _gsm("N1", "ycr001w-del+ydr001c-del", [2.0, 4.0, 8.0], [1.0, 4.0, 2.0]),
            _two_probe_gsm("N2", "ycr001w-del+ydr001c-del-b", "refpool"),
        ],
    )
    double = m.DmMicroarraySameith2015Dataset(root=str(double_root))
    assert len(double) == 2
    reference = double[1]["reference"]["phenotype_reference"]
    assert reference["expression"] == {"YAL001C": 2.0, "YCR001W": 2.0}
    assert reference["n_replicates"] == {"YAL001C": 1, "YCR001W": 1}


def test_spots_of_one_gene_are_combined_inside_the_array_record(tmp_path: Path) -> None:
    """A gene printed twice (probes 1 and 4 both YAL001C, as GPL11232 prints every ORF
    twice) contributes both spots to its array's record: mutant and refpool are the
    geometric means of the spots and the log2 ratio is the mean of the spot log2
    ratios. A1 (mutant Cy5): YAL001C spots (2 / 1) and (8 / 4) -> mutant 4, refpool 2,
    log2 1; YCR001W spot (1 / 4) -> log2 -2. Before 2026-11-24 the loader kept only the
    last spot (8 / 4). A2 (mutant Cy3) spots (1 / 4) and (4 / 1) for YAL001C -> mutant
    2, refpool 2, log2 mean(-2, 2) = 0.
    """
    platform = pd.DataFrame(
        {"ID": [1, 2, 3, 4], "ORF": ["YAL001C", "YBR001C", "YCR001W", "YAL001C"]}
    )

    def four_probe_gsm(
        name: str, title: str, cy5: list[float], cy3: list[float], source: str
    ) -> GSM:
        table = pd.DataFrame(
            {"ID_REF": [1, 2, 3, 4], "Signal Norm_Cy5": cy5, "Signal Norm_Cy3": cy3}
        )
        metadata = {"title": [title], "source_name_ch1": [source]}
        return GSM(name=name, metadata=metadata, table=table, columns=_describe(table))

    root = tmp_path / "sm_spots"
    _write_raw(
        root / "raw",
        [
            four_probe_gsm(
                "A1", "yal001c-del-a", [2.0, 1.0, 1.0, 8.0], [1.0, 1.0, 4.0, 4.0], ""
            ),
            four_probe_gsm(
                "A2",
                "yal001c-del-b",
                [4.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 4.0],
                "refpool",
            ),
        ],
        platform,
    )
    dataset = m.SmMicroarraySameith2015Dataset(root=str(root))
    assert len(dataset) == 2
    first = dataset[0]["experiment"]["phenotype"]
    assert first["expression"] == {"YAL001C": 4.0, "YBR001C": 1.0, "YCR001W": 1.0}
    assert first["expression_log2_ratio"] == {
        "YAL001C": 1.0,
        "YBR001C": 0.0,
        "YCR001W": -2.0,
    }
    assert dataset[0]["reference"]["phenotype_reference"]["expression"] == {
        "YAL001C": 2.0,
        "YBR001C": 1.0,
        "YCR001W": 4.0,
    }
    second = dataset[1]["experiment"]["phenotype"]
    assert second["expression"]["YAL001C"] == 2.0
    assert second["expression_log2_ratio"]["YAL001C"] == 0.0
    assert (
        dataset[1]["reference"]["phenotype_reference"]["expression"]["YAL001C"] == 2.0
    )


# ---------------------------------------------------------------------------
# 2026.10.06 (Phase 21): download, the parallel batch path and the static helpers.
# ---------------------------------------------------------------------------

_SUPPL_URL = (
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs12915-015-0222-5/"
    "MediaObjects/12915_2015_222_MOESM1_ESM.xlsx"
)
_PROBES = {"1": "YAL001C", "2": "YBR001C", "3": "YCR001W"}


class _GeoRecorder:
    """Stand-in for ``GEOparse.get_GEO`` that records its keyword arguments."""

    def __init__(self, result: object, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _RetrieveRecorder:
    """Stand-in for ``urllib.request.urlretrieve`` that writes hand-made bytes."""

    def __init__(self, payload: bytes, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, path: str) -> None:
        self.calls.append((url, path))
        if self.error is not None:
            raise self.error
        Path(path).write_bytes(self.payload)


def _fresh_raw(dataset: Any) -> Path:
    """Empty the dataset's raw dir so ``download`` writes into a known state."""
    raw = Path(dataset.raw_dir)
    for child in raw.iterdir():
        child.unlink()
    return raw


@pytest.mark.parametrize("fixture", ["single", "double"])
def test_download_pickles_the_geo_object_and_fetches_the_si_workbook(
    fixture: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: ``download`` (sameith2015.py lines 155-185 and 928-960) writes the
    pickled GEO object and the SI workbook with no sha256, no
    ``source_url`` / ``retrieval_command`` record, so a rebuilt raw dir cannot be
    checked against the bytes the served build consumed. ``get_GEO`` receives exactly
    ``geo="GSE42536", destdir=<raw_dir>, silent=False`` and ``urlretrieve`` the
    Springer ESM URL and ``<raw_dir>/12915_2015_222_MOESM1_ESM.xlsx``; the pickle
    round-trips the object ``get_GEO`` returned. Under the stub (which writes no SOFT
    file) ``raw/`` then holds exactly those two files; a real ``get_GEO`` also leaves
    ``GSE42536_family.soft.gz`` there, likewise unrecorded. Pinned until download records a
    retrieval manifest with the sha256 of each file.
    """
    dataset = request.getfixturevalue(fixture)
    raw = _fresh_raw(dataset)
    geo = _GeoRecorder(result={"stand-in": "GSE42536"})
    retrieve = _RetrieveRecorder(b"hand-made workbook bytes")
    monkeypatch.setattr(GEOparse, "get_GEO", geo)
    monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
    dataset.download()
    assert geo.calls == [{"geo": "GSE42536", "destdir": str(raw), "silent": False}]
    assert retrieve.calls == [(_SUPPL_URL, str(raw / "12915_2015_222_MOESM1_ESM.xlsx"))]
    assert sorted(p.name for p in raw.iterdir()) == [
        "12915_2015_222_MOESM1_ESM.xlsx",
        "GSE42536.pkl",
    ]
    with open(raw / "GSE42536.pkl", "rb") as handle:
        assert pickle.load(handle) == {"stand-in": "GSE42536"}
    assert (raw / "12915_2015_222_MOESM1_ESM.xlsx").read_bytes() == (
        b"hand-made workbook bytes"
    )


def test_single_download_refetches_an_existing_workbook_double_does_not(
    single: m.SmMicroarraySameith2015Dataset,
    double: m.DmMicroarraySameith2015Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the workbook already in ``raw/``, the single-mutant ``download`` calls
    ``urlretrieve`` again and overwrites it (line 179, no existence check) while the
    double-mutant one leaves it alone (line 950, ``if not osp.exists``).
    """
    monkeypatch.setattr(GEOparse, "get_GEO", _GeoRecorder(result="gse"))
    for dataset, expected_calls, expected_bytes in (
        (single, 1, b"new"),
        (double, 0, b"old"),
    ):
        raw = _fresh_raw(dataset)
        (raw / "12915_2015_222_MOESM1_ESM.xlsx").write_bytes(b"old")
        retrieve = _RetrieveRecorder(b"new")
        monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
        dataset.download()
        assert len(retrieve.calls) == expected_calls
        assert (raw / "12915_2015_222_MOESM1_ESM.xlsx").read_bytes() == expected_bytes


@pytest.mark.parametrize(
    ("fixture", "geo_message"),
    [
        ("single", "GEO download failed"),
        ("double", "Failed to download GSE42536 from GEO"),
    ],
)
def test_download_refusals(
    fixture: str,
    geo_message: str,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``get_GEO`` failure raises ``RuntimeError`` with the class's own message and
    writes no pickle and fetches no workbook; a workbook failure raises ``Failed to
    download supplementary data`` after the pickle is already on disk.
    """
    dataset = request.getfixturevalue(fixture)
    raw = _fresh_raw(dataset)
    retrieve = _RetrieveRecorder(b"x")
    monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
    monkeypatch.setattr(
        GEOparse, "get_GEO", _GeoRecorder(result=None, error=OSError("offline"))
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(geo_message)}$"):
        dataset.download()
    assert retrieve.calls == []
    assert list(raw.iterdir()) == []

    monkeypatch.setattr(GEOparse, "get_GEO", _GeoRecorder(result="gse"))
    monkeypatch.setattr(
        urllib.request, "urlretrieve", _RetrieveRecorder(b"x", error=OSError("404"))
    )
    with pytest.raises(
        RuntimeError, match=f"^{re.escape('Failed to download supplementary data')}$"
    ):
        dataset.download()
    assert [p.name for p in raw.iterdir()] == ["GSE42536.pkl"]


@pytest.mark.parametrize(
    ("cls", "gsms", "expected"),
    [
        (m.SmMicroarraySameith2015Dataset, _SINGLE_GSMS, _SINGLE_EXPECTED),
        (m.DmMicroarraySameith2015Dataset, _DOUBLE_GSMS, _DOUBLE_EXPECTED),
    ],
)
def test_process_refetches_the_geo_object_when_the_pickle_is_missing(
    cls: Any,
    gsms: list[GSM],
    expected: list[tuple[dict[str, Any], dict[str, Any]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``GSE42536.pkl``, ``process`` calls ``get_GEO(geo="GSE42536",
    destdir=<raw_dir>, silent=False)`` once (lines 222 and 1045) and builds the same
    records from the object it returns as from the pickle.
    """
    root = tmp_path / "nopkl"
    _write_raw(root / "raw", gsms)
    (root / "raw" / "GSE42536.pkl").unlink()
    geo = _GeoRecorder(result=_gse(gsms))
    monkeypatch.setattr(GEOparse, "get_GEO", geo)
    dataset = cls(root=str(root))
    assert geo.calls == [
        {"geo": "GSE42536", "destdir": str(root / "raw"), "silent": False}
    ]
    assert len(dataset) == len(expected)
    for i, (experiment, reference) in enumerate(expected):
        assert _nan_safe(dataset[i]["experiment"]) == experiment
        assert dataset[i]["reference"] == reference


def _info(
    accession: str, genes: list[str], gsm: GSM | object, double: bool = True
) -> dict[str, Any]:
    return {
        "geo_accession": accession,
        "is_double_mutant": double,
        "is_single_mutant": not double,
        "gene_names": genes,
        "gsm_object": gsm,
    }


def _no_id_ref_gsm() -> GSM:
    table = pd.DataFrame(
        {"PROBE": [1], "Signal Norm_Cy5": [1.0], "Signal Norm_Cy3": [1.0]}
    )
    return GSM(
        name="X1",
        metadata={"title": ["x"], "source_name_ch1": [""]},
        table=table,
        columns=_describe(table),
    )


_D1, _D2, _D3 = _DOUBLE_GSMS[2], _DOUBLE_GSMS[3], _DOUBLE_GSMS[4]
_BATCH_GROUPS = [
    [
        _info("D1", ["YCR001W", "YDR001C"], _D1),
        _info("D2", ["YCR001W", "YDR001C"], _D2),
    ],
    [_info("N1", ["YCR001W", "YDR001C"], _D1, double=False)],
    [_info("N2", ["YAL001C"], _D3)],
    [_info("D3", ["YAL001C", "YBR001C"], _D3)],
    [_info("U1", ["YFL001W", "YEL001C"], _DOUBLE_GSMS[5])],
]
_BATCH_EXPECTED = [
    *_DOUBLE_EXPECTED,
    _record(
        _DM,
        "BY4742",
        [_kan("YFL001W", "FFA1"), _nat("YEL001C", "EAA1")],
        _ALL_ONES,
        _ALL_ONES,
        _three(0.0, 0.0, 0.0),
        "U1",
    ),
]


def test_double_process_batch_matches_process_sequential_record_by_record(
    double: m.DmMicroarraySameith2015Dataset,
) -> None:
    """The static batch path (run in worker processes under ``process_workers > 0``)
    and the sequential path (which now writes what ``_process_batch`` returns) produce
    the same records, in order, on five hand-made groups: D1 + D2 (two records) and D3
    as in ``_DOUBLE_EXPECTED``; a group flagged not double and a one-gene group are
    skipped; the pair (YFL001W, YEL001C), absent from the SI pairs (D4's pair is
    ``failed``), takes the default strain BY4742 and all-ones signals (log2 0, n 1).
    A group whose array has no ``ID_REF`` column raises instead of being skipped.
    """
    batch = m.DmMicroarraySameith2015Dataset._process_batch(
        _BATCH_GROUPS, _PROBES, _DM, double.gstf_pairs, double.gene_symbols
    )
    batch_records = [pickle.loads(blob) for blob in batch]
    assert len(batch_records) == 4
    for record, (experiment, reference) in zip(
        batch_records, _BATCH_EXPECTED, strict=True
    ):
        assert _nan_safe(record["experiment"]) == experiment
        assert record["reference"] == reference
        assert record["publication"] == _PUBLICATION.model_dump()

    double.close_lmdb()
    lmdb_dir = Path(double.processed_dir) / "lmdb"
    for child in lmdb_dir.iterdir():
        child.unlink()
    double._process_sequential(_BATCH_GROUPS, _PROBES)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    with env.begin() as txn:
        sequential = [pickle.loads(value) for _, value in txn.cursor()]
    env.close()
    assert [_nan_safe(r) for r in sequential] == [_nan_safe(r) for r in batch_records]

    no_data = [[_info("X1", ["YEL001C", "YFL001W"], _no_id_ref_gsm())]]
    with pytest.raises(RuntimeError, match=r"^array X1 .* yields no expression value"):
        m.DmMicroarraySameith2015Dataset._process_batch(
            no_data, _PROBES, _DM, double.gstf_pairs, double.gene_symbols
        )


def test_double_process_batch_takes_strain_from_the_sorted_upper_pair() -> None:
    """The batch path looks the strain up under the SORTED pair, so a title that names
    the genes in reverse SI order still finds its row. The first title gene still
    gets the KanMX marker and the second NatMX; ``Genotype`` then stores the
    perturbations sorted by gene name, so YAL001C (NatMX) comes first.
    """
    pairs = {("YAL001C", "YBR001C"): {"strain": "BY4741"}}
    group = [[_info("D3", ["YBR001C", "YAL001C"], _D3)]]
    [blob] = m.DmMicroarraySameith2015Dataset._process_batch(
        group, _PROBES, _DM, pairs, {}
    )
    record = pickle.loads(blob)
    assert record["reference"]["genome_reference"]["strain"] == "BY4741"
    assert [
        (p["systematic_gene_name"], p["strain_id"])
        for p in record["experiment"]["genotype"]["perturbations"]
    ] == [("YAL001C", "NatMX_YAL001C"), ("YBR001C", "KanMX_YBR001C")]


def test_combine_spots_takes_geometric_channel_means_and_the_mean_log_ratio() -> None:
    """Spots (2 / 1) and (8 / 4): mutant sqrt(16) = 4, refpool sqrt(4) = 2, log2 mean(1,
    1) = 1. Spots (1 / 4) and (4 / 1): mutant 2, refpool 2, log2 mean(-2, 2) = 0, which
    is log2 of the ratio of the two geometric means. One spot returns its own values.
    A zero, negative or NaN signal raises, naming the array and the gene.
    """
    assert m.combine_spots("G", "A", [(2.0, 1.0), (8.0, 4.0)]) == (4.0, 2.0, 1.0)
    assert m.combine_spots("G", "A", [(1.0, 4.0), (4.0, 1.0)]) == (2.0, 2.0, 0.0)
    assert m.combine_spots("G", "A", [(3.0, 7.0)]) == (3.0, 7.0, math.log2(3.0 / 7.0))
    for bad in ((0.0, 1.0), (1.0, -2.0), (_NANF, 1.0)):
        with pytest.raises(
            ValueError, match=r"^G A: spot signals .* not both positive"
        ):
            m.combine_spots("G", "A", [(1.0, 1.0), bad])


class _NoTable:
    """An object with no ``table`` attribute."""


def _gsm_table(table: pd.DataFrame, source: str = "") -> GSM:
    return GSM(
        name="T",
        metadata={"title": ["t"], "source_name_ch1": [source]},
        table=table,
        columns=_describe(table),
    )


_EXTRACT_CASES: list[
    tuple[str, object, dict[str, str] | None, tuple[dict[str, float], ...]]
] = [
    ("no table attribute", _NoTable(), _PROBES, ({}, {}, {})),
    ("no ID_REF", _no_id_ref_gsm(), _PROBES, ({}, {}, {})),
    (
        "no Cy3 column",
        _gsm_table(pd.DataFrame({"ID_REF": [1], "Signal Norm_Cy5": [2.0]})),
        _PROBES,
        ({}, {}, {}),
    ),
    ("no probe map", _D3, None, ({}, {}, {})),
    (
        "unmapped probe",
        _gsm_table(
            pd.DataFrame(
                {
                    "ID_REF": [1, 3, 9],
                    "Signal Norm_Cy5": [4.0, 2.0, 7.0],
                    "Signal Norm_Cy3": [1.0, 2.0, 7.0],
                }
            )
        ),
        _PROBES,
        (
            {"YAL001C": 4.0, "YCR001W": 2.0},
            {"YAL001C": 1.0, "YCR001W": 2.0},
            {"YAL001C": 2.0, "YCR001W": 0.0},
        ),
    ),
    (
        "two spots of one gene",
        _gsm_table(
            pd.DataFrame(
                {
                    "ID_REF": [1, 4],
                    "Signal Norm_Cy5": [2.0, 8.0],
                    "Signal Norm_Cy3": [1.0, 4.0],
                }
            )
        ),
        {"1": "YAL001C", "4": "YAL001C"},
        ({"YAL001C": 4.0}, {"YAL001C": 2.0}, {"YAL001C": 1.0}),
    ),
    (
        "refpool in ch1 swaps the channels",
        _gsm_table(
            pd.DataFrame(
                {"ID_REF": [1], "Signal Norm_Cy5": [2.0], "Signal Norm_Cy3": [8.0]}
            ),
            source="WT RefPool",
        ),
        _PROBES,
        ({"YAL001C": 8.0}, {"YAL001C": 2.0}, {"YAL001C": 2.0}),
    ),
]


@pytest.mark.parametrize(
    ("label", "gsm", "probes", "expected"),
    _EXTRACT_CASES,
    ids=[case[0] for case in _EXTRACT_CASES],
)
def test_static_expression_extraction_equals_the_instance_method_on_every_branch(
    label: str,
    gsm: object,
    probes: dict[str, str] | None,
    expected: tuple[dict[str, float], ...],
    single: m.SmMicroarraySameith2015Dataset,
    double: m.DmMicroarraySameith2015Dataset,
) -> None:
    """Each branch returns (mutant, refpool, log2): a missing table, a missing
    ``ID_REF`` or Cy3 column, or no probe map give three empty dicts; probe 9 is not
    mapped; two probes of one gene are combined (``combine_spots``); a ``refpool``
    source (any case) makes Cy3 the mutant, log2(8/2) = 2. The static,
    double-instance and single-instance extractors agree on every case.
    """
    static = m.DmMicroarraySameith2015Dataset._extract_expression_from_gsm_static
    for extractor in (
        static,
        double._extract_expression_from_gsm,
        single._extract_expression_from_gsm,
    ):
        assert tuple(dict(d) for d in extractor(gsm, probes)) == expected


def test_a_non_numeric_or_zero_cell_raises_in_every_extractor(
    single: m.SmMicroarraySameith2015Dataset, double: m.DmMicroarraySameith2015Dataset
) -> None:
    """A ``"bad"`` cell fails float conversion and a 0 signal has no log ratio: both
    raise instead of being skipped (until 2026-11-24 the first was skipped and the
    second kept without a log2). GSE42536 has neither.
    """
    bad_cell = _gsm_table(
        pd.DataFrame(
            {"ID_REF": [2], "Signal Norm_Cy5": ["bad"], "Signal Norm_Cy3": [1.0]}
        )
    )
    zero = _gsm_table(
        pd.DataFrame(
            {"ID_REF": [3], "Signal Norm_Cy5": [0.0], "Signal Norm_Cy3": [2.0]}
        )
    )
    for extractor in (
        m.DmMicroarraySameith2015Dataset._extract_expression_from_gsm_static,
        double._extract_expression_from_gsm,
        single._extract_expression_from_gsm,
    ):
        with pytest.raises(ValueError, match="could not convert string to float"):
            extractor(bad_cell, _PROBES)
        with pytest.raises(ValueError, match=r"^T YCR001W: spot signals mutant=0\.0"):
            extractor(zero, _PROBES)


@pytest.mark.parametrize("fixture", ["single", "double"])
def test_probe_mapping_branches(fixture: str, request: pytest.FixtureRequest) -> None:
    """Finding: the "Clean and validate gene name" block (sameith2015.py lines
    605-613 and 1631-1639) has three arms that all store ``gene_name.upper()``, so a
    control probe named ``Empty`` becomes the gene ``EMPTY`` exactly like an ORF, and a
    ``None`` cell becomes the gene ``NONE`` (``str(None)``); only the string ``"nan"``
    (a float NaN cell) is dropped. ``SPOT`` is accepted as the ID
    column, ``Gene`` as the gene column when ``ORF`` is absent; a GSE without
    platforms, and a platform without an ID or gene column, map nothing. ``EMPTY`` and
    ``NONE`` do not occur in the real platform, but the same ``.upper()`` path stores
    ``SNR10``, a non-systematic name, as an expression key in every served record (82
    single-mutant, 72 double-mutant; audit 1, 2026.10.06). Pinned until the validation
    arms reject names that are not ORFs.
    """
    dataset = request.getfixturevalue(fixture)

    def platform(table: pd.DataFrame) -> GSE:
        gpl = GPL(name="GPL", metadata={}, table=table, columns=_describe(table))
        return GSE(name="G", metadata={}, gpls={"GPL": gpl}, gsms={})

    spot = pd.DataFrame(
        {
            "SPOT": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Gene": ["yal001c", "q0010", "Empty", None, float("nan")],
        }
    )
    assert dataset._extract_probe_to_gene_mapping(platform(spot)) == {
        "1": "YAL001C",
        "2": "Q0010",
        "3": "EMPTY",
        "4": "NONE",
    }
    assert (
        dataset._extract_probe_to_gene_mapping(
            GSE(name="G", metadata={}, gpls={}, gsms={})
        )
        == {}
    )
    no_gene = pd.DataFrame({"ID": [1], "Description": ["x"]})
    assert dataset._extract_probe_to_gene_mapping(platform(no_gene)) == {}
    no_id = pd.DataFrame({"NAME": [1], "ORF": ["YAL001C"]})
    assert dataset._extract_probe_to_gene_mapping(platform(no_id)) == {}


def test_double_convert_to_systematic_and_name_validation(
    double: m.DmMicroarraySameith2015Dataset, single: m.SmMicroarraySameith2015Dataset
) -> None:
    """The double-mutant ``_convert_to_systematic`` resolves through the same workbook
    map as the single one. ``_is_valid_systematic_name`` rejects the empty string and
    accepts a ``-A`` suffix, in both classes.
    """
    convert = double._convert_to_systematic
    assert [
        convert(name) for name in ("", "yal001c", "nth2", "swt1", "ccr1", "none9")
    ] == [None, "YAL001C", "YBR001C", "YOR166C", "YCR001W", None]
    assert double.gene_symbols == single.gene_symbols
    for dataset in (single, double):
        assert [
            dataset._is_valid_systematic_name(name)
            for name in ("", "YBR089C-A", "ybr089c", "YZR001C", "YAL01C")
        ] == [False, True, True, False, False]


def test_single_process_sequential_skips_non_single_empty_and_no_data_groups(
    single: m.SmMicroarraySameith2015Dataset,
) -> None:
    """Three groups: one flagged not single, one with no gene, and S1 + S2 for
    YAL001C. Only the last writes, one record per array at keys 0 and 1, equal to the
    fixture's first two records. A group whose array has no ``ID_REF`` (no data) raises
    instead of being skipped.
    """

    def info(
        accession: str, genes: list[str], gsm: object, is_single: bool = True
    ) -> dict[str, Any]:
        return {
            "geo_accession": accession,
            "is_single_mutant": is_single,
            "gene_names": genes,
            "gsm_object": gsm,
        }

    groups = [
        [info("S1", ["YAL001C"], _S1, is_single=False)],
        [info("S1", [], _S1)],
        [info("S1", ["YAL001C"], _S1), info("S2", ["YAL001C"], _S2)],
    ]
    single.close_lmdb()
    lmdb_dir = Path(single.processed_dir) / "lmdb"
    for child in lmdb_dir.iterdir():
        child.unlink()
    single._process_sequential(groups, _PROBES)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
    with env.begin() as txn:
        stored = [(key, pickle.loads(value)) for key, value in txn.cursor()]
    env.close()
    assert [key for key, _ in stored] == [b"0", b"1"]
    for (_, record), (experiment, reference) in zip(
        stored, _SINGLE_EXPECTED[:2], strict=True
    ):
        assert _nan_safe(record["experiment"]) == experiment
        assert record["reference"] == reference

    for child in lmdb_dir.iterdir():
        child.unlink()
    with pytest.raises(RuntimeError, match=r"^array X1 .* yields no expression value"):
        single._process_sequential(
            [[info("X1", ["YBR001C"], _no_id_ref_gsm())]], _PROBES
        )


def test_double_class_surface_and_empty_helpers(
    single: m.SmMicroarraySameith2015Dataset, double: m.DmMicroarraySameith2015Dataset
) -> None:
    """The double-mutant class's schema classes and ``preprocess_raw`` pass-through;
    an empty input gives two (wildtype reference) empty SortedDicts; one wildtype
    array (_S4, refpool Cy3 5) gives mean 5 and std 0.0, the ``len(values) > 1``
    guard's fallback instead of a ddof=1 NaN.
    """
    assert double.experiment_class is MicroarrayExpressionExperiment
    assert double.reference_class is MicroarrayExpressionExperimentReference
    frame = pd.DataFrame({"a": [1]})
    assert double.preprocess_raw(frame) is frame
    assert double._calculate_wt_reference_with_std([], _PROBES) == ({}, {})
    mean, std = double._calculate_wt_reference_with_std([_S4], {"1": "YAL001C"})
    assert (dict(mean), dict(std)) == ({"YAL001C": 5.0}, {"YAL001C": 0.0})


# ---------------------------------------------------------------------------
# 2026.11.24: the released GSE42536, one record per array (#1174, #1178).
# ---------------------------------------------------------------------------

_DEV_RAW = {
    kind: Path(
        os.environ["DATA_ROOT"],
        "data",
        "torchcell",
        f"{kind}_microarray_sameith2015",
        "raw",
    )
    for kind in ("sm", "dm")
}
_RAW_FILES = ("GSE42536_family.soft.gz", "GSE42536.pkl", m.WORKBOOK_FILE_NAME)


def _link_raw(kind: str, root: Path) -> None:
    (root / "raw").mkdir(parents=True)
    for name in _RAW_FILES:
        (root / "raw" / name).symlink_to(_DEV_RAW[kind] / name)


def _arrays_per_genotype(dataset: Any) -> dict[tuple[str, ...], list[str]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        key = tuple(
            sorted(
                p["systematic_gene_name"]
                for p in experiment["genotype"]["perturbations"]
            )
        )
        groups.setdefault(key, []).append(experiment["phenotype"]["replicate_id"])
    return groups


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    not all((_DEV_RAW[k] / n).exists() for k in _DEV_RAW for n in _RAW_FILES),
    reason=f"requires the GSE42536 raw files under {_DEV_RAW['sm']} and {_DEV_RAW['dm']}",
)
def test_the_released_series_builds_one_record_per_array(tmp_path: Path) -> None:
    """GSE42536 (287 arrays) builds 144 single-mutant records over 82 mutants (62 x 2
    arrays, 20 x 1) and 143 double-mutant records over 72 pairs (71 x 2, cin5+yap6 x 1),
    one per array, ``replicate_id`` the GSM accession, n = 1 and no SE on every record.
    The 20 one-array singles are exactly the Additional file 1 "Single mutants - info"
    rows whose ``comments`` reads "2 replicates" (one culture, two spots). On
    GSM636383 (dot6-del-1-a, refpool in ch1) the YAL001C value is the mean of its two
    spot log2(Cy3 / Cy5) ratios.
    """
    sm_root, dm_root = tmp_path / "sm", tmp_path / "dm"
    _link_raw("sm", sm_root)
    _link_raw("dm", dm_root)
    single = m.SmMicroarraySameith2015Dataset(root=str(sm_root))
    double = m.DmMicroarraySameith2015Dataset(root=str(dm_root))
    assert (len(single), len(double)) == (144, 143)

    sm_groups = _arrays_per_genotype(single)
    dm_groups = _arrays_per_genotype(double)
    assert len(sm_groups) == 82 and len(dm_groups) == 72
    assert sorted(len(v) for v in sm_groups.values()) == [1] * 20 + [2] * 62
    assert sorted(len(v) for v in dm_groups.values()) == [1] + [2] * 71
    assert [k for k, v in dm_groups.items() if len(v) == 1] == [("YDR259C", "YOR028C")]
    ids = [a for g in (sm_groups, dm_groups) for v in g.values() for a in v]
    assert len(ids) == len(set(ids)) == 287 and all(a.startswith("GSM") for a in ids)

    workbook = pd.read_excel(
        _DEV_RAW["sm"] / m.WORKBOOK_FILE_NAME, sheet_name="Single mutants - info"
    )
    two_replicates = {
        str(orf).strip().upper()
        for orf, comment in zip(workbook["systematic name"], workbook["comments"])
        if isinstance(comment, str) and "2 replicates" in comment
    }
    assert {k[0] for k, v in sm_groups.items() if len(v) == 1} == two_replicates

    for dataset in (single, double):
        for i in range(len(dataset)):
            phenotype = dataset[i]["experiment"]["phenotype"]
            assert set(phenotype["n_replicates"].values()) == {1}
            assert phenotype["expression_log2_ratio_se"] is None
            assert (
                dataset[i]["reference"]["phenotype_reference"]["replicate_id"] is None
            )

    with open(_DEV_RAW["sm"] / "GSE42536.pkl", "rb") as handle:
        gse = pickle.load(handle)
    probes = gse.gpls["GPL11232"].table
    spot_ids = set(probes.loc[probes["ORF"] == "YAL001C", "ID"].astype(int))
    table = gse.gsms["GSM636383"].table
    spots = table[table["ID_REF"].astype(int).isin(spot_ids)]
    assert len(spots) == 2
    expected = float(
        np.mean(np.log2(spots["Signal Norm_Cy3"] / spots["Signal Norm_Cy5"]))
    )
    [record] = [
        single[i]
        for i in range(len(single))
        if single[i]["experiment"]["phenotype"]["replicate_id"] == "GSM636383"
    ]
    assert record["experiment"]["phenotype"]["expression_log2_ratio"][
        "YAL001C"
    ] == pytest.approx(expected, abs=1e-12)

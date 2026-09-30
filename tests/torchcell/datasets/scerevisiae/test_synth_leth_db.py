# tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
# [[tests.torchcell.datasets.scerevisiae.test_synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
"""SynLethDB yeast synthetic-lethality and synthetic-rescue loaders, built hermetically.

The loaders resolve common names to systematic names through ``genome.db.all_features()``
BEFORE the PyG base class runs, so the genome is a duck-typed stub yielding gffutils-like
features (``featuretype``, ``id``, ``attributes``). The raw CSV is written into
``<root>/raw/`` so ``download()`` is never called. Expected records are hand-built from
the schema classes and compared by ``model_dump`` equality. Nothing touches ``$DATA_ROOT``.

The stub genome has six ``gene`` features plus a ``CDS`` and an ``mRNA`` that must be
ignored. The SL fixture has five rows (common name, alias, systematic passthrough, a
name-less ORF, and a NaN score); the SR fixture has two rows (one NaN score).

2026.09.30 (Phase 14): a second SL build (``edge_sl``) on a two-gene stub whose alias
``SHARED`` is listed on BOTH YAL001C and YAL002W (the later feature wins the dict
write), with five rows: ``TFC3,VPS8,0.2,111``; the same pair swapped
(``VPS8,TFC3,0.2,111``); a synonym self-pair ``TFC3,TSV115,0.7,222`` (both YAL001C);
``TFC3,SHARED,0.4,333``; and ``TFC3,VPS8,0.5,`` with a blank PMID. Expected: five
records, the swapped pair stored as an identical second record (the genotype sorts its
perturbations), the self-pair stored as a double deletion of YAL001C, ``SHARED``
resolved to YAL002W, and, because one blank cell makes pandas read the PMID column as
float, every record's ``pubmed_id`` is ``"111.0"``-shaped and the blank one ``"nan"``.
One reference covers [0..4]; the gene set is [YAL001C, YAL002W]. Also pinned: both
``download`` methods against a fake ``requests.Session`` (the plain response, the
``download_warning`` confirm round trip, and an HTTP error that writes no file) with
their exact Google Drive URLs, and ``main`` against a stub genome.

Findings pinned: the PMID float cast (line 229), the stored duplicate and self-pair
(no deduplication in ``process``, lines 157-170), the alias last-write-wins (line 77),
and ``main`` building both datasets under the working directory (the default relative
roots, lines 49 and 243) while the genome is read from ``$DATA_ROOT``.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from pydantic import ValidationError

from torchcell.data import ExperimentDataset
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    SyntheticRescuePhenotype,
    Temperature,
)
from torchcell.datasets.scerevisiae import synth_leth_db as s
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class _Feature:
    """The three attributes ``_build_gene_name_mapping`` reads off a gffutils feature."""

    def __init__(
        self, featuretype: str, id: str, attributes: dict[str, list[str]]
    ) -> None:
        self.featuretype = featuretype
        self.id = id
        self.attributes = attributes


class _Db:
    def __init__(self, features: list[_Feature]) -> None:
        self._features = features

    def all_features(self) -> list[_Feature]:
        return list(self._features)


class _StubGenome:
    def __init__(self) -> None:
        self.db = _Db(_FEATURES)


_FEATURES = [
    _Feature("gene", "YAL001C", {"gene": ["TFC3"], "Alias": ["TSV115", "FUN24"]}),
    _Feature("CDS", "YAL001C_CDS", {"gene": ["TFC3"]}),
    _Feature("gene", "YAL002W", {"gene": ["VPS8"], "Alias": ["FUN15"]}),
    _Feature("mRNA", "YAL002W_mRNA", {"gene": ["VPS8"]}),
    _Feature("gene", "YAL003W", {"gene": ["EFB1"], "Alias": ["TEF5"]}),
    _Feature("gene", "YAL005C", {"gene": ["SSA1"], "Alias": ["YG100"]}),
    _Feature("gene", "YAL008W", {"gene": ["FUN14"]}),
    _Feature("gene", "YAL012W", {}),
]
_EXPECTED_MAPPING = {
    "YAL001C": "YAL001C",
    "TFC3": "YAL001C",
    "TSV115": "YAL001C",
    "FUN24": "YAL001C",
    "YAL002W": "YAL002W",
    "VPS8": "YAL002W",
    "FUN15": "YAL002W",
    "YAL003W": "YAL003W",
    "EFB1": "YAL003W",
    "TEF5": "YAL003W",
    "YAL005C": "YAL005C",
    "SSA1": "YAL005C",
    "YG100": "YAL005C",
    "YAL008W": "YAL008W",
    "FUN14": "YAL008W",
    "YAL012W": "YAL012W",
}

_HEADER = "n1.name,n2.name,r.statistic_score,r.pubmed_id\n"
# Row 1 uses an alias, row 2 a systematic name and a primed name, row 3 a name-less
# ORF, row 4 an empty score (NaN).
_SL_ROWS = [
    "TFC3,VPS8,0.85,12345678\n",
    "TSV115,EFB1,0.5,23456789\n",
    "YAL005C,FUN14',0.1,34567890\n",
    "YAL012W,TEF5,0.3,45678901\n",
    "SSA1,VPS8,,56789012\n",
]
_SR_ROWS = ["TFC3,VPS8,0.42,11111111\n", "SSA1,YAL012W,,22222222\n"]

_ENVIRONMENT = Environment(
    media=Media(name="YEPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30),
)
_GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _write_raw(root: Path, filename: str, rows: list[str]) -> None:
    (root / "raw").mkdir(parents=True)
    (root / "raw" / filename).write_text(_HEADER + "".join(rows), encoding="utf-8")


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _pair(systematic: tuple[str, str], perturbed: tuple[str, str]) -> Genotype:
    return Genotype(
        perturbations=[
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=systematic[0],
                perturbed_gene_name=perturbed[0],
                strain_id="S288C",
            ),
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=systematic[1],
                perturbed_gene_name=perturbed[1],
                strain_id="S288C",
            ),
        ]
    )


def _publication(pubmed_id: str) -> Publication:
    return Publication(
        pubmed_id=pubmed_id,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
        doi=None,
        doi_url=None,
    )


@pytest.fixture(scope="module")
def sl(
    tmp_path_factory: pytest.TempPathFactory,
) -> s.SynthLethalityYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "sl"
    _write_raw(root, "Yeast_SL.csv", _SL_ROWS)
    return s.SynthLethalityYeastSynthLethDbDataset(root=str(root), genome=_genome())


@pytest.fixture(scope="module")
def sr(
    tmp_path_factory: pytest.TempPathFactory,
) -> s.SynthRescueYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "sr"
    _write_raw(root, "Yeast_SR.csv", _SR_ROWS)
    return s.SynthRescueYeastSynthLethDbDataset(root=str(root), genome=_genome())


def test_gene_name_mapping_reads_gene_features_only_and_drops_the_genome(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """The mapping is exactly ``_EXPECTED_MAPPING``: every ``gene`` feature maps its id,
    its ``gene`` attribute, and each ``Alias`` to the id (16 keys); the ``CDS`` and
    ``mRNA`` features contribute nothing; the name-less ORF maps only its id. The
    ``genome`` attribute is deleted after the mapping so the dataset stays picklable.
    """
    assert sl.gene_name_to_systematic == _EXPECTED_MAPPING
    assert len(sl.gene_name_to_systematic) == 16
    assert "YAL001C_CDS" not in sl.gene_name_to_systematic
    assert "genome" not in vars(sl)
    assert sl.get_systematic_name("FUN14'") == "YAL008W"
    assert sl.raw_file_names == ["Yeast_SL.csv"]
    assert sl.processed_file_names == ["lmdb"]


def test_sl_common_name_pair_record_matches_the_source_row(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Record 0 (``TFC3,VPS8,0.85,12345678``): two SGA KanMX deletions with
    ``strain_id="S288C"`` on YAL001C / YAL002W, YEPD solid non-synthetic at 30 C,
    ``is_synthetic_lethal True`` with score 0.85; reference ``False`` with score None;
    publication PMID 12345678 with no DOI.
    """
    expected = SyntheticLethalityExperiment(
        dataset_name="SynthLethalityYeastSynthLethDbDataset",
        genotype=_pair(("YAL001C", "YAL002W"), ("TFC3", "VPS8")),
        environment=_ENVIRONMENT,
        phenotype=SyntheticLethalityPhenotype(
            is_synthetic_lethal=True, synthetic_lethality_statistic_score=0.85
        ),
    )
    expected_reference = SyntheticLethalityExperimentReference(
        dataset_name="SynthLethalityYeastSynthLethDbDataset",
        genome_reference=_GENOME,
        environment_reference=_ENVIRONMENT,
        phenotype_reference=SyntheticLethalityPhenotype(
            is_synthetic_lethal=False, synthetic_lethality_statistic_score=None
        ),
    )
    assert len(sl) == 5
    record = sl[0]
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == _publication("12345678").model_dump()


@pytest.mark.parametrize(
    ("index", "systematic", "perturbed", "score", "pubmed_id"),
    [
        (1, ["YAL001C", "YAL003W"], ["TSV115", "EFB1"], 0.5, "23456789"),
        (2, ["YAL005C", "YAL008W"], ["YAL005C", "FUN14_prime"], 0.1, "34567890"),
        (3, ["YAL003W", "YAL012W"], ["TEF5", "YAL012W"], 0.3, "45678901"),
    ],
)
def test_sl_alias_systematic_and_primed_names_resolve(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
    index: int,
    systematic: list[str],
    perturbed: list[str],
    score: float,
    pubmed_id: str,
) -> None:
    """Row 1: alias TSV115 -> YAL001C. Row 2: YAL005C passes through and ``FUN14'`` maps
    to YAL008W with the prime normalized to ``FUN14_prime`` by the schema. Row 3: the
    name-less ORF YAL012W maps to itself and sorts after YAL003W. Scores and PMIDs are
    the raw column values (PMID stringified from the integer column).
    """
    typed = sl.transform_item(sl[index])
    genotype = typed["experiment"].genotype
    assert genotype.systematic_gene_names == systematic
    assert genotype.perturbed_gene_names == perturbed
    assert typed["experiment"].phenotype.synthetic_lethality_statistic_score == score
    assert typed["publication"] == _publication(pubmed_id)


def test_sl_nan_statistic_score_is_stored_as_nan_not_none(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Finding: the SL loader does ``float(row["r.statistic_score"])`` with no NaN guard,
    so an empty score cell is stored as ``nan`` (the SR loader maps the same cell to
    ``None``). Record 4 (``SSA1,VPS8,,56789012``) pins this asymmetry as it behaves.
    """
    phenotype = sl[4]["experiment"]["phenotype"]
    assert phenotype["is_synthetic_lethal"] is True
    assert math.isnan(phenotype["synthetic_lethality_statistic_score"])
    assert sl[4]["publication"]["pubmed_id"] == "56789012"
    assert sl.transform_item(sl[4])["experiment"].genotype.systematic_gene_names == [
        "YAL002W",
        "YAL005C",
    ]


def test_sl_side_files_single_reference_gene_set_manifest_and_no_interning(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``preprocess/`` holds gene_set, reference index, and manifest but NO data.csv (the
    SL loader never saves a preprocessed frame, so ``df`` is None); one reference covers
    members [0..4]; gene set = the six systematic names sorted; the records LMDB is
    written directly with ``pickle`` so no ``processed/interned`` env exists.
    """
    assert sorted(os.listdir(sl.preprocess_dir)) == [
        "build_manifest.json",
        "experiment_reference_index.json",
        "gene_set.json",
    ]
    assert sl.df is None
    with open(osp.join(sl.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == [
            "YAL001C",
            "YAL002W",
            "YAL003W",
            "YAL005C",
            "YAL008W",
            "YAL012W",
        ]
    with open(osp.join(sl.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [[0, 1, 2, 3, 4]]
    assert stored[0]["reference"]["phenotype_reference"]["is_synthetic_lethal"] is False
    with open(osp.join(sl.preprocess_dir, "build_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["loader_class"] == "SynthLethalityYeastSynthLethDbDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.synth_leth_db"
    assert manifest["dataset_name"] == "sl"
    assert sorted(os.listdir(sl.processed_dir)) == [
        "lmdb",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_sr_records_match_the_source_rows_and_nan_score_becomes_none(
    sr: s.SynthRescueYeastSynthLethDbDataset,
) -> None:
    """Record 0 (``TFC3,VPS8,0.42,11111111``): ``is_synthetic_rescue True`` score 0.42,
    reference ``False`` / None. Record 1 (``SSA1,YAL012W,,22222222``): the empty score is
    ``None`` via ``pd.notna``; both perturbations carry ``strain_id="S288C"``.
    """
    expected = SyntheticRescueExperiment(
        dataset_name="SynthRescueYeastSynthLethDbDataset",
        genotype=_pair(("YAL001C", "YAL002W"), ("TFC3", "VPS8")),
        environment=_ENVIRONMENT,
        phenotype=SyntheticRescuePhenotype(
            is_synthetic_rescue=True, synthetic_rescue_statistic_score=0.42
        ),
    )
    expected_reference = SyntheticRescueExperimentReference(
        dataset_name="SynthRescueYeastSynthLethDbDataset",
        genome_reference=_GENOME,
        environment_reference=_ENVIRONMENT,
        phenotype_reference=SyntheticRescuePhenotype(
            is_synthetic_rescue=False, synthetic_rescue_statistic_score=None
        ),
    )
    assert len(sr) == 2
    assert sr[0]["experiment"] == expected.model_dump()
    assert sr[0]["reference"] == expected_reference.model_dump()
    assert sr[0]["publication"] == _publication("11111111").model_dump()
    second = sr.transform_item(sr[1])
    assert second["experiment"].phenotype.synthetic_rescue_statistic_score is None
    assert second["experiment"].phenotype.is_synthetic_rescue is True
    assert second["experiment"].genotype.systematic_gene_names == ["YAL005C", "YAL012W"]
    assert [p.strain_id for p in second["experiment"].genotype.perturbations] == [
        "S288C",
        "S288C",
    ]
    assert second["publication"] == _publication("22222222")
    assert sr.raw_file_names == ["Yeast_SR.csv"]


def test_sr_side_files_single_reference_and_gene_set(
    sr: s.SynthRescueYeastSynthLethDbDataset,
) -> None:
    """One reference with members [0, 1]; gene set = the four systematic names sorted;
    the manifest names the SR loader.
    """
    with open(osp.join(sr.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == ["YAL001C", "YAL002W", "YAL005C", "YAL012W"]
    with open(osp.join(sr.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [[0, 1]]
    assert stored[0]["reference"]["experiment_reference_type"] == "synthetic rescue"
    with open(osp.join(sr.preprocess_dir, "build_manifest.json")) as f:
        assert json.load(f)["loader_class"] == "SynthRescueYeastSynthLethDbDataset"


@pytest.mark.parametrize(
    ("cls", "filename"),
    [
        (s.SynthLethalityYeastSynthLethDbDataset, "Yeast_SL.csv"),
        (s.SynthRescueYeastSynthLethDbDataset, "Yeast_SR.csv"),
    ],
)
def test_unknown_gene_name_falls_back_to_itself_and_fails_schema_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cls: type[Any], filename: str
) -> None:
    """Finding: ``get_systematic_name`` returns the raw name for an unmapped gene
    (``NOTAGENE``), which then fails ``GenePerturbation``'s systematic-name regex, so the
    build raises ``ValidationError("Invalid systematic gene name format")`` after printing
    the loader's warning line. The documented "fall back" is therefore always a crash.
    """
    root = tmp_path / "bad"
    _write_raw(root, filename, ["NOTAGENE,VPS8,0.9,99999999\n"])
    with pytest.raises(ValidationError, match="Invalid systematic gene name format"):
        cls(root=str(root), genome=_genome())
    assert (
        "Warning: No systematic name found for gene NOTAGENE" in capsys.readouterr().out
    )


# --------------------------------------------------------------------------- #
# Phase 14: duplicates, synonyms, the PMID column type, download, main
# --------------------------------------------------------------------------- #

_EDGE_FEATURES = [
    _Feature("gene", "YAL001C", {"gene": ["TFC3"], "Alias": ["TSV115", "SHARED"]}),
    _Feature("gene", "YAL002W", {"gene": ["VPS8"], "Alias": ["SHARED"]}),
]
_EDGE_ROWS = [
    "TFC3,VPS8,0.2,111\n",
    "VPS8,TFC3,0.2,111\n",
    "TFC3,TSV115,0.7,222\n",
    "TFC3,SHARED,0.4,333\n",
    "TFC3,VPS8,0.5,\n",
]


class _EdgeGenome:
    def __init__(self) -> None:
        self.db = _Db(_EDGE_FEATURES)


@pytest.fixture(scope="module")
def edge_sl(
    tmp_path_factory: pytest.TempPathFactory,
) -> s.SynthLethalityYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "edge"
    _write_raw(root, "Yeast_SL.csv", _EDGE_ROWS)
    return s.SynthLethalityYeastSynthLethDbDataset(
        root=str(root), genome=cast(SCerevisiaeGenome, _EdgeGenome())
    )


def _pairs(
    dataset: s.SynthLethalityYeastSynthLethDbDataset,
) -> list[list[tuple[str, str]]]:
    return [
        [
            (p["systematic_gene_name"], p["perturbed_gene_name"])
            for p in dataset[i]["experiment"]["genotype"]["perturbations"]
        ]
        for i in range(len(dataset))
    ]


def test_a_shared_alias_resolves_to_the_later_gene(
    edge_sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Finding: ``SHARED`` is an alias of both genes; the mapping keeps the last write
    (line 77), so row 3 is stored as a deletion of YAL002W with no warning. Pinned until
    an alias on more than one gene is refused or logged as ambiguous.
    """
    assert edge_sl.gene_name_to_systematic == {
        "YAL001C": "YAL001C",
        "TFC3": "YAL001C",
        "TSV115": "YAL001C",
        "SHARED": "YAL002W",
        "YAL002W": "YAL002W",
        "VPS8": "YAL002W",
    }
    assert _pairs(edge_sl)[3] == [("YAL001C", "TFC3"), ("YAL002W", "SHARED")]


def test_a_swapped_duplicate_pair_is_stored_twice_and_a_synonym_pair_as_a_self_pair(
    edge_sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Finding: rows 0 and 1 are one pair in two orders and become two identical records;
    row 2 names YAL001C twice (TFC3 and its alias TSV115) and is stored as a double
    deletion of one ORF. ``process`` writes every row (lines 157-170). Pinned until a
    repeated or self pair is refused or merged.
    """
    assert len(edge_sl) == 5
    assert edge_sl[1]["experiment"] == edge_sl[0]["experiment"]
    assert _pairs(edge_sl) == [
        [("YAL001C", "TFC3"), ("YAL002W", "VPS8")],
        [("YAL001C", "TFC3"), ("YAL002W", "VPS8")],
        [("YAL001C", "TFC3"), ("YAL001C", "TSV115")],
        [("YAL001C", "TFC3"), ("YAL002W", "SHARED")],
        [("YAL001C", "TFC3"), ("YAL002W", "VPS8")],
    ]
    index = edge_sl.experiment_reference_index
    assert index is not None
    assert [e.member_indices for e in index] == [[0, 1, 2, 3, 4]]
    with open(osp.join(edge_sl.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == ["YAL001C", "YAL002W"]


def test_one_blank_pmid_turns_every_pmid_into_a_float_string(
    edge_sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Finding: the blank PMID in row 4 makes ``pd.read_csv`` type the column float, and
    ``str(row["r.pubmed_id"])`` (line 229) then stores ``"111.0"`` and the URL
    ``.../111.0/`` for EVERY row, and ``"nan"`` for the blank one. Pinned until the PMID
    column is read as a string and a blank PMID is refused.
    """
    assert [edge_sl[i]["publication"] for i in range(len(edge_sl))] == [
        _publication(p).model_dump()
        for p in ["111.0", "111.0", "222.0", "333.0", "nan"]
    ]
    assert edge_sl[0]["publication"]["pubmed_url"] == (
        "https://pubmed.ncbi.nlm.nih.gov/111.0/"
    )


class _Response:
    def __init__(
        self, chunks: list[bytes], cookies: dict[str, str], status: int
    ) -> None:
        self.cookies = cookies
        self._chunks = chunks
        self._status = status
        self.chunk_sizes: list[int] = []

    def raise_for_status(self) -> None:
        if self._status != 200:
            raise requests.HTTPError(f"{self._status} Client Error")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        self.chunk_sizes.append(chunk_size)
        return self._chunks


def _fake_session(
    monkeypatch: pytest.MonkeyPatch, responses: list[_Response]
) -> list[tuple[str, dict[str, str] | None, bool]]:
    calls: list[tuple[str, dict[str, str] | None, bool]] = []

    class _Session:
        def get(
            self, url: str, params: dict[str, str] | None = None, stream: bool = False
        ) -> _Response:
            calls.append((url, params, stream))
            return responses[len(calls) - 1]

    monkeypatch.setattr(requests, "Session", _Session)
    return calls


_DOWNLOADS = [
    (
        s.SynthLethalityYeastSynthLethDbDataset,
        "Yeast_SL.csv",
        "https://drive.google.com/uc?export=download&id=1_56ebyBatapNml8S5HlJW7Dz1l0DZZIq",
    ),
    (
        s.SynthRescueYeastSynthLethDbDataset,
        "Yeast_SR.csv",
        "https://drive.google.com/uc?export=download&id=1lBaApm70E05JnkrE1Hwmn8gT1cV5Bzlt",
    ),
]


def _bare(cls: type[ExperimentDataset], root: Path) -> ExperimentDataset:
    dataset = cls.__new__(cls)
    dataset.root = str(root)
    return dataset


@pytest.mark.parametrize(("cls", "filename", "url"), _DOWNLOADS)
def test_download_writes_the_streamed_chunks_in_one_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[ExperimentDataset],
    filename: str,
    url: str,
) -> None:
    response = _Response([b"n1.name,", b"n2.name\n"], {}, 200)
    calls = _fake_session(monkeypatch, [response])
    _bare(cls, tmp_path).download()
    assert calls == [(url, None, True)]
    assert response.chunk_sizes == [8192]
    assert (tmp_path / "raw" / filename).read_bytes() == b"n1.name,n2.name\n"


@pytest.mark.parametrize(("cls", "filename", "url"), _DOWNLOADS)
def test_download_confirms_a_drive_warning_cookie_and_keeps_the_second_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[ExperimentDataset],
    filename: str,
    url: str,
) -> None:
    warning = _Response([b"<html>virus scan</html>"], {"download_warning": "tok"}, 200)
    body = _Response([b"payload"], {}, 200)
    calls = _fake_session(monkeypatch, [warning, body])
    _bare(cls, tmp_path).download()
    assert calls == [(url, None, True), (url, {"confirm": "tok"}, True)]
    assert warning.chunk_sizes == []
    assert (tmp_path / "raw" / filename).read_bytes() == b"payload"


@pytest.mark.parametrize(("cls", "filename", "url"), _DOWNLOADS)
def test_download_raises_on_an_http_error_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[ExperimentDataset],
    filename: str,
    url: str,
) -> None:
    _fake_session(monkeypatch, [_Response([b"x"], {}, 404)])
    with pytest.raises(requests.HTTPError) as excinfo:
        _bare(cls, tmp_path).download()
    assert str(excinfo.value) == "404 Client Error"
    assert os.listdir(tmp_path / "raw") == []


def test_main_builds_both_datasets_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: ``main`` reads the genome from ``$DATA_ROOT`` but builds both datasets at
    their default RELATIVE roots (lines 49 and 243), so the LMDBs land under the current
    working directory, not under ``$DATA_ROOT``. Pinned until ``main`` joins the roots
    onto ``$DATA_ROOT``.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    genome_kwargs: list[dict[str, Any]] = []

    class _MainGenome(_StubGenome):
        def __init__(self, **kwargs: Any) -> None:
            genome_kwargs.append(kwargs)
            super().__init__()

    monkeypatch.setattr(s, "SCerevisiaeGenome", _MainGenome)
    cwd = tmp_path / "cwd"
    _write_raw(cwd / "data/torchcell/syn_leth_db_yeast", "Yeast_SL.csv", _SL_ROWS)
    _write_raw(cwd / "data/torchcell/syn_rescue_db_yeast", "Yeast_SR.csv", _SR_ROWS)
    monkeypatch.chdir(cwd)
    s.main()
    assert genome_kwargs == [
        {
            "genome_root": osp.join(str(data_root), "data/sgd/genome"),
            "go_root": osp.join(str(data_root), "data/go"),
            "overwrite": False,
        }
    ]
    lines = capsys.readouterr().out.splitlines()
    assert "SynthLethalityYeastSynthLethDbDataset(5)" in lines
    assert "SynthRescueYeastSynthLethDbDataset(2)" in lines
    assert (cwd / "data/torchcell/syn_leth_db_yeast/processed/lmdb").is_dir()
    assert (cwd / "data/torchcell/syn_rescue_db_yeast/processed/lmdb").is_dir()
    assert not data_root.exists()


def test_lethality_items_retype_through_the_lethality_classes(
    edge_sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``transform_item`` rebuilds a stored lethality item as a
    ``SyntheticLethalityExperiment`` with its reference, dumping to exactly the stored
    dictionaries (``experiment_dataset.py`` lines 638 to 641). The rescue dataset has no
    synthetic build in this file, so its classes are read from a bare instance: a
    ``SyntheticRescueExperiment`` refuses the lethality item's experiment dictionary,
    which is the observable difference between the two wirings.
    """
    item = edge_sl[0]
    typed = edge_sl.transform_item(item)
    assert type(typed["experiment"]) is SyntheticLethalityExperiment
    assert type(typed["reference"]) is SyntheticLethalityExperimentReference
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    rescue = s.SynthRescueYeastSynthLethDbDataset.__new__(
        s.SynthRescueYeastSynthLethDbDataset
    )
    assert rescue.processed_file_names == ["lmdb"]
    assert rescue.experiment_class is SyntheticRescueExperiment
    assert rescue.reference_class is SyntheticRescueExperimentReference
    with pytest.raises(ValidationError):
        rescue.experiment_class(**item["experiment"])

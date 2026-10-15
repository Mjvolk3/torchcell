# tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
# [[tests.torchcell.datasets.scerevisiae.test_synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
"""SynLethDB yeast synthetic-lethality and synthetic-rescue loaders, built hermetically.

2026.10.12 (#1045): each side of a row is stored exactly as released, on
``ReleasedGeneNamePerturbation``: the name verbatim (``n1.name`` / ``n2.name``), the
Entrez id verbatim (``n1.identifier`` / ``n2.identifier``, namespace
``ncbi_entrez_gene``) and a typed gap for the perturbation kind the release does not
state. No genome or GFF is read, nothing is resolved or dropped, and a row naming one
gene on both sides is stored as released. A repeated unordered (name, id) pair and a
blank PMID refuse the build.

The raw CSV is written into ``<root>/raw/`` so ``download()`` is never called; the
build-time CSV pin is swapped for a presence recorder by ``tests/torchcell/conftest.py``.
Expected records are hand-built from the schema classes. Nothing touches ``$DATA_ROOT``
except the ``data``-marked test at the end, which reads the real pinned files.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from torchcell.data import ExperimentDataset
from torchcell.datamodels.schema import (
    Genotype,
    Publication,
    ReferenceGenome,
    ReleasedGeneIdNamespace,
    ReleasedGeneNamePerturbation,
    SyntheticLethalityExperiment,
    SyntheticLethalityExperimentReference,
    SyntheticLethalityPhenotype,
    SyntheticRescueExperiment,
    SyntheticRescueExperimentReference,
    SyntheticRescuePhenotype,
)
from torchcell.datasets.scerevisiae import synth_leth_db as s
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)

_HEADER = (
    "n1.name,n1.identifier,n2.name,n2.identifier,r.statistic_score,r.pubmed_id,"
    "r.source\n"
)
_SL_ROWS = [
    "TFC3,1001,VPS8,1002,0.85,12345678,Low Throughput\n",  # 0: standard names
    "TSV115,1001,EFB1,1003,0.5,23456789,Low Throughput\n",  # 1: an alias, as released
    "YAL005C,1005,IMP2',1011,0.1,34567890,Low Throughput\n",  # 2: a systematic name, a primed name
    "YAL014C,1014,TEF5,1003,0.3,45678901,Low Throughput\n",  # 3: a name-less ORF
    "SSA1,1005,VPS8,1002,,56789012,Low Throughput\n",  # 4: empty score
    "STM1,1012,EFB1,1003,0.2,333;444,Low Throughput\n",  # 5: two PMIDs
    "VPS8,1001,EFB1,1003,0.4,222,Low Throughput\n",  # 6: a name on another gene's id
    "TFC3,1001,GHOST,9999,0.4,333,Low Throughput\n",  # 7: an id no annotation carries
    "TFC3,1001,TFC3,1001,0.7,444,Low Throughput\n",  # 8: one gene on both sides
    "SHARED,1002,TFC3,1001,0.6,555,Low Throughput\n",  # 9: an alias of two genes
]
_SR_ROWS = [
    "TFC3,1001,VPS8,1002,0.42,11111111,BioGRID\n",
    "SSA1,1005,YAL014C,1014,,22222222,BioGRID\n",
    "EFB1,1003,EFB1,1003,,33333333,BioGRID\n",
]

_ENVIRONMENT = s.SYNLETHDB_ENVIRONMENT
_GENOME = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


def _write_raw(root: Path, filename: str, rows: list[str]) -> None:
    (root / "raw").mkdir(parents=True)
    (root / "raw" / filename).write_text(_HEADER + "".join(rows), encoding="utf-8")


def _side(name: str, entrez: int) -> ReleasedGeneNamePerturbation:
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=name,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=name,
        released_gene_name=name,
        released_gene_id=str(entrez),
        released_gene_id_namespace=ReleasedGeneIdNamespace.ncbi_entrez_gene,
        released_kind=None,
        provenance_gaps=[s.SYNLETHDB_KIND_GAP],
    )


def _pair(a: tuple[str, int], b: tuple[str, int]) -> Genotype:
    return Genotype(perturbations=[_side(*a), _side(*b)])


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
    _write_raw(root, s.SL_CSV_NAME, _SL_ROWS)
    return s.SynthLethalityYeastSynthLethDbDataset(root=str(root))


@pytest.fixture(scope="module")
def sr(
    tmp_path_factory: pytest.TempPathFactory,
) -> s.SynthRescueYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "sr"
    _write_raw(root, s.SR_CSV_NAME, _SR_ROWS)
    return s.SynthRescueYeastSynthLethDbDataset(root=str(root))


def _pairs(dataset: ExperimentDataset) -> list[list[tuple[str, str, str | None]]]:
    return [
        [
            (p["systematic_gene_name"], p["perturbed_gene_name"], p["released_gene_id"])
            for p in dataset[i]["experiment"]["genotype"]["perturbations"]
        ]
        for i in range(len(dataset))
    ]


def _ledger(dataset: ExperimentDataset) -> s.DropLog:
    with open(osp.join(dataset.preprocess_dir, "dropped_records.json")) as f:
        return s.DropLog.model_validate_json(f.read())


# --------------------------------------------------------------------------- #
# Synthetic lethality build
# --------------------------------------------------------------------------- #


def test_sl_keeps_every_row_in_source_order_as_released(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """All ten rows are records 0-9 under the names and ids the file gives: ``TSV115``
    stays TSV115 (never its standard name), ``IMP2'`` keeps its prime in the identity
    and the schema spells the perturbed name ``IMP2_prime``, ``GHOST`` (an id no
    annotation carries) and ``SHARED`` (an alias of two genes) are records, and row 8
    names TFC3 on both sides. ``Genotype`` sorts the two sides by name.
    """
    assert len(sl) == 10
    assert _pairs(sl) == [
        [("TFC3", "TFC3", "1001"), ("VPS8", "VPS8", "1002")],
        [("EFB1", "EFB1", "1003"), ("TSV115", "TSV115", "1001")],
        [("IMP2'", "IMP2_prime", "1011"), ("YAL005C", "YAL005C", "1005")],
        [("TEF5", "TEF5", "1003"), ("YAL014C", "YAL014C", "1014")],
        [("SSA1", "SSA1", "1005"), ("VPS8", "VPS8", "1002")],
        [("EFB1", "EFB1", "1003"), ("STM1", "STM1", "1012")],
        [("EFB1", "EFB1", "1003"), ("VPS8", "VPS8", "1001")],
        [("GHOST", "GHOST", "9999"), ("TFC3", "TFC3", "1001")],
        [("TFC3", "TFC3", "1001"), ("TFC3", "TFC3", "1001")],
        [("SHARED", "SHARED", "1002"), ("TFC3", "TFC3", "1001")],
    ]
    assert "genome" not in vars(sl)
    assert sl.raw_file_names == ["Yeast_SL.csv"]
    assert sl.processed_file_names == ["lmdb"]


def test_sl_record_matches_the_source_row_with_a_gapped_kind(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Record 0 (``TFC3,1001,VPS8,1002,0.85,12345678``): two released-gene-name
    perturbations carrying the name, the Entrez id (namespace ``ncbi_entrez_gene``)
    and a ``not_reported_by_primary`` gap on ``released_kind`` (SynLethDB states the
    pair, not the perturbation); the ``SYNLETHDB_ENVIRONMENT`` (medium and
    temperature not carried, issue #622); ``is_synthetic_lethal True`` with score
    0.85; reference ``False`` with score None; publication PMID 12345678 with no DOI.
    """
    expected = SyntheticLethalityExperiment(
        dataset_name="SynthLethalityYeastSynthLethDbDataset",
        genotype=_pair(("TFC3", 1001), ("VPS8", 1002)),
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
    record = sl[0]
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == _publication("12345678").model_dump()
    side = record["experiment"]["genotype"]["perturbations"][0]
    assert side["perturbation_type"] == "released_gene_name"
    assert side["released_kind"] is None
    (gap,) = side["provenance_gaps"]
    assert (gap["field"], gap["reason"]) == ("released_kind", "not_reported_by_primary")
    assert gap["looked_in"]["sha256"] == s.SYNLETHDB_PAPER_SHA256
    assert s.SYNLETHDB_ENTRY_FIELDS_QUOTE in gap["note"]


def test_sl_nan_statistic_score_is_stored_as_nan_not_none(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Finding: the SL loader does ``float(row["r.statistic_score"])`` with no NaN guard,
    so an empty score cell is stored as ``nan`` (the SR loader maps the same cell to
    ``None``). Record 4 (``SSA1,1005,VPS8,1002,,56789012``) pins this asymmetry.
    """
    phenotype = sl[4]["experiment"]["phenotype"]
    assert phenotype["is_synthetic_lethal"] is True
    assert math.isnan(phenotype["synthetic_lethality_statistic_score"])
    assert sl[4]["publication"]["pubmed_id"] == "56789012"


def test_pmids_are_stored_verbatim_as_text(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Contract: ``r.pubmed_id`` is read as text, so each record stores the cell
    verbatim, including the two-PMID cell ``"333;444"``, and the URL is built from it.
    """
    assert [sl[i]["publication"] for i in range(6)] == [
        _publication(p).model_dump()
        for p in ["12345678", "23456789", "34567890", "45678901", "56789012", "333;444"]
    ]
    assert sl[5]["publication"]["pubmed_url"] == (
        "https://pubmed.ncbi.nlm.nih.gov/333;444/"
    )


def test_sl_ledger_counts_every_row_as_a_record(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``dropped_records.json``: 10 source rows, 10 kept, 0 dropped, no rule, and the
    one row that names one gene on both sides counted for the verifier.
    """
    ledger = _ledger(sl)
    assert ledger == s.DropLog(
        dataset="SynthLethalityYeastSynthLethDbDataset",
        source_records=10,
        kept_records=10,
        dropped_records=0,
        rule=s.RETENTION_RULE,
        n_self_pairs=1,
        rules=[],
    )


def test_sl_side_files_single_reference_gene_set_and_ledger(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``preprocess/`` holds the ledger, gene set, reference index and manifest but no
    data.csv; one reference covers members [0..9]; the gene set is every released
    name, the primed ``IMP2'`` included.
    """
    assert sorted(os.listdir(sl.preprocess_dir)) == [
        "build_manifest.json",
        "dropped_records.json",
        "experiment_reference_index.json",
        "gene_set.json",
    ]
    assert sl.df is None
    with open(osp.join(sl.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == [
            "EFB1",
            "GHOST",
            "IMP2'",
            "SHARED",
            "SSA1",
            "STM1",
            "TEF5",
            "TFC3",
            "TSV115",
            "VPS8",
            "YAL005C",
            "YAL014C",
        ]
    with open(osp.join(sl.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [list(range(10))]
    with open(osp.join(sl.preprocess_dir, "build_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["loader_class"] == "SynthLethalityYeastSynthLethDbDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.synth_leth_db"
    assert sorted(os.listdir(sl.processed_dir)) == [
        "lmdb",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


# --------------------------------------------------------------------------- #
# Synthetic rescue build
# --------------------------------------------------------------------------- #


def test_sr_records_match_the_source_rows_and_nan_score_becomes_none(
    sr: s.SynthRescueYeastSynthLethDbDataset,
) -> None:
    """Record 0 (``TFC3,1001,VPS8,1002,0.42,11111111``): ``is_synthetic_rescue True``,
    reference ``False`` / None. The released 0.42 has no sourced definition (#903), so
    the score is ``None`` with a ``deferred_pending_source_review`` gap whose note
    carries 0.42, the source and the PMID verbatim. Record 1
    (``SSA1,1005,YAL014C,1014,,...``): the empty score is ``None`` with NO gap (the
    release states none). Row 2 (``EFB1`` twice, Entrez 1003 twice) is stored as
    released, so the store holds 3 records.
    """
    expected = SyntheticRescueExperiment(
        dataset_name="SynthRescueYeastSynthLethDbDataset",
        genotype=_pair(("TFC3", 1001), ("VPS8", 1002)),
        environment=_ENVIRONMENT,
        phenotype=SyntheticRescuePhenotype(
            is_synthetic_rescue=True,
            synthetic_rescue_statistic_score=None,
            provenance_gaps=[s.sr_score_gap(0.42, "BioGRID", "11111111")],
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
    assert len(sr) == 3
    assert sr[0]["experiment"] == expected.model_dump()
    assert sr[0]["reference"] == expected_reference.model_dump()
    assert sr[0]["publication"] == _publication("11111111").model_dump()
    (gap,) = sr[0]["experiment"]["phenotype"]["provenance_gaps"]
    assert gap["field"] == "synthetic_rescue_statistic_score"
    assert gap["reason"] == "deferred_pending_source_review"
    assert gap["resolve_with"]["source_uri"] == "https://doi.org/10.1093/nar/gky1079"
    assert gap["note"].startswith(
        "Yeast_SR.csv states r.statistic_score 0.42 (r.source BioGRID, r.pubmed_id "
        "11111111)"
    )
    second = sr.transform_item(sr[1])
    assert second["experiment"].phenotype.synthetic_rescue_statistic_score is None
    assert second["experiment"].phenotype.provenance_gaps == []
    assert second["experiment"].genotype.systematic_gene_names == ["SSA1", "YAL014C"]
    assert second["publication"] == _publication("22222222")
    assert sr.transform_item(sr[2])["experiment"].genotype.systematic_gene_names == [
        "EFB1",
        "EFB1",
    ]
    assert sr.raw_file_names == ["Yeast_SR.csv"]


def test_sr_ledger_side_files_and_gene_set(
    sr: s.SynthRescueYeastSynthLethDbDataset,
) -> None:
    """One reference with members [0, 1, 2]; gene set = the five released names; the
    ledger drops nothing and counts the one self pair.
    """
    with open(osp.join(sr.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == ["EFB1", "SSA1", "TFC3", "VPS8", "YAL014C"]
    with open(osp.join(sr.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [[0, 1, 2]]
    assert stored[0]["reference"]["experiment_reference_type"] == "synthetic rescue"
    ledger = _ledger(sr)
    assert ledger.dataset == "SynthRescueYeastSynthLethDbDataset"
    assert (ledger.kept_records, ledger.dropped_records, ledger.n_self_pairs) == (
        3,
        0,
        1,
    )
    assert ledger.rules == []


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

_CLASSES = [
    (s.SynthLethalityYeastSynthLethDbDataset, "Yeast_SL.csv"),
    (s.SynthRescueYeastSynthLethDbDataset, "Yeast_SR.csv"),
]


@pytest.mark.parametrize(("cls", "filename"), _CLASSES)
def test_a_repeated_released_pair_refuses_the_build(
    tmp_path: Path, cls: type[Any], filename: str
) -> None:
    """One (name, id) pair in two orders (rows 0 and 2) is one released pair twice:
    ``DuplicatePairError`` names the rows and the pair, and no record is written. Row
    3 names the same genes by other names and ids, which is another released pair.
    """
    root = tmp_path / "dup"
    _write_raw(
        root,
        filename,
        [
            "TFC3,1001,VPS8,1002,0.2,111\n",
            "TFC3,1001,EFB1,1003,0.2,111\n",
            "VPS8,1002,TFC3,1001,0.2,111\n",
            "TSV115,1001,FUN15,1002,0.2,111\n",
        ],
    )
    with pytest.raises(
        s.DuplicatePairError,
        match=r"1 released pair\(s\) repeat across rows \[0, 2\]: "
        r"\[\[\('TFC3', 1001\), \('VPS8', 1002\)\]\]",
    ):
        cls(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


@pytest.mark.parametrize(("cls", "filename"), _CLASSES)
def test_a_blank_pmid_refuses_the_build_by_name(
    tmp_path: Path, cls: type[Any], filename: str
) -> None:
    """Contract: an empty PMID in row 1 and a whitespace-only PMID (``" "``) in row 2
    are both blank, so ``BlankPubmedIdError`` names the file, the count 2 and the first
    blank row, before any LMDB is written.
    """
    root = tmp_path / "blank_pmid"
    _write_raw(
        root,
        filename,
        [
            "TFC3,1001,VPS8,1002,0.2,111\n",
            "SSA1,1005,VPS8,1002,0.5,\n",
            "TFC3,1001,EFB1,1003,0.3, \n",
        ],
    )
    with pytest.raises(s.BlankPubmedIdError) as excinfo:
        cls(root=str(root))
    assert str(excinfo.value) == (
        f"{root / 'raw' / filename}: 2 row(s) with a blank r.pubmed_id (first at row 1)"
    )
    assert not (root / "processed" / "lmdb").exists()


# --------------------------------------------------------------------------- #
# download, main, item retyping
# --------------------------------------------------------------------------- #


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


def test_main_builds_both_datasets_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Contract: ``main`` builds both datasets under
    ``$DATA_ROOT/data/torchcell/synth_{lethality,rescue}_yeast_synth_leth_db``, the
    directories the knowledge-graph configs read, with no genome; nothing is written
    under the working directory.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    sl_root = data_root / "data/torchcell/synth_lethality_yeast_synth_leth_db"
    sr_root = data_root / "data/torchcell/synth_rescue_yeast_synth_leth_db"
    _write_raw(sl_root, "Yeast_SL.csv", _SL_ROWS)
    _write_raw(sr_root, "Yeast_SR.csv", _SR_ROWS)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    s.main()
    lines = capsys.readouterr().out.splitlines()
    assert "SynthLethalityYeastSynthLethDbDataset(10)" in lines
    assert "SynthRescueYeastSynthLethDbDataset(3)" in lines
    assert (sl_root / "processed/lmdb").is_dir()
    assert (sr_root / "processed/lmdb").is_dir()
    assert os.listdir(cwd) == []


def test_lethality_items_retype_through_the_lethality_classes(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``transform_item`` rebuilds a stored lethality item as a
    ``SyntheticLethalityExperiment`` with its reference, dumping to exactly the stored
    dictionaries; a ``SyntheticRescueExperiment`` refuses the lethality item's
    experiment dictionary.
    """
    item = sl[0]
    typed = sl.transform_item(item)
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


# --------------------------------------------------------------------------- #
# The real pinned files (#1045)
# --------------------------------------------------------------------------- #


@pytest.mark.data
def test_real_files_store_every_row_with_the_self_pairs_counted(tmp_path: Path) -> None:
    """On the pinned ``Yeast_SL.csv`` / ``Yeast_SR.csv``: every released row is a
    record (14,000 SL, 6,948 SR); SL rows 3152 (PUS1), 5719 (TAF1) and 8663 (SBA1) and
    SR rows 370, 1182, 3138, 3154, 6616 and 6672 name one gene on both sides and are
    stored as released; ``YPR108W-A`` (Entrez 1466522, in no R64 annotation) is a
    record; no released pair repeats.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    for slug, filename, cls, expected in (
        (
            "syn_leth_db_yeast",
            s.SL_CSV_NAME,
            s.SynthLethalityYeastSynthLethDbDataset,
            14000,
        ),
        (
            "syn_rescue_db_yeast",
            s.SR_CSV_NAME,
            s.SynthRescueYeastSynthLethDbDataset,
            6948,
        ),
    ):
        source = osp.join(data_root, "data/torchcell", slug, "raw", filename)
        root = tmp_path / slug
        (root / "raw").mkdir(parents=True)
        (root / "raw" / filename).write_bytes(Path(source).read_bytes())
        dataset = cls(root=str(root))
        assert len(dataset) == expected
        ledger = _ledger(dataset)
        assert (ledger.kept_records, ledger.dropped_records) == (expected, 0)
        assert (
            ledger.n_self_pairs
            == {"syn_leth_db_yeast": 3, "syn_rescue_db_yeast": 6}[slug]
        )
        if slug == "syn_leth_db_yeast":
            names = {
                p["systematic_gene_name"]
                for p in dataset[5232]["experiment"]["genotype"]["perturbations"]
            }
            assert "YPR108W-A" in names
            sides = dataset[3152]["experiment"]["genotype"]["perturbations"]
            assert [p["systematic_gene_name"] for p in sides] == ["PUS1", "PUS1"]

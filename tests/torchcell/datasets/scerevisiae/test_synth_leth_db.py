# tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
# [[tests.torchcell.datasets.scerevisiae.test_synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py
"""SynLethDB yeast synthetic-lethality and synthetic-rescue loaders, built hermetically.

2026.10.02 (issue #597): each side of a row is resolved by its Entrez id
(``n1.identifier`` / ``n2.identifier``) through the pinned NCBI GFF, with the gene name as
a cross-check through ``SCerevisiaeGenome.resolve_gene_name`` (standard name before
alias, the prime kept). A row whose Entrez id is not in the GFF, whose name disagrees
with its id, or that names one gene on both sides is dropped under a named rule in
``preprocess/dropped_records.json``; a repeated unordered ORF pair refuses the build.

The genome is a duck-typed stub: ``genome_root`` (holding a fixture
``ncbi_genomic.gff`` at ``NCBI_GFF_RELPATH``), a ``feature_index`` in the shape the
real genome builds, and the REAL ``SCerevisiaeGenome.resolve_gene_name`` bound onto it,
so the precedence under test is the genome's own. The module pin ``NCBI_GFF_SHA256`` is
patched to the fixture GFF's digest for the module. The stub mirrors the two issue
cases: ``STM1`` is the standard name of YAL012W and an alias of the LATER gene YAL013W
(the real YLR150W / YPR163C), and ``IMP2`` is the standard name of YAL010C while
``IMP2'`` is an alias of YAL011W (the real YMR035W / YIL154C).

The raw CSV is written into ``<root>/raw/`` so ``download()`` is never called; the
build-time CSV pin is swapped for a presence recorder by ``tests/torchcell/conftest.py``.
Expected records are hand-built from the schema classes. Nothing touches ``$DATA_ROOT``
except the ``data``-marked test at the end, which resolves the real pinned files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from pydantic import ValidationError

from torchcell.data import ExperimentDataset, RawSha256MismatchError
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

#: The module's real GFF pin, read before the module fixture patches it.
_REAL_NCBI_GFF_SHA256 = s.NCBI_GFF_SHA256

#: ``(ORF, Entrez id, standard name or None, aliases)`` of the stub genome, in GFF order.
_GENES: list[tuple[str, int, str | None, list[str]]] = [
    ("YAL001C", 1001, "TFC3", ["TSV115", "FUN24"]),
    ("YAL002W", 1002, "VPS8", ["FUN15", "SHARED"]),
    ("YAL003W", 1003, "EFB1", ["TEF5", "SHARED"]),
    ("YAL005C", 1005, "SSA1", ["YG100"]),
    ("YAL010C", 1010, "IMP2", []),
    ("YAL011W", 1011, "IMP21", ["IMP2'", "IMP2"]),
    ("YAL012W", 1012, "STM1", ["MPT4"]),
    ("YAL013W", 1013, "TIF3", ["STM1"]),
    ("YAL014C", 1014, None, []),
]


def _gff_text() -> str:
    """A fixture GFF in the RefSeq shape: genes, a pseudogene, and features to skip."""
    lines = ["##gff-version 3", "#!annotation-source SGD R64-4-1"]
    for orf, entrez, standard, _ in _GENES:
        name = f";gene={standard}" if standard else ""
        lines.append(
            f"NC_001133.9\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene-{orf};"
            f"Dbxref=GeneID:{entrez}{name};locus_tag={orf}"
        )
        lines.append(
            f"NC_001133.9\tRefSeq\tmRNA\t1\t100\t.\t+\t.\tID=rna-{orf};"
            f"Dbxref=GeneID:{entrez + 50000};locus_tag=NOT_{orf}"
        )
    lines.append(
        "NC_001133.9\tRefSeq\tpseudogene\t1\t100\t.\t+\t.\tID=gene-YAL099W;"
        "Dbxref=GeneID:1099,SGD:S000000001;locus_tag=YAL099W"
    )
    return "\n".join(lines) + "\n"


def _feature_index() -> dict[str, Any]:
    """The four keys ``SCerevisiaeGenome.feature_index`` builds, from ``_GENES``."""
    standard: dict[str, list[str]] = {}
    alias: dict[str, list[str]] = {}
    for orf, _, name, aliases in _GENES:
        if name:
            standard.setdefault(name.upper(), []).append(orf)
        for a in aliases:
            alias.setdefault(a.upper(), []).append(orf)
    return {
        "genes": {orf for orf, *_ in _GENES},
        "locus_type": {},
        "standard_to_ids": standard,
        "alias_to_ids": alias,
    }


class _StubGenome:
    """``genome_root``, ``feature_index`` and the real ``resolve_gene_name``."""

    resolve_gene_name = SCerevisiaeGenome.resolve_gene_name

    def __init__(self, genome_root: str) -> None:
        self.genome_root = genome_root
        self.feature_index = _feature_index()


def _write_gff(genome_root: Path) -> str:
    path = genome_root / s.NCBI_GFF_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_gff_text(), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def genome_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A genome root holding the fixture GFF, with the module pin set to its digest."""
    root = tmp_path_factory.mktemp("genome")
    digest = _write_gff(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s, "NCBI_GFF_SHA256", digest)
        yield root


def _genome(root: Path) -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome(str(root)))


_HEADER = "n1.name,n1.identifier,n2.name,n2.identifier,r.statistic_score,r.pubmed_id\n"
_SL_ROWS = [
    "TFC3,1001,VPS8,1002,0.85,12345678\n",  # 0: standard names
    "TSV115,1001,EFB1,1003,0.5,23456789\n",  # 1: an alias
    "YAL005C,1005,IMP2',1011,0.1,34567890\n",  # 2: systematic name, primed alias
    "YAL014C,1014,TEF5,1003,0.3,45678901\n",  # 3: a name-less ORF
    "SSA1,1005,VPS8,1002,,56789012\n",  # 4: empty score
    "STM1,1012,EFB1,1003,0.2,333;444\n",  # 5: standard name shadowed by an alias
    "VPS8,1001,EFB1,1003,0.4,222\n",  # 6: name disagrees with its Entrez id
    "TFC3,1001,GHOST,9999,0.4,333\n",  # 7: Entrez id not in the GFF
    "TFC3,1001,TSV115,1001,0.7,444\n",  # 8: one gene on both sides
    "SHARED,1002,TFC3,1001,0.6,555\n",  # 9: an alias on two genes (ambiguous)
]
_SR_ROWS = [
    "TFC3,1001,VPS8,1002,0.42,11111111\n",
    "SSA1,1005,YAL014C,1014,,22222222\n",
    "EFB1,1003,EFB1,1003,,33333333\n",
]

_ENVIRONMENT = Environment(
    media=Media(name="YEPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30),
)
_GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _write_raw(root: Path, filename: str, rows: list[str]) -> None:
    (root / "raw").mkdir(parents=True)
    (root / "raw" / filename).write_text(_HEADER + "".join(rows), encoding="utf-8")


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
    tmp_path_factory: pytest.TempPathFactory, genome_root: Path
) -> s.SynthLethalityYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "sl"
    _write_raw(root, s.SL_CSV_NAME, _SL_ROWS)
    return s.SynthLethalityYeastSynthLethDbDataset(
        root=str(root), genome=_genome(genome_root)
    )


@pytest.fixture(scope="module")
def sr(
    tmp_path_factory: pytest.TempPathFactory, genome_root: Path
) -> s.SynthRescueYeastSynthLethDbDataset:
    root = tmp_path_factory.mktemp("synlethdb") / "sr"
    _write_raw(root, s.SR_CSV_NAME, _SR_ROWS)
    return s.SynthRescueYeastSynthLethDbDataset(
        root=str(root), genome=_genome(genome_root)
    )


def _pairs(dataset: ExperimentDataset) -> list[list[tuple[str, str]]]:
    return [
        [
            (p["systematic_gene_name"], p["perturbed_gene_name"])
            for p in dataset[i]["experiment"]["genotype"]["perturbations"]
        ]
        for i in range(len(dataset))
    ]


def _ledger(dataset: ExperimentDataset) -> s.DropLog:
    with open(osp.join(dataset.preprocess_dir, "dropped_records.json")) as f:
        return s.DropLog.model_validate_json(f.read())


# --------------------------------------------------------------------------- #
# Entrez map
# --------------------------------------------------------------------------- #


def test_entrez_map_reads_gene_and_pseudogene_locus_tags_only(
    genome_root: Path,
) -> None:
    """Every ``gene`` feature maps its ``GeneID`` to its ``locus_tag`` and the
    ``pseudogene`` (with a second, non-GeneID cross-reference) counts too; the
    ``mRNA`` features (GeneID + 50000, ``NOT_`` locus tags) contribute nothing.
    """
    mapping = s.load_entrez_to_orf(str(genome_root / s.NCBI_GFF_RELPATH))
    assert mapping == {entrez: orf for orf, entrez, *_ in _GENES} | {1099: "YAL099W"}


def test_entrez_map_refuses_bytes_off_the_pin(tmp_path: Path) -> None:
    """The GFF is hashed before it is read: other bytes raise
    ``RawSha256MismatchError`` naming the file and both digests.
    """
    path = tmp_path / "ncbi_genomic.gff"
    path.write_text(_gff_text() + "# edited\n", encoding="utf-8")
    with pytest.raises(RawSha256MismatchError, match=str(path)):
        s.load_entrez_to_orf(str(path))


def test_entrez_map_refuses_one_gene_id_on_two_locus_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``GeneID`` that names two different locus tags raises
    ``EntrezGeneIdConflictError`` naming the id and both tags.
    """
    path = tmp_path / "ncbi_genomic.gff"
    path.write_text(
        _gff_text() + "NC_001134.8\tRefSeq\tgene\t1\t9\t.\t+\t.\tID=gene-YBL001C;"
        "Dbxref=GeneID:1001;locus_tag=YBL001C\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        s, "NCBI_GFF_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(
        s.EntrezGeneIdConflictError, match="GeneID 1001 names both YAL001C and YBL001C"
    ):
        s.load_entrez_to_orf(str(path))


# --------------------------------------------------------------------------- #
# Synthetic lethality build
# --------------------------------------------------------------------------- #


def test_sl_keeps_six_rows_in_source_order_with_entrez_orfs(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Rows 0-5 are kept as records 0-5 with the ORF each Entrez id names: ``TSV115``
    (an alias) is YAL001C, ``IMP2'`` keeps its prime and is YAL011W (stripping it would
    give IMP2 = YAL010C), and ``STM1`` is YAL012W, the gene whose STANDARD name it is,
    not YAL013W, the later gene listing it as an alias (the old name map's last write).
    The perturbed name is the raw name (the schema spells the prime ``_prime``).
    """
    assert len(sl) == 6
    assert _pairs(sl) == [
        [("YAL001C", "TFC3"), ("YAL002W", "VPS8")],
        [("YAL001C", "TSV115"), ("YAL003W", "EFB1")],
        [("YAL005C", "YAL005C"), ("YAL011W", "IMP2_prime")],
        [("YAL003W", "TEF5"), ("YAL014C", "YAL014C")],
        [("YAL002W", "VPS8"), ("YAL005C", "SSA1")],
        [("YAL003W", "EFB1"), ("YAL012W", "STM1")],
    ]
    assert "genome" not in vars(sl)
    assert sl.raw_file_names == ["Yeast_SL.csv"]
    assert sl.processed_file_names == ["lmdb"]


def test_sl_common_name_pair_record_matches_the_source_row(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """Record 0 (``TFC3,1001,VPS8,1002,0.85,12345678``): two SGA KanMX deletions with
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
    record = sl[0]
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == _publication("12345678").model_dump()


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
    assert [sl[i]["publication"] for i in range(len(sl))] == [
        _publication(p).model_dump()
        for p in ["12345678", "23456789", "34567890", "45678901", "56789012", "333;444"]
    ]
    assert sl[5]["publication"]["pubmed_url"] == (
        "https://pubmed.ncbi.nlm.nih.gov/333;444/"
    )


def test_sl_ledger_names_every_dropped_row_and_its_rule(
    sl: s.SynthLethalityYeastSynthLethDbDataset, genome_root: Path
) -> None:
    """``dropped_records.json``: 10 source rows, 6 kept, 4 dropped. Row 6 (``VPS8`` on
    Entrez 1001, which is TFC3) and row 9 (``SHARED``, an alias of YAL002W and YAL003W,
    so ambiguous) fail the name cross-check; row 7 (Entrez 9999) is not in the GFF; row
    8 (``TFC3`` and its alias ``TSV115``, both Entrez 1001) is one gene on both sides.
    The ledger records the GFF it read and that file's pin.
    """
    ledger = _ledger(sl)
    assert (ledger.source_records, ledger.kept_records, ledger.dropped_records) == (
        10,
        6,
        4,
    )
    assert ledger.entrez_source_path == str(genome_root / s.NCBI_GFF_RELPATH)
    assert ledger.entrez_source_sha256 == s.NCBI_GFF_SHA256
    by_rule = {r.rule: r for r in ledger.rules}
    assert list(by_rule) == [
        "entrez_id_not_in_ncbi_gff",
        "gene_name_disagrees_with_entrez_id",
        "same_gene_on_both_sides",
    ]
    assert {k: [row.source_row for row in r.rows] for k, r in by_rule.items()} == {
        "entrez_id_not_in_ncbi_gff": [7],
        "gene_name_disagrees_with_entrez_id": [6, 9],
        "same_gene_on_both_sides": [8],
    }
    assert [r.n_records for r in ledger.rules] == [1, 2, 1]
    disagree = by_rule["gene_name_disagrees_with_entrez_id"].rows
    assert disagree[0].model_dump() == {
        "source_row": 6,
        "n1_name": "VPS8",
        "n1_entrez": 1001,
        "n2_name": "EFB1",
        "n2_entrez": 1003,
        "detail": (
            "n1 VPS8 (Entrez 1001) -> YAL001C by id, but the name resolves renamed "
            "to YAL002W (candidates []); n2 EFB1 (Entrez 1003) -> YAL003W"
        ),
    }
    assert disagree[1].detail.startswith(
        "n1 SHARED (Entrez 1002) -> YAL002W by id, but the name resolves ambiguous "
        "to None (candidates ['YAL002W', 'YAL003W'])"
    )
    assert by_rule["entrez_id_not_in_ncbi_gff"].rows[0].detail == (
        "n1 TFC3 (Entrez 1001) -> YAL001C; n2 GHOST (Entrez 9999): not in the GFF"
    )


def test_sl_side_files_single_reference_gene_set_and_ledger(
    sl: s.SynthLethalityYeastSynthLethDbDataset,
) -> None:
    """``preprocess/`` holds the ledger, gene set, reference index and manifest but no
    data.csv; one reference covers members [0..5]; the gene set is the seven kept ORFs
    (YAL010C, the unprimed IMP2, and YAL013W, the alias-only STM1, are absent).
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
            "YAL001C",
            "YAL002W",
            "YAL003W",
            "YAL005C",
            "YAL011W",
            "YAL012W",
            "YAL014C",
        ]
    with open(osp.join(sl.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [[0, 1, 2, 3, 4, 5]]
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
    """Record 0 (``TFC3,1001,VPS8,1002,0.42,11111111``): ``is_synthetic_rescue True``
    score 0.42, reference ``False`` / None. Record 1 (``SSA1,1005,YAL014C,1014,,...``):
    the empty score is ``None``. Row 2 (``EFB1`` twice, Entrez 1003 twice) is a self
    pair and is dropped, so the store holds 2 records.
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
    assert second["experiment"].genotype.systematic_gene_names == ["YAL005C", "YAL014C"]
    assert second["publication"] == _publication("22222222")
    assert sr.raw_file_names == ["Yeast_SR.csv"]


def test_sr_ledger_side_files_and_gene_set(
    sr: s.SynthRescueYeastSynthLethDbDataset,
) -> None:
    """One reference with members [0, 1]; gene set = the four kept ORFs; the ledger
    drops source row 2 as a self pair and nothing else.
    """
    with open(osp.join(sr.preprocess_dir, "gene_set.json")) as f:
        assert json.load(f) == ["YAL001C", "YAL002W", "YAL005C", "YAL014C"]
    with open(osp.join(sr.preprocess_dir, "experiment_reference_index.json")) as f:
        stored = json.load(f)
    assert [item["member_indices"] for item in stored] == [[0, 1]]
    assert stored[0]["reference"]["experiment_reference_type"] == "synthetic rescue"
    ledger = _ledger(sr)
    assert ledger.dataset == "SynthRescueYeastSynthLethDbDataset"
    assert [(r.rule, [x.source_row for x in r.rows]) for r in ledger.rules] == [
        ("entrez_id_not_in_ncbi_gff", []),
        ("gene_name_disagrees_with_entrez_id", []),
        ("same_gene_on_both_sides", [2]),
    ]


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

_CLASSES = [
    (s.SynthLethalityYeastSynthLethDbDataset, "Yeast_SL.csv"),
    (s.SynthRescueYeastSynthLethDbDataset, "Yeast_SR.csv"),
]


@pytest.mark.parametrize(("cls", "filename"), _CLASSES)
def test_a_repeated_orf_pair_refuses_the_build(
    tmp_path: Path, genome_root: Path, cls: type[Any], filename: str
) -> None:
    """One pair in two orders (rows 0 and 2) and the same pair named by an alias (row
    3) resolve to one unordered ORF pair: ``DuplicateOrfPairError`` names the rows and
    the pair, and no record is written.
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
        s.DuplicateOrfPairError,
        match=r"1 ORF pair\(s\) repeat across kept rows \[0, 2, 3\]: "
        r"\[\['YAL001C', 'YAL002W'\]\]",
    ):
        cls(root=str(root), genome=_genome(genome_root))
    assert not (root / "processed" / "lmdb").exists()


@pytest.mark.parametrize(("cls", "filename"), _CLASSES)
def test_a_build_without_the_genome_is_refused(
    tmp_path: Path, cls: type[Any], filename: str
) -> None:
    """The name cross-check and the GFF location both come from the genome, so a build
    started with ``genome=None`` raises ``MissingGenomeError`` before reading the CSV.
    """
    root = tmp_path / "nogenome"
    _write_raw(root, filename, ["TFC3,1001,VPS8,1002,0.2,111\n"])
    with pytest.raises(s.MissingGenomeError):
        cls(root=str(root), genome=None)


@pytest.mark.parametrize(("cls", "filename"), _CLASSES)
def test_a_blank_pmid_refuses_the_build_by_name(
    tmp_path: Path, genome_root: Path, cls: type[Any], filename: str
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
        cls(root=str(root), genome=_genome(genome_root))
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    genome_root: Path,
) -> None:
    """Contract: ``main`` reads the genome from ``$DATA_ROOT`` and builds both datasets
    under ``$DATA_ROOT/data/torchcell/synth_{lethality,rescue}_yeast_synth_leth_db``,
    the directories the knowledge-graph configs read; the GFF is read from that
    genome's root; nothing is written under the working directory.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    _write_gff(data_root / "data/sgd/genome")
    genome_kwargs: list[dict[str, Any]] = []

    class _MainGenome(_StubGenome):
        def __init__(self, **kwargs: Any) -> None:
            genome_kwargs.append(kwargs)
            super().__init__(kwargs["genome_root"])

    monkeypatch.setattr(s, "SCerevisiaeGenome", _MainGenome)
    sl_root = data_root / "data/torchcell/synth_lethality_yeast_synth_leth_db"
    sr_root = data_root / "data/torchcell/synth_rescue_yeast_synth_leth_db"
    _write_raw(sl_root, "Yeast_SL.csv", _SL_ROWS)
    _write_raw(sr_root, "Yeast_SR.csv", _SR_ROWS)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
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
    assert "SynthLethalityYeastSynthLethDbDataset(6)" in lines
    assert "SynthRescueYeastSynthLethDbDataset(2)" in lines
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
# The real pinned files (issue #597)
# --------------------------------------------------------------------------- #


@pytest.mark.data
def test_real_files_resolve_by_entrez_with_the_issue_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the pinned ``Yeast_SL.csv`` / ``Yeast_SR.csv`` with the genome at
    ``$DATA_ROOT/data/sgd/genome`` and its pinned ``ncbi_genomic.gff``: no name
    disagrees with its Entrez id; SL drops source row 5232 (``YPR108W-A``, Entrez
    1466522, absent from the GFF) and the self pairs 3152 (PUS1), 5719 (TAF1) and 8663
    (SBA1); SR drops the self pairs 370, 1182, 3138, 3154, 6616 and 6672; no unordered
    ORF pair repeats among kept rows; and every name in the issue's table resolves to
    its Entrez ORF (``STM1`` -> YLR150W, ..., ``IMP2'`` -> YIL154C).
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    # The module fixture may still hold the fixture GFF's digest; the real file is
    # checked against the real pin.
    monkeypatch.setattr(s, "NCBI_GFF_SHA256", _REAL_NCBI_GFF_SHA256)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    entrez = s.load_entrez_to_orf(s.ncbi_gff_path(genome))
    expected_drops = {
        "synth_lethality_yeast_synth_leth_db": (
            s.SL_CSV_NAME,
            {
                "entrez_id_not_in_ncbi_gff": [5232],
                "same_gene_on_both_sides": [3152, 5719, 8663],
            },
        ),
        "synth_rescue_yeast_synth_leth_db": (
            s.SR_CSV_NAME,
            {"same_gene_on_both_sides": [370, 1182, 3138, 3154, 6616, 6672]},
        ),
    }
    issue_table = {
        "STM1": "YLR150W",
        "SDC1": "YDR469W",
        "MFT1": "YML062C",
        "NSP1": "YJL041W",
        "RPL37A": "YLR185W",
        "CCS1": "YMR038C",
        "YPK1": "YKL126W",
        "TAF1": "YGR274C",
        "SSL2": "YIL143C",
        "HAP1": "YLR256W",
        "IMP2'": "YIL154C",
    }
    seen: dict[str, set[str]] = {}
    for slug, (csv_name, drops) in expected_drops.items():
        df = s._read_synlethdb_csv(
            osp.join(data_root, "data/torchcell", slug, "raw", csv_name)
        )
        resolved = s.resolve_pairs(df, genome, entrez)
        got = {
            str(rule): sorted(int(i) for i in group.index)
            for rule, group in resolved.groupby("drop_rule")
        }
        assert got == drops, slug
        kept = resolved[resolved["drop_rule"].isna()]
        s.refuse_duplicate_pairs(kept)
        for side in ("n1", "n2"):
            for name, orf in zip(
                kept[f"{side}.name"], kept[f"{side}.systematic_name"], strict=True
            ):
                if name in issue_table:
                    seen.setdefault(name, set()).add(orf)
    assert seen == {name: {orf} for name, orf in issue_table.items()}

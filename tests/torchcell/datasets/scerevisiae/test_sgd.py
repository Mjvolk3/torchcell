# tests/torchcell/datasets/scerevisiae/test_sgd.py
# [[tests.torchcell.datasets.scerevisiae.test_sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_sgd.py
"""Hermetic build of the SGD gene-essentiality loader on a fixture dump and a fake Entrez.

The loader reads SGD's per-gene API dump (``$DATA_ROOT/data/sgd/genome/genes/*.json``)
and refuses bytes off its digest pin, so each build here writes a small dump into
``tmp_path`` and sets the module pin to that dump's digest. ``Entrez.efetch`` and
``Entrez.read`` are replaced on ``Bio.Entrez`` itself (the module object the loader
holds), so ``get_publication_info`` runs for real against canned records.

Fixture dump (2026.10.11, #901: allele rule, carried conditions, typed gaps):
    YAL001C  TFC3: one inviable / null / S288C ``tfc3-Δ`` (PubMed 111) -> record; a
             viable, a "reduction of function" and a W303 entry -> skipped
    YBR001C  no phenotype_details -> nothing
    YCR001W  two ``ycr001w-Δ`` annotations (PubMed 222 and 333) -> two records
    YDR001C  ATG9: ``atg9-Δ`` under "nitrogen starvation" (PubMed 444) -> record with a
             nitrogen-source edit; ``atg9-K12A`` and ``atg9Δ10-20`` (PubMed 444, same
             condition) -> ledgered, never stored
    YEL001C  ``yel001c-∆`` spelled with U+2206, under "elevated temperature, 37 °C"
             (PubMed 111) -> record at 37 C

Canned Entrez: 111 has the DOI ``10.1/aaa`` in ELocationID; 222 has a ``pii`` there and the
DOI ``10.2/bbb`` in ArticleIdList; 333 and 444 have no DOI.

History: the 2024 loader walked ``SCerevisiaeGraph.G_raw`` (a pickle built from the same
dump), stored every annotation as a whole-gene SGA KanMX deletion on a guessed YEPD
solid medium at 30 C, and downloaded the dump when fewer than 100 files were present.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from Bio import Entrez

from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatementRole,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    NamedAllelePerturbation,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import sgd as m
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason


class _Elem(str):
    """A string carrying Entrez-style ``.attributes`` (as ``Bio.Entrez`` StringElement)."""

    attributes: dict[str, str]

    def __new__(cls, value: str, attributes: dict[str, str]) -> _Elem:
        obj = super().__new__(cls, value)
        obj.attributes = attributes
        return obj


def _article(elocation: list[_Elem], article_ids: list[_Elem]) -> dict[str, Any]:
    return {
        "PubmedArticle": [
            {
                "MedlineCitation": {"Article": {"ELocationID": elocation}},
                "PubmedData": {"ArticleIdList": article_ids},
            }
        ]
    }


_RECORDS: dict[str, dict[str, Any]] = {
    "111": _article([_Elem("10.1/aaa", {"EIdType": "doi"})], []),
    "222": _article(
        [_Elem("S0000-1", {"EIdType": "pii"})],
        [_Elem("22222", {"IdType": "pubmed"}), _Elem("10.2/bbb", {"IdType": "doi"})],
    ),
    "333": _article([], [_Elem("33333", {"IdType": "pubmed"})]),
    "444": _article([], [_Elem("44444", {"IdType": "pubmed"})]),
}


class _Handle:
    def __init__(self, pubmed_id: str) -> None:
        self.pubmed_id = pubmed_id

    def close(self) -> None:
        pass


def _install_entrez(monkeypatch: pytest.MonkeyPatch, fail_first: int = 0) -> list[str]:
    """Patch efetch/read on the module Entrez; the first ``fail_first`` calls raise."""
    fetched: list[str] = []
    state = {"failures": fail_first}

    def efetch(**kwargs: Any) -> _Handle:
        assert kwargs == {
            "db": "pubmed",
            "id": kwargs["id"],
            "rettype": "xml",
            "retmode": "text",
        }
        if state["failures"] > 0:
            state["failures"] -= 1
            raise RuntimeError("boom")
        fetched.append(kwargs["id"])
        return _Handle(kwargs["id"])

    def read(handle: _Handle) -> dict[str, Any]:
        return _RECORDS[handle.pubmed_id]

    monkeypatch.setattr(Entrez, "efetch", efetch)
    monkeypatch.setattr(Entrez, "read", read)
    return fetched


_ANNOTATION_ID = iter(range(1000, 2000))


def _annotation(
    pubmed_id: str,
    locus: str,
    allele: str,
    *,
    conditions: list[dict[str, Any]] | None = None,
    note: str | None = None,
    mutant_type: str = "null",
    strain: str = "S288C",
    phenotype: str = "inviable",
) -> dict[str, Any]:
    """One ``phenotype_details`` entry in the shape SGD's locus API returns it."""
    return {
        "id": next(_ANNOTATION_ID),
        "mutant_type": mutant_type,
        "locus": {"display_name": locus},
        "experiment": {"display_name": "classical genetics"},
        "strain": {"display_name": strain},
        "properties": [
            {
                "class_type": "BIOITEM",
                "bioitem": {"display_name": allele},
                "note": "",
                "role": "Allele",
            },
            *(conditions or []),
        ],
        "note": note,
        "phenotype": {"display_name": phenotype},
        "reference": {"pubmed_id": int(pubmed_id)},
    }


_NITROGEN = {"class_type": "media", "note": "nitrogen starvation", "unit": ""}
_HOT = {
    "class_type": "temperature",
    "note": "elevated temperature, 37 °C",
    "unit": "°C",
}

_DUMP: dict[str, list[dict[str, Any]] | None] = {
    "YAL001C": [
        _annotation("111", "TFC3", "tfc3-Δ"),
        _annotation("111", "TFC3", "tfc3-Δ", phenotype="viable"),
        _annotation("111", "TFC3", "tfc3-Δ", mutant_type="reduction of function"),
        _annotation("111", "TFC3", "tfc3-Δ", strain="W303"),
    ],
    "YBR001C": None,
    "YCR001W": [
        _annotation("222", "YCR001W", "ycr001w-Δ"),
        _annotation("333", "YCR001W", "ycr001w-Δ"),
    ],
    "YDR001C": [
        _annotation("444", "ATG9", "atg9-Δ", conditions=[_NITROGEN]),
        _annotation("444", "ATG9", "atg9-K12A", conditions=[_NITROGEN]),
        _annotation("444", "ATG9", "atg9Δ10-20", conditions=[_NITROGEN], note="n"),
    ],
    "YEL001C": [_annotation("111", "YEL001C", "yel001c-∆", conditions=[_HOT])],
}


def _write_dump(data_root: Path, dump: dict[str, list[dict[str, Any]] | None]) -> Path:
    genes = data_root / "data" / "sgd" / "genome" / "genes"
    genes.mkdir(parents=True)
    for orf, details in dump.items():
        (genes / f"{orf}.json").write_text(
            json.dumps({"locus": {"format_name": orf}, "phenotype_details": details}),
            encoding="utf-8",
        )
    (genes / "README.txt").write_text("not a gene file", encoding="utf-8")
    return genes


def _pin(monkeypatch: pytest.MonkeyPatch, genes: Path) -> None:
    monkeypatch.setattr(m, "SGD_GENES_DIGEST", m.sgd_genes_digest(str(genes)))


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dump: dict[str, list[dict[str, Any]] | None],
) -> m.GeneEssentialitySgdDataset:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _pin(monkeypatch, _write_dump(data_root, dump))
    _install_entrez(monkeypatch)
    return m.GeneEssentialitySgdDataset(root=str(tmp_path / "gene_essentiality_sgd"))


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.GeneEssentialitySgdDataset:
    return _build(tmp_path, monkeypatch, _DUMP)


_GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")
_PIN = "1cb4ed59556b1534e28eb79ad21135cdb5a149f78dca8df4924f473e39eef409"


def _unstated(silence: str) -> Media:
    """The gapped medium a record must store, spelled out here rather than read from
    the module: name, state and class each a ``not_carried_by_curation`` gap looked for
    in the pinned dump, the note quoting what the annotation carries instead.
    """
    note = (
        "the SGD phenotype annotation states a phenotype ('inviable', mutant_type "
        "'null', strain 'S288C') and an allele ('<gene>-Δ') but no growth medium: "
        + silence
        + "; the medium, if stated, is in the annotation's own reference "
        "(Publication.pubmed_id)"
    )
    return Media(
        name=None,
        state=None,
        is_synthetic=None,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=m.SGD_DUMP,
                note=note,
            )
            for field in ("name", "state", "is_synthetic")
        ],
    )


_NO_CONDITION_ENVIRONMENT = Environment(
    media=_unstated("phenotype_details[].properties carries no condition property"),
    temperature=None,
    provenance_gaps=[m.SGD_TEMPERATURE_GAP],
)
_NITROGEN_ENVIRONMENT = Environment(
    media=_unstated(
        "its one condition property (class_type 'media') reads 'nitrogen "
        "starvation', a nutrient edit to a base medium it does not name"
    ),
    temperature=None,
    perturbations=[
        EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.nitrogen_source, description="nitrogen starvation"
        )
    ],
    provenance_gaps=[m.SGD_TEMPERATURE_GAP],
)
_HOT_ENVIRONMENT = Environment(
    media=_unstated(
        "its one condition property is a temperature ('elevated temperature, "
        "37 °C', class_type 'temperature'), which names no medium"
    ),
    temperature=Temperature(value=37.0),
)


def _experiment(
    gene: str, locus: str, environment: Environment, allele: str | None = None
) -> dict[str, Any]:
    perturbation: SgaKanMxDeletionPerturbation | NamedAllelePerturbation
    if allele is None:
        perturbation = SgaKanMxDeletionPerturbation(
            systematic_gene_name=gene, perturbed_gene_name=locus, strain_id="S288C"
        )
    else:
        perturbation = NamedAllelePerturbation(
            systematic_gene_name=gene,
            perturbed_gene_name=locus,
            allele_name=allele,
            strain_id="S288C",
        )
    return GeneEssentialityExperiment(
        dataset_name="GeneEssentialitySgdDataset",
        genotype=Genotype(perturbations=[perturbation]),
        environment=environment,
        phenotype=GeneEssentialityPhenotype(
            is_essential=True, evidence=m.sgd_evidence("classical genetics")
        ),
    ).model_dump()


def _reference(environment: Environment) -> dict[str, Any]:
    return GeneEssentialityExperimentReference(
        dataset_name="GeneEssentialitySgdDataset",
        genome_reference=_GENOME,
        environment_reference=environment,
        phenotype_reference=GeneEssentialityPhenotype(
            is_essential=False, evidence=m.sgd_evidence()
        ),
    ).model_dump()


def test_one_record_per_annotation_in_file_order(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """Seven records, files in sorted order and annotations in SGD's order: TFC3 once
    (its other three entries fail the null / S288C / inviable filter), YCR001W twice,
    ATG9 three times (the whole-gene deletion and, since #1045, its two named alleles
    ``atg9-K12A`` and ``atg9Δ10-20`` on ``NamedAllelePerturbation``) and YEL001C once.
    Each record's environment is the one its condition property states; the reference
    is viable in that same environment.
    """
    assert len(dataset) == 7
    expected: list[tuple[str, str, Environment, str | None]] = [
        ("YAL001C", "TFC3", _NO_CONDITION_ENVIRONMENT, None),
        ("YCR001W", "YCR001W", _NO_CONDITION_ENVIRONMENT, None),
        ("YCR001W", "YCR001W", _NO_CONDITION_ENVIRONMENT, None),
        ("YDR001C", "ATG9", _NITROGEN_ENVIRONMENT, None),
        ("YDR001C", "ATG9", _NITROGEN_ENVIRONMENT, "atg9-K12A"),
        ("YDR001C", "ATG9", _NITROGEN_ENVIRONMENT, "atg9Δ10-20"),
        ("YEL001C", "YEL001C", _HOT_ENVIRONMENT, None),
    ]
    for index, (gene, locus, environment, allele) in enumerate(expected):
        assert dataset[index]["experiment"] == _experiment(
            gene, locus, environment, allele
        )
        assert dataset[index]["reference"] == _reference(environment)
    named = dataset[4]["experiment"]["genotype"]["perturbations"][0]
    assert (named["perturbation_type"], named["allele_name"], named["strain_id"]) == (
        "named_allele",
        "atg9-K12A",
        "S288C",
    )
    assert (named["mechanism_so_id"], named["mechanism_so_name"]) == (
        "SO:0001060",
        "sequence_variant",
    )
    assert [dataset[i]["publication"] for i in (0, 1, 2, 3, 6)] == [
        Publication(
            pubmed_id="111",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/111/",
            doi="10.1/aaa",
            doi_url="https://doi.org/10.1/aaa",
        ).model_dump(),
        Publication(
            pubmed_id="222",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/222/",
            doi="10.2/bbb",
            doi_url="https://doi.org/10.2/bbb",
        ).model_dump(),
        *(
            Publication(
                pubmed_id=pmid,
                pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=None,
                doi_url=None,
            ).model_dump()
            for pmid in ("333", "444")
        ),
        Publication(
            pubmed_id="111",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/111/",
            doi="10.1/aaa",
            doi_url="https://doi.org/10.1/aaa",
        ).model_dump(),
    ]


def test_no_record_carries_a_guessed_medium_or_temperature(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """The old loader stored YEPD solid at 30 C on every record. Now every record
    stores a medium whose name, state and class are ``not_carried_by_curation`` gaps
    (anchored to the pinned dump), and the temperature is a gap except where SGD states
    37 C.
    """
    for index in range(len(dataset)):
        environment = dataset[index]["experiment"]["environment"]
        media = environment["media"]
        assert (media["name"], media["state"], media["is_synthetic"]) == (
            None,
            None,
            None,
        )
        media_gaps = {gap["field"]: gap for gap in media["provenance_gaps"]}
        assert set(media_gaps) == {"name", "state", "is_synthetic"}
        for gap in media_gaps.values():
            assert gap["reason"] == "not_carried_by_curation"
            assert gap["looked_in"]["sha256"] == _PIN
        gaps = {gap["field"]: gap for gap in environment["provenance_gaps"]}
        if index == 6:
            assert environment["temperature"] == {"value": 37.0, "unit": "Celsius"}
            assert "temperature" not in gaps
        else:
            assert environment["temperature"] is None
            assert gaps["temperature"]["reason"] == "not_carried_by_curation"


def test_the_ledger_accounts_for_every_annotation_and_drops_none(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """Seven annotations, seven records: five whole-gene deletions and two named
    alleles; no rule drops a row (#1045).
    """
    ledger = m.SgdDropLog.model_validate_json(
        (Path(dataset.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert ledger == m.SgdDropLog(
        dataset="GeneEssentialitySgdDataset",
        source_annotations=7,
        kept_records=7,
        dropped_records=0,
        n_whole_gene_deletion=5,
        n_named_allele=2,
        sgd_genes_digest=m.SGD_GENES_DIGEST,
        rule="none",
        description=m._RULE_DESCRIPTION,
        rows=[],
    )


def test_side_files(dataset: m.GeneEssentialitySgdDataset) -> None:
    """No raw files, no ``data.csv``; the gene set is the four genes with records and
    each distinct environment has its own reference.
    """
    assert dataset.raw_file_names == []
    preprocess = Path(dataset.root) / "preprocess"
    assert not (preprocess / "data.csv").exists()
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YCR001W",
        "YDR001C",
        "YEL001C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert sorted(entry["member_indices"] for entry in index) == [
        [0, 1, 2],
        [3, 4, 5],
        [6],
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "gene_essentiality_sgd"
    assert manifest["loader_class"] == "GeneEssentialitySgdDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.sgd"
    assert {
        "GeneEssentialityExperiment",
        "SgaKanMxDeletionPerturbation",
        "NamedAllelePerturbation",
        "Environment",
    } <= set(manifest["closure"])


@pytest.mark.parametrize(
    ("allele", "locus", "whole"),
    [
        ("atg8-Δ", "ATG8", True),
        ("yil102c-A-∆", "YIL102C-A", True),
        ("ATG8-Δ", "ATG8", True),
        ("sac1Δ114-503", "SAC1", False),
        ("cdc24p-∆CH", "CDC24", False),
        ("cdc42-111", "CDC42", False),
        ("Nog1-(1-426)", "NOG1", False),
        ("rsc3-C15G", "RSC30", False),
        ("atg8-Δ", "ATG80", False),
    ],
)
def test_whole_gene_deletion_is_sgds_gene_delta_name(
    allele: str, locus: str, whole: bool
) -> None:
    """``<gene>-Δ`` with either delta code point and any case; nothing else."""
    assert m.is_whole_gene_deletion(allele, locus) is whole


def test_every_condition_of_the_pinned_dump_maps_to_its_own_environment() -> None:
    """The four verbatim pairs map to four distinct environments: nitrogen starvation
    is a qualitative nitrogen-source edit, 2% glucose a carbon-source edit of D-glucose
    at a basis-free 2 percent, the 37 C pair the temperature, each on a medium whose
    labels are gaps; the stateless YPD is its stated name on the library YPD base with
    only the state a gap that quotes it.
    """
    conditions = m.SGD_CONDITIONS
    assert set(conditions) == {
        ("media", "nitrogen starvation"),
        ("media", "2% glucose"),
        ("media", "YPD with 2% glucose"),
        ("temperature", "elevated temperature, 37 °C"),
    }
    assert m.SGD_ENVIRONMENT == _NO_CONDITION_ENVIRONMENT
    assert conditions[("media", "nitrogen starvation")] == _NITROGEN_ENVIRONMENT
    assert conditions[("temperature", "elevated temperature, 37 °C")] == (
        _HOT_ENVIRONMENT
    )
    assert conditions[("media", "2% glucose")].media == _unstated(
        "its one condition property (class_type 'media') reads '2% glucose', a "
        "carbon-source edit to a base medium it does not name"
    )
    (glucose,) = conditions[("media", "2% glucose")].perturbations
    assert isinstance(glucose, EnvironmentPhysicalPerturbation)
    assert glucose.factor is PhysicalFactor.carbon_source
    assert glucose.magnitude == Concentration(value=2.0, unit=ConcentrationUnit.percent)
    assert glucose.agent is not None
    assert glucose.agent.inchikey == "WQZGKKKJIJFFOK-GASJEMHNSA-N"
    ypd = conditions[("media", "YPD with 2% glucose")]
    assert ypd.perturbations == []
    assert ypd.media is m.SGD_STATELESS_YPD
    assert (
        ypd.media.name,
        ypd.media.state,
        ypd.media.is_synthetic,
        ypd.media.base_medium,
    ) == ("YPD with 2% glucose", None, False, "YPD")
    assert ypd.media.gapped_fields() == {"state"}
    (state_gap,) = ypd.media.provenance_gaps
    assert state_gap.reason is ProvenanceGapReason.not_carried_by_curation
    assert state_gap.looked_in == m.SGD_DUMP
    assert "'YPD with 2% glucose'" in str(state_gap.note)
    assert [g.field for g in ypd.provenance_gaps] == ["temperature"]
    dumps = {
        json.dumps(env.model_dump(), sort_keys=True) for env in conditions.values()
    }
    assert (
        len(dumps | {json.dumps(m.SGD_ENVIRONMENT.model_dump(), sort_keys=True)}) == 5
    )
    assert m.SGD_UNSTATED_MEDIUM.gapped_fields() == {"name", "state", "is_synthetic"}


def test_an_unknown_condition_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A condition pair outside the closed vocabulary is named, never guessed."""
    dump = dict(_DUMP)
    dump["YFL001W"] = [
        _annotation(
            "111",
            "DEG1",
            "deg1-Δ",
            conditions=[{"class_type": "media", "note": "YPGal", "unit": ""}],
        )
    ]
    with pytest.raises(m.SgdConditionError, match=r"YFL001W .*'YPGal'"):
        _build(tmp_path, monkeypatch, dump)
    assert not (tmp_path / "gene_essentiality_sgd" / "processed" / "lmdb").exists()


def test_two_kept_annotations_of_one_record_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-gene allele twice for one gene, condition and paper is one fact twice;
    two alleles of one gene (``cdc42-111`` beside ``cdc42-Δ``) are two records.
    """
    dump = dict(_DUMP)
    dump["YCR001W"] = [
        _annotation("222", "YCR001W", "ycr001w-Δ"),
        _annotation("222", "YCR001W", "ycr001w-Δ"),
    ]
    with pytest.raises(m.DuplicateSgdAnnotationError, match="YCR001W"):
        _build(tmp_path, monkeypatch, dump)
    m.refuse_duplicate_annotations(
        [
            m.SgdAnnotation(
                sgd_annotation_id=1,
                systematic_name="YLR229C",
                locus_display_name="CDC42",
                allele=allele,
                pubmed_id="1",
                experiment="classical genetics",
                conditions=(),
                note=None,
            )
            for allele in ("cdc42-Δ", "cdc42-111")
        ]
    )


def test_an_annotation_without_one_allele_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = _annotation("111", "TFC3", "tfc3-Δ")
    detail["properties"] = []
    with pytest.raises(
        m.SgdAlleleError, match=r"YAL001C.json annotation \d+: 0 alleles"
    ):
        _build(tmp_path, monkeypatch, {"YAL001C": [detail]})


def test_bytes_off_the_pin_and_a_missing_dump_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest covers every ``.json`` name and its bytes; a non-JSON file does not
    count. Editing one file, or building with no dump, raises ``SgdDumpDigestError``
    before any Entrez call or store write.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    fetched = _install_entrez(monkeypatch)
    with pytest.raises(m.SgdDumpDigestError, match="does not exist"):
        m.GeneEssentialitySgdDataset(root=str(tmp_path / "a"))
    genes = _write_dump(data_root, _DUMP)
    _pin(monkeypatch, genes)
    (genes / "README.txt").write_text("edited", encoding="utf-8")
    assert m.sgd_genes_digest(str(genes)) == m.SGD_GENES_DIGEST
    (genes / "YBR001C.json").write_text("{}", encoding="utf-8")
    with pytest.raises(m.SgdDumpDigestError, match="is not the pinned"):
        m.GeneEssentialitySgdDataset(root=str(tmp_path / "b"))
    assert fetched == []
    assert not (tmp_path / "b" / "processed" / "lmdb").exists()


def test_the_digest_is_sha256_over_sorted_name_and_file_sha256_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "B.json").write_bytes(b"b")
    (tmp_path / "A.json").write_bytes(b"a")
    lines = [
        f"A.json {hashlib.sha256(b'a').hexdigest()}",
        f"B.json {hashlib.sha256(b'b').hexdigest()}",
    ]
    assert (
        m.sgd_genes_digest(str(tmp_path))
        == hashlib.sha256("\n".join(lines).encode()).hexdigest()
    )


def test_get_publication_info_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two failing fetches then success: sleeps 1 s and 2 s (``base_delay * 2**attempt``
    with the random jitter pinned to 0), prints one retry line per failure, and returns
    the parsed record.
    """
    fetched = _install_entrez(monkeypatch, fail_first=2)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    assert m.get_publication_info("111") == {
        "pubmed_id": "111",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "doi": "10.1/aaa",
        "doi_url": "https://doi.org/10.1/aaa",
    }
    assert sleeps == [1, 2]
    assert fetched == ["111"]
    assert capsys.readouterr().out == (
        "Attempt 1 failed. Retrying in 1.00 seconds...\n"
        "Attempt 2 failed. Retrying in 2.00 seconds...\n"
    )


def test_get_publication_info_gives_up_after_five_attempts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Five failures: four sleeps (1, 2, 4, 8 s), a final error line, and None."""
    _install_entrez(monkeypatch, fail_first=5)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    assert m.get_publication_info("999") is None
    assert sleeps == [1, 2, 4, 8]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Error fetching info for PubMed ID 999: boom"
    )


def test_create_experiment_raises_when_publication_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "get_publication_info", lambda pubmed_id: None)
    annotation = m.SgdAnnotation(
        sgd_annotation_id=1,
        systematic_name="YAL001C",
        locus_display_name="TFC3",
        allele="tfc3-Δ",
        pubmed_id="999",
        experiment="classical genetics",
        conditions=(),
        note=None,
    )
    with pytest.raises(
        ValueError,
        match="Unable to retrieve publication information for PubMed ID: 999",
    ):
        m.GeneEssentialitySgdDataset.create_experiment(
            "GeneEssentialitySgdDataset", annotation, m.SGD_ENVIRONMENT
        )


def test_stored_items_retype_through_the_essentiality_classes(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes: a
    ``GeneEssentialityExperiment`` whose phenotype is essential, a
    ``GeneEssentialityExperimentReference`` whose phenotype is not, and a ``Publication``,
    each dumping back to exactly the stored dictionary, gapped medium included.
    """
    for index in range(len(dataset)):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is GeneEssentialityExperiment
        assert type(typed["reference"]) is GeneEssentialityExperimentReference
        assert type(typed["publication"]) is Publication
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == item["publication"]
        assert typed["experiment"].phenotype.is_essential is True
        assert typed["reference"].phenotype_reference.is_essential is False
        assert typed["experiment"].environment.media.gapped_fields() == {
            "name",
            "state",
            "is_synthetic",
        }


def test_publication_is_outside_the_content_hash_so_repeat_annotations_collapse(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """The KG node id of an experiment is the sha256 of its JSON dump, and the stored
    experiment carries no publication. The two YCR001W records differ only in their
    publication (PubMed 222 and 333), so they map to one node id; the three ATG9
    records (the deletion and two named alleles) and the 37 C record are their own
    nodes.
    """
    ids = [
        hashlib.sha256(
            json.dumps(
                GeneEssentialityExperiment(**dataset[i]["experiment"]).model_dump()
            ).encode("utf-8")
        ).hexdigest()
        for i in range(len(dataset))
    ]
    assert ids[1] == ids[2]
    assert len(set(ids)) == 6


def test_unset_data_root_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``DATA_ROOT`` unset the build refuses with the named ``data_root`` error
    instead of reading a directory relative to the working directory.
    """
    monkeypatch.delenv("DATA_ROOT")
    monkeypatch.chdir(tmp_path)
    _write_dump(tmp_path / "data", _DUMP)
    _install_entrez(monkeypatch)
    root = tmp_path / "gene_essentiality_sgd"
    with pytest.raises(ValueError) as info:
        m.GeneEssentialitySgdDataset(root=str(root))
    assert str(info.value) == (
        "DATA_ROOT environment variable is not set. Please set it in your .env file."
    )
    assert not (root / "processed" / "lmdb").exists()


def test_no_raw_stage(dataset: m.GeneEssentialitySgdDataset) -> None:
    """The loader has no raw files: ``download`` writes nothing into ``raw/`` and
    ``preprocess_raw`` hands back the very frame it was given, unchanged.
    """
    raw = Path(dataset.raw_dir)
    before = sorted(p.name for p in raw.iterdir()) if raw.exists() else []
    dataset.download()
    after = sorted(p.name for p in raw.iterdir()) if raw.exists() else []
    assert after == before
    frame = pd.DataFrame({"gene": ["YAL001C"], "is_essential": [True]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame


def test_main_builds_the_dataset_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` roots the dataset at ``$DATA_ROOT/data/torchcell/gene_essentiality_sgd``
    and prints it; no genome or graph is built, since the source is the dump itself.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[dict[str, Any]] = []

    class _Dataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def __repr__(self) -> str:
            return "GeneEssentialitySgdDataset(5)"

    monkeypatch.setattr(m, "GeneEssentialitySgdDataset", _Dataset)
    m.main()
    assert calls == [{"root": f"{tmp_path}/data/torchcell/gene_essentiality_sgd"}]
    assert capsys.readouterr().out == "GeneEssentialitySgdDataset(5)\n"


@pytest.mark.data
def test_the_pinned_dump_holds_the_issue_counts() -> None:
    """On the real ``$DATA_ROOT/data/sgd/genome/genes`` (#901, #1045): the digest is the
    pin; 1,329 inviable null S288C annotations; 1,287 name the whole-gene deletion and
    42 another allele, every one a record; no (gene, allele, condition, PubMed id) key
    repeats; 21 whole-gene deletions carry a condition (18 nitrogen starvation, one
    each of the other three pairs).
    """
    import os
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    genes_dir = osp.join(os.environ["DATA_ROOT"], m.SGD_GENES_RELPATH)
    m.verify_sgd_dump(genes_dir)
    annotations = m.inviable_null_annotations(genes_dir)
    kept, dropped = m.split_by_allele(annotations)
    assert (len(annotations), len(kept), len(dropped)) == (1329, 1287, 42)
    m.refuse_duplicate_annotations(annotations)
    assert {type(m.sgd_perturbation(a)).__name__ for a in dropped} == {
        "NamedAllelePerturbation"
    }
    assert Counter(a.conditions for a in kept if a.conditions) == {
        (("media", "nitrogen starvation"),): 18,
        (("media", "2% glucose"),): 1,
        (("media", "YPD with 2% glucose"),): 1,
        (("temperature", "elevated temperature, 37 °C"),): 1,
    }
    assert Counter(m.condition_environment(a).media.name for a in kept) == {
        None: 1286,
        "YPD with 2% glucose": 1,
    }


# --------------------------------------------------------------------------- #
# #924: the curated-annotation evidence
# --------------------------------------------------------------------------- #
def test_sgd_evidence_is_the_curated_null_phenotype_with_its_experiment_type() -> None:
    """The annotation fields a call reads and its definition, anchored to SGD's own
    description of a phenotype annotation (Engel 2022), the experiment-type definition,
    and the annotation's experiment type on an experiment only. The loader's own allele
    rule (#901) is not SGD's, so the evidence does not carry it.
    """
    evidence = m.sgd_evidence("systematic mutation set")
    assert evidence.evidence_type is EssentialityEvidenceType.curated_null_phenotype
    assert evidence.evidence_class is EssentialityEvidenceClass.curated_annotation
    assert evidence.curated_experiment_type == "systematic mutation set"
    assert evidence.scoring_rule is m.SGD_SCORING_RULE
    assert evidence.scoring_rule.value == {
        "observable": "inviable",
        "mutant_type": "null",
        "strain_background": "S288C",
    }
    assert evidence.call_definition is m.SGD_CALL_DEFINITION
    assert evidence.statements_of(EssentialityStatementRole.caveat) == []
    assert evidence.statements_of(EssentialityStatementRole.statistic_definition) == [
        m.SGD_EXPERIMENT_TYPES
    ]
    assert {sv.provenance.citation_key for sv in m.SOURCED_VALUES.values()} == {
        "engelNewDataCollaborations2022"
    }
    assert m.sgd_evidence().curated_experiment_type is None


def test_a_stored_record_carries_its_annotations_experiment_type(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """Every fixture annotation is 'classical genetics'; the reference states none."""
    for index in range(len(dataset)):
        item = dataset[index]
        evidence = item["experiment"]["phenotype"]["evidence"]
        assert evidence["curated_experiment_type"] == "classical genetics"
        assert evidence["evidence_type"] == "curated_null_phenotype"
        reference = item["reference"]["phenotype_reference"]["evidence"]
        assert reference["curated_experiment_type"] is None


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim_in_the_literature_mirror(key: str) -> None:
    import os
    from pathlib import Path as _Path

    from dotenv import load_dotenv

    from torchcell.verification.sourced import audit_sourced_value

    load_dotenv()
    library = _Path(os.environ["DATA_ROOT"]) / "torchcell-library"
    result = audit_sourced_value(m.SOURCED_VALUES[key], library)
    assert result.passed, result.message


@pytest.mark.data
def test_the_pinned_dump_splits_the_kept_annotations_by_experiment_type() -> None:
    """Of the 1,287 stored annotations, 1,121 are 'systematic mutation set' and 166
    'classical genetics' (measured 2026-10-12 on the pinned dump).
    """
    import os
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    genes_dir = osp.join(os.environ["DATA_ROOT"], m.SGD_GENES_RELPATH)
    kept, _ = m.split_by_allele(m.inviable_null_annotations(genes_dir))
    assert Counter(a.experiment for a in kept) == {
        "systematic mutation set": 1121,
        "classical genetics": 166,
    }

# tests/torchcell/verification/test_gene_essentiality.py
# [[tests.torchcell.verification.test_gene_essentiality]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_gene_essentiality.py
"""The gene-essentiality verifier on hand-built, schema-valid records.

Fixture: three single-gene deletions (YAL001C from PMID 1, YBR085W from PMID 2, YAL001C
again from PMID 3) on ``SC`` at 30 C, each ``is_essential`` True against a viable
reference. A correct store emits, in order: structural, reference_structural, count,
single_gene, one_record_per_gene_and_publication, essential_label,
released_annotations, reference_viable, then the caller's extra rows and the six
shared rules (and the two L4 gene rules with a universe).

Derived expectations: the released multiset {(YAL001C, 1), (YBR085W, 2), (YAL001C, 3)}
matches the fixture; adding a fourth record identical to the first gives one key with
two records (one extra), and makes the stored multiset one (YAL001C, 1) larger than the
release.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from tests.torchcell.datamodels._essentiality_evidence import curated_evidence
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Environment,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.gene_essentiality import (
    essential_gene_set,
    essential_without_viable_deletion_result,
    evidence_result,
    released_annotations_result,
    states_no_condition,
    verify_gene_essentiality_dataset,
)
from torchcell.verification.report import Level, LevelResult, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="cherrySGD1998")
ROWS = [("YAL001C", "1"), ("YBR085W", "2"), ("YAL001C", "3")]
RELEASED: Counter[tuple[str, str]] = Counter(ROWS)
FAMILY_NAMES = [
    "structural",
    "reference_structural",
    "count",
    "single_gene",
    "one_record_per_gene_and_publication",
    "essential_label",
    "released_annotations",
    "reference_viable",
    "evidence",
]
SHARED_NAMES = [
    "provenance_gaps",
    "canonical_gene_names",
    "uncertainty_sanity",
    "compound_identity",
    "media_compound_identity",
    "media_membership",
]


def _record(
    genes: list[str],
    pubmed_id: str,
    *,
    essential: bool = True,
    reference_essential: bool = False,
    reference_temperature: float = 30.0,
) -> dict[str, Any]:
    environment = Environment(media=SC, temperature=Temperature(value=30.0))
    experiment = GeneEssentialityExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    strain_id="S288C",
                )
                for gene in genes
            ]
        ),
        environment=environment,
        phenotype=GeneEssentialityPhenotype(
            is_essential=essential, evidence=curated_evidence("classical genetics")
        ),
    )
    reference = GeneEssentialityExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="S288C",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=Environment(
            media=SC, temperature=Temperature(value=reference_temperature)
        ),
        phenotype_reference=GeneEssentialityPhenotype(
            is_essential=reference_essential, evidence=curated_evidence()
        ),
    )
    return {
        "experiment": experiment.model_dump(),
        "reference": reference.model_dump(),
        "publication": {"pubmed_id": pubmed_id},
    }


def _records() -> list[dict[str, Any]]:
    return [_record([gene], pubmed) for gene, pubmed in ROWS]


def _verify(records: list[dict[str, Any]], **kwargs: Any) -> Any:
    arguments: dict[str, Any] = {
        "dataset_name": "ess",
        "provenance": PROV,
        "expected_count": len(records),
        "released": RELEASED,
        "pinned_digest": "d",
        "observed_digest": "d",
    }
    arguments.update(kwargs)
    return verify_gene_essentiality_dataset(records, **arguments)


def _result(report: Any, name: str) -> LevelResult:
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    result: LevelResult = matches[0]
    return result


def test_a_correct_store_emits_every_row_in_order_and_passes() -> None:
    report = _verify(_records())
    assert [r.name for r in report.results] == FAMILY_NAMES + SHARED_NAMES
    assert report.passed is True
    assert _result(report, "released_annotations").message == (
        "the 3 stored (gene, PubMed id) records are the 3 inviable null S288C "
        "annotations SGD releases"
    )
    assert _result(report, "one_record_per_gene_and_publication").message == (
        "3 (gene, environment, publication) keys, one record each"
    )


def test_a_record_identical_down_to_the_publication_is_a_duplicate_and_unreleased() -> (
    None
):
    records = [*_records(), _record(["YAL001C"], "1")]
    report = _verify(records)
    duplicate = _result(report, "one_record_per_gene_and_publication")
    assert duplicate.passed is False
    assert duplicate.details["n_duplicated"] == 1
    assert duplicate.details["n_extra_records"] == 1
    assert duplicate.details["examples"] == ["YAL001C PMID 1"]
    released = _result(report, "released_annotations")
    assert released.passed is False
    assert released.details["only_stored"] == ["('YAL001C', '1')"]
    assert released.details["only_released"] == []


def test_a_multi_gene_record_a_false_label_and_a_bad_reference_fail_their_rows() -> (
    None
):
    records = [
        _record(["YAL001C", "YBR085W"], "1"),
        _record(["YBR085W"], "2", essential=False),
        _record(["YAL001C"], "3", reference_essential=True, reference_temperature=26.0),
    ]
    report = _verify(records, released=None, observed_digest="other")
    assert _result(report, "single_gene").details == {
        "n_bad": 1,
        "examples": [{"index": 0, "genes": ["YAL001C", "YBR085W"]}],
    }
    assert _result(report, "essential_label").details == {
        "n_bad": 1,
        "example_indices": [1],
    }
    assert _result(report, "reference_viable").details == {
        "problems": {
            "reference is_essential is not False": 1,
            "reference environment differs": 1,
        }
    }
    drift = _result(report, "released_annotations")
    assert drift.passed is False
    assert drift.message == (
        "the SGD per-gene JSON release drifted from its pin: not compared"
    )
    assert drift.details["observed_digest"] == "other"


def test_released_annotations_report_both_directions() -> None:
    result = released_annotations_result(
        Counter({("A", "1"): 1}),
        Counter({("B", "2"): 2}),
        pinned_digest="p",
        observed_digest="p",
    )
    assert result.passed is False
    assert result.message == (
        "1 stored records have no SGD annotation and 2 SGD annotations have no record"
    )


def test_extra_rows_and_gene_rules_are_placed_after_the_family_rows() -> None:
    extra = LevelResult(level=Level.L4, name="cross", passed=True, message="m")
    report = _verify(
        _records(), extra_results=[extra], sgd_genes={"YAL001C", "YBR085W"}
    )
    assert [r.name for r in report.results] == (
        FAMILY_NAMES
        + ["cross"]
        + SHARED_NAMES
        + ["gene_containment_sgd", "current_genome_genes"]
    )
    assert report.passed is True


def test_an_empty_store_fails_the_rows_that_need_a_record() -> None:
    report = _verify([], expected_count=0, released=Counter())
    failed = {r.name for r in report.results if not r.passed}
    assert {
        "structural",
        "single_gene",
        "one_record_per_gene_and_publication",
        "essential_label",
        "reference_viable",
    } <= failed


def test_essential_genes_grown_as_viable_deletions_contradict() -> None:
    essential = essential_gene_set(_records())
    assert essential == {"YAL001C", "YBR085W"}
    clean = essential_without_viable_deletion_result(
        essential, {"YJR155W": "YJR155W_dma1"}, viable_store="smf"
    )
    assert clean.passed is True
    assert clean.name == "no_viable_deletion_in_smf"
    assert clean.message == (
        "none of the 2 essential genes is a viable full-deletion strain of smf"
    )
    contradicted = essential_without_viable_deletion_result(
        essential, {"YAL001C": "YAL001C_dma1"}, viable_store="smf"
    )
    assert contradicted.passed is False
    assert contradicted.level is Level.L4
    assert contradicted.message == (
        "1 of 2 essential genes are viable full-deletion strains of smf: "
        "YAL001C (YAL001C_dma1)"
    )
    assert (
        essential_without_viable_deletion_result(set(), {}, viable_store="smf").passed
        is False
    )


def test_essential_gene_set_skips_a_record_that_is_not_essential() -> None:
    records = [_record(["YAL001C"], "1"), _record(["YBR085W"], "2", essential=False)]
    assert essential_gene_set(records) == {"YAL001C"}


# --------------------------------------------------------------------------- #
# #901: the loader's drop ledger and SGD's condition qualifiers
# --------------------------------------------------------------------------- #
def test_released_annotations_take_the_ledgered_annotations_out_of_the_release() -> (
    None
):
    """The release has YAL001C / PubMed 1 twice (two alleles); the loader stores one and
    ledgers the other, so the store matches the release less the ledger.
    """
    released = RELEASED + Counter({("YAL001C", "1"): 1})
    ledgered = Counter({("YAL001C", "1"): 1})
    report = _verify(_records(), released=released, ledgered=ledgered)
    result = _result(report, "released_annotations")
    assert result.passed is True
    assert result.message == (
        "the 3 stored (gene, PubMed id) records are the 4 inviable null S288C "
        "annotations SGD releases, less the 1 the loader ledgers"
    )
    assert result.details["n_ledgered"] == 1
    without_ledger = _result(
        _verify(_records(), released=released), "released_annotations"
    )
    assert without_ledger.passed is False
    assert without_ledger.message == (
        "0 stored records have no SGD annotation and 1 SGD annotations have no record"
    )


def test_a_ledgered_annotation_sgd_does_not_release_fails_the_rule() -> None:
    result = released_annotations_result(
        Counter(ROWS),
        RELEASED,
        pinned_digest="p",
        observed_digest="p",
        ledgered=Counter({("YJR155W", "9"): 1}),
    )
    assert result.passed is False
    assert result.message == "1 ledgered annotations are not in the SGD release"
    assert result.details["unreleased_drops"] == ["('YJR155W', '9')"]


def test_only_records_stating_no_condition_are_unconditionally_essential() -> None:
    """The fixture's records state SC at 30 C, so none is unconditional; a record whose
    medium names nothing (its name a typed gap), with no temperature and no edit, is.
    A stated medium name, a temperature or an edit each make a record conditional.
    """
    unnamed = {"name": None, "state": None, "is_synthetic": None}
    assert essential_gene_set(_records(), unconditional_only=True) == set()
    bare = _record(["YJR155W"], "9")
    bare["experiment"]["environment"] = {
        "media": unnamed,
        "temperature": None,
        "perturbations": [],
    }
    assert essential_gene_set([*_records(), bare], unconditional_only=True) == {
        "YJR155W"
    }
    assert states_no_condition(bare["experiment"]["environment"]) is True
    assert (
        states_no_condition(
            {"media": unnamed, "temperature": {"value": 37.0}, "perturbations": []}
        )
        is False
    )
    assert (
        states_no_condition(
            {
                "media": unnamed,
                "temperature": None,
                "perturbations": [{"perturbation_type": "environment_physical"}],
            }
        )
        is False
    )
    assert (
        states_no_condition(
            {
                "media": {"name": "YPD with 2% glucose", "state": None},
                "temperature": None,
                "perturbations": [],
            }
        )
        is False
    )


def test_the_scope_names_the_compared_set_in_the_message() -> None:
    result = essential_without_viable_deletion_result(
        {"YAL001C"},
        {"YAL001C": "YAL001C_sn1"},
        viable_store="smf",
        scope="unconditionally essential",
    )
    assert result.message == (
        "1 of 1 unconditionally essential genes are viable full-deletion strains of "
        "smf: YAL001C (YAL001C_sn1)"
    )
    assert result.details["scope"] == "unconditionally essential"


# --------------------------------------------------------------------------- #
# #924: the evidence rule
# --------------------------------------------------------------------------- #
def test_the_evidence_row_names_the_one_type_and_the_experiment_types() -> None:
    result = _result(_verify(_records()), "evidence")
    assert result.passed is True
    assert result.message == (
        "all 3 experiments and references carry curated_null_phenotype evidence"
    )
    assert result.details["type_pairs"] == {
        "curated_null_phenotype / curated_null_phenotype": 3
    }
    assert result.details["curated_experiment_types"] == {"classical genetics": 3}


def test_the_evidence_row_fails_on_mixed_types_or_misplaced_experiment_types() -> None:
    pairs = Counter(
        {
            ("curated_null_phenotype", "curated_null_phenotype"): 2,
            ("construction_outcome", "curated_null_phenotype"): 1,
        }
    )
    result = evidence_result(pairs, Counter({"x": 3}), Counter({None: 3}))
    assert result.passed is False
    assert result.details["problems"] == [
        "2 evidence-type pairs {('curated_null_phenotype', 'curated_null_phenotype'): "
        "2, ('construction_outcome', 'curated_null_phenotype'): 1}",
        "1 experiments of construction_outcome evidence have references of "
        "curated_null_phenotype evidence",
    ]
    one = Counter({("curated_null_phenotype", "curated_null_phenotype"): 2})
    result = evidence_result(one, Counter({None: 1, "x": 1}), Counter({"x": 2}))
    assert result.details["problems"] == [
        "1 curated experiments state no experiment type",
        "2 references state an experiment type",
    ]
    empty = evidence_result(Counter(), Counter(), Counter())
    assert empty.passed is False

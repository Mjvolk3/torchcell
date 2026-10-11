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
from torchcell.verification.gene_essentiality import (
    essential_gene_set,
    essential_without_viable_deletion_result,
    released_annotations_result,
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
        phenotype=GeneEssentialityPhenotype(is_essential=essential),
    )
    reference = GeneEssentialityExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=Environment(
            media=SC, temperature=Temperature(value=reference_temperature)
        ),
        phenotype_reference=GeneEssentialityPhenotype(is_essential=reference_essential),
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

# tests/torchcell/verification/test_finding_class.py
# [[tests.torchcell.verification.test_finding_class]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_finding_class.py
"""Genome-reference rules report, never gate (owner, 2026-10-12).

``torchcell.verification.report`` classifies every rule result by its NAME: a rule whose
subject is the genome reference (containment, name currency, canonical spellings,
locus resolution, marker alleles, MAT and cis positions, assembly and parent pins) is
``report``; every other rule is ``gate``. A store's verdict reads only the gate rules,
and the summary lists the genome rows under their own header. These tests pin the
registry, the verdict, the summary, the JSON round trip, and the shared rules' classes.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import torchcell
from torchcell.verification.common import shared_rule_results
from torchcell.verification.report import (
    DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES,
    GENOME_REFERENCE_NAME_VOCABULARY,
    GENOME_REFERENCE_RULE_PREFIXES,
    GENOME_REFERENCE_RULES,
    FindingClass,
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    finding_class_of,
)

PROVENANCE = Provenance(source_uri="test://finding-class")
REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent


def _result(name: str, passed: bool, level: Level = Level.L4) -> LevelResult:
    return LevelResult(level=level, name=name, passed=passed, message=f"{name} msg")


def _report(*results: LevelResult) -> VerificationReport:
    report = VerificationReport(dataset_name="toy", provenance=PROVENANCE)
    for result in results:
        report.add(result)
    return report


def test_the_registry_is_exactly_the_reclassified_rules() -> None:
    """The names and prefixes reclassified on 2026-10-12, pinned so a change is deliberate.

    ``construct_placement_states`` joined on 2026-10-15 with the construct store: it
    counts how many of a store's gene additions point at deposited construct sequence,
    which is the record against our sequence references rather than against its release.
    The three ``*_annotation`` rules joined the same day (lane R): each reads a stored or
    released identifier against OUR annotation of the host and says so in its own
    message, and each was a ``gate`` until then because its name was off this list.
    """
    assert sorted(GENOME_REFERENCE_RULES) == [
        "canonical_gene_names",
        "cis_concordance",
        "construct_placement_states",
        "current_genome_genes",
        "deleted_gene_containment_sgd",
        "deleted_symbols_are_one_contiguous_run_of_the_pinned_assembly",
        "feature_containment",
        "induced_gene_containment",
        "marker_reference",
        "mat_fixation",
        "mat_orientation",
        "pangenome_orf_containment",
        "parents_pinned",
        "perturbed_locus_in_bw25113_annotation",
        "promoter_gene_is_a_locus",
        "promoter_genes_in_the_host_gene_universe",
        "released_b_numbers_on_mg1655_annotation",
        "released_tags_on_kt2440_annotation",
        "s288c_gene_containment",
        "spacer_matches_the_released_window_on_the_pinned_sequence",
        "stored_loci_are_bw25113_loci_derived_from_the_released_b_number",
        "stored_protein_keys_are_loci_of_the_pinned_assembly",
        "stored_tags_are_loci",
        "stored_tags_are_loci_of_the_pinned_assembly",
        "stored_targets_are_loci_of_the_pinned_assembly",
        "variant_reference",
    ]
    assert GENOME_REFERENCE_RULE_PREFIXES == (
        "gene_containment_",
        "perturbed_gene_containment_",
        "protein_and_perturbed_locus_containment_",
        "assembly_pin",
        "no_viable_deletion_in_",
    )


@pytest.mark.parametrize(
    "name",
    [
        *sorted(GENOME_REFERENCE_RULES),
        # one rendered name per prefix family, as the verifiers and loaders write them
        "gene_containment_sgd",
        "gene_containment_assembly",
        "gene_containment_scmd_ohya2005",
        "gene_containment_mg1655_b_numbers",
        "gene_containment_kt2440_tested_and_perturbed_loci",
        "perturbed_gene_containment_assembly",
        "perturbed_gene_containment_bw25113_locus_tags",
        "protein_and_perturbed_locus_containment_assembly",
        "assembly_pin",
        "assembly_pin_resolves",
        "assembly_pin_is_mg1655_genbank_with_the_ncm3722_background",
        "no_viable_deletion_in_smf_costanzo2016",
    ],
)
def test_genome_reference_rules_are_report(name: str) -> None:
    assert finding_class_of(name) is FindingClass.report
    assert _result(name, False).finding_class is FindingClass.report


@pytest.mark.parametrize(
    "name",
    [
        # L0 structure, L1 counts and uniqueness, L2 value fidelity and uncertainty, L3
        # vocabulary / reference / compound / media membership, L4 against a release
        "structural",
        "count",
        "pair_uniqueness",
        "orf_uniqueness",
        "one_record_per_bw25113_locus",
        "provenance_gaps",
        "value_fidelity",
        "released_annotations",
        "uncertainty_sanity",
        "measurement_type_consistent",
        "compound_identity",
        "media_compound_identity",
        "media_membership",
        "environment_perturbed",
        # rules beside the genome rows that read no annotation stay gates
        "measured_genes_present",
        "fitness_companion_containment",
        "caudal_genotype_agreement",
        "guide_targets",
        "strain_background",
        "reference_genome_is_the_named_denominator_strain",
        "every_record_measures_the_same_kt2440_locus_set",
        "site_identifiers_deleted_loci",
    ],
)
def test_data_fidelity_rules_are_gate(name: str) -> None:
    assert finding_class_of(name) is FindingClass.gate
    assert _result(name, False).finding_class is FindingClass.gate


def test_the_verdict_reads_only_gate_rules() -> None:
    """A failing genome row leaves the store PASS and is listed as an open finding; a
    failing gate row fails it whatever the genome rows say.
    """
    structural = _result("structural", True, Level.L0)
    current = _result("current_genome_genes", False)
    containment = _result("gene_containment_sgd", True)
    passing = _report(structural, current, containment)
    assert passing.passed is True
    assert passing.gate_results == [structural]
    assert passing.findings == [current, containment]
    assert passing.open_findings == [current]
    assert [r.is_open_finding for r in passing.results] == [False, True, False]

    failing = _report(_result("count", False, Level.L1), containment)
    assert failing.passed is False
    assert failing.open_findings == []


def test_a_report_with_no_gate_rule_does_not_pass() -> None:
    """An empty report never passes, and neither does one that holds genome rows only:
    nothing about the data's fidelity to its release was checked.
    """
    assert _report().passed is False
    assert _report(_result("gene_containment_sgd", True)).passed is False


def test_summary_lists_the_findings_under_the_verdict() -> None:
    report = _report(
        _result("current_genome_genes", False),
        _result("structural", True, Level.L0),
        _result("uncertainty_sanity", True, Level.L2),
        _result("canonical_gene_names", True, Level.L1),
    )
    assert report.summary().split("\n") == [
        "toy: PASS (1 genome finding(s) reported, not gated)",
        "  [ok] L0 structural: structural msg",
        "  [ok] L2 uncertainty_sanity: uncertainty_sanity msg",
        "  genome-reference findings (reported, never gated):",
        "  [ok] L1 canonical_gene_names: canonical_gene_names msg",
        "  [rp] L4 current_genome_genes: current_genome_genes msg",
    ]
    gates_only = _report(_result("count", False, Level.L1))
    assert gates_only.summary().split("\n") == [
        "toy: FAIL",
        "  [XX] L1 count: count msg",
    ]


def test_the_class_is_serialized_and_round_trips() -> None:
    report = _report(
        _result("structural", True, Level.L0), _result("current_genome_genes", False)
    )
    dumped = json.loads(report.model_dump_json())
    assert [(r["name"], r["finding_class"]) for r in dumped["results"]] == [
        ("structural", "gate"),
        ("current_genome_genes", "report"),
    ]
    again = VerificationReport.model_validate_json(report.model_dump_json())
    assert again == report
    assert again.passed is True


def test_a_report_written_before_the_class_existed_still_loads() -> None:
    """A ``verification_report.json`` without ``finding_class`` (every report written
    before 2026-10-12) loads, and its rows are classified by name.
    """
    old = {
        "dataset_name": "old",
        "provenance": {"source_uri": "u"},
        "results": [
            {"level": 0, "name": "structural", "passed": True, "message": "m"},
            {
                "level": 4,
                "name": "current_genome_genes",
                "passed": False,
                "message": "m",
            },
        ],
    }
    report = VerificationReport.model_validate(old)
    assert [r.finding_class for r in report.results] == [
        FindingClass.gate,
        FindingClass.report,
    ]
    assert report.passed is True


def test_a_stored_class_that_contradicts_the_registry_is_refused() -> None:
    with pytest.raises(ValueError, match="rule 'structural' was written as report"):
        LevelResult.model_validate(
            {
                "level": 0,
                "name": "structural",
                "passed": True,
                "message": "m",
                "finding_class": "report",
            }
        )


def test_a_renamed_copy_is_reclassified_with_its_name() -> None:
    """``model_copy(update={"name": ...})`` skips validation; the class follows the name."""
    renamed = _result("gene_containment_assembly", False).model_copy(
        update={"name": "count"}
    )
    assert renamed.finding_class is FindingClass.gate


def test_the_shared_gene_rules_report_and_the_rest_gate() -> None:
    """Over one record on a gene the universe lacks, the shared rules' classes and
    outcomes: the three genome rows are ``report`` (two do not hold), every other
    shared row is a gate, and the report built from them passes.
    """
    record = {
        "experiment": {
            "genotype": {
                "perturbations": [
                    {
                        "systematic_gene_name": "YAL001C",
                        "perturbed_gene_name": "TFC3",
                        "perturbation_type": "deletion",
                    }
                ]
            },
            "phenotype": {"label_name": "fitness", "fitness": 1.0},
            "environment": {"media": {}, "perturbations": []},
        }
    }
    results = shared_rule_results([record], sgd_genes={"YBR085W"})
    assert [(r.name, r.finding_class.value, r.passed) for r in results] == [
        ("provenance_gaps", "gate", True),
        ("canonical_gene_names", "report", True),
        ("uncertainty_sanity", "gate", True),
        ("compound_identity", "gate", True),
        ("media_compound_identity", "gate", True),
        ("media_membership", "gate", False),
        ("gene_containment_sgd", "report", False),
        ("current_genome_genes", "report", False),
    ]
    report = _report(*[r for r in results if r.name != "media_membership"])
    assert report.passed is True
    assert [r.name for r in report.open_findings] == [
        "gene_containment_sgd",
        "current_genome_genes",
    ]


def _static_rule_names() -> dict[str, str]:
    """Every rule NAME a verification module or a loader writes as a literal.

    ``name=`` of a ``LevelResult(...)`` or of an ``l0_``..``l4_`` helper call, over
    ``torchcell/verification`` and ``torchcell/datasets``. Names built at runtime (an
    f-string naming a gene universe or a store) are the prefix families
    :data:`GENOME_REFERENCE_RULE_PREFIXES` covers and are not read here.
    """
    names: dict[str, str] = {}
    for root in ("verification", "datasets"):
        for path in sorted((REPO_ROOT / "torchcell" / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if called != "LevelResult" and not (
                    called is not None and re.fullmatch(r"l[0-4]_[a-z_]+", called)
                ):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "name" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        value = keyword.value.value
                        if isinstance(value, str):
                            names.setdefault(value, str(path.relative_to(REPO_ROOT)))
    return names


def test_the_registry_is_exhaustive_over_every_rule_name_the_fleet_emits() -> None:
    """A rule named like a genome-reference rule is classified ``report`` or exempted.

    The registry above is a list of names, so nothing in the schema derives it and the
    only way it can be wrong is by omission -- which is what happened: three L4 rules
    that read a stored identifier against our own annotation
    (``perturbed_locus_in_bw25113_annotation``, ``released_b_numbers_on_mg1655_annotation``,
    ``released_tags_on_kt2440_annotation``) said "a finding, never a gate" in their own
    messages and classified ``gate`` because nobody added their names. This test sweeps
    every literal rule name in the fleet for the genome-reference vocabulary and demands
    that each match be either classified ``report`` or listed in
    ``DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES`` with its reason, so the next such
    rule fails here instead of silently gating a store on our annotation.
    """
    emitted = _static_rule_names()
    assert len(emitted) > 200, "the sweep found almost no rule names; it is not reading"
    misclassified = {
        name: emitted[name]
        for name in emitted
        if any(word in name for word in GENOME_REFERENCE_NAME_VOCABULARY)
        and finding_class_of(name) is FindingClass.gate
        and name not in DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES
    }
    assert misclassified == {}
    # the exemptions are real rules, not stale names
    unknown = sorted(DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES - set(emitted))
    assert unknown == []

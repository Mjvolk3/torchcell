# tests/torchcell/verification/test_environment_response_verification.py
"""Unit tests for the WS15 environment-response verifier + the ProvenanceGap pass.

Closes a real coverage gap: no env-response verification test existed. Also exercises
the ProvenanceGap affordance end-to-end -- the Phenotype-level honesty invariants (L0)
and the informational L1 ``provenance_gaps`` census emitted by both the eager and
streaming verifiers (a documented gap is a PASS, and its deferred fields form a worklist).

The second half covers the rules shared by every dataset-family verifier
(``torchcell.verification.common``): compound identity, media membership, the carrier-wide
gap census, canonical gene names, uncertainty sanity, and the two L4 gene rules. The
fixtures here are therefore compliant records -- a shared ``MEDIA_LIBRARY`` medium and an
identified compound -- and each rule is failed on purpose by one test.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels.media import YP_GALACTOSE
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    DoseBasis,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.verification import Provenance, ProvenanceGap, ProvenanceGapReason
from torchcell.verification.environment_response import (
    _condition_signature,
    _study_key,
    environment_response_gene_set,
    verify_environment_response_dataset,
    verify_environment_response_dataset_streaming,
)
from torchcell.verification.report import Level

PROV = Provenance(
    source_uri="test://synthetic", citation_key="turcoGlobalAnalysisYeast2023"
)
GENES = ["YAL001C", "YBR085W", "YJR155W"]
SGD = set(GENES)
# Real InChIKeys: the compound identity rule wants a resolvable structure, and a made-up
# string would not pass the schema's 14-10-1 block validator either.
HYDROQUINONE = "QIGBRXMKCJKVMJ-UHFFFAOYSA-N"
QUININE = "LOUPRKONTZGTKE-WZBLMQSHSA-N"


def _compound(
    name: str = "hydroquinone", inchikey: str | None = HYDROQUINONE
) -> Compound:
    return Compound(name=name, inchikey=inchikey)


def _env(
    *, media: Media | None = None, compound: Compound | None = None
) -> Environment:
    return Environment(
        media=media if media is not None else YP_GALACTOSE,
        temperature=Temperature(value=30),
        perturbations=[
            SmallMoleculePerturbation(
                compound=compound if compound is not None else _compound(),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )


def _phenotype(
    value: float, *, gaps: list[ProvenanceGap] | None = None
) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        environment_response=value,
        units="log2(treatment/control)",
        provenance_gaps=gaps or [],
    )


def _record(
    gene: str,
    value: float,
    *,
    gaps: list[ProvenanceGap] | None = None,
    common_name: str | None = None,
    environment: Environment | None = None,
) -> dict[str, Any]:
    env = environment if environment is not None else _env()
    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=common_name or gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(value, gaps=gaps),
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(0.0),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]


# --- verifier core ----------------------------------------------------------- #
def test_good_dataset_passes_all_levels():
    report = verify_environment_response_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_duplicate_pair_fails_uniqueness():
    records = _good_records()
    records.append(_record("YAL001C", -1.2))  # same strain + condition -> duplicate
    report = verify_environment_response_dataset(
        records, dataset_name="dup", provenance=PROV, expected_count=4
    )
    u = [r for r in report.results if r.name == "pair_uniqueness"]
    assert u and not u[0].passed


def test_gene_set_helper():
    assert environment_response_gene_set(_good_records()) == SGD


# --- ProvenanceGap: schema-level honesty invariants (L0) --------------------- #
def test_gapped_field_must_be_none():
    """Cannot both store a value and declare it missing."""
    with pytest.raises(ValidationError):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=-1.0,
            n_samples=3,
            sample_unit=SampleUnit.biological_replicate,
            provenance_gaps=[
                ProvenanceGap(
                    field="n_samples",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_gap_field_must_exist():
    with pytest.raises(ValidationError):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=-1.0,
            provenance_gaps=[
                ProvenanceGap(
                    field="not_a_real_field",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_valid_gap_leaves_field_none():
    p = _phenotype(
        -1.0,
        gaps=[
            ProvenanceGap(
                field="n_samples",
                reason=ProvenanceGapReason.deferred_pending_source_review,
                looked_in=PROV,
            )
        ],
    )
    assert p.n_samples is None
    assert len(p.provenance_gaps) == 1


# --- ProvenanceGap: verifier L1 census pass (eager + streaming) -------------- #
def _gapped_records() -> list[dict[str, Any]]:
    """One record fully sourced, two with deferred n_samples gaps (the worklist)."""
    gap = [
        ProvenanceGap(
            field="n_samples",
            reason=ProvenanceGapReason.deferred_pending_source_review,
            looked_in=PROV,
        )
    ]
    return [
        _record(GENES[0], -1.2),
        _record(GENES[1], 0.8, gaps=gap),
        _record(GENES[2], -0.3, gaps=gap),
    ]


def test_gap_pass_is_emitted_and_passes_eager():
    report = verify_environment_response_dataset(
        _gapped_records(), dataset_name="gaps", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()  # documented gaps do NOT fail the build
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed
    assert g[0].details["n_gaps"] == 2
    assert g[0].details["n_records_with_gaps"] == 2
    assert g[0].details["worklist_fields"] == ["n_samples"]


def test_gap_pass_is_emitted_streaming():
    report = verify_environment_response_dataset_streaming(
        iter(_gapped_records()),
        dataset_name="gaps-stream",
        provenance=PROV,
        expected_count=3,
        sgd_genes=SGD,
    )
    assert report.passed, report.summary()
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed
    assert g[0].details["n_gaps"] == 2
    assert g[0].details["worklist_fields"] == ["n_samples"]


def test_no_gaps_reports_fully_sourced():
    report = verify_environment_response_dataset(
        _good_records(), dataset_name="clean", provenance=PROV, expected_count=3
    )
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed and g[0].details["n_gaps"] == 0


# --- Environment-level ProvenanceGap (temperature not carried by curation) ----- #
def test_environment_temperature_is_optional_and_gappable():
    """A curation layer (YeastPhenome) may not carry temperature -> typed absence."""
    env = Environment(
        media=Media(name="YPD", state="liquid", is_synthetic=False),
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature",
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=PROV,
            )
        ],
    )
    assert env.temperature is None
    assert env.provenance_gaps[0].field == "temperature"


def test_environment_gap_honesty_invariant():
    """A gapped temperature must be None -- cannot both set it and declare it missing."""
    with pytest.raises(ValidationError):
        Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=30),
            provenance_gaps=[
                ProvenanceGap(
                    field="temperature",
                    reason=ProvenanceGapReason.not_carried_by_curation,
                )
            ],
        )


def _temp_gapped_record(gene: str, value: float) -> dict[str, Any]:
    """A record whose environment carries a temperature ProvenanceGap (temp=None)."""
    env = Environment(
        media=YP_GALACTOSE,
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature",
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=PROV,
            )
        ],
        perturbations=[
            SmallMoleculePerturbation(
                compound=_compound("quinine", QUININE),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )
    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(value),
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(0.0),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def test_environment_gap_counted_in_census_eager_and_streaming():
    """The L1 census aggregates BOTH phenotype and environment gaps; a documented
    temperature gap passes and appears in by_field.
    """
    recs = [_temp_gapped_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]
    eager = verify_environment_response_dataset(
        recs, dataset_name="tgap", provenance=PROV, expected_count=3
    )
    stream = verify_environment_response_dataset_streaming(
        iter(recs),
        dataset_name="tgap-s",
        provenance=PROV,
        expected_count=3,
        sgd_genes=SGD,
    )
    for report in (eager, stream):
        assert report.passed, report.summary()  # documented env gap does NOT fail
        g = [r for r in report.results if r.name == "provenance_gaps"][0]
        assert g.passed
        assert g.details["by_field"]["temperature"] == 3


# --- shared rules: compound identity, media, names, uncertainty, L4 ---------- #
def _result(report: Any, name: str) -> Any:
    """The one level result with this name (a rule is emitted exactly once)."""
    matches = [r for r in report.results if r.name == name]
    assert len(matches) == 1, [r.name for r in report.results]
    return matches[0]


def _verify(records: list[dict[str, Any]], **kwargs: Any) -> Any:
    return verify_environment_response_dataset(
        records,
        dataset_name="rules",
        provenance=PROV,
        expected_count=len(records),
        **kwargs,
    )


def test_name_only_compound_fails_l3_with_record_counts():
    """A compound with no identifier and no typed gap cannot be encoded or joined."""
    records = [
        _record(g, v, environment=_env(compound=Compound(name="CMB2138")))
        for g, v in zip(GENES, [-1.2, 0.8, -0.3])
    ]
    result = _result(_verify(records), "compound_identity")
    assert not result.passed
    assert result.details["n_name_only"] == 3
    assert result.details["name_only_records"] == {"CMB2138 (perturbation.compound)": 3}


def test_gapped_compound_is_reported_but_does_not_fail():
    """An unresolvable compound whose absence is TYPED is honest, and stays visible."""
    compound = Compound(
        name="CMB2138",
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=PROV,
            )
        ],
    )
    records = [
        _record(g, v, environment=_env(compound=compound))
        for g, v in zip(GENES, [-1.2, 0.8, -0.3])
    ]
    result = _result(_verify(records), "compound_identity")
    assert result.passed
    assert result.details["n_gapped"] == 3
    assert result.details["gapped_records"] == {"CMB2138 (perturbation.compound)": 3}


def test_free_text_media_fails_l3_and_a_derived_medium_passes():
    """A medium must BE a library object or name a library key as its base."""
    free_text = Media(name="SynBase", state="liquid", is_synthetic=True)
    records = [
        _record(g, v, environment=_env(media=free_text))
        for g, v in zip(GENES, [-1.2, 0.8, -0.3])
    ]
    result = _result(_verify(records), "media_membership")
    assert not result.passed
    assert result.details["free_text_records"] == {"SynBase (base_medium=None)": 3}

    derived = Media(
        name="YPD pH 4.5", state="solid", is_synthetic=False, base_medium="YPD"
    )
    records = [
        _record(g, v, environment=_env(media=derived))
        for g, v in zip(GENES, [-1.2, 0.8, -0.3])
    ]
    result = _result(_verify(records), "media_membership")
    assert result.passed
    assert result.details["matched_media"] == {"YPD pH 4.5": "derived:YPD"}


def test_census_counts_undeclared_nones_on_every_carrier():
    """A None with no gap is counted per carrier field, so "fully sourced" is earned."""
    result = _result(_verify(_good_records()), "provenance_gaps")
    assert result.passed  # informational: a census never fails a build
    assert result.details["n_gaps"] == 0
    # the compound carriers are reached (they were invisible to the old census)
    assert result.details["silent_none_by_field"]["Compound.chebi_id"] > 0
    assert result.details["carriers_by_class"]["Compound"] > 0
    assert result.details["n_silent_none"] > 0
    assert "fully sourced" not in result.message


def test_two_spellings_of_one_gene_fail_canonical_names():
    """TOR1 and Tor1 in one release split the perturbation identity in the graph."""
    records = [
        _record("YJR066W", -1.2, common_name="TOR1"),
        _record(
            "YJR066W",
            0.8,
            common_name="Tor1",
            environment=_env(compound=_compound("quinine", QUININE)),
        ),
    ]
    result = _result(_verify(records), "canonical_gene_names")
    assert not result.passed
    assert result.details["split_spellings"] == {"YJR066W": ["TOR1", "Tor1"]}
    assert result.details["n_records_with_split_spelling"] == 2


class _Resolution:
    """Stand-in for GeneNameResolution (only status + systematic_name are read)."""

    def __init__(self, status: str, systematic_name: str | None) -> None:
        self.status = status
        self.systematic_name = systematic_name


def _resolver(name: str) -> _Resolution:
    table = {
        "YAL001C": _Resolution("current", "YAL001C"),
        "YBR085W": _Resolution("current", "YBR085W"),
        "YJR155W": _Resolution("current", "YJR155W"),
        "YFL012C": _Resolution("retired", "YFL012C"),
        "AAD15": _Resolution("renamed", "YOL165C"),
    }
    return table.get(name, _Resolution("retired", name))


def test_resolver_flags_a_retired_systematic_name():
    records = [_record("YFL012C", -1.2)]
    result = _result(
        _verify(records, resolve_gene_name=_resolver), "canonical_gene_names"
    )
    assert not result.passed
    assert result.details["not_current"] == ["YFL012C (retired -> YFL012C)"]
    assert _result(
        _verify(_good_records(), resolve_gene_name=_resolver), "canonical_gene_names"
    ).passed


def test_resolver_flags_a_common_name_that_resolves_elsewhere():
    records = [_record("YAL001C", -1.2, common_name="AAD15")]
    result = _result(
        _verify(records, resolve_gene_name=_resolver), "canonical_gene_names"
    )
    assert not result.passed
    assert result.details["common_name_mismatch"] == [
        "AAD15 -> YOL165C (stored YAL001C)"
    ]


def _uncertain_record(gene: str, uncertainty: float, n_samples: int) -> dict[str, Any]:
    env = _env()
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        environment_response=-1.0,
        environment_response_uncertainty=uncertainty,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=n_samples,
        sample_unit=SampleUnit.biological_replicate,
        units="log2(treatment/control)",
    )
    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=phenotype,
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(0.0),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def test_zero_sample_sd_fails_uncertainty_sanity():
    """SD == 0 across replicates is a pseudocount artifact, not perfect precision."""
    records = [_uncertain_record("YAL001C", 0.0, 3)]
    result = _result(_verify(records), "uncertainty_sanity")
    assert not result.passed
    assert result.details["n_zero_dispersion"] == 1

    result = _result(
        _verify([_uncertain_record("YAL001C", 0.4, 3)]), "uncertainty_sanity"
    )
    assert result.passed
    assert result.details["n_checked"] == 1


def test_replicates_without_an_uncertainty_are_reported():
    records = [_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]
    for record in records:
        record["experiment"]["phenotype"]["n_samples"] = 3
    result = _result(_verify(records), "uncertainty_sanity")
    assert result.passed  # reported, not failed
    assert result.details["n_no_uncertainty_with_replicates"] == 3


def _categorical_record(
    gene: str, category: str, reference_category: str | None
) -> dict[str, Any]:
    env = _env()

    def phenotype(call: str | None) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            category=ResponseCategory(call) if call is not None else None,
            units="spot-assay susceptibility call",
        )

    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=phenotype(category),
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=phenotype(reference_category),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def test_categorical_dataset_gets_the_categorical_reference_rule():
    """The numeric rule passes over zero values on a categorical set; this one does not."""
    records = [_categorical_record(g, "sensitive", "no_change") for g in GENES]
    result = _result(_verify(records), "reference_zero")
    assert result.passed
    assert result.details["rule"] == "categorical_baseline"
    assert result.details["reference_categories"] == {"no_change": 3}

    # a census screen scores every strain, so the baseline is also a measured call
    # (Smith 2006 grades wild type as the modal call); reported, never failed
    records = [_categorical_record(g, "sensitive", "sensitive") for g in GENES]
    result = _result(_verify(records), "reference_zero")
    assert result.passed
    assert result.details["baseline_used_as_measured_call"] == {"sensitive": 3}


def test_retired_orf_fails_l4_while_the_containment_floor_still_passes():
    """Mota's 0.993 containment hid seven records keyed to ORFs the genome dropped."""
    records = [
        _record("YAL001C", -1.2),
        _record("YBR085W", 0.8),
        _record("YFL012C", -0.3),  # not in the current gene set
    ]
    report = _verify(records, sgd_genes=SGD, min_containment=0.5)
    containment = _result(report, "gene_containment_sgd")
    assert containment.passed and containment.details["overlap"] == pytest.approx(2 / 3)
    strict = _result(report, "current_genome_genes")
    assert not strict.passed
    assert strict.details["missing_records"] == {"YFL012C": 1}
    assert strict.details["n_records"] == 1


def test_screen_id_joins_the_study_and_condition_keys():
    """Two screens of the same compound at the same dose are two measurements."""
    record = _record("YAL001C", -1.2)
    other = _record("YAL001C", -0.4)
    assert _study_key(record) == _study_key(other)
    assert _condition_signature(record["experiment"]) == _condition_signature(
        other["experiment"]
    )
    record["experiment"]["phenotype"]["screen_id"] = "screen-1"
    other["experiment"]["phenotype"]["screen_id"] = "screen-2"
    assert _study_key(record) != _study_key(other)
    assert _condition_signature(record["experiment"]) != _condition_signature(
        other["experiment"]
    )

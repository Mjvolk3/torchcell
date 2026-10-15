# tests/torchcell/datamodels/test_phosphosite_schema.py
# [[tests.torchcell.datamodels.test_phosphosite_schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_phosphosite_schema.py
"""The phosphosite-intensity leaf (#921): its key grammar, its refusals and its place.

The fixtures are released rows of Potel 2018 Supplementary Table 2: ``gnd`` S86 (one
protein), ``tufA;tufB`` H119 (two genes under the one accession ``P0CE47``) and
``pflB;tdcE;grcA`` S734/738/101 (three accessions). Every refusal names the field that
broke it, so a change to a validator's contract fails here by its message.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from pydantic import ValidationError

from torchcell.datamodels import schema as s
from torchcell.datamodels.pydant import ModelStrict
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason


def _site(**kw: Any) -> s.Phosphosite:
    fields: dict[str, Any] = dict(
        protein_ids=["P00350"],
        protein_id_namespace=s.ProteinIdNamespace.uniprotkb_accession,
        gene_names=["gnd"],
        positions=[86],
        residue=s.PhosphoResidue.serine,
        sequence_windows=["LLMVKAGAGTDAAIDSLKPYLDKGDIIIDGG"],
        localization_probability=0.999615,
        posterior_error_probability=4.27602e-05,
        identification_score=66.401,
        delta_score=58.487,
        diagnostic_peak=False,
    )
    fields.update(kw)
    return s.Phosphosite(**fields)


TUF = dict(
    protein_ids=["P0CE47"],
    gene_names=["tufA", "tufB"],
    positions=[119, 119],
    residue=s.PhosphoResidue.histidine,
    sequence_windows=["LVVAATDGPMPQTREHILLGRQVGVPYIIVF"],
    diagnostic_peak=True,
)
PFL = dict(
    protein_ids=["P09373", "P42632", "P68066"],
    gene_names=["pflB", "tdcE", "grcA"],
    positions=[734, 738, 101],
    sequence_windows=[
        "AIEHPEKYPNLTIRVSGYAVRFNALTREQQQ",
        "AMENPEKYPQLTIRVSGYAVRFNSLTKEQQQ",
        "AVKHPEKYPQLTIRVSGYAVRFNSLTPEQQR",
    ],
)


def _phenotype(**kw: Any) -> s.PhosphositeIntensityPhenotype:
    gnd, tuf = _site(), _site(**TUF)
    fields: dict[str, Any] = dict(
        phosphosite_intensity={gnd.key(): 2714000.0},
        quantification={
            gnd.key(): s.PhosphositeQuantification.quantified,
            tuf.key(): s.PhosphositeQuantification.not_quantified,
        },
        sites={gnd.key(): gnd, tuf.key(): tuf},
        measurement_type="fe3_imac_lc_msms_maxquant_summed_xic",
        sample_label="Intensity glucose_exp1",
        replicate=1,
        replicate_unit=s.SampleUnit.technical_replicate,
        harvest_growth_phase=s.CultureGrowthPhase.exponential,
        harvest_optical_density=0.6,
    )
    fields.update(kw)
    return s.PhosphositeIntensityPhenotype(**fields)


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.deferred_pending_source_review
    )


# --------------------------------------------------------------------- the key
def test_a_site_key_is_the_released_proteins_residue_and_positions() -> None:
    assert _site().key() == "P00350|S|86"
    assert _site(**TUF).key() == "P0CE47|H|119;119"
    assert _site(**PFL).key() == "P09373;P42632;P68066|S|734;738;101"


def test_only_histidine_is_the_n_phosphate() -> None:
    assert [r.value for r in s.PhosphoResidue if r.is_phosphohistidine] == ["H"]
    assert [r.value for r in s.PhosphoResidue] == ["S", "T", "Y", "H"]


# ---------------------------------------------------------- Phosphosite refusals
@pytest.mark.parametrize(
    "change,message",
    [
        (dict(protein_ids=[]), "names at least one protein id"),
        (dict(protein_ids=["gnd"]), "'gnd' is not a uniprotkb_accession id"),
        (dict(gene_names=[""]), "at least one non-empty gene name"),
        (
            dict(positions=[86, 87]),
            "2 positions for 1 gene names; the release pairs one position",
        ),
        (dict(positions=[0]), "positions are 1-based, got [0]"),
        (dict(sequence_windows=[]), "at least one sequence window"),
        (
            dict(sequence_windows=["LLMVKAGAGTDAAIDSLKPYLDKGDIIIDG"]),
            "no center residue",
        ),
        (
            dict(sequence_windows=["LLMVKAGAGTDAAIDTLKPYLDKGDIIIDGG"]),
            "does not center on S",
        ),
        (
            dict(localization_probability=1.2),
            "localization_probability is 1.2, not a probability",
        ),
        (
            dict(posterior_error_probability=float("nan")),
            "posterior_error_probability is nan, not a probability",
        ),
        (dict(identification_score=float("inf")), "identification_score must be"),
    ],
)
def test_a_site_refuses_what_the_release_could_not_have_written(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _site(**change)
    assert message in str(excinfo.value)


# ------------------------------------------------------------ phenotype contract
def test_a_released_zero_is_a_status_not_a_value() -> None:
    phenotype = _phenotype()
    assert phenotype.phosphosite_intensity == {"P00350|S|86": 2714000.0}
    assert phenotype.quantification["P0CE47|H|119;119"] == "not_quantified"
    assert "P0CE47|H|119;119" in phenotype.sites
    assert (phenotype.label_name, phenotype.label_statistic_name) == (
        "phosphosite_intensity",
        None,
    )


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(sites={}), "sites cannot be empty"),
        (
            dict(sites={"P00350|S|87": _site(), "P0CE47|H|119;119": _site(**TUF)}),
            "site key 'P00350|S|87' is not its site's key 'P00350|S|86'",
        ),
        (
            dict(quantification={"P00350|S|86": "quantified"}),
            "quantification keys must match sites keys",
        ),
        (
            dict(
                phosphosite_intensity={
                    "P00350|S|86": 2714000.0,
                    "P0CE47|H|119;119": 5.0,
                }
            ),
            "phosphosite_intensity keys must be exactly the quantified sites",
        ),
        (
            dict(phosphosite_intensity={"P00350|S|86": 0.0}),
            "is 0.0; an unquantified site is a not_quantified status",
        ),
        (dict(sample_label=""), "sample_label and measurement_type are non-empty"),
        (dict(replicate=0), "replicate is 1-based, got 0"),
        (
            dict(harvest_optical_density=-0.1),
            "harvest_optical_density must be positive, got -0.1",
        ),
        (
            dict(replicate_unit=None),
            "PhosphositeIntensityPhenotype.replicate_unit is unset and carries no "
            "ProvenanceGap",
        ),
        (
            dict(harvest_growth_phase=None),
            "harvest_growth_phase is unset and carries no ProvenanceGap",
        ),
    ],
)
def test_the_phenotype_refuses_disagreeing_maps_and_silent_sample_fields(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _phenotype(**change)
    assert message in str(excinfo.value)


def test_a_contradicted_replicate_unit_is_a_typed_gap() -> None:
    phenotype = _phenotype(
        replicate_unit=None, provenance_gaps=[_gap("replicate_unit")]
    )
    assert phenotype.replicate_unit is None
    assert phenotype.gapped_fields() == {"replicate_unit"}
    with pytest.raises(ValidationError, match="cannot both store a value"):
        _phenotype(provenance_gaps=[_gap("replicate_unit")])


def test_the_phenotype_round_trips_through_json() -> None:
    phenotype = _phenotype(
        replicate_unit=None, provenance_gaps=[_gap("replicate_unit")]
    )
    again = s.PhosphositeIntensityPhenotype.model_validate_json(
        phenotype.model_dump_json()
    )
    assert again == phenotype
    assert again.sites["P0CE47|H|119;119"].residue is s.PhosphoResidue.histidine


# --------------------------------------------------------------- DAG placement
def test_the_leaf_is_a_sibling_of_protein_abundance_not_a_child() -> None:
    """A direct ``Phenotype`` subclass beside ``ProteinAbundancePhenotype``; the served
    abundance leaf keeps its bases.
    """
    assert s.PhosphositeIntensityPhenotype.__bases__ == (s.Phenotype, ModelStrict)
    assert s.ProteinAbundancePhenotype.__bases__ == (s.Phenotype, ModelStrict)
    assert not issubclass(s.PhosphositeIntensityPhenotype, s.ProteinAbundancePhenotype)
    assert not issubclass(s.ProteinAbundancePhenotype, s.PhosphositeIntensityPhenotype)


def test_the_pair_is_registered_host_neutral_and_inherits_the_pinned_slot() -> None:
    """P2: the pair carries no host prefix; the pin lives on ``ReferenceGenome``."""
    experiment = s.PhosphositeIntensityExperiment
    reference = s.PhosphositeIntensityExperimentReference
    assert s.EXPERIMENT_TYPE_MAP["phosphosite_intensity"] is experiment
    assert s.EXPERIMENT_REFERENCE_TYPE_MAP["phosphosite_intensity"] is (reference)
    assert "bacterial_phosphosite_intensity" not in s.EXPERIMENT_TYPE_MAP
    assert s.RETIRED_EXPERIMENT_TYPES["bacterial_phosphosite_intensity"] == (
        "phosphosite_intensity"
    )
    assert experiment in get_args(s.ExperimentType)
    assert reference in get_args(s.ExperimentReferenceType)
    assert s.PhosphositeIntensityPhenotype in get_args(s.PhenotypeType)
    assert (
        reference.model_fields["genome_reference"].annotation
        == s.ExperimentReference.model_fields["genome_reference"].annotation
    )
    assert experiment.model_fields["phenotype"].annotation is (
        s.PhosphositeIntensityPhenotype
    )

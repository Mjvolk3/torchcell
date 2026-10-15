# tests/torchcell/verification/test_phosphosite.py
# [[tests.torchcell.verification.test_phosphosite]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_phosphosite.py
"""The phosphosite-intensity family gate (#921): each rule passes on a well-formed store
and fails on the one defect it exists to catch.

Records are built from the schema classes and dumped the way a loader stores them, so L0
validates real records. The family reads no strain, so the reference pins MG1655.
"""

from __future__ import annotations

import copy
from typing import Any

from torchcell.datamodels.schema import (
    CultureGrowthPhase,
    Environment,
    Genotype,
    Media,
    PhosphoResidue,
    Phosphosite,
    PhosphositeIntensityExperiment,
    PhosphositeIntensityExperimentReference,
    PhosphositeIntensityPhenotype,
    PhosphositeQuantification,
    ProteinIdNamespace,
    ReferenceGenome,
    SampleUnit,
    Temperature,
)
from torchcell.verification.phosphosite import verify_phosphosite_dataset
from torchcell.verification.report import Provenance

GND = Phosphosite(
    protein_ids=["P00350"],
    protein_id_namespace=ProteinIdNamespace.uniprotkb_accession,
    gene_names=["gnd"],
    positions=[86],
    residue=PhosphoResidue.serine,
    sequence_windows=["LLMVKAGAGTDAAIDSLKPYLDKGDIIIDGG"],
    localization_probability=0.999615,
    posterior_error_probability=4.27602e-05,
    identification_score=66.401,
    delta_score=58.487,
    diagnostic_peak=False,
)
TUF = Phosphosite(
    protein_ids=["P0CE47"],
    protein_id_namespace=ProteinIdNamespace.uniprotkb_accession,
    gene_names=["tufA", "tufB"],
    positions=[119, 119],
    residue=PhosphoResidue.histidine,
    sequence_windows=["LVVAATDGPMPQTREHILLGRQVGVPYIIVF"],
    localization_probability=0.5,
    posterior_error_probability=0.001,
    identification_score=80.0,
    delta_score=20.0,
    diagnostic_peak=True,
)
PROVENANCE = Provenance(source_uri="test", citation_key="test")


def _phenotype(label: str, gnd: float | None) -> PhosphositeIntensityPhenotype:
    status = PhosphositeQuantification
    return PhosphositeIntensityPhenotype(
        phosphosite_intensity={} if gnd is None else {GND.key(): gnd},
        quantification={
            GND.key(): status.not_quantified if gnd is None else status.quantified,
            TUF.key(): status.not_quantified,
        },
        sites={GND.key(): GND, TUF.key(): TUF},
        measurement_type="fe3_imac_lc_msms_maxquant_summed_xic",
        sample_label=label,
        replicate=1,
        replicate_unit=SampleUnit.technical_replicate,
        harvest_growth_phase=CultureGrowthPhase.exponential,
        harvest_optical_density=0.6,
    )


def _record(label: str, gnd: float | None) -> dict[str, Any]:
    env = Environment(
        media=Media(name="M9", state="liquid", is_synthetic=True),
        temperature=Temperature(value=37),
    )
    experiment = PhosphositeIntensityExperiment(
        dataset_name="test",
        genotype=Genotype(perturbations=[]),
        environment=env,
        phenotype=_phenotype(label, gnd),
    )
    reference = PhosphositeIntensityExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Escherichia coli",
            strain="MG1655",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            assembly_accession="GCA_000005845.2",
        ),
        environment_reference=env,
        phenotype_reference=_phenotype("s1", 10.0),
    )
    return {
        "experiment": experiment.model_dump(mode="json"),
        "reference": reference.model_dump(mode="json"),
    }


def _store() -> list[dict[str, Any]]:
    return [_record("s1", 10.0), _record("s2", None), _record("s3", 3.5)]


def _results(records: list[dict[str, Any]], expected: int = 3) -> dict[str, Any]:
    report = verify_phosphosite_dataset(
        records, dataset_name="test", provenance=PROVENANCE, expected_count=expected
    )
    return {r.name: r for r in report.results}


def test_a_well_formed_store_passes_every_family_rule() -> None:
    results = _results(_store())
    assert list(results) == [
        "structural",
        "count",
        "one_record_per_sample",
        "site_list_is_shared",
        "intensities_finite_positive",
        "site_evidence_is_per_site",
        "measurement_type_consistent",
    ]
    assert all(r.passed for r in results.values())
    assert results["intensities_finite_positive"].message == (
        "2 intensities; 0 non-finite or non-positive"
    )
    assert results["site_list_is_shared"].details == {"n_key_sets": 1, "sizes": [2]}


def test_a_repeated_sample_label_fails_one_record_per_sample() -> None:
    records = _store()
    records[2]["experiment"]["phenotype"]["sample_label"] = "s1"
    result = _results(records)["one_record_per_sample"]
    assert (result.passed, result.details) == (
        False,
        {"n_samples": 2, "repeated": ["s1"]},
    )


def test_a_record_missing_a_site_fails_site_list_is_shared() -> None:
    records = _store()
    del records[1]["experiment"]["phenotype"]["sites"][TUF.key()]
    result = _results(records)["site_list_is_shared"]
    assert (result.passed, result.details["sizes"]) == (False, [1, 2])


def test_a_non_positive_intensity_fails_its_l2_rule() -> None:
    records = _store()
    records[0]["experiment"]["phenotype"]["phosphosite_intensity"][GND.key()] = 0.0
    result = _results(records)["intensities_finite_positive"]
    assert (result.passed, result.details["bad"]) == (False, [GND.key()])


def test_a_site_whose_evidence_differs_between_records_fails() -> None:
    records = _store()
    site = copy.deepcopy(records[2]["experiment"]["phenotype"]["sites"][GND.key()])
    site["localization_probability"] = 0.5
    records[2]["experiment"]["phenotype"]["sites"][GND.key()] = site
    result = _results(records)["site_evidence_is_per_site"]
    assert (result.passed, result.details["differing"]) == (False, [GND.key()])


def test_two_measurement_types_fail_and_a_wrong_count_fails() -> None:
    records = _store()
    records[1]["experiment"]["phenotype"]["measurement_type"] = "other"
    results = _results(records, expected=4)
    assert results["measurement_type_consistent"].passed is False
    assert (results["count"].passed, results["count"].message) == (
        False,
        "observed 3, expected 4",
    )

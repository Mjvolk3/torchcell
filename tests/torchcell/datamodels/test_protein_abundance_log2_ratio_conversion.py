"""Tests for the build-time log2-ratio conversion of protein-abundance records."""

import math
from pathlib import Path

import pytest

from torchcell.datamodels.protein_abundance_log2_ratio_conversion import (
    ProteinAbundanceLog2RatioConverter,
    convert_protein_abundance_pair,
    measurement_type_for_log2_ratio,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ReferenceGenome,
    Temperature,
)

MTYPE = "swath_ms_maxlfq_batch_corrected_quantity"


def _pair(
    abundance: dict[str, float],
    ref_abundance: dict[str, float],
    ref_se: dict[str, float],
) -> tuple[ProteinAbundanceExperiment, ProteinAbundanceExperimentReference]:
    env = Environment(
        media=Media(name="SM", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = ProteinAbundanceExperiment(
        dataset_name="ProteomeMessner2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YAL059W", perturbed_gene_name="ECM1"
                )
            ]
        ),
        environment=env,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance=abundance,
            protein_abundance_se=None,
            n_replicates={k: 1 for k in abundance},
            measurement_type=MTYPE,
        ),
    )
    reference = ProteinAbundanceExperimentReference(
        dataset_name="ProteomeMessner2023Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance=ref_abundance,
            protein_abundance_se=ref_se,
            n_replicates={k: 388 for k in ref_abundance},
            measurement_type=MTYPE,
        ),
    )
    return experiment, reference


def test_pair_is_log2_ratio_and_reference_is_zero() -> None:
    exp, ref = _pair(
        {"YA": 200.0, "YB": 50.0}, {"YA": 100.0, "YB": 100.0}, {"YA": 10.0, "YB": 5.0}
    )
    new_exp, new_ref = convert_protein_abundance_pair(exp, ref)
    assert new_exp.phenotype.protein_abundance == {"YA": 1.0, "YB": -1.0}
    assert new_exp.phenotype.protein_abundance_se is None
    assert new_exp.phenotype.n_replicates == {"YA": 1, "YB": 1}
    assert new_exp.phenotype.measurement_type == measurement_type_for_log2_ratio(MTYPE)
    assert new_ref.phenotype_reference.protein_abundance == {"YA": 0.0, "YB": 0.0}
    se = new_ref.phenotype_reference.protein_abundance_se
    assert se is not None
    assert se["YA"] == pytest.approx(10.0 / (100.0 * math.log(2.0)))
    assert se["YB"] == pytest.approx(5.0 / (100.0 * math.log(2.0)))
    # genotype, environment and provenance travel unchanged
    assert new_exp.genotype == exp.genotype
    assert new_exp.environment == exp.environment
    assert new_ref.genome_reference == ref.genome_reference


def test_missing_reference_protein_raises() -> None:
    exp, ref = _pair({"YA": 2.0, "YB": 3.0}, {"YA": 1.0}, {"YA": 0.1})
    with pytest.raises(ValueError, match="no reference value"):
        convert_protein_abundance_pair(exp, ref)


def test_non_positive_quantity_raises() -> None:
    exp, ref = _pair({"YA": 0.0}, {"YA": 1.0}, {"YA": 0.1})
    with pytest.raises(ValueError, match="non-positive"):
        convert_protein_abundance_pair(exp, ref)


def test_double_conversion_raises() -> None:
    exp, ref = _pair({"YA": 2.0}, {"YA": 1.0}, {"YA": 0.1})
    once_exp, once_ref = convert_protein_abundance_pair(exp, ref)
    with pytest.raises(ValueError, match="already a log2 ratio"):
        convert_protein_abundance_pair(once_exp, once_ref)


def test_converter_passes_other_types_through(tmp_path: Path) -> None:
    converter = ProteinAbundanceLog2RatioConverter(root=str(tmp_path), query=None)  # type: ignore[arg-type]
    env = Environment(
        media=Media(name="YPD", state="liquid", is_synthetic=False),
        temperature=Temperature(value=30),
    )
    fit = FitnessExperiment(
        dataset_name="SmfCostanzo2016Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YAL059W", perturbed_gene_name="ECM1"
                )
            ]
        ),
        environment=env,
        phenotype=FitnessPhenotype(fitness=0.9, fitness_std=0.01),
    )
    fit_ref = FitnessExperimentReference(
        dataset_name="SmfCostanzo2016Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=0.0),
    )
    out = converter.convert({"experiment": fit, "experiment_reference": fit_ref})
    assert out["experiment"] is fit
    assert out["experiment_reference"] is fit_ref

    exp, ref = _pair({"YA": 4.0}, {"YA": 1.0}, {"YA": 0.1})
    out = converter.convert({"experiment": exp, "experiment_reference": ref})
    assert out["experiment"].phenotype.protein_abundance == {"YA": 2.0}  # type: ignore[union-attr]

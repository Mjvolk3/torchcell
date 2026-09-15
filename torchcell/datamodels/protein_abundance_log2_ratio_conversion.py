# torchcell/datamodels/protein_abundance_log2_ratio_conversion
# [[torchcell.datamodels.protein_abundance_log2_ratio_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/protein_abundance_log2_ratio_conversion
# Test file: tests/torchcell/datamodels/test_protein_abundance_log2_ratio_conversion.py
"""Build-time conversion of a protein-abundance experiment to the log2 ratio to its reference.

A ``ProteinAbundanceExperiment`` stores the per-strain quantity the assay reports (for
Messner 2023 a linear, batch-corrected MaxLFQ quantity) and its reference carries the
wild-type profile the same quantity was measured against (the 388-replicate HIS3 strain).
The expression datasets the multitask model trains on store ``log2(mutant / reference)``
per gene, so a proteome that is to be predicted the same way needs the same transform,
and it needs the reference to do it, which the per-experiment conversion hook of
:class:`~torchcell.datamodels.conversion.Converter` does not see. This converter overrides
``convert`` to read both halves of the record and rewrites

    protein_abundance[k]  ->  log2(protein_abundance[k] / reference[k])

for every protein ``k`` the experiment measured, leaving every other experiment type
untouched. The reference profile becomes exactly zero (a ratio to itself), its standard
error is carried to the log2 scale by the delta method, ``se / (value * ln 2)``, and the
phenotype's ``measurement_type`` is rewritten to name the transform and the original
quantity, so a record read back from the LMDB says what it holds. A protein missing from
the reference, or a non-positive quantity on either side, is an error: the loader
guarantees neither can happen for Messner, and a silent NaN would train as a zero.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torchcell.datamodels.conversion import ConversionMap, Converter
from torchcell.datamodels.schema import (
    ExperimentReferenceType,
    ExperimentType,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
)

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw

LOG2_RATIO_PREFIX = "log2_ratio_to_reference"
_LN2 = math.log(2.0)


def measurement_type_for_log2_ratio(original: str) -> str:
    """Name of the converted quantity, ``log2_ratio_to_reference(<original>)``."""
    return f"{LOG2_RATIO_PREFIX}({original})"


def convert_protein_abundance_pair(
    experiment: ProteinAbundanceExperiment,
    reference: ProteinAbundanceExperimentReference,
) -> tuple[ProteinAbundanceExperiment, ProteinAbundanceExperimentReference]:
    """Return the (experiment, reference) pair on the log2-ratio scale."""
    exp_ph = experiment.phenotype
    ref_ph = reference.phenotype_reference
    if exp_ph.measurement_type.startswith(LOG2_RATIO_PREFIX):
        raise ValueError(
            f"{experiment.dataset_name}: measurement_type {exp_ph.measurement_type!r} "
            "is already a log2 ratio; converting twice would be wrong"
        )
    ref_ab = ref_ph.protein_abundance
    missing = sorted(k for k in exp_ph.protein_abundance if k not in ref_ab)
    if missing:
        raise ValueError(
            f"{experiment.dataset_name}: {len(missing)} measured proteins have no "
            f"reference value (first: {missing[:5]})"
        )
    ratio: dict[str, float] = {}
    for k, v in exp_ph.protein_abundance.items():
        r = ref_ab[k]
        if not (v > 0.0 and r > 0.0):
            raise ValueError(
                f"{experiment.dataset_name}: non-positive quantity for {k} "
                f"(experiment {v}, reference {r}); log2 is undefined"
            )
        ratio[k] = math.log2(v / r)

    exp_se = exp_ph.protein_abundance_se
    exp_se_log2 = (
        {k: exp_se[k] / (exp_ph.protein_abundance[k] * _LN2) for k in exp_se}
        if exp_se is not None
        else None
    )
    new_mtype = measurement_type_for_log2_ratio(exp_ph.measurement_type)
    new_exp_ph = ProteinAbundancePhenotype(
        protein_abundance=ratio,
        protein_abundance_se=exp_se_log2,
        n_replicates=exp_ph.n_replicates,
        measurement_type=new_mtype,
    )
    ref_se = ref_ph.protein_abundance_se
    ref_se_log2 = (
        {k: ref_se[k] / (ref_ab[k] * _LN2) for k in ref_se}
        if ref_se is not None
        else None
    )
    new_ref_ph = ProteinAbundancePhenotype(
        protein_abundance={k: 0.0 for k in ref_ab},
        protein_abundance_se=ref_se_log2,
        n_replicates=ref_ph.n_replicates,
        measurement_type=measurement_type_for_log2_ratio(ref_ph.measurement_type),
    )
    new_experiment = ProteinAbundanceExperiment(
        dataset_name=experiment.dataset_name,
        genotype=experiment.genotype,
        environment=experiment.environment,
        phenotype=new_exp_ph,
    )
    new_reference = ProteinAbundanceExperimentReference(
        dataset_name=reference.dataset_name,
        genome_reference=reference.genome_reference,
        environment_reference=reference.environment_reference,
        phenotype_reference=new_ref_ph,
    )
    return new_experiment, new_reference


class ProteinAbundanceLog2RatioConverter(Converter):
    """Rewrite every protein-abundance record to ``log2(experiment / reference)``.

    Other experiment types pass through unchanged, so the converter can sit in a build
    that unions the proteome with expression and fitness datasets.
    """

    def __init__(self, root: str, query: Neo4jQueryRaw):
        """Store the root and query; no sub-converters."""
        super().__init__(root, query)

    @property
    def conversion_map(self) -> ConversionMap:
        """Empty: ``convert`` is overridden because it needs the reference too."""
        return ConversionMap(entries=[])

    def convert(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> dict[str, ExperimentType | ExperimentReferenceType | None]:
        """Convert one record; non-proteome records are returned as they came."""
        if "experiment" not in data or "experiment_reference" not in data:
            raise ValueError(
                "Input data must contain both 'experiment' and 'experiment_reference' keys"
            )
        experiment = data["experiment"]
        reference = data["experiment_reference"]
        if not isinstance(experiment, ProteinAbundanceExperiment):
            return dict(data)
        if not isinstance(reference, ProteinAbundanceExperimentReference):
            raise TypeError(
                f"{experiment.dataset_name}: protein-abundance experiment paired with a "
                f"{type(reference).__name__} reference"
            )
        new_experiment, new_reference = convert_protein_abundance_pair(
            experiment, reference
        )
        out: dict[str, ExperimentType | ExperimentReferenceType | None] = dict(data)
        out["experiment"] = new_experiment
        out["experiment_reference"] = new_reference
        return out

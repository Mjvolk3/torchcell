# torchcell/data/genotype_environment_aggregate
# [[torchcell.data.genotype_environment_aggregate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/genotype_environment_aggregate
# Test file: tests/torchcell/data/test_genotype_environment_aggregate.py
"""Aggregator keyed on the (genotype, environment) cell an experiment measures.

``GenotypeAggregator`` keys on the perturbed gene set alone, which is the right identity for
a solid-growth build, where every record of a genotype is the same condition and grouping
them is what lets a read-time label policy choose among measurements. A chemogenomic record
is a strain in a condition, and one gene is measured under 41 to 5,170 conditions, so the
gene-set key would put every condition of a gene into one entry and the label table would
keep the first. This aggregator keys on the cell instead: the genotype AND the environment.

The genotype half is the sorted list of (gene, perturbation type, copy number) plus the
reference genome's ploidy. Perturbation type is part of the key here, unlike
``GenotypeAggregator``, because a heterozygous deletion (``engineered_copy_number``, one
of two copies) and a homozygous one (``kanmx_deletion``, both copies) of the same gene are
different strains with different measured responses, and Hoepfner 2014 carries both for
648,977 gene-by-compound cells. Ploidy is included for the same reason at the genome level.

The environment half is ``environment_identity`` from ``torchcell.datamodels.identity``,
the same content address the graph adapter uses for its environment nodes, so an entry here
groups exactly the records the graph would hang off one environment node: same medium
composition, temperature, added compounds at their doses, aerobicity and duration.
Provenance quotes and free-text names are not part of it.

What groups together, measured on the served chemogenomic records: repeated screens of the
same compound at the same concentration (Hoepfner's fourteen compounds screened in more
than one study, 71 condition pairs), and nothing else within a dataset, because every other
record differs in gene, compound, dose or generation count. Across datasets no environment
is shared, since the media, durations and dose units differ, so this key never merges two
sources. Each record's raw environment is parsed once in aggregation pass 1, at 0.12 to
0.36 ms per record on the four served datasets.
"""

from __future__ import annotations

from typing import Any

from torchcell.data.aggregate import Aggregator
from torchcell.datamodels import Environment, ExperimentReferenceType, ExperimentType
from torchcell.datamodels.identity import (
    _canonical,
    environment_identity,
    identity_sha256,
)


def _perturbation_identity(
    gene: str,
    perturbation_type: str,
    copy_number: float | None,
    reference_copy_number: float | None,
) -> dict[str, Any]:
    """The slots of a gene perturbation that make it a different strain."""
    return {
        "gene": gene,
        "perturbation_type": perturbation_type,
        "copy_number": copy_number,
        "reference_copy_number": reference_copy_number,
    }


def cell_identity(
    perturbations: list[dict[str, Any]], ploidy: str, environment: Environment
) -> dict[str, Any]:
    """The (genotype, environment) identity an aggregation key is the hash of."""
    return {
        "ploidy": ploidy,
        "perturbations": sorted(perturbations, key=_canonical),
        "environment": environment_identity(environment),
    }


class GenotypeEnvironmentAggregator(Aggregator):
    """Group experiments that measure the same strain in the same environment."""

    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """Return the sha256 of the record's (genotype, environment) identity."""
        experiment = data["experiment"]
        reference = data["experiment_reference"]
        perturbations = [
            _perturbation_identity(
                pert.systematic_gene_name,
                pert.perturbation_type,
                getattr(pert, "copy_number", None),
                getattr(pert, "reference_copy_number", None),
            )
            for pert in experiment.genotype.perturbations  # type: ignore[union-attr]
        ]
        return identity_sha256(
            cell_identity(
                perturbations,
                reference.genome_reference.ploidy,  # type: ignore[union-attr]
                experiment.environment,  # type: ignore[union-attr]
            )
        )

    def aggregate_key_raw(self, record: dict[str, Any]) -> str:
        """Return the same key off the stored JSON, parsing only the environment."""
        experiment = record["experiment"]
        perturbations = [
            _perturbation_identity(
                pert["systematic_gene_name"],
                pert["perturbation_type"],
                pert.get("copy_number"),
                pert.get("reference_copy_number"),
            )
            for pert in experiment["genotype"]["perturbations"]
        ]
        return identity_sha256(
            cell_identity(
                perturbations,
                record["experiment_reference"]["genome_reference"]["ploidy"],
                Environment(**experiment["environment"]),
            )
        )

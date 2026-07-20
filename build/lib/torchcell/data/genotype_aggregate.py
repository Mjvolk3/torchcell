"""Aggregator that groups experiments by their genotype perturbation set."""

import hashlib
from typing import cast

from torchcell.data import Aggregator
from torchcell.datamodels import Experiment, ExperimentReferenceType, ExperimentType


class GenotypeAggregator(Aggregator):
    """Aggregator keyed by a hash of the experiment's perturbation set."""

    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """Return a SHA-256 hash of the experiment's perturbation tuple."""
        # data["experiment"] is always an Experiment (single Genotype) here, not a reference.
        genotype = cast(Experiment, data["experiment"]).genotype
        perturbations = genotype.perturbations

        # Create a tuple of (systematic_gene_name, perturbation_type) for each perturbation
        perturbation_key = tuple(
            (pert.systematic_gene_name, pert.perturbation_type)
            for pert in perturbations
        )

        # Hash the perturbation key
        return hashlib.sha256(str(perturbation_key).encode()).hexdigest()

import hashlib
from torchcell.data import Aggregator
from torchcell.datamodels import ExperimentType, ExperimentReferenceType


class GenotypeAggregator(Aggregator):
    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        genotype = data["experiment"].genotype
        perturbations = genotype.perturbations

        # Create a tuple of (systematic_gene_name, perturbation_type) for each perturbation
        perturbation_key = tuple(
            (pert.systematic_gene_name, pert.perturbation_type)
            for pert in perturbations
        )

        # Hash the perturbation key
        return hashlib.sha256(str(perturbation_key).encode()).hexdigest()

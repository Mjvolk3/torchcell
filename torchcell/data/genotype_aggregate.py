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
        """Return a SHA-256 hash of the experiment's perturbed gene set.

        Keyed on the sorted SET of systematic_gene_names ONLY (perturbation_type
        is intentionally dropped) so this matches how
        ``MeanExperimentDeduplicator.duplicate_check`` groups genotypes. This lets
        the same genotype expressed across modalities co-locate: a *singleton*
        expression record (marker-specific type, e.g. ``kanmx_deletion``) and a
        *replicated* fitness record rewritten to ``mean_deletion`` by the
        deduplicator share one bucket because their perturbed gene set is equal.
        (The deduplicator additionally keys on ``experiment_type``; the aggregator
        must NOT, since merging across modalities is the whole point here.)

        # TODO When non-deletion perturbation axes are introduced (overexpression,
        # allele swaps, integrations, etc.), this key MUST become
        # ``(gene-set, perturbation-axis)`` so that a deletion and an
        # overexpression of the SAME gene are not merged into one bucket. Today
        # every perturbation is a deletion, so the gene set alone is unambiguous.
        """
        # data["experiment"] is always an Experiment (single Genotype) here, not a reference.
        genotype = cast(Experiment, data["experiment"]).genotype
        perturbations = genotype.perturbations

        # Key on the sorted SET of systematic gene names only.
        sorted_gene_names = sorted(
            {pert.systematic_gene_name for pert in perturbations}
        )

        # Hash the gene-set key
        return hashlib.sha256(str(sorted_gene_names).encode()).hexdigest()

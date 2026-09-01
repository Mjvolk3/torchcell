"""Aggregators that group experiments by their genotype perturbation set."""

import hashlib
from collections.abc import Iterable
from typing import Any, cast

from torchcell.data import Aggregator
from torchcell.datamodels import Experiment, ExperimentReferenceType, ExperimentType
from torchcell.datamodels.schema import GenePerturbationType

#: A perturbation is a *deletion* when its ``perturbation_type`` ends with
#: ``"deletion"``. This covers ``deletion``, ``kanmx_deletion``, ``natmx_deletion``,
#: ``sga_kanmx_deletion``, ``sga_natmx_deletion``, ``mean_deletion``,
#: ``marker_deletion`` and ``crispr_deletion``, and excludes ``gene_addition``,
#: ``allele``, ``damp`` and the natural-variation axes. It is the same predicate the
#: Fig-6 Cypher uses (``pert.perturbation_type ENDS WITH 'deletion'``), so the build
#: query and the aggregator agree on what a deletion is.
DELETION_TYPE_SUFFIX = "deletion"


def _hash_gene_set(gene_names: Iterable[str]) -> str:
    """Return a SHA-256 hash of a sorted, de-duplicated gene-name set."""
    return hashlib.sha256(str(sorted(set(gene_names))).encode()).hexdigest()


def _perturbations(
    data: dict[str, ExperimentType | ExperimentReferenceType],
) -> list[GenePerturbationType]:
    """Return the perturbations of the experiment half of an aggregation pair."""
    # data["experiment"] is always an Experiment (single Genotype) here, not a reference.
    return cast(Experiment, data["experiment"]).genotype.perturbations


class GenotypeAggregator(Aggregator):
    """Aggregator keyed by a hash of the experiment's full perturbation set."""

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

        Every perturbation counts toward the key, including heterologous
        ``gene_addition`` cassette genes. That makes a cassette-bearing production
        strain key-disjoint from an otherwise identical single-deletion strain --
        see ``DeletionKeyedGenotypeAggregator`` for the variant that treats a
        constant cassette as reference-strain background.

        # TODO When non-deletion perturbation axes are introduced (overexpression,
        # allele swaps, integrations, etc.), this key MUST become
        # ``(gene-set, perturbation-axis)`` so that a deletion and an
        # overexpression of the SAME gene are not merged into one bucket. Today
        # every perturbation is a deletion, so the gene set alone is unambiguous.
        """
        return _hash_gene_set(
            pert.systematic_gene_name for pert in _perturbations(data)
        )

    def aggregate_key_raw(self, record: dict[str, Any]) -> str:
        """Return the gene-set grouping key for one raw record dict.

        Same hash as :meth:`aggregate_check`, read off the stored JSON so the
        streaming ``process`` never builds pydantic objects to group.
        """
        return _hash_gene_set(
            pert["systematic_gene_name"]
            for pert in record["experiment"]["genotype"]["perturbations"]
        )


class DeletionKeyedGenotypeAggregator(GenotypeAggregator):
    """Aggregator keyed on the DELETION gene set only; other axes are background.

    Use this when a dataset's non-deletion perturbations are a *constant
    engineered background* rather than the experimental variable. The motivating
    case is the pigment production screens: every Cachera betaxanthin strain
    carries the same 4-gene cassette (``CYP76AD1``, ``DOD``, ``ARO4`` K229L,
    ``ARO7`` G141S) and every Ozaydin beta-carotene strain the same 3-gene
    cassette (``crtYB``, ``crtI``, ``BTS1``), emitted as ``gene_addition`` /
    ``allele`` perturbations. Members of those screens differ ONLY by which ORF is
    deleted, so the deletion gene-set is the genotype identity that matters.

    Because :class:`GenotypeAggregator` keys on the FULL perturbation set, a
    cassette-bearing pigment genotype can never share a bucket with a single-KO
    metabolome genotype -- measured co-location is exactly ZERO. Keying on
    deletions alone recovers it (betaxanthin/metabolome and
    beta-carotene/metabolome both in the thousands), which is what makes any
    "does the metabolome improve production prediction" contrast possible.

    This is the axis-aware-key follow-up WS2b deferred, and it implements Design
    Decision 3 of [[plan.simb-2026-multimodal-cgt.2026.07.21]] -- the cassette
    belongs to the *reference cell*, not to the perturbation, so a production
    phenotype is predicted as a function of the deletion applied to a
    pathway-carrying reference strain.

    A genotype whose perturbations are ALL non-deletions (e.g. cassette-only, no
    ORF knocked out) hashes to the empty set and therefore groups with other
    such genotypes. The Fig-6 build queries require at least one deletion per
    strain, so that bucket is empty in practice.
    """

    def aggregate_check(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> str:
        """Return a SHA-256 hash of the experiment's DELETED gene set."""
        return _hash_gene_set(
            pert.systematic_gene_name
            for pert in _perturbations(data)
            if pert.perturbation_type.endswith(DELETION_TYPE_SUFFIX)
        )

    def aggregate_key_raw(self, record: dict[str, Any]) -> str:
        """Return the deletion-gene-set grouping key for one raw record dict."""
        return _hash_gene_set(
            pert["systematic_gene_name"]
            for pert in record["experiment"]["genotype"]["perturbations"]
            if pert["perturbation_type"].endswith(DELETION_TYPE_SUFFIX)
        )

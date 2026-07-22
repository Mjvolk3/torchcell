# torchcell/data/mean_experiment_deduplicate
# [[torchcell.data.mean_experiment_deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/mean_experiment_deduplicate
# Test file: tests/torchcell/data/test_mean_experiment_deduplicate.py

"""Deduplicator that merges duplicate experiments by averaging their values."""

import hashlib
import logging
from typing import Any, cast

import numpy as np
from scipy.stats import t

from torchcell.data import Deduplicator
from torchcell.datamodels import (
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    MeanDeletionPerturbation,
)
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    CalMorphPhenotype,
    GenePerturbationType,
    MetabolitePhenotype,
    MicroarrayExpressionPhenotype,
    Phenotype,
    ProteinAbundancePhenotype,
    RNASeqExpressionPhenotype,
    VisualScorePhenotype,
)

log = logging.getLogger(__name__)

# Phenotype experiment types that are merged by elementwise dict/scalar averaging.
# Their duplicate groups are keyed the same way as fitness (experiment_type + sorted
# perturbed gene names in duplicate_check), so cross-modality records never collide.
_VECTOR_EXPERIMENT_TYPES = frozenset(
    {
        "microarray_expression",
        "rnaseq_expression",
        "calmorph",
        "metabolite",
        "protein_abundance",
        "visual_score",
    }
)


def _mean_float_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    """Elementwise mean over the union of keys, averaging only the present values."""
    keys: set[str] = set()
    for d in dicts:
        keys.update(d.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [d[k] for d in dicts if k in d]
        out[k] = float(np.mean(vals))
    return out


def _rms_pool_float_dict(
    dicts: list[dict[str, float] | None],
) -> dict[str, float] | None:
    """RMS-pool per-key stds: sqrt(mean(std^2)) over present values.

    Null-safe: dicts that are ``None`` are skipped; if every entry is ``None`` the
    field stays ``None`` (e.g. Mulleder metabolite_level_se is null, n_replicates=1).
    """
    present = [d for d in dicts if d is not None]
    if not present:
        return None
    keys: set[str] = set()
    for d in present:
        keys.update(d.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [d[k] for d in present if k in d]
        out[k] = float(np.sqrt(np.mean(np.array(vals) ** 2)))
    return out


def _sum_int_dict(dicts: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-key integer counts (e.g. n_replicates, read counts) over the union."""
    keys: set[str] = set()
    for d in dicts:
        keys.update(d.keys())
    out: dict[str, int] = {}
    for k in sorted(keys):
        out[k] = int(sum(d[k] for d in dicts if k in d))
    return out


class MeanExperimentDeduplicator(Deduplicator):
    """Deduplicate experiments by averaging duplicates of the same genotype."""

    def duplicate_check(self, data: list[dict[str, Any]]) -> dict[str, list[int]]:
        """Group data indices by hash of experiment type and perturbed genes."""
        duplicate_check: dict[str, list[int]] = {}
        for idx, item in enumerate(data):
            experiment = item["experiment"]
            experiment_type = experiment.experiment_type
            perturbations = experiment.genotype.perturbations
            sorted_gene_names = sorted(
                [pert.systematic_gene_name for pert in perturbations]
            )

            # Create a hash key that includes both experiment type and perturbations
            hash_input = f"{experiment_type}:{str(sorted_gene_names)}"
            hash_key = hashlib.sha256(hash_input.encode()).hexdigest()

            if hash_key not in duplicate_check:
                duplicate_check[hash_key] = []
            duplicate_check[hash_key].append(idx)
        return duplicate_check

    def create_deduplicate_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return a single mean entry for a group of duplicate experiments."""
        experiment_type = duplicate_experiments[0]["experiment"].experiment_type
        if experiment_type == "fitness":
            return self._create_mean_fitness_entry(duplicate_experiments)
        elif experiment_type == "gene interaction":
            return self._create_mean_gene_interaction_entry(duplicate_experiments)
        elif experiment_type in _VECTOR_EXPERIMENT_TYPES:
            return self._create_mean_vector_entry(duplicate_experiments)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def _create_mean_vector_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Mean-merge phenotype-family duplicates (expression/morphology/metabolite/etc.).

        Mirrors the fitness rule: elementwise mean of the per-key value dict, RMS-pooled
        std for the uncertainty, MeanDeletionPerturbation genotype, and dataset_name
        joined as "a+b+...". The concrete Experiment/ExperimentReference classes are
        resolved generically from the type maps so a new family (e.g. Messner protein
        abundance) works with no further edits.
        """
        experiment_type = duplicate_experiments[0]["experiment"].experiment_type
        experiment_cls = EXPERIMENT_TYPE_MAP[experiment_type]
        reference_cls = EXPERIMENT_REFERENCE_TYPE_MAP[experiment_type]

        mean_phenotype = self._merge_phenotype(
            [exp["experiment"].phenotype for exp in duplicate_experiments]
        )
        mean_genotype = self._create_mean_genotype(duplicate_experiments)
        dataset_name = ("+").join(
            sorted([i["experiment"].dataset_name for i in duplicate_experiments])
        )
        mean_experiment = experiment_cls(
            dataset_name=dataset_name,
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        mean_reference_phenotype = self._merge_phenotype(
            [
                exp["experiment_reference"].phenotype_reference
                for exp in duplicate_experiments
            ]
        )
        reference_dataset_name = ("+").join(
            sorted(
                [i["experiment_reference"].dataset_name for i in duplicate_experiments]
            )
        )
        mean_reference = reference_cls(
            dataset_name=reference_dataset_name,
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_reference_phenotype,
        )

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}

    def _merge_phenotype(self, phenotypes: list[Phenotype]) -> Phenotype:
        """Dispatch elementwise merge on the concrete phenotype type.

        Value fields are averaged elementwise (per gene/feature/metabolite), std/SE
        fields are RMS-pooled (null-safe), and per-key replicate counts are summed.
        Non-numeric metadata (measurement_type, ordinal scale, semantics) is taken from
        the first record, which is identical across a duplicate group.
        """
        first = phenotypes[0]
        if isinstance(first, MicroarrayExpressionPhenotype):
            micro = cast(list[MicroarrayExpressionPhenotype], phenotypes)
            return MicroarrayExpressionPhenotype(
                expression_log2_ratio=_mean_float_dict(
                    [p.expression_log2_ratio for p in micro]
                ),
                expression_log2_ratio_se=_rms_pool_float_dict(
                    [p.expression_log2_ratio_se for p in micro]
                ),
                expression=_mean_float_dict([p.expression for p in micro]),
                expression_log2_ratio_variance=_rms_pool_float_dict(
                    [p.expression_log2_ratio_variance for p in micro]
                ),
                n_replicates=_sum_int_dict([p.n_replicates for p in micro]),
            )
        if isinstance(first, RNASeqExpressionPhenotype):
            rna = cast(list[RNASeqExpressionPhenotype], phenotypes)
            n_mapped = [p.n_mapped_reads for p in rna if p.n_mapped_reads is not None]
            return RNASeqExpressionPhenotype(
                expression_tpm=_mean_float_dict([p.expression_tpm for p in rna]),
                expression_count=_sum_int_dict([p.expression_count for p in rna]),
                measurement_type=first.measurement_type,
                n_mapped_reads=int(sum(n_mapped)) if n_mapped else None,
            )
        if isinstance(first, CalMorphPhenotype):
            morph = cast(list[CalMorphPhenotype], phenotypes)
            cv_dicts = [p.calmorph_coefficient_of_variation for p in morph]
            merged_cv = (
                _mean_float_dict([d for d in cv_dicts if d is not None])
                if any(d is not None for d in cv_dicts)
                else None
            )
            return CalMorphPhenotype(
                calmorph=_mean_float_dict([p.calmorph for p in morph]),
                calmorph_coefficient_of_variation=merged_cv,
            )
        if isinstance(first, MetabolitePhenotype):
            metab = cast(list[MetabolitePhenotype], phenotypes)
            target_ids = next(
                (p.target_metabolite_ids for p in metab if p.target_metabolite_ids),
                None,
            )
            return MetabolitePhenotype(
                metabolite_level=_mean_float_dict([p.metabolite_level for p in metab]),
                metabolite_level_se=_rms_pool_float_dict(
                    [p.metabolite_level_se for p in metab]
                ),
                n_replicates=_sum_int_dict([p.n_replicates for p in metab]),
                measurement_type=first.measurement_type,
                target_metabolite_ids=target_ids,
            )
        if isinstance(first, ProteinAbundancePhenotype):
            prot = cast(list[ProteinAbundancePhenotype], phenotypes)
            return ProteinAbundancePhenotype(
                protein_abundance=_mean_float_dict([p.protein_abundance for p in prot]),
                protein_abundance_se=_rms_pool_float_dict(
                    [p.protein_abundance_se for p in prot]
                ),
                n_replicates=_sum_int_dict([p.n_replicates for p in prot]),
                measurement_type=first.measurement_type,
            )
        if isinstance(first, VisualScorePhenotype):
            vis = cast(list[VisualScorePhenotype], phenotypes)
            score_mins = [
                p.visual_score_min for p in vis if p.visual_score_min is not None
            ]
            comments = next(
                (p.comment_annotations for p in vis if p.comment_annotations), None
            )
            score_texts = [p.score_text for p in vis if p.score_text is not None]
            return VisualScorePhenotype(
                visual_score=float(np.mean([p.visual_score for p in vis])),
                visual_score_min=min(score_mins) if score_mins else None,
                n_replicates=int(sum(p.n_replicates for p in vis)),
                score_scale_min=first.score_scale_min,
                score_scale_max=first.score_scale_max,
                score_semantics=first.score_semantics,
                target_product=first.target_product,
                target_metabolite_id=first.target_metabolite_id,
                score_text=score_texts[0] if score_texts else None,
                comment_annotations=comments,
            )
        raise ValueError(
            f"Unsupported phenotype type for mean merge: {type(first).__name__}"
        )

    def _create_mean_fitness_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        fitness_values = [
            exp["experiment"].phenotype.fitness
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness is not None
        ]
        fitness_stds = [
            exp["experiment"].phenotype.fitness_std
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness_std is not None
        ]

        mean_fitness = np.mean(fitness_values) if fitness_values else None
        mean_fitness_std = (
            np.sqrt(np.mean(np.array(fitness_stds) ** 2)) if fitness_stds else None
        )

        mean_phenotype = FitnessPhenotype(
            fitness=cast(float, mean_fitness), fitness_std=mean_fitness_std
        )

        mean_genotype = self._create_mean_genotype(duplicate_experiments)

        dataset_name = ("+").join(
            sorted([i["experiment"].dataset_name for i in duplicate_experiments])
        )

        mean_experiment = FitnessExperiment(
            dataset_name=dataset_name,
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        mean_reference = self._create_mean_fitness_reference(duplicate_experiments)

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}

    def _create_mean_gene_interaction_entry(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        interaction_values = [
            exp["experiment"].phenotype.gene_interaction
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.gene_interaction is not None
        ]
        p_values = [
            exp["experiment"].phenotype.gene_interaction_p_value
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.gene_interaction_p_value is not None
        ]

        mean_interaction = np.mean(interaction_values) if interaction_values else None
        aggregated_p_value = self._compute_p_value_for_mean(
            interaction_values, p_values
        )

        # Preserve the source arity (digenic -> "edge", trigenic -> "hyperedge");
        # do not fall back to the class default, which would relabel digenic means.
        mean_phenotype = GeneInteractionPhenotype(
            gene_interaction=cast(float, mean_interaction),
            gene_interaction_p_value=aggregated_p_value,
            graph_level=duplicate_experiments[0]["experiment"].phenotype.graph_level,
        )

        mean_genotype = self._create_mean_genotype(duplicate_experiments)

        dataset_name = ("+").join(
            sorted([i["experiment"].dataset_name for i in duplicate_experiments])
        )

        mean_experiment = GeneInteractionExperiment(
            dataset_name=dataset_name,
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        mean_reference = self._create_mean_gene_interaction_reference(
            duplicate_experiments
        )

        return {"experiment": mean_experiment, "experiment_reference": mean_reference}

    def _create_mean_genotype(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> Genotype:
        mean_perturbations: list[GenePerturbationType] = []
        for pert in duplicate_experiments[0]["experiment"].genotype.perturbations:
            mean_pert = MeanDeletionPerturbation(
                systematic_gene_name=pert.systematic_gene_name,
                perturbed_gene_name=pert.perturbed_gene_name,
                num_duplicates=len(duplicate_experiments),
            )
            mean_perturbations.append(mean_pert)
        return Genotype(perturbations=mean_perturbations)

    def _create_mean_fitness_reference(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> FitnessExperimentReference:
        fitness_ref_values = [
            exp["experiment_reference"].phenotype_reference.fitness
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.fitness is not None
        ]
        fitness_ref_stds = [
            exp["experiment_reference"].phenotype_reference.fitness_std
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.fitness_std is not None
        ]

        mean_fitness_ref = np.mean(fitness_ref_values) if fitness_ref_values else None
        mean_fitness_ref_std = (
            np.sqrt(np.mean(np.array(fitness_ref_stds) ** 2))
            if fitness_ref_stds
            else None
        )

        mean_phenotype_reference = FitnessPhenotype(
            fitness=cast(float, mean_fitness_ref), fitness_std=mean_fitness_ref_std
        )

        dataset_name = ("+").join(
            sorted(
                [i["experiment_reference"].dataset_name for i in duplicate_experiments]
            )
        )

        return FitnessExperimentReference(
            dataset_name=dataset_name,
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

    def _create_mean_gene_interaction_reference(
        self, duplicate_experiments: list[dict[str, Any]]
    ) -> GeneInteractionExperimentReference:
        interaction_ref_values = [
            exp["experiment_reference"].phenotype_reference.gene_interaction
            for exp in duplicate_experiments
            if exp["experiment_reference"].phenotype_reference.gene_interaction
            is not None
        ]

        mean_interaction_ref = (
            np.mean(interaction_ref_values) if interaction_ref_values else None
        )

        mean_phenotype_reference = GeneInteractionPhenotype(
            gene_interaction=cast(float, mean_interaction_ref),
            gene_interaction_p_value=None,
            graph_level=duplicate_experiments[0][
                "experiment_reference"
            ].phenotype_reference.graph_level,
        )

        dataset_name = ("+").join(
            sorted(
                [i["experiment_reference"].dataset_name for i in duplicate_experiments]
            )
        )

        return GeneInteractionExperimentReference(
            dataset_name=dataset_name,
            genome_reference=duplicate_experiments[0][
                "experiment_reference"
            ].genome_reference,
            environment_reference=duplicate_experiments[0][
                "experiment_reference"
            ].environment_reference,
            phenotype_reference=mean_phenotype_reference,
        )

    def _compute_p_value_for_mean(self, x: list[float], p_values: list[float]) -> float:
        if len(x) != len(p_values):
            raise ValueError("x and p_values must have the same length.")

        n = len(x)

        if n < 2:
            raise ValueError("At least two data points are required.")

        mean_x = np.mean(x)
        sample_std_dev = np.std(x, ddof=1)
        sem = sample_std_dev / np.sqrt(n)
        t_stat = mean_x / sem
        p_value_for_mean = t.sf(np.abs(t_stat), df=n - 1) * 2

        return cast(float, p_value_for_mean)


if __name__ == "__main__":
    pass

# tests/torchcell/data/test_mean_experiment_deduplicate.py
# [[tests.torchcell.data.test_mean_experiment_deduplicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_mean_experiment_deduplicate.py
"""``MeanExperimentDeduplicator`` on three in-memory duplicates, no LMDB.

Three fitness records of the same single deletion with fitness 1, 2 and 6 (stds 0.1, 0.2
and 0.2) reduce to one record: mean fitness 3.0, RMS-pooled std sqrt((0.01 + 0.04 +
0.04) / 3) = sqrt(0.03) = 0.1732050808, a ``MeanDeletionPerturbation`` with
``num_duplicates`` 3, and the dataset names joined in sorted order ``a+b+c``. The
reference is pooled the same way (1.0, 1.0, 1.0 -> 1.0). The grouping key is the
experiment type plus the sorted perturbed gene names, so gene order never splits a group
and a different type always does. Gene-interaction duplicates get a one-sample t-test
p-value of the scores against zero; the vector families (metabolite here) merge every
per-key dict elementwise through ``_create_mean_vector_entry``.
"""

import math
from typing import Any

import pytest

from torchcell.data.mean_experiment_deduplicate import (
    MeanExperimentDeduplicator,
    _mean_float_dict,
    _rms_pool_float_dict,
    _sum_int_dict,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeanDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ReferenceGenome,
    Temperature,
)

ENVIRONMENT = Environment(
    media=Media(name="YPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30.0),
)
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _deletion(gene: str) -> KanMxDeletionPerturbation:
    return KanMxDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=gene
    )


def _fitness(
    dataset: str, genes: list[str], fitness: float, std: float
) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_std=std),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=0.05),
        ),
    }


def _interaction(
    dataset: str, genes: list[str], score: float, p: float
) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion(g) for g in genes]),
            environment=ENVIRONMENT,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=score, gene_interaction_p_value=p
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


def _dedup(tmp_path: Any) -> MeanExperimentDeduplicator:
    return MeanExperimentDeduplicator(root=str(tmp_path))


def test_dict_helpers_mean_rms_and_sum_over_the_key_union() -> None:
    """mean: (1 + 3) / 2 and a lone 5; rms: sqrt((0.09 + 0.16) / 2) = 0.3535533906; sum: 2 + 3."""
    assert _mean_float_dict([{"a": 1.0, "b": 5.0}, {"a": 3.0}]) == {"a": 2.0, "b": 5.0}
    pooled = _rms_pool_float_dict([{"a": 0.3}, None, {"a": 0.4}])
    assert pooled is not None
    assert pooled["a"] == pytest.approx(math.sqrt((0.09 + 0.16) / 2))
    assert _rms_pool_float_dict([None, None]) is None
    assert _sum_int_dict([{"a": 2, "b": 1}, {"a": 3}]) == {"a": 5, "b": 1}


def test_duplicate_check_groups_by_type_and_sorted_genes(tmp_path: Any) -> None:
    """Gene order does not split a group; the experiment type does."""
    data = [
        _fitness("a", ["YAL001C", "YAL002W"], 1.0, 0.1),
        _fitness("b", ["YAL002W", "YAL001C"], 2.0, 0.2),
        _fitness("c", ["YAL003W"], 0.5, 0.1),
        _interaction("d", ["YAL001C", "YAL002W"], 0.1, 0.5),
    ]
    groups = _dedup(tmp_path).duplicate_check(data)
    assert sorted(groups.values()) == [[0, 1], [2], [3]]


def test_duplicate_key_on_raw_json_matches_the_pydantic_grouping(tmp_path: Any) -> None:
    """The streaming key computed off model_dump() equals the duplicate_check key."""
    dedup = _dedup(tmp_path)
    record = _fitness("a", ["YAL002W", "YAL001C"], 1.0, 0.1)
    raw = {k: v.model_dump() for k, v in record.items()}
    (key,) = dedup.duplicate_check([record])
    assert dedup.duplicate_key(raw) == key
    assert len(key) == 64  # sha256 hex digest


def test_fitness_duplicates_reduce_to_the_mean_with_pooled_std(tmp_path: Any) -> None:
    """1, 2, 6 -> 3.0; stds 0.1, 0.2, 0.2 -> sqrt(0.03); three duplicates recorded."""
    data = [
        _fitness("b", ["YAL001C"], 1.0, 0.1),
        _fitness("c", ["YAL001C"], 2.0, 0.2),
        _fitness("a", ["YAL001C"], 6.0, 0.2),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    experiment = entry["experiment"]
    assert isinstance(experiment, FitnessExperiment)
    assert experiment.phenotype.fitness == pytest.approx(3.0)
    assert experiment.phenotype.fitness_std == pytest.approx(math.sqrt(0.03))
    assert experiment.dataset_name == "a+b+c"
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, MeanDeletionPerturbation)
    assert perturbation.num_duplicates == 3
    assert perturbation.systematic_gene_name == "YAL001C"
    assert experiment.environment == ENVIRONMENT
    reference = entry["experiment_reference"]
    assert reference.phenotype_reference.fitness == pytest.approx(1.0)
    assert reference.phenotype_reference.fitness_std == pytest.approx(0.05)
    assert reference.dataset_name == "a+b+c"
    assert reference.genome_reference == GENOME


def test_gene_interaction_duplicates_average_the_score(tmp_path: Any) -> None:
    """0.1 and 0.3 -> 0.2 with the source arity kept. The merged p-value is a one-sample
    t-test of the scores against zero: mean 0.2, sample sd 0.1414213562, sem 0.1, t = 2,
    df = 1, so 2 * t.sf(2, 1) = 0.2951672353. The records' own p-values (0.5, 0.5) do not
    enter that number; ``_compute_p_value_for_mean`` reads them only for a length check.
    """
    data = [
        _interaction("x", ["YAL001C", "YAL002W"], 0.1, 0.5),
        _interaction("y", ["YAL001C", "YAL002W"], 0.3, 0.5),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    experiment = entry["experiment"]
    assert isinstance(experiment, GeneInteractionExperiment)
    assert experiment.phenotype.gene_interaction == pytest.approx(0.2)
    assert experiment.phenotype.gene_interaction_p_value == pytest.approx(
        0.2951672353, abs=1e-9
    )
    assert experiment.phenotype.graph_level == "hyperedge"
    assert experiment.dataset_name == "x+y"
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    assert all(isinstance(p, MeanDeletionPerturbation) for p in genotype.perturbations)
    assert [getattr(p, "num_duplicates") for p in genotype.perturbations] == [2, 2]
    assert entry["experiment_reference"].phenotype_reference.gene_interaction == 0.0


def test_gene_interaction_merge_refuses_a_record_without_a_p_value(
    tmp_path: Any,
) -> None:
    """One record with p = None leaves fewer p-values than scores; the merge is refused."""
    data = [
        _interaction("x", ["YAL001C", "YAL002W"], 0.1, 0.5),
        _interaction("y", ["YAL001C", "YAL002W"], 0.3, 0.5),
    ]
    data[1]["experiment"] = data[1]["experiment"].model_copy(
        update={
            "phenotype": GeneInteractionPhenotype(
                gene_interaction=0.3, gene_interaction_p_value=None
            )
        }
    )
    with pytest.raises(ValueError, match="x and p_values must have the same length"):
        _dedup(tmp_path).create_deduplicate_entry(data)


def _metabolite(
    dataset: str, level: float, se: float, n: int, target: dict[str, str] | None = None
) -> dict[str, Any]:
    phenotype = MetabolitePhenotype(
        metabolite_level={"betaxanthin": level},
        metabolite_level_se={"betaxanthin": se},
        n_replicates={"betaxanthin": n},
        measurement_type="cri_spa_corrected_fluorescence_intensity",
        target_metabolite_ids=target,
    )
    reference_phenotype = MetabolitePhenotype(
        metabolite_level={"betaxanthin": 1.0},
        n_replicates={"betaxanthin": 1},
        measurement_type="cri_spa_corrected_fluorescence_intensity",
    )
    return {
        "experiment": MetaboliteExperiment(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion("YAL001C")]),
            environment=ENVIRONMENT,
            phenotype=phenotype,
        ),
        "experiment_reference": MetaboliteExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=reference_phenotype,
        ),
    }


def test_metabolite_duplicates_merge_elementwise_with_summed_replicates(
    tmp_path: Any,
) -> None:
    """The vector family: levels 1 and 3 -> 2.0, se sqrt((0.09 + 0.16) / 2) = 0.3535533906,
    replicates 2 + 3 = 5, measurement type from the first record, and the target-id map
    from the first record that carries one (here the second).
    """
    data = [
        _metabolite("cachera", 1.0, 0.3, 2),
        _metabolite("other", 3.0, 0.4, 3, target={"betaxanthin": "s_9999"}),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    experiment = entry["experiment"]
    assert isinstance(experiment, MetaboliteExperiment)
    phenotype = experiment.phenotype
    assert phenotype.metabolite_level == {"betaxanthin": pytest.approx(2.0)}
    assert phenotype.metabolite_level_se is not None
    assert phenotype.metabolite_level_se["betaxanthin"] == pytest.approx(
        math.sqrt((0.09 + 0.16) / 2)
    )
    assert phenotype.n_replicates == {"betaxanthin": 5}
    assert phenotype.measurement_type == "cri_spa_corrected_fluorescence_intensity"
    assert phenotype.target_metabolite_ids == {"betaxanthin": "s_9999"}
    assert experiment.dataset_name == "cachera+other"
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, MeanDeletionPerturbation)
    assert perturbation.num_duplicates == 2
    reference = entry["experiment_reference"]
    assert isinstance(reference, MetaboliteExperimentReference)
    assert reference.phenotype_reference.metabolite_level == {"betaxanthin": 1.0}
    assert reference.phenotype_reference.n_replicates == {"betaxanthin": 2}
    assert reference.phenotype_reference.metabolite_level_se is None


def test_unsupported_experiment_type_raises(tmp_path: Any) -> None:
    """A type outside fitness, gene interaction and the vector families is refused."""
    record = _fitness("a", ["YAL001C"], 1.0, 0.1)
    record["experiment"] = record["experiment"].model_copy(
        update={"experiment_type": "growth"}
    )
    with pytest.raises(ValueError, match="Unsupported experiment type: growth"):
        _dedup(tmp_path).create_deduplicate_entry([record])

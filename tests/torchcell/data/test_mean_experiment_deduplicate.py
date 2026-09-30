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

2026.09.30 (Phase 15): every vector family on two or three hand-built duplicates, each
value worked in closed form in its test. Values are averaged over the key union (a key
present in one record keeps that record's value), SE and std dicts are RMS-pooled over
the records that carry them (all absent stays None), replicate and read counts are
summed, the CalMorph CV is a plain mean, the visual score takes the minimum of the
present minima and the first present text and comment map, and scalar metadata comes
from the first record. Context is kept from the FIRST record in input order
(environment, genome, environment reference) while dataset names are sorted. Three
digenic interactions (0.1, 0.2, 0.6) give t = sqrt(27 / 7) on 2 df and a two-sided p of
1 - sqrt(27 / 41) = 0.1884973288, with the ``edge`` level kept. Two Findings are pinned:
the microarray variance is RMS-pooled like a std, and two valid microarray records whose
SE is present on fewer genes than their values merge to an invalid phenotype.
"""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.data.mean_experiment_deduplicate import (
    MeanExperimentDeduplicator,
    _mean_float_dict,
    _rms_pool_float_dict,
    _sum_int_dict,
)
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
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
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
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


# --- 2026.09.30 (Phase 15): the remaining families, the null paths and the refusals -- #
def _vector_record(
    experiment_cls: type[Any],
    reference_cls: type[Any],
    dataset: str,
    phenotype: Any,
    reference_phenotype: Any | None = None,
) -> dict[str, Any]:
    """One record of a vector family; the reference phenotype defaults to the record's."""
    return {
        "experiment": experiment_cls(
            dataset_name=dataset,
            genotype=Genotype(perturbations=[_deletion("YAL001C")]),
            environment=ENVIRONMENT,
            phenotype=phenotype,
        ),
        "experiment_reference": reference_cls(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=ENVIRONMENT,
            phenotype_reference=reference_phenotype or phenotype,
        ),
    }


def _microarray(
    log2: dict[str, float],
    se: dict[str, float] | None,
    expression: dict[str, float],
    variance: dict[str, float] | None,
    n: dict[str, int],
) -> MicroarrayExpressionPhenotype:
    return MicroarrayExpressionPhenotype(
        expression_log2_ratio=log2,
        expression_log2_ratio_se=se,
        expression=expression,
        expression_log2_ratio_variance=variance,
        n_replicates=n,
    )


def test_microarray_duplicates_merge_per_gene_over_the_key_union(tmp_path: Any) -> None:
    """Record 1: log2 {A: 1, B: -1}, se {A: 0.3, B: 0.2}, expression {A: 100, B: 50},
    variance {A: 0.5, B: 0.1}, n {A: 2, B: 2}. Record 2: log2 {A: 3}, se {A: 0.4},
    expression {A: 300}, no variance, n {A: 3}. Merged: log2 {A: (1 + 3) / 2 = 2, B: -1
    (one value)}, se {A: sqrt((0.09 + 0.16) / 2) = 0.3535533906, B: 0.2}, expression
    {A: 200, B: 50}, variance from the one record that has it, n {A: 5, B: 2}; the
    reference is merged the same way.
    """
    data = [
        _vector_record(
            MicroarrayExpressionExperiment,
            MicroarrayExpressionExperimentReference,
            "kemmeren",
            _microarray(
                {"A": 1.0, "B": -1.0},
                {"A": 0.3, "B": 0.2},
                {"A": 100.0, "B": 50.0},
                {"A": 0.5, "B": 0.1},
                {"A": 2, "B": 2},
            ),
        ),
        _vector_record(
            MicroarrayExpressionExperiment,
            MicroarrayExpressionExperimentReference,
            "sameith",
            _microarray({"A": 3.0}, {"A": 0.4}, {"A": 300.0}, None, {"A": 3}),
        ),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    experiment = entry["experiment"]
    assert type(experiment) is MicroarrayExpressionExperiment
    phenotype = experiment.phenotype
    assert dict(phenotype.expression_log2_ratio) == {"A": 2.0, "B": -1.0}
    assert phenotype.expression_log2_ratio_se == pytest.approx(
        {"A": math.sqrt((0.09 + 0.16) / 2), "B": 0.2}
    )
    assert dict(phenotype.expression) == {"A": 200.0, "B": 50.0}
    assert phenotype.expression_log2_ratio_variance == {"A": 0.5, "B": 0.1}
    assert dict(phenotype.n_replicates) == {"A": 5, "B": 2}
    assert experiment.dataset_name == "kemmeren+sameith"
    reference = entry["experiment_reference"]
    assert type(reference) is MicroarrayExpressionExperimentReference
    assert reference.phenotype_reference == phenotype
    assert reference.dataset_name == "kemmeren+sameith"


def test_microarray_variance_is_rms_pooled_like_a_standard_deviation(
    tmp_path: Any,
) -> None:
    """Finding: ``expression_log2_ratio_variance`` goes through ``_rms_pool_float_dict``
    (mean_experiment_deduplicate.py:224-226), the pooling written for standard deviations,
    so variances 0.5 and 0.1 merge to sqrt((0.25 + 0.01) / 2) = 0.3605551275 rather than
    their mean 0.3 (a variance pools by averaging, which is what RMS-pooling the std does).
    With every SE absent the merged SE stays None. Pinned until variances are averaged.
    """
    phenotypes = [
        _microarray({"A": 0.0}, None, {"A": 1.0}, {"A": 0.5}, {"A": 1}),
        _microarray({"A": 0.0}, None, {"A": 1.0}, {"A": 0.1}, {"A": 1}),
    ]
    merged = _dedup(tmp_path)._merge_phenotype(list(phenotypes))
    assert isinstance(merged, MicroarrayExpressionPhenotype)
    assert merged.expression_log2_ratio_variance == pytest.approx(
        {"A": math.sqrt((0.25 + 0.01) / 2)}
    )
    assert merged.expression_log2_ratio_variance != pytest.approx({"A": 0.3})
    assert merged.expression_log2_ratio_se is None


def test_microarray_merge_of_two_valid_records_can_fail_validation(
    tmp_path: Any,
) -> None:
    """Finding: the value dict is merged over the key UNION of every record while the SE
    dict is merged only over the records that carry one (mean_experiment_deduplicate.py:
    217-222), so record 1 (genes A and B, no SE) plus record 2 (gene A, SE {A: 0.4})
    merges to log2 keys {A, B} and SE keys {A}, which the phenotype's own validator
    rejects: two valid duplicates make the merge raise. Pinned until the SE is either
    dropped or completed on the union.
    """
    phenotypes = [
        _microarray(
            {"A": 1.0, "B": -1.0}, None, {"A": 1.0, "B": 1.0}, None, {"A": 1, "B": 1}
        ),
        _microarray({"A": 3.0}, {"A": 0.4}, {"A": 1.0}, None, {"A": 1}),
    ]
    with pytest.raises(
        ValidationError,
        match="expression_log2_ratio_se must have the same keys as expression_log2_ratio",
    ):
        _dedup(tmp_path)._merge_phenotype(list(phenotypes))


def test_rnaseq_duplicates_average_tpm_and_sum_counts_and_mapped_reads(
    tmp_path: Any,
) -> None:
    """TPM {A: 10, B: 20} and {A: 30, B: 40} -> {A: 20, B: 30}; counts {A: 100, B: 200}
    and {A: 300, B: 400} -> {A: 400, B: 600}; mapped reads 1,000,000 plus a missing value
    -> 1,000,000 (the None is skipped, not counted as 0); the measurement type comes from
    the first record. Two missing mapped-read values stay None.
    """
    first = RNASeqExpressionPhenotype(
        expression_tpm={"A": 10.0, "B": 20.0},
        expression_count={"A": 100, "B": 200},
        measurement_type="rnaseq_tpm_batch1",
        n_mapped_reads=1_000_000,
    )
    second = RNASeqExpressionPhenotype(
        expression_tpm={"A": 30.0, "B": 40.0},
        expression_count={"A": 300, "B": 400},
        measurement_type="rnaseq_tpm_batch2",
    )
    data = [
        _vector_record(
            RNASeqExpressionExperiment, RNASeqExpressionExperimentReference, "c1", first
        ),
        _vector_record(
            RNASeqExpressionExperiment,
            RNASeqExpressionExperimentReference,
            "c2",
            second,
        ),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    phenotype = entry["experiment"].phenotype
    assert type(entry["experiment"]) is RNASeqExpressionExperiment
    assert dict(phenotype.expression_tpm) == {"A": 20.0, "B": 30.0}
    assert dict(phenotype.expression_count) == {"A": 400, "B": 600}
    assert phenotype.n_mapped_reads == 1_000_000
    assert phenotype.measurement_type == "rnaseq_tpm_batch1"
    no_reads = _dedup(tmp_path)._merge_phenotype([second, second])
    assert isinstance(no_reads, RNASeqExpressionPhenotype)
    assert no_reads.n_mapped_reads is None


def test_calmorph_duplicates_average_features_and_the_present_cv(tmp_path: Any) -> None:
    """Calmorph {C101_A1B: 1, C101_C: 4} and {C101_A1B: 3} -> {C101_A1B: 2, C101_C: 4};
    the coefficient of variation is the plain MEAN over the records that carry one (only
    the second here, {CCV101_A1B: 0.2}; CVs 0.1 and 0.3 average to 0.2 where an RMS pool
    would give sqrt(0.05) = 0.2236067977); two records without one stay None.
    """
    first = CalMorphPhenotype(calmorph={"C101_A1B": 1.0, "C101_C": 4.0})
    second = CalMorphPhenotype(
        calmorph={"C101_A1B": 3.0},
        calmorph_coefficient_of_variation={"CCV101_A1B": 0.2},
    )
    data = [
        _vector_record(CalMorphExperiment, CalMorphExperimentReference, "ohya", first),
        _vector_record(CalMorphExperiment, CalMorphExperimentReference, "x", second),
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    phenotype = entry["experiment"].phenotype
    assert type(entry["experiment"]) is CalMorphExperiment
    assert dict(phenotype.calmorph) == {"C101_A1B": 2.0, "C101_C": 4.0}
    assert phenotype.calmorph_coefficient_of_variation == {"CCV101_A1B": 0.2}
    cv_pair = _dedup(tmp_path)._merge_phenotype(
        [
            CalMorphPhenotype(
                calmorph={"C101_A1B": 1.0},
                calmorph_coefficient_of_variation={"CCV101_A1B": 0.1},
            ),
            CalMorphPhenotype(
                calmorph={"C101_A1B": 1.0},
                calmorph_coefficient_of_variation={"CCV101_A1B": 0.3},
            ),
        ]
    )
    assert isinstance(cv_pair, CalMorphPhenotype)
    assert cv_pair.calmorph_coefficient_of_variation == pytest.approx(
        {"CCV101_A1B": 0.2}
    )
    bare = _dedup(tmp_path)._merge_phenotype([first, first])
    assert isinstance(bare, CalMorphPhenotype)
    assert bare.calmorph_coefficient_of_variation is None


def test_protein_abundance_three_duplicates_in_closed_form(tmp_path: Any) -> None:
    """Abundance 1, 2, 6 -> 3.0; SE 0.1, 0.2, 0.2 -> sqrt((0.01 + 0.04 + 0.04) / 3) =
    sqrt(0.03) = 0.1732050808; replicates 2 + 2 + 2 = 6; ``num_duplicates`` 3.
    """
    data = [
        _vector_record(
            ProteinAbundanceExperiment,
            ProteinAbundanceExperimentReference,
            name,
            ProteinAbundancePhenotype(
                protein_abundance={"P1": level},
                protein_abundance_se={"P1": se},
                n_replicates={"P1": 2},
                measurement_type="dia_ms_log2",
            ),
        )
        for name, level, se in (("m3", 1.0, 0.1), ("m1", 2.0, 0.2), ("m2", 6.0, 0.2))
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    experiment = entry["experiment"]
    assert type(experiment) is ProteinAbundanceExperiment
    phenotype = experiment.phenotype
    assert phenotype.protein_abundance == {"P1": pytest.approx(3.0)}
    assert phenotype.protein_abundance_se == pytest.approx({"P1": math.sqrt(0.03)})
    assert phenotype.n_replicates == {"P1": 6}
    assert phenotype.measurement_type == "dia_ms_log2"
    assert experiment.dataset_name == "m1+m2+m3"
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    assert [getattr(p, "num_duplicates") for p in genotype.perturbations] == [3]


def _visual(
    score: float,
    score_min: float | None,
    n: int,
    text: str | None,
    comments: dict[str, bool] | None,
    semantics: str,
) -> VisualScorePhenotype:
    return VisualScorePhenotype(
        visual_score=score,
        visual_score_min=score_min,
        n_replicates=n,
        score_scale_min=-5,
        score_scale_max=5,
        score_semantics=semantics,
        target_product="beta-carotene",
        score_text=text,
        comment_annotations=comments,
    )


def test_visual_score_duplicates_mean_min_sum_and_first_present_annotations(
    tmp_path: Any,
) -> None:
    """Scores 1, 2, 4 -> 7/3 = 2.3333333333; the minimum over the present minima (None,
    1.0, 3.0) is 1.0; replicates 1 + 2 + 3 = 6; the first non-None score text ("pet");
    the first truthy comment map (the empty map of record 2 is skipped); the scale and
    semantics from record 1.
    """
    phenotypes = [
        _visual(1.0, None, 1, None, None, "first semantics"),
        _visual(2.0, 1.0, 2, "pet", {}, "second semantics"),
        _visual(4.0, 3.0, 3, "tiny", {"flag_petite": True}, "third semantics"),
    ]
    data = [
        _vector_record(VisualScoreExperiment, VisualScoreExperimentReference, n, p)
        for n, p in zip(("o1", "o2", "o3"), phenotypes)
    ]
    entry = _dedup(tmp_path).create_deduplicate_entry(data)
    phenotype = entry["experiment"].phenotype
    assert type(entry["experiment"]) is VisualScoreExperiment
    assert phenotype.visual_score == pytest.approx(7 / 3)
    assert phenotype.visual_score_min == 1.0
    assert phenotype.n_replicates == 6
    assert phenotype.score_text == "pet"
    assert phenotype.comment_annotations == {"flag_petite": True}
    assert (phenotype.score_scale_min, phenotype.score_scale_max) == (-5, 5)
    assert phenotype.score_semantics == "first semantics"
    assert phenotype.target_product == "beta-carotene"
    single_rep = _dedup(tmp_path)._merge_phenotype([phenotypes[0], phenotypes[0]])
    assert isinstance(single_rep, VisualScorePhenotype)
    assert (single_rep.visual_score_min, single_rep.score_text) == (None, None)


def test_merge_phenotype_refuses_a_phenotype_outside_the_vector_families(
    tmp_path: Any,
) -> None:
    """A fitness phenotype handed to the vector merge is refused by name."""
    with pytest.raises(ValueError) as excinfo:
        _dedup(tmp_path)._merge_phenotype([FitnessPhenotype(fitness=1.0)])
    assert str(excinfo.value) == (
        "Unsupported phenotype type for mean merge: FitnessPhenotype"
    )


def test_fitness_merge_skips_missing_stds_and_keeps_the_first_records_context(
    tmp_path: Any,
) -> None:
    """Stds 0.3 and None pool to 0.3 (the None is dropped, not read as 0, which would give
    sqrt(0.09 / 2) = 0.2121320344); two None stds stay None. Environment, genome and
    environment reference come from the FIRST record in input order ("b", 30 C), while
    the dataset names are sorted ("a+b").
    """
    warm = _fitness("b", ["YAL001C"], 1.0, 0.3)
    cool_environment = Environment(
        media=Media(name="YPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=26.0),
    )
    other_genome = ReferenceGenome(species="Saccharomyces cerevisiae", strain="W303")
    cool = {
        "experiment": FitnessExperiment(
            dataset_name="a",
            genotype=Genotype(perturbations=[_deletion("YAL001C")]),
            environment=cool_environment,
            phenotype=FitnessPhenotype(fitness=3.0, fitness_std=None),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name="a",
            genome_reference=other_genome,
            environment_reference=cool_environment,
            phenotype_reference=FitnessPhenotype(fitness=1.0, fitness_std=None),
        ),
    }
    entry = _dedup(tmp_path).create_deduplicate_entry([warm, cool])
    experiment = entry["experiment"]
    assert experiment.phenotype.fitness == pytest.approx(2.0)
    assert experiment.phenotype.fitness_std == pytest.approx(0.3)
    assert experiment.environment == ENVIRONMENT
    assert experiment.dataset_name == "a+b"
    reference = entry["experiment_reference"]
    assert reference.genome_reference == GENOME
    assert reference.environment_reference == ENVIRONMENT
    assert reference.phenotype_reference.fitness_std == pytest.approx(0.05)
    no_std = _dedup(tmp_path).create_deduplicate_entry([cool, cool])
    assert no_std["experiment"].phenotype.fitness_std is None
    assert no_std["experiment_reference"].phenotype_reference.fitness_std is None


def test_three_digenic_interactions_keep_the_edge_level_and_t_test_p_value(
    tmp_path: Any,
) -> None:
    """Scores 0.1, 0.2, 0.6: mean 0.3, sample sd sqrt(0.07), sem sqrt(0.07 / 3), t =
    0.3 / sem = sqrt(27 / 7) = 1.9639610121 on 2 df. For 2 df the two-sided tail is
    1 - t / sqrt(2 + t^2) = 1 - sqrt(27 / 41) = 0.1884973288. The digenic ``edge`` level
    of the records survives on the experiment and the reference.
    """
    data = []
    for name, score in (("c", 0.1), ("a", 0.2), ("b", 0.6)):
        record = _interaction(name, ["YAL001C", "YAL002W"], score, 0.01)
        record["experiment"] = record["experiment"].model_copy(
            update={
                "phenotype": GeneInteractionPhenotype(
                    gene_interaction=score,
                    gene_interaction_p_value=0.01,
                    graph_level="edge",
                )
            }
        )
        record["experiment_reference"] = record["experiment_reference"].model_copy(
            update={
                "phenotype_reference": GeneInteractionPhenotype(
                    gene_interaction=0.0, graph_level="edge"
                )
            }
        )
        data.append(record)
    entry = MeanExperimentDeduplicator(root=str(tmp_path)).create_deduplicate_entry(
        data
    )
    phenotype = entry["experiment"].phenotype
    assert phenotype.gene_interaction == pytest.approx(0.3)
    assert phenotype.gene_interaction_p_value == pytest.approx(
        1 - math.sqrt(27 / 41), abs=1e-12
    )
    assert phenotype.graph_level == "edge"
    assert entry["experiment"].dataset_name == "a+b+c"
    reference_phenotype = entry["experiment_reference"].phenotype_reference
    assert reference_phenotype.graph_level == "edge"
    assert reference_phenotype.gene_interaction_p_value is None


def test_p_value_for_a_single_score_is_refused(tmp_path: Any) -> None:
    """One score and one p-value pass the length check and fail the n >= 2 check."""
    with pytest.raises(ValueError) as excinfo:
        _dedup(tmp_path)._compute_p_value_for_mean([0.1], [0.5])
    assert str(excinfo.value) == "At least two data points are required."

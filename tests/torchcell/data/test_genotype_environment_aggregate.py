"""GenotypeEnvironmentAggregator keys on the (genotype, environment) cell.

The two things the key must do that the gene-set key does not: keep two conditions of one
gene apart, and keep a heterozygous and a homozygous deletion of one gene apart. And the
two things it must share with every aggregator: the pydantic and the raw-JSON paths agree,
and nothing that is not identity (order, provenance, the measured value) changes the key.
"""

from types import SimpleNamespace

from torchcell.data.genotype_environment_aggregate import GenotypeEnvironmentAggregator
from torchcell.datamodels import schema as s

AGG = GenotypeEnvironmentAggregator(root="/tmp/unused")


def _media() -> s.Media:
    return s.Media(name="YPD", state="liquid", is_synthetic=False)


def _dosed(name: str, inchikey: str, micromolar: float) -> s.SmallMoleculePerturbation:
    return s.SmallMoleculePerturbation(
        compound=s.Compound(name=name, inchikey=inchikey),
        concentration=s.Concentration(
            value=micromolar, unit=s.ConcentrationUnit.micromolar
        ),
    )


def _environment(*perturbations: s.SmallMoleculePerturbation) -> s.Environment:
    return s.Environment(
        media=_media(),
        temperature=s.Temperature(value=30.0),
        perturbations=list(perturbations),
        duration_generations=20.0,
    )


BENOMYL_7 = _dosed("benomyl", "RIOXQFHNBCJBCZ-UHFFFAOYSA-N", 6.9)
BENOMYL_14 = _dosed("benomyl", "RIOXQFHNBCJBCZ-UHFFFAOYSA-N", 13.8)
BENOMYL_7_RENAMED = _dosed("Benomyl (Sigma)", "RIOXQFHNBCJBCZ-UHFFFAOYSA-N", 6.9)

HOM = s.KanMxDeletionPerturbation(
    systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
)
HET = s.EngineeredCopyNumberPerturbation(
    systematic_gene_name="YAL001C",
    perturbed_gene_name="TFC3",
    copy_number=1,
    reference_copy_number=2,
    marker="KanMX",
)
OTHER = s.KanMxDeletionPerturbation(
    systematic_gene_name="YAL002W", perturbed_gene_name="VPS8"
)


def _data(perturbations, environment, ploidy="diploid"):
    """The minimal object shape ``aggregate_check`` reads."""
    return {
        "experiment": SimpleNamespace(
            genotype=SimpleNamespace(perturbations=list(perturbations)),
            environment=environment,
        ),
        "experiment_reference": SimpleNamespace(
            genome_reference=SimpleNamespace(ploidy=ploidy)
        ),
    }


def _record(perturbations, environment, ploidy="diploid"):
    """The stored-JSON shape ``aggregate_key_raw`` reads."""
    return {
        "experiment": {
            "experiment_type": "environment_response",
            "genotype": {"perturbations": [p.model_dump() for p in perturbations]},
            "environment": environment.model_dump(),
        },
        "experiment_reference": {
            "genome_reference": {
                "species": "Saccharomyces cerevisiae",
                "strain": "BY4743",
                "ploidy": ploidy,
            }
        },
    }


def test_two_doses_of_one_compound_are_two_cells() -> None:
    assert AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7))) != (
        AGG.aggregate_check(_data([HOM], _environment(BENOMYL_14)))
    )


def test_two_compounds_are_two_cells() -> None:
    nocodazole = _dosed("nocodazole", "KYRVNWMVYQXFEU-UHFFFAOYSA-N", 6.9)
    assert AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7))) != (
        AGG.aggregate_check(_data([HOM], _environment(nocodazole)))
    )


def test_heterozygous_and_homozygous_deletions_are_two_cells() -> None:
    env = _environment(BENOMYL_7)
    assert AGG.aggregate_check(_data([HOM], env)) != (
        AGG.aggregate_check(_data([HET], env))
    )


def test_ploidy_is_part_of_the_cell() -> None:
    env = _environment(BENOMYL_7)
    assert AGG.aggregate_check(_data([HOM], env, "haploid")) != (
        AGG.aggregate_check(_data([HOM], env, "diploid"))
    )


def test_different_genes_are_two_cells() -> None:
    env = _environment(BENOMYL_7)
    assert AGG.aggregate_check(_data([HOM], env)) != (
        AGG.aggregate_check(_data([OTHER], env))
    )


def test_a_repeated_screen_of_the_same_cell_shares_the_key() -> None:
    """The measured value is the phenotype, not the cell, so it never enters the key."""
    assert AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7))) == (
        AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7)))
    )


def test_a_compound_rename_under_one_inchikey_shares_the_key() -> None:
    """Environment identity is composition, so a name is not part of it."""
    assert AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7))) == (
        AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7_RENAMED)))
    )


def test_perturbation_order_is_not_identity() -> None:
    env = _environment(BENOMYL_7)
    assert AGG.aggregate_check(_data([HOM, OTHER], env)) == (
        AGG.aggregate_check(_data([OTHER, HOM], env))
    )


def test_raw_and_pydantic_paths_agree() -> None:
    env = _environment(BENOMYL_7)
    for perts, ploidy in (
        ([HOM], "diploid"),
        ([HET], "diploid"),
        ([HOM, OTHER], "haploid"),
    ):
        assert AGG.aggregate_key_raw(_record(perts, env, ploidy)) == (
            AGG.aggregate_check(_data(perts, env, ploidy))
        )


def test_an_undosed_environment_is_its_own_cell() -> None:
    """A media-swap record with no added compound must not join a dosed one."""
    assert AGG.aggregate_check(_data([HOM], _environment())) != (
        AGG.aggregate_check(_data([HOM], _environment(BENOMYL_7)))
    )


def _culture_environment(
    *perturbations: s.SmallMoleculePerturbation, pre_culture: bool = True
) -> s.CultureEnvironment:
    """The strain-resolved family's environment: the base fields plus a protocol."""
    return s.CultureEnvironment(
        media=_media(),
        temperature=s.Temperature(value=30.0),
        perturbations=list(perturbations),
        duration_generations=20.0,
        culture_format=s.CultureFormat(vessel="microwell", working_volume_ul=100.0),
        pre_culture=(
            s.PreCulture(source=s.PreCultureSource.frozen_stock, source_label="-5gen")
            if pre_culture
            else None
        ),
    )


def _strain_record(perturbations, environment, ploidy="haploid"):
    """The stored-JSON shape of a strain-resolved record, which names its family."""
    record = _record(perturbations, environment, ploidy)
    record["experiment"]["experiment_type"] = "strain_environment_response"
    return record


def test_raw_path_rebuilds_the_declared_environment_class() -> None:
    """The base ``Environment`` forbids the protocol fields a ``CultureEnvironment``
    carries, so the raw path must construct the class the record's family declares;
    with it the raw and pydantic keys agree on a strain-resolved record.
    """
    env = _culture_environment(BENOMYL_7)
    assert AGG.aggregate_key_raw(_strain_record([HOM], env)) == (
        AGG.aggregate_check(_data([HOM], env, "haploid"))
    )


def test_the_culture_protocol_is_part_of_the_cell() -> None:
    """Two screens of one strain and compound that differ in pre-culture are two cells."""
    assert AGG.aggregate_check(
        _data([HOM], _culture_environment(BENOMYL_7, pre_culture=True), "haploid")
    ) != AGG.aggregate_check(
        _data([HOM], _culture_environment(BENOMYL_7, pre_culture=False), "haploid")
    )

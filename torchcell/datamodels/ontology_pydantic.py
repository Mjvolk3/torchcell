# torchcell/datamodels/ontology_pydantic.py
# [[torchcell.datamodels.ontology_pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/ontology_pydantic.py
# Test file: torchcell/datamodels/test_ontology_pydantic.py

import json
from typing import List, Union

from pydantic import BaseModel, Field, field_validator, root_validator

from torchcell.datamodels.pydant import ModelStrict


# Genotype
class ReferenceGenome(ModelStrict):
    species: str
    strain: str


class SysGeneName(ModelStrict):
    name: str = Field(description="Systematic gene name", min_length=7, max_length=7)


# class ExpressionRangeMultiplier(ModelStrict):
#     min: float
#     max: float


# class DampPerturbation(ModelStrict):
#     description: str = "4-10 decreased expression via KANmx gene replacement"

#     expression_range: ExpressionRangeMultiplier = Field(
#         default=ExpressionRangeMultiplier(min=1 / 4.0, max=1 / 10.0),
#         description="Range of gene expression levels",
#     )


# class TsAllelePerturbation(ModelStrict):
#     allele_name: str
#     # Still now sure how to deal with since this is really defined by the sequences of the new allele. You can think if it as gene substitution.
#     description: str = "Temperature sensitive allele"


class GenePerturbation(ModelStrict):
    sys_gene_name: SysGeneName
    perturbed_gene_name: str


class DeletionPerturbation(GenePerturbation, ModelStrict):
    description: str = "Deletion via KANMX gene replacement"


class BaseGenotype(ModelStrict):
    perturbation: GenePerturbation | list[GenePerturbation] = Field(
        description="Gene perturbation"
    )


# Environment
class Media(ModelStrict):
    name: str
    state: str

    @field_validator("state")
    def validate_state(cls, v):
        if v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v


class Temperature(ModelStrict):
    # in celsius - not sure how to enforce units
    Celsius: float


class BaseEnvironment(ModelStrict):
    media: Media
    temperature: Temperature


# Phenotype


class BasePhenotype(ModelStrict):
    graph_level: str
    label: str
    label_error: str

    @field_validator("graph_level")
    def validate_level(cls, v):
        levels = {"edge", "node", "subgraph", "global", "metabolism"}

        if v not in levels:
            raise ValueError("level must be one of: edge, node, global, metabolism")

        return v


class FitnessPhenotype(BasePhenotype, ModelStrict):
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_std: float = Field(description="fitness standard deviation")


class ExperimentReferenceState(ModelStrict):
    reference_genome: ReferenceGenome
    reference_environment: BaseEnvironment
    reference_phenotype: BasePhenotype


class BaseExperiment(ModelStrict):
    genotype: BaseGenotype
    environment: BaseEnvironment
    phenotype: BasePhenotype


class DeletionGenotype(BaseGenotype, ModelStrict):
    perturbation: DeletionPerturbation | list[DeletionPerturbation]


class FitnessExperiment(BaseExperiment):
    experiment_reference_state: ExperimentReferenceState
    genotype: DeletionGenotype | list[DeletionGenotype]
    phenotype: FitnessPhenotype


if __name__ == "__main__":
    reference_genome = ReferenceGenome(
        species="saccharomyces Cerevisiae", strain="s288c"
    )
    genotype = DeletionGenotype(
        perturbation=DeletionPerturbation(
            sys_gene_name=SysGeneName(name="YAL001C"),
            perturbed_gene_name="YAL001C_damp174",
        )
    )
    environment = BaseEnvironment(
        media=Media(name="YPD", state="solid"), temperature=Temperature(Celsius=30.0)
    )
    reference_environment = environment.model_copy()
    phenotype = FitnessPhenotype(
        graph_level="global",
        label="smf",
        label_error="smf_std",
        fitness=0.94,
        fitness_std=0.10,
    )

    reference_phenotype = FitnessPhenotype(
        graph_level="global",
        label="smf",
        label_error="smf_std",
        fitness=1.0,
        fitness_std=0.03,
    )
    experiment_reference_state = ExperimentReferenceState(
        reference_genome=reference_genome,
        reference_environment=reference_environment,
        reference_phenotype=reference_phenotype,
    )
    experiment = FitnessExperiment(
        experiment_reference_state=experiment_reference_state,
        genotype=genotype,
        environment=environment,
        phenotype=phenotype,
    )

    print(experiment.model_dump_json(indent=2))
    temp_data = json.loads(experiment.model_dump_json())
    FitnessExperiment.model_validate(temp_data)
    print("success")
    print("==================")
    # print(json.dumps(FitnessExperiment.model_json_schema(), indent=2))

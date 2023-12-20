# torchcell/datamodels/ontology_pydantic.py
# [[torchcell.datamodels.ontology_pydantic]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/ontology_pydantic.py
# Test file: torchcell/datamodels/test_ontology_pydantic.py

import json
from typing import List, Union

from pydantic import BaseModel, Field, field_validator, root_validator
from enum import Enum, auto
from torchcell.datamodels.pydant import ModelStrict


# Genotype
class ReferenceGenome(ModelStrict):
    species: str
    strain: str


class SysGeneName(ModelStrict):
    name: str = Field(description="Systematic gene name", min_length=7, max_length=9)


class GenePerturbation(ModelStrict):
    systematic_gene_name: SysGeneName
    perturbed_gene_name: str


class DeletionPerturbation(GenePerturbation, ModelStrict):
    description: str = "Deletion via KANMX gene replacement"
    perturbation_type: str = Field(default="deletion", Literal=True)


class BaseGenotype(ModelStrict):
    perturbation: GenePerturbation | list[GenePerturbation] = Field(
        description="Gene perturbation"
    )

class ExpressionRangeMultiplier(ModelStrict):
    min: float = Field(
        ..., description="Minimum range multiplier of gene expression levels"
    )
    max: float = Field(
        ..., description="Maximum range multiplier of gene expression levels"
    )


class DampPerturbation(GenePerturbation, ModelStrict):
    description: str = "4-10 decreased expression via KANmx insertion at the the 3' UTR of the target gene."
    expression_range: ExpressionRangeMultiplier = Field(
        default=ExpressionRangeMultiplier(min=1 / 10.0, max=1 / 4.0),
        description="Gene expression is descreased by 4-10 fold",
    )
    perturbation_type: str = Field(default="damp", Literal=True)

class TsAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "Temperature sensitive allele compromised by amino acid substitution."
    )
    seq: str = "NOT IMPLEMENTED"
    # TODO add specifics of allele
    # [[2023.12.15|dendron://torchcell/user.Mjvolk3.torchcell.tasks#20231215]] Many of these are unknown.
    perturbation_type: str = Field(default="ts_allele", Literal=True)


# Environment
class Media(ModelStrict):
    name: str
    state: str

    @field_validator("state")
    def validate_state(cls, v):
        if v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v


class Temperature(BaseModel):
    scalar: float
    description: str = "Temperature in degrees Celsius."
    
    @field_validator('scalar')
    def check_temperature(cls, v):
        if v < -273:
            raise ValueError("Temperature cannot be below -273 degrees Celsius")
        return v


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

# TODO when we only do BasePhenotype during serialization, we will lose the other information. It might be good to make refs for each phenotype, 
class ExperimentReference(ModelStrict):
    reference_genome: ReferenceGenome
    reference_environment: BaseEnvironment
    reference_phenotype: BasePhenotype


class BaseExperiment(ModelStrict):
    genotype: BaseGenotype
    environment: BaseEnvironment
    phenotype: BasePhenotype


class DeletionGenotype(BaseGenotype, ModelStrict):
    perturbation: DeletionPerturbation | list[DeletionPerturbation]


class InterferenceGenotype(BaseGenotype, ModelStrict):
    perturbation: DampPerturbation | list[
        DampPerturbation
    ] | TsAllelePerturbation | list[TsAllelePerturbation]


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    reference_phenotype: FitnessPhenotype


class FitnessExperiment(BaseExperiment):
    genotype: DeletionGenotype | list[DeletionGenotype] | InterferenceGenotype | list[
        InterferenceGenotype
    ]
    phenotype: FitnessPhenotype
    

if __name__ == "__main__":
    # Primary Data
    genotype = DeletionGenotype(
        perturbation=DeletionPerturbation(
            systematic_gene_name=SysGeneName(name="YAL001C"),
            perturbed_gene_name="YAL001C_damp174",
        )
    )
    environment = BaseEnvironment(
        media=Media(name="YPD", state="solid"), temperature=Temperature(scalar=30.0)
    )
    phenotype = FitnessPhenotype(
        graph_level="global",
        label="smf",
        label_error="smf_std",
        fitness=0.94,
        fitness_std=0.10,
    )

    # Reference
    reference_genome = ReferenceGenome(
        species="saccharomyces Cerevisiae", strain="s288c"
    )
    reference_environment = environment.model_copy()
    reference_phenotype = FitnessPhenotype(
        graph_level="global",
        label="smf",
        label_error="smf_std",
        fitness=1.0,
        fitness_std=0.03,
    )
    experiment_reference_state = FitnessExperimentReference(
        reference_genome=reference_genome,
        reference_environment=reference_environment,
        reference_phenotype=reference_phenotype,
    )
    
    # Final Experiment
    experiment = FitnessExperiment(
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

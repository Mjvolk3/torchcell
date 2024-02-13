# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py


import json
from typing import List, Union, Optional

from pydantic import BaseModel, Field, field_validator, root_validator
from enum import Enum, auto
from torchcell.datamodels.pydant import ModelStrict


# Genotype
class ReferenceGenome(ModelStrict):
    species: str
    strain: str


class GenePerturbation(ModelStrict):
    systematic_gene_name: str
    perturbed_gene_name: str

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v):
        if len(v) < 7 or len(v) > 9:
            raise ValueError("Systematic gene name must be between 7 and 9 characters")
        return v

    @field_validator("perturbed_gene_name", mode="after")
    @classmethod
    def validate_pert_gene_name(cls, v):
        if v.endswith("'"):
            v = v[:-1] + "_prime"
        return v


class DeletionPerturbation(GenePerturbation, ModelStrict):
    description: str = "Deletion via KanMX or NatMX gene replacement"
    perturbation_type: str = "deletion"


class KanMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    deletion_description: str = "Deletion via KanMX gene replacement."
    deletion_type: str = "KanMX"


class NatMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    deletion_description: str = "Deletion via NatMX gene replacement."
    deletion_type: str = "NatMX"


class SgaKanMxDeletionPerturbation(KanMxDeletionPerturbation, ModelStrict):
    kan_mx_description: str = (
        "KanMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    kanmx_deletion_type: str = "SGA"


class SgaNatMxDeletionPerturbation(NatMxDeletionPerturbation, ModelStrict):
    nat_mx_description: str = (
        "NatMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    natmx_deletion_type: str = "SGA"

    # @classmethod
    # def _process_perturbation_data(cls, perturbation_data):
    #     if isinstance(perturbation_data, list):
    #         return [cls._create_perturbation_from_dict(p) for p in perturbation_data]
    #     elif isinstance(perturbation_data, dict):
    #         return cls._create_perturbation_from_dict(perturbation_data)
    #     return perturbation_data


class ExpressionRangeMultiplier(ModelStrict):
    min: float = Field(
        ..., description="Minimum range multiplier of gene expression levels"
    )
    max: float = Field(
        ..., description="Maximum range multiplier of gene expression levels"
    )


class DampPerturbation(GenePerturbation, ModelStrict):
    description: str = "4-10 decreased expression via KANmx insertion at the "
    "the 3' UTR of the target gene."
    expression_range: ExpressionRangeMultiplier = Field(
        default=ExpressionRangeMultiplier(min=1 / 10.0, max=1 / 4.0),
        description="Gene expression is decreased by 4-10 fold",
    )
    perturbation_type: str = "damp"


class SgaDampPerturbation(DampPerturbation, ModelStrict):
    damp_description: str = "Damp Perturbation information specific to SGA experiments."
    strain_id: str = Field(description="'Strain ID' in raw data.")
    damp_perturbation_type: str = "SGA"


class TsAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "Temperature sensitive allele compromised by amino acid substitution."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: str = "temperature_sensitive_allele"


class AllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "Allele compromised by amino acid substitution without more generic"
        "phenotypic information specified."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: str = "allele"


class SuppressorAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "suppressor allele that results in higher fitness in the presence"
        "of a perturbation, compared to the fitness of the perturbation alone."
    )
    perturbation_type: str = "suppressor_allele"


class SgaSuppressorAllelePerturbation(SuppressorAllelePerturbation, ModelStrict):
    suppressor_description: str = (
        "Suppressor Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    suppressor_allele_perturbation_type: str = "SGA"


class SgaTsAllelePerturbation(TsAllelePerturbation, ModelStrict):
    ts_allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    temperature_sensitive_allele_perturbation_type: str = "SGA"


class SgaAllelePerturbation(AllelePerturbation, ModelStrict):
    allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    allele_perturbation_type: str = "SGA"


SgaPerturbationType = Union[
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaDampPerturbation,
    SgaTsAllelePerturbation,
    SgaSuppressorAllelePerturbation,
    SgaAllelePerturbation,
]

GenePerturbationType = Union[SgaPerturbationType]


class Genotype(ModelStrict):
    perturbations: list[GenePerturbationType] = Field(description="Gene perturbation")

    @field_validator("perturbations", mode="after")
    @classmethod
    def sort_perturbations(cls, perturbations):
        return sorted(
            perturbations,
            key=lambda p: (
                p.systematic_gene_name,
                p.perturbation_type,
                p.perturbed_gene_name,
            ),
        )

    @property
    def systematic_gene_names(self):
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.systematic_gene_name for p in sorted_perturbations]

    @property
    def perturbed_gene_names(self):
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbed_gene_name for p in sorted_perturbations]

    @property
    def perturbation_types(self):
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbation_type for p in sorted_perturbations]

    def __len__(self):
        return len(self.perturbations)

    # we would use set, but need serialization to be a list
    def __eq__(self, other):
        if not isinstance(other, Genotype):
            return NotImplemented

        return set(self.perturbations) == set(other.perturbations)


# Environment
class Media(ModelStrict):
    name: str
    state: str

    @field_validator("state", mode="after")
    @classmethod
    def validate_state(cls, v):
        if v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v


class Temperature(BaseModel):
    value: float  # Renamed from scalar to value
    unit: str = "Celsius"  # Simplified unit string

    @field_validator("value", mode="after") 
    @classmethod
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

    @field_validator("graph_level", mode="after")
    @classmethod
    def validate_level(cls, v):
        levels = {"edge", "node", "subgraph", "global", "metabolism"}

        if v not in levels:
            raise ValueError("level must be one of: edge, node, global, metabolism")

        return v


class FitnessPhenotype(BasePhenotype, ModelStrict):
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_std: Optional[float] = Field(None, description="fitness standard deviation")


# TODO when we only do BasePhenotype during serialization, we will lose the other information. It might be good to make refs for each phenotype,
class ExperimentReference(ModelStrict):
    reference_genome: ReferenceGenome
    reference_environment: BaseEnvironment
    reference_phenotype: BasePhenotype


class BaseExperiment(ModelStrict):
    genotype: Genotype
    environment: BaseEnvironment
    phenotype: BasePhenotype


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    reference_phenotype: FitnessPhenotype


class FitnessExperiment(BaseExperiment):
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: FitnessPhenotype


if __name__ == "__main__":
    pass

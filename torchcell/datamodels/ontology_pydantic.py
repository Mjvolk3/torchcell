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


class GenePerturbation(ModelStrict):
    systematic_gene_name: str
    perturbed_gene_name: str

    @field_validator("systematic_gene_name")
    def validate_sys_gene_name(cls, v):
        if len(v) < 7 or len(v) > 9:
            raise ValueError("Systematic gene name must be between 7 and 9 characters")
        return v

    @field_validator("perturbed_gene_name")
    def validate_pert_gene_name(cls, v):
        if v.endswith("'"):
            v = v[:-1] + "_prime"
        return v


class DeletionPerturbation(GenePerturbation, ModelStrict):
    description: str = "Deletion via KanMX or NatMX gene replacement"
    perturbation_type: str = Field(default="deletion", Literal=True)


class KanMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    description: str = "Deletion via KanMX gene replacement."
    deletion_type: str = "KanMX"


class NatMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    description: str = "Deletion via NatMX gene replacement."
    deletion_type: str = "NatMX"


class SgaKanMxDeletionPerturbation(KanMxDeletionPerturbation, ModelStrict):
    description: str = (
        "KanMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    kanmx_deletion_type: str = "SGA"


class SgaNatMxDeletionPerturbation(NatMxDeletionPerturbation, ModelStrict):
    description: str = (
        "NatMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    natmx_deletion_type: str = "SGA"


class BaseGenotype(ModelStrict):
    perturbation: GenePerturbation | list[GenePerturbation] = Field(
        description="Gene perturbation"
    )

    # Important for disambiguating between single and multiple perturbations
    # and perturbation types
    @root_validator(pre=True)
    def validate_perturbation(cls, values):
        if "perturbation" in values:
            perturbation_data = values["perturbation"]
            validated_perturbations = cls._process_perturbation_data(perturbation_data)
            values["perturbation"] = validated_perturbations
        elif isinstance(values, list):
            # Assuming values is a list of dictionaries, each with a 'perturbation' key
            validated_perturbations = []
            for item in values:
                if "perturbation" in item:
                    perturbation_data = item["perturbation"]
                    validated_perturbation = cls._process_perturbation_data(
                        perturbation_data
                    )
                    validated_perturbations.append(validated_perturbation)
            return validated_perturbations
        return values

    @classmethod
    def _process_perturbation_data(cls, perturbation_data):
        if isinstance(perturbation_data, list):
            return [cls._create_perturbation_from_dict(p) for p in perturbation_data]
        elif isinstance(perturbation_data, dict):
            return cls._create_perturbation_from_dict(perturbation_data)
        return perturbation_data

    @classmethod
    def _create_perturbation_from_dict(cls, perturbation_dict):
        pert_type = perturbation_dict.get("perturbation_type")
        deletion_type = perturbation_dict.get("deletion_type")
        damp_perturbation_type = perturbation_dict.get("damp_perturbation_type")
        temperature_sensitive_allele_perturbation_type = perturbation_dict.get(
            "temperature_sensitive_allele_perturbation_type"
        )
        suppressor_allele_perturbation_type = perturbation_dict.get(
            "suppressor_allele_perturbation_type"
        )

        if pert_type == "deletion":
            if deletion_type == "NatMX":
                return SgaNatMxDeletionPerturbation(**perturbation_dict)
            elif deletion_type == "KanMX":
                return SgaKanMxDeletionPerturbation(**perturbation_dict)
            return DeletionPerturbation(**perturbation_dict)
        elif pert_type == "damp":
            if damp_perturbation_type == "SGA":
                return SgdDampPerturbation(**perturbation_dict)
        elif pert_type == "temperature_sensitive_allele":
            if temperature_sensitive_allele_perturbation_type == "SGA":
                return SgdTsAllelePerturbation(**perturbation_dict)
        elif pert_type == "suppressor_allele":
            if suppressor_allele_perturbation_type == "SGA":
                return SgdSuppressorAllelePerturbation(**perturbation_dict)
        return None


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
    perturbation_type: str = "damp"


class SgdDampPerturbation(DampPerturbation, ModelStrict):
    description: str = "Damp Perturbation information specific to SGA experiments."
    strain_id: str = Field(description="'Strain ID' in raw data.")
    damp_perturbation_type: str = "SGA"


class TsAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "Temperature sensitive allele compromised by amino acid substitution."
    )
    # seq: str = "NOT IMPLEMENTED"
    # TODO add specifics of allele
    # [[2023.12.15|dendron://torchcell/user.Mjvolk3.torchcell.tasks#20231215]] Many of these are unknown.
    perturbation_type: str = "temperature_sensitive_allele"


class SuppressorAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "suppressor allele that results in higher fitness in the presence"
        "of a perturbation, compared to the fitness of the perturbation alone."
    )
    perturbation_type: str = "suppressor_allele"


class SgdSuppressorAllelePerturbation(GenePerturbation, ModelStrict):
    description: str = (
        "Suppressor Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    suppressor_allele_perturbation_type: str = "SGA"


class SgdTsAllelePerturbation(TsAllelePerturbation, ModelStrict):
    description: str = "Ts Allele Perturbation information specific to SGA experiments."
    strain_id: str = Field(description="'Strain ID' in raw data.")
    temperature_sensitive_allele_perturbation_type: str = "SGA"


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
    value: float  # Renamed from scalar to value
    unit: str = "Celsius"  # Simplified unit string

    @field_validator("value")  # Updated to reflect the new attribute name
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
    perturbation: DeletionPerturbation | list[
        DeletionPerturbation
    ] | KanMxDeletionPerturbation | list[
        KanMxDeletionPerturbation
    ] | NatMxDeletionPerturbation | list[
        NatMxDeletionPerturbation
    ] | SgaKanMxDeletionPerturbation | list[
        SgaKanMxDeletionPerturbation
    ] | SgaNatMxDeletionPerturbation | list[
        SgaNatMxDeletionPerturbation
    ]


class InterferenceGenotype(BaseGenotype, ModelStrict):
    perturbation: DampPerturbation | list[
        DampPerturbation
    ] | TsAllelePerturbation | list[TsAllelePerturbation] | SgdDampPerturbation | list[
        SgdDampPerturbation
    ] | SgdTsAllelePerturbation | list[
        SgdTsAllelePerturbation
    ]


class SuppressorGenotype(BaseGenotype, ModelStrict):
    perturbation: SuppressorAllelePerturbation | list[
        SuppressorAllelePerturbation
    ] | SgdSuppressorAllelePerturbation | list[SgdSuppressorAllelePerturbation]


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    reference_phenotype: FitnessPhenotype


class FitnessExperiment(BaseExperiment):
    genotype: Union[
        DeletionGenotype,
        InterferenceGenotype,
        SuppressorGenotype,
        List[DeletionGenotype],
        List[InterferenceGenotype],
        List[SuppressorGenotype],
        List[Union[DeletionGenotype, InterferenceGenotype, SuppressorGenotype]],
    ]
    phenotype: FitnessPhenotype


if __name__ == "__main__":
    # Primary Data
    genotype = DeletionGenotype(
        perturbation=DeletionPerturbation(
            systematic_gene_name="YAL001C", perturbed_gene_name="YAL001C"
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
        genotype=genotype, environment=environment, phenotype=phenotype
    )

    print(experiment.model_dump_json(indent=2))
    temp_data = json.loads(experiment.model_dump_json())
    FitnessExperiment.model_validate(temp_data)
    print("success")
    print("==================")
    # print(json.dumps(FitnessExperiment.model_json_schema(), indent=2))

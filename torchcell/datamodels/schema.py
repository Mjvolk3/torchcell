# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

import re
from typing import List, Union, Dict, Type, Optional, Any
import json
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum, auto
from torchcell.datamodels.pydant import ModelStrict

# causes circular import
# from torchcell.datasets.dataset_registry import dataset_registry


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
        if not re.match(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d+)$", v):
            raise ValueError("Invalid systematic gene name format")
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


# Change to AggregateDeletionPerturbation, or AggDeletionPerturbation
class MeanDeletionPerturbation(DeletionPerturbation, ModelStrict):
    description: str = "Mean deletion perturbation representing duplicate experiments"
    deletion_type: str = "mean"
    num_duplicates: int = Field(
        description="Number of duplicate experiments used to compute the mean and std."
    )


SgaPerturbationType = Union[
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaDampPerturbation,
    SgaTsAllelePerturbation,
    SgaSuppressorAllelePerturbation,
    SgaAllelePerturbation,
]

GenePerturbationType = Union[SgaPerturbationType, MeanDeletionPerturbation]


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


class Environment(ModelStrict):
    media: Media
    temperature: Temperature


# Phenotype
class Phenotype(ModelStrict):
    graph_level: str = Field(
        description="most natural level of graph at which phenotype is observed"
    )
    label_name: str = Field(description="name of label")
    label_statistic_name: Optional[str] = Field(
        default=None,
        description="name of error or confidence statistic related to label",
    )
    label: Any = Field(description="value of the label")
    label_statistic: Optional[Any] = Field(
        default=None,
        description="value of the error or confidence statistic related to label",
    )

    @model_validator(mode="after")
    def validate_fields(self):
        valid_graph_levels = {
            "edge",
            "node",
            "subgraph",
            "global",
            "metabolism",
            "gene ontology",
        }
        if self.graph_level not in valid_graph_levels:
            raise ValueError(
                f"graph_level must be one of: {', '.join(valid_graph_levels)}"
            )
        return self

    def __getitem__(self, key):
        return getattr(self, key)


class FitnessPhenotype(Phenotype, ModelStrict):
    graph_level: str = "global"
    label_name: str = Field(
        default="fitness", description="wt_growth_rate/ko_growth_rate"
    )
    label_statistic_name: str | None = Field(
        default="std", description="fitness standard deviation"
    )

    @field_validator("label")
    def validate_fitness(cls, v):
        if v <= 0:
            raise ValueError("Fitness must be greater than 0")
        return v


class GeneEssentialityPhenotype(Phenotype, ModelStrict):
    gene_essentiality: bool = Field(
        default=True, description="gene knockout leading cell death."
    )


class SyntheticLethalityPhenotype(Phenotype, ModelStrict):
    synthetic_lethality: bool = Field(
        default=True,
        description="synthetic lethality occurs when the combination of mutations in"
        "two or more genes leads to cell death, whereas a mutation in only one of these"
        "genes does not affect the viability of the cell.",
    )
    statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )


class SyntheticRescuePhenotype(Phenotype, ModelStrict):
    synthetic_rescue: bool = Field(
        default=True,
        description="synthetic rescue occurs when a mutation in one gene compensates"
        "for the deleterious effects of a mutation in another gene, thereby restoring"
        "normal function or viability to the cell",
    )
    statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )


class GeneInteractionPhenotype(Phenotype, ModelStrict):
    interaction: float = Field(
        description="""epsilon, tau, or analogous interaction value.
        Computed from composite fitness phenotypes."""
    )
    p_value: float | None = Field(default=None, description="p-value of interaction")


class Publication(ModelStrict):
    pubmed_id: str | None = None
    pubmed_url: str | None = None
    doi: str | None = None
    doi_url: str | None = None

    @model_validator(mode="after")
    def check_pub_info(self):
        if self.pubmed_id is None and self.doi is None:
            raise ValueError("At least one of PubMed ID or DOI must be provided")
        if self.pubmed_url is None and self.doi_url is None:
            raise ValueError("At least one of PubMed URL or DOI URL must be provided")
        return self


class ExperimentReference(ModelStrict):
    experiment_reference_type: str = "base"
    dataset_name: str
    genome_reference: ReferenceGenome
    environment_reference: Environment
    phenotype_reference: Phenotype


class Experiment(ModelStrict):
    experiment_type: str = "base"
    dataset_name: str
    genotype: Genotype
    environment: Environment
    phenotype: Phenotype


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    experiment_reference_type: str = "fitness"
    phenotype_reference: FitnessPhenotype


class FitnessExperiment(Experiment, ModelStrict):
    experiment_type: str = "fitness"
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: FitnessPhenotype


class GeneInteractionExperimentReference(ExperimentReference, ModelStrict):
    experiment_reference_type: str = "gene interaction"
    phenotype_reference: GeneInteractionPhenotype


class GeneInteractionExperiment(Experiment, ModelStrict):
    experiment_type: str = "gene interaction"
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: GeneInteractionPhenotype


class GeneEssentialityExperimentReference(ExperimentReference, ModelStrict):
    experiment_reference_type: str = "gene essentiality"
    phenotype_reference: GeneEssentialityPhenotype


# shouldn't it jut be one gene for genotype?
class GeneEssentialityExperiment(Experiment, ModelStrict):
    experiment_type: str = "gene essentiality"
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: GeneEssentialityPhenotype


class SyntheticLethalityExperimentReference(ExperimentReference, ModelStrict):
    experiment_reference_type: str = "synthetic lethality"
    phenotype_reference: SyntheticLethalityPhenotype


class SyntheticLethalityExperiment(Experiment, ModelStrict):
    experiment_type: str = "synthetic lethality"
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: SyntheticLethalityPhenotype


class SyntheticRescueExperimentReference(ExperimentReference, ModelStrict):
    experiment_reference_type: str = "synthetic rescue"
    phenotype_reference: SyntheticRescuePhenotype


class SyntheticRescueExperiment(Experiment, ModelStrict):
    experiment_type: str = "synthetic rescue"
    genotype: Union[Genotype, List[Genotype,]]
    phenotype: SyntheticRescuePhenotype


ExperimentType = Union[
    Experiment,
    FitnessExperiment,
    GeneInteractionExperiment,
    GeneEssentialityExperiment,
    SyntheticLethalityExperiment,
    SyntheticRescueExperiment,
]

ExperimentReferenceType = Union[
    ExperimentReference,
    FitnessExperimentReference,
    GeneInteractionExperimentReference,
    GeneEssentialityExperimentReference,
    SyntheticLethalityExperimentReference,
    SyntheticRescueExperimentReference,
]


EXPERIMENT_TYPE_MAP = {
    "fitness": FitnessExperiment,
    "gene interaction": GeneInteractionExperiment,
    "gene essentiality": GeneEssentialityExperiment,
    "synthetic lethality": SyntheticLethalityExperiment,
    "synthetic rescue": SyntheticRescueExperiment,
}

EXPERIMENT_REFERENCE_TYPE_MAP = {
    "fitness": FitnessExperimentReference,
    "gene interaction": GeneInteractionExperimentReference,
    "gene essentiality": GeneEssentialityExperimentReference,
    "synthetic lethality": SyntheticLethalityExperimentReference,
    "synthetic rescue": SyntheticRescueExperimentReference,
}


if __name__ == "__main__":
    pass

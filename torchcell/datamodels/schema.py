# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

"""Pydantic data models for torchcell genotypes, environments, and phenotypes."""

import math
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from sortedcontainers import SortedDict

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.pydant import ModelStrict

# causes circular import
# from torchcell.datasets.dataset_registry import dataset_registry


# Genotype
class ReferenceGenome(ModelStrict):
    """Reference genome identified by species and strain."""

    species: str
    strain: str


class GenePerturbation(ModelStrict):
    """Base perturbation of a single gene by systematic and common name."""

    systematic_gene_name: str
    perturbed_gene_name: str

    @field_validator("systematic_gene_name", mode="after")
    @classmethod
    def validate_sys_gene_name(cls, v: str) -> str:
        """Validate the systematic gene name matches an allowed feature pattern."""
        # Define named patterns for clarity based on genome feature types
        # protein-coding genes, pseudogenes, transposable_element_genes
        coding_gene_pattern = r"Y[A-P][LR]\d{3}[WC](-[A-Z])?"
        # mitochondrial genes
        mitochondrial_gene_pattern = r"Q\d{4}"
        # ncRNA_gene, rRNA_gene, snRNA_gene, snoRNA_gene, tRNA_gene, telomerase_RNA_gene
        noncoding_gene_pattern = r"YNC[A-Q]\d{4}[WC]"
        # Combine patterns
        full_pattern = f"^({coding_gene_pattern}|{mitochondrial_gene_pattern}|{noncoding_gene_pattern})$"

        if not re.match(full_pattern, v):
            raise ValueError("Invalid systematic gene name format")
        return v

    @field_validator("perturbed_gene_name", mode="after")
    @classmethod
    def validate_pert_gene_name(cls, v: str) -> str:
        """Normalize a trailing prime in the perturbed gene name to ``_prime``."""
        if v.endswith("'"):
            v = v[:-1] + "_prime"
        return v


class DeletionPerturbation(GenePerturbation, ModelStrict):
    """Gene deletion via KanMX or NatMX gene replacement."""

    description: str = "Deletion via KanMX or NatMX gene replacement"
    perturbation_type: str = "deletion"


class KanMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via KanMX gene replacement."""

    deletion_description: str = "Deletion via KanMX gene replacement."
    deletion_type: str = "KanMX"


class NatMxDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Gene deletion via NatMX gene replacement."""

    deletion_description: str = "Deletion via NatMX gene replacement."
    deletion_type: str = "NatMX"


class SgaKanMxDeletionPerturbation(KanMxDeletionPerturbation, ModelStrict):
    """KanMX deletion perturbation specific to SGA experiments."""

    kan_mx_description: str = (
        "KanMX Deletion Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    kanmx_deletion_type: str = "SGA"


class SgaNatMxDeletionPerturbation(NatMxDeletionPerturbation, ModelStrict):
    """NatMX deletion perturbation specific to SGA experiments."""

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
    """Min/max multiplier bounds on a gene's expression level."""

    min: float = Field(
        ..., description="Minimum range multiplier of gene expression levels"
    )
    max: float = Field(
        ..., description="Maximum range multiplier of gene expression levels"
    )


class DampPerturbation(GenePerturbation, ModelStrict):
    """Decreased-abundance-by-mRNA-perturbation (DAmP) allele perturbation."""

    description: str = "4-10 decreased expression via KANmx insertion at the "
    "the 3' UTR of the target gene."
    expression_range: ExpressionRangeMultiplier = Field(
        default=ExpressionRangeMultiplier(min=1 / 10.0, max=1 / 4.0),
        description="Gene expression is decreased by 4-10 fold",
    )
    perturbation_type: str = "damp"


class SgaDampPerturbation(DampPerturbation, ModelStrict):
    """DAmP perturbation specific to SGA experiments."""

    damp_description: str = "Damp Perturbation information specific to SGA experiments."
    strain_id: str = Field(description="'Strain ID' in raw data.")
    damp_perturbation_type: str = "SGA"


class TsAllelePerturbation(GenePerturbation, ModelStrict):
    """Temperature-sensitive allele perturbation via amino acid substitution."""

    description: str = (
        "Temperature sensitive allele compromised by amino acid substitution."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: str = "temperature_sensitive_allele"


class AllelePerturbation(GenePerturbation, ModelStrict):
    """Generic allele perturbation via amino acid substitution."""

    description: str = (
        "Allele compromised by amino acid substitution without more generic "
        "phenotypic information specified."
    )
    # seq: str = "NOT IMPLEMENTED"
    perturbation_type: str = "allele"


class SuppressorAllelePerturbation(GenePerturbation, ModelStrict):
    """Suppressor allele that raises fitness in the presence of a perturbation."""

    description: str = (
        "suppressor allele that results in higher fitness in the presence"
        "of a perturbation, compared to the fitness of the perturbation alone."
    )
    perturbation_type: str = "suppressor_allele"


class SgaSuppressorAllelePerturbation(SuppressorAllelePerturbation, ModelStrict):
    """Suppressor allele perturbation specific to SGA experiments."""

    suppressor_description: str = (
        "Suppressor Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    suppressor_allele_perturbation_type: str = "SGA"


class SgaTsAllelePerturbation(TsAllelePerturbation, ModelStrict):
    """Temperature-sensitive allele perturbation specific to SGA experiments."""

    ts_allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    temperature_sensitive_allele_perturbation_type: str = "SGA"


class SgaAllelePerturbation(AllelePerturbation, ModelStrict):
    """Generic allele perturbation specific to SGA experiments."""

    allele_description: str = (
        "Ts Allele Perturbation information specific to SGA experiments."
    )
    strain_id: str = Field(description="'Strain ID' in raw data.")
    allele_perturbation_type: str = "SGA"


# Change to AggregateDeletionPerturbation, or AggDeletionPerturbation
class MeanDeletionPerturbation(DeletionPerturbation, ModelStrict):
    """Deletion perturbation aggregating duplicate experiments by their mean."""

    description: str = "Mean deletion perturbation representing duplicate experiments"
    deletion_type: str = "mean"
    num_duplicates: int = Field(
        description="Number of duplicate experiments used to compute the mean and std."
    )


SgaPerturbationType = (
    SgaKanMxDeletionPerturbation
    | SgaNatMxDeletionPerturbation
    | SgaDampPerturbation
    | SgaTsAllelePerturbation
    | SgaSuppressorAllelePerturbation
    | SgaAllelePerturbation
)

GenePerturbationType = (
    SgaPerturbationType
    | MeanDeletionPerturbation
    | KanMxDeletionPerturbation
    | NatMxDeletionPerturbation
)


class Genotype(ModelStrict):
    """Collection of gene perturbations defining a strain's genotype."""

    perturbations: list[GenePerturbationType] = Field(description="Gene perturbation")

    @field_validator("perturbations", mode="after")
    @classmethod
    def sort_perturbations(
        cls, perturbations: list[GenePerturbationType]
    ) -> list[GenePerturbationType]:
        """Sort perturbations by gene name, type, and perturbed name for stable order."""
        return sorted(
            perturbations,
            key=lambda p: (
                p.systematic_gene_name,
                p.perturbation_type,
                p.perturbed_gene_name,
            ),
        )

    @property
    def systematic_gene_names(self) -> list[str]:
        """Return systematic gene names ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.systematic_gene_name for p in sorted_perturbations]

    @property
    def perturbed_gene_names(self) -> list[str]:
        """Return perturbed gene names ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbed_gene_name for p in sorted_perturbations]

    @property
    def perturbation_types(self) -> list[str]:
        """Return perturbation types ordered by systematic gene name."""
        sorted_perturbations = sorted(
            self.perturbations, key=lambda p: p.systematic_gene_name
        )
        return [p.perturbation_type for p in sorted_perturbations]

    def __len__(self) -> int:
        """Return the number of perturbations."""
        return len(self.perturbations)

    # we would use set, but need serialization to be a list
    def __eq__(self, other: object) -> bool:
        """Return True if both genotypes contain the same set of perturbations."""
        if not isinstance(other, Genotype):
            return NotImplemented

        return set(self.perturbations) == set(other.perturbations)


# Environment
class Media(ModelStrict):
    """Growth medium identified by name and physical state."""

    name: str
    state: str

    @field_validator("state", mode="after")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Validate that state is one of solid, liquid, or gas."""
        if v not in ["solid", "liquid", "gas"]:
            raise ValueError('state must be one of "solid", "liquid", or "gas"')
        return v


class Temperature(BaseModel):
    """Temperature value with unit (defaults to Celsius)."""

    value: float  # Renamed from scalar to value
    unit: str = "Celsius"  # Simplified unit string

    @field_validator("value", mode="after")
    @classmethod
    def check_temperature(cls, v: float) -> float:
        """Validate that the temperature is not below absolute zero in Celsius."""
        if v < -273:
            raise ValueError("Temperature cannot be below -273 degrees Celsius")
        return v


class Environment(ModelStrict):
    """Experimental environment combining media and temperature."""

    media: Media
    temperature: Temperature


# Phenotype
class Phenotype(ModelStrict):
    """Base phenotype describing an observed label and its graph level."""

    graph_level: str = Field(
        description="most natural level of graph at which phenotype is observed"
    )
    label_name: str = Field(description="name of label")
    label_statistic_name: str | None = Field(
        default=None,
        description="name of error or confidence statistic related to label",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "Phenotype":
        """Validate that graph_level is one of the supported graph levels."""
        valid_graph_levels = {
            "edge",
            "node",
            "hyperedge",
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

    def __getitem__(self, key: str) -> Any:  # heterogeneous phenotype field values
        """Return the attribute value for the given field name."""
        return getattr(self, key)


class FitnessPhenotype(Phenotype, ModelStrict):
    """Fitness phenotype with fitness value and uncertainty statistics."""

    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_se"
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_se: float | None = Field(
        default=None,
        description="fitness standard error (primary uncertainty statistic)",
    )
    fitness_std: float | None = Field(
        default=None,
        description="fitness standard deviation (raw data from publication)",
    )
    n_samples: int | None = Field(
        default=None,
        description="""Number of replicate measurements of the fitness ratio.
        For experiment: n independent measurements of strain_of_interest/wt.
        For reference: n independent measurements of wt control.
        Note: numerator and denominator may have different sample sizes;
        this tracks the complete ratio measurement.""",
    )

    @field_validator("fitness")
    def validate_fitness(cls, v: float) -> float:
        """Reject NaN fitness and clamp non-positive values to zero."""
        if math.isnan(v):
            raise ValueError("Fitness cannot be NaN")
        if v <= 0:
            return 0.0
        return v

    @field_validator("n_samples")
    def validate_n_samples(cls, v: int | None) -> int | None:
        """Validate that n_samples is a positive integer or None."""
        if v is not None and (not isinstance(v, int) or v < 1):
            raise ValueError(f"n_samples must be a positive integer or None, got: {v}")
        return v

    @model_validator(mode="after")
    def validate_label_fields(cls, values: "FitnessPhenotype") -> "FitnessPhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class GeneEssentialityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating whether a gene knockout is lethal."""

    graph_level: str = "node"
    label_name: str = "is_essential"
    is_essential: bool = Field(
        default=True, description="gene knockout leading cell death."
    )

    # IDEA
    # This is going to be standard for all child classes of Phenotype
    # This could alternatively be moved to testing
    @model_validator(mode="after")
    def validate_label_fields(
        cls, values: "GeneEssentialityPhenotype"
    ) -> "GeneEssentialityPhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        # Check if label_name is a class attribute
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        # Check if label_statistic_name is a class attribute (if it's not None)
        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class SyntheticLethalityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating synthetic lethality between perturbed genes."""

    graph_level: str = "hyperedge"
    label_name: str = "is_synthetic_lethal"
    label_statistic_name: str = "synthetic_lethality_statistic_score"
    is_synthetic_lethal: bool = Field(
        default=True,
        description="synthetic lethality occurs when the combination of mutations in"
        "two or more genes leads to cell death, whereas a mutation in only one of these"
        "genes does not affect the viability of the cell.",
    )
    synthetic_lethality_statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )

    # IDEA
    # This is going to be standard for all child classes of Phenotype
    # This could alternatively be moved to testing
    @model_validator(mode="after")
    def validate_label_fields(
        cls, values: "SyntheticLethalityPhenotype"
    ) -> "SyntheticLethalityPhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        # Check if label_name is a class attribute
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        # Check if label_statistic_name is a class attribute (if it's not None)
        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class SyntheticRescuePhenotype(Phenotype, ModelStrict):
    """Phenotype indicating one perturbation rescues another's deleterious effect."""

    graph_level: str = "hyperedge"
    label_name: str = "is_synthetic_rescue"
    label_statistic_name: str = "synthetic_rescue_statistic_score"
    is_synthetic_rescue: bool = Field(
        default=True,
        description="synthetic rescue occurs when a mutation in one gene compensates"
        "for the deleterious effects of a mutation in another gene, thereby restoring"
        "normal function or viability to the cell",
    )
    synthetic_rescue_statistic_score: float | None = Field(
        default=None,
        description="statistical score computed in [SynLethDB](https://synlethdb.sist.shanghaitech.edu.cn/#/",
    )

    # IDEA
    # This is going to be standard for all child classes of Phenotype
    # This could alternatively be moved to testing
    @model_validator(mode="after")
    def validate_label_fields(
        cls, values: "SyntheticRescuePhenotype"
    ) -> "SyntheticRescuePhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        # Check if label_name is a class attribute
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        # Check if label_statistic_name is a class attribute (if it's not None)
        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class GeneInteractionPhenotype(Phenotype, ModelStrict):
    """Phenotype holding a gene interaction score and its p-value."""

    graph_level: str = "hyperedge"
    label_name: str = "gene_interaction"
    label_statistic_name: str = "gene_interaction_p_value"
    gene_interaction: float = Field(
        description="""epsilon, tau, or analogous gene interaction value.
        Computed from composite fitness phenotypes."""
    )
    gene_interaction_p_value: float | None = Field(
        default=None, description="p-value of gene interaction"
    )

    @field_validator("gene_interaction")
    def validate_fitness(cls, v: float) -> float:
        """Reject NaN gene interaction values."""
        if math.isnan(v):
            raise ValueError("Gene interaction cannot be NaN")
        return v

    # IDEA
    # This is going to be standard for all child classes of Phenotype
    # This could alternatively be moved to testing
    @model_validator(mode="after")
    def validate_label_fields(
        cls, values: "GeneInteractionPhenotype"
    ) -> "GeneInteractionPhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        # Check if label_name is a class attribute
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        # Check if label_statistic_name is a class attribute (if it's not None)
        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class CalMorphPhenotype(Phenotype, ModelStrict):
    """Phenotype holding CalMorph morphological measurements and CV statistics."""

    graph_level: str = "global"
    label_name: str = "calmorph"
    label_statistic_name: str = "calmorph_coefficient_of_variation"
    calmorph: dict[str, float] = Field(
        description="Dictionary of CalMorph base morphological measurements (281 parameters)"
    )
    calmorph_coefficient_of_variation: dict[str, float] | None = Field(
        default=None,
        description="Dictionary of coefficient of variation values for CalMorph parameters (220 parameters)",
    )

    # CALMORPH_PARAMETERS: All 501 parameters from Ohya et al. 2005
    # CALMORPH_LABELS: 281 base morphological measurements
    # CALMORPH_STATISTICS: 220 coefficient of variation parameters

    @field_validator("calmorph")
    def validate_calmorph(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate CalMorph base parameters against CALMORPH_LABELS and reject NaN."""
        if not v:
            raise ValueError("calmorph measurements cannot be empty")
        for key, value in v.items():
            if key not in CALMORPH_LABELS:
                raise ValueError(
                    f"Invalid CalMorph base parameter: {key}. "
                    f"Must be one of the 281 base parameters in CALMORPH_LABELS."
                )
            if math.isnan(value):
                raise ValueError(f"calmorph measurement {key} cannot be NaN")
        return v

    @field_validator("calmorph_coefficient_of_variation")
    def validate_cv(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate CV parameters against CALMORPH_STATISTICS and reject NaN."""
        if v is None:
            return v
        for key, value in v.items():
            if key not in CALMORPH_STATISTICS:
                raise ValueError(
                    f"Invalid CalMorph CV parameter: {key}. "
                    f"Must be one of the 220 CV parameters in CALMORPH_STATISTICS."
                )
            if math.isnan(value):
                raise ValueError(f"CV measurement {key} cannot be NaN")
        return v

    @model_validator(mode="after")
    def validate_label_fields(
        cls, values: "CalMorphPhenotype"
    ) -> "CalMorphPhenotype":
        """Validate that label_name and label_statistic_name are class attributes."""
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values


class Publication(ModelStrict):
    """Publication reference identified by PubMed ID and/or DOI."""

    pubmed_id: str | None = None
    pubmed_url: str | None = None
    doi: str | None = None
    doi_url: str | None = None

    @model_validator(mode="after")
    def check_pub_info(self) -> "Publication":
        """Require at least one of PubMed ID/DOI and at least one URL."""
        if self.pubmed_id is None and self.doi is None:
            raise ValueError("At least one of PubMed ID or DOI must be provided")
        if self.pubmed_url is None and self.doi_url is None:
            raise ValueError("At least one of PubMed URL or DOI URL must be provided")
        return self


class ExperimentReference(ModelStrict):
    """Reference (wildtype/control) context for an experiment."""

    experiment_reference_type: str = "base"
    dataset_name: str
    genome_reference: ReferenceGenome
    environment_reference: Environment
    phenotype_reference: Phenotype


class Experiment(ModelStrict):
    """Base experiment pairing a genotype and environment with a phenotype."""

    experiment_type: str = "base"
    dataset_name: str
    genotype: Genotype
    environment: Environment
    phenotype: Phenotype


class FitnessExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a fitness experiment."""

    experiment_reference_type: str = "fitness"
    phenotype_reference: FitnessPhenotype


class FitnessExperiment(Experiment, ModelStrict):
    """Experiment measuring a fitness phenotype."""

    experiment_type: str = "fitness"
    genotype: Genotype | list[Genotype,]
    phenotype: FitnessPhenotype


class GeneInteractionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene interaction experiment."""

    experiment_reference_type: str = "gene interaction"
    phenotype_reference: GeneInteractionPhenotype


class GeneInteractionExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene interaction phenotype."""

    experiment_type: str = "gene interaction"
    genotype: Genotype | list[Genotype,]
    phenotype: GeneInteractionPhenotype


class GeneEssentialityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene essentiality experiment."""

    experiment_reference_type: str = "gene essentiality"
    phenotype_reference: GeneEssentialityPhenotype


# shouldn't it jut be one gene for genotype?
class GeneEssentialityExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene essentiality phenotype."""

    experiment_type: str = "gene essentiality"
    genotype: Genotype | list[Genotype,]
    phenotype: GeneEssentialityPhenotype


class SyntheticLethalityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic lethality experiment."""

    experiment_reference_type: str = "synthetic lethality"
    phenotype_reference: SyntheticLethalityPhenotype


class SyntheticLethalityExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic lethality phenotype."""

    experiment_type: str = "synthetic lethality"
    genotype: Genotype | list[Genotype,]
    phenotype: SyntheticLethalityPhenotype


class SyntheticRescueExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic rescue experiment."""

    experiment_reference_type: str = "synthetic rescue"
    phenotype_reference: SyntheticRescuePhenotype


class SyntheticRescueExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic rescue phenotype."""

    experiment_type: str = "synthetic rescue"
    genotype: Genotype | list[Genotype,]
    phenotype: SyntheticRescuePhenotype


class CalMorphExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a CalMorph experiment."""

    experiment_reference_type: str = "calmorph"
    phenotype_reference: CalMorphPhenotype


class CalMorphExperiment(Experiment, ModelStrict):
    """Experiment measuring a CalMorph phenotype."""

    experiment_type: str = "calmorph"
    genotype: Genotype | list[Genotype,]
    phenotype: CalMorphPhenotype


class MicroarrayExpressionPhenotype(Phenotype, ModelStrict):
    """Microarray expression phenotype with canonical log2 ratio convention.

    CONVENTION: All expression_log2_ratio values follow the torchcell standard:
        expression_log2_ratio = log2(sample_of_interest / reference)

    Where:
        - sample_of_interest = mutant/deletion strain being studied
        - reference = wildtype or common reference pool

    This means:
        - Positive values: gene is MORE expressed in sample vs reference (upregulated)
        - Negative values: gene is LESS expressed in sample vs reference (downregulated)

    NOTE: Source data (GEO) may use different conventions (e.g., log2(reference/sample)).
    Dataset loaders MUST transform to this canonical representation for consistency.

    PRIMARY FIELDS (for BioCypher/ML):
        - expression_log2_ratio: log2 fold change relative to reference
        - expression_log2_ratio_se: Standard error of log2 ratios

    SECONDARY FIELDS (for QC/reproducibility):
        - expression: Absolute expression measurements on linear scale
        - expression_se: Standard error on linear scale
        - expression_log2_ratio_variance: Variance of log2 ratios
        - n_samples: Number of independent biological samples
    """

    graph_level: str = "node"
    label_name: str = "expression_log2_ratio"
    label_statistic_name: str = "expression_log2_ratio_se"

    # PRIMARY FIELDS - for BioCypher/Neo4j and ML training
    expression_log2_ratio: dict[str, float] = Field(
        description=(
            "SortedDict of log2 fold change ratios relative to wildtype reference. "
            "CONVENTION: log2(sample/reference) where positive = upregulated, "
            "negative = downregulated. All datasets transformed to this standard."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    expression_log2_ratio_se: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of standard errors (SE) for log2 ratios. "
            "SE = SD / sqrt(n_samples). "
            "None when unavailable; NaN when n_samples = 1 (SE undefined)."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )

    # SECONDARY FIELDS - for QC and reproducibility
    expression: dict[str, float] = Field(
        description=(
            "SortedDict of per-gene expression measurements on linear scale. "
            "May be raw probe intensities, background-subtracted, or normalized "
            "(e.g., quantile normalization, housekeeping gene normalization). "
            "NOT the ratio to reference - this is the sample's absolute expression."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    expression_se: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of standard errors on linear scale. "
            "None when unavailable; NaN when n_samples = 1 (SE undefined)."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    expression_log2_ratio_variance: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of variance for log2 ratios. Variance = SE^2 * n_samples."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    n_samples: dict[str, int] = Field(
        description="SortedDict of number of independent biological samples per gene",
        repr=False,  # Hide in repr to avoid clutter
    )

    def __repr__(self) -> str:
        """Custom repr that shows summary statistics instead of full data."""
        expr_count = len(self.expression) if self.expression else 0
        log2_count = (
            len(self.expression_log2_ratio) if self.expression_log2_ratio else 0
        )
        se_count = (
            len(self.expression_log2_ratio_se) if self.expression_log2_ratio_se else 0
        )
        n_samples_count = len(self.n_samples) if self.n_samples else 0

        return (
            f"MicroarrayExpressionPhenotype("
            f"expression_genes={expr_count}, "
            f"log2_ratio_genes={log2_count}, "
            f"log2_se_genes={se_count}, "
            f"n_samples_genes={n_samples_count})"
        )

    @field_validator("expression", mode="before")
    def convert_and_validate_expression(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce expression to a SortedDict and reject empty or infinite values."""
        if v is None:
            raise ValueError("expression measurements cannot be None")
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression measurements cannot be empty")
        for key, value in v.items():
            # Accept any gene name, not just systematic patterns
            if math.isinf(value):
                raise ValueError(f"Invalid expression value for gene {key}: {value}")
        return v

    @field_validator("expression_log2_ratio", mode="before")
    def convert_and_validate_log2_ratio(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce log2 ratios to a SortedDict and reject empty input."""
        if v is None:
            raise ValueError("expression_log2_ratio cannot be None")
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("expression_log2_ratio cannot be empty")
        # Accept any gene name keys, no validation needed
        return v

    @field_validator("expression_log2_ratio_se", mode="before")
    def convert_and_validate_log2_se(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce log2 ratio SE to a SortedDict and reject negative finite values."""
        if v is None:
            return v
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        # SE can be NaN or Inf when n=1
        for key, value in v.items():
            if not (math.isnan(value) or math.isinf(value)) and value < 0:
                raise ValueError(f"SE for {key} cannot be negative: {value}")
        return v

    @field_validator("expression_se", mode="before")
    def convert_and_validate_expression_se(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce expression SE to a SortedDict and reject negative finite values."""
        if v is None:
            return v
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        # SE can be NaN or Inf when n=1
        for key, value in v.items():
            if not (math.isnan(value) or math.isinf(value)) and value < 0:
                raise ValueError(f"SE for {key} cannot be negative: {value}")
        return v

    @field_validator("expression_log2_ratio_variance", mode="before")
    def convert_and_validate_variance(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce variance to a SortedDict and reject negative non-NaN values."""
        if v is None:
            return v
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        # Variance must be non-negative
        for key, value in v.items():
            if not math.isnan(value) and value < 0:
                raise ValueError(f"Variance for {key} cannot be negative: {value}")
        return v

    @field_validator("n_samples", mode="before")
    def convert_and_validate_n_samples(cls, v: Any) -> Any:  # raw pre-validation input
        """Coerce n_samples to a SortedDict and require positive integer counts."""
        if v is None:
            raise ValueError(
                "n_samples cannot be None - it is required for SE interpretation"
            )
        # Convert to SortedDict for consistent ordering
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("n_samples cannot be empty")
        # Validate that all n_samples are positive integers
        for key, value in v.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"n_samples for {key} must be a positive integer, got: {value}"
                )
        return v

    @model_validator(mode="after")
    def validate_matching_keys(self) -> "MicroarrayExpressionPhenotype":
        """Ensure all secondary fields have same keys as expression."""
        # n_samples must match expression keys
        if set(self.n_samples.keys()) != set(self.expression.keys()):
            raise ValueError("n_samples must have the same keys as expression")

        # Optional fields should match if present
        if self.expression_se is not None:
            if set(self.expression_se.keys()) != set(self.expression.keys()):
                raise ValueError("expression_se must have the same keys as expression")

        if self.expression_log2_ratio_se is not None:
            if set(self.expression_log2_ratio_se.keys()) != set(
                self.expression_log2_ratio.keys()
            ):
                raise ValueError(
                    "expression_log2_ratio_se must have the same keys as expression_log2_ratio"
                )

        if self.expression_log2_ratio_variance is not None:
            if set(self.expression_log2_ratio_variance.keys()) != set(
                self.expression_log2_ratio.keys()
            ):
                raise ValueError(
                    "expression_log2_ratio_variance must have the same keys as expression_log2_ratio"
                )

        return self


class MicroarrayExpressionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a microarray expression experiment."""

    experiment_reference_type: str = "microarray_expression"
    phenotype_reference: MicroarrayExpressionPhenotype


class MicroarrayExpressionExperiment(Experiment, ModelStrict):
    """Experiment measuring a microarray expression phenotype."""

    experiment_type: str = "microarray_expression"
    genotype: Genotype | list[Genotype,]
    phenotype: MicroarrayExpressionPhenotype


PhenotypeType = (
    Phenotype
    | FitnessPhenotype
    | GeneInteractionPhenotype
    | GeneEssentialityPhenotype
    | SyntheticLethalityPhenotype
    | SyntheticRescuePhenotype
    | CalMorphPhenotype
    | MicroarrayExpressionPhenotype
)

ExperimentType = (
    Experiment
    | FitnessExperiment
    | GeneInteractionExperiment
    | GeneEssentialityExperiment
    | SyntheticLethalityExperiment
    | SyntheticRescueExperiment
    | CalMorphExperiment
    | MicroarrayExpressionExperiment
)

ExperimentReferenceType = (
    ExperimentReference
    | FitnessExperimentReference
    | GeneInteractionExperimentReference
    | GeneEssentialityExperimentReference
    | SyntheticLethalityExperimentReference
    | SyntheticRescueExperimentReference
    | CalMorphExperimentReference
    | MicroarrayExpressionExperimentReference
)


EXPERIMENT_TYPE_MAP = {
    "fitness": FitnessExperiment,
    "gene interaction": GeneInteractionExperiment,
    "gene essentiality": GeneEssentialityExperiment,
    "synthetic lethality": SyntheticLethalityExperiment,
    "synthetic rescue": SyntheticRescueExperiment,
    "calmorph": CalMorphExperiment,
    "microarray_expression": MicroarrayExpressionExperiment,
}

EXPERIMENT_REFERENCE_TYPE_MAP = {
    "fitness": FitnessExperimentReference,
    "gene interaction": GeneInteractionExperimentReference,
    "gene essentiality": GeneEssentialityExperimentReference,
    "synthetic lethality": SyntheticLethalityExperimentReference,
    "synthetic rescue": SyntheticRescueExperimentReference,
    "calmorph": CalMorphExperimentReference,
    "microarray_expression": MicroarrayExpressionExperimentReference,
}


if __name__ == "__main__":
    pass

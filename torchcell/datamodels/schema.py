# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py

"""Pydantic data models for torchcell genotypes, environments, and phenotypes."""

import math
import re
from enum import StrEnum
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

    @model_validator(mode="after")
    def validate_label_fields(self) -> "Phenotype":
        """label_name / label_statistic_name must name fields on the concrete class.

        Inherited by every Phenotype subclass (replaces the formerly identical
        per-subclass copies). ``type(self).__annotations__`` is the concrete
        subclass's own field set, so each subclass is checked against its own
        declared fields.
        """
        own_fields = type(self).__annotations__
        if self.label_name not in own_fields:
            raise ValueError(
                f"label_name '{self.label_name}' must be a class attribute"
            )
        if (
            self.label_statistic_name is not None
            and self.label_statistic_name not in own_fields
        ):
            raise ValueError(
                f"label_statistic_name '{self.label_statistic_name}' "
                "must be a class attribute"
            )
        return self

    def __getitem__(self, key: str) -> Any:  # heterogeneous phenotype field values
        """Return the attribute value for the given field name."""
        return getattr(self, key)


class UncertaintyType(StrEnum):
    """What a reported uncertainty number IS, so it converts to an SE correctly.

    Rigor comes from naming the statistic: a bootstrap SD of an estimator is
    already an SE (never divide it by sqrt(n) again), whereas a sample SD of
    observations must be divided. There is deliberately NO ``unknown`` -- strict
    labelling: if the kind is unknown we do not ingest the value.
    """

    sample_sd = "sample_sd"  # SD of observations -> SE = sd / sqrt(n)
    standard_error = "standard_error"  # already SE of the mean -> use as-is
    bootstrap_se = "bootstrap_se"  # bootstrap SD of the estimator ~ SE -> as-is
    variance = "variance"  # sample variance -> SE = sqrt(var / n)
    ci95 = "ci95"  # 95% CI half-width -> SE = hw / 1.96


class SampleUnit(StrEnum):
    """What one sample in ``n_samples`` physically is. Add values as datasets need
    them (do not pre-populate). ``screen`` vs ``colony`` matters: Costanzo colonies
    are pseudoreplicates; the independent unit is the screen.
    """

    colony = "colony"
    screen = "screen"
    biological_replicate = "biological_replicate"
    technical_replicate = "technical_replicate"
    pooled = "pooled"


_Z95 = 1.959963984540054  # standard-normal two-sided 95% quantile


def derive_se(
    uncertainty: float | None,
    uncertainty_type: UncertaintyType | None,
    n_samples: int | None,
) -> float | None:
    """Derive the standard error of the mean from a reported uncertainty + its kind.

    ``standard_error``/``bootstrap_se`` -> as-is (already an SE); ``sample_sd`` ->
    sd/sqrt(n); ``variance`` -> sqrt(var/n); ``ci95`` -> half-width/1.96. n_samples
    is required for the kinds that divide. Returns None when nothing is reported.
    """
    if uncertainty is None or uncertainty_type is None:
        return None
    if uncertainty_type in (
        UncertaintyType.standard_error,
        UncertaintyType.bootstrap_se,
    ):
        return uncertainty
    if uncertainty_type is UncertaintyType.ci95:
        return uncertainty / _Z95
    if n_samples is None or n_samples < 1:
        raise ValueError(
            f"n_samples (>=1) required to derive SE from {uncertainty_type}"
        )
    if uncertainty_type is UncertaintyType.sample_sd:
        return uncertainty / math.sqrt(n_samples)
    if uncertainty_type is UncertaintyType.variance:
        return math.sqrt(uncertainty / n_samples)
    raise ValueError(f"unhandled uncertainty_type: {uncertainty_type}")


class FitnessPhenotype(Phenotype, ModelStrict):
    """Fitness phenotype with fitness value and uncertainty statistics.

    Uncertainty ontology: the source-reported number lives in ``fitness_uncertainty``
    with its ``fitness_uncertainty_type``; ``n_samples`` + ``sample_unit`` give the
    replicate design; ``fitness_se`` is the DERIVED, ML-facing standard error
    (auto-computed via ``derive_se`` when not supplied). ``fitness_std`` is
    DEPRECATED (superseded by uncertainty/type; retained until loaders migrate).
    """

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
    fitness_uncertainty: float | None = Field(
        default=None,
        description="Source-reported uncertainty number, verbatim (its meaning is "
        "given by fitness_uncertainty_type).",
    )
    fitness_uncertainty_type: UncertaintyType | None = Field(
        default=None, description="What fitness_uncertainty IS (sample_sd, ...)."
    )
    sample_unit: SampleUnit | None = Field(
        default=None,
        description="What one sample in n_samples is (colony, screen, ...).",
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

    @model_validator(mode="before")
    @classmethod
    def _fill_fitness_se(cls, data: Any) -> Any:
        """Derive the ML-facing fitness_se from the reported uncertainty.

        ModelStrict is frozen, so we fill fitness_se BEFORE construction rather than
        assign after. Skips when not safely derivable (e.g. sample_sd without n);
        ``_check_uncertainty`` then raises the precise error.
        """
        if not isinstance(data, dict):
            return data
        unc = data.get("fitness_uncertainty")
        typ = data.get("fitness_uncertainty_type")
        if unc is None or typ is None or data.get("fitness_se") is not None:
            return data
        typ = UncertaintyType(typ)
        n = data.get("n_samples")
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and n is None:
            return data
        data["fitness_se"] = derive_se(unc, typ, n)
        return data

    @model_validator(mode="after")
    def _check_uncertainty(self) -> "FitnessPhenotype":
        """Strict invariant: no unlabelled uncertainty (reported<->type both-or-
        neither); n_samples + sample_unit required for kinds that divide.
        """
        unc, typ = self.fitness_uncertainty, self.fitness_uncertainty_type
        if (unc is None) != (typ is None):
            raise ValueError(
                "fitness_uncertainty and fitness_uncertainty_type must both be set "
                "or both be None (no unlabelled uncertainty)"
            )
        if typ in (UncertaintyType.sample_sd, UncertaintyType.variance) and (
            self.n_samples is None or self.sample_unit is None
        ):
            raise ValueError(f"n_samples and sample_unit are required for {typ}")
        return self


class GeneEssentialityPhenotype(Phenotype, ModelStrict):
    """Phenotype indicating whether a gene knockout is lethal."""

    graph_level: str = "node"
    label_name: str = "is_essential"
    is_essential: bool = Field(
        default=True, description="gene knockout leading cell death."
    )


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
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: FitnessPhenotype


class GeneInteractionExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene interaction experiment."""

    experiment_reference_type: str = "gene interaction"
    phenotype_reference: GeneInteractionPhenotype


class GeneInteractionExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene interaction phenotype."""

    experiment_type: str = "gene interaction"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GeneInteractionPhenotype


class GeneEssentialityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a gene essentiality experiment."""

    experiment_reference_type: str = "gene essentiality"
    phenotype_reference: GeneEssentialityPhenotype


# shouldn't it jut be one gene for genotype?
class GeneEssentialityExperiment(Experiment, ModelStrict):
    """Experiment measuring a gene essentiality phenotype."""

    experiment_type: str = "gene essentiality"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: GeneEssentialityPhenotype


class SyntheticLethalityExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic lethality experiment."""

    experiment_reference_type: str = "synthetic lethality"
    phenotype_reference: SyntheticLethalityPhenotype


class SyntheticLethalityExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic lethality phenotype."""

    experiment_type: str = "synthetic lethality"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: SyntheticLethalityPhenotype


class SyntheticRescueExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a synthetic rescue experiment."""

    experiment_reference_type: str = "synthetic rescue"
    phenotype_reference: SyntheticRescuePhenotype


class SyntheticRescueExperiment(Experiment, ModelStrict):
    """Experiment measuring a synthetic rescue phenotype."""

    experiment_type: str = "synthetic rescue"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: SyntheticRescuePhenotype


class CalMorphExperimentReference(ExperimentReference, ModelStrict):
    """Reference context for a CalMorph experiment."""

    experiment_reference_type: str = "calmorph"
    phenotype_reference: CalMorphPhenotype


class CalMorphExperiment(Experiment, ModelStrict):
    """Experiment measuring a CalMorph phenotype."""

    experiment_type: str = "calmorph"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
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
        - expression: Absolute expression measurements on linear scale (reference only)
        - expression_log2_ratio_variance: Variance of log2 ratios
        - n_replicates: Number of independent biological/technical replicates
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
            "SE = SD / sqrt(n_replicates). "
            "None when unavailable; NaN when n_replicates = 1 (SE undefined)."
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
    expression_log2_ratio_variance: dict[str, float] | None = Field(
        default=None,
        description=(
            "SortedDict of variance for log2 ratios. Variance = SE^2 * n_replicates."
        ),
        repr=False,  # Hide in repr to avoid clutter
    )
    n_replicates: dict[str, int] = Field(
        description="SortedDict of number of independent biological/technical replicates per gene",
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
        n_replicates_count = len(self.n_replicates) if self.n_replicates else 0

        return (
            f"MicroarrayExpressionPhenotype("
            f"expression_genes={expr_count}, "
            f"log2_ratio_genes={log2_count}, "
            f"log2_se_genes={se_count}, "
            f"n_replicates_genes={n_replicates_count})"
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

    @field_validator("n_replicates", mode="before")
    def convert_and_validate_n_replicates(
        cls, v: Any
    ) -> Any:  # raw pre-validation input
        """Coerce n_replicates to a SortedDict and require positive-integer counts."""
        if v is None:
            raise ValueError(
                "n_replicates cannot be None - it is required for SE interpretation"
            )
        # Must be a per-gene mapping. Raise a clean ValueError (not crash on .items())
        # so union resolution can skip this member when a scalar-n_replicates phenotype
        # (e.g. VisualScorePhenotype) is the real match.
        if not isinstance(v, dict):
            raise ValueError(
                f"n_replicates must be a per-gene dict for this phenotype, got {type(v).__name__}"
            )
        # Convert to SortedDict for consistent ordering
        if not isinstance(v, SortedDict):
            v = SortedDict(v)
        if not v:
            raise ValueError("n_replicates cannot be empty")
        # Validate that all n_replicates are positive integers
        for key, value in v.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"n_replicates for {key} must be a positive integer, got: {value}"
                )
        return v

    @model_validator(mode="after")
    def validate_matching_keys(self) -> "MicroarrayExpressionPhenotype":
        """Ensure all secondary fields have same keys as expression."""
        # n_replicates must match expression keys
        if set(self.n_replicates.keys()) != set(self.expression.keys()):
            raise ValueError("n_replicates must have the same keys as expression")

        # Optional fields should match if present
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
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: MicroarrayExpressionPhenotype


class VisualScorePhenotype(Phenotype, ModelStrict):
    """Ordinal visual-inspection score as a proxy for a metabolite/product level.

    For screens where a metabolite is read out by a qualitative visual signal rather
    than a quantitative assay. Ozaydin 2005/2013: colony COLOR on a -5..+5 scale is a
    visual proxy for carotenoid (beta-carotene) accumulation -- more orange/red colony
    = more product. The score is a SUBJECTIVE ORDINAL, not a quantitative abundance;
    downstream models must treat it ordinally.

    Metabolite linkage (Yeast9): ``target_product`` names the metabolite the score is a
    proxy for (e.g. "beta-carotene"). Heterologous products (carotenoids, betalains)
    are NOT native to Yeast9, so ``target_metabolite_id`` -- an optional Yeast9
    ``s_NNNN`` metabolite id for constraint-based-model linkage -- is left ``None``
    until the metabolic-model mapping is decided; the fields here capture the data a
    CBM would need, without committing to that mapping.
    """

    graph_level: str = "global"
    label_name: str = "visual_score"
    label_statistic_name: str | None = None

    visual_score: float = Field(
        description="aggregated ordinal visual score (e.g. colony color intensity)"
    )
    visual_score_min: float | None = Field(
        default=None,
        description="min score across replicates (reproducibility; None if 1 replicate)",
    )
    n_replicates: int = Field(
        description="number of independent visually-scored replicates for this strain"
    )
    score_scale_min: int = Field(description="lower bound of the ordinal scale")
    score_scale_max: int = Field(description="upper bound of the ordinal scale")
    score_semantics: str = Field(
        description=(
            "what higher vs lower means, e.g. "
            "'higher = more orange colony = more carotenoid/beta-carotene'"
        )
    )
    target_product: str = Field(
        description="metabolite/product the score is a visual proxy for, e.g. 'beta-carotene'"
    )
    target_metabolite_id: str | None = Field(
        default=None,
        description="optional Yeast9 s_NNNN metabolite id for CBM linkage; None until modeling decided",
    )
    score_text: str | None = Field(
        default=None,
        description="non-numeric score annotations from the source (e.g. 'pet', 'tiny')",
    )
    qc_flags: dict[str, bool] | None = Field(
        default=None,
        description="QC/phenotype flags parsed from source comments (e.g. petite, tiny)",
    )

    @model_validator(mode="after")
    def validate_visual_score(self) -> "VisualScorePhenotype":
        """Enforce a coherent ordinal scale and score/replicate bounds."""
        if self.score_scale_min >= self.score_scale_max:
            raise ValueError("score_scale_min must be < score_scale_max")
        if not (self.score_scale_min <= self.visual_score <= self.score_scale_max):
            raise ValueError(
                f"visual_score {self.visual_score} outside scale "
                f"[{self.score_scale_min}, {self.score_scale_max}]"
            )
        if self.visual_score_min is not None and not (
            self.score_scale_min <= self.visual_score_min <= self.score_scale_max
        ):
            raise ValueError("visual_score_min outside the declared scale")
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be >= 1")
        return self


class VisualScoreExperimentReference(ExperimentReference, ModelStrict):
    """Reference (control colony) context for a visual-score experiment."""

    experiment_reference_type: str = "visual_score"
    phenotype_reference: VisualScorePhenotype


class VisualScoreExperiment(Experiment, ModelStrict):
    """Experiment measuring a visual-score phenotype."""

    experiment_type: str = "visual_score"
    genotype: Genotype | list[Genotype,]  # type: ignore[assignment]  # pydantic intentionally widens base Genotype field in subclass
    phenotype: VisualScorePhenotype


PhenotypeType = (
    Phenotype
    | FitnessPhenotype
    | GeneInteractionPhenotype
    | GeneEssentialityPhenotype
    | SyntheticLethalityPhenotype
    | SyntheticRescuePhenotype
    | CalMorphPhenotype
    | MicroarrayExpressionPhenotype
    | VisualScorePhenotype
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
    | VisualScoreExperiment
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
    | VisualScoreExperimentReference
)


EXPERIMENT_TYPE_MAP = {
    "fitness": FitnessExperiment,
    "gene interaction": GeneInteractionExperiment,
    "gene essentiality": GeneEssentialityExperiment,
    "synthetic lethality": SyntheticLethalityExperiment,
    "synthetic rescue": SyntheticRescueExperiment,
    "calmorph": CalMorphExperiment,
    "microarray_expression": MicroarrayExpressionExperiment,
    "visual_score": VisualScoreExperiment,
}

EXPERIMENT_REFERENCE_TYPE_MAP = {
    "fitness": FitnessExperimentReference,
    "gene interaction": GeneInteractionExperimentReference,
    "gene essentiality": GeneEssentialityExperimentReference,
    "synthetic lethality": SyntheticLethalityExperimentReference,
    "synthetic rescue": SyntheticRescueExperimentReference,
    "calmorph": CalMorphExperimentReference,
    "microarray_expression": MicroarrayExpressionExperimentReference,
    "visual_score": VisualScoreExperimentReference,
}


if __name__ == "__main__":
    pass

"""Pydantic schemas validating the SGD locus JSON record and its sub-structures."""

# torchcell/graph/validation/locus_related/locus.py
# [[torchcell.graph.validation.locus_related.locus]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/validation/locus_related/locus.py
# Test file: torchcell/graph/validation/locus_related/test_locus.py

from typing import Any

from pydantic import Field

from torchcell.datamodels import ModelStrict


# qualities
class ReferenceURL(ModelStrict):
    """External URL associated with a literature reference."""

    display_name: str
    link: str


class Reference(ModelStrict):
    """Literature reference with citation metadata and linked URLs."""

    id: int
    display_name: str
    citation: str
    pubmed_id: int | None
    link: str
    year: int
    urls: list[ReferenceURL] | None


class QualitiesAttribute(ModelStrict):
    """References backing a single locus quality attribute."""

    references: list[Reference]


class Qualities(ModelStrict):
    """References backing each curated quality field of a locus."""

    gene_name: QualitiesAttribute
    feature_type: QualitiesAttribute
    qualifier: QualitiesAttribute
    description: QualitiesAttribute
    name_description: QualitiesAttribute
    id: QualitiesAttribute


# aliases
class Alias(ModelStrict):
    """Alternative name or identifier for a locus."""

    id: int
    display_name: str
    link: str | None
    category: str
    references: list[Reference]
    source: dict[str, str]
    protein: bool | None = None


# urls
class LocusDataUrl(ModelStrict):
    """Categorized external URL listed on a locus page."""

    category: str
    link: str
    display_name: str


# alliance_icon_link
class AllianceIconLink(ModelStrict):
    """Alliance of Genome Resources model-organism database icon link."""

    mod: str
    icon_url: str


# protein_overview
class ProteinOverview(ModelStrict):
    """Summary protein properties such as length, weight, and isoelectric point."""

    length: int
    molecular_weight: float
    pi: float
    median_value: int | None = None
    median_abs_dev_value: int | None = None


# go_overview
class EvidenceCode(ModelStrict):
    """GO annotation evidence code with its display name and link."""

    display_name: str
    link: str


class GoTerm(ModelStrict):
    """Gene Ontology term annotation with namespace, qualifiers, and evidence."""

    namespace: str
    qualifiers: list[str]
    term: dict[str, Any]
    evidence_codes: list[EvidenceCode]


class GoSlim(ModelStrict):
    """GO Slim mapping entry for a locus."""

    slim_name: str
    go_id: int
    link: str
    display_name: str


class GoOverview(ModelStrict):
    """Aggregated GO annotations (manual, HTP, computational) and GO Slim for a locus."""

    manual_molecular_function_terms: list[GoTerm]
    manual_biological_process_terms: list[GoTerm]
    manual_cellular_component_terms: list[GoTerm]
    htp_molecular_function_terms: list[GoTerm]
    htp_biological_process_terms: list[GoTerm]
    htp_cellular_component_terms: list[GoTerm]
    computational_annotation_count: int
    go_slim: list[GoSlim]
    go_slim_grouped: list[GoSlim]
    date_last_reviewed: str
    paragraph: str


# alleles
class Allele(ModelStrict):
    """Named allele of a locus with its detail page link."""

    display_name: str
    link_url: str


# phenotype_overview
class Phenotype(ModelStrict):
    """Single phenotype annotation with name, link, and identifier."""

    display_name: str
    link: str
    id: int


class LargeScalePhenotypes(ModelStrict):
    """Phenotypes derived from large-scale studies, grouped by perturbation."""

    null: list[Phenotype]
    overexpression: list[Phenotype]


class ClassicalPhenotypes(ModelStrict):
    """Phenotypes from classical experiments, grouped by perturbation type."""

    null: list[Phenotype] | None = None
    overexpression: list[Phenotype] | None = None
    reuction_of_function: list[Phenotype] | None = Field(
        None, alias="reduction of function"
    )


class PhenotypeOverview(ModelStrict):
    """Aggregated phenotype data, strains, and experiment categories for a locus."""

    paragraph: str | None
    classical_phenotypes: ClassicalPhenotypes
    large_scale_phenotypes: LargeScalePhenotypes
    strains: list[list[str | int]]  # List of lists with either string or integer
    experiment_categories: list[list[str | int]]  # Same as above


class PhysicalExperiments(ModelStrict):
    """Counts of physical interaction experiment types for a locus."""

    # We default to 0 to see all possible experiments for any gene.
    affinity_capture_rna: int = Field(0, alias="Affinity Capture-RNA")
    pca: int = Field(0, alias="PCA")
    two_hybrid: int = Field(0, alias="Two-hybrid")


class GeneticExperiments(ModelStrict):
    """Counts of genetic interaction experiment types for a locus."""

    negative_genetic: int = Field(0, alias="Negative Genetic")
    positive_genetic: int = Field(0, alias="Positive Genetic")
    phenotypic_enhancement: int = Field(0, alias="Phenotypic Enhancement")
    synthetic_growth_defect: int = Field(0, alias="Synthetic Growth Defect")
    synthetic_lethality: int = Field(0, alias="Synthetic Lethality")
    dosage_rescue: int = Field(0, alias="Dosage Rescue")


class InteractionOverview(ModelStrict):
    """Summary of physical and genetic interactions and visualization sizing."""

    total_interactions: int
    total_interactors: int
    num_phys_interactors: int
    num_gen_interactors: int
    num_both_interactors: int
    physical_experiments: PhysicalExperiments
    genetic_experiments: GeneticExperiments
    gen_circle_size: float
    phys_circle_size: float
    circle_distance: float


# literature_overview
class LiteratureOverview(ModelStrict):
    """Counts of literature references by category for a locus."""

    primary_count: int
    additional_count: int
    review_count: int
    go_count: int
    phenotype_count: int
    disease_count: int
    interaction_count: int
    regulation_count: int
    htp_count: int
    total_count: int


# disease_overview
class DiseaseOverview(ModelStrict):
    """Summary of manual and high-throughput disease annotations for a locus."""

    manual_disease_terms: list[str]  # Update str if you know the type
    htp_disease_terms: list[str]  # Same as above
    computational_annotation_count: int
    date_last_reviewed: str | None  # None


# reulation overview
class RegulationOverview(ModelStrict):
    """Counts of regulators and targets for a locus."""

    regulator_count: int
    target_count: int


# history
class History(ModelStrict):
    """Curation history entry for a locus with type, note, and references."""

    category: str
    history_type: str
    note: str
    date_created: str
    references: list[Reference]


# reserved_name
class ReservedNameLocus(ModelStrict):
    """Locus referenced by a reserved gene name."""

    display_name: str
    systematic_name: str
    link: str


# ReservedName
class ReservedName(ModelStrict):
    """Reserved gene name record with reservation status and dates."""

    id: int
    display_name: str
    reservation_date: str
    expiration_date: str
    locus: ReservedNameLocus
    reference: Reference
    reservation_status: str
    name_description: str
    link: str
    class_type: str


# Locus Data
class LocusData(ModelStrict):
    """Full SGD locus record aggregating all locus-related sub-structures."""

    id: int  # 1266542
    display_name: str  # "YDR210W"
    format_name: str  # "YDR210W"
    gene_name: str | None  # None
    link: str  # "/locus/S000002618"
    sgdid: str  # "S000002618"
    qualities: Qualities
    aliases: list[Alias]
    references: list[Reference]
    locus_type: str  # "ORF"
    qualifier: str  # "Uncharacterized"
    bioent_status: str  # "Active"
    description: str
    name_description: str
    paralogs: list[str]
    complements: list[str]  # Can be None
    urls: list[LocusDataUrl]
    alliance_icon_links: list[AllianceIconLink]  # BOOK
    protein_overview: ProteinOverview
    go_overview: GoOverview
    pathways: list[str]  # Can be None
    alleles: list[Allele]
    sequence_summary: str
    protein_summary: str
    regulation_summary: str
    phenotype_overview: PhenotypeOverview
    interaction_overview: InteractionOverview
    paragraph: str | None
    literature_overview: LiteratureOverview
    disease_overview: DiseaseOverview
    ecnumbers: list[str]
    URS_ID: str | None
    main_strain: str
    regulation_overview: RegulationOverview
    reference_mapping: dict[str, int] | None
    history: list[History]
    complexes: list[str]
    reserved_name: ReservedName | None = None


# Validation
def validate_data(data: dict[str, Any]) -> LocusData:
    """Validate a raw locus dictionary against the LocusData schema."""
    return LocusData(**data)


if __name__ == "__main__":
    pass

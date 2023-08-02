# Locus.py
# Example given with gene "YDR210W"
from typing import Dict, List, Optional, Union

from pydantic import Field, validator

from torchcell.data_models import BaseModelStrict


# qualities
class ReferenceURL(BaseModelStrict):
    display_name: str
    link: str


class Reference(BaseModelStrict):
    id: int
    display_name: str
    citation: str
    pubmed_id: Optional[int]
    link: str
    year: int
    urls: Optional[List[ReferenceURL]]


class QualitiesAttribute(BaseModelStrict):
    references: List[Reference]


class Qualities(BaseModelStrict):
    gene_name: QualitiesAttribute
    feature_type: QualitiesAttribute
    qualifier: QualitiesAttribute
    description: QualitiesAttribute
    name_description: QualitiesAttribute
    id: QualitiesAttribute


# aliases
class Alias(BaseModelStrict):
    id: int
    display_name: str
    link: Optional[str]
    category: str
    references: List[Reference]
    source: dict[str, str]
    protein: Optional[bool] = None


# urls
class LocusDataUrl(BaseModelStrict):
    category: str
    link: str
    display_name: str


# alliance_icon_link
class AllianceIconLink(BaseModelStrict):
    mod: str
    icon_url: str


# protein_overview
class ProteinOverview(BaseModelStrict):
    length: int
    molecular_weight: float
    pi: float
    median_value: Optional[int] = None
    median_abs_dev_value: Optional[int] = None


# go_overview
class EvidenceCode(BaseModelStrict):
    display_name: str
    link: str


class GoTerm(BaseModelStrict):
    namespace: str
    qualifiers: List[str]
    term: dict
    evidence_codes: List[EvidenceCode]


class GoSlim(BaseModelStrict):
    slim_name: str
    go_id: int
    link: str
    display_name: str


class GoOverview(BaseModelStrict):
    manual_molecular_function_terms: List[GoTerm]
    manual_biological_process_terms: List[GoTerm]
    manual_cellular_component_terms: List[GoTerm]
    htp_molecular_function_terms: List[GoTerm]
    htp_biological_process_terms: List[GoTerm]
    htp_cellular_component_terms: List[GoTerm]
    computational_annotation_count: int
    go_slim: List[GoSlim]
    go_slim_grouped: List[GoSlim]
    date_last_reviewed: str
    paragraph: str


# alleles
class Allele(BaseModelStrict):
    display_name: str
    link_url: str


# phenotype_overview
class Phenotype(BaseModelStrict):
    display_name: str
    link: str
    id: int


class LargeScalePhenotypes(BaseModelStrict):
    null: List[Phenotype]
    overexpression: List[Phenotype]


class ClassicalPhenotypes(BaseModelStrict):
    null: Optional[List[Phenotype]] = None
    overexpression: Optional[List[Phenotype]] = None
    reuction_of_function: Optional[List[Phenotype]] = Field(
        None, alias="reduction of function"
    )


class PhenotypeOverview(BaseModelStrict):
    paragraph: Optional[str]
    classical_phenotypes: ClassicalPhenotypes
    large_scale_phenotypes: LargeScalePhenotypes
    strains: List[List[Union[str, int]]]  # List of lists with either string or integer
    experiment_categories: List[List[Union[str, int]]]  # Same as above


class PhysicalExperiments(BaseModelStrict):
    # We default to 0 to see all possible experiments for any gene.
    affinity_capture_rna: int = Field(0, alias="Affinity Capture-RNA")
    pca: int = Field(0, alias="PCA")
    two_hybrid: int = Field(0, alias="Two-hybrid")


class GeneticExperiments(BaseModelStrict):
    negative_genetic: int = Field(0, alias="Negative Genetic")
    positive_genetic: int = Field(0, alias="Positive Genetic")
    phenotypic_enhancement: int = Field(0, alias="Phenotypic Enhancement")
    synthetic_growth_defect: int = Field(0, alias="Synthetic Growth Defect")
    synthetic_lethality: int = Field(0, alias="Synthetic Lethality")
    dosage_rescue: int = Field(0, alias="Dosage Rescue")


class InteractionOverview(BaseModelStrict):
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
class LiteratureOverview(BaseModelStrict):
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
class DiseaseOverview(BaseModelStrict):
    manual_disease_terms: List[str]  # Update str if you know the type
    htp_disease_terms: List[str]  # Same as above
    computational_annotation_count: int
    date_last_reviewed: Optional[str]  # None


# reulation overview
class RegulationOverview(BaseModelStrict):
    regulator_count: int
    target_count: int


# history
class History(BaseModelStrict):
    category: str
    history_type: str
    note: str
    date_created: str
    references: List[Reference]


# reserved_name
class ReservedNameLocus(BaseModelStrict):
    display_name: str
    systematic_name: str
    link: str


# ReservedName
class ReservedName(BaseModelStrict):
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
class LocusData(BaseModelStrict):
    id: int  # 1266542
    display_name: str  # "YDR210W"
    format_name: str  # "YDR210W"
    gene_name: Optional[str]  # None
    link: str  # "/locus/S000002618"
    sgdid: str  # "S000002618"
    qualities: Qualities
    aliases: List[Alias]
    references: List[Reference]
    locus_type: str  # "ORF"
    qualifier: str  # "Uncharacterized"
    bioent_status: str  # "Active"
    description: str
    name_description: str
    paralogs: List[str]
    complements: List[str]  # Can be None
    urls: List[LocusDataUrl]
    alliance_icon_links: List[AllianceIconLink]  # BOOK
    protein_overview: ProteinOverview
    go_overview: GoOverview
    pathways: List[str]  # Can be None
    alleles: List[Allele]
    sequence_summary: str
    protein_summary: str
    regulation_summary: str
    phenotype_overview: PhenotypeOverview
    interaction_overview: InteractionOverview
    paragraph: Optional[str]
    literature_overview: LiteratureOverview
    disease_overview: DiseaseOverview
    ecnumbers: List[str]
    URS_ID: Optional[str]
    main_strain: str
    regulation_overview: RegulationOverview
    reference_mapping: Optional[dict[str, int]]
    history: List[History]
    complexes: List[str]
    reserved_name: Optional[ReservedName] = None


# Validation
def validate_data(data: dict) -> LocusData:
    return LocusData(**data)


if __name__ == "__main__":
    pass

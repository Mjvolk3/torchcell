# %%
# Relies on env-intermine (python 3.5)
import json
import os
import os.path as osp

import pandas as pd
from dask import compute, delayed
from intermine.webservice import Service
from tqdm import tqdm

from torchcell.sc_graph import get_gene_list


def get_regulators(gene: str = "YOR202W") -> int:
    # Gene Regulation
    # Retrieve genes that are regulators of a given target gene.
    # Gene(target) -> Gene(Regulators).
    # "YOR202W" == "HIS3"
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("GeneTarget_GeneFactor")
    rows = template.rows(
        A={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    data_dict_list = []
    reg_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["regulatoryRegions.regulator.symbol"] = row[
            "regulatoryRegions.regulator.symbol"
        ]
        data_dict["regulatoryRegions.regulator.secondaryIdentifier"] = row[
            "regulatoryRegions.regulator.secondaryIdentifier"
        ]
        data_dict["symbol"] = row["symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["regulatoryRegions.regEvidence.ontologyTerm.name"] = row[
            "regulatoryRegions.regEvidence.ontologyTerm.name"
        ]
        data_dict["regulatoryRegions.regEvidence.ontologyTerm.identifier"] = row[
            "regulatoryRegions.regEvidence.ontologyTerm.identifier"
        ]
        data_dict["regulatoryRegions.experimentCondition"] = row[
            "regulatoryRegions.experimentCondition"
        ]
        data_dict["regulatoryRegions.strainBackground"] = row[
            "regulatoryRegions.strainBackground"
        ]
        data_dict["regulatoryRegions.regulationDirection"] = row[
            "regulatoryRegions.regulationDirection"
        ]
        data_dict["regulatoryRegions.publications.pubMedId"] = row[
            "regulatoryRegions.publications.pubMedId"
        ]
        data_dict["regulatoryRegions.regulatorType"] = row[
            "regulatoryRegions.regulatorType"
        ]
        data_dict["regulatoryRegions.datasource"] = row["regulatoryRegions.datasource"]
        data_dict["regulatoryRegions.annotationType"] = row[
            "regulatoryRegions.annotationType"
        ]
        data_dict_list.append(data_dict)
    reg_dict["regulators"] = data_dict_list
    # if reg_dict is empty set to None
    if reg_dict == {}:
        reg_dict["regulators"] = None
    return reg_dict


def get_physical_interactions(gene: str = "YFL039C") -> dict:
    # act1  == YFL039C
    # Physical Interactions
    # Retrieve all physical interactions for a specific gene. Gene(target) -> Physical Interactions.
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_Physical_Interaction")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})

    data_dict_list = []
    physical_int_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["name"] = row["name"]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict["interactions.details.annotationType"] = row[
            "interactions.details.annotationType"
        ]
        data_dict["interactions.details.role1"] = row["interactions.details.role1"]
        data_dict["interactions.participant2.symbol"] = row[
            "interactions.participant2.symbol"
        ]
        data_dict["interactions.participant2.secondaryIdentifier"] = row[
            "interactions.participant2.secondaryIdentifier"
        ]
        data_dict[
            "interactions.details.experiment.interactionDetectionMethods.identifier"
        ] = row[
            "interactions.details.experiment.interactionDetectionMethods.identifier"
        ]
        data_dict["interactions.details.experiment.name"] = row[
            "interactions.details.experiment.name"
        ]
        data_dict["interactions.details.relationshipType"] = row[
            "interactions.details.relationshipType"
        ]
        data_dict["interactions.details.note"] = row["interactions.details.note"]
        data_dict_list.append(data_dict)
    physical_int_dict["physical_interactions"] = data_dict_list
    # if physical_int_dict is empty set to None
    if physical_int_dict == {}:
        physical_int_dict["physical_interactions"] = None
    return physical_int_dict


def get_gene_interactions(gene: str = "YDR210W") -> dict:
    # Gene Interactions
    # Retrieve all gene interactions for a specific gene. Gene(target) -> Gene Interactions.
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_Genetic_Interaction")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})

    data_dict_list = []
    gene_int_dict = {}
    for i, row in enumerate(rows):
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["name"] = row["name"]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict["interactions.details.annotationType"] = row[
            "interactions.details.annotationType"
        ]
        data_dict["interactions.details.phenotype"] = row[
            "interactions.details.phenotype"
        ]
        # TODO - What is interactions.details.role1 describing? Looks ok for now... Can check after
        data_dict["interactions.details.role1"] = row["interactions.details.role1"]
        data_dict["interactions.participant2.symbol"] = row[
            "interactions.participant2.symbol"
        ]
        data_dict["interactions.participant2.secondaryIdentifier"] = row[
            "interactions.participant2.secondaryIdentifier"
        ]
        data_dict[
            "interactions.details.experiment.interactionDetectionMethods.identifier"
        ] = row[
            "interactions.details.experiment.interactionDetectionMethods.identifier"
        ]
        data_dict["interactions.details.experiment.name"] = row[
            "interactions.details.experiment.name"
        ]
        data_dict["interactions.details.relationshipType"] = row[
            "interactions.details.relationshipType"
        ]
        data_dict["interactions.details.note"] = row["interactions.details.note"]
        data_dict["interactions.alleleinteractions.allele1.name"] = row[
            "interactions.alleleinteractions.allele1.name"
        ]
        data_dict["interactions.alleleinteractions.allele2.name"] = row[
            "interactions.alleleinteractions.allele2.name"
        ]
        data_dict["interactions.alleleinteractions.pvalue"] = row[
            "interactions.alleleinteractions.pvalue"
        ]
        data_dict["interactions.alleleinteractions.sgaScore"] = row[
            "interactions.alleleinteractions.sgaScore"
        ]
        data_dict_list.append(data_dict)
    gene_int_dict["gene_interactions"] = data_dict_list
    # if gene_int_dict is empty set to None
    if gene_int_dict == {}:
        gene_int_dict["gene_interactions"] = None
    return gene_int_dict


def get_protein_abundance(gene: str = "tfc3") -> dict:
    # Retrieve protein abundance for the protein(s) encoded by a gene
    # Gene(target) -> Protein Abundance Experiments.
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_ProteinAbundance")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})
    data_dict_list = []
    protein_abundance_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["proteins.symbol"] = row["proteins.symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["qualifier"] = row["qualifier"]
        data_dict["proteins.proteinAbundance.abundance"] = row[
            "proteins.proteinAbundance.abundance"
        ]
        data_dict["proteins.proteinAbundance.units"] = row[
            "proteins.proteinAbundance.units"
        ]
        data_dict["proteins.proteinAbundance.media"] = row[
            "proteins.proteinAbundance.media"
        ]
        data_dict["proteins.proteinAbundance.treatment"] = row[
            "proteins.proteinAbundance.treatment"
        ]
        data_dict["proteins.proteinAbundance.treatmentTime"] = row[
            "proteins.proteinAbundance.treatmentTime"
        ]
        data_dict["proteins.proteinAbundance.foldChange"] = row[
            "proteins.proteinAbundance.foldChange"
        ]
        data_dict["proteins.proteinAbundance.visualization"] = row[
            "proteins.proteinAbundance.visualization"
        ]
        data_dict["proteins.proteinAbundance.strainBackground"] = row[
            "proteins.proteinAbundance.strainBackground"
        ]
        data_dict["proteins.proteinAbundance.origPublication.pubMedId"] = row[
            "proteins.proteinAbundance.origPublication.pubMedId"
        ]
        data_dict["proteins.proteinAbundance.publication.pubMedId"] = row[
            "proteins.proteinAbundance.publication.pubMedId"
        ]
        data_dict_list.append(data_dict)
    protein_abundance_dict["protein_abundance"] = data_dict_list
    # if protein_abundance_dict is empty set to None
    if protein_abundance_dict == {}:
        protein_abundance_dict["protein_abundance"] = None
    return protein_abundance_dict


def get_median_protein_abundance(gene: str = "YAL001C") -> dict:
    # Gene median protein abundance
    # Retrieve all protein median abundance for a specific gene. Gene(target) -> median protein abundance.
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Protein_Median_Abundance")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})
    data_dict_list = []
    median_protein_abundance_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["proteins.symbol"] = row["proteins.symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["qualifier"] = row["qualifier"]
        data_dict["proteins.median"] = row["proteins.median"]
        data_dict["proteins.units"] = row["proteins.units"]
        data_dict["proteins.MAD"] = row["proteins.MAD"]
        data_dict["proteins.proteinAbundance.publication.pubMedId"] = row[
            "proteins.proteinAbundance.publication.pubMedId"
        ]
        data_dict_list.append(data_dict)
    median_protein_abundance_dict["median_protein_abundance"] = data_dict_list
    # if median_protein_abundance_dict is empty set to None
    if median_protein_abundance_dict == {}:
        median_protein_abundance_dict["median_protein_abundance"] = None
    return median_protein_abundance_dict


def get_protein_sequence(gene: str = "rad54") -> dict:
    # Gene(target) -> Gene protein sequence
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_ProteinSequence")
    rows = template.rows(
        B={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    data_dict_list = []
    protein_sequence_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["name"] = row["name"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["proteins.sequence.residues"] = row["proteins.sequence.residues"]
        data_dict["proteins.sequence.length"] = row["proteins.sequence.length"]
        data_dict["featureType"] = row["featureType"]
        data_dict["proteins.symbol"] = row["proteins.symbol"]
        data_dict_list.append(data_dict)
    protein_sequence_dict["protein_sequence"] = data_dict_list
    # if protein_sequence_dict is empty set to None
    if protein_sequence_dict == {}:
        protein_sequence_dict["protein_sequence"] = None
    return protein_sequence_dict


def get_gene_sequence(gene: str = "rad54") -> dict:
    # Retrieve genomic DNA (DNA sequence with introns)
    # # Gene(target) -> Gene sequence
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_GenomicDNA")
    rows = template.rows(
        E={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    data_dict_list = []
    gene_sequence_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["name"] = row["name"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict["sequence.length"] = row["sequence.length"]
        data_dict["sequence.residues"] = row["sequence.residues"]
        data_dict["description"] = row["description"]
        data_dict["qualifier"] = row["qualifier"]
        data_dict_list.append(data_dict)
    gene_sequence_dict["gene_sequence"] = data_dict_list
    # if gene_sequence_dict is empty set to None
    if gene_sequence_dict == {}:
        gene_sequence_dict["gene_sequence"] = None
    return gene_sequence_dict


def get_go(gene: str = "YAL018C") -> dict:
    # Gene(target) -> GO terms
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_GO")
    data_dict_list = []
    go_dict = {}
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["featureType"] = row["featureType"]
        data_dict["qualifier"] = row["qualifier"]
        data_dict["goAnnotation.ontologyTerm.identifier"] = row[
            "goAnnotation.ontologyTerm.identifier"
        ]
        data_dict["goAnnotation.ontologyTerm.name"] = row[
            "goAnnotation.ontologyTerm.name"
        ]
        data_dict["goAnnotation.ontologyTerm.namespace"] = row[
            "goAnnotation.ontologyTerm.namespace"
        ]
        data_dict["goAnnotation.evidence.code.code"] = row[
            "goAnnotation.evidence.code.code"
        ]
        data_dict["goAnnotation.qualifier"] = row["goAnnotation.qualifier"]
        data_dict["goAnnotation.evidence.code.withText"] = row[
            "goAnnotation.evidence.code.withText"
        ]
        data_dict["goAnnotation.annotationExtension"] = row[
            "goAnnotation.annotationExtension"
        ]
        data_dict["goAnnotation.evidence.code.annotType"] = row[
            "goAnnotation.evidence.code.annotType"
        ]
        data_dict["goAnnotation.evidence.publications.pubMedId"] = row[
            "goAnnotation.evidence.publications.pubMedId"
        ]
        data_dict["goAnnotation.evidence.publications.citation"] = row[
            "goAnnotation.evidence.publications.citation"
        ]
        data_dict_list.append(data_dict)
    go_dict["go"] = data_dict_list
    # if go_dict is empty set to None
    if go_dict == {}:
        go_dict["go"] = None
    return go_dict


def get_protein_half_life(gene: str = "act1") -> dict:
    # Gene(target) -> protein half life
    # Retrieve Protein half-life for a given gene(s). This study had reported value
    # of >=100 hours for the most stable proteins. To facilitate sorting and
    # searching this values have been substituted  '10000' hours.

    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_Protein_Half_Life")
    rows = template.rows(
        A={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    data_dict_list = []
    protein_half_life_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["description"] = row["description"]
        data_dict["proteins.symbol"] = row["proteins.symbol"]
        data_dict["proteins.proteinHalfLife.experiment"] = row[
            "proteins.proteinHalfLife.experiment"
        ]
        data_dict["proteins.proteinHalfLife.value"] = row[
            "proteins.proteinHalfLife.value"
        ]
        data_dict["proteins.proteinHalfLife.units"] = row[
            "proteins.proteinHalfLife.units"
        ]
        data_dict["proteins.proteinHalfLife.publication.pubMedId"] = row[
            "proteins.proteinHalfLife.publication.pubMedId"
        ]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict_list.append(data_dict)
    protein_half_life_dict["protein_half_life"] = data_dict_list
    # if protein_half_life_dict is empty set to None
    if protein_half_life_dict == {}:
        protein_half_life_dict["protein_half_life"] = None
    return protein_half_life_dict


# Retrieve the chromosomal location for a gene
def get_chromosomal_location(gene: str = "act1") -> dict:
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_ChromosomeLocation")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})
    data_dict_list = []
    chromosomal_location_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["name"] = row["name"]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict["chromosome.primaryIdentifier"] = row["chromosome.primaryIdentifier"]
        data_dict["chromosomeLocation.start"] = row["chromosomeLocation.start"]
        data_dict["chromosomeLocation.end"] = row["chromosomeLocation.end"]
        data_dict["chromosomeLocation.strand"] = row["chromosomeLocation.strand"]
        data_dict_list.append(data_dict)
    chromosomal_location_dict["chromosomal_location"] = data_dict_list
    # if chromosomal_location_dict is empty set to None
    if chromosomal_location_dict == {}:
        chromosomal_location_dict["chromosomal_location"] = None
    return chromosomal_location_dict


def get_feature_type(gene: str = "YIL080W") -> dict:
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    # Get a new query on the class (table) you will be querying:
    query = service.new_query("Gene")
    # The view specifies the output columns
    query.add_view("featureType")
    # Uncomment and edit the line below (the default) to select a custom sort order:
    # query.add_sort_order("Gene.description", "ASC")
    query.add_constraint("organism.shortName", "=", "S. cerevisiae", code="F")
    query.add_constraint("Gene", "LOOKUP", gene, code="A")
    data_dict_list = []
    description_dict = {}
    for row in query.rows():
        data_dict = {}
        data_dict["featureType"] = row["featureType"]
        data_dict_list.append(data_dict)
    description_dict["featureType"] = data_dict_list
    # if description_dict is empty set to None
    if description_dict == {}:
        description_dict["featureType"] = None
    return description_dict


def get_all_feature_types() -> dict:
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    # Get a new query on the class (table) you will be querying:
    query = service.new_query("Gene")
    # The view specifies the output columns
    query.add_view("featureType", "secondaryIdentifier")
    # Uncomment and edit the line below (the default) to select a custom sort order:
    # query.add_sort_order("Gene.featureType", "ASC")
    data_dict = {}
    for row in query.rows():
        data_dict[row["secondaryIdentifier"]] = row["featureType"]
    return data_dict


def get_pathways(gene: str = "fas1") -> dict:
    # Gene associated pathways
    # Retrieve all associated pathways for a specific gene. Gene(target) -> Gene Phenotypes.
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_Pathways")
    rows = template.rows(A={"op": "LOOKUP", "value": gene, "extra_value": ""})
    data_dict_list = []
    pathways_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["name"] = row["name"]
        data_dict["organism.shortName"] = row["organism.shortName"]
        data_dict["pathways.identifier"] = row["pathways.identifier"]
        data_dict["pathways.name"] = row["pathways.name"]
        data_dict_list.append(data_dict)
    pathways_dict["pathways"] = data_dict_list
    if pathways_dict == {}:
        pathways_dict["pathways"] = None
    return pathways_dict


def get_phenotype(gene: str = "act1") -> dict:
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    template = service.get_template("Gene_Phenotype_New")
    rows = template.rows(
        A={"op": "LOOKUP", "value": gene, "extra_value": "S. cerevisiae"}
    )
    data_dict_list = []
    phenotype_dict = {}
    for row in rows:
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["qualifier"] = row["qualifier"]
        data_dict["phenotypes.experimentType"] = row["phenotypes.experimentType"]
        data_dict["phenotypes.mutantType"] = row["phenotypes.mutantType"]
        data_dict["phenotypes.observable"] = row["phenotypes.observable"]
        data_dict["phenotypes.qualifier"] = row["phenotypes.qualifier"]
        data_dict["phenotypes.allele"] = row["phenotypes.allele"]
        data_dict["phenotypes.alleleDescription"] = row["phenotypes.alleleDescription"]
        data_dict["phenotypes.strainBackground"] = row["phenotypes.strainBackground"]
        data_dict["phenotypes.chemical"] = row["phenotypes.chemical"]
        data_dict["phenotypes.condition"] = row["phenotypes.condition"]
        data_dict["phenotypes.details"] = row["phenotypes.details"]
        data_dict["phenotypes.reporter"] = row["phenotypes.reporter"]
        data_dict["phenotypes.publications.pubMedId"] = row[
            "phenotypes.publications.pubMedId"
        ]
        data_dict["phenotypes.publications.citation"] = row[
            "phenotypes.publications.citation"
        ]
        data_dict_list.append(data_dict)
    phenotype_dict["phenotype"] = data_dict_list
    if phenotype_dict == {}:
        phenotype_dict["phenotype"] = None
    return phenotype_dict


# Was originally used to get the "symbol" gene names (like act1) for genes with standard names. Used for the Mechanisitc-Aware comparison. Not yet written to ym_attrs.
def get_gene_external_id(gene: str = "act1") -> dict:
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    query = service.new_query("Gene")
    query.add_view(
        "primaryIdentifier",
        "secondaryIdentifier",
        "symbol",
        "name",
        "sgdAlias",
        "crossReferences.identifier",
        "crossReferences.source.name",
    )
    query.add_constraint("organism.shortName", "=", "S. cerevisiae", code="B")
    query.add_constraint("Gene", "LOOKUP", gene, code="A")
    data_dict_list = []
    external_id_dict = {}
    for row in query.rows():
        data_dict = {}
        data_dict["primaryIdentifier"] = row["primaryIdentifier"]
        data_dict["secondaryIdentifier"] = row["secondaryIdentifier"]
        data_dict["symbol"] = row["symbol"]
        data_dict["name"] = row["name"]
        data_dict["sgdAlias"] = row["sgdAlias"]
        data_dict["crossReferences.identifier"] = row["crossReferences.identifier"]
        data_dict["crossReferences.source.name"] = row["crossReferences.source.name"]
        data_dict_list.append(data_dict)

    external_id_dict["external_ids"] = data_dict_list
    if external_id_dict == {}:
        external_id_dict["external_ids"] = None
    return external_id_dict


def write_gene_id_translation_table() -> tuple[pd.DataFrame, list[str]]:
    # This function takes 2.5 hours to run. Could probably pull from somewhere else... but this is convenient in that it is specific to the gene_list.
    gene_list = get_gene_list()
    list_of_dicts = []
    for i, gene in enumerate(tqdm(gene_list)):
        errors = []
        try:
            symbol = get_gene_external_id(gene)["external_ids"][0]["symbol"]
        except IndexError as error:
            errors.append(f"gene: {error}")
        list_of_dicts.append({"secondaryIdentifier": gene, "symbol": symbol})
    df = pd.DataFrame(list_of_dicts)
    path = "data/preprocessed/gene_id_translation_table.csv"
    df.to_csv(path, index=False)
    print(f"Table writen to: {path}")
    return df, errors


def merge_dicts(dict_list: list[dict]) -> dict:
    result = {}
    for d in dict_list:
        result.update(d)
    return result


def get_ym_attrs_list(gene: str) -> list[dict]:
    regulators = delayed(get_regulators)(gene)
    physical_interaction = delayed(get_physical_interactions)(gene)
    gene_interaction = delayed(get_gene_interactions)(gene)
    phenotype = delayed(get_phenotype)(gene)
    protein_abundance = delayed(get_protein_abundance)(gene)
    median_protein_abundance = delayed(get_median_protein_abundance)(gene)
    protein_sequence = delayed(get_protein_sequence)(gene)
    gene_sequence = delayed(get_gene_sequence)(gene)
    go = delayed(get_go)(gene)
    protein_half_life = delayed(get_protein_half_life)(gene)
    chromosomal_location = delayed(get_chromosomal_location)(gene)
    feature_type = delayed(get_feature_type)(gene)
    pathways = delayed(get_pathways)(gene)

    ym_attrs_list = [
        regulators,
        physical_interaction,
        gene_interaction,
        phenotype,
        protein_abundance,
        median_protein_abundance,
        protein_sequence,
        gene_sequence,
        go,
        protein_half_life,
        chromosomal_location,
        feature_type,
        pathways,
    ]
    return ym_attrs_list


def create_node_ym_attrs(gene_list: list) -> dict:
    # Failures occur do to requests connections errors.
    file_names_list = []
    node_ym_attrs_dir = "data/preprocessed/node_ym_attrs"
    if not osp.exists(node_ym_attrs_dir):
        os.mkdir(node_ym_attrs_dir)
    else:
        # sort by date modified
        files = os.listdir(node_ym_attrs_dir)
        files.sort(key=lambda x: os.path.getmtime(osp.join(node_ym_attrs_dir, x)))
    file_names_list = [f.split(".")[0] for f in files]
    for gene in tqdm(gene_list):
        # Overwrite the last file in list, where previous run failed.
        if gene not in file_names_list[:-1]:
            print(gene)
            ym_attrs = compute(get_ym_attrs_list(gene))[0]
            ym_attrs_dict = {gene: merge_dicts(ym_attrs)}
            # write ym_attrs_list to json with files name indexed by ith gene
            with open(f"{node_ym_attrs_dir}/{gene}.json", "w") as f:
                json.dump(ym_attrs_dict, f, indent=4)
    return ym_attrs_dict


if __name__ == "__main__":
    # gene_list = get_gene_list()
    # create_node_ym_attrs(gene_list)
    ###
    # df, errors = write_gene_id_translation_table()
    # print(df)
    # print(errors)
    ###
    # get_gene_sequence()
    get_regulators()
    pass

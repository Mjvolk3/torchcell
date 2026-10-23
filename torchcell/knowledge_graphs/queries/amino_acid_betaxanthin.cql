// torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql
// [[torchcell.knowledge_graphs.queries.amino_acid_betaxanthin]]
//
// Supported query: amino acids + betaxanthin (showcase group 3).
//
// Datasets selected (by Dataset.id in the served graph), all MetabolitePhenotype with
// graph_level 'metabolism' (torchcell/datamodels/schema.py):
//   - AminoAcidMulleder2016Dataset    intracellular amino acids, deletion collection
//   - AminoAcidCooper2010Dataset      amino-acid pools (served in 2026.09.21-ab6d8c5d)
//   - BetaxanthinCachera2023Dataset   betaxanthin fluorescence, deletions on the Btx cassette
// Other metabolite datasets of that release (MetaboliteZelezniak2018Dataset,
// MetaboliteDaSilveira2014Dataset, FattyAcid*, OrganicAcidYoshida2012Dataset) are not
// selected; they are not amino-acid or betaxanthin measurements.
//
// Perturbation filter of experiments/019-simb-multimodal/queries/fig6_build.cql: every
// DELETION must be of a gene in $gene_set and at least one deletion is required, while
// non-deletion perturbations pass: the Btx-cassette gene additions (CYP76AD1 and DOD
// from plants, and the feedback-resistant ARO4 K229L and ARO7 G141S alleles). The 025
// rule (every perturbed gene in $gene_set) would drop every Cachera record unless
// $gene_set carried CYP76AD1 and DOD, which are not yeast genes. RETURN is
// property-shaped (serialized_data strings).
//
// Released identifiers (2026-10-23). Since PR #1085 (#1045) a strain whose source names the
// deleted gene by a SYMBOL sits on ReleasedIdentifierPerturbation (perturbation_type
// 'released_identifier', released_kind deletion, systematic_gene_name = the symbol): 3,592 of
// Cachera's 4,759 served experiments, and Mulleder's non-systematic strain entries. Under
// the clause above alone the Cachera block returned 1,101 of 4,759 against release
// 2026.10.17-4295f7ca (experiment 046, jobs 6154 and 6155). The Mulleder and Cachera blocks
// therefore admit a released_identifier perturbation as a deletion WITHOUT the gene-set
// test (the served node carries no released_kind property, so the type is what the query
// can see); the symbol is resolved to its R64 ORF at read time by
// torchcell.data.graph_processor.Perturbation(released_name_resolver=...), every refusal
// listed on the record. Cooper 2010 writes no released identifiers and keeps the clause.

// AminoAcidMulleder2016Dataset (MetabolitePhenotype, graph_level 'metabolism')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'AminoAcidMulleder2016Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'metabolism'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE dp.perturbation_type = 'released_identifier'
 OR (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
WHERE pert.perturbation_type ENDS WITH 'deletion'
 OR pert.perturbation_type = 'released_identifier' | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// AminoAcidCooper2010Dataset (MetabolitePhenotype, graph_level 'metabolism')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'AminoAcidCooper2010Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'metabolism'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// BetaxanthinCachera2023Dataset (MetabolitePhenotype, graph_level 'metabolism')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'BetaxanthinCachera2023Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'metabolism'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE dp.perturbation_type = 'released_identifier'
 OR (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
WHERE pert.perturbation_type ENDS WITH 'deletion'
 OR pert.perturbation_type = 'released_identifier' | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

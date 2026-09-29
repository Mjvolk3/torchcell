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

// AminoAcidMulleder2016Dataset (MetabolitePhenotype, graph_level 'metabolism')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'AminoAcidMulleder2016Dataset'
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
WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

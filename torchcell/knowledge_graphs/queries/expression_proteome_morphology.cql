// torchcell/knowledge_graphs/queries/expression_proteome_morphology.cql
// [[torchcell.knowledge_graphs.queries.expression_proteome_morphology]]
//
// Supported query: expression + proteome + morphology (showcase group 2).
//
// Datasets selected (by Dataset.id in the served graph), with the graph_level their
// phenotype class declares in torchcell/datamodels/schema.py:
//   - MicroarrayKemmeren2014Dataset   MicroarrayExpressionPhenotype, 'node'
//   - SmMicroarraySameith2015Dataset  MicroarrayExpressionPhenotype, 'node'
//   - DmMicroarraySameith2015Dataset  MicroarrayExpressionPhenotype, 'node'
//   - ProteomeMessner2023Dataset      ProteinAbundancePhenotype, 'node'
//   - ScmdOhya2005Dataset             CalMorphPhenotype, 'global'
//
// Block shape of experiments/025-solid-growth/queries/001_all_solid_growth.cql: the
// $gene_set parameter (every perturbed gene in the set, at least one perturbation) and
// the property-shaped RETURN (serialized_data strings, never whole nodes). No medium
// filter and no Environment/Media match: the loaders record liquid media for all five (SC for
// Kemmeren and Sameith, SM for Messner, YPD for Ohya), and the query returns them all.
//
// The Kemmeren and Sameith loaders are corrected on PR #460 (channel and strain
// assignment); the served release predates that fix until re-admission.

// MicroarrayKemmeren2014Dataset (MicroarrayExpressionPhenotype, graph_level 'node')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'MicroarrayKemmeren2014Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'node'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// SmMicroarraySameith2015Dataset (MicroarrayExpressionPhenotype, graph_level 'node')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'SmMicroarraySameith2015Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'node'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmMicroarraySameith2015Dataset (MicroarrayExpressionPhenotype, graph_level 'node')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmMicroarraySameith2015Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'node'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// ProteomeMessner2023Dataset (ProteinAbundancePhenotype, graph_level 'node')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'ProteomeMessner2023Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'node'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// ScmdOhya2005Dataset (CalMorphPhenotype, graph_level 'global')
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'ScmdOhya2005Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'global'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

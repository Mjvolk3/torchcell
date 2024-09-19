// Query for the first dataset
// We can order by id which is essentially a random ordering
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'hyperedge'
 AND m.name = 'YEPD'
 AND t.value = 30
 AND ALL(pert IN [(g)<-[:PerturbationMemberOf]-(p) | p]
WHERE pert.perturbation_type = 'deletion'
 AND pert.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(p) | p]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 500000
RETURN e, ref

UNION ALL

// // Query for the second dataset
// // We can order by id which is essentially a random ordering
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmiKuzmin2018Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'hyperedge'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(pert IN [(g)<-[:PerturbationMemberOf]-(p) | p]
// WHERE pert.perturbation_type = 'deletion'
//  AND pert.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(p) | p]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// LIMIT 500000
// RETURN e, ref

// UNION ALL

// // Query third dataset.
// // We can order by id which is essentially a random ordering
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmiCostanzo2016Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'hyperedge'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(pert IN [(g)<-[:PerturbationMemberOf]-(p) | p]
// WHERE pert.perturbation_type = 'deletion'
//  AND pert.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(p) | p]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// LIMIT 550482
// RETURN e, ref

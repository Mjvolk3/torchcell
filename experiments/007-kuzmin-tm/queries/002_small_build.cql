// // SynthLethalityYeastSynthLethDbDataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'SynthLethalityYeastSynthLethDbDataset'
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
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 10
// RETURN e, ref

// UNION ALL

// // GeneEssentialitySgdDataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'GeneEssentialitySgdDataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'node'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 10
// RETURN e, ref

// UNION ALL

// // Singles - SmfCostanzo2016Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'SmfCostanzo2016Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 10
// RETURN e, ref

// UNION ALL

// // Singles - SmfKuzmin2018Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'SmfKuzmin2018Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 10
// RETURN e, ref

// UNION ALL

// // Singles - SmfKuzmin2020Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'SmfKuzmin2020Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 10
// RETURN e, ref

// UNION ALL

// //  DmfCostanzo2016Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmfCostanzo2016Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

// UNION ALL

// // DmfKuzmin2018Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmfKuzmin2018Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

// UNION ALL

// // DmfKuzmin2020Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmfKuzmin2020Dataset'
// MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
// MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
// MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
// MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
// MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
// MATCH (env)<-[:MediaMemberOf]-(m:Media)
// MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
// WHERE phen.graph_level = 'global'
//  AND m.name = 'YEPD'
//  AND t.value = 30
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

// UNION ALL

// TmfKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmfKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'global'
 AND m.name = 'YEPD'
 AND t.value = 30
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
 AND p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 100
RETURN e, ref

UNION ALL

// TmfKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmfKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature)
WHERE phen.graph_level = 'global'
 AND m.name = 'YEPD'
 AND t.value = 30
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
 AND p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 100
RETURN e, ref

UNION ALL

// DmiCostanzo2016Dataset
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
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // // LIMIT 100
// RETURN e, ref

// UNION ALL

// // DmiKuzmin2018Dataset
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
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

// UNION ALL

// // DmiKuzmin2020Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'DmiKuzmin2020Dataset'
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
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

// UNION ALL

// TmiKuzmin2018Dataset
// MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
// WHERE dataset.id = 'TmiKuzmin2018Dataset'
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
//  AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
// WHERE p.perturbation_type = 'deletion'
//  AND p.systematic_gene_name IN $gene_set)
//  AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
// WITH DISTINCT e, ref
//  ORDER BY e.id
// // LIMIT 100
// RETURN e, ref

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
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
 AND p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 100
RETURN e, ref

UNION ALL

// TmiKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2020Dataset'
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
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.perturbation_type = 'deletion'
 AND p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
// LIMIT 100
RETURN e, ref

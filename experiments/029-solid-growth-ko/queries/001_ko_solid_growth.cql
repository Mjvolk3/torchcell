// experiments/029-solid-growth-ko/queries/001_ko_solid_growth.cql
// [[experiments.029-solid-growth-ko.queries.001_ko_solid_growth]]
//
// Every solid-medium growth record in the served graph whose perturbations are ALL
// deletions: fitness (smf/dmf/tmf), gene interactions (dmi/tmi), gene essentiality
// and synthetic lethality across Costanzo 2016, Kuzmin 2018, Kuzmin 2020, SGD and
// SynthLethDB. Same fifteen blocks, media rule and gene-set rule as
// experiments/025-solid-growth/queries/001_all_solid_growth.cql, with two changes:
//
//   - EVERY perturbation must satisfy p.perturbation_type ENDS WITH 'deletion'
//     (sga_kanmx_deletion, sga_natmx_deletion). Temperature-sensitive alleles, DAmP,
//     suppressor and generic alleles are excluded, so an essential gene has no
//     measured allele in this build and its SGD essentiality entry (fitness 0 after
//     conversion) is the only record for it. Measured in the served graph on
//     2026-09-18 (perturbation nodes per dataset): Costanzo 2016 doubles carry
//     7.85M temperature-sensitive, 1.39M DAmP and 0.41M suppressor perturbations
//     beside 31.8M deletions; Kuzmin triples carry 63k TS and 27k allele beside
//     1.09M deletions, so about a fifth of the trigenic rows leave with this rule.
//   - The SynthLethDB block requires TWO perturbation nodes. A record that lists the
//     same gene twice (a single gene stored as a pair; YPL212C, YHL047C, YKL117W in
//     025, where it put two fitness values in one record) has one Perturbation node
//     related twice, so its node count is 1 and it is dropped. Verified on the
//     served graph 2026-09-18.
//
// Temperature is NOT filtered: Costanzo 26 C and 30 C screens and Kuzmin's 26 C
// screens all stay, each as its own entry, and the label policy chooses at read
// time. There is no deduplication stage in this build (see scripts/query.py).

// Singles - SmfCostanzo2016Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'SmfCostanzo2016Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// Singles - SmfKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'SmfKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// Singles - SmfKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'SmfKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmfCostanzo2016Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmfCostanzo2016Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmfKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmfKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmfKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmfKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// TmfKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmfKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// TmfKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmfKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'global'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmiCostanzo2016Dataset (graph_level 'edge' in the served build)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmiCostanzo2016Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'edge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmiKuzmin2018Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmiKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'edge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// DmiKuzmin2020Dataset
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DmiKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'edge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// TmiKuzmin2018Dataset (same record population as 010's live block)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2018Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'hyperedge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// TmiKuzmin2020Dataset (same record population as 010's live block)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'TmiKuzmin2020Dataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'hyperedge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// GeneEssentialitySgdDataset (converted to fitness downstream)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'GeneEssentialitySgdDataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'node'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

UNION ALL

// SynthLethalityYeastSynthLethDbDataset (converted to fitness downstream;
// graph_level 'edge' in the served build)
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'SynthLethalityYeastSynthLethDbDataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media)
WHERE phen.graph_level = 'edge'
 AND m.state = 'solid'
 AND ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set
       AND p.perturbation_type ENDS WITH 'deletion')
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) = 2
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized

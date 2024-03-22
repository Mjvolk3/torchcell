MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e, perturbations
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global'
AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
AND phen.fitness_std < 0.15
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
MATCH (ref)<-[:EnvironmentMemberOf]-(ref_env:Environment)
MATCH (ref_env)<-[:MediaMemberOf]-(ref_m:Media {name: 'YEPD'})
MATCH (ref_env)<-[:TemperatureMemberOf]-(ref_t:Temperature {value: 30})
WITH e, ref, perturbations
WHERE ALL(p IN perturbations WHERE p.systematic_gene_name IN $gene_set)
RETURN e, ref
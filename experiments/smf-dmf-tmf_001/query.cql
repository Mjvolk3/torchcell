MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE 
    ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
WITH e, COLLECT(g) AS genotypes
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
    AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
    AND phen.fitness_std < 0.005
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN e, ref
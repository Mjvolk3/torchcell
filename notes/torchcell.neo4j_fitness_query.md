---
id: 8t4hjrl71hnr0iocad2je2j
title: Neo4j_fitness_query
desc: ''
updated: 1710538608925
created: 1706648223300
---
## Simple counting of experiment nodes

```bash
MATCH (e:Experiment)
RETURN COUNT(e) AS TotalExperiments
```

## Smf Dmf Benchmark Query Results on Small Knowledge Graph

[[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]

This graph only uses `1e6` Dmf to keep things smallish for local builds.

Query to count all data

```bash
MATCH (e:Experiment)
RETURN COUNT(e) AS TotalExperiments
>>> 1523533
```

Query to count selection:

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf')
RETURN COUNT(DISTINCT e) AS ExperimentCount
433940
```

Query to get data:

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf')
RETURN e, ref
```

Query with std constraints:

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
AND (phen.label = 'smf' OR phen.label = 'dmf')
AND phen.fitness_std < 0.05
RETURN COUNT(e) AS TotalExperiments
277363
```

Query with added temperature and media constraints.

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
AND (phen.label = 'smf' OR phen.label = 'dmf')
AND phen.fitness_std < 0.05
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN e, ref
RETURN COUNT(e) AS TotalExperiments
```

## Smf Dmf Tmf Benchmark Query Results on Small Knowledge Graph

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf' or phen.label = 'tmf')
RETURN COUNT(DISTINCT e) AS ExperimentCount
509858
```

Added 75,918 triple gene deletions.

Quey to get data:

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' AND (phen.label = 'smf' OR phen.label = 'dmf' or phen.label = 'tmf')
RETURN e, ref
```

Query with std constraints:

``` bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
AND phen.fitness_std < 0.05
RETURN COUNT(e) AS TotalExperiments
309248
```

Query with added temperature and media constraints.

```bash
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
AND phen.fitness_std < 0.05
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN COUNT(e) AS TotalExperiments
```

```bash
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global' 
AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = "tmf")
AND phen.fitness_std < 0.05
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
RETURN COUNT(e) AS TotalExperiments
```

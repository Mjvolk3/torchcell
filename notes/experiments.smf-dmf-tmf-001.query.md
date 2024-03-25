---
id: 34jdlcd1bhq2o5agu6ug0y6
title: Query
desc: ''
updated: 1710987251178
created: 1710987039912
---

## 2024.03.20 - Current Query

- When we didn't put the restrictions on ref, we got some `ref` at 26C and some and some `e` at this ref dragged along, which is strange because they should have `Temperature.value=26`

```cypher
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
```

## 2024.03.20 - A Proposed Query to Solve the ref Issue

```cypher
MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
WITH e, g, COLLECT(p) AS perturbations
WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
WITH DISTINCT e, perturbations
MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
WHERE phen.graph_level = 'global'
  AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
  AND phen.fitness_std < 0.15
WITH e, perturbations
WHERE ALL(p IN perturbations WHERE p.systematic_gene_name IN $gene_set)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
RETURN e, ref
```

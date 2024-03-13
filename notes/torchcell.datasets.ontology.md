---
id: 4h1tibaou9xlykrxy3gwh2g
title: Ontology
desc: ''
updated: 1701564169649
created: 1700002180524
---
## Introduction to Basic Formal Ontology

[Introduction to Basic Formal Ontology](https://www.youtube.com/watch?v=p0buEjR3t8A)
- Terms in an ontology should only contain singular nouns - each term in an ontology represent one universal
- Classes should form a DAG


## Neo Semantics Import of Ontology Only Support Specific Strucutres

1. Named class (category) declarations with both `rdfs:Class` and `owl:Class`.
2. Explicit class hierarchies defined with `rdf:subClassOf` statements.
3. Property definitions with `owl:ObjectProperty`, owl:DatatypeProperty and `rdfs:Property`
4. Explicit property hierarchies defined with `rdfs:subPropertyOf` statements.
5. Domain and range information for properties described as `rdfs:domain` and `rdfs:range` statements.


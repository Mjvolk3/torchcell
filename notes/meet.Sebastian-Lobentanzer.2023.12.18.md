---
id: miusjrp9f93a0zjdss9lrj7
title: '18'
desc: ''
updated: 1705661546021
created: 1702920282578
---

Questions:

- [x] Suggestions for removing duplicates.
  - Answer: This is really for us to decided when the removal of duplicates is necessary. Could let `Biocypher` do it, or could handle it explicitly.
- [x] Ask about difference between property query v node query. Media v Temperature.
  - There are two different philosophies, the subtype philosophy and the instance philosophy. To choose one it is best to consider the types of queries that will be run over the graph. Biocypher is more inline with the instance philosophy and I think this aligns well with our goals considering we don't know excactly how the hierarchy will change.

## ChatGPT on Subtype vs. Instance

### Subtype Philosophy in Neo4j

In the subtype philosophy, you would use labels to define a hierarchy of types. Neo4j allows nodes to have multiple labels, enabling the representation of subtype relationships.

**Example**: Modeling biological organisms

- **Base Type**: `Organism`
- **Subtypes**: `Plant`, `Animal`
- **Further Subtypes for Animal**: `Mammal`, `Bird`

```cypher
// Create base type Organism
CREATE (:Organism {name: 'Generic Organism'})

// Create subtype Plant
CREATE (:Plant:Organism {name: 'Generic Plant'})

// Create subtype Animal and further subtypes Mammal and Bird
CREATE (:Animal:Organism {name: 'Generic Animal'}),
       (:Mammal:Animal:Organism {name: 'Generic Mammal'}),
       (:Bird:Animal:Organism {name: 'Generic Bird'})
```

In this model, `Plant`, `Animal`, `Mammal`, and `Bird` are all subtypes of `Organism`. `Mammal` and `Bird` are further specialized types of `Animal`. This hierarchical model is useful for queries that need to consider entities at different levels of specificity.

### Instance Philosophy in Neo4j

In the instance philosophy, each node is considered a unique instance, and the relationships and properties are emphasized over the type hierarchy.

**Example**: Research samples in a lab

- **Instances**: Individual organisms, each with unique data
- **Relationships**: Can include relationships to experiments, other organisms, etc.

```cypher
// Create individual instances
CREATE (o1:Organism {name: 'Organism 1', id: 1}),
       (o2:Organism {name: 'Organism 2', id: 2})

// Create relationships to experiments
CREATE (e1:Experiment {name: 'Experiment A'}),
       (e2:Experiment {name: 'Experiment B'})

// Link organisms to experiments
CREATE (o1)-[:PART_OF]->(e1),
       (o2)-[:PART_OF]->(e2)
```

In this model, each organism is an instance with its unique identifier and properties. The relationships, such as `PART_OF`, link these instances to experiments or other relevant entities.

### Conclusion

- **Subtype Philosophy**: Ideal for scenarios requiring a clear hierarchical classification of entities (like taxonomic classification in biology).
- **Instance Philosophy**: Better suited for cases where the focus is on the individual characteristics and relationships of each entity (like tracking individual samples or experiments in a lab).

Neo4j's flexibility allows it to accommodate both philosophies, often blending them as needed for a given application.

- 🔲 Add all data, or just the data instances?
  - Never asked, but the answer would really be, it depends on the type of problems we are trying to model.
- [x] What happens when we create an edge to a duplicated node? → Messaged on [Zulip](https://biocypher.zulipchat.com/#narrow/dm/590747-Sebastian-Lobentanzer) about this. → Don't have to worry about this because after deduplication the edges is connected to the same node.

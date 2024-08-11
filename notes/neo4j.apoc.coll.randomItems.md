---
id: fstw7q1ufyu7cqi477ha0xa
title: randomItems
desc: ''
updated: 1723348207649
created: 1723348126279
---
Non-reproducible random query. We could always gather and `experiment.id` list to reproduce the original query.

```cypher
// Query all experiments
MATCH (e:Experiment)
WITH collect(e) AS experiments

// Use apoc.coll.randomItems to select 100 random experiments
WITH apoc.coll.randomItems(experiments, 100, false) AS randomExperiments

// Unwind the collection to return individual experiments
UNWIND randomExperiments AS experiment

// Return the experiment properties you're interested in
RETURN experiment.id
LIMIT 2;

// Output
// experiment.id
// "725f6b84f2c6725a8eb5c02effa33129564729ee2779b3ec0c078011fb8d7b26"
// "273ff6f69e98d1a2865a6e7e8856578dacbc30d349738285a37bd5271c734fae"
```

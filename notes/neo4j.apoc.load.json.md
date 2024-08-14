---
id: pukn4qve8yg1z41y9zdnw21
title: Json
desc: ''
updated: 1723595497598
created: 1723595454237
---

If you save gene_set.json in volume mapped `import` dir. You can run the following. This is nice for debugging queries prior to running.

```cypher
CALL apoc.load.json("file:///gene_set.json") YIELD value
WITH value.result AS gene_set
RETURN gene_set;
```

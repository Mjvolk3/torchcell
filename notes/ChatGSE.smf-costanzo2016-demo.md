---
id: e9t31mxl5d35r2ad3anhnz5
title: smf-costanzo2016-dem
desc: ''
updated: 1705028983257
created: 1703980111875
---

## SMF Costanzo 2016 Query Example

![](./assets/images/ChatGSE.smf-costanzo2016-demo.md.query-example.png)

![](./assets/images/ChatGSE-query-example.gif)

```Cypher
MATCH (g:Genotype)-[:GenotypeMemberOf]->(e:Experiment)
WHERE g.perturbation_type = 'gene deletion'
MATCH (e)-[:ExperimentMemberOf]->(d:Dataset)
MATCH (e)-[:ExperimentMemberOf]->(t:Temperature)
WHERE t.scalar = 30 AND t.description = 'degrees celsius'
RETURN e
```
o
---
id: n51wqr0f4p24e6o9rvkvw54
title: smf-costanzo2016-dem
desc: ''
updated: 1705648644266
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

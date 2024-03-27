---
id: xtvbc1lz7aq62xwwam0h1kc
title: Create_scerevisiae_kg_small
desc: ''
updated: 1711386983899
created: 1709787278856
---

## 2024.03.25 - Query to Check Database Edges

Used this query [[2024.03.21 - Query that Identified ExperimentReferenceOfCount Issue in Database|dendron://torchcell/cypher-queries#20240321---query-that-identified-experimentreferenceofcount-issue-in-database]] and we find that we have all of the proper edges.

```cypher
MATCH (d:Dataset)
WHERE d.id IN ["SmfCostanzo2016Dataset","DmfCostanzo2016Dataset", "SmfKuzmin2018Dataset", "DmfKuzmin2018Dataset", "TmfKuzmin2018Dataset"]
OPTIONAL MATCH (d)<-[r1:ExperimentMemberOf]-(e:Experiment)
WITH d, COUNT(r1) AS ExperimentMemberOfCount
OPTIONAL MATCH (d)<-[:ExperimentMemberOf]-(e:Experiment)<-[r2:ExperimentReferenceOf]-(ref:ExperimentReference)
RETURN d.id AS DatasetID, ExperimentMemberOfCount, COUNT(DISTINCT r2) AS ExperimentReferenceOfCount;
+---------------------------------------------------------------------------------+
| DatasetID                | ExperimentMemberOfCount | ExperimentReferenceOfCount |
+---------------------------------------------------------------------------------+
| "SmfCostanzo2016Dataset" | 20484                   | 20484                      |
| "SmfKuzmin2018Dataset"   | 1539                    | 1539                       |
| "DmfKuzmin2018Dataset"   | 410399                  | 410399                     |
| "TmfKuzmin2018Dataset"   | 91111                   | 91111                      |
| "DmfCostanzo2016Dataset" | 1000000                 | 1000000                    |
```

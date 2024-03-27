---
id: ydbxmqdyihrdq3031le22al
title: Cypher Queries
desc: ''
updated: 1711386752708
created: 1710904047926
---

## 2024.03.21 - Query that Identified ExperimentReferenceOfCount Issue in Database

We have a mismatch in this row that was due trying to compute the `experiment_reference_index` in parallel. We now just compute serially and run an assertion check that the index is correct. [[torchcell.dataset.experiment_dataset]]

```cypher
neo4j@torchcell> MATCH (d:Dataset)
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
| "DmfCostanzo2016Dataset" | 20705612                | 647050                     |
+---------------------------------------------------------------------------------+
```


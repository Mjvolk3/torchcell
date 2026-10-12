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

## 2026.10.12 - `--organisms`: building one organism's datasets

`__main__` takes `--organisms <set>` off `sys.argv` (`take_organisms_flag`, after `take_include_private_flag`) into the module global `ORGANISMS`; every other entry sees every organism. In `main`:

- the visible map is the public map, unioned with the private map under `--include-private`, as before;
- with a `datasets` list, `refuse_organism_mismatch` names any listed class outside `ORGANISMS` (after the private refusal), so an increment cannot name a dataset the build was not asked to serve;
- the build map is `select_organisms(visible, ORGANISMS)`, so `datasets: null` builds every mapped dataset of the asked organisms;
- the membership line is printed (it was only logged), `Build membership: ALL (include_private=True, organisms=scerevisiae, 67 mapped datasets)`, so the generation log the live rebuild reads names the graph it wrote.

Tests: `test_an_organism_set_narrows_the_full_build_and_the_log_names_it` and `test_a_named_dataset_outside_the_organism_set_is_refused_by_name` in `tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py`. The shared `FakeDataset` stand-in now declares `organism = Organism.scerevisiae`, since every build reads it off the class.

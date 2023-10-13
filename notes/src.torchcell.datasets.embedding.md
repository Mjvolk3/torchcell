---
id: z5hyzc2e7haaa4zkk417q1m
title: Embedding
desc: ''
updated: 1697130190970
created: 1692275134695
---

## Disabling Genome Due to Sqlite and Parsed Genome

We commented out genome. I am not sure if this is necessary because returning `None` in `parse_genome` method in [[Parse Genome and Return None For Dunder Add|dendron://torchcell/src.torchcell.datasets.fungal_up_down_transformer#parse-genome-and-return-none-for-dunder-add]] solved the issue. It would be worth seeing if we could pass `genome` as it could be useful for data joins. Obviously the only reason we could comment it out is because it is not yet being used for joins.

## Difficult to Add Datasets with Sum operator

`a + b` and `sum([a, b])` seem to work slightly different in python. `sum([a, b])` might use `__radd__` and has and tries to add 0 first. I think to use `sum` with will need a dummy data.

Also another issue is that when adding datasets if one of the datasets is not of the same size. It's data does not get combined.

---
id: 3h2mege61o7winu1x2j0w9j
title: Cell_adapter
desc: ''
updated: 1721797777190
created: 1711381015227
---

## 2024.07.18 - Config Refactor Time Test

`CellAdaptor` with `yaml` config refactor

![](./assets/images/torchcell.adapters.cell_adapter.md.cell-adaptor-with-yaml-config-refactor.png)

`461 s` or `7.68 m`

`CellAdaptor` without `yaml` config refactor

![](./assets/images/torchcell.adapters.cell_adapter.md.cell-adaptor-without-yaml-config-refactor.png)

`545 s` or `9 m`

The few min discrepancy is noise. They are equivalent so we will make the refactor.

## 2024.07.23

A key aspect of `CellAdapter` methods is that they should be defined in a minimal manner to forcing deduplication upon graph import. If we add metadata to say the `experiment`, or `experiment reference` nodes then duplicate measurement data will not be deduplicated on import.

Most often queries will not take place directly at the experiment level but levels above or below. Also experiments should be considered primary, where as references are primary purpose is for harmonization. In other words it is expected that almost if not all experiments will be unique, whereas we want a lot of references to be duplicated. This will allow for queries directed at references. e.g. "get all data with this same experimental reference." In the case of gene essentiality collected from primary sources we would rather them just get deduplicated on import... 😠 No good answer. I don't think we need the dataset name in the experiment object, as long as it is in graph it can be used for querying and that is really all we need it for.

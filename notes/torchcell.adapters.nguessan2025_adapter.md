---
id: 0blybq6fgrk5wrb15q3pixj
title: Nguessan2025_adapter
desc: ''
updated: 1791733649770
created: 1791733649770
---

## 2026.10.11 - Adapter and conf, not yet in the map

`Nguessan2025Adapter` (`torchcell/adapters/nguessan2025_adapter.py`) and `conf/nguessan2025_adapter.yaml` serve [[torchcell.datasets.scerevisiae.nguessan2025]] with the method list of [[torchcell.adapters.boocock2025_adapter]]: segregant genotype nodes, the single cell expression phenotype pair, no perturbation or environment-perturbation methods. The temperature methods stay listed; they emit nothing for these records because temperature is a typed gap, which `CellAdapter._temperature_node` already handles. A gapped stage partition projects as the JSON string `null`.

The adapter is exported from `torchcell.adapters` (`segregant_adapters`) but NOT entered in `dataset_adapter_map`: the map is the rebuild membership list, and this dataset's store cannot be built until the walled Dryad genotypes are deposited (#1014). Its construction test names the dataset pairing itself, since the shared harness (`_adapter_init_harness.paired_dataset_class`) pairs through the map.

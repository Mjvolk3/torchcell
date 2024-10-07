---
id: h59o9thoyum3pqfxrxqbrcu
title: '40'
desc: ''
updated: 1728315170988
created: 1727811879911
---
## 2024.10.01

- [x] Composition converter. → We have a decent composite strategy for now [[torchcell.datamodels.fitness_composite_conversion]]
- [x] Handle `None` in deduplicate. → We seem to already be handing `None` fine. → [[torchcell.data.mean_experiment_deduplicate]]
- [x] Unique naming of statistic field. → we also did not remove `dataset_name` from the reference like we were thinking of doing because the uniqueness of the reference would be maintained without the `dataset_name`. References can be converted to be unified.
- [x] Rebuild database.

## 2024.10.02

- [x] GH maintenance.

## 2024.10.03

- [x] Synthetic Lethality phenotypes not linked to query. Breaks `PhenotypeMemberOf` relation. → Checked two of these broken edge types.
- [x] `SynthLethalityYeastSynthLethDbDataset` is a missing node. Breaks `ExperimentMemberOf` relation.
- [x] `Mentions` relation is broken. All the `to` and `from` look mostly different and this is really only in the case of a secondary data source like synthetic lethality.
- [x] Fixed typo in `cell_adapter`
- [x] Test smaller build with just synthetic lethality. Now import errors.
- [x] Run small build again with synthetic lethality fix.

## 2024.10.04

- [x] Query dataset → failed due to typos. fixed. →failed rerun due to low memory. → rerun

## 2024.10.05

- [x] Meet with Rana → Discussed making dataset easily accessible.
- [x] Database failed due to low memory. → Reran after deleting `database/biocypher-out`

## 2024.10.06

- [x] Make CellModule more robust and reproducible with pydantic data passing

---
id: 8uokcgutd227uercrfpk2sd
title: Dataset_adapter_map
desc: ''
updated: 1791801771706
created: 1791801771706
---

## 2026.10.12 - The organism switch

A build used to serve every class in `dataset_adapter_map` (plus `PRIVATE_DATASET_ADAPTER_MAP` under `--include-private`), so a yeast-only graph was not buildable. The organism switch narrows the map the same way the visibility gate does, reading the class and never a config.

- Every mapped class declares `organism: ClassVar[Organism]` (`torchcell.data.experiment_dataset.Organism`: `scerevisiae`, `ecoli`, `pputida`). [[torchcell.data.experiment_dataset]] has the enum and why the base class carries no value.
- `take_organisms_flag(argv)` removes `--organisms <names>` (or `--organisms=<names>`) from `sys.argv` before hydra parses it and returns the parsed set; absent means every organism. A flag with no value, a flag given twice and an unknown name raise `ValueError`.
- `select_organisms(adapter_map, organisms)` keeps the entries whose class declares an organism in the set. It reads every class through `dataset_organism`, so a mapped class with no declaration stops every selection, the full one included.
- `build_adapter_map(include_private, organisms)` is the public map (or the union with the private map) passed through `select_organisms`; the default `organisms` is every organism, so existing callers are unchanged.
- `refuse_organism_mismatch(classes, organisms)` raises `OrganismRefused` naming each configured class outside the set with its organism, the set, and the flag. It is the twin of `refuse_private_datasets`: nothing named is dropped silently.

Map sizes from a dry run of the generator's own selection (`take_include_private_flag`, `take_organisms_flag`, `select_organisms` on the merged tree at `eb07bd213` plus this branch):

| `--include-private` | `--organisms` | mapped datasets |
|---|---|---:|
| no | (absent) `scerevisiae,ecoli,pputida` | 152 |
| no | `scerevisiae` | 66 |
| no | `ecoli` | 58 |
| no | `pputida` | 28 |
| no | `ecoli,pputida` | 86 |
| yes | (absent) | 153 |
| yes | `scerevisiae` | 67 |
| yes | `ecoli` | 58 |
| yes | `pputida` | 28 |

The private map holds one class (`InhibitorBioscreenVolk2021Dataset`), which is yeast, so it appears only in the yeast and the full sets. The per-organism maps partition the full map in both modes (`tests/torchcell/datasets/test_dataset_organism.py::test_the_organism_maps_partition_the_full_map`). The gate itself is pinned in `tests/torchcell/knowledge_graphs/test_dataset_organism_gate.py`.

---
id: tt069dm8v74kuh8drjk9xo8
title: Generation_tiers.yaml
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - The resource tiers of the generation array

`torchcell/knowledge_graphs/conf/generation_tiers.yaml` keys adapter class names to a tier and gives each tier its slurm reservation, container limits, time limit and concurrency cap. The sizes are the overestimated ones of the Path U decision ([[scratch.2026.10.15.ontology-unification-decision]], "Slurm sizing, overestimated so nothing ends short"):

| tier | cpus | memory | time | `%N` | members |
|---|---|---|---|---|---|
| standard | 16 | 64 G | 8 h | 4 | every adapter not listed |
| large | 32 | 160 G | 16 h | 2 | Costanzo DMF and DMI, Hillenmeyer het and hom, Vaishnav 2022, Hale 2024, Piotrowski 2017, Caudal 2024, Peter 2018, the two Muenzner 2024, Nadal-Ribelles 2025 |
| hoepfner | 64 | 320 G | 36 h | 1 | EnvChemgenHoepfner2014Adapter |

To move an adapter, edit `adapters:`; `tests/torchcell/knowledge_graphs/test_generation_array.py` fails on a name that is not a mapped adapter class and pins the sizes above. The table is read by `generation_array.load_tier_table` (`TierTable`, pydantic), and the conductor passes `--tiers` to point at another file for a measurement.

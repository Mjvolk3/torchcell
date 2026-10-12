---
id: yyoryvrut55twthnatab8c0
title: Fitness_browser_inventory
desc: ''
updated: 1791803967375
created: 1791803967375
---

## 2026.10.12 - G3 inventory of the Fitness Browser deposit

`experiments/database/scripts/fitness_browser_inventory.py` opens the pinned `feba.db` read-only (sha256 re-verified unless `--no-verify`, which still checks the size and says so in the JSON) and writes, per organism and per experiment, the publication, set, description, group, condition columns, medium, temperature, library, start date, generations, loci with both a fit and a t value, Price 2018 Table S5 membership (joined on `orgId` and the experiment name over every organism of the table) and the served store that keeps the sample, computed through the loaders' own rules from their pinned release bytes (`price2018.classify_samples`, `wetmore2015.classify_samples`, `borchert2024.drop_reason`).

```bash
python experiments/database/scripts/fitness_browser_inventory.py
```

Outputs (untracked, `experiments/database/results/` is generated): `fitness_browser_inventory.json`, `fitness_browser_inventory_tables.md`. The tables and what they establish are in [[torchcell.candidates.aggregation-decomposition]] (2026.10.12). Measured 2026-10-12 with the sha256 verified: 72 organisms, 10,053 experiments, 28 publications, 4,797 Table S5 rows present; Keio 183 experiments over 3,789 loci, Putida 314 over 4,778.

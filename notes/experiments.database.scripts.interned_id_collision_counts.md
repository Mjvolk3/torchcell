---
id: 2y7fvvi4x5oww7ngl815zzc
title: Interned_id_collision_counts
desc: ''
updated: 1792072582832
created: 1792072582832
---

## 2026.10.15 - The interned-id collision counted on the job 4901 partial

`experiments/database/scripts/interned_id_collision_counts.py` reads a BioCypher output directory and counts, independently of lane H's attribution script, the genotype nodes an output lost to the interned-id collision ([[torchcell.datamodels.interned_constant]]) and, in the other direction, the Experiment blob pointers with no constant. Run on `/db/database/biocypher-out/2026-10-14_03-53-33` (read-only; the pointers come from a `grep` over the 261 GB Experiment file, 2 m 20 s user, then 133 s for the script). Result: `experiments/database/results/interned_id_collision_4901.json` (tracked; the partial is deleted after the build).

| measure | value |
|---|---|
| InternedConstant rows by kind | environment 24,591; genotype 218,258; genotype.background 135 |
| genotype-kind ids also in a genotype node file | 0 (the sink wrote each id once) |
| PerturbationMemberOf rows (end side) | 47,251,386 total; 121,180 end at an interned-only id (121,143 distinct); 1,931 at nothing |
| GenotypeMemberOf rows (start side) | 95,910,118 total; 1,948,098 start at an interned-only id (218,258 distinct); 22,241 at nothing |
| SegregantBackgroundOf rows (start side) | 894,907 total; 709,716 start at an interned-only id (135 distinct); 185,191 at nothing (the `SegregantGenotype` file is empty on the partial) |
| blob pointers | genotype 1,948,098; genotype.background 6,629,132; every one resolves to an interned constant |

Per dataset, genotype nodes lost: CrispriSegregantHale2024 121,140; Boocock2025 79,443; Bloom2019 13,720; Nguessan2025 2,948; IsolateGrowthPeter2018 971; CaudalPanTranscriptome2024 943; CellYieldDeChiara2022 906; DoublingTimeDeChiara2022 906; LifeCycleDeChiara2022 905; ProteomeMuenzner2024 794; ProteinTurnoverMuenzner2024 55; the four Brunk2016 leaves 2 each; GrowthRateChoe2019 1; 135 unattributed (the Hale backgrounds, which have no `GenotypeMemberOf` row). The rows at nothing are the partial's unflushed buffer tails, lane H's finding, and are not this defect.

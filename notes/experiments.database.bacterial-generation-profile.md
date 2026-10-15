---
id: 2w8bp4dgtp9ge4jj9xotv5p
title: Bacterial Generation Profile
desc: ''
updated: 1792098419171
created: 1792098419171
---

## 2026.10.15 - The 46 adapters job 4901 never reached, measured through the generation array

Job 4901, the serial KG 4.0 generation, walked positions 0 to 123 of the build order and
died on its 12 h slurm limit, so the 46 adapters after `EnvChemgenShiver2016Adapter`
(index 123) had never been generated once, in any configuration, and the standard tier's
own comment in [[torchcell.knowledge_graphs.conf.generation_tiers.yaml]] said their
per-task peak was unmeasured. Lane P listed them as the largest open gap before the 4.0
build ("the other 45 bacterial adapters' generation peaks"). This lane generated all 46
through the per-adapter array ([[torchcell.knowledge_graphs.generation_array]]), plus
index 108 (`PhageRbTnseqMutalik2020Adapter`), which job 4901 generated only inside its
cumulative container and which lane P flagged at 37 G of store-build RSS against the
64 G standard tier.

**All 47 tasks COMPLETED; 47 of 47 directories carry a DONE marker and none left an
`.incomplete` directory, so nothing here is projected over a partial set.** Job 4974,
array `108,124-169%3`, 16 CPU and 64 G per task, `tc-neo4j-gen:0a238c9679fb`, commit
`0a238c9679fbcd1be07723e52561af3d09d42962`, torchcell 1.7.5, output
`/scratch/projects/torchcell/database/bacterial-gen-profile/biocypher-out/bacterial-gen-2026-10-15`
(19 GB, a scratch directory, never a build directory).

Script: `experiments/database/scripts/bacterial_generation_profile.py`
([[experiments.database.scripts.bacterial_generation_profile]]), which recomputes the
membership from `build_adapter_map` every run and reads only what the array wrote.
Results: `experiments/database/results/bacterial_generation_profile_bacterial-tail.json`
and `experiments/database/results/bacterial_generation_csv_endpoint_gate.json`.

### How far job 4901 actually got, measured

The prose of this program says "124 adapters" in one place and "adapter 65 of 170" in
another, so it was measured rather than repeated. Folding job 4901's surviving
`resource_samples.csv` (`/db/database/biocypher-out/2026-10-14_03-53-33`) against the
build order: **100 adapters produced at least one 5 s sample, the highest build index
sampled is 122 (`MetabolomeSchastnaya2021Adapter`), 23 indices at or below it produced no
sample at all (adapters that finish inside one sampling interval, index 123 among them),
and no index at or above 124 was ever sampled.** Both numbers are right about different
things: 124 positions were walked, and the 65 adapter class names the generate log prints
are a logging artifact, not the progress. The tail of 46 is confirmed untouched.

### The 46, and why the prose list is one short

The lane reports list the tail by author as Nichols, Schmidt x5, Tong, Wang x5, Fang,
Banerjee, Borchert x3, Fitness Browser, Carruthers x4, DeSiqueira x4, Kang, Thompson x2,
Lim x4, Menasalvas x4, Royet x2, Yunus x5 and the private Bioscreen, which is **45**. The
map has 46: the list carries one `Banerjee` where there are two separate stores,
`ProteomeBanerjee2025Adapter` at 137 (E. coli) and `IsoprenolTiterBanerjee2024Adapter` at
151 (P. putida). Every adapter of the tail is in the `standard` tier; the tier table keys
no bacterial adapter to `large` or `hoepfner`.

#### The membership, from `build_adapter_map`

| index | dataset | adapter | tier |
|---:|---|---|---|
| 124 | EnvChemgenNichols2011Dataset | EnvChemgenNichols2011Adapter | standard |
| 125 | ProteomeSchmidt2016Dataset | ProteomeSchmidt2016Adapter | standard |
| 126 | ProteomeSrmSet1Schmidt2016Dataset | ProteomeSrmSet1Schmidt2016Adapter | standard |
| 127 | ProteomeSrmSet2Schmidt2016Dataset | ProteomeSrmSet2Schmidt2016Adapter | standard |
| 128 | GrowthRateSchmidt2016Dataset | GrowthRateSchmidt2016Adapter | standard |
| 129 | GrowthRateS23Schmidt2016Dataset | GrowthRateS23Schmidt2016Adapter | standard |
| 130 | CarbonSourceTong2020Dataset | CarbonSourceTong2020Adapter | standard |
| 131 | EnvChemgenWang2015Dataset | EnvChemgenWang2015Adapter | standard |
| 132 | GrowthWang2015Dataset | GrowthWang2015Adapter | standard |
| 133 | CrispriGuideFfaEnrichmentFang2025Dataset | CrispriGuideFfaEnrichmentFang2025Adapter | standard |
| 134 | CrispriGuideFitnessWang2018Dataset | CrispriGuideFitnessWang2018Adapter | standard |
| 135 | CrispriGeneFitnessWang2018Dataset | CrispriGeneFitnessWang2018Adapter | standard |
| 136 | EnvChemgenWang2024Dataset | EnvChemgenWang2024Adapter | standard |
| 137 | ProteomeBanerjee2025Dataset | ProteomeBanerjee2025Adapter | standard |
| 138 | RbTnseqBorchert2023Dataset | RbTnseqBorchert2023Adapter | standard |
| 139 | FitnessContrastBorchert2023Dataset | FitnessContrastBorchert2023Adapter | standard |
| 140 | RbTnseqBorchert2024Dataset | RbTnseqBorchert2024Adapter | standard |
| 141 | FitnessBrowserPutidaDataset | RbTnseqFitnessBrowserPutidaAdapter | standard |
| 142 | IsoprenolTiterCarruthers2025Dataset | IsoprenolTiterCarruthers2025Adapter | standard |
| 143 | ProteomeCarruthers2025Dataset | ProteomeCarruthers2025Adapter | standard |
| 144 | CampaignProteomeCarruthers2025Dataset | CampaignProteomeCarruthers2025Adapter | standard |
| 145 | ProteomeFoldChangeCarruthers2025Dataset | ProteomeFoldChangeCarruthers2025Adapter | standard |
| 146 | ProteomeDeSiqueira2025Dataset | ProteomeDeSiqueira2025Adapter | standard |
| 147 | ProteomePercentDeSiqueira2025Dataset | ProteomePercentDeSiqueira2025Adapter | standard |
| 148 | ProteomeLog10PercentDeSiqueira2025Dataset | ProteomeLog10PercentDeSiqueira2025Adapter | standard |
| 149 | IsoprenolTiterDeSiqueira2025Dataset | IsoprenolTiterDeSiqueira2025Adapter | standard |
| 150 | IsoprenylAcetateTiterKang2026Dataset | IsoprenylAcetateTiterKang2026Adapter | standard |
| 151 | IsoprenolTiterBanerjee2024Dataset | IsoprenolTiterBanerjee2024Adapter | standard |
| 152 | ValerolactamTiterThompson2019Dataset | ValerolactamTiterThompson2019Adapter | standard |
| 153 | LactamGrowthRateThompson2019Dataset | LactamGrowthRateThompson2019Adapter | standard |
| 154 | PutidaPrecise321Lim2022Dataset | PutidaPrecise321Lim2022Adapter | standard |
| 155 | IsoprenolToleranceLim2025Dataset | IsoprenolToleranceLim2025Adapter | standard |
| 156 | ProteomeLim2025Dataset | ProteomeLim2025Adapter | standard |
| 157 | ProteomeFoldChangeLim2025Dataset | ProteomeFoldChangeLim2025Adapter | standard |
| 158 | IsoprenolSelectionMenasalvas2025Dataset | IsoprenolSelectionMenasalvas2025Adapter | standard |
| 159 | ProteomeMenasalvas2025Dataset | ProteomeMenasalvas2025Adapter | standard |
| 160 | MetaboliteGrowthPhaseMenasalvas2025Dataset | MetaboliteGrowthPhaseMenasalvas2025Adapter | standard |
| 161 | MetaboliteProductionPhaseMenasalvas2025Dataset | MetaboliteProductionPhaseMenasalvas2025Adapter | standard |
| 162 | EnvMetalTnseqRoyet2025Dataset | EnvMetalTnseqRoyet2025Adapter | standard |
| 163 | GradedGeneEssentialityRoyet2025Dataset | GradedGeneEssentialityRoyet2025Adapter | standard |
| 164 | CrispriArrayYunus2026Dataset | CrispriArrayYunus2026Adapter | standard |
| 165 | CrispriDifferentialProteomeYunus2026Dataset | CrispriDifferentialProteomeYunus2026Adapter | standard |
| 166 | CrispriKnockdownYunus2026Dataset | CrispriKnockdownYunus2026Adapter | standard |
| 167 | IsoprenolTiterYunus2026Dataset | IsoprenolTiterYunus2026Adapter | standard |
| 168 | CrispriPanelProteomeYunus2026Dataset | CrispriPanelProteomeYunus2026Adapter | standard |
| 169 | InhibitorBioscreenVolk2021Dataset | InhibitorBioscreenVolk2021Adapter | standard |

### Per adapter, measured

Every number in this table is measured by job 4974's own task: `wall s` and `peak GiB`
are the DONE marker's `generation_wall_s` and `peak_memory_gb`, the row counts are the
sink's, and `cores mean` and `recycles` are folded from that task's 5 s cgroup samples
and its generate log. A dash means the task finished inside fewer than two sampling
intervals, so there is no fold to report; those tasks also read 0.00 GiB, which is the
sampler never having run rather than a zero peak.

Totals: **4,685,825 records, 9,275,542 node rows and 28,353,191 edge rows in 1,591 s of
generation across the 47 tasks** (the 46 of the tail alone: 4,398,010 records, 1,513 s).


| index | adapter | records | node rows | edge rows | wall s | peak GiB | % of tier | cores mean | recycles |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 108 | PhageRbTnseqMutalik2020Adapter | 287,815 | 583,435 | 1,731,128 | 78 | 9.85 | 15.4% | 13.0 | 0 |
| 124 | EnvChemgenNichols2011Adapter | 1,286,962 | 2,582,031 | 7,728,021 | 206 | 11.73 | 18.3% | 14.0 | 0 |
| 125 | ProteomeSchmidt2016Adapter | 21 | 112 | 199 | 2 | 0.00 | 0.0% | - | - |
| 126 | ProteomeSrmSet1Schmidt2016Adapter | 18 | 97 | 170 | 1 | 0.00 | 0.0% | - | - |
| 127 | ProteomeSrmSet2Schmidt2016Adapter | 21 | 110 | 195 | 1 | 0.00 | 0.0% | - | - |
| 128 | GrowthRateSchmidt2016Adapter | 6 | 33 | 53 | 1 | 0.00 | 0.0% | - | - |
| 129 | GrowthRateS23Schmidt2016Adapter | 26 | 121 | 235 | 1 | 0.00 | 0.0% | - | - |
| 130 | CarbonSourceTong2020Adapter | 111,420 | 230,282 | 672,564 | 54 | 8.82 | 13.8% | 12.3 | 0 |
| 131 | EnvChemgenWang2015Adapter | 46 | 195 | 330 | 1 | 0.00 | 0.0% | - | - |
| 132 | GrowthWang2015Adapter | 46 | 192 | 329 | 1 | 0.00 | 0.0% | - | - |
| 133 | CrispriGuideFfaEnrichmentFang2025Adapter | 15,711 | 74,674 | 126,227 | 12 | 2.18 | 3.4% | 4.6 | 0 |
| 134 | CrispriGuideFitnessWang2018Adapter | 240,481 | 630,988 | 1,550,546 | 80 | 13.50 | 21.1% | 14.3 | 0 |
| 135 | CrispriGeneFitnessWang2018Adapter | 20,158 | 67,717 | 148,604 | 16 | 2.42 | 3.8% | 5.3 | 0 |
| 136 | EnvChemgenWang2024Adapter | 26,514 | 59,831 | 163,545 | 18 | 1.58 | 2.5% | 3.6 | 0 |
| 137 | ProteomeBanerjee2025Adapter | 6 | 37 | 56 | 3 | 0.00 | 0.0% | - | - |
| 138 | RbTnseqBorchert2023Adapter | 10,824 | 22,355 | 65,507 | 22 | 1.79 | 2.8% | 1.9 | 0 |
| 139 | FitnessContrastBorchert2023Adapter | 64,853 | 139,837 | 394,250 | 15 | 3.33 | 5.2% | 11.1 | 0 |
| 140 | RbTnseqBorchert2024Adapter | 1,466,920 | 2,532,825 | 8,812,762 | 522 | 33.93 | 53.0% | 14.4 | 0 |
| 141 | RbTnseqFitnessBrowserPutidaAdapter | 1,118,052 | 2,256,121 | 6,719,300 | 364 | 11.79 | 18.4% | 14.3 | 0 |
| 142 | IsoprenolTiterCarruthers2025Adapter | 502 | 1,634 | 6,358 | 8 | 0.97 | 1.5% | 0.9 | 0 |
| 143 | ProteomeCarruthers2025Adapter | 21 | 97 | 280 | 4 | 0.00 | 0.0% | - | - |
| 144 | CampaignProteomeCarruthers2025Adapter | 465 | 1,488 | 5,851 | 20 | 1.14 | 1.8% | 1.0 | 0 |
| 145 | ProteomeFoldChangeCarruthers2025Adapter | 16 | 110 | 275 | 4 | 0.00 | 0.0% | - | - |
| 146 | ProteomeDeSiqueira2025Adapter | 17 | 178 | 293 | 4 | 0.00 | 0.0% | - | - |
| 147 | ProteomePercentDeSiqueira2025Adapter | 17 | 178 | 293 | 4 | 0.00 | 0.0% | - | - |
| 148 | ProteomeLog10PercentDeSiqueira2025Adapter | 17 | 178 | 293 | 4 | 0.00 | 0.0% | - | - |
| 149 | IsoprenolTiterDeSiqueira2025Adapter | 10 | 164 | 273 | 3 | 0.00 | 0.0% | - | - |
| 150 | IsoprenylAcetateTiterKang2026Adapter | 26 | 150 | 408 | 5 | 0.00 | 0.0% | - | - |
| 151 | IsoprenolTiterBanerjee2024Adapter | 20 | 119 | 305 | 3 | 0.00 | 0.0% | - | - |
| 152 | ValerolactamTiterThompson2019Adapter | 8 | 41 | 87 | 2 | 0.00 | 0.0% | - | - |
| 153 | LactamGrowthRateThompson2019Adapter | 9 | 39 | 78 | 2 | 0.00 | 0.0% | - | - |
| 154 | PutidaPrecise321Lim2022Adapter | 321 | 1,166 | 2,311 | 40 | 1.41 | 2.2% | 1.0 | 0 |
| 155 | IsoprenolToleranceLim2025Adapter | 3 | 35 | 51 | 2 | 0.00 | 0.0% | - | - |
| 156 | ProteomeLim2025Adapter | 3 | 51 | 79 | 3 | 0.00 | 0.0% | - | - |
| 157 | ProteomeFoldChangeLim2025Adapter | 4 | 55 | 83 | 3 | 0.00 | 0.0% | - | - |
| 158 | IsoprenolSelectionMenasalvas2025Adapter | 58 | 258 | 888 | 3 | 0.00 | 0.0% | - | - |
| 159 | ProteomeMenasalvas2025Adapter | 6 | 122 | 161 | 3 | 0.00 | 0.0% | - | - |
| 160 | MetaboliteGrowthPhaseMenasalvas2025Adapter | 3 | 113 | 136 | 2 | 0.00 | 0.0% | - | - |
| 161 | MetaboliteProductionPhaseMenasalvas2025Adapter | 3 | 115 | 140 | 2 | 0.00 | 0.0% | - | - |
| 162 | EnvMetalTnseqRoyet2025Adapter | 21,583 | 49,971 | 135,017 | 20 | 1.81 | 2.8% | 2.9 | 0 |
| 163 | GradedGeneEssentialityRoyet2025Adapter | 11,458 | 33,269 | 74,489 | 27 | 1.45 | 2.3% | 1.0 | 0 |
| 164 | CrispriArrayYunus2026Adapter | 25 | 132 | 284 | 3 | 0.00 | 0.0% | - | - |
| 165 | CrispriDifferentialProteomeYunus2026Adapter | 1 | 18 | 18 | 2 | 0.00 | 0.0% | - | - |
| 166 | CrispriKnockdownYunus2026Adapter | 102 | 717 | 1,222 | 10 | 0.96 | 1.5% | 1.0 | 0 |
| 167 | IsoprenolTiterYunus2026Adapter | 125 | 633 | 1,010 | 3 | 0.00 | 0.0% | - | - |
| 168 | CrispriPanelProteomeYunus2026Adapter | 125 | 633 | 1,010 | 3 | 0.00 | 0.0% | - | - |
| 169 | InhibitorBioscreenVolk2021Adapter | 977 | 2,885 | 7,477 | 3 | 0.00 | 0.0% | - | - |

#### The slowest five and the largest five


| rank | slowest (wall s) | largest (peak GiB) |
|---:|---|---|
| 1 | RbTnseqBorchert2024Adapter 522 s (1,466,920 records) | RbTnseqBorchert2024Adapter 33.93 GiB (53.0% of 64 G) |
| 2 | RbTnseqFitnessBrowserPutidaAdapter 364 s (1,118,052 records) | CrispriGuideFitnessWang2018Adapter 13.50 GiB (21.1% of 64 G) |
| 3 | EnvChemgenNichols2011Adapter 206 s (1,286,962 records) | RbTnseqFitnessBrowserPutidaAdapter 11.79 GiB (18.4% of 64 G) |
| 4 | CrispriGuideFitnessWang2018Adapter 80 s (240,481 records) | EnvChemgenNichols2011Adapter 11.73 GiB (18.3% of 64 G) |
| 5 | PhageRbTnseqMutalik2020Adapter 78 s (287,815 records) | PhageRbTnseqMutalik2020Adapter 9.85 GiB (15.4% of 64 G) |

### Every adapter over half its tier, and the tier proposed for it

The threshold is half the tier's memory, which is the owner's overestimate rule applied
to a number that was previously a guess: at 50 percent a task has a 2x margin for the
chunk-size and page-cache variation the Hoepfner study measured.

**One adapter of the 47 passes it, and only one.**

| adapter | tier | peak GiB | % of tier | proposed tier | reason |
|---|---|---:|---:|---|---|
| RbTnseqBorchert2024Adapter | standard | 33.93 | 53.0% | large | 33.9 GiB is 21% of the 160 G large tier against 53% of standard |

`RbTnseqBorchert2024Adapter` (index 140, 1,466,920 records, the largest record count of
the whole tail) peaked at 33.93 GiB of a 64 G container and took 522 s, with 0 pool
recycles at 14.4 of 16 cores, so it completed comfortably at the standard tier in this
run. The proposal is still to move it: 53 percent leaves a 1.9x margin, and the Hoepfner
study measured what happens to an adapter whose peak approaches its container limit, a
5.1x slowdown from pool recycling alone once `memory.current` crosses 0.8 of the limit
(139 recycles at 256 G against 4 at 320 G, [[experiments.database.hoepfner-generation-study]]).
At 33.9 GiB the 0.8 threshold of a 64 G container is 51.2 G, which this run came within
17 GiB of, with the page cache of a 5.7 GB LMDB counted into `memory.current` on a quiet
filesystem. A busier `/db` or a wider row would put it over. Moving it to `large`
(32 CPU, 160 G, 16 h) costs the array one more `large` task, which the projection below
charges.

Second and third are `CrispriGuideFitnessWang2018Adapter` at 13.50 GiB (21.1 percent)
and `RbTnseqFitnessBrowserPutidaAdapter` at 11.79 GiB (18.4 percent). Nothing else in
the tail passes 19 percent, and 29 of the 47 never ran long enough to be sampled.

### Mutalik 2020, settled

Lane P named `PhageRbTnseqMutalik2020Adapter` as the one to watch, on a measured 37 GB
of store-build RSS against the 64 G standard tier.

**Measured here: 287,815 records generated in 78 s at a 9.85 GiB peak, 15.4 percent of
the 64 G standard tier, 13.0 of 16 cores, 0 pool recycles. It stays in the standard
tier, with a 6.5x margin.** The 37 GB lane P measured is the RSS of
`build_dataset_lmdb` building the store, which holds the release's frames in memory
while it serializes; generation streams the finished LMDB in 128 MiB chunks and never
holds the store. The two numbers measure different programs on the same dataset, and
only the generation number bears on the generation tier. The store-build number stands
as a sweep-side constraint, not a generation one.

The adapter has a second, separate finding of its own: `test_loader_lane_exhaustiveness`
on main xfails it because `released_addition` (`mutalik2020.py:1769`) appends a
`SmallMoleculePerturbation` to `Environment.perturbations` under a phage-only conf. Its
generated output carries `PhagePerturbation` and `EnvironmentPerturbationMemberOf` with
0 dangling rows, so no stored record takes that branch today. Reading, labeled: the gap
is latent in the current store on the evidence of the gate's zero, and it would surface
as a dangling environment-perturbation row if a record ever did; the unification closes
it either way.

### The CSV endpoint gate: NOT clean, and it found the defect it was built for

`python -m torchcell.knowledge_graphs.csv_endpoint_gate <out>` over all 47 directories as
one output, with lane N's label and collision checks, exactly as the array follow-up runs
it. **9,275,589 node rows in 25 labels (9,195,342 distinct ids), 28,353,191 edge rows in
13 labels, 80 s, 2.1 GB peak RSS, `Mentions` the one undeclared edge label, 0 mislabeled
endpoints, 0 id collisions, and 12 DANGLING edge rows.** Exit 1.

| finding | count | where |
|---|---:|---|
| `PerturbationMemberOf` `:START_ID` with no node row | 6 rows over 6 distinct ids | `142_IsoprenolTiterCarruthers2025Adapter` (of 3,125 rows) |
| `PerturbationMemberOf` `:START_ID` with no node row | 6 rows over 6 distinct ids | `143_ProteomeCarruthers2025Adapter` (of 121 rows) |
| mislabeled endpoints | 0 | - |
| id collisions | 0 | - |

The 12 ids appear in no node part file of any of the 47 directories. The two Carruthers
adapters write `BacterialPerturbation` nodes (151 and 21 rows) and no gene-addition node
at all, so the missing nodes are the `GeneAdditionPerturbation` leaves. This is the
`#1105`-style defect the gate's own docstring names: the leaf sits under a conf whose
only genotype lane is `bacterial perturbation`, the edge method walks it anyway, and
`skip_bad_relationships: true` would discard the row at import without stopping the
build.

**It is main's defect, and the ontology unification closes it. That is measured on both
sides, not assumed:**

- Main's `tests/torchcell/adapters/test_loader_lane_exhaustiveness.py` xfails exactly
  three datasets, and two of them are this pair, with the reason "GeneAdditionPerturbation
  (carruthers2025.py 1825-1853) under a bacterial-perturbation-only conf, **6 leaves in 2
  records**, closed by the ontology unification". Six is the number this gate found, per
  arm, independently and from the bytes.
- Lane P generated `IsoprenolTiterCarruthers2025Dataset` from `ontology-unify` and its
  CSV carries **157** perturbation rows with **0 dangling endpoints**
  ([[database.kg-4-pre-build-verification]]). Main writes 151 here. 151 + 6 = 157, so the
  branch writes precisely the six nodes this run finds missing.
- `ProteomeCarruthers2025Dataset` was not in lane P's nine, so its six are covered by the
  same xfail and the same lane fix rather than by a direct branch measurement. That is an
  inference from the shared leaf and the shared conf, labeled as one; it is not measured
  on the branch.

So the gate's verdict on the bacterial tail, read against the branch the build will use:
clean except for a known main-side gap that the branch under integration fixes, with no
mislabeled endpoint and no id collision anywhere in 9.2 M node ids and 28.4 M edge rows.
Lane N's namespaced interned-constant id holds across 47 directories.

One accounting note, so the two totals are not read as a discrepancy: the gate counts
9,275,589 node rows against the markers' 9,275,542, a difference of exactly 47, one per
directory. It is BioCypher's own `Schema_info` node, which `write_import_call` writes and
the sink's counter does not include.

### Container startup on the prebuilt image, measured

Lane C2's image landed as a module with no caller: the check-in log of 08:35 CDT records
that "its three hooks into lane C's array files wait on PR #1110 landing (follow-up
branch needed)". This lane wired them, so job 4974 is the first array to run on the image
and the first measurement of it inside the real array task.

| path | container start to the generator's first line | source |
|---|---|---|
| install, serial build | 278 s | job 4919_64, [[torchcell.knowledge_graphs.generation_array]] |
| install, measured by lane C2 | 255.9 s (239 s of it pip) | lane C2, 2026-10-15 |
| image, measured by lane C2 | 17.3, 17.0, 13.1 s | lane C2, 2026-10-15 |
| **image, in the real array task** | **median 11.0 s, mean 11.2 s, range 10 to 14 s over 47 tasks** | **job 4974, this run** |

Cold image build: **434 s** on a busy node, against lane C2's 216 to 259 s on a quiet
one; a same-commit rebuild is free on the layer cache, so a re-run of failed tasks pays
no image time. Across the real 170-task array the image saves 170 x about 245 s, which is
**11.6 h of container CPU** for one 7 min build.

### Projection of the full 170-task array

Measured and projected are separated strictly. Every one of the 170 tasks is charged a
number that was MEASURED somewhere; nothing is extrapolated. The charge per task, in
order of preference: this run's own measurement at the task's real tier (47 tasks); a
measurement another run made at that tier (Hoepfner at 4,558 s from job 4919_64's 320 G
baseline, Caudal at 727 s from lane P's job 4962); job 4901's per-adapter fold (97 tasks,
measured in ONE 64 CPU 256 G container, not in a per-adapter container); and one sampling
interval, 5 s, for the 24 adapters job 4901 never sampled even once, which bounds them
above. What is PROJECTED is only the arithmetic: the sum over a tier divided by its
concurrency.

| tier | tasks | concurrency | measured tasks | task seconds | projected wall h | basis |
|---|---:|---:|---:|---:|---:|---|
| standard | 157 | 4 | 47 | 14,368 | 1.00 | 86 the serial run, 47 this run, 24 under one sampling interval; 11 s of startup each, at %4 |
| large | 12 | 2 | 1 | 18,759 | 2.61 | 1 another run at this tier, 11 the serial run; 11 s of startup each, at %2 |
| hoepfner | 1 | 1 | 1 | 4,569 | 1.27 | 1 another run at this tier; 11 s of startup each, at %1 |

Serial sum of the tiers: **4.88 h projected**; the tiers run concurrently in the real build, so the array's calendar is at least the largest row. Critical path adapter: PooledPromoterExpressionVaishnav2022Adapter at 6175 s (1.72 h).

**The array's calendar is the largest tier row, not their sum, because the three arrays
are submitted together and run concurrently: about 2.6 h projected, set by the `large`
tier.** The standard tier, which carries 157 of the 170 tasks including the whole
bacterial tail, projects to 1.00 h at `%4`, so the tail this lane existed to measure is
not the cost; the 12 `large` tasks at `%2` are.

**Critical path adapter: `PooledPromoterExpressionVaishnav2022Adapter`, 6,175 s
(1.72 h).** It is the longest single task of the array, longer than Hoepfner's 4,558 s at
320 G, and it is the reason the `large` tier does not go below 1.72 h however the other
11 are packed. Job 4901 measured it at 1.1 cores mean of 64 over 1.72 h with 0 recycles
at a mean 110 G, so it is neither core-bound nor memory-throttled but parent-bound, and
the `large` tier's 32 CPUs will not shorten it. Hoepfner is second at 1.27 h and runs in
a tier of its own, so it is not on the critical path any more; the Hoepfner study's 320 G
result is what took it off.

Caveats on the projection, each labeled:

- **The 97 serial-run charges are measurements of a different configuration.** Job 4901
  ran every adapter in one 64 CPU, 256 G container with a cumulative heap. The array
  gives a `standard` task 16 CPUs. For an adapter that was core-bound in the serial
  container the array will be SLOWER than charged, and the core-bound ones are
  identifiable from the 4901 fold: `DmfCostanzo2016Adapter` and `DmiCostanzo2016Adapter`
  at 60.8 and 60.0 cores mean, `Hale2024Adapter` at 51.4. All three are in the `large`
  tier at 32 CPUs, so the understatement is bounded at about 2x on three tasks of the
  tier that sets the calendar. A `large` tier that runs at 2x its charge is 5.2 h rather
  than 2.6 h.
- **The 24 five-second charges are an upper bound, not an estimate.** Those adapters
  produced no sample in a 5 s sampler, so each ran for less than 5 s in the serial
  container. 47 of the 47 tasks this run measured at the real tier confirm the shape:
  29 of them also finished inside two intervals.
- **Pool recycling is the one thing that could invalidate this.** Every task of this run
  recycled 0 times. The Hoepfner study measured a 5.1x penalty when a container crosses
  0.8 of its memory limit, so an adapter whose peak moves toward its tier is not 10
  percent slower, it is multiples slower. That is the whole argument for moving Borchert
  2024 off the 53 percent row.
- **Lane V's collision guard is a SWEEP cost, not a generation cost**, and it is not in
  the table above. It adds about 48 min to Hoepfner's manifest write and about 30 min per
  Costanzo store, which is `build_dataset_lmdb` time in the 170-store sweep that precedes
  generation, not array time. A sweep projection that uses this lane's numbers should add
  it there; folding it into a generation tier would double count it.

### What this lane could not measure, and why

1. **The branch, on these adapters.** Lane S was asked to branch from `origin/ontology-unify`
   and the measurement ran at `origin/main` instead. The reason is measured, not a
   preference: the shared dev stores are built from main and a main-built record unpickles
   a `torchcell.datamodels.schema.ReleasedStrainKind`, an enum main has and
   `ontology-unify` does not, so branch code raises
   `AttributeError: Can't get attribute 'ReleasedStrainKind'` on
   `EnvChemgenNichols2011Dataset[0]` and cannot read any of the 47 stores. Generating the
   tail at the branch therefore requires the 170-store sweep to rebuild them first, which
   is a different lane. The transfer from main to the branch is calibrated on the one
   adapter both sides have measured: Nichols at **205.7 s and 11.73 GiB** here on main
   against lane P's **219 s and 12.5 GiB** on the branch, so the branch is 6.5 percent
   slower and takes 6.6 percent more memory on the largest E. coli store. Applying that
   factor, Borchert 2024's 53.0 percent of the standard tier becomes about **56.5 percent
   on the branch**, which strengthens the move to `large` rather than weakening it.
   (Labeled: 6.6 percent is one adapter's ratio, not a fitted model.)
2. **A branch measurement of `ProteomeCarruthers2025Dataset`'s six dangling rows**, for
   the reason in the gate section: no branch-built store for it exists.
3. **The merge and the duplicate check over these 47 directories.** The merge requires
   every slot of the recorded order to carry a marker and this run generated 47 of 170 on
   purpose, so `merge_outputs` refuses by design. Its cost is measured separately in
   [[torchcell.knowledge_graphs.generation_array]] (the two-copy worst case at 1 h 55 min
   and the 10 to 30 min production estimate).
4. **A second sample of the three tasks over 300 s.** Each adapter ran once. The wall
   times carry the node's contention at the time (two 031 training packs were running
   beside this array) and are single observations, not means.

### Reproducing this

```bash
# the array, at the committed tier sizes with the standard tier's concurrency at 3
OUT_NAME=bacterial-gen-2026-10-15 TORCHCELL_SRC=$PWD \
  BUILD_COMMIT=0a238c9679fbcd1be07723e52561af3d09d42962 \
  SERVE_ROOT=/scratch/projects/torchcell/database/bacterial-gen-profile \
  TIERS=$PWD/experiments/database/scripts/bacterial_generation_tiers.yaml \
  INDICES=108,124-169 JOB_NAME_PREFIX=bact_gen GEN_IMAGE_MODE=image \
  TCDB_TAGS=lane-S,bacterial-gen-profile \
  bash database/scripts/kg_generation_array_conductor.sh

# the readout and every table above
PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python \
  experiments/database/scripts/bacterial_generation_profile.py \
  --out-dir /scratch/projects/torchcell/database/bacterial-gen-profile/biocypher-out/bacterial-gen-2026-10-15 \
  --job 4974 --slurm-output /scratch/projects/torchcell/database/slurm/output \
  --serial-telemetry /db/database/biocypher-out/2026-10-14_03-53-33/telemetry/resource_samples.csv \
  --extra-index 108 --startup-s 11.2 --label bacterial-tail --table

# the gate, over the 47 directories as one output
PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python -m torchcell.knowledge_graphs.csv_endpoint_gate \
  /scratch/projects/torchcell/database/bacterial-gen-profile/biocypher-out/bacterial-gen-2026-10-15 \
  --report experiments/database/results/bacterial_generation_csv_endpoint_gate.json
```

`--list` prints the membership alone, without an array. The 19 GB of CSV under
`SERVE_ROOT` is reproducible from the line above and holds no unique provenance, so it
can be reclaimed before the 4.0 build writes its own.

### Three things the build should take from this

1. **Move `RbTnseqBorchert2024Adapter` to the `large` tier.** It is the only one of 47
   past half its tier, at 53.0 percent measured on main and about 56.5 percent projected
   on the branch, and the Hoepfner study already measured the penalty for being near a
   container limit.
2. **Run the array on the prebuilt image.** The hooks are wired and measured at 11.2 s
   of startup against 255.9 s, which is 11.6 h of container CPU over 170 tasks for one
   7 min build.
3. **Carruthers' twelve dangling rows must be fixed before import, and the branch does
   fix them.** Importing main's CSVs would silently drop them under
   `skip_bad_relationships: true`. The gate catches it; keep the gate as the hard stop
   it is.

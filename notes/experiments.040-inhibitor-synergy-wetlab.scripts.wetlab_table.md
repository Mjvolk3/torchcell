---
id: ijxexutprq1fap6i6acppxn
title: Wetlab_table
desc: ''
updated: 1791617246355
created: 1791617246355
---

## 2026.10.10 - Tidy well table and the two growth calls

Reads the private dev LMDB (`InhibitorBioscreenVolk2021Dataset`, 977 records) into `results/wetlab_wells.csv`: one row per well, with the run, the run length, the compounds (loader names, sorted, pipe-joined), the dose of each inhibitor in g/L and in mM (standard formula weights: furfural 96.08, acetic acid 60.05, 5-HMF 126.11, formic acid 46.03, levulinic acid 116.12, lactic acid 90.08 g/mol), the relative growth rate, the growth call and the category label. The biological replicate id comes from the loader's plate layout (`bioscreen.layout`). `results/wetlab_provenance.json` (pydantic `WetlabProvenance`) records the publication identifier with the report sha256, the raw-mirror manifest path and its sha256, and the consumed raw-file pins.

Counts. Wells by number of compounds: 0: 32, 1: 288, 2: 531, 3: 60, 4: 45, 5: 18, 6: 3. Wells grown per run: ex21 117 of 180, ex23 69 of 197, ex26 45, ex27 76, ex28 53 of 200. The isobole counts equal 039's.

**The served call does not reproduce 039's ex23 counts.** ex23 combinations with at least one grown well:

| inhibitors | combinations | grew, served (raw curve) | grew, software (039) | mean fitness of grown, served | software |
|---|---|---|---|---|---|
| 1 | 6 | 6 | 6 | 0.805 | 0.788 |
| 2 | 15 | 12 | 14 | 0.437 | 0.465 |
| 3 | 20 | 3 | 5 | 0.426 | 0.461 |
| 4 to 6 | 22 | 0 | 0 | | |

The software call (`bioscreen.software_generation_times`) reproduces 039 exactly, so the gap is the call, not the table. The two calls disagree on 12 of 197 ex23 wells, all three replicates of HMF + LA, HMF + LVA, FF + AA + HMF and FF + HMF + FA. The software assigned them generation times of 6.6 to 10.6 h. Their baseline-subtracted OD rose only 0.085 to 0.277, below the loader's 0.3 growth rise (`results/wetlab_growth_call_check.csv`). The loader pins this agreement (0.939086 = 185 of 197). The fitness scales also differ: the ex23 control generation time is 1.345 h from the raw curves and 1.815 h from the software (n = 8 control wells).

Decision (coordinator, 2026-10-10): the served call is primary (21 grown combinations). The software call is carried as a sensitivity column for ex23 only (`fitness_software`, `grew_software`, each on its own scale), and every claim-1 score is reported under both.

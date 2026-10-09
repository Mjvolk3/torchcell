---
id: in3at1i4mj9wzo17yat6mtn
title: 039 Inhibitor Combinations Wetlab
desc: ''
updated: 1791404285725
created: 1791404285725
---

## 2026.10.07 - The 2021 inhibitor runs as a wet-lab validation target

The Bioscreen C runs of 2021 on the MAGIC strain BY4742-iAID6 in YPD, read from the
thesis archive on `/bulk` (`/bulk/thesis/thesis_archive`, 29,438 data files, every one
sha256-verified against its `MANIFEST.tsv` on 2026-10-07). Six sorghum-hydrolysate
inhibitors: furfural (FF), acetic acid (AA), 5-HMF, formic acid (FA), levulinic acid
(LVA), lactic acid (LA). The intent is a prediction target for the chemogenomic models
once they are developed: single inhibitors, their 63 combinations, and two-inhibitor
isoboles, all measured in one strain.

Fitness throughout is the definition of the analysis repo's `process_bsc.py`: the mean
wild-type generation time divided by the well's generation time, so 1 is wild-type growth.
A culture whose curve never rose within the run has no generation time; those wells are
shown as their own category, not dropped. Figures and tables from
[[experiments.039-inhibitor-combinations-wetlab.scripts.plot_inhibitor_runs]].

### ex23: all 63 combinations at one concentration each (the main target)

One dispensed concentration per inhibitor (FF 1.5, AA 2, HMF 2.522, FA 1, LVA 6, LA 20 g/L;
see [[2026.10.08 - Dose corrections|dendron://torchcell/experiments.039-inhibitor-combinations-wetlab#20261008---dose-corrections]]), three
biological replicates, 200 wells (8 uninhibited controls, 3 blanks). `results/ex23_conditions.csv`.

| number of inhibitors | combinations | grew within 85 h | mean fitness of those that grew |
|---|---|---|---|
| 1 | 6 | 6 | 0.788 |
| 2 | 15 | 14 | 0.465 |
| 3 | 20 | 5 | 0.461 |
| 4 | 15 | 0 | |
| 5 | 6 | 0 | |
| 6 | 1 | 0 | |

![](assets/images/039-inhibitor-combinations-wetlab/ex23_combinations_fitness_2026-10-09-05-08-37.svg)

![](assets/images/039-inhibitor-combinations-wetlab/ex23_pair_matrix_2026-10-09-05-08-37.svg)

![](assets/images/039-inhibitor-combinations-wetlab/ex23_single_inhibitor_curves_2026-10-09-05-08-38.svg)

Read off the data: HMF is the strongest single inhibitor at its concentration (fitness
0.44), the pair that fails to grow is FA + LA, and every combination of four or more fails.
The target is therefore two-part: growth or no growth over all 63, and a fitness over the
25 that grew.

### ex21: single-inhibitor titrations

Nine concentrations per inhibitor, three biological replicates; wells in blocks of ten
(control, then the titration from the highest concentration down). The concentration
labels are the processing notebook's in g/L, except two notebook typos replaced by the
dispensed doses (FF 18, FA 0.125 g/L; see the 2026.10.08 section); the x axis is the
titration step. The control wells of this run grew slower (mean generation time
2.62 h) than most low-concentration wells, so fitness exceeds 1 at the low end.
`results/ex21_titration.csv`.

![](assets/images/039-inhibitor-combinations-wetlab/ex21_titrations_2026-10-09-05-08-38.svg)

### The isoboles: ex26 furfural, ex27 formic acid and ex28 5-HMF, each against acetic acid

A 10 x 10 grid per run (0 to 9 steps of 10 uL of inhibitor stock in 200 uL: FF 0 to 3,
FA 0 to 1.8, HMF 0 to 4.5, AA 0 to 3.6 g/L, from the design sheet
`BioscreenC_FF_AA_Isobole_exp.xlsx`), two plates each. Grid colors: white at wild-type
growth down through the palette red to its dark red at 0; a well that did not grow is
drawn at 0, the call being "no growth" rather than "missing", and the black front traces
the boundary between the wells that grew and those that did not, the detection limit of
the run.

Only ex26 was processed by the Bioscreen software (`MV_ex26_..._Traits.txt`, 48 of 200
wells grew). `results/ex26_isobole.csv`.

![](assets/images/039-inhibitor-combinations-wetlab/ex26_isobole_furfural_acetic_acid_2026-10-09-05-08-38.svg)

For ex27 and ex28 the generation time is derived here from the raw curves
(`generation_time` in the script: baseline-subtracted OD, growth if the rise is at least
0.3, the rate from the steepest 3 h log-linear stretch). The same derivation on ex26's raw
curves agrees with the software's traits on 98.5% of the 200 grew / no-grew calls and
ranks the 45 wells both call grown with a Spearman of 0.80 (`results/ex26_trait_check.csv`),
so the two unprocessed isoboles are read on a checked footing. Wells grown: ex26 45, ex27
76, ex28 53 of 200 (`results/isoboles_from_raw.csv`).

![](assets/images/039-inhibitor-combinations-wetlab/isoboles_from_raw_2026-10-09-05-08-38.svg)

Formic acid x acetic acid is the cleanest trade-off (a diagonal front from FA 1.6 g/L
alone to AA 2.8 g/L alone). 5-HMF x acetic acid has a ragged front with growth islands at
HMF 2.0 g/L; those wells are read from a single run with the threshold above and would
need a repeat before being trusted.

### What a dataset would read

`01_bioscreenc_raw/MV_ex23_Magic_inhibitor_combinations.csv` (OD600 every 15 min, 341
rows x 200 wells, UTF-16), the well map
`multi_knockout/inhibitor_tolerance/1_inhibitor-screen_2021-04-14_205953.xlsx` (volumes of
each stock and of YPD per well), the stock table `02_wet_lab_records/experiments/inhibitors.xlsx`,
and the strain: BY4742-iAID6, the MAGIC background that `CrisprMagicLian2019Dataset`
already carries. Admission as a dataset is a separate decision.

## 2026.10.08 - Dose corrections

Two sets of doses in the 2026.10.07 figures were wrong; the figures above are regenerated
with the dispensed doses and the superseded files removed.

- **ex23.** The script plotted the literature `C2` column of `inhibitors.xlsx` (FF 6, AA 4,
  HMF 2.522, FA 1, LVA 2, LA 40 g/L), which is the fifth titration step, not what ex23
  dispensed. The dispensed doses are FF 1.5, AA 2, HMF 2.522, FA 1, LVA 6, LA 20 g/L:
  uL of stock in the well map `1_inhibitor-screen_2021-04-14_205953.xlsx` (42.5, 8.5, 17,
  17, 51, 85) x stock concentration (`inhibitors.xlsx` C12:C17: 60, 400, 252.2, 100, 200,
  400 g/L) / the 1700 uL tube (`1_inhibitor-screen.ipynb`, "ypd_v =  1700"). The Fig 11b
  dose table of the 2021 prelim report
  (`/bulk/thesis/thesis_archive/reports/Prelim_Report_2021_Michael_Volk.pdf`, p. 13) shows
  the same six values. Only axis and legend labels change; the fitness values do not.
- **ex21.** Two titration labels were notebook typos carried verbatim: the dispensed
  volumes (uL of stock per 4 mL tube in `inhibitors_titration_array.xlsx` x stock / 4000 uL)
  give FF 18 g/L where `MV_ex21_data_processing_v2.ipynb` says "28", and FA 0.125 g/L where
  it says "1.25". With these corrected both titrations are monotone. The `label` column of
  `results/ex21_titration.csv` changes accordingly.

The derivation and its sourced values are in
[[torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen]]
(`torchcell/datasets/private_torchcell/bioscreen.py`, `volk2021_sources.py`, on branch
`feat/private-torchcell-bioscreen-volk2021`).

## 2026.10.09 - Run lengths in the figure labels

No run was 48 h: the Bioscreen logs give 71.97 h for ex21, 84.97 h for ex23 and about 96 h for the isoboles (the 48 h was the lag the software assigns to a well that never grew). The ex21 and ex23 labels now say 72 h and 85 h; the figures were regenerated and relinked.

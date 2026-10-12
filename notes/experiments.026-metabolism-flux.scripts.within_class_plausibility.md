---
id: 7wjmoyhuz3hjrgnsctkrjcy
title: Within_class_plausibility
desc: ''
updated: 1788925193523
created: 1788925193523
---

## 2026.09.08 - How many predictions land inside the within-class spread, by model

`experiments/026-metabolism-flux/scripts/within_class_plausibility.py` makes the fraction
that the agreement figure's panel e gives as a caption range into a first-class result,
computed four ways: a shared band at each EC depth (the median within-class standard
deviation over all classes at EC1 to EC4), each pair's own EC4 class band, the EC-median
lookup scored the same way, and the betaxanthin supply genes scored for class plausibility
since none of them has a measured yeast value. It also counts how many of the 3,728
catalytic units any measured source can price: the Open Enzyme Database 138 (3.7 percent),
EnzEngDB 0, with the verified reason copied from `predictor_vs_measured.json`.

Writes `results/within_class_plausibility.json` and
`results/within_class_plausibility_betaxanthin.csv`. Reads the OED mirrors under
`$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/`, the predictor parquets under
`$DATA_ROOT/data/torchcell/kinetics/`, the yeast-GEM SBML for EC annotations, and imports
its class-spread and baseline functions from `ec_class_variance.py`.

Measured on 2026-09-08, 57 pairs per parameter, 47 scorable against their own class:

| parameter | shared EC4 band | predictors inside | EC-median lookup | betaxanthin units inside (of 16) |
| --- | --- | --- | --- | --- |
| k_cat | 0.75 | 0.56 to 0.84 | 0.53 | DLKcat 16, UniKP 15, DeepEnzyme 13, TurNuP 10, EZKinetics 10, EITLEM 6 |
| K_M | 0.58 | 0.49 to 0.63 | 0.60 | EITLEM 9, EZKinetics 7, UniKP 5, Boost_KM 4 |

Figure: [[experiments.026-metabolism-flux.scripts.plot_within_class_plausibility]].
Document: `notes-tex/026-kinetics-thermodynamics`, the two figures following the per-gene
agreement figures.

## 2026.09.10 - The 138 reconciled against the flux layer's 148, and two selection rules

**138 and 148 are both right and they count different things.** `measured_source_coverage`
here requires **every** member gene of a catalytic unit to have a measured accession, since
it is asking which units a measured source can price outright: 138. The flux layer's
`resolve_kcat_table` takes the minimum over a unit's member genes, so a unit is measured as
soon as **any** member gene resolves: 148. Verified directly by counting both ways on the
same mirror, and the `wildtype` filter changes neither number. Nothing here needs changing;
the two counts should just be quoted with their rule.

**The selection rule in this study and the one the flux layer runs are not the same rule,
and they disagree on K_M.** `PredictorSelection.from_study` reads this file's
`own_class_band` block and keeps a predictor when its Wilson 95 percent lower bound exceeds
the EC-median lookup's point fraction: EZKinetics, EITLEM and UniKP for k_cat, and
**EZKinetics alone** for K_M. `parameter_coverage_selected.json`, written by
`plot_parameter_coverage.py`, ranks on the mean of the shared and own-class band fractions,
takes the top 3 and drops anything at or below the lookup, which keeps EZKinetics and EITLEM
for K_M. The flux layer follows the first. Both files are in the repo, so a reader comparing
them sees two different admitted sets for the same parameter. This is a decision to make.

**Coverage moved after this study was run.** The 3,552 units the predictor parquets covered
are now 3,717, because 165 units were unblocked by resolving substrate structures the input
builder had missed. The predictors this study does not admit were not re-run on the new rows,
so their tables stop at 3,552; a re-run of this study would compare unequal unit sets across
predictors unless they are brought level first. Details:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

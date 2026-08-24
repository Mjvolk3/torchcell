---
id: c1rv6t37ofwqhp965jaepok
title: run4_measured_summary
desc: ''
updated: 1787598257356
created: 1787598257356
---

## 2026.08.24 - What run 4 measured, and what is missing

The build list says which strains exist; this says what each one scored. It generates the
measured columns pasted into the two inventory tables of
[[experiments.W019-echo-crispr-array.build-list]], and the gap table under "What is
missing, and why". `verify_triple_build_list.py` checks the pasted values against the CSVs
row by row, so the note cannot drift from the data.

Run from repo root:

```bash
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/run4_measured_summary.py
```

### Coverage is not reference coverage

The distinction the script exists to make legible: **12 of 12 singles and 13 of 13 built
doubles were measured in run 4**, but a strain can be measured here and still have nothing
published to compare against, and two are in that position.

| | measured | with a published value |
|:--|:--|:--|
| singles | 12 of 12 built | 11 |
| doubles | 13 of 13 built, 14 designed | 12 |
| triples | 0 | -- |

- **`YLR104W` (LCL2)** has no published SMF because the reference panel was assembled for
  the earlier plate design, which carried LCL1 (`YPL056C`) in that slot. Run 4 kept LCL2.
  This is the same root cause as the frozen 10-gene / 31-target basis being wrong.
- **`YPL046C + YPL081W` (d2)** has no published DMF because it is tier `novel`, the only
  one of the 45 candidate pairs with no measurement in Costanzo 2016, Kuzmin 2018 or
  Kuzmin 2020.
- **`YKL033W-A x YJR060W`** was designed and never built. Reported failed at the
  2026-08-11 handover with neither an attempt date nor a cause recorded, so no reason can
  be given beyond that. It blocks 0 of the 39 target triples.
- **No triples exist** in any round.

### Fitness only, deliberately

Run 4's epsilon is **not reportable**. The round's denominator rests on a single WT colony
and the measured double/single ratio is 0.758 where a multiplicative model wants ~1.07,
which no choice of reference can produce
([[experiments.W019-echo-crispr-array.run4-handoff]]). Emitting eps beside fitness would
invite it to be read as a result, so the tables carry fitness, bootstrap SE and
across-plate SD, and the eps situation is named in prose instead.

### Outputs

- `results/run4_measured_summary_singles.csv` -- id, ORF, common, n_plates, fitness,
  boot_se, across_plate_sd, costanzo_smf, costanzo_se
- `results/run4_measured_summary_doubles.csv` -- id, pair, n_plates, fitness, boot_se,
  across_plate_sd, tier, costanzo_dmf
- `results/run4_measured_summary_gaps.csv` -- one row per missing thing, with its reason

### Related

- [[experiments.W019-echo-crispr-array.build-list]]
- [[experiments.W019-echo-crispr-array.run4-handoff]]
- [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]

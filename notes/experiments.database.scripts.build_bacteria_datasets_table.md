---
id: m0mfrzux198bf2jk0gebh6e
title: Build_bacteria_datasets_table
desc: ''
updated: 1791955907930
created: 1791955907930
---

## 2026.10.14 - Bacteria Instances vs Signal table, with the Signal controls (temporary pre-build version)

Owner asked (2026-10-13) for the bacteria counterpart of the yeast table, and for the
Signal-column controls of #1099, as temporary versions now rather than at the next table
build. Both are measured in one script and rendered as views:

- `experiments/database/scripts/build_bacteria_datasets_table.py` scans every bacterial
  LMDB the adapter map serves (`build_adapter_map(organisms={ecoli, pputida})`, 100
  datasets, all built in the dev tree, 32.6 GB) and writes
  `experiments/database/results/pre-build/2026-10-14/ecoli-pputida_datasets.json`.
  Per-dataset rows are cached by LMDB fingerprint in
  `experiments/database/scripts/dataset_probe_cache.json` (committed, like the yeast
  table's cache; one file for every organism), so a rerun recomputes only rebuilt stores. Two streaming passes, 24 workers,
  about 8 minutes wall clock (Fuhrer 2017's wide metabolome was the straggler at 489 s).
- `render_bacteria_datasets_table.py` writes the generated note
  [[experiments.database.bacteria-datasets-table]], the `.tex` fragment beside the JSON and
  the preview PDF `notes/assets/pdf-output/experiments.database.bacteria-datasets-table.pdf`.
- `plot_bacteria_datasets_signal.py` draws the figure below (`ASSET_IMAGES_DIR/database/`).

Nothing in the row list is hand-curated: membership comes from the adapter map, the LMDB
path from each loader's default root, and Genotypes / Env are distinct stored genotype and
environment payloads counted in the pass, not curated text as in the yeast table.

![](./assets/images/database/bacteria_datasets_signal.svg)

Panel a: Instances vs Signal (gzip bits of the stored instance, the yeast table's
definition), colored by section, host as marker. Panel b: each control divided by Signal,
against Signal, one series per control (definitions in the generated note).

### Totals (measured, 100 datasets, 7,257,468 instances, 115,325,823 numeric leaves)

| Quantity | bits | ratio to Signal | median per-dataset ratio |
| :-- | --: | --: | --: |
| Signal (gzip of stored instances) | 7.73e9 | 1.000 | 1.000 |
| Structure only (leaves replaced by type tokens) | 2.26e9 | 0.292 | 0.199 |
| Shuffled within key path | 7.90e9 | 1.022 | 1.005 |
| Digit noise at stored precision | 8.25e9 | 1.066 | 1.087 |
| 1-bit z-score, in place | 4.08e9 | 0.528 | 0.678 |
| 1-bit z-score, packed | 6.98e7 | 0.009 | 0.009 |

E. coli holds 4,561,865 instances over 68 datasets and P. putida 2,695,603 over 32. The
Environmental / chemogenomic response section carries 6.56 M of the 7.26 M instances.

### Findings (measured in this run)

1. **Shuffling the numeric values within their key path does not change the gzip size**
   (ratio 1.022 overall; per dataset median 1.005, range 0.992 to 1.48). gzip finds no
   cross-record regularity in the values: a dataset's real values and a permutation of them
   compress to the same length.
2. **Pure digit noise at the stored precision is only 7% larger than the real data**
   (1.066 overall; median 1.087). The value bytes of a stored record compress like noise.
3. **Structure is 20 to 30% of Signal.** The schema copies (keys, nesting, list lengths,
   type tokens) cost 2.26e9 bits; the remaining 70 to 80% is value bytes that behave as in
   finding 2.
4. **The one-bit floor is about 1% of Signal**: 6.98e7 bits packed, 0.60 bits per numeric
   leaf (below 1 because constant paths such as `n_samples` and skewed sign distributions
   compress). Written in place, the same bits cost 0.528 of Signal, which is the structure
   plus the `1`/`0` tokens.

Reading: as defined, Signal measures (number of numeric leaves) times (stored digits) plus
the schema copies. It ranks datasets by volume and precision, not by regularity gzip can
exploit, which is consistent with the paper's framing of `L_C(D)` as a codelength under
gzip's model and an upper bound, not an information content. For the yeast snapshot of
2026-07-17 the same decomposition is not yet measured; a run with `--organisms scerevisiae`
was started on 2026-10-14 (377 GB, Hoepfner 2014 alone 30.3 M records) and is in flight, so
no yeast number is reported here.

Hypothesis (untested): the yeast table decomposes the same way, with a larger structure
share for the natural-isolate datasets whose perturbation lists are long and repetitive.

Hypothesis (untested): a compressor that models numbers (quantize to k bits, then gzip, or
an arithmetic coder over per-path histograms) would separate datasets that gzip cannot, and
would bring Signal closer to the information a model can learn from.

Caveats: several bacterial stores intern the environment as a `$ref` digest plus a name, so
their environment bytes are short; the yeast 2026-07-17 snapshot predates interning, so the
two tables' Signal columns are not directly comparable on that axis. The binarization sets a
constant path and a NaN leaf to 0. Noise keeps the digit count of `repr`, so a random digit
string ending in 0 can print one character shorter than the original.

## 2026.10.14 - Yeast run of the same probes (hypothesis above now measured)

`--organisms scerevisiae` over the 69 yeast datasets the adapter map serves (all built in the
dev tree, 377 GB; 24 workers, 3 h 32 min wall clock, Hoepfner 2014 with 30.3 M records took
12,717 s on its own). Output
`experiments/database/results/pre-build/2026-10-14/scerevisiae_datasets.json`, generated view
[[experiments.database.yeast-datasets-table]], preview PDF
`notes/assets/pdf-output/experiments.database.yeast-datasets-table.pdf`.

![](./assets/images/database/yeast_datasets_signal.svg)

### Totals (measured, 69 datasets, 93,266,786 instances, 886,671,302 numeric leaves)

| Quantity | bits | ratio to Signal | median per-dataset ratio | bacteria ratio (above) |
| :-- | --: | --: | --: | --: |
| Signal | 7.19e10 | 1.000 | 1.000 | 1.000 |
| Structure only | 1.27e10 | 0.176 | 0.159 | 0.292 |
| Shuffled within key path | 7.35e10 | 1.023 | 1.012 | 1.022 |
| Digit noise at stored precision | 7.92e10 | 1.102 | 1.096 | 1.066 |
| 1-bit z-score, in place | 4.83e10 | 0.672 | 0.724 | 0.528 |
| 1-bit z-score, packed | 5.60e8 | 0.008 | 0.007 | 0.009 |

Packed floor 0.63 bits per numeric leaf (bacteria 0.60). The yeast dev-tree Signal total here
(7.19e10 bits) is not the paper table's 2026-07-17 number: the set of built stores, the
interned environments and several rebuilt loaders differ, so compare ratios, not totals.

### Findings (measured)

1. **Yeast decomposes the same way as bacteria.** Shuffled 1.023 and digit noise 1.102 of
   Signal; the hypothesis in the section above is now measured. The largest stores are the
   cleanest cases: Hoepfner 2014 shuffled 1.001, noise 1.015, packed 0.0017; Costanzo dmi
   shuffled 1.001, noise 1.090.
2. **Structure is a smaller share in yeast** (0.176 vs 0.292) because the big yeast stores
   are long scalar tables whose per-record schema is small relative to their digits; the one
   yeast store above 0.5 is Jackson 2020 pseudobulk (0.72), the two bacterial ones are
   Fuhrer 2017 (0.56) and Caglar 2017 proteome (0.67).
3. **Datasets where shuffling does change the size are the exception and are small.** The
   yeast maximum is 1.67 and the bacterial 1.48, both on stores with few records where the
   per-path permutation breaks a within-record pattern gzip had found.

Hypothesis (untested): the within-record pattern of finding 3 is repeated identical values
across a record's vector (replicate counts, shared SE), not biological structure.

## 2026.10.15 - Totals per organism as grouped bars

The totals above, split into S. cerevisiae, E. coli and P. putida and drawn as grouped bars,
are in [[experiments.database.scripts.plot_organism_signal_controls]], generated by
`experiments/database/scripts/plot_organism_signal_controls.py` from
`dataset_probe_cache.json`. That note also holds a one-line definition of each control and
the 1-bit in place vs packed comparison.

![](./assets/images/database/organism_signal_controls.svg)

Correction the same day: the bacteria section above first gave the split as 58 E. coli and 42
P. putida datasets, and the yeast totals table gave structure as 1.26e10 and packed as 5.75e8
bits. The generated tables and the cache give 68 and 32 datasets, 1.27e10 and 5.60e8 bits;
the text now carries those values.

## 2026.10.15 - Expanded tables with a description of each dataset

Owner asked for a new version of the tables carrying a short description of each dataset.
The existing views are unchanged; each organism set gains an expanded view beside them:

- [[experiments.database.bacteria-datasets-table-expanded]], PDF
  `notes/assets/pdf-output/experiments.database.bacteria-datasets-table-expanded.pdf`
- [[experiments.database.yeast-datasets-table-expanded]], PDF
  `notes/assets/pdf-output/experiments.database.yeast-datasets-table-expanded.pdf`

**Where a description comes from.** `docstring_summary` in the build script takes the first
paragraph of the dataset class's own docstring (never an inherited one), unwraps RST roles
and literals, and collapses whitespace. It is code metadata, not read from the LMDB, so the
build script attaches it fresh to every row when it writes the JSON rather than caching it.
All 169 served classes have a docstring; their summary paragraphs run from 50 to 149
characters. A class without a docstring raises, and the renderer refuses a row with an empty
description. A better description is made by improving the docstring and rerunning both
scripts; nothing is typed into the table.

**Layout.** Description sits after Host. Description and Phenotype are wrapping LaTeX
columns (`Column.width` in `torchcell/paper/tables.py`, a ragged-right `p` column), and a
Phenotype cell may break after each underscore so long measurement types fit. The preview
canvas is sized from the wrapped line count of each row.

**Rebuild on 2026-10-15.** Every one of the 169 dev stores still matched its cached
fingerprint, so both JSONs were rewritten without rescanning an LMDB
(`results/pre-build/2026-10-15/`); every number in the standard views is unchanged and only
their source-JSON line moved.

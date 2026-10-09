---
id: o45gzyig7leypviww88dfjc
title: Wang2015_growth
desc: ''
updated: 1791555936023
created: 1791555936023
---

## 2026.10.09 - Table S3's no-isoprenol column loaded as 46 fitness ratios (#826, rank 12)

Rank 12 of [[plan.bacteria-si-phenotype-audit-ecoli]] called this "a recorded decision
rather than a release defect", and the decision is recorded here. The loader is
`torchcell/datasets/ecoli/wang2015_growth.py`, class `GrowthWang2015Dataset`, store
`data/torchcell/growth_wang2015`: 46 records plus one reference, no schema change.

### The pinned bytes, and the column

`si1.pdf`, sha256 `ed2a029aef1d7d45aa9aa853162ba7960132b7ebc674a9c92fbc618aae605952`,
1,098,793 bytes, read with the same `pdftotext -layout -enc UTF-8` recipe
(`pdftotext version 21.01.0`) and checked against the same parsed-Table-S3 digest
`ce6d952c1058695fe5912353de563faa0224ad5ce8c16c5a8812e62da3b41093` that
`wang2015.py` pins. Nothing new is retrieved.

Table S3 has exactly three columns, verbatim from the text layer:

```
Table S3. Cell growth of the MDT null mutants in the absence and presence of isoprenol.

                     Cell growth (OD600)
    Strains          No         0.5% (v/v)
                  isoprenol      isoprenol
  BW25113        8.30 ± 0.16     4.12 ± 0.01
...
Note: The results are presented as means ± standard divisions.
```

47 rows: BW25113 plus 46 `BWD*` Keio deletions. The loader consumes the SECOND column
(`No isoprenol`), which `wang2015.py` already reads as the per-strain normalizer of its
log2 relative tolerance.

### It is not a second copy of a stored number

Measured over the 46 mutants:

| statistic | value |
|---|---|
| Pearson r(`od600_without`, `od600_with`) | 0.6667 |
| Pearson r(`od600_without`, the stored log2) | 0.5454 |
| Pearson r(`od600_with`, the stored log2) | 0.9774 |
| stored values equal to any no-isoprenol value | 0 of 46 |
| max abs difference, log2(od600_without ratio) against stored | 0.6632 |

The stored log2 is a function of BOTH columns, so neither column is recoverable from it.
The Methods state the cultures are independent, verbatim: "4 mL of fresh media with (or
without) isoprenol or other $\mathrm{C}_{\mathrm{m}}\mathrm{OHs}$ at a given
concentration".

### Why `FitnessPhenotype` and not the absolute environment-response branch

The readout is a 12 h endpoint OD600 of a DELETION strain against its parent in the same
medium, which is a genotype ratio. The absolute branch PR #836 landed does not reach it:
that branch refuses any record whose `measurement_type` is not in
`ABSOLUTE_MEASUREMENT_TYPES`, the set is `{growth_rate, colony_size}`, and
`MeasurementType` has no optical-density member, so an endpoint OD600 would have to be
mislabeled a growth rate to take it. `FitnessPhenotype` also carries no
`measurement_type` field at all.

Measured: the 46 ratios run 0.877108 to 1.030120, median 0.956024, 0 non-positive, so
`FitnessPhenotype.validate_fitness`'s clamp never fires and nothing is destroyed. The
range is a checked oracle (`MIN_FITNESS` / `MAX_FITNESS`), not a sentence, so a
re-extracted table whose ratios moved stops the build.

Honest caveat, a reading and not a measurement: `FitnessPhenotype.fitness` is documented
`ko_growth_rate/wt_growth_rate` and this ratio is an endpoint BIOMASS ratio, not a rate
ratio. The Schmidt 2016 Table S24 precedent used real rates in h^-1.

### The uncertainty is exact arithmetic, not a gap

Unlike the stored log2 record, whose dispersion is a typed gap because a ratio of ratios
mixes four means, this ratio mixes only two:

| stored field | value | measured range |
|---|---|---|
| `fitness_uncertainty` | `sd_without(strain) / od600_without(parent)`, `sample_sd`, `n_samples = 2` | 0.00120 to 0.04096 |
| `fitness_se` | delta method, `fitness * sqrt((SE_s/mean_s)^2 + (SE_p/mean_p)^2)`, `SE = SD/sqrt(2)` | 0.01261 to 0.03142 |

The uncertainty is exact: it is the sample standard deviation of the n released ratio
observations, so the released statistic's kind and n survive the change of units. The SE
is always at least the auto-derived `fitness_uncertainty / sqrt(n)`, which conditions on
the parent mean as a fixed denominator and understates the spread; the build asserts that
direction per record and L2 `se_is_not_the_conditioned_derivation` re-asserts it on the
STORED numbers. 0 of 46 `sd_without` cells are zero. 34 of the 46 ratios sit more than
2 SE from 1.0.

`n_samples = 2` is STATED, not inferred: the Fig. 1C caption, verbatim, "The growth
inhibition of the wild type strain is at approximate $5 0 \%$ as a reference. Results are
the means of two biological replicates.", which is
`wang2015.SOURCED_VALUES["n_samples"]`, reused rather than re-sourced. The kind is the
table's own note, "means ± standard divisions" (a typo for deviations), corroborated by
Figs. 3C, 4 and 5, "Error bars represent the standard deviations of two biological
replicates."

### The environment

2YT at pH 7.0 and 30 C, shaken, 12 h: the stored dataset's environment MINUS the
isoprenol `SmallMoleculePerturbation`. The pH perturbation keeps its agent gap, because
the paper names no acid or base ("adjusted to $\mathrm{pH}7.0$"). Isoprenol is ABSENT
here, not gapped. Carried over from the stored dataset: the `acrA`/`acrB` and
`emrK`/`emrY` Table S2 number swaps and `mdtD`'s `JW2077`, so the record's locus comes
from the strain NAME and `strain_accession` is written only when the Table S2 number
names the same locus (`preprocess/identifier_reconciliation.json`).

### Why a separate module, measured

`loader_closure` on `wang2015.py` is 69 symbols and equals the 69 the served
`env_chemgen_wang2015` store records in its `preprocess/build_manifest.json`. Adding
`BacterialFitnessExperiment`, `BacterialFitnessExperimentReference` and
`FitnessPhenotype` to that module's schema import raises it to 72, so the served store
would read STALE for a change touching none of its 46 records. The pinned artifact, the
SI parsers, the strain resolver, the media object and the sourced values are imported
FROM `wang2015`.

### L0 to L4, measured on the built dev store

`PYTHONPATH=<wt> python -m torchcell.datasets.ecoli.wang2015_growth verify`,
`DATA_ROOT=/scratch/projects/torchcell-scratch`. PASS, including the 15 provenance-audit
rows that re-read every sourced quote out of the literature mirror.

| level | rule | result |
|---|---|---|
| L0 | structural | 46 records validated |
| L1 | count | observed 46, expected 46 |
| L1 | pair_uniqueness | 46 unique (strain, environment) records, one each |
| L1 | provenance_gaps | 46 documented gaps over 46/46 records; 0 deferred fields |
| L1 | canonical_gene_names | 46 systematic names, one canonical spelling each, each current in the genome |
| L2 | value_fidelity | 46 values checked |
| L2 | se_nonnegative | 46 values checked |
| L2 | uncertainty_sanity | 46 labeled uncertainties, none a zero dispersion |
| L2 | se_is_not_the_conditioned_derivation | all 46 stored SEs are at least the parent-conditioned derivation |
| L3 | reference_one | reference fitness == 1.0 for all 46 records |
| L3 | compound_identity | 0 environment-edit compound references (the pH edit names no agent) |
| L3 | media_compound_identity | 46 medium-component references carry a structure identifier |
| L3 | media_membership | 46 records on a shared MEDIA_LIBRARY medium (1 distinct medium) |
| L3 | provenance_audit x15 | every sourced value backed by a verbatim quote in `paper.md` or `si1.md` |
| L4 | gene_containment_sgd | 1.000 of 46 measured genes are BW25113 genes |
| L4 | current_genome_genes | every one of the 46 systematic names is a gene of the current genome |

### Schema impact

None. No symbol of `torchcell/datamodels/schema.py` is touched, so no served dataset is
staled; the new class is additive and reaches the graph through its own adapter module,
conf and `kg_bacteria.yaml` entry.

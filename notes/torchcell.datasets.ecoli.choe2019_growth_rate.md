---
id: qx4o5li6bayqloomipp75of
title: Choe2019_growth_rate
desc: ''
updated: 1791451590997
created: 1791451590997
---

## 2026.10.08 - What the Release Gives, and the Two Arms That Load

Row 47 of the bacterial schedule (`experiments/database/scripts/build_bacteria_candidate_datasets_table.py`,
`name="Choe 2019 genome-reduced ALE"`, E. coli, tier 3). Every number below is measured by
`experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py` and
regenerates into `experiments/036-dataset-fixes-before-kg-build/results/`.

### What the release gives per sample

A **population allele frequency**, never a clone genotype, and never a called variant
list for the strain the paper is about.

| file | item | calls | frequency axis |
|---|---|---|---|
| `si/si5.xlsx` | Supplementary Data 2, MS56 ALE | 117 | 20 timepoints, day 0 to day 62 |
| `si/si6.xlsx` | Supplementary Data 3, MG1655 ALE | 101 | 3 replicate populations |
| `si/si7.xlsx` | Supplementary Data 4, 300 extra generations | 145 | 2 days x 4 populations, plus a count |

Each row gives gene, position, reference base, alternate base, mutation type, amino-acid
change, and a frequency per column. 363 calls in total. The flagship evolved strain
eMS57 has **no variant list of its own**: a per-clone genotype would have to be produced
by thresholding the day-62 column, which leaves 11 calls at 100%, 30 above 50%, 23
between 5% and 95% and 79 at zero. The authors never made that call. The one call they
did make is a LINEAGE call, not a clone one: "Sequence variants with allelic frequency
$ { > } 0 . 8 $ once or $ { > } 0 . 5 $ at more than two time points were subjected to
hierarchical clustering", giving three clonal lineages with five sub-lineages drawn in
Fig. 2e. Supplementary Data 2's `Lineage Analysis` column marks INCLUSION in that
clustering, not membership of a clone.

### Whether an existing perturbation leaf can carry it

No. Four refusals, each constructed on a real row rather than read off the class:

1. `SequenceVariantPerturbation(systematic_gene_name="b0078", ...)` raises
   `Invalid systematic gene name format`: it inherits `GenePerturbation`'s R64 ORF regex
   and has no `gene_namespace` field (issue #749).
2. `BacterialBackgroundAllele` requires a non-optional `functional: bool`
   (`Input should be a valid boolean`), so an unknown missense consequence cannot be a
   `ProvenanceGap` -- a gap must name a field that is `None`.
3. `BacterialStrainBackground` permits one allele entry per locus, which refuses the
   seven loci carrying more than one call in Supplementary Data 2 alone (`yafF` 5,
   `ydjN` 4, `iscR` 3, `rrsH` / `tufA` / `yeaM` 2 each).
4. 22 of the 117 rows are intergenic and name no locus, so there is nothing to key to.

**The designed point mutants show the same gap with no evolved-clone excuse, and that is
what this row adds to issue #731.** Supplementary Table 3 releases three strains the
authors BUILT on MS56, each a single exactly specified allele with its own isogenic
kan-marked control: `MS56, cspC::cspC(G37A)-kanR`, `MS56, ilvN::ilvN(C202T)-kanR` and
`MS56, yifB::yifB(1169C insertion)-kanR`. Fig. 2f releases their growth rates as three
paired mutant-over-parent ratios each. Designed strains, alleles given to the base,
ratios released, and not one record can be written.

### What loaded

Two classes in `torchcell/datasets/ecoli/choe2019_growth_rate.py`, both
`BacterialFitnessExperiment`, both L0-L4 PASS on every row.

| class | dev store | records | reference |
|---|---|---|---|
| `GrowthRateChoe2019Dataset` | `growth_rate_choe2019` | 2 | MS56 |
| `TranscriptionFactorKnockoutChoe2019Dataset` | `tf_knockout_growth_choe2019` | 2 | BW25113 |

The designed-deletion arm, from the Fig. 2c sheet of the Source Data workbook. Released
rates, means over three biological replicates:

| strain | Rep1 | Rep2 | Rep3 | mean | fitness vs MS56 |
|---|---|---|---|---|---|
| MG1655 | 0.5828 | 0.5970 | 0.58785 | 0.589217 | not a record |
| MS56 | 0.1834 | 0.1876 | 0.1760 | 0.182333 | 1.0 (the reference) |
| eMS57 | 0.5148 | 0.5152 | 0.5103 | 0.513433 | refused |
| MS56 Delta21kb | 0.4073 | 0.3891 | 0.3993 | 0.398567 | 2.185923 |
| MS56 DeltarpoS | 0.4158 | 0.4215 | 0.3897 | 0.409000 | 2.243144 |

Both deletions grow faster than their reduced parent, which is the paper's point. The
Keio arm is the two BW25113 single deletions whose growth rate the Supplementary Fig. 6
legend states: `ydaS -> BW25113_1357` at 0.724 and `abgR -> BW25113_1339` at 0.698 of
wild type.

### Three cross-checks the build asserts

- **The Fig. 2c rates reproduce the paper's own prose.** `check_recovery_fraction`:
  Delta21kb reaches 0.7763 and DeltarpoS 0.7966 of eMS57, against a stated "recovered its
  growth rate to $ 8 0 \% $ of that of eMS57". MS56 itself is 0.3094 of MG1655, which is
  the stated "severe growth reduction in M9 minimal medium".
- **The two growth panels disagree, so nothing may be derived across them.**
  `check_panels_disagree`: Fig. 2c's rates give eMS57/MS56 = 2.8159 while Fig. 2f
  releases 1.3990 for the same strain pair, a factor of 2.013. The build refuses to
  proceed if they ever agree, because that would mean a sheet changed meaning.
- **The 21 named genes are a contiguous MG1655 run and the released coordinates are
  not MG1655's.** `check_deleted_region`: all 21 symbols of
  `(hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan` resolve to `b2721` to
  `b2741`, spanning 2,844,762 to 2,867,551 (22,790 bp) with no unlisted gene inside. The
  paper's "genomic coordinates from 2,038,496 to 2,059,460 bp" (20,965 bp) are 806,266 bp
  upstream of that: they are MS56 coordinates, on an undeposited assembly, so no
  `GenomicSpan` can carry them and the gene-keyed deletions are what the records state.

### The one absence with no typed home

MS56 is carried as a `BacterialStrainBackground` with `genotype_statement` verbatim
("E. coli MG1655 with large deletions MD1 toMD56") and an EMPTY `alleles` list, the same
statement the landed Girgis 2009 loader makes for MG1655 delta-lacZ. This paper
enumerates none of the 55 regions and defers them to its reference 4, and MS56 has no
deposited assembly. Measured: a `ProvenanceGap` on `alleles` is REFUSED, because the
field defaults to `[]` and `ProvenanceGapMixin` requires a gapped field to be `None`
(`field 'alleles' has a ProvenanceGap but is not None`). The absence therefore lives in
`preprocess/genotype_gaps.json`, here, and in the issue.

Consequence: **MS56's own Fig. 2c growth rate is not loaded.** Written as a record it
would need an empty genotype against an MG1655 reference, which asserts that MS56 is
genotypically MG1655 while 1.1 Mbp of it is missing.

### Per-strain writability, from Supplementary Table 3

| strain | genotype, verbatim | verdict |
|---|---|---|
| MG1655 | Laboratory E. coli, train K-12, substr. MG1655 | writable (reference) |
| eMG1655 | E. coli MG1655 adaptively evolved in M9glucose medium | blocked, evolved clone |
| MS56 | E. coli MG1655 with large deletions MD1 toMD56 | background only |
| eMS57 | E. coli MS56 adaptively evolved in M9 glucosemedium | blocked, evolved clone |
| eMS57mutS+ | eMS57, puuP::mutS-kan | blocked, parent is evolved |
| MS56 Delta21kb | MS56, (hycEDCBA-hypABCDE-fhlA-ygbA-mutS-pphB-ygbIJKLMN-rpoS)::kan | **loaded** |
| MS56 DeltarpoS | MS56, rpoS::kan | **loaded** |
| cspCmut | MS56, cspC::cspC(G37A)-kanR | blocked, no sequence-variant leaf |
| ilvNmut | MS56, ilvN::ilvN(C202T)-kanR | blocked, no sequence-variant leaf |
| yifBmut | MS56, yifB::yifB(1169C insertion)-kanR | blocked, no sequence-variant leaf |

### What else is released and why it is not here

- **RNA-seq (Supplementary Data 6, 3,457 genes x 6 samples) and Ribo-seq (Supplementary
  Data 7)**: RPKM only. `RNASeqExpressionPhenotype` requires a per-gene integer
  `expression_count` alongside `expression_tpm` (`Field required`), and the count does
  not back-solve. Measured over the 3,391 b-numbers with a gene span: no assumed total
  read scale from 1 to 8 puts more than 3.4% of `RPKM x span` products within 0.01 of an
  integer. The MG1655 columns are wild type and would otherwise be writable, so this is a
  phenotype-class gap, not a genotype one. Gene-label resolution is good (3,447 of 3,457
  reach a b-number; 10 do not: `ade`, `rdoA`, `rhmA`, `rhmR`, `rhmT`, `spr`, `tsaA`,
  `waaJ`, `waaR`, `ybfK`).
- **Translation level and translational efficiency**: no phenotype class at all.
- **Supplementary Data 5, 839 sigma-70 ChIP-seq peaks**: no class.
- **Supplementary Data 1, the Biolog phenotype microarray** (384 wells over 4 plates x
  2 MG1655 + 2 eMS57 columns): the readout is "cellular respiration ... using an Omnilog
  instrument", an absolute endpoint dye-reduction signal. `MeasurementType` has no member
  for it.
- **Fig. 1a / 1b / 1d / 1h**: OD-over-time growth CURVES, not rates. No class holds a
  growth curve.
- **Fig. 1g, Fig. 3d and Supplementary Table 1**: metabolite levels and a fed-batch
  biomass plus specific growth rate for MG1655 and eMS57. The only non-wild-type strain in
  any of them is the evolved clone.
- **Fig. 2g**: MG1655's M9-plus-valine cell releases no number at all (the cells did not
  grow) and eMS57 is the only other strain in the panel.

### Measurement corrections over the first pass

Four, each measured:

- Supplementary Data 2 has **117** variant rows, not 118. The workbook carries a stray
  single-cell row (a lone `3` at column J, twelve rows past the last call) which a
  row-is-non-empty filter counts as a variant, and which also produced a phantom `None`
  mutation type. A row is a variant only when it has BOTH a gene and a position cell.
- Supplementary Data 3 writes its intergenic calls as a bare `-` and Supplementary Data 4
  as the bare word, so a `startswith("intergenic")` test reported **0** intergenic rows
  for Data 3 when it has **20**.
- The Keio panel is Supplementary Fig. **6**, not Fig. 5 (Fig. 5 is the `eMS57mutS+`
  construction).
- Supplementary Data 3 and 4 each carry a SECOND worksheet, 44 rows with no header,
  identical in both workbooks. It is a working sheet, not an independent release.

### Writable but not loaded

The 60 remaining Keio strains of Supplementary Fig. 6 have a stated qualitative call
("No strain showed significant growth retardation in M9 glucose medium") and no released
number, and the 62 deleted transcription factors are never listed, so they could not be
keyed to a locus even if a number existed. The legend also states that "Growth
retardation of DeltaydaS did not result from ydaS deletion1", citing Bindal 2017 on RacR.
That is a causal claim about the strain, so the measured rate is stored and the claim is
recorded in `preprocess/dropped_records.json` rather than altering the value.

### One correction the schedule row needs, not made here

`experiments/database/scripts/build_bacteria_candidate_datasets_table.py` gives this row
`genotypes_n=3`, "3 strains, 31 resequenced populations". Supplementary Table 3 lists
**13** strains: MG1655, eMG1655, MS56, eMS57, eMS57mutS+, the two designed MS56
deletions, and six reconstruction strains (three mutant alleles, each with its own
kan-marked wild-type-allele control). The undercount is load-bearing, because two of the
ten strains it omits are exactly the ones that loaded. The row is NOT edited in this
branch: the script writes nine committed `.tex` files and a counts table into
`notes-tex/database/database-expansion-bacteria/`, so the correction belongs with the
next regeneration of that document rather than in a dataset PR. Everything else the row
asserts is confirmed by this measurement, including the population-only genotype, the
absent MS56 accession and the bare gene names in both expression tables.

### Files

- loader: `torchcell/datasets/ecoli/choe2019_growth_rate.py`
- adapters: `torchcell/adapters/choe2019_growth_rate_adapter.py`,
  `torchcell/adapters/choe2019_tf_knockout_adapter.py`
- confs: `torchcell/adapters/conf/growth_rate_choe2019_adapter.yaml`,
  `torchcell/adapters/conf/tf_knockout_growth_choe2019_adapter.yaml`
- measurement: [[experiments.036-dataset-fixes-before-kg-build.scripts.choe2019_release_shape]]

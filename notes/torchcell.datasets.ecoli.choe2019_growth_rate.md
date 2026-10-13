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

## 2026.10.12 - #1057 on main: released rows stored, rebuilt and verified

Owner rule of 2026-10-12: a released row is stored under the identifier its release gives, and the genome never gates storage. Rebuilt from main `db632f2b0` with `--retire-existing --verify` (slurm array 4506; the stores with no per-dataset verifier entry point verified by job 4582).

### `growth_rate_choe2019`: 2 -> 5 records

Row 22: the three Fig. 2f reconstructed alleles against their kan-marked controls.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 5 records validated |
| L1 | `count` | pass | observed 5, expected 5 |
| L1 | `pair_uniqueness` | pass | 5 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 5 values checked |
| L2 | `se_nonnegative` | pass | 5 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 5 records |
| L1 | `provenance_gaps` | pass | 5 documented provenance gaps over 5/5 records; 0 deferred field(s): []; 65 undeclared None values over 7 carrier fields (top: Compound.inchi x35, Compound.chebi_id x5, Environme... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 3 systematic names are not the genome's current name; 0 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 5 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 35 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 5 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.875 of 24 measured genes are MG1655 (ecoli_K12_MG1655_ASM584v2) genes (>= 0.9) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 3 systematic names are absent from the current genome over 3 records: cspC x1, ilvN x1, yifB x1 |

### `tf_knockout_growth_choe2019`: 2 -> 2 records

Rebuilt with its module; records unchanged.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 2 records validated |
| L1 | `count` | pass | observed 2, expected 2 |
| L1 | `pair_uniqueness` | pass | 2 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 2 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 2 records |
| L1 | `provenance_gaps` | pass | 0 documented provenance gaps over 0/2 records; 0 deferred field(s): []; 36 undeclared None values over 12 carrier fields (top: Compound.inchi x14, Compound.chebi_id x2, Environm... |
| L1 | `canonical_gene_names` | pass | 2 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 14 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 2 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 2 measured genes are BW25113 (ecoli_K12_BW25113_ASM75055v1) genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 2 measured systematic names is a gene of the current genome |

## 2026.10.12 - #1045 part 2: the rest of the released rows, under their released identifiers

Every row #1057 still dropped that is not an author withdrawal is stored, and no identifier is translated: the stored key is the one the release prints, and a crosswalk to an annotation is a modeling-time join outside the record. Rebuilt from the part-2 branch with `--retire-existing --verify` (slurm array 4583; the stores with no per-dataset verifier entry point verified by jobs 4645 and 4644).

### `growth_rate_choe2019`: 5 -> 5 records

The 22 designed deletions under their released symbol; the contiguous-run check is a genome-reference verification row.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 5 records validated |
| L1 | `count` | pass | observed 5, expected 5 |
| L1 | `pair_uniqueness` | pass | 5 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 5 values checked |
| L2 | `se_nonnegative` | pass | 5 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 5 records |
| L1 | `provenance_gaps` | pass | 5 documented provenance gaps over 5/5 records; 0 deferred field(s): []; 65 undeclared None values over 7 carrier fields (top: Compound.inchi x35, Compound.chebi_id x5, Environme... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 24 systematic names are not the genome's current name; 21 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 5 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 35 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 5 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.000 of 24 measured genes are MG1655 (ecoli_K12_MG1655_ASM584v2) genes (>= 0.9) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 24 systematic names are absent from the current genome over 5 records: cspC x1, fhlA x1, hycA x1, hycB x1, hycC x1, hycD x1, hycE x1, hypA x1, hypB x1, hypC x1 |
| L4 | `deleted_symbols_are_one_contiguous_run_of_the_pinned_assembly` | pass | the 21 released deletion symbols are one contiguous run of ecoli_K12_MG1655_ASM584v2 with no unlisted gene inside |

### `tf_knockout_growth_choe2019`: 2 -> 2 records

The two Keio deletions under their released symbol.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 2 records validated |
| L1 | `count` | pass | observed 2, expected 2 |
| L1 | `pair_uniqueness` | pass | 2 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 2 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 2 records |
| L1 | `provenance_gaps` | pass | 0 documented provenance gaps over 0/2 records; 0 deferred field(s): []; 36 undeclared None values over 12 carrier fields (top: Compound.inchi x14, Compound.chebi_id x2, Environm... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 2 systematic names are not the genome's current name; 2 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 14 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 2 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.000 of 2 measured genes are BW25113 (ecoli_K12_BW25113_ASM75055v1) genes (>= 0.9) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 2 systematic names are absent from the current genome over 2 records: abgR x1, ydaS x1 |

## 2026.10.13 - #1045 part 3: the oxygen, growth-phase and released-strain samples stored

The schema gained a typed gap on Environment.aerobicity, the regime 'transition', Environment.growth_phase and PhysicalFactor.dissolved_oxygen (record 2026-10-13-environment-aerobicity-gap-transition-growth-phase-dissolved-oxygen), so no released sample of these loaders is dropped. Rebuilt from the part-3 branch (`c61b1310a`) with `--retire-existing --verify` (slurm arrays 4640 and 4643; the stores with no per-dataset verifier entry point verified by job 4641).

### `growth_rate_choe2019`: 5 -> 8 records

MS56 over MG1655 on MS56's named background, and eMS57 over MS56 from the Fig. 2c rates and from the Fig. 2f released ratios (screen_id Fig2c and Fig2f; the panels disagree by 2.013), on a background named eMS57 whose genotype statement is its Supplementary Table 3 row; the fitness verifier's L1 strain key now names the reference background.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 8 records validated |
| L1 | `count` | pass | observed 8, expected 8 |
| L1 | `pair_uniqueness` | pass | 8 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 8 values checked |
| L2 | `se_nonnegative` | pass | 8 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 8 records |
| L1 | `provenance_gaps` | pass | 8 documented provenance gaps over 8/8 records; 0 deferred field(s): []; 102 undeclared None values over 7 carrier fields (top: Compound.inchi x56, Compound.chebi_id x8, Environm... |
| L1 | `canonical_gene_names` | finding (genome reference; reports, does not gate) | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 24 systematic names are not the genome's current name; 21 common names resolve to another gene |
| L2 | `uncertainty_sanity` | pass | 8 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 56 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 8 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | finding (genome reference; reports, does not gate) | 0.000 of 24 measured genes are MG1655 (ecoli_K12_MG1655_ASM584v2) genes (>= 0.9) |
| L4 | `current_genome_genes` | finding (genome reference; reports, does not gate) | 24 systematic names are absent from the current genome over 5 records: cspC x1, fhlA x1, hycA x1, hycB x1, hycC x1, hycD x1, hycE x1, hypA x1, hypB x1, hypC x1 |
| L4 | `deleted_symbols_are_one_contiguous_run_of_the_pinned_assembly` | pass | the 21 released deletion symbols are one contiguous run of ecoli_K12_MG1655_ASM584v2 with no unlisted gene inside |

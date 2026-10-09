---
id: sl1yrqpjczo1b9hculo6yrz
title: Borchert2023
desc: ''
updated: 1791449698939
created: 1791449698939
---

## 2026.10.08 - Row 29: what the compendium already serves, and the 271 loci it dropped

Borchert, Bleem and Beckham 2023 (Metab Eng 77:208-218, doi 10.1016/j.ymben.2023.04.007,
citation key `borchertRBTnSeqIdentifiesGenetic2023`) is row 29 of the fifty bacterial rows.
Fourteen RB-TnSeq experiments in biological triplicate on the KT2440 `ML-5` library: M9 +
20 mM glucose against eleven lignin-relevant stressors, a second glucose reference and a
protocatechuate enrichment run on a later day.

Almost all of it is already served. `RbTnseqBorchert2024Dataset` stores the Borchert 2024
compendium, and 42 of its 332 samples ARE these 42 cultures. This note records the
measurement that proves it, the one thing that is genuinely new, and the three released
quantities that have no home in the schema.

### The subsumption measurement, reproduced and extended to all 13 sheets

`si/si1.xlsx` holds 13 pairwise comparison sheets. Each has four identifier columns, then
three reference replicates and their mean, three condition replicates and their mean, then
`t-statistic`, `p-value`, `q-value` and `adjusted_q-value`. Across the 13 sheets the
replicate columns resolve to **42 distinct (experiment, replicate) arms**, which is Table
S3's own `Exp./Rep.` key `Exp1A` to `Exp14C`.

Measured by `match_arms` in `torchcell/datasets/pputida/borchert2023.py`, which the build
re-runs: each arm is joined to the compendium's `fitness_measurements` sheet on the locus
tag and compared against all 332 sample columns.

| statistic | value |
|---|---|
| arms measured | 42 of 42 |
| shared loci per arm | 4,732 |
| largest absolute difference, over all 42 arms | 0.000500 |
| smallest runner-up distance, over all 42 arms | 1.7885 |
| (locus, arm) fitness values that are therefore already served | 198,744 |

0.0005 is exactly the half-unit of the compendium's three-decimal rounding, and Borchert
2023 releases full precision, so this is one measurement exported twice. The runner-up
column is never within 1.78, so the identification is not a coincidence.

The earlier audit compared ONE sheet (`Glu_v_Glu_Van`) and header-inspected the other
twelve. All thirteen are now compared numerically and all thirteen follow. The per-arm
table is written to `preprocess/subsumption.json`.

**An independent corroboration that was not used as the proof.** Table S3 gives the BarSeq
`IT` index of every (experiment, replicate): `1A` is `IT08`, `1B` `IT09`, `1C` `IT10`, and
so on. The compendium's sample names encode the same index: `set100IT008`, `set100IT009`,
`set100IT010`. All 42 agree with the numeric match, including the two later experiments,
which sit in `set101` rather than `set100`. The build asserts the numeric argmin equals the
declared sample, so a drift in either direction stops it.

**Experiments 1 and 13 share a column name and are not the same culture.** Twelve sheets
carry a column literally named `M9_Glucose_RepA`. Measured: the eleven day-1 sheets agree
to exactly 0.0, and the `Glu_v_Glu_PCA` sheet differs from them by up to 6.93. That is the
paper's own statement made numeric, "The protocatechuate enrichment experiment (with
corresponding M9 + 20 mM glucose alone enrichment) was performed on a separate day from all
other experiments". They are two arms, `Exp1*` and `Exp13*`, matching `set100IT008-010` and
`set101IT007-009`.

**The `_mean` columns are derived.** Measured over all 26 of them: each is the arithmetic
mean of its own three replicates to at most 7.3e-14. Not a measurement, not stored.

### What IS new: 271 loci the compendium eliminated

The compendium states its own filter: "In instances where gene fitness data for a
particular gene did not exist across all 332 data sets, the gene was eliminated from
analysis", which took 832 of the 5,564 protein-coding genes. Measured here: **271 of those
eliminated loci carry full triplicate fitness in at least one Borchert 2023 comparison
sheet.** All 271 resolve as current `pputida_KT2440_ASM756v2` locus tags, 0 remapped.

Summed over the arms that carry each one, that is **10,824 (locus, arm) values the
compendium does not have**, and they are what `RbTnseqBorchert2023Dataset` stores. The
per-arm lists are in `preprocess/new_loci.json`.

### Records

One `BacterialEnvironmentResponseExperiment` per (locus, arm).

- **Genotype**: `TransposonInsertionPerturbation` of the locus in `Putida_ML5_JBEI`, built
  through `borchert2024.build_genotype`, so a record joins the served ones on the strain.
  Borchert 2023 names the library "(ML-5)"; the compendium labels these same 42 cultures
  `Putida_ML5_JBEI`, which is the `library_pool` stored.
- **Environment**: `borchert2024.BORCHERT2023_M9` (the same `Media` object the served
  records use, so the graph keeps one media node), 30 C, D-glucose 20 mM as an
  `EnvironmentPhysicalPerturbation`, and for the twelve stressed arms the stressor as a
  `SmallMoleculePerturbation` at the Methods dose.
- **Phenotype**: `EnvironmentResponsePhenotype`, `measurement_type=log2_ratio`,
  `assay_type=pooled_competitive_growth_barcode`, `screen_id` the arm name. `FitnessPhenotype`
  clamps non-positive values to 0.0, which would erase every fitness defect, so the signed
  log2 ratio belongs here, exactly as the compendium loader stores the same measurement.
- **Reference**: the typical gene of the same culture, fitness 0.0, by the paper's own
  normalization ("normalized by subtracting the median unnormalized fitness within a 251
  gene sliding window").

### Growth duration: a gap on the served records, a value here

Table S4 releases the OD600 reached and the **total growth duration per replicate** at the
time of sampling. The compendium has no growth-time column, so `borchert2024` records
`duration_hours` as a `ProvenanceGap` for all 332 samples, these 42 included. Here it is
filled, per replicate, from Table S4.

| experiment | condition | A | B | C |
|---|---|---|---|---|
| 1 | M9 + 20 mM glucose | 10:40 | 10:05 | 10:05 |
| 2 | + 60 mM 4-coumarate | 12:45 | 12:25 | 11:55 |
| 3 | + 60 mM ferulate | 14:25 | 15:10 | 15:20 |
| 4 | + 10 mM 4-hydroxybenzaldehyde | 19:30 | 19:30 | 17:50 |
| 5 | + 60 mM 4-hydroxybenzoate | 9:30 | 10:04 | 9:30 |
| 6 | + 10 mM vanillin | 13:15 | 15:10 | 13:15 |
| 7 | + 60 mM vanillate | 18:40 | 18:40 | 20:00 |
| 8 | + 500 mM NaCl | 19:30 | 20:30 | 20:00 |
| 9 | + 500 mM Na2SO4 | 24:40 | 24:40 | 24:00 |
| 10 | + 75 mM acetate | 30:25 | 30:25 | 31:00 |
| 11 | + 500 mM lactate | 25:10 | 26:20 | 24:40 |
| 12 | + 125 mM glycolate | 15:20 | 16:40 | 16:40 |
| 13 | *M9 + 20 mM glucose | 10:10 | 10:50 | 10:10 |
| 14 | *+ 30 mM protecatechuate | 20:20 | 20:20 | 18:10 |

Each row is a `SourcedValue` carrying the pinned OCR's own `<tr>` as its quote, and the
whole table was re-read off page S5 of `si/si2.pdf` to catch an OCR digit. The asterisk is
Table S4's own, "These experiments were performed later than the preceding experiments".

**Follow-up on the served dataset.** Borchert 2024's `duration_hours` gap is now closable
for its 42 Borchert 2023 samples from this same table. Doing it changes served records, so
it is a full-rebuild change, not an increment, and it is not done here.

### The compound naming decision, measured

Borchert 2023's Methods name the conjugate bases ("4-coumarate", "ferulate", "vanillate").
Measured over the twelve stressors through the shared compound-identity table: 4 of 12
resolve to a structure under both spellings, and 8 of 12 resolve only under the
compendium's acid spellings (`4-coumarate` gives a name-only compound with no InChIKey,
`p-Coumaric acid` gives `NGSWKAQJJWESNS-ZZXKWVIFSA-N`). Using Borchert 2023's spellings
would put two compound nodes in the graph for one chemical and fail the L3
`compound_identity` rule. So the agent is named as the served records name it, and the
build asserts Borchert 2023's own Methods dose equals the compendium's for every arm.

### A released dose disagreement, 3 to 1

The `Read_Me` description of sheet `Glu_v_Glu_4HBald` says "20 mM 4-hydroxybenzaldehyde".
The Methods say 10 mM, Table S4 says 10 mM and the compendium's `set100IT017-019` metadata
says 10 mM. 10 mM is stored. Recorded in `preprocess/not_stored.json` as
`read_me_conflict`.

### What is NOT stored, and why

| quantity | where | values | why |
|---|---|---|---|
| per-replicate fitness of the 4,732 compendium loci | 13 sheets, 42 arms | 198,744 | measured identical to the served Borchert 2024 records, to 0.0005 |
| the `_mean` of each arm triple | 13 sheets, 26 columns | 129,706 | measured derived from the three replicates, to 7.3e-14 |
| `t-statistic`, `p-value`, `q-value`, `adjusted_q-value` | 13 sheets, 4 columns each | 259,412 | **no phenotype class carries a significance triple** |
| per-barcode read counts | `Exp1_All_Poolcount`, `Exp2_All_Poolcount` | 7,852,194 | raw sequencing read tallies; no loader ingests raw reads |
| OD600 growth curves | `Figure_2..6_growth_data` | time series | no optical-density `MeasurementType` member, no time-series carrier |
| OD600 at sampling, per replicate | Table S4 | 42 | the same missing enum member |

#### The significance triple has no home, and that is the schema finding

Borchert 2023's own description: "Comparison of mean fitness values between enrichment and
medium reference cultures (M9 + 20 mM glucose alone) was performed using a two-sample t
test, where the p value was corrected for multiple testing via the positive false discovery
rate (pFDR) method", and "pFDR q values were then adjusted for monotonicity (Yekutieli and
Benjamini, 1999), and both unadjusted and adjusted q values are reported". That is 64,853
(gene, comparison) rows, each with four numbers.

`EnvironmentResponsePhenotype` has no field for any of them. Its uncertainty axis is
`environment_response_se` plus `environment_response_uncertainty` and an
`UncertaintyType`, which are dispersions of the value, not a test of it.
`GeneInteractionPhenotype.gene_interaction_p_value` is the only p-value field anywhere in
the schema and it is not applicable to a fitness contrast. `UncertaintyType` has
deliberately no `unknown`, so there is no honest slot to put a q value in.

A second, structural reason the triple cannot simply be added to a record: it is a
different GRAIN. A stored record is one (locus, culture) fitness. A t, p, q and adjusted-q
belong to one (locus, comparison of two cultures) contrast. Even with a field, the triple
would need a record whose subject is a pair of environments, which no experiment class
expresses.

Storing the fitness and dropping the significance call silently is what the refusal avoids.
The measurement is recorded on **issue #776**, which already asks for fields on this same
class and is already scoped as a full knowledge-graph rebuild. No near-duplicate issue was
opened.

#### The barcode grain: the perturbation leaf exists, the phenotype does not

`Exp1_All_Poolcount` and `Exp2_All_Poolcount` carry 186,957 unique barcodes with
`barcode`, `rcbarcode`, `scaffold`, `strand`, `pos`, `locusId` and `f`, then 42 count
columns. Measured: 152,809 of the 186,957 rows carry a `locusId`, the scaffolds are
`AE015451` and the `pastEnd` pseudo-scaffold, and `f` runs 0.0 to 1.0.

`TransposonInsertionPerturbation` ALREADY carries `barcode`, `insertion_position` and
`insertion_strand`, so a barcode-grain genotype is expressible today and this is not a
perturbation-leaf gap. The blocker is entirely on the phenotype side: the released value is
a raw sequencing read tally per barcode per sample, which is the input to the fitness
calculation rather than a phenotype, no class holds a read count and `MeasurementType` has
no member for one. 34,148 rows have no `locusId` at all, so a gene-keyed record could not
be written for them either.

### Verification

`verify_build` on the dev store, 16 rules, all pass.

| level | rule | result |
|---|---|---|
| L0 | structural | PASS, 10,824 records validated |
| L1 | count | PASS, observed 10,824, expected 10,824 |
| L1 | pair_uniqueness | PASS, 10,824 unique (study, strain, condition) |
| L1 | provenance_gaps | PASS, 57,120 documented gaps over 10,824 records |
| L1 | canonical_gene_names | PASS, 271 names, each current in the genome |
| L2 | value_fidelity | PASS, 10,824 values checked |
| L2 | se_nonnegative | PASS, 0 values (no SE is released per replicate) |
| L2 | uncertainty_sanity | PASS, 0 labeled uncertainties |
| L3 | measurement_type_consistent | PASS, single `log2_ratio` |
| L3 | reference_zero | PASS, numeric rule, reference response 0 for all |
| L3 | environment_perturbed | PASS, all 10,824 carry an environmental edit |
| L3 | compound_identity | PASS, 17,241 structure identifiers, 3,000 typed gaps |
| L3 | media_compound_identity | PASS, 75,768 structure identifiers, 0 gaps |
| L3 | media_membership | PASS, 10,824 on a medium deriving from a library medium |
| L4 | gene_containment_sgd | PASS, 1.000 of 271 |
| L4 | current_genome_genes | PASS, all 271 |

### The build-time partition, asserted in both directions

`process()` refuses to write unless all of the following hold, so a drift stops the build
rather than producing a silent duplicate:

1. every arm matches its declared compendium sample within 0.0005 on all 4,732 shared loci,
   with the runner-up at least 0.5 away (`match_arms`);
2. an arm exported by more than one sheet carries identical values in each, and the two
   glucose arms differ (`assert_arms_agree_across_sheets`);
3. every `_mean` is the mean of its replicates (`assert_means_are_derived`);
4. the new-locus count is exactly 271;
5. against the **already-served store** (`assert_served_partition`, written to
   `preprocess/served_partition.json`): its gene set is exactly the pinned compendium's
   4,732, none of the 271 stored loci is among them, and all 42 matched sample names appear
   as served `screen_id` values. The served LMDB is streamed once through
   `stream_records` and fully consumed, so the environment is closed before anything
   re-reads that path.

### Mirror

Raw mirror `$DATA_ROOT/torchcell-raw/borchertRBTnSeqIdentifiesGenetic2023/data/si1.xlsx`,
sha256 `b80c6866c6fbb95696067532f4890951fcbbf2dc72f4351df0bdec86956a0784`, 70,872,248 bytes,
retrieved through `torchcell.literature.retrieve.elsevier_mmc` from
`https://ars.els-cdn.com/content/image/1-s2.0-S1096717623000599-mmc1.xlsx`, with a live
`last_check` recorded on 2026-10-08 that reproduced the same sha256. `si/si2.pdf` is NOT
deposited in the raw mirror: only its Table S4 is read, as verbatim quotes against the
library mirror's pinned OCR `si/si2.md`
(`568d7edfa237314d9ac775b1cb4bd69645dbf4e3ba4ba03e106f1f5760c23891`). All 30 sourced quotes
pass `audit_sourced_value` against the library mirror.

### Files

- Loader: `torchcell/datasets/pputida/borchert2023.py`
- Adapter: `torchcell/adapters/borchert2023_adapter.py`, conf
  `torchcell/adapters/conf/rbtnseq_borchert2023_adapter.yaml` (measured identical to the
  compendium's enable-list, which is why no new graph node class is declared)
- Tests: `tests/torchcell/datasets/pputida/test_borchert2023.py`,
  `tests/torchcell/adapters/test_borchert2023_adapter.py`
- Dev store: `$DATA_ROOT/data/torchcell/rbtnseq_borchert2023/`

## 2026.10.09 - The pair-of-environments question (#776): NOT a new experiment class

#776's second, structural half asks whether an experiment class whose SUBJECT is a pair
of environments is warranted now, for Borchert 2023's 259,412 significance values. The
answer measured here is no, and the reason is that the pair-of-environments grain is
already expressible; what is missing is a phenotype field, which is a different ask.

### The counts, re-read off the built store

`preprocess/not_stored.json` of
`$DATA_ROOT/data/torchcell/rbtnseq_borchert2023`, measured on `si1.xlsx` (sha256
`b80c6866c6fbb95696067532f4890951fcbbf2dc72f4351df0bdec86956a0784`):

| | value |
|---|---|
| comparison sheets | 13 |
| (gene, comparison) rows | **64,853** (5002 + 5001 + 4997 + 5002 + 5000 + 5000 + 4931 + 4993 + 5000 + 4999 + 4998 + 5000 + 4930) |
| significance columns per sheet | 4 (`t-statistic`, `p-value`, `q-value`, `adjusted_q-value`) |
| values with nowhere to go | **259,412** |

### Why a pair of environments needs no new experiment class

A record already names TWO environments: `Experiment.environment` is the condition
measured and `ExperimentReference.environment_reference` is the condition it is referenced
against, with the asymmetry (which one is the control) carried by the reference. That is a
pair, and it is already used as one. Measured on the served Bloom 2019 store's condition
table: **36 of its 38 conditions** have a reference environment that DIFFERS from the
experiment environment, and for exactly those 36 the stored value is
`MeasurementType.control_regression_residual`, a contrast statistic of the two plates. A
Borchert comparison sheet is the same shape: the enrichment culture is the experiment's
environment, the M9 + 20 mM glucose medium reference culture is the reference's.

So building a class whose subject is a pair would duplicate a capability the family has,
and would do it by making the reference a second first-class subject, which every
`ExperimentReference` consumer (the reconstruction maps, the adapters, the L3
`reference_zero` family of rules) reads as a CONTEXT. That is a large, cross-cutting
change bought for nothing: it would still not give the triple a field.

### What IS missing, and why it is still not stored

A phenotype whose value is a contrast statistic WITH a significance triple. The loader's
current records are per-culture fitness, and a `t`, `p`, `q` and adjusted `q` of a
contrast attached to one of its two arms would misfile the statistic, which is the
objection #776 raises and it stands. The honest form is a contrast-valued phenotype that
declares `p_value` and an adjustment method alongside its value, so the number and the
test of it sit on the same record.

That shape is being built on a different phenotype in this same wave (#770's
`ProteinFoldChangePhenotype`, which carries `log2FoldChange` with `lfcSE`, `pvalue` and
`padj`), so the precedent for the fields will exist. Reusing it for a FITNESS contrast is
a loader-plus-phenotype decision of its own, with its own sourcing (the pFDR and
monotonicity-adjustment methods Borchert states verbatim), and it is not inside #776's
scope. Until then the triple stays refused and declared in `not_stored.json` with its
count and the issue number, which is the discipline that keeps a stored fitness value from
silently losing its significance call.

Recommendation recorded for the owner: open a follow-up for a fitness-contrast phenotype
with a significance triple, and close #776's structural half against it. Nothing about the
pair-of-environments grain blocks it.

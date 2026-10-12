---
id: f76ow55c2g80hhpspg6snzi
title: Vaishnav2022
desc: ''
updated: 1791808952211
created: 1791808952211
---

## 2026.10.12 - Vaishnav 2022 sort-seq promoter expression (#991)

The first dataset on the pooled sort-seq family: one record per released promoter
sequence, the sequence itself on the genotype. Schema PR #1055 landed the two classes it
needs (`VariablePromoterReporterPerturbation`, `PooledSortSeqExpressionPhenotype`) and
the `ReporterReadout.sort_seq_fluorescence` member.

- loader: `torchcell/datasets/scerevisiae/vaishnav2022.py`
  (`PooledPromoterExpressionVaishnav2022Dataset`)
- verifier: `torchcell/verification/pooled_sort_seq.py`, registered as
  `POOLED_SORT_SEQ_DATASETS` and in `run_all`
  ([[torchcell.verification.pooled_sort_seq]])
- adapter: `torchcell/adapters/vaishnav2022_adapter.py`
  ([[torchcell.adapters.vaishnav2022_adapter]])
- inventory script: `experiments/database/scripts/vaishnav2022_release_inventory.py`
  (its JSON lands in the untracked `experiments/database/results/`)
- raw mirror: `$DATA_ROOT/torchcell-raw/vaishnavEvolutionEvolvabilityEngineering2022/`
- candidate verdict: `database/candidates/vaishnavEvolutionEvolvabilityEngineering2022.json`,
  **admissible** (G1-G5 pass) on this tree

### What the release is, measured

Counted by the inventory script off the sha256-pinned mirror, never from the abstract:

| quantity | value | where |
|---|---|---|
| rows in `GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz` | 75,487 | GEO GSE163045 |
| distinct `seqID` groups | 71,158 | same file |
| distinct `seq110` sequences | 71,158 | same file |
| rows whose `origID` is its own `seqID` | 67,690 | same file |
| groups whose `ExpressionLevel` cell is the verbatim `NA` | 104 (110 rows) | same file |
| **records the loader builds** | **71,158** | `read_measurements` on the real files |
| of those, carrying an estimate | 71,054 | same |
| of those, carrying a typed gap on the estimate | 104 | same |
| expression range | 0.3206 to 17.0 | same file |
| Zenodo defined-medium test-set rows | 5,327 | Zenodo 4436477 |
| of those, joining the GEO table by `origID` | 5,327, all to distinct `seqID`s | measured |
| largest absolute disagreement between `meanEL` and `ExpressionLevel` | 7.1e-15 | measured |
| read counts (`proTotal`) | 1 to 22,680; 194 rows under 100 | measured |

The file name is the decision: it is `splitByOrigID`, one ROW per design label of a
measurement whose own identity is `seqID`. Within a group there is exactly one sequence
and exactly one `ExpressionLevel` cell, and every row's `origID` is a member of its own
`seqID` list (all three asserted by `read_measurements`, not assumed), so one record per
`seqID` drops no row's label and a row-per-record store would have held 4,433
measurements more than once.

**The release decides membership, and nothing else does.** Owner rule, 2026-10-12: "the
data is the data ... the genome is separate.. it doesn't gate anything". Every `seqID`
the file lists is a record, including the 104 whose cell is the verbatim `NA`: those
carry `expression_level = None` beside a typed `ProvenanceGap` on that field, whose
`looked_in` names the pinned file and whose note quotes the cell. An earlier revision of
this loader refused them, which made the store hold 71,054 rather than the released
71,158 and turned a release absence into a silent drop. The leaf enforces the rule
(`_require_value_or_gap` on `expression_level`), and the verifier's
`estimate_is_stated_or_gapped` and `released_values` levels both check it: an absence
the release states must be an absence in the store, and a sequence the release does not
list at all is not a record. No lookup against the genome or any other tree takes part
in that decision: this loader injects no genome and resolves no gene name.

The Zenodo test set is the same measurements, so it is not loaded as records. It supplies
the two released per-record values the GEO table does not carry: its `expt` token
(`NBT_S288CdU_SCUra`, stored as `library_id`) and `proTotal` (stored as `read_count`), on
the 5,327 records it covers. The other 65,727 carry `None` for both, which is what the
release states about them.

### Store build and L0-L4: QUEUED, not yet measured

The dev store is slurm array job **4269** (`991-build-vaishnav2022`, the committed
`gilahyper_build_dataset_lmdbs_array.slurm` with `--retire-existing --verify`), submitted
2026-10-12 and still PENDING behind 58 other jobs on the node. So the L0-L4 table is NOT
in this note yet, and nothing here claims a verified store.

What IS measured, in-process on the real sha256-pinned files rather than on a built store:
`read_measurements` returns 71,158 rows (71,054 with an estimate, 104 gapped), 5,327 of
them carrying `library_id` and `read_count`, and records built from sampled rows carry
the 18-97 span. The verifier's own
run (L0 structural, L1 count and sequence uniqueness, L2 value fidelity and released-value
fidelity, L3 units, sequence-is-the-genotype, typed gaps and stated-or-gapped estimate,
L4 read-count coverage) lands
in `preprocess/verification_report.json` when the job runs, and this section gets its table
then.

### Sourced values

Every hardcoded schema value is in `SOURCED_VALUES` with its verbatim quote, and the
verifier re-reads each one against the mirror when the library is mounted. The two that
decide what the number means:

- the estimator, Vaishnav 2022 Methods: "For each sequence, the expression level was the
  average of the expression bins in which it was observed, weighted by the number of times
  it was observed in each bin."
- the same estimator in the method's deferral target, de Boer 2020: "Expression levels for
  each promoter sequence were estimated as the weighted average of bins in which the
  promoter was observed."

Three of the ten values are sourced to de Boer 2020 rather than to this paper, because the
quote must be auditable and `audit_sourced_value` reads its source as TEXT: this paper's
Supplementary Notes are a PDF the mirror has not OCR'd, and de Boer 2020 (mirrored, the
method's deferral target) states the same facts in prose. The growth temperature is
sourced both ways, from de Boer's GPRA culture sentence and from this paper's own "Unless
otherwise noted, yeast were grown at 30 C" line.

So the stored number is a read-weighted mean index over the 18 uniform log(YFP:RFP) sort
bins, and `read_count` counts the observations it averaged.

### Three decisions worth stating

1. **Strain.** The loaded table is written against `S288C::ura3`, on the SI sentence
   ("For random libraries, the strain Y8205 was used, but later experiments including the
   designed libraries were performed in S288C::ura3"), the Zenodo `expt` token on the
   overlapping rows, and the fact that the Y8205 random training set is a separate GEO
   file whose own name carries the strain. The GEO table has no per-row strain column, so
   this is a file-level assignment and is labeled as one in the loader docstring.
2. **No uncertainty is stored.** MEASURED: `sdEL == sdScale * relativeSD` on all 5,327
   Zenodo rows to 2.0e-14, so `sdEL` is an author-scaled quantity rather than a dispersion
   of the reads, and neither this paper (Methods, Supplementary Notes) nor de Boer 2020
   defines any of the three columns. The uncertainty type therefore cannot be labeled, and
   the schema admits no unlabeled uncertainty, so all four uncertainty and replicate fields
   carry typed `ProvenanceGap`s naming the columns. `n_samples` is gapped for its own
   reason: the estimate averages reads, and a read is not a replicate of the construct.
   **Owner question:** ask the authors what `sdEL`, `relativeSD` and `sdScale` are, or
   leave the gap standing.
3. **The reference restates the record.** The loaded files carry no control-promoter arm:
   the library's internal control is the constitutive RFP and the released level is already
   a ratio against it ("Cells were sorted by the log ratio of RFP to YFP signal"). The
   reference phenotype is therefore the record's own reading, as the untreated arm of
   Mohiuddin 2022 is its own reference, rather than a baseline computed here. The paper's
   measured native-promoter arm is in a Zenodo file this loader does not read.

### Deferred, with sizes

A later admission, each a size decision for the owner rather than a schema or provenance
one (byte counts from the hosts' own listings, in the inventory JSON):

| file | host | bytes | what |
|---|---|---|---|
| `defined_media_training_data_SC_Ura.txt` | Zenodo 4436477 | 2,473,729,542 | 20,616,659 random sequences, defined medium |
| `complex_media_training_data_Glu.txt` | Zenodo 4436477 | 3,592,193,655 | 30,722,376 random sequences, complex medium |
| `GSE163045_average_promoter_ELs_per_seq_N80_SD-Ura_Y8205_ALL.shuffled.txt.gz` | GEO | 668,401,864 | the Y8205 random training library |
| `Random_testdata_complex_media.txt` | Zenodo 4436477 | 427,196 | the YPD arm's test slice |

The YPD slice is deferred for a different reason than size: its arm's full measured
release is the 3.59 GB file above, so loading the slice alone would make that medium's
coverage an artifact of the size decision. `si/si3.xlsx` and
`S288C_reference_promoter_expression_in_both_media.tsv` are model PREDICTIONS and are not
candidates.

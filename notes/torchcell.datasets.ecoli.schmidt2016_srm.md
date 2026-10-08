---
id: l9a9obfza72hv43xyx8sc6c
title: Schmidt2016_srm
desc: ''
updated: 1791435711053
created: 1791435711053
---

## 2026.10.08 - Loader, sourcing and build

Schmidt et al. 2016's Supplementary Tables 2 and 3: the 41-protein SRM + stable-isotope-
dilution panel the proteome-wide estimates were anchored on, a DIFFERENT assay of proteins
the stored Table S6 block also covers. The loader is
`torchcell/datasets/ecoli/schmidt2016_srm.py`, the classes are
`ProteomeSrmSet1Schmidt2016Dataset` (Table S2) and `ProteomeSrmSet2Schmidt2016Dataset`
(Table S3), and the test module is
`tests/torchcell/datasets/ecoli/test_schmidt2016_srm.py`. The paper, the citation key, the
pinned `si2.xlsx` and the media objects are the ones
[[torchcell.datasets.ecoli.schmidt2016]] already records; this note states only what is
new.

### Why a separate module, and why TWO classes

**Separate module from `schmidt2016.py`**, for the reason
[[torchcell.datasets.ecoli.schmidt2016_growth_rate]] states in full: `build_manifest` keys
a built store's staleness on the schema closure of the loader MODULE's own
`torchcell.datamodels` imports, so co-locating would have marked the served
`proteome_schmidt2016` store stale for a change that touches none of its records.
Measured: it reads `fresh` before and after this branch.

**Two classes, not one**, because `verify_protein_dataset`'s L3
`measurement_type_consistent` requires every record of one dataset to share a single
`measurement_type` -- the rule that exists so heterogeneous proteomics assays are never
silently mixed. The two arms differ in what their released dispersion IS (two SRM
injections of one sample against three independently grown cultures), so they are two
measurement types and therefore two datasets. Merging them would also put two records on
the same (genotype, environment) for the twelve conditions both arms measured, with nothing
on `ProteinAbundancePhenotype` to tell them apart: unlike `FitnessPhenotype` and
`EnvironmentResponsePhenotype` it has no `screen_id`. Making the arm the DATASET identity
is what the knowledge graph already distinguishes by.

### What is stored

| field | `...SrmSet1...` | `...SrmSet2...` |
|---|---|---|
| sheet | Table S2 (data set 1, OGE-fractionated) | Table S3 (data set 2, unfractionated) |
| released data rows | 779 = 41 proteins x 19 conditions | 682 = 31 proteins x 22 conditions |
| records | 11 | 14 |
| protein keys per record | 41 | 31 |
| `measurement_type` | `absolute_protein_copies_per_cell_srm_sid_dataset1_technical_sd` | `absolute_protein_copies_per_cell_srm_sid_dataset2_biological_sd` |
| `n_replicates` | 2 (SRM technical duplicate) | 3 (biological triplicate) |
| stored SE | released SD / sqrt(2) | released SD / sqrt(3) |
| store | `$DATA_ROOT/data/torchcell/proteome_srm_set1_schmidt2016` | `$DATA_ROOT/data/torchcell/proteome_srm_set2_schmidt2016` |

Both: `BacterialProteinAbundanceExperiment` / `...Reference`, protein copies per cell keyed
by BW25113 locus tag, `Genotype(perturbations=[])` on every record, the glucose arm as the
`phenotype_reference`, assembly pin `ecoli_K12_BW25113_ASM75055v1` / `GCA_000750555.1`, and
one consumed file, `$DATA_ROOT/torchcell-raw/<citation key>/data/si2.xlsx`.
779 + 682 = **1,461 released cell values**, the count the E. coli SI audit measured for
rank 14.

Every panel protein carries exactly ONE proteotypic peptide ("For each protein, one heavy
reference peptide was synthesized"), so the release's (protein, peptide, condition) key has
a peptide axis of length one and the record grain is the same (protein, condition) grain
the stored Table S6 block uses. Set 2's 31 accessions are a measured SUBSET of set 1's 41.

### Separability from the stored Table S6 block, asserted at build time

The audit's caveat is that this block re-measures proteins Table S6 already covers, so it
must not be mixed with the stored one. Three things keep them separable and the build
asserts all three (`preprocess/released_statistics_check.json`, key
`not_the_stored_block`):

1. **Separate datasets, separate stores, separate `Dataset` nodes.** Nothing joins a
   record of one to a record of another.
2. **`check_not_the_stored_block` refuses a non-distinct `measurement_type`.** It asserts
   that the stored block's type and the two arms' types are pairwise distinct, that this
   dataset's type is one of the two declared SRM types, and that it is not the stored
   block's.
3. **It also refuses any shared cell that AGREES with the stored block.** A cell whose SRM
   number equalled the stored label-free number is the only way the two blocks could be
   mixed without anyone noticing, so zero agreement to `IDENTITY_RTOL = 1e-6` is a build
   condition, not an observation. Measured:

| arm against Table S6 | shared (accession, condition) cells | agreeing to 1e-6 | Pearson r on log10 | median abs log2 ratio | median SRM / stored |
|---|---|---|---|---|---|
| Table S2 (set 1) | 738 | **0** | 0.6642 | 1.3518 | 0.4892 |
| Table S3 (set 2) | 682 | **0** | 0.9281 | 0.6164 | 1.0935 |

The two arms also disagree with EACH OTHER on every one of the 558 cells they share
(0 agreeing to 1e-6, median ratio 0.394), which is the record-level form of the
two-classes argument. All 41 accessions are filed under `Dataset 2` in Table S6's own
`Dataset` column, so the anchors have no dataset-1 label-free counterpart there at all.

### No duplication against Mori 2021, measured rather than assumed

Mori 2021 is the other absolute *E. coli* proteome in the store and it does not duplicate
Schmidt's Table S6 block (0 of 1,812 proteins agree to 1e-6, Pearson r 0.796, pinned in
`tests/torchcell/datasets/ecoli/test_mori2021.py`). The SRM panel overlaps Mori's proteins
completely, so it was measured too: all 41 set-1 and all 31 set-2 accessions are also
quantified in Mori's `A1-1` sample, and **0 of 41 and 0 of 31 agree to 1e-6** on the raw
numbers, at Pearson r 0.709 and 0.914 on log10 of a copies-like conversion. Mori's loaded
records are MG1655 (EQ353) in Neidhardt MOPS glucose and these are BW25113 in M9, so the
two sets share no (protein, strain, condition) triple either.

### Which table is which data set is VERIFIED, not read off the title

Table S6's coefficient-of-variance headers are proven swapped in this same release, so the
"dataset 1" / "dataset 2" in the two titles is held against Table S6's own coverage
pattern. Table S6 writes `NA` where a data set did not measure a cell, and measured over
its 301 dataset-1 rows, **every one is `NA` in exactly four conditions**: Glycerol + AA,
Xylose, Mannose and Fructose. Then:

- Table S2's 19 condition labels are exactly the remaining **18** Table S6 columns plus
  `anaerobic`.
- Table S3's 22 labels are exactly **all 22** Table S6 columns.
- The Methods independently put data set 1 at 19 samples: "Then, all 19 peptide mixtures
  were separated on a 12-cm pH 3-10 immobilized PH gradient strip".

`check_dataset_arm` asserts that pattern per arm plus a uniform row count per condition, so
a swapped pair of titles stops the build. **Verdict: the two titles are correct, a clean
negative.**

### The dispersion labels are checked the same way

Table S2 says its standard deviation comes from technical replicates and Table S3 from
biological triplicates, and a dispersion from two injections of one sample must be smaller
than one from three grown cultures. `check_dispersion_labels` reads BOTH sheets in either
build and asserts that ordering, because the claim is a comparison between them:

| measured | Table S2 (technical) | Table S3 (biological) |
|---|---|---|
| median relative spread | **1.7707 %** | **6.4891 %** |
| rows with SD at or above the abundance | 0 of 779 | 2 of 682 |
| rows with SD exactly 0 | 98 of 779 | 0 of 682 |

A 3.7-fold difference in the direction the labels predict. **Verdict: the two dispersion
labels are correct, a clean negative.** The abundance and standard-deviation columns are
not swapped either: the median SD/abundance ratio is 0.0177 and 0.0649, and the two Table
S3 rows at or above 1.0 are `ybhA` in stationary phase (13.72 +/- 16.49) and `glcB` in LB
(1487.68 +/- 1779.76), both low-abundance proteins across biological triplicates rather
than a column fault.

### A released condition the paper never describes

Table S2's 19th condition is labelled `anaerobic`, and the string "anaerob" appears
**nowhere** in the pinned `paper.md` or in the Supplementary `si1.md` -- no medium, no
cultivation protocol, no oxygen regime. It is in no other released table either: not Table
S6's 22 columns, not Table S23's experimental details, not Table S25's sample-name map.
There is nothing to build an `Environment` from, and `aerobicity="aerobic"`, which every
described condition of this paper carries, would be flatly wrong for it. It is dropped
under the new rule `condition_not_described_by_the_source` and ledgered rather than
ignored, because the Methods' own "all 19 peptide mixtures" makes it a real 19th
data-set-1 sample.

### A released standard deviation of exactly zero, kept verbatim

98 of Table S2's 779 rows carry a standard deviation of exactly 0, across 13 of its 41
proteins; four of them (`ytjC`, `acnA`, `pgk`, `aceA`) carry it in **all 19** conditions,
`pntB` in 10. Table S3 has none. The released value is stored as given, so the stored SE is
0.0: reading it as "not measured" would be a reinterpretation rather than a reading, and
the loader's no-fallback stance is to keep the bytes and make the finding visible. The
count is asserted at build time, so a corrected re-export stops the build and says so.
**Hypothesis (untested): those rows were quantified from a single usable SRM injection
rather than the stated duplicate.** Nothing in the release states that, and it is recorded
as a hypothesis, not as the reason.

### The retention ledger, with the arithmetic

The three structural condition rules are `schmidt2016.py`'s own `DropReason` objects,
reused rather than restated, so there is exactly one statement of each.

**Set 1: 19 released conditions = 11 records + 1 reference + 7 dropped.**

| rule | n | items |
|---|---|---|
| `condition_not_described_by_the_source` | 1 | `anaerobic` |
| `culture_not_batch` | 4 | chemostat µ=0.12, 0.20, 0.35, 0.5 |
| `growth_phase_not_representable` | 2 | stationary 1 day, stationary 3 days |

**Set 2: 22 released conditions = 14 records + 1 reference + 7 dropped.**

| rule | n | items |
|---|---|---|
| `medium_has_no_media_library_entry` | 1 | glycerol + AA |
| `culture_not_batch` | 4 | chemostat µ=0.12, 0.2, 0.35, 0.5 |
| `growth_phase_not_representable` | 2 | stationary 1 day, stationary 3 day |

Note the label spellings: Table S3 writes `chemostat µ=0.2` and `stationary 3 day` where
Table S2 and Table S6 write `chemostat µ=0.20` and `Stationary phase 3 days`, which is why
`CONDITION_LABELS` is a declared map rather than derived by string munging, and why both
spellings are in it.

### Identifiers

These tables release no b-number at all, so the stored key is the released `Gene` symbol
resolved against the pinned BW25113 GenBank annotation through `reconcile_locus_tags` --
the same derived route the stored block takes, and for the same reason (the records are
BW25113). Measured: **41 of 41** and **31 of 31** symbols resolve, 39 and 29 through the
gene-symbol layer and 2 and 2 through a gene synonym, with 0 collisions, 0 ambiguities and
0 outside the namespace. `MIN_RESOLVED_FRACTION` is 1.0, so a renamed symbol stops the
build.

**One identifier inconsistency inside the release, pinned rather than smoothed over.**
`P0ACP1` is `cra` in Table S1 and in its own UniProt description's `GN=` token but `fruR`
in Tables S2 and S3. They are synonyms of one gene and both resolve to the same locus.
`check_identifier_columns` asserts that this is the ONLY accession whose gene name
disagrees between the panel table and the measurement tables, and that every released
peptide sequence matches Table S1's selected proteotypic peptide exactly -- a mismatch
there would mean a measurement of a peptide no heavy standard was spiked for.

### Sourcing table

Every quote is a verbatim substring of the pinned bytes (`paper.md` sha256
`67bedae8f934421086c23b7fa9582b0950a07206bffe7c4ffdd11e4d003a8710`, `si2.xlsx` sha256
`3280a13ff67a73f25440cff6ee73fb99b5ce3ef57854213dbbf6272be241912f`).

| value | source | section | quote (abridged where long) |
|---|---|---|---|
| the assay and the 41-protein panel | paper.md | Online Methods, *Absolute quantification of selected proteins by targeted LC-MS* | "41 proteins covering key enzymes and iso-enzymes of carbohydrate metabolic pathways were selected for absolute quantification by SRM and SID (Supplementary Table 1)." |
| `n_replicates = 2` for set 1 | paper.md | same section | "Each sample was analyzed in duplicate." |
| `n_replicates = 3` for set 2 | paper.md | Online Methods, *Sample preparation* | "All samples of data set 2 were prepared in biological triplicates." |
| data set 1 is 19 samples | paper.md | Online Methods, *OFFGEL electrophoresis* | "Then, all 19 peptide mixtures were separated on a 12-cm pH 3-10 immobilized PH gradient strip" |
| the stored quantity | paper.md | Online Methods, *Absolute quantification ...* | "Based on the number of cells counted by fluorescenceactivated cell sorting for each sample, absolute abundances for the selected proteins (in copies/cell) could be calculated across all samples in both data sets (Supplementary Tables 2 and 3)." |
| what data set 1 was used for in the paper | paper.md | Online Methods, *Proteome-wide estimation of protein abundances* | "Data generated in data set 1 were included only in the qualitative analysis of identified protein modifications illustrated in Table 1 and Figure 5a,b ." |
| the panel table | si2.xlsx | Table S1 title | "Proteins selected for absolute quantification and their selected proteotypic peptides for which heavy reference peptides were synthesized and employed for quantification by stable isotope dilution" |
| the set-1 table | si2.xlsx | Table S2 title | "Absolute quantification of selected proteins for dataset 1 (see supplemental Figure 4 for dataset details)" |
| the set-2 table | si2.xlsx | Table S3 title | "Absolute quantification of selected proteins for dataset 2 (see supplemental Figure 4 for dataset details)" |

Table S1's released heavy-reference spike concentration (20 or 200 fmol/ug of total
protein, 21 and 20 proteins respectively) is a design parameter, not a phenotype; it is
kept in `preprocess/protein_identifiers.csv` beside each protein's peptide.

### Build

```
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset ProteomeSrmSet1Schmidt2016Dataset
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset ProteomeSrmSet2Schmidt2016Dataset
```

| metric | set 1 | set 2 |
|---|---|---|
| records | 11 | 14 |
| gene-set size | 41 | 31 |
| references | 1 | 1 |
| build time | 3 s | 3 s |
| `provenance.build_manifest` | `fresh` | `fresh` |

`compute_gene_set` is overridden on the shared base: every record is wild type, so the base
class's genotype scan would return the empty set it refuses, and the dataset's genes are
the loci it measures. That is the override `ProteomeSchmidt2016Dataset` already carries.

### Verification, L0 to L4

`verify_build(root, arm=...)` runs the shared `verify_protein_dataset` plus
`schmidt2016`'s own three supplementary rows and its host-aware L4 containment, reused
because these records have the same shape as that block's. Both arms: **PASS**.

```
proteome_srm_set1_schmidt2016: PASS
  [ok] L0 structural: 11 records validated
  [ok] L1 count: observed 11, expected 11
  [ok] L1 orf_uniqueness: 0 unique knocked-out ORFs, one record each
  [ok] L1 environment_uniqueness: SUPPLEMENTARY: 11 distinct environments over 11 records
  [ok] L2 value_fidelity: 451 values checked
  [ok] L2 se_nonnegative: 451 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 451 values
  [ok] L3 measurement_type_consistent: single measurement_type:
       'absolute_protein_copies_per_cell_srm_sid_dataset1_technical_sd'
  [ok] L3 assembly_pin: SUPPLEMENTARY: ('ecoli_K12_BW25113_ASM75055v1', 'GCA_000750555.1')
  [ok] L4 gene_containment_bw25113: 41 measured protein keys; 0 outside the locus universe

proteome_srm_set2_schmidt2016: PASS
  [ok] L0 structural: 14 records validated
  [ok] L1 count: observed 14, expected 14
  [ok] L1 orf_uniqueness: 0 unique knocked-out ORFs, one record each
  [ok] L1 environment_uniqueness: SUPPLEMENTARY: 14 distinct environments over 14 records
  [ok] L2 value_fidelity: 434 values checked
  [ok] L2 se_nonnegative: 434 values checked
  [ok] L3 reference_finite: reference abundance finite + key-matched for all 434 values
  [ok] L3 measurement_type_consistent: single measurement_type:
       'absolute_protein_copies_per_cell_srm_sid_dataset2_biological_sd'
  [ok] L3 assembly_pin: SUPPLEMENTARY: ('ecoli_K12_BW25113_ASM75055v1', 'GCA_000750555.1')
  [ok] L4 gene_containment_bw25113: 31 measured protein keys; 0 outside the locus universe
```

### BioCypher adapters and their enable-lists

`ProteomeSrmSet1Schmidt2016Adapter` and `ProteomeSrmSet2Schmidt2016Adapter`
(`torchcell/adapters/schmidt2016_srm_set1_adapter.py` and `..._set2_adapter.py`) serve the
two arms, with enable-lists in
`torchcell/adapters/conf/proteome_srm_set{1,2}_schmidt2016_adapter.yaml`. Each module holds
exactly one adapter class, so `kg_manifest._CONF_RE` reads its own conf rather than a
sibling's (issue #743). Both are registered in `dataset_adapter_map`, re-exported from
`torchcell/adapters/__init__.py` and named in
`torchcell/knowledge_graphs/conf/kg_bacteria.yaml`.

Neither serves a gene perturbation: every record is wild-type BW25113 with an empty
genotype, the paper's three deletion strains carry no SRM abundance, and their growth rates
are served by `GrowthRateSchmidt2016Dataset` instead. `genotype (chunked)` IS served (one
empty genotype per record, because the genotype is what the experiment points at), and the
environment-perturbation pair is on, since every record but the LB one and the glucose
reference carry a carbon-source or stress edit. The paired test files pin that in both
directions, on the conf (hermetic) and on the emitted graph (data-gated), and also pin that
each arm's conf is NOT the stored block's conf.

---
id: 55160mu9r2ontn9xzxty4b7
title: Smith2016
desc: ''
updated: 1789274048457
created: 1789274048457
---

## 2026.09.12 - Serve-50 build: the vendor-code drop, SC-Ura, 20 doublings

`CrispriChemgenSmith2016Dataset` -- per-guide dCas9-Mxi1 CRISPRi chemical-genetic fitness,
one record per (pool x guide x drug-condition) row of Additional file 10. This is the note's
first entry; the loader existed since 2026-07-12 but had no note and no test file.

The 2026-07-12 build failed L0 structural on all 14,463 records (`Media` gained a required
`is_synthetic` after they were serialized), so a rebuild was mandatory and everything below
rides on it. Nothing was served, so nothing is updated in place.

### Records dropped: 7,410 of 14,463 (51.2%)

Ten of the nineteen drug labels are vendor catalog ids whose structure the primary never
released: `1181-0519`, `4130-1276` and `0KPI-0099` (ChemDiv), `9121982`, `9125678`,
`7312221`, `CBF-666774`, `6630449` and `9150499` (ChemBridge), `ST016598` (TimTec).
Additional file 8 gives the vendor and the dose but no SMILES, CAS or structure for any of
them, so `resolve_compound_identity` returns `PROPRIETARY` with no InChIKey, ChEBI id or
PubChem CID. Per the Hoepfner precedent those records are dropped rather than served under a
bare name, and the rule plus the per-compound counts are written to
`preprocess/dropped_records.json`. **Kept: 7,053.**

The other three retention rules fire zero times today and are kept as live gates: rows with
no `A`/`var(A)` (0 -- Additional file 10 is complete), unresolved target ORFs (0 -- all 20
are current R64) and guides with no released spacer (0 -- all 977 join Additional file 4).

`NSC-180973` is the one label that could have been lost and was not. Additional file 8
releases the alias `NSC-180973 (tamoxifen)`, and the pinned compound identity table carries
that synonym onto the tamoxifen row (`NKANXQFJJICGDU-QPLCGJKRSA-N`, CID 2733526), so its 456
records are kept. The old loader passed `known_proprietary=True` for every non-DMSO drug,
which would have mislabelled eight ordinary public compounds' gaps as terminal; the flag is
gone, because the table's own per-row `resolution_status` is authoritative.

### Encoding changes

- **Medium**: the shared `media.SC_URA` object replaces `Media(name="SCM-Ura")`. Smith
  releases no SCM-Ura recipe, so the library object's own component gaps are the honest
  state, and using it joins every other SC-Ura dataset.
- **`assay_type = pooled_competitive_growth_barcode`**, sourced from the Methods sentence
  that makes the spacer a barcode ("the short specificity-determining regions of gRNAs ...
  can act as unique identifiers of individual strains").
- **`duration_generations = 20.0`** ("This amounted to approximately 20 culture doublings
  from the beginning of the experiment"). Exposure duration is part of environment identity
  and the old build left it None. `duration_hours` is a typed `ProvenanceGap`: the cultures
  were held in log phase by repeated dilution and no wall time is reported.
- **Common names** now come from the genome's own standard name for the resolved ORF, via
  the shared `resolve_gene_name`.
- The record grain and the `library_pool` discriminator are unchanged: 272 (guide, drug,
  concentration) triples appear twice as the pool pair (`broad_tiling`,
  `gene_tiling_20bp`), and those are two independent pooled measurements, not replicates.

### `n_samples = 1` is a decision, not an oversight

The review flagged it as contradicting the released replicate structure (eight replicate
experiments for 1% DMSO, three for 20 uM fluconazole). It is kept at 1 deliberately. The
stored `var(A)` is the variance of the SINGLE released estimate: for those two conditions
the paper has ALREADY inverse-variance-combined the replicates into it ("We combined the R
replicates ... to obtain the variance"). `derive_se` computes `sqrt(var/n)`, so recording 8
would shrink the stored SE by `sqrt(8)` against a variance that is already the combined
estimate's. The replicate structure is recorded instead in the module-level
`REPLICATE_STRUCTURE` SourcedValue, where it cannot corrupt arithmetic.

### Two things deliberately NOT asserted

- **The ATc dose.** ATc induction is what turns the CRISPRi genotype on, but the pooled
  screen's Methods say only "grown +/- a drug listed in Additional file 8, and +/- ATc". The
  one released number, 250 ng/mL, is from the qPCR section and Additional file 5's
  sample-metadata header, neither of which is this screen. The induced-vs-uninduced
  definition rides in the phenotype `units` string and in the uninduced (-ATc) reference.
  `Environment` has no inducer field, so there is nothing to attach a `ProvenanceGap` to;
  this is flagged here rather than typed.
- **The solvent.** "Drugs were dissolved in DMSO" sits in the individual-strain growth-assay
  section and the pooled section does not restate it, so no `Solvent` is asserted on the
  pooled drugs. The 1% DMSO vehicle control is served as a compound in its own right.

The ~20%-growth-inhibition dose rule IS released (Additional file 8 ReadMe, sha256
`a22ef00ace51...`: "These concentrations were selected to inhibit growth of a wild-type
strain by ~20%"), but `DoseBasis` has no `IC20` member and adding one changes a class in all
36 served dataset closures, so the rule is recorded here and in the loader rather than in
`Concentration.basis`.

### Provenance

Raw mirror `$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/` holds exactly
the two files the loader consumes, `si/si_data/13059_2016_900_MOESM10_ESM.xlsx` (Additional
file 10, sha256 `02962e51...`) and `si/si_data/13059_2016_900_MOESM4_ESM.xlsx` (Additional
file 4, sha256 `e5eb4e3c...`), each with a `springer_esm` retrieval record. Both URLs were
re-retrieved on 2026-09-12 and reproduced the pinned hashes exactly, so the retrieval chain
is live rather than aspirational.

### Verifier result (verbatim, scratch runner with this entry's parameters)

```
crispri_chemgen_smith2016: PASS
  [ok] L0 structural: 7053 records validated
  [ok] L1 count: observed 7053, expected 7053
  [ok] L1 pair_uniqueness: 7053 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 7053 documented provenance gaps over 7053/7053 records; 0 deferred field(s)
  [ok] L1 canonical_gene_names: 20 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 7053 values checked
  [ok] L2 se_nonnegative: 7053 values checked
  [ok] L2 uncertainty_sanity: 7053 labelled uncertainties, none a zero dispersion
  [ok] L3 measurement_type_consistent: single measurement_type: MeasurementType.log2_ratio
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 7053 records
  [ok] L3 environment_perturbed: all 7053 experiments carry an environmental edit
  [ok] L3 compound_identity: 7053 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 218643 compound references carry a structure identifier
  [ok] L3 media_membership: 7053 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 20 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 20 measured systematic names is current
```

### Open flags

- The class was NOT in `torchcell/datasets/scerevisiae/__init__.py` when this group was
  reviewed, which made `kg_manifest admit --dataset CrispriChemgenSmith2016Dataset` raise
  `KeyError` at `kg_manifest.py:489`. Registration is the orchestrator's file, not this
  agent's; it has to land before admission can run.
- Resolving the 10 vendor catalog codes through a pinned PubChem SID lookup would recover
  7,410 records. ChemDiv / ChemBridge / TimTec ids ARE carried by PubChem as SUBSTANCE
  records, so it is a real retrieval rather than a guess, and it is the single highest-yield
  compound-table item in this group.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: partial.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `EFFECT_SHA256`, `GUIDE_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.09.30 - Module constant is the one pin at download (issue #561)

Before this change, `download()` verified the mirror bytes against the digest recorded in the raw-mirror `manifest.json`, and `process()` verified them against the module constant. Because `deposit_raw_mirror` writes the manifest from the constant, the two were equal by construction, but the loader still carried two pins. Now `download()` verifies the bytes against the module constant, and `check_manifest_pin` refuses a manifest that records any other digest, raising `ManifestPinMismatchError` named by path with both digests. The manifest stays the retrieval record. Built records are unchanged. Tests: `test_download_refuses_a_manifest_digest_off_the_module_pin` in [[tests.torchcell.datasets.scerevisiae.test_raw_pins]].

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: 7,410 of 14,463 rows were dropped because the Drug label is a ChemDiv / ChemBridge / TimTec catalog id with no released structure (`resolve_compound_identity` PROPRIETARY); the target ORF was resolved through the genome (0 dropped) and the perturbed name was the genome's standard name.

Now: every row is a record. The vendor-code drugs are stored under the released label with the typed `inchikey` gap `resolved_compound` attaches (`not_reported_by_primary`), listed under `compound_identity_gaps` in the ledger with their record counts (the Parsons 2006 route of #1053). The target is the released `ORF` cell and `perturbed_gene_name` the released `Gene` cell (the ORF where blank); no genome is injected. 14,463 records (was 7,053). An `ORF` cell with no ORF form is counted under `deferred_to_released_gene_name_leaf` (0 of 14,463).

## 2026.10.12 - The deferred name is a record on the released-gene-name leaf (#1045, PR #1085)

A target whose ORF cell is not ORF-shaped goes on `ReleasedGeneNamePerturbation` with kind `crispr_interference` and its construct (0 of the 20 released targets); the ledger rule `deferred_to_released_gene_name_leaf` is renamed `stored_on_released_gene_name_leaf` with `n_records` 0. The count stays 14,463.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**crispri_chemgen_smith2016**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/crispri_chemgen_smith2016/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 14463 records validated |
| L1 | `count` | gate | ok | observed 14463, expected 14463 |
| L1 | `pair_uniqueness` | gate | ok | 14463 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | gate | ok | 14463 values checked |
| L2 | `se_nonnegative` | gate | ok | 14463 values checked |
| L2 | `interval_orientation` | gate | ok | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | gate | ok | numeric rule: reference response == 0 for all 14463 records |
| L3 | `environment_perturbed` | gate | ok | all 14463 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='SC-Ura (synthetic complete minus uracil; URA3 plasmid selection)') |
| L1 | `provenance_gaps` | gate | ok | 21873 documented provenance gaps over 14463/14463 records; 0 deferred field(s): []; 1064460 undeclared None values over 17 carrier fields (top: Compound.inchi x477279, Compound.chebi_id x384342, Environment.dilution_rate_per_hour x14463, EnvironmentResponsePhenotype.category x14463, EnvironmentResponsePhenotype.category_label x14463) |
| L1 | `canonical_gene_names` | report | ok | 20 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | gate | ok | 14463 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | gate | ok | environment edits: 7053 compound references carry a structure identifier; 7410 declare a typed gap (10 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | gate | ok | medium components: 448353 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | gate | ok | 14463 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | report | ok | 1.000 of 20 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | report | ok | every one of the 20 measured systematic names is a gene of the current genome |

## 2026.11.24 - One record per released replicate (#1178)

The store held one record per Additional file 10 row and called the per-replicate tables
unread. Additional files 6 and 7 release every replicate culture of the two
multi-replicate conditions, so each is now its own record, and nothing is averaged.

| | before | after |
|---|---|---|
| records | 7,053 | 17,707 |
| A records (Additional file 10) | 7,053, `replicate_id` absent | 7,053, same values, `replicate_id = "1"` |
| culture records | none | 10,654 (7,216 from Additional file 6, 3,438 from Additional file 7) |
| files read | Additional files 10, 4 | Additional files 10, 4, 5, 6, 7 |
| ATc | not asserted anywhere | 0.25 ug/mL on an induced culture's environment |

### What a replicate is

"The guide counts following competitive growth were highly reproducible between
biological replicates, indicating that the assay is robust (Additional file 2: Figure
S3A-B, Additional files 6 and 7)." (`paper.md` line 35, sha256 `346be696...`). The
replicate tables the sentence cites are the two files this loader now reads, so a
replicate there is an independent pooled culture pair. The counts are Methods line 150:
"For the control condition (1 % DMSO), we had eight replicate experiments for the tiling
pools, and three replicate experiments for 20 uM fluconazole."

Measured on the pinned files (2026.11.24):

- Additional file 6 `DMSO_replicate_data` (sha256 `7f70c60d...`): 476 guide rows, 451 with
  data, 8 replicates x 2 arms = 7,216 released measurements, every row complete in all 8.
- Additional file 7 `Fluconazole_replicate_data` (sha256 `3d26fb3e...`): 676 guide rows,
  637 with data; 637 carry replicates 1 and 2 and 445 also carry replicate 3, so
  (637 + 637 + 445) x 2 = 3,438.
- The count, the fitness and its variance of one replicate arm are present or absent
  together on every row of both files, so `process()` refuses a partial triple.

### Additional file 10's row is replicate 1, not the combination

The previous docstring and verifier entry said the 1% DMSO and 20 uM fluconazole rows of
Additional file 10 were the paper's inverse-variance combination of the eight and three
replicates. They are replicate 1 alone:

- Additional file 10's `-ATC_count` and `+ATC_count` equal Additional files 6 / 7's
  `Rep-1` counts on 2,148 of 2,148 comparable cells (440 of 440 DMSO rows, 634 of 646
  fluconazole rows; the other 12 release no Rep-1 cells). Replicates 2 to 8 match on
  0.0 to 3.0 percent of rows.
- Its `A` equals its own `+ATC_f - -ATC_f` to 0.001, and differs from `Combined_A_mean`
  by a median 0.245 (max 3.521).
- Additional file 9's `Count_perfect` for replicate 1 equals Additional file 10's counts
  in all 67 (pool, drug, concentration) groups, and Additional file 5 lists replicate 1
  for every one of those groups.

So `replicate_id = "1"` on an A record, and the released `var(A)` is one replicate's
read-resampling variance, which is why `n_samples = 1` and `derive_se` dividing by 1 was
right all along (for a different reason than the docstring gave). `process()` re-proves
both checks on every build and refuses otherwise.

### The authors' combined statistics are a derivation, so they are not stored

Measured over the 451 + 637 guides with data: `Combined_A_mean` is the PLAIN MEAN of the
per-replicate `A_k = f(+ATc) - f(-ATc)` (max absolute difference 0.000875 for Additional
file 6 and 0.001 for Additional file 7, the rounding of the released three-decimal cells),
and `Combined_A_var` is their sample variance with ddof 1 (max absolute difference 0.0015
and 0.0035). It is NOT the inverse-variance combination the Methods describe: that
formula gives a median absolute difference of 0.027 on Additional file 6 and matches only
20 percent of rows to 0.01. The per-arm `Combined_*ATC_mean` ARE the inverse-variance
combination of the per-replicate `f` (max 0.0008, every row). Either way both are
functions of the records now stored, so they stay out of the store and the mean is a
modeling-time question.

### What a culture record holds

A culture record is one culture, so it stores what the source releases for one culture:
its guide fitness `f` ("we calculated the log2-scale median centered counts l_ijk =
log2(c_ijk + 0.5) - median_i' in pool j (log2(c_i'jk + 0.5))", `paper.md` line 146) with
that cell's released `f_var`, `UncertaintyType.variance`. `A_k` for a replicate is the
difference of its two records and is left to model time, under the policy's rule that a
loader stores the per-culture value and not a loader-computed contrast.

Its target comes from Additional file 4, the only file that gives a per-replicate row's
ORF (the per-replicate sheets carry a pool and a guide name only). A guide name repeated
there (63 of them, a broad-tiling guide that is also in a gene-tiling set) agrees on ORF,
gene and spacer in every case, and agrees with Additional file 10 on every row where both
name the guide's target; `process()` refuses a disagreement.

`units` is `CULTURE_UNITS`, which is what keeps a culture record and an A record of the
same strain and condition L1-distinct (the environment-response verifier's study key
reads `units` and `replicate_id`). The reference is f = 0, the pool's median guide, by the
median centring itself.

### ATc enters as a dose, on a culture record only

An A record is the contrast between the induced and uninduced cultures, so it carries no
ATc and its reference is the uninduced culture at A = 0. A culture record is one culture,
so its environment states whether ATc was in it: `SmallMoleculePerturbation` with
`resolved_compound("anhydrotetracycline")` at 0.25 ug/mL, parsed at build time from
Additional file 5's own column header `ATC (250ng/ml)` (README: '"yes" if ATC was
included, or "no" if it was not'). The 2026.09.12 note said the only released ATc number
was from the qPCR section and not the pooled screen; Additional file 5 IS the pooled
screen's sample sheet (its 100 rows are the broad_tiling, gene_tiling18 & gene_tiling20
and mutant18 & mutant20 samples), so the dose is sourced for these cultures.

### Released replicates that reach no record

`dropped_records.json` gained `culture_replicates` (what each per-replicate file
produced, per replicate label) and `replicates_not_stored`, and the build refuses if a
replicate Additional file 5 releases reaches no record without a recorded reason. Two
remain, both named:

- `mutant18 & mutant20` 1% DMSO replicate 2: released only as read counts (Additional
  file 9). No `MeasurementType` holds a read count, and computing `f` from the counts
  would be a loader derivation. Additional file 6 covers the gene-tiling pools only.
- `gene_tiling18 & gene_tiling20` Rapamycin 10 nM replicate 1: Additional file 5 lists the
  sequenced sample pair and no released file carries a count, a fitness or an effect for
  it (Additional file 9 has no 10 nM rows; Additional file 10 releases 2.2 nM and
  4.375 nM).

The 1181-0519 25 uM replicate 2 of the tiling and mutant pools is NOT in that list: that
compound is a vendor catalog code held out for every replicate under
`UNIDENTIFIED_COMPOUND_RULE`, which already accounts for it.

### Raw mirror

Additional files 5, 6 and 7 were retrieved from the Springer ESM CDN on 2026-11-25 UTC
with `torchcell.literature.retrieve.springer_esm` and deposited in
`$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/si/si_data/`, each
recorded in the mirror `manifest.json` with its `source_url`, retrieval command,
`sha256` and `retrieved_at`. Additional file 5's bytes are identical to the literature
mirror's copy (sha256 `9477066f...`), which is the pin; Additional files 6 and 7 were not
in either mirror before.

| file | sha256 |
|---|---|
| Additional file 5 `13059_2016_900_MOESM5_ESM.xlsx` | `9477066f538248c18b256ed2fec811bbd5914b2039bd2f78baa41000e70eaf9c` |
| Additional file 6 `13059_2016_900_MOESM6_ESM.xlsx` | `7f70c60de3702e49caa5142b73b32b2de64ad1cbfd4db47e3a58a76b616ae676` |
| Additional file 7 `13059_2016_900_MOESM7_ESM.xlsx` | `3d26fb3e6e613e21e427abeaa7a8cc6a9053b34d4770ecc185db7900053a592b` |

### L0-L4 on a throwaway build of the per-replicate loader

Built into `$DATA_ROOT/tmp/scratch/per-replicate-smith2016/crispri_chemgen_smith2016`
(not the shared dev store) and verified with `verify_environment_response_entry` under the
`crispri_chemgen_smith2016` runner configuration, whose `expected_count` is now 17,707.
Verdict PASS.

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 17707 records validated |
| L1 | `count` | gate | ok | observed 17707, expected 17707 |
| L1 | `pair_uniqueness` | gate | ok | 17707 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | gate | ok | 17707 values checked |
| L2 | `se_nonnegative` | gate | ok | 17707 values checked |
| L2 | `uncertainty_sanity` | gate | ok | 17707 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | gate | ok | numeric rule: reference response == 0 for all 17707 records |
| L3 | `environment_perturbed` | gate | ok | all 17707 experiments carry an environmental edit |
| L3 | `compound_identity` | gate | ok | environment edits: 23034 compound references carry a structure identifier; 0 declare a typed gap |
| L3 | `media_membership` | gate | ok | 17707 records on a shared MEDIA_LIBRARY medium (1 distinct media) |
| L4 | `gene_containment_sgd` | report | ok | 1.000 of 20 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | report | ok | every one of the 20 measured systematic names is a gene of the current genome |

The 7,053 A records of the new build were compared record by record against the previous
dev store: the same 7,053 (strain, condition) keys, max absolute difference 0.0 on
`environment_response`, identical references, and `replicate_id` the only field that
changed on any of them.

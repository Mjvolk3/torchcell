---
id: zyo4mkoh8sv4wyosjmn2f1h
title: Goodall2018
desc: ''
updated: 1791389068340
created: 1791389068340
---

## 2026.10.07 - Loader, sourcing decisions and the first build

`torchcell/datasets/ecoli/goodall2018.py`, class `GeneEssentialityGoodall2018Dataset`
(`REFERENCE_STRAIN = "BW25113"`, takes `ecoli_genome`), rank 11 of
[[plan.bacteria-ontology-genome]] section 6. Templates: [[torchcell.datasets.ecoli.fuhrer2017]]
(BW25113, raw mirror, verifier shape) and [[torchcell.datasets.ecoli.lamoureux2023]]
(a loader-built medium deriving from a library key). Shared API:
[[torchcell.datasets.bacteria_common]]; schema: [[torchcell.datamodels.bacterial-perturbation-ontology]];
media: [[torchcell.datamodels.media]]. Tests: `tests/torchcell/datasets/ecoli/test_goodall2018.py`.
Every number below is read from the build's ledgers in
`$DATA_ROOT/data/torchcell/gene_essentiality_goodall2018/preprocess/`
(`dropped_records.json`, `identifier_reconciliation.json`, `call_rule.json`, `calls.csv`,
`perturbation_field_gaps.json`, `verification_report.json`).

### Where the data is

Goodall et al. 2018, mBio 9:e02096-17, doi 10.1128/mBio.02096-17, PMC5821084, PubMed
29463657 (PMC id converter, 2026-10-07), citation key `goodallEssentialGenomeEscherichia2018`,
`paper.md` sha256 `facfe1fac8ab6dab5cd0ecb88876b33616cc00de7c1fe68b60590f7a1883b429`.

The per-gene calls are two publisher SI workbooks, captured from the PMC Article Datasets
bucket by the literature mirror. The raw mirror
`$DATA_ROOT/torchcell-raw/goodallEssentialGenomeEscherichia2018/` holds exactly those two
files (`deposit_raw_mirror`, idempotent by sha256, refuses a differing file; deposited
twice, second run a no-op):

| file | what | sha256 | retrieval |
|---|---|---|---|
| `si2.xlsx` | Table S1, "Essentiality classification for genes of the TL data" | `db85743f...` | `pmc_cloud`, key `PMC5821084.1/mbo001183726st1.xlsx` |
| `si7.xlsx` | Table S4, "Essentiality classification for genes following outgrowth in LB" | `8a4c6eb8...` | `pmc_cloud`, key `PMC5821084.1/mbo001183726st4.xlsx` |

The recorded retrieval was re-run on 2026-10-07 (`pmc_cloud_object`) and reproduced both
pinned digests. Columns of both: `Gene`, `Insertion Index Score`, `Log Likelihood Ratio`,
`Essential`, `Non-essential`, `Unclear` (one-hot), 4,313 data rows each. Not mirrored,
because nothing reads them: the TraDIS reads (ENA PRJEB24436) and Tables S2/S3.

### Strain and identifiers

- Strain: `E. coli K-12 strain BW25113, the parent strain of the Keio library, was used for construction of a transposon library.`
  Records pin `assembly_reference("BW25113")` = `ecoli_K12_BW25113_ASM75055v1`, `GCA_000750555.1`.
- Mapping reference: `Trimmed, filtered sequences were then aligned to the reference genome E. coli BW25113 (accession no. CP009273.1), obtained from the NCBI genome repository (69).`
  CP009273.1 is the set's replicon; `check_replicon` refuses a genome whose loci sit anywhere else.
- Names: `Where gene names differed between databases, the BW25113 annotation was used.`
  So `Gene` is a BW25113 GenBank symbol and resolves at the gene-symbol layer. No MG1655
  b-number, ECK or JW route is used; every record's identity comes from the BW25113
  symbol route.

`reconcile_locus_tags` over the 4,269 distinct names of both tables (the two tables list
the same genes):

| status | names | layer |
|---|---|---|
| renamed (gene, symbol to locus tag) | 4,082 | gene symbol |
| non_gene_feature (pseudogene) | 178 | gene symbol |
| ambiguous, kept as given | 9 | gene symbol |
| current, retired, collision, case-insensitive | 0 | |

Resolved 4,260 / 4,269 = 0.9979, above the stop threshold `MIN_RESOLVED_FRACTION = 0.99`
(the paper says the names ARE BW25113 symbols, so a lower fraction means the wrong
annotation). The 9 ambiguous names are the multi-copy insertion-sequence symbols
(`insA` 6 loci, `insB1` 5, `insC1` 6, `insD1` 6, `insE1` 5, `insF1` 5, `insH1` 10,
`insI1` 3, `insL1` 3), 53 rows per table. Each row is one copy, and the table runs in
genome order, so a positional assignment between resolved neighbors is possible; it is
not done, because it is a derived mapping the transposon leaf has no field for, and reads
that map to identical copies make each copy's index unreliable anyway (TL rows 21 and
3376, both `insA`, carry identical values).

### Conditions and environments

| condition | table | environment |
|---|---|---|
| `TL` | S1 | loader-built `selection_medium()`: `LB` library components + agar + chloramphenicol, `base_medium="LB"`, solid; temperature and duration gapped |
| `LB` | S4 | library `LB` (LB Miller, which already cites this paper's amounts), 37 C, aerobic; `duration_generations` gapped |

- TL: `Transposon mutants were selected by growth on LB agar supplemented with chloramphenicol.`
  and `... grown overnight on selective medium.` No agar amount, chloramphenicol dose or
  incubation temperature is stated, so both components carry `concentration=None`.
  `LB_AGAR` is not reused: its 2% (w/v) is Menasalvas 2025's and Schmidt 2016's value,
  and asserting it here would fabricate a number (the Bloom/Hillenmeyer rule of the media
  note). `aerobicity` keeps the field default because the field cannot be None.
  Chloramphenicol has no row in the pinned compound table (only `l-erythro-chloramphenicol`,
  a different stereoisomer, CID 146033), so it resolves with a typed
  `deferred_pending_source_review` gap on `inchikey`.
- LB: `two independent samples of the transposon library were grown in Luria broth (LB) at $3 7 ^ { \circ } \mathsf { C }$ for 5 or 6 generations to an optical density at $6 0 0 ~ \mathsf { n m }$ $( \mathrm { O D } _ { 6 0 0 } )$ of 1.0 and were then sequenced.`
  `5 or 6` has no per-culture value and the Methods sentence lost its number in the OCR
  (`grown for generations`), so `duration_generations` is a typed gap, not either end.
  These cells passed the TL selection first; `Environment` cannot say so, and the
  experiment pair declares `environment: Environment`, so a `CultureEnvironment.pre_culture`
  would serialize away.

### Genotype: a gene-level transposon disruption

One `TransposonInsertionPerturbation` per record: `systematic_gene_name` = the BW25113
locus tag, `perturbed_gene_name` = that locus's own GenBank symbol, `gene_namespace =
ecoli_k12_bw25113_locus_tag`, `transposon = "mini-Tn5"` (`The main differences were that a mini-Tn 5 transposon coding for a chloramphenicol resistance cassette was used.`;
the OCR writes `Tn 5`). The record stands for every insertion mutant of that gene in the
pool. The coordinate fields:

| field | value | why |
|---|---|---|
| `barcode` | None, not applicable | TraDIS has no per-strain barcode; the only barcode is a per-sample index: `Raw data were checked for the presence of an inline index barcode to identify independently processed samples (Table 1).` |
| `insertion_position`, `insertion_strand` | None, typed absence | a gene-level record has no single site, and the tables release none; per-insertion sites would come from the reads |
| `library_pool` | None | one library |

The leaf has no `provenance_gaps` slot (it does not take `HashableProvenanceGapMixin`), so
the two coordinate absences are `ProvenanceGap(reason=not_reported_by_primary)` objects in
`PERTURBATION_FIELD_GAPS`, written to `perturbation_field_gaps.json`; a test pins that
they name real leaf fields.

### Phenotype, n and the uncertainty type

`BacterialGeneEssentialityExperiment` / `...Reference` (the assembly-pinned pair),
`GeneEssentialityPhenotype(is_essential=...)`: True for `Essential`, False for
`Non-essential`. What True means here: `which have a low number of transposon insertions, are either essential for survival or genes that, when disrupted, confer a very severe fitness cost (Fig. 1D).`
The reference is the unperturbed BW25113 parent in the same environment,
`is_essential=False` (the library was built in it and plated), one per condition.

**The call rule, measured.** Methods: `Genes with log likelihood scores between the upper and lower $\mathsf { l o g } _ { 2 } 1 2$ threshold values of 3.6 and $- 3 . 6 ,$ , respectively, were deemed “unclear.”`
Every released call of both tables (4,313 / 4,313 each) equals ratio < -3.6 essential,
> 3.6 non-essential, else unclear (`call_rule.json`; `process()` refuses any
disagreement). The exact log2(12) = 3.585 would move one unclear gene per table to
non-essential: `ydhW` (S1, 3.5852) and `grxD` (S4, 3.5881). So the rule applied is the
printed 3.6.

**Counts.** Table S1: 358 essential, 3,793 non-essential, 162 unclear, equal to the
paper's `Using this approach, sufficient insertions were found in 3,793 genes for them to be classed as nonessential, 162 genes were situated between the two modes and classed as unclear, and 358 genes in the mutant library were identified as essential (Table S1).`
(required at build time). Table S4: 356 / 3,802 / 155 (not stated in the paper; measured).

**n_samples, sourced but not storable.** Two sequenced samples per condition, pooled
before calling, so one call per gene and no per-replicate spread:

- TL: `DNA was extracted from two samples of the transposon library glycerol stock to generate TraDIS data referred to as TL1 and TL2 in the text.`
  then `The data were therefore combined to give a total of 8,279,309 sequences that were mapped to 901,383 unique insertion sites throughout the genome.`
- LB: `In addition, DNA was extracted from two independent cultures, LB1 and LB2, of the library grown in Luria broth (LB)`
  then `As there was a high correlation coefficient of 0.97 between the gene insertion index scores of each technical replicate (Fig. 1C), the data were combined to give a pool of 10,584,188 sequences.`
- Kind, a recorded conflict: the Methods call LB1/LB2 independent cultures; the Fig. 1
  legend calls both pairs `two sequenced technical replicates`. Not resolved.

`GeneEssentialityPhenotype` has no `n_samples`, `sample_unit` or uncertainty field, so n = 2
lives in `SOURCED_VALUES` and here. There is no uncertainty to type: the released values
per gene are the insertion index (a ratio of counts on pooled reads), the mixture log
likelihood ratio (the evidence for the call, not a spread) and the call. Combed for
`replicate`, `independent`, `triplicate` (the beta-galactosidase assay only), `standard
deviation`, `n =`, `bootstrap`, `error`, `confidence` over `paper.md` and every SI
OCR file; the only standard deviation is the beta-galactosidase assay's, and no per-gene
uncertainty is reported. Text S1 (`si4.docx`,
read, no OCR in the mirror) is the insertion-free-region statistics and adds none.

### Records and drops

Rules in order, first match wins:

| rule | TL | LB | total |
|---|---|---|---|
| `symbol_not_on_one_bw25113_locus` (the 9 multi-copy IS symbols) | 53 | 53 | 106 |
| `unclear_call_not_representable` | 162 | 155 | 317 |
| kept | **4,098** (358 E, 3,740 N) | **4,105** (356 E, 3,749 N) | **8,203** |

8,626 table rows, 8,203 records, 4,172 distinct loci (`gene_set`). Of the 4,031 loci kept
in both conditions, 332 are essential in both and 3,694 non-essential in both; 3 flip
non-essential to essential after LB (`degS`, the paper's conditional example, plus `ptsH`
and `tusC`) and 2 flip essential to non-essential (`ycaR`, `yqeL`). 178 pseudogene names
are kept as records on their pseudogene loci (`ttcC` BW25113_4638 is essential in both).
`calls.csv` keeps every row's insertion index, ratio and call, with its locus, rule and
LMDB index, so nothing released is lost by the drops.

The stored call is the paper's STATISTICAL call. The text's manual reassessments (for
example `ftsK`, essential by its N-terminal insertion-free region but non-essential by
index) are not in the tables and are not applied.

### Build and verification

`PYTHONPATH=<wt> python -m torchcell.database.build_dataset_lmdb --dataset GeneEssentialityGoodall2018Dataset`:
8,203 records, 2 references, gene set 4,172, 5 s (12 s wall with imports), peak RSS 1.3 GB,
LMDB 15 MB. `torchcell.provenance.build_manifest.check_all` reads
`gene_essentiality_goodall2018` as fresh.

`python -m torchcell.datasets.ecoli.goodall2018 verify` (`run_verification`): PASS.

- L0: 8,203 experiments and references validate as the bacterial essentiality pair.
- L1: count 8,203 = the drop log's kept; one record per (condition, locus), each a single
  insertion; gap census 20,497 documented gaps = 4 per TL record (temperature, duration,
  and the agar and chloramphenicol `inchikey`) + 1 per LB record (generations); spelling
  one per locus.
- L2: every record's call and locus equal its Table S1/S4 row, re-read from `raw/` and
  re-resolved on BW25113 (independent of the build path).
- L3: the +/-3.6 rule holds on 4,313 / 4,313 rows of each table; Table S1 counts equal the
  paper's; media: 4,105 records on library `LB`, 4,098 on a medium deriving from it;
  25 / 25 sourced values verbatim at their pinned sha256.
- L4: 4,172 / 4,172 disrupted loci are BW25113 GenBank gene rows.

The shared rules run without a resolver: their canonical-name rule requires `CURRENT`, and
a pseudogene tag resolves `NON_GENE_FEATURE`; the L2 check accepts both by name.

### Deduplication (checklist item 7)

No superset in the fifty covers this TraDIS library. Keio-derived and RB-TnSeq BW25113
rows are different libraries and assays, so nothing is subsumed either way.

### Open, and what is a gap

- **Schema, the main gap.** `GeneEssentialityPhenotype` stores only `is_essential`, so the
  317 unclear calls, the insertion index, the likelihood ratio and n = 2 have no field. The
  additive change that would carry them (no served closure moves, since
  `BacterialGeneEssentialityExperiment` serves no built dataset yet) is in the PR body.
- **Leaf gap slot.** `TransposonInsertionPerturbation` cannot carry `provenance_gaps`; the
  coordinate absences live in `PERTURBATION_FIELD_GAPS` until it takes
  `HashableProvenanceGapMixin`.
- The 106 multi-copy IS rows: dropped; recoverable by positional assignment if a derived
  mapping field exists.
- TL: temperature, duration, agar amount and chloramphenicol dose unstated; chloramphenicol
  needs a compound-table row (one curator row; outside this branch).
- LB: generations `5 or 6` gapped; the TL pre-selection is not representable on
  `Environment`.
- Per-insertion records (positions, strands) would need the ENA reads; not in scope.
- Not registered in `torchcell/verification/runners.py` (outside this branch's files);
  `run_verification` here runs the gate. No KG adapter or graph class handles
  `bacterial_gene_essentiality` yet; this branch builds only the dev LMDB.

## 2026.10.11 - Chloramphenicol resolves (#726)

The TL selection plates' `chloramphenicol` now resolves to the compound-identity row the
#726 curator pass added (PubChem CID 5959, `WIIZWVCIJKGZOK-RKDXNWHRSA-N`, CHEBI:17698;
[[torchcell.datamodels.compound_identity]], 2026.10.11). No loader code changed; the store
was rebuilt by slurm array 3809 (task 13) and both chloramphenicol references now carry the
InChIKey (0 of 2 before). Agar, tryptone and yeast extract keep their terminal rows.

### L0-L4 after the rebuild

#### `GeneEssentialityGoodall2018Dataset`

8,203 records; store built 2026-10-11 07:02 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 08:04 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 8203 records validated |
| L1 | `count` | pass | observed 8203, expected 8203 |
| L1 | `one_record_per_gene_and_condition` | pass | 8203 (condition, locus) pairs, one single-insertion record each |
| L2 | `calls_match_released_tables` | pass | 8203 records equal their Table S1/S4 row's call and locus |
| L3 | `call_threshold_convention` | pass | TL: 4313/4313 calls follow +/-3.6, LB: 4313/4313 calls follow +/-3.6 |
| L3 | `table_s1_counts_equal_the_paper` | pass | Table S1 {'essential': 358, 'non_essential': 3793, 'unclear': 162}; the paper states {'essential': 358, 'non_essential': 3793, 'unclear': 162} |
| L1 | `provenance_gaps` | pass | 16399 documented provenance gaps over 8203/8203 records; 0 deferred field(s): []; 127136 undeclared None values over 9 carrier fields (top: Compound.inchi x32805, Compound.smiles x20504, Compound.chebi_id x16406, Compound.inchikey x16406, Compound.pubchem_cid x16406) |
| L1 | `canonical_gene_names` | pass | 4172 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not) |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 16399 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 4105 records on a shared MEDIA_LIBRARY medium, 4098 on a medium deriving from one (2 distinct media) |
| L4 | `gene_containment_bw25113_locus_tags` | pass | 4172 of 4172 disrupted loci are BW25113 GenBank gene rows |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | pass | value backed by verbatim quote in paper.md |

## 2026.10.12 - Every released symbol keys a record on the released transposon leaf

The owner's rule of 2026-10-12 ([[database.genome-gating-sweep.2026.10.12]], rows 147 and 377): the loader reads no annotation. Each record is a `BacterialReleasedTransposonPerturbation` (#1077) keyed by the table's `Gene` symbol verbatim, with `strain_label` the same row label, `identifier_kind` "gene symbol, as the Table S1 and S4 Gene column writes it" and `transposon` mini-Tn5; `reconcile_locus_tags`, `canonical_symbol`, `MIN_RESOLVED_FRACTION`, the `ecoli_genome` constructor parameter, `preprocess/identifier_reconciliation.json` and the `symbol_not_on_one_bw25113_locus` rule are gone. The nine multi-copy insertion-sequence symbols on no single BW25113 locus (106 rows over the two tables) are records now; the L4 row, renamed `gene_containment_bw25113_annotation`, reports what the annotation's symbols and tags make of the 4,181 stored symbols (#1069), and `check_replicon` runs in the verifier, which opens the genome, not in the build. The one remaining rule is `unclear_call_not_representable` (317 rows; a graded sibling leaf follows #1066).

Released vs stored: 8,626 released rows = 8,309 records + 317 unclear calls; 4,260 released symbols, 4,181 of them on a kept row, all stored.

### 2026.10.12 - The released row is the record identity

The 8,309-record build of job 4483 failed: nine insertion-sequence symbols head SEVERAL rows of each table (`insH1` 11, `insD1` 7, `insB1`/`insC1`/`insA` 6, `insE1`/`insF1` 5, `insI1` 4, `insL1` 3; 53 rows per table), so `(condition, Gene)` is not a record identity. Each of those rows is a distinct released measurement with its own insertion index and likelihood ratio, which is exactly what the dropped `symbol_not_on_one_bw25113_locus` rule used to hide. `strain_label` is now the release's own coordinate of the row, `"<condition> row <n>"`; `classify_calls` refuses a repeated `(condition, row)` rather than a repeated symbol, the L1 row keys `(condition, strain_label)`, and the L2 row looks the table row up by that label before checking that the stored identifier IS that row's gene. Counts are unchanged: 4,151 TL + 4,158 LB = 8,309 records.

## 2026.10.12 - The unclear calls get a graded store (#909, PR #1083)

The 317 TraDIS calls Tables S1 and S4 release as `unclear` (the log likelihood ratio inside +/-3.6) have no boolean, so the boolean store keeps counting them under `unclear_call_not_representable`. They are now stored in a second store of the same release, `GradedGeneEssentialityGoodall2018Dataset` (`data/torchcell/graded_gene_essentiality_goodall2018`), as `GradedGeneEssentialityPhenotype` records with `essentiality_call = "unclear"` and the same `GeneEssentialityEvidence` (`tradis_insertion_index`, `tradis_call = unclear`, the insertion index and the ratio as released). A store holds one experiment class, which is why the graded calls are a second store rather than a second class in the first.

- **Keyed as the boolean records (#1082, #1086).** The genotype is the shared `insertion_genotype`: a `BacterialReleasedTransposonPerturbation` with the released `Gene` symbol as the identifier and the row's own coordinate (`TL row <n>`, `LB row <n>`) as `strain_label`; no annotation is read. The graded L2 finds each record's table row by that coordinate and checks the stored gene is the row's.
- **Every released row is in exactly one store.** `GradedLedger` (`preprocess/dropped_records.json`) counts the rows of both tables as graded (317) or boolean (8,309), 8,626 in all, and the build refuses a row in neither. `read_released_calls` is the one reader both stores call, so they split the same parsed rows; `classify_calls` refuses a repeated (condition, row) for every row.
- Reference: the unperturbed BW25113 parent with `essentiality_call = "non_essential"` (`build_graded_reference`).
- Verification: `verify_graded_build`, registered in `runners.BACTERIAL_GRADED_ESSENTIALITY_DATASETS`. Its L4 (`gene_containment_bw25113_annotation`, annotation symbols and tags) and the family's `perturbed_gene_containment_assembly` are genome-reference findings, never gates.

### L0-L4 of the graded store

Built and verified by slurm array 4584 (task 1, `--retire-existing --verify`, source `1c55407e4`); rendered by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py graded_gene_essentiality_goodall2018`. The one failing row is the family's `perturbed_gene_containment_assembly`, a report-class genome-reference finding: it compares the released symbols with locus tags, as on the boolean store; the module's own L4 against the annotation's symbols and tags holds 229 of 229.

`graded_gene_essentiality_goodall2018`: **PASS**, 40 rules, 1 failing (report written 2026-10-12T23:39:50)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 317 records validated |
| L1 | `count` | PASS | observed 317, expected 317 |
| L1 | `one_record_per_gene_and_condition` | PASS | 317 (condition, released row) pairs, one single-insertion record each |
| L2 | `unclear_calls_match_released_tables` | PASS | 317 records are unclear rows of Tables S1/S4 with their insertion index and likelihood ratio (317 unclear rows released) |
| L3 | `call_threshold_convention` | PASS | TL: 4313/4313 calls follow +/-3.6, LB: 4313/4313 calls follow +/-3.6 |
| L3 | `evidence_is_tradis_insertion_index` | PASS | 317 of 317 records and references carry tradis_insertion_index evidence (insertion_depletion) with the sourced rule, definition and caveat |
| L3 | `reference_is_the_parent_called_non_essential` | PASS | 317 of 317 references state the parent's non_essential call with the TraDIS evidence |
| L1 | `provenance_gaps` | PASS | 641 documented provenance gaps over 317/317 records; 0 deferred field(s): []; 12215 undeclared None values over 32 carrier fields (top: Compound.inchi x1275, Compound.smiles x796, Compound.chebi_id x634, Compound.inch... |
| L1 | `canonical_gene_names` | PASS | 229 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not) |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 641 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 155 records on a shared MEDIA_LIBRARY medium, 162 on a medium deriving from one (2 distinct media) |
| L4 | `gene_containment_bw25113_annotation` | PASS | 229 of 229 released gene symbols are a symbol or tag of the BW25113 GenBank annotation |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L4 | `perturbed_gene_containment_assembly` | **FAIL** | 0.000 of 229 measured genes are loci of ecoli_K12_BW25113_ASM75055v1 (>= 1.0) |

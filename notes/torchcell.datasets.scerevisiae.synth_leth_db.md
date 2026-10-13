---
id: qdfgck00436h71efgxwc5jf
title: Syn_leth_db_yeast
desc: ''
updated: 1721782510231
created: 1721782460950
---

## 2024.07.23 - We Will Not Be Able to Manually Populate Meta Data for all Synthetic Lethality and Synthetic Rescue Experiments

```python
len(set([i['experiment']['pubmed_id'] for i in lethality_dataset]))
1741
len(set([i['experiment']['pubmed_id'] for i in rescue_dataset]))
1918
```

## 2026.10.01 - PMID read as text, blank PMID refused, main under DATA_ROOT (issue #528)

- Before: `pd.read_csv` typed `r.pubmed_id` by content, so one blank cell turned the column float and every record stored `"111.0"` with a `.../111.0/` URL and the blank one `"nan"`; `main` built both datasets at the default relative roots under the working directory.
- Now: `_read_synlethdb_csv` reads `r.pubmed_id` as text and raises `BlankPubmedIdError` ("<path>: <n> row(s) with a blank r.pubmed_id (first at row <i>)"); `main` builds under `$DATA_ROOT/data/torchcell/synth_lethality_yeast_synth_leth_db` and `.../synth_rescue_yeast_synth_leth_db`, the directories the knowledge-graph configs read.
- Record-neutral: 0 blank PMIDs in `Yeast_SL.csv` (14,000 rows) and `Yeast_SR.csv` (6,948 rows); the text read stores the same string as before on every row (0 mismatches; SL already read as text because one row cites `24125552;19918932`). The built dev stores hold `"18676811"`-shaped PMIDs on all 14,000 and 6,948 records.
- Left open (record-changing on the pinned files, measured with the genome at `$DATA_ROOT/data/sgd/genome/data.db`): the name mapping lets another gene's alias overwrite a standard name (868 SL and 115 SR rows would map differently under standard-name precedence, for example SDC1 stored as YJR090C (GRR1) instead of YDR469W); 7 SL and 2 SR records repeat an ORF pair, all of them artifacts of that overwrite (0 under standard-name precedence); 3 SL and 6 SR rows name one gene twice (same Entrez id both sides); 0 swapped-order pairs. The raw CSV carries Entrez ids (`n1.identifier`, `n2.identifier`) that could resolve names without the alias table.
- Tests: `test_pmids_are_stored_verbatim_as_text`, `test_a_blank_pmid_refuses_the_build_by_name`, `test_main_builds_both_datasets_under_data_root`.

## 2026.10.01 - Review corrections: primed names and whitespace PMIDs

- A whitespace-only PMID (`" "`) counts as blank and raises `BlankPubmedIdError` like an empty cell (it used to be stored as-is). The pinned files have 0 such cells.
- A second wrong-ORF source, also left open: `get_systematic_name` strips the prime from `IMP2'`, so it resolves to YMR035W (IMP2) while its Entrez id 854652 is YIL154C (IMP21): 4 SL records (keys 241, 6067, 8568, 10553) and 4 SR records (keys 356, 357, 3705, 5131), verified in the built dev stores. With the alias overwrite, 872 SL and 119 SR records carry the wrong ORF. The mapping fix is record-changing and goes to its own dataset issue.

## 2026.10.02 - Genes resolved by Entrez id, self pairs dropped (issue #597)

- Was wrong: the name map let a later gene's `Alias` overwrite an earlier gene's standard name (last write wins), and `get_systematic_name` stripped the prime from `IMP2'`, so 872 of 14,000 SL and 119 of 6,948 SR records carried the wrong ORF; 7 SL and 2 SR records repeated an ORF pair; 3 SL and 6 SR records named one gene on both sides.
- Now: each side resolves by its Entrez id (`n1.identifier`, `n2.identifier`) through the RefSeq `ncbi_genomic.gff` (`GeneID` -> `locus_tag`, gene and pseudogene features), and the gene name must resolve to that same current gene through `SCerevisiaeGenome.resolve_gene_name` (standard name before alias; `IMP2'` keeps its prime and is an alias of YIL154C). A row is dropped, never re-resolved by name, under one of three rules recorded in `preprocess/dropped_records.json` (`DropLog`, with every dropped source row): `entrez_id_not_in_ncbi_gff`, `gene_name_disagrees_with_entrez_id`, `same_gene_on_both_sides`. A repeated unordered ORF pair among kept rows raises `DuplicateOrfPairError`. LMDB keys are now contiguous over kept rows, so they are no longer source-row indices.
- The loader is now pinned: `process()` verifies `Yeast_SL.csv` (`091e04db...56ca`) and `Yeast_SR.csv` (`d84fba78...18f4`), the dev-tree raw bytes the served build consumed, and `load_entrez_to_orf` verifies the GFF before reading it. `synth_leth_db` left `UNPINNED_LOADERS` in `test_raw_pins.py`.
- Files the resolution reads: `$DATA_ROOT/data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/ncbi_genomic.gff`, sha256 `8200def54936659721e3f7b4484d725667ed4e23492edefbbca667b8430209c4` (6,470 Entrez ids, header `annotation-source SGD R64-4-1`), and the genome's `data.db` for the name cross-check, sha256 `feadb0f5d7662606c5a6e1cfba873cdcd852900da9ebcb920bd4b6ef38530ded`, content digest `9fae73b7...c1f4`, 28 chrmt gene features. `data.db.bak` and `data_alt.db` have 0 chrmt gene features (different content digests); the build reads only `data.db`.
- Measured (`experiments/036-dataset-fixes-before-kg-build/scripts/synth_leth_db_entrez_resolution.py`, scratch build of both datasets against the dev stores; output `results/synth_leth_db_entrez_resolution.json` and `results/synth_leth_db_entrez_resolution_changed_rows.csv`):
  - SL: 14,000 source rows, 13,996 records. Old-store rows disagreeing with the Entrez ORF pair: 872. Of these, 870 kept rows now carry a different ORF pair and 2 were dropped (row 5232 by `entrez_id_not_in_ncbi_gff`, row 5719 by `same_gene_on_both_sides`); no kept row outside the 872 changed. Dropped: row 5232 (`TAF1,853191,YPR108W-A,1466522`; Entrez 1466522 is in neither the pinned GFF nor the current R64-5-1 NCBI GFF, while SGD R64-4-1 has YPR108W-A) and self pairs 3152 (PUS1), 5719 (TAF1), 8663 (SBA1). Duplicate unordered ORF pairs: 7 old, 0 new.
  - SR: 6,948 source rows, 6,942 records. Old-store disagreements: 119; kept rows changed: 119, all of them disagreements. Dropped: self pairs 370, 1182, 3138, 3154, 6616, 6672. Duplicates: 2 old, 0 new.
  - Name rule: 0 rows in either file (every name agrees with its Entrez id).
  - The ten issue names plus `IMP2'`, ORFs stored old -> new: STM1 YPR163C -> YLR150W, SDC1 YJR090C -> YDR469W, MFT1 YMR177W -> YML062C, NSP1 YLR178C -> YJL041W (SL only), RPL37A YPL143W -> YLR185W, CCS1 YOL081W -> YMR038C, YPK1 YNL307C -> YKL126W, TAF1 YHL047C -> YGR274C, SSL2 YLR452C -> YIL143C, HAP1 YPL101W -> YLR256W, IMP2' YMR035W -> YIL154C.
- Tests: `tests/torchcell/datasets/scerevisiae/test_synth_leth_db.py` rewritten around a stub genome carrying the real `resolve_gene_name` and a fixture GFF (the STM1 standard-vs-alias and `IMP2'` prime cases are mirrored); `test_real_files_resolve_by_entrez_with_the_issue_counts` (`--data`) pins the drop rows, 0 duplicates and the eleven names on the real files.
- Open:
  - `ncbi_genomic.gff` is not in the genomes tier (`torchcell-genomes/sgd_S288C_R64-4-1_20230830` excludes it) and NCBI now serves R64-5-1 under GCF_000146045.2, so its retrieval cannot be reproduced; the sha256 pin is the only record. Depositing it as its own tier set with `provenance_complete=False` needs a go-ahead.
  - Row 5232 is dropped because its Entrez id is absent from the pinned GFF, although its name `YPR108W-A` is a current SGD ORF. Resolving it by name would be a fallback; reversing the drop is a decision for the user.
  - The self-pair drop assigns no meaning to a gene paired with itself; if SynLethDB documents one, the rule can be revisited.

## 2026.10.02 - Sourced medium replaces the inline stub (issue #622)

Issue #622: both datasets built `Media(name="YEPD", state="solid", is_synthetic=False)` at `Temperature(30)`. Neither value is stated anywhere the loader reads. SynLethDB aggregates SL pairs from many papers; its release paper (Wang 2022, mirror `wangSynLethDB20Webbased2022/paper.md` line 49, sha256 `acaee2c7...`) lists what an entry carries: "The species, references to PubMed, supporting evidence, cell lines and other relevant information about an SL entry are stored as properties of the edge". The released CSV header is `n1.name, n1.identifier, n2.name, n2.identifier, r.cell_line, r.pubmed_id, r.source, r.statistic_score`. No medium, no temperature.

The fix records the absence instead of a default:

- `SYNLETHDB_MEDIUM_NOT_CARRIED`: one `composition_deferred` component naming the record's primary study as the source of the medium, with the line-49 quote as provenance. A true typed gap on the medium is not expressible in the current schema (`Environment.media` is required, `Media` has no `provenance_gaps`), so this object is the in-schema stand-in. `state="solid"` and `is_synthetic=False` are schema-required and NOT sourced; they keep the old stub's values and say so in the SourcedValue note.
- `SYNLETHDB_TEMPERATURE_GAP`: `temperature=None` with `ProvenanceGap(field="temperature", reason=not_carried_by_curation)`, looked-in the same paper. This goes beyond the medium named in the issue; it is the same unsourced default on the same environment.

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/media_stubs_seven_loaders.py` (output `experiments/036-dataset-fixes-before-kg-build/results/media_stubs_seven_loaders.csv`): scratch builds hold 13,996 SL and 6,942 SR records (the #597 counts; the dev stores, built before #597, hold 14,000 and 6,948) and every environment carries the new medium and the temperature gap; the dev stores carry the `YEPD` stub on all 28,000 and 13,896 environments.

Open: making the medium a real typed gap needs a schema decision (`Media.state` optional plus `ProvenanceGapMixin` on `Media`, or `Environment.media` optional), which touches every adapter that reads `media.state`.

## 2026.10.10 - L0-L4 verification of the dev stores (#889)

Both stores run the new SynLethDB battery (`torchcell/verification/synthetic_lethality.py`). `released_rows` re-reads the pinned CSV, skips the ledgered drops by source row, resolves each side by its Entrez id through the pinned NCBI GFF and compares the (unordered ORF pair, score, PubMed id) multiset: both pass. `release_accounting` passes (SL: 14,000 = 13,996 + 4 = the paper's "14 000 of Saccharomyces cerevisiae"; SR: 6,948 = 6,942 + 6, no stated count). Skipped, with reason: no cross-dataset L4 containment, because neither file claims its pairs are a subset of a served store; the L4 rows are the S288C gene rules. SL scores must lie in [0, 1], the range of the paper's normalized confidence score; the paper defines no SR score, so SR runs `score_defined_by_source` instead.

Findings, left failing and filed: #902 (`syn_leth_db_yeast`, the #622 medium stand-in joins nothing), #903 (`syn_rescue_db_yeast`, the same plus 74 negative BioGRID scores, -1.0 to -0.21, that no source defines).

Tables generated by `experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_report_tables.py` from the reports the sweep wrote (`kg4_pre_build_verification_sweep.py --dataset <Class>`, read-only on `processed/`); count oracles derived by `no_verifier_store_release_counts.py` (results `results/no_verifier_store_release_counts.json`).

`syn_leth_db_yeast`: **FAIL**, 21 rules, 1 failing (report written 2026-10-10T21:29:17)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 13996 records validated as SyntheticLethalityExperiment |
| L0 | `reference_structural` | PASS | 13996 records validated as SyntheticLethalityExperimentReference |
| L1 | `count` | PASS | observed 13996, expected 13996 |
| L1 | `two_distinct_genes` | PASS | all 13996 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 13996 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 13996 experiments say is_synthetic_lethal True |
| L2 | `score_in_unit_interval` | PASS | 13996 stated scores lie in [0, 1] (0 records state none) |
| L2 | `released_rows` | PASS | the 13996 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 14000 released rows = 13996 stored + 4 ledgered drops; the release paper states 14000 |
| L3 | `reference_negative` | PASS | every one of the 13996 references says is_synthetic_lethal False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 4 ledgered drops meets its rule on the released row |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L1 | `provenance_gaps` | PASS | 13996 documented provenance gaps over 13996/13996 records; 0 deferred field(s): []; 111968 undeclared None values over 8 carrier fields (top: Compound.chebi_id x13996, Compound.inchi x13996, Compound.inchikey x13996, ... |
| L1 | `canonical_gene_names` | PASS | 3547 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | **FAIL** | 1 free-text media over 13996 records join nothing: growth medium not carried by SynLethDB (stated, if at all, by each record's primary paper) (base_medium=None) x13996 |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 3547 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 3547 measured systematic names is a gene of the current genome |

`syn_rescue_db_yeast`: **FAIL**, 19 rules, 2 failing (report written 2026-10-10T21:29:30)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 6942 records validated as SyntheticRescueExperiment |
| L0 | `reference_structural` | PASS | 6942 records validated as SyntheticRescueExperimentReference |
| L1 | `count` | PASS | observed 6942, expected 6942 |
| L1 | `two_distinct_genes` | PASS | all 6942 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 6942 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 6942 experiments say is_synthetic_rescue True |
| L2 | `score_defined_by_source` | **FAIL** | 74 of 6942 records state a score the source never defines (74 of them outside [0, 1]) |
| L2 | `released_rows` | PASS | the 6942 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 6948 released rows = 6942 stored + 6 ledgered drops; the release paper states no count for this file |
| L3 | `reference_negative` | PASS | every one of the 6942 references says is_synthetic_rescue False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 6 ledgered drops meets its rule on the released row |
| L1 | `provenance_gaps` | PASS | 6942 documented provenance gaps over 6942/6942 records; 0 deferred field(s): []; 62404 undeclared None values over 9 carrier fields (top: Compound.chebi_id x6942, Compound.inchi x6942, Compound.inchikey x6942, Compoun... |
| L1 | `canonical_gene_names` | PASS | 2980 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | **FAIL** | 1 free-text media over 6942 records join nothing: growth medium not carried by SynLethDB (stated, if at all, by each record's primary paper) (base_medium=None) x6942 |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 2980 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 2980 measured systematic names is a gene of the current genome |

## 2026.10.11 - Typed medium gap, the 74 undefined SR scores gapped, per-source counts (#902, #903) [the loader half superseded the same day, see below]

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings.py` (results `experiments/036-dataset-fixes-before-kg-build/results/sgd_synlethdb_findings.json`) on the pinned `Yeast_SL.csv` (`091e04db...56ca`) and `Yeast_SR.csv` (`d84fba78...18f4`) and on the dev stores before and after the rebuild.

### Medium (#902, #903)

The #622 stand-in (a `Media` named "growth medium not carried by SynLethDB (stated, if at all, by each record's primary paper)") joined no shared medium and failed L3 `media_membership` on all 13,996 SL and 6,942 SR records. No medium is sourced per record: the release header is `n1.name, n1.identifier, n2.name, n2.identifier, r.cell_line, r.pubmed_id, r.source, r.statistic_score`, and `r.cell_line` is null on all 14,000 and 6,948 rows. Every environment is now `media=None` with `ProvenanceGap(field='media', reason=not_carried_by_curation, looked_in=paper.md)`, whose note quotes line 49 ("The species, references to PubMed, supporting evidence, cell lines and other relevant information about an SL entry are stored as properties of the edge") and the header; the temperature gap is unchanged. This needed the schema to carry a medium gap (`Environment.media` optional only beside the gap, [[torchcell.datamodels.schema]]), and `media_membership` counts a typed medium gap as a declared absence ([[torchcell.verification.common]]).

### The 74 SR scores (#903)

`Yeast_SR.csv` states `r.statistic_score` on 74 of 6,948 rows; all 74 are kept rows. Every one has `r.source` BioGRID and `r.pubmed_id` 24470217 (Atencio et al. 2014, G3, "The yeast Ess1 prolyl isomerase controls Swi6 and Whi5 nuclear localization", DOI 10.1534/g3.113.008763; PubMed metadata only), and they run from -1.0 to -0.2. Looking for a definition:

- SynLethDB 2.0 defines its score for SL pairs only, `paper.md` line 75: "we integrated the scores of different types of sources into a normalized confdence score for every SL pair" (OCR drops the 'fi' ligature).
- For SR it says only that the pairs were collected, line 38: "We collected 16 207 SR gene pairs and 5798 non-synthetic lethal (non-SL) gene pairs from the above three sources".
- Its BioGRID source is reference 22, line 38: "The second source is public databases containing SL data such as GenomeRNAi (21) and BioGRID (22)." The reference list in `paper.pdf` (sha256 `21e8c003...70f7`, read with `pdftotext -layout`) gives reference 22 as "The BioGRID interaction database: update. Nucleic Acids Res., 47, D529–D541", the BioGRID 2019 update (DOI 10.1093/nar/gky1079). Neither that paper nor Atencio 2014 is in the mirror.

No mirrored source defines these numbers, so they are not stored as SynLethDB's statistic score and are not transformed: each record keeps `synthetic_rescue_statistic_score=None` with `ProvenanceGap(field='synthetic_rescue_statistic_score', reason=deferred_pending_source_review, looked_in=paper.md, resolve_with=the BioGRID 2019 update)` whose note carries the released value, source and PubMed id verbatim (`sr_score_gap`). The four `paper.md` sentences are `SYNLETHDB_SOURCED_QUOTES`, audited against the mirror bytes. Mirroring the BioGRID 2019 update and Atencio 2014 is the way to close the gap (an owner go-ahead).

### Per-source counts (the aggregation record, plan decision 15)

SynLethDB is an aggregation: each row names its source type (`r.source`) and its source study (`r.pubmed_id`). Counted here, not loaded (owner policy: the compendium is a sanity check; its original studies are loaded later from their own releases).

| File | Rows | `r.source` | Rows by source | Distinct PubMed ids by source |
|---|---|---|---|---|
| SL | 14,000 | High Throughput | 7,721 | 62 |
| SL | | Low Throughput | 4,944 | 1,660 |
| SL | | High Throughput\|Low Throughput | 1,248 | 21 |
| SL | | Text Mining | 87 | 40 |
| SR | 6,948 | BioGRID | 6,948 | 1,918 |

Source studies (distinct PubMed ids, a multi-paper cell split): SL 1,741 (one row cites two papers, `24125552;19918932`, both also cited alone), SR 1,918, either file 3,308. Every SL score is the source type's value (High Throughput 0.5, Low Throughput 0.8, both 0.9; Text Mining 0.25 to 1.0), in [0.25, 1].

### Stores and L0-L4

Rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset <Class> --retire-existing --verify`: SL 13,996 records (5 s; old store `processed.superseded.20261011-014903`), SR 6,942 (3 s; `processed.superseded.20261011-014915`), counts unchanged. The #889 verifier's `released_rows` now reads the SR file's released score as None where the source defines no score (`keep_scores=False`), the same statement `score_defined_by_source` makes; on the as-written verifier the 74 gapped records were the only `released_rows` mismatch (74 stored rows not released, 74 released rows not stored), and `score_defined_by_source` and `media_membership` already passed. Verified by `kg4_pre_build_verification_sweep.py --dataset <Class>`; tables rendered by `no_verifier_store_report_tables.py`:

`syn_leth_db_yeast`: **PASS**, 21 rules, 0 failing (report written 2026-10-11T02:15:07)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 13996 records validated as SyntheticLethalityExperiment |
| L0 | `reference_structural` | PASS | 13996 records validated as SyntheticLethalityExperimentReference |
| L1 | `count` | PASS | observed 13996, expected 13996 |
| L1 | `two_distinct_genes` | PASS | all 13996 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 13996 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 13996 experiments say is_synthetic_lethal True |
| L2 | `score_in_unit_interval` | PASS | 13996 stated scores lie in [0, 1] (0 records state none) |
| L2 | `released_rows` | PASS | the 13996 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 14000 released rows = 13996 stored + 4 ledgered drops; the release paper states 14000 |
| L3 | `reference_negative` | PASS | every one of the 13996 references says is_synthetic_lethal False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 4 ledgered drops meets its rule on the released row |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | PASS | value backed by verbatim quote in paper.md |
| L1 | `provenance_gaps` | PASS | 27992 documented provenance gaps over 13996/13996 records; 0 deferred field(s): []; 41988 undeclared None values over 3 carrier fields (top: Environment.dilution_rate_per_hour x13996, Environment.duration_generations ... |
| L1 | `canonical_gene_names` | PASS | 3547 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 13996 with a typed medium gap (not_carried_by_curation x13996) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 3547 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 3547 measured systematic names is a gene of the current genome |

`syn_rescue_db_yeast`: **PASS**, 19 rules, 0 failing (report written 2026-10-11T02:15:19)

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 6942 records validated as SyntheticRescueExperiment |
| L0 | `reference_structural` | PASS | 6942 records validated as SyntheticRescueExperimentReference |
| L1 | `count` | PASS | observed 6942, expected 6942 |
| L1 | `two_distinct_genes` | PASS | all 6942 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 6942 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 6942 experiments say is_synthetic_rescue True |
| L2 | `score_defined_by_source` | PASS | no record states a score (6942 records), and the source defines none |
| L2 | `released_rows` | PASS | the 6942 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 6948 released rows = 6942 stored + 6 ledgered drops; the release paper states no count for this file |
| L3 | `reference_negative` | PASS | every one of the 6942 references says is_synthetic_rescue False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 6 ledgered drops meets its rule on the released row |
| L1 | `provenance_gaps` | PASS | 13958 documented provenance gaps over 6942/6942 records; 1 deferred field(s): ['synthetic_rescue_statistic_score']; 27694 undeclared None values over 4 carrier fields (top: Environment.dilution_rate_per_hour x6942, En... |
| L1 | `canonical_gene_names` | PASS | 2980 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | PASS | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 6942 with a typed medium gap (not_carried_by_curation x6942) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 2980 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 2980 measured systematic names is a gene of the current genome |

## 2026.10.11 - NOT LOADED: no SynLethDB field states a growth condition (owner decision)

`Environment.media` stays required (owner, [[torchcell.datamodels.schema]]): a synthetic-lethal call is a growth measurement, so the record names its medium, and a source that does not state one produces no record. Both `SynthLethalityYeastSynthLethDbDataset` and `SynthRescueYeastSynthLethDbDataset` left the served graph and `torchcell/datasets/scerevisiae/synth_leth_db.py` is a provenance record, registering no dataset, in the shape of MCF2Chem 2023 ([[torchcell.datasets.ecoli.cai2023]]) and CeCaFDB 2015 ([[torchcell.datasets.ecoli.zhang2015]]).

### Storable records: 0 of 14,000 SL and 0 of 6,948 SR

Both released files carry the same eight columns, on verified bytes (`SL_CSV_SHA256` = `091e04db...`, `SR_CSV_SHA256` = `d84fba78...`): `n1.name`, `n1.identifier`, `n2.name`, `n2.identifier`, `r.cell_line`, `r.pubmed_id`, `r.source`, `r.statistic_score`. Not one names a medium, a temperature, a carbon source or a culture state, and `r.cell_line`, the one column that could carry a growth context, is empty on all 20,948 rows. The paper says the same of its own data model, and the sentence is the pinned quote `SYNLETHDB_ENTRY_FIELDS_QUOTE`. `release_inventory` measures this and raises `ConditionColumnFound` rather than absorbing a release that grew a condition column; `test_the_real_release_matches_the_pins_and_every_recorded_count` pins both counts at 0.

### The decomposition, which is the loadable path

Under the 2026-10-10 policy ([[torchcell.candidates.aggregation-decomposition]]) a served compendium is a processing reference, not the source of record, and each original study inside it is loaded from its own release. Measured by `source_decomposition`:

| File | Rows | Distinct source PubMed ids | Evidence labels (`r.source`) |
|---|---:|---:|---|
| `Yeast_SL.csv` | 14,000 | 1,741 | High Throughput 7,721 / Low Throughput 4,944 / both 1,248 / Text Mining 87 |
| `Yeast_SR.csv` | 6,948 | 1,918 | BioGRID 6,948 |

3,308 distinct PubMed ids across both files, every one an original study whose own paper states its medium. The whole SR file is labeled BioGRID, so every SR row is a database re-export rather than a reading of the paper its PubMed id names.

### The SL score is a label, the 74 SR scores are undefined

All 14,000 SL rows carry a score and it is a function of the evidence label, not of the pair: 0.5 for High Throughput, 0.8 for Low Throughput, 0.9 for both, and one of 0.25 / 0.5 / 0.75 / 1.0 for Text Mining (`score_by_source`). 74 SR rows carry a score (-1.0 to -0.2, all `r.source` BioGRID, all PubMed 24470217) that no mirrored document defines: the paper defines its confidence score for SL pairs only, and the BioGRID 2019 update it cites (`BIOGRID_2019_REFERENCE`, doi:10.1093/nar/gky1079) is not in the mirror. The numbers are recorded here rather than stored.

### What the module keeps

Both release pins and the paper pin with its four audited quotes; the release inventory, the decomposition and the score measurement; the Entrez-to-ORF map (`load_entrez_to_orf`, `ncbi_gff_path`, `resolved_orf_pairs`) against the pinned RefSeq R64 annotation, because a per-study loader of any SynLethDB source paper resolves its pairs with the same map; and the #597 finding as constants (`OLD_RESOLUTION_WRONG_ORF`: 872 SL and 119 SR records stored under the wrong ORF by the old name map, which also produced 7 SL and 2 SR duplicate unordered pairs).

### What read the classes, and what loses them

- Removed with them: `torchcell/adapters/synth_leth_db_adapter.py` and both its confs, the two per-dataset KG confs, their entries in `scerevisiae_global_kg.yaml`, `scerevisiae_small_global_kg.yaml` and `dataset_adapter_map`, `torchcell/verification/synthetic_lethality.py` with `run_synthetic_pairs`, and both rows of `candidates/store.py`'s `GRANDFATHERED` (134 to 131, with the SGD class).
- **One supported query loses a block and is reported, not rewritten**: `torchcell/knowledge_graphs/queries/solid_growth_025.cql`, whose `SynthLethalityYeastSynthLethDbDataset` block converts SL calls to fitness downstream (it also loses its `GeneEssentialitySgdDataset` block).
- The SynLethDB arm of `experiments/036-dataset-fixes-before-kg-build/scripts/{media_stubs_seven_loaders,synth_leth_db_entrez_resolution}.py` now raises `SynLethDbLoaderRemoved` with the reason; their committed results stand as the record of what they measured while the loaders existed.
- The dev stores `$DATA_ROOT/data/torchcell/syn_{leth,rescue}_db_yeast` are orphaned (no loader builds them) and read stale. Deprecating them is an owner decision; nothing here deletes them.

## 2026.10.12 - Loaded again: a typed gap on Media, both classes kept (owner decision)

The owner reversed the removal of the section above, verbatim: "we are fine with irreversible updates to schema, as long as you are convinced it is correct ... why would you get rid of old classes?" `SynthLethalityYeastSynthLethDbDataset` and `SynthRescueYeastSynthLethDbDataset` are back in the registry, the KG mapping (their two KG confs and their entries in `scerevisiae_global_kg.yaml`, `scerevisiae_small_global_kg.yaml` and `dataset_adapter_map`), their adapter, and the verifiers (`run_synthetic_pairs`), restored from the loader of the 2026.10.11 typed-gap section: resolution by Entrez id, the 74 undefined SR scores as `deferred_pending_source_review` gaps, the pubmed dtype spelled as a string for CI's pandas stubs. What changed is where the unstated medium lives: `Environment.media` stays a required `Media`, and `SYNLETHDB_MEDIUM` is a `Media` whose `name`, `state` and `is_synthetic` are each `None` beside a `ProvenanceGap(reason=not_carried_by_curation, looked_in=paper.md)`. The note of each gap is `SYNLETHDB_MEDIUM_SILENCE`, which quotes line 49 verbatim ("The species, references to PubMed, supporting evidence, cell lines and other relevant information about an SL entry are stored as properties of the edge"), names the released header, and states that `r.cell_line` is empty on every row (`n_cell_line_stated` 0 of 14,000 SL and 0 of 6,948 SR in `results/sgd_synlethdb_findings.json`). The temperature stays a gap on the environment; no YPD and no 30 °C is stated anywhere.

### The rebuilt dev stores (slurm 4184_2 and 4184_3, verified by 4194)

Built from `86c7ae573` with `--retire-existing --verify` (the main-code builds of job 3953 moved aside): **13,996 SL records** (14,000 released less 4 ledgered drops, 3,547 genes) and **6,942 SR records** (6,948 less 6, 2,980 genes). The store served today holds 14,000 and 6,948 under the old loader. Verified by the family runner through `kg4_pre_build_verification_sweep.py` (slurm 4194, HEAD `86c7ae573`, reports `$DATA_ROOT/data/torchcell/{syn_leth_db_yeast,syn_rescue_db_yeast}/preprocess/verification_report.json`); `--list-stale --include-private` names neither.

`syn_leth_db_yeast`, PASS on all 21 rules:

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 13996 records validated as SyntheticLethalityExperiment |
| L0 | `reference_structural` | PASS | 13996 records validated as SyntheticLethalityExperimentReference |
| L1 | `count` | PASS | observed 13996, expected 13996 |
| L1 | `two_distinct_genes` | PASS | all 13996 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 13996 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 13996 experiments say is_synthetic_lethal True |
| L2 | `score_in_unit_interval` | PASS | 13996 stated scores lie in [0, 1] (0 records state none) |
| L2 | `released_rows` | PASS | the 13996 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 14000 released rows = 13996 stored + 4 ledgered drops; the release paper states 14000 |
| L3 | `reference_negative` | PASS | every one of the 13996 references says is_synthetic_lethal False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 4 ledgered drops meets its rule on the released row |
| L3 | `provenance_audit` (x2) | PASS | value backed by verbatim quote in paper.md |
| L1 | `provenance_gaps` | PASS | 55984 documented provenance gaps over 13996/13996 records; 0 deferred field(s) |
| L1 | `canonical_gene_names` | PASS | 3547 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references; 0 declare a typed gap |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references; 0 declare a typed gap |
| L3 | `media_membership` | PASS | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 13996 on a medium the source does not name, as a typed gap (not_carried_by_curation x13996) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 3547 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 3547 measured systematic names is a gene of the current genome |

`syn_rescue_db_yeast`, PASS on all 19 rules:

| Level | Rule | Verdict | Message |
|---|---|---|---|
| L0 | `structural` | PASS | 6942 records validated as SyntheticRescueExperiment |
| L0 | `reference_structural` | PASS | 6942 records validated as SyntheticRescueExperimentReference |
| L1 | `count` | PASS | observed 6942, expected 6942 |
| L1 | `two_distinct_genes` | PASS | all 6942 records perturb two distinct named genes at graph level 'edge' |
| L1 | `unordered_pair_uniqueness` | PASS | 6942 unordered gene pairs, one record each |
| L2 | `positive_label` | PASS | all 6942 experiments say is_synthetic_rescue True |
| L2 | `score_defined_by_source` | PASS | no record states a score (6942 records), and the source defines none |
| L2 | `released_rows` | PASS | the 6942 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side resolved by its Entrez id |
| L2 | `release_accounting` | PASS | 6948 released rows = 6942 stored + 6 ledgered drops; the release paper states no count for this file |
| L3 | `reference_negative` | PASS | every one of the 6942 references says is_synthetic_rescue False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | PASS | each of the 6 ledgered drops meets its rule on the released row |
| L1 | `provenance_gaps` | PASS | 27842 documented provenance gaps over 6942/6942 records; 1 deferred field(s): ['synthetic_rescue_statistic_score'] |
| L1 | `canonical_gene_names` | PASS | 2980 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place |
| L2 | `uncertainty_sanity` | PASS | 0 labeled uncertainties, none a zero dispersion |
| L3 | `compound_identity` | PASS | environment edits: 0 compound references; 0 declare a typed gap |
| L3 | `media_compound_identity` | PASS | medium components: 0 compound references; 0 declare a typed gap |
| L3 | `media_membership` | PASS | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 6942 on a medium the source does not name, as a typed gap (not_carried_by_curation x6942) |
| L4 | `gene_containment_sgd` | PASS | 1.000 of 2980 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | PASS | every one of the 2980 measured systematic names is a gene of the current genome |

The gap counts check by hand: four gaps per record (three on the medium, one on the temperature), so 13,996 x 4 = 55,984 for SL and 6,942 x 4 = 27,768 plus the 74 SR score gaps = 27,842 for SR. Both #902 and #903 failures pass now: `media_membership` on both stores, and `score_defined_by_source` on SR (the 74 released scores are gaps, not stored values).

**Rebuilt once more at `8d1e8f423` (slurm 4246, verified by 4247).** The schema-change record commit declared `Media.name` and `state` without their documentation-only `Field` wrapper, which moved `Media`'s contract fingerprint (not its dump), so the stores above read stale until rebuilt. The rebuild gives the same counts (1,287 SGD, 13,996 SL, 6,942 SR) and the sweep the same verdict and the same message on every rule as job 4194 (compared row by row, timestamps and HEAD aside); `--list-stale --include-private` names none of the three.

## 2026.10.12 - Every released name stored as released on the released-gene-name leaf (#1045, PR #1085)

Both loaders drop the genome and the GFF. Each side of every row is stored on `ReleasedGeneNamePerturbation` exactly as released: `n1.name` / `n2.name` verbatim in `systematic_gene_name` and `released_gene_name`, the Entrez id verbatim in `released_gene_id` (namespace `ncbi_entrez_gene`), and `released_kind` a typed `not_reported_by_primary` gap, because SynLethDB names the pair, its source, PubMed id and score and not the perturbation that made it (the gap quotes the release paper's statement of what an SL entry stores). The 2026.10.02 drop rules are gone: the row whose Entrez id is in no R64 GFF (`YPR108W-A`, 1466522) is a record, and the rows that name one gene on both sides (3 SL, 6 SR) are stored as released, their genotype the one perturbation. The ledger counts them (`n_self_pairs`) and the runner spec declares them through `expected_self_pairs`, which the pair verifier's `two_distinct_genes` row now takes. The verifier runner keys the released rows by name and reads no GFF; `load_entrez_to_orf` and `ncbi_gff_path` stay in the module as retained utilities for `experiments/036`. A repeated unordered (name, id) pair refuses the build (`DuplicatePairError`; none in the pinned files).

Measured on the pinned files: 14,000 SL and 6,948 SR records, every released row, against 13,996 and 6,942 before. The L0-L4 tables from the rebuild follow below once job 4606 finishes.

### Family verification (job 4606): the self pairs in the released-rows comparison

The pair verifier set the 3 SL and 6 SR self pairs aside for the declared `two_distinct_genes` count and left them out of the stored multiset, so `released_rows` reported them as kept released rows with no record. A self pair now counts as a stored row as well (never as a repeated gene pair); verified again by job 4658.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**syn_leth_db_yeast**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/syn_leth_db_yeast/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 14000 records validated as SyntheticLethalityExperiment |
| L0 | `reference_structural` | gate | ok | 14000 records validated as SyntheticLethalityExperimentReference |
| L1 | `count` | gate | ok | observed 14000, expected 14000 |
| L1 | `two_distinct_genes` | gate | ok | all 13997 records perturb two distinct named genes at graph level 'edge' and 3 name one gene on both sides, as the loader declares |
| L1 | `unordered_pair_uniqueness` | gate | ok | 13997 unordered gene pairs, one record each |
| L2 | `positive_label` | gate | ok | all 14000 experiments say is_synthetic_lethal True |
| L2 | `score_in_unit_interval` | gate | ok | 13997 stated scores lie in [0, 1] (3 records state none) |
| L2 | `released_rows` | gate | ok | the 14000 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side under its released name |
| L2 | `release_accounting` | gate | ok | 14000 released rows = 14000 stored + 0 ledgered drops; the release paper states 14000 |
| L3 | `reference_negative` | gate | ok | every one of the 14000 references says is_synthetic_lethal False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | gate | ok | each of the 0 ledgered drops meets its rule on the released row |
| L3 | `provenance_audit` | gate | ok | value backed by verbatim quote in paper.md |
| L3 | `provenance_audit` | gate | ok | value backed by verbatim quote in paper.md |
| L1 | `provenance_gaps` | gate | ok | 84000 documented provenance gaps over 14000/14000 records; 0 deferred field(s): []; 168000 undeclared None values over 8 carrier fields (top: ReleasedGeneNamePerturbation.collection x28000, ReleasedGeneNamePerturbation.crispr x28000, ReleasedGeneNamePerturbation.marker x28000, ReleasedGeneNamePerturbation.strain_id x28000, Environment.dilution_rate_per_hour x14000) |
| L1 | `canonical_gene_names` | report | finding | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 3435 systematic names are not the genome's current name; 3434 common names resolve to another gene |
| L2 | `uncertainty_sanity` | gate | ok | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | gate | ok | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | gate | ok | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | gate | ok | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 14000 on a medium the source does not name, as a typed gap (not_carried_by_curation x14000) |
| L4 | `gene_containment_sgd` | report | finding | 0.032 of 3548 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | report | finding | 3435 systematic names are absent from the current genome over 14000 records: AAC1 x1, AAC3 x1, AAD4 x2, AAH1 x3, AAR2 x2, AAT1 x1, AAT2 x10, ABF1 x4, ABM1 x2, ABP1 x5 |

**syn_rescue_db_yeast**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/syn_rescue_db_yeast/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 6948 records validated as SyntheticRescueExperiment |
| L0 | `reference_structural` | gate | ok | 6948 records validated as SyntheticRescueExperimentReference |
| L1 | `count` | gate | ok | observed 6948, expected 6948 |
| L1 | `two_distinct_genes` | gate | ok | all 6942 records perturb two distinct named genes at graph level 'edge' and 6 name one gene on both sides, as the loader declares |
| L1 | `unordered_pair_uniqueness` | gate | ok | 6942 unordered gene pairs, one record each |
| L2 | `positive_label` | gate | ok | all 6948 experiments say is_synthetic_rescue True |
| L2 | `score_defined_by_source` | gate | ok | no record states a score (6948 records), and the source defines none |
| L2 | `released_rows` | gate | ok | the 6948 stored (gene pair, score, PubMed id) rows are the release's kept rows, each side under its released name |
| L2 | `release_accounting` | gate | ok | 6948 released rows = 6948 stored + 0 ledgered drops; the release paper states no count for this file |
| L3 | `reference_negative` | gate | ok | every one of the 6948 references says is_synthetic_rescue False with no score, in its experiment's own environment |
| L3 | `drop_ledger` | gate | ok | each of the 0 ledgered drops meets its rule on the released row |
| L1 | `provenance_gaps` | gate | ok | 41762 documented provenance gaps over 6948/6948 records; 1 deferred field(s): ['synthetic_rescue_statistic_score']; 90250 undeclared None values over 9 carrier fields (top: ReleasedGeneNamePerturbation.collection x13896, ReleasedGeneNamePerturbation.crispr x13896, ReleasedGeneNamePerturbation.marker x13896, ReleasedGeneNamePerturbation.strain_id x13896, Environment.dilution_rate_per_hour x6948) |
| L1 | `canonical_gene_names` | report | finding | 0 genes carry conflicting common-name spellings (0 records; 0 case-only); 2871 systematic names are not the genome's current name; 2870 common names resolve to another gene |
| L2 | `uncertainty_sanity` | gate | ok | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | gate | ok | environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | gate | ok | medium components: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | gate | ok | 0 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (0 distinct media), 6948 on a medium the source does not name, as a typed gap (not_carried_by_curation x6948) |
| L4 | `gene_containment_sgd` | report | finding | 0.037 of 2980 measured genes are S288C reference genes (>= 0.9) |
| L4 | `current_genome_genes` | report | finding | 2871 systematic names are absent from the current genome over 6948 records: AAC1 x1, AAD3 x1, AAR2 x1, AAT1 x1, AAT2 x1, ABP1 x2, ABP140 x2, ACC1 x5, ACE2 x3, ACF4 x3 |

---
id: td54n8bmmlrlc87ceeb4svg
title: Cooper2010
desc: ''
updated: 1789463989696
created: 1789463989696
---

## 2026.09.15 - Built as the 51st dataset: what Table 4 holds and what the loader does

Plan: [[plan.cooper2010-amino-acid-metabolome.2026.09.15]]. Loader
`torchcell/datasets/scerevisiae/cooper2010.py`, class `AminoAcidCooper2010Dataset`, root
`data/torchcell/amino_acid_cooper2010`. Adapter: [[torchcell.adapters.cooper2010_adapter]].

### Provenance

- Paper mirror key `cooperHighthroughputProfilingAmino2010`; `paper.md` sha256
  `b7739b1eeaf68b609a3e13b870f28649c331fe36b3c930bbbfd8a4c4fd19e52a`. Every `SourcedValue`
  quote hits `paper.md` exactly once (`test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper`).
- Raw mirror `$DATA_ROOT/torchcell-raw/cooperHighthroughputProfilingAmino2010/` written by
  `deposit_raw_mirror(source_dir=<staged deposit>)` on 2026-09-15, run twice (idempotent):
  `data/SupplementalTable4.txt` (874,149 B, sha256
  `3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb`, role `raw_data`) and
  `data/Supplemental_Table_Legends.doc` (20,992 B, sha256
  `ca4983cda4c34318dbc3a05d10edf8e05df11f1560ffc99442df83a6a4399184`, role `si_data`), each with a
  `RetrievalRecord(method=manual_browser, source_url=https://genome.cshlp.org/content/20/9/1288,
  retriever="manual", retrieved_at="2026-09-15")` whose `params.retrieval_command` is the browser
  recipe (institutional login; scripted access returned 429/403/404) and whose sha256 was checked
  against the staged `SHA256SUMS.txt` before copying. Only the two consumed files are deposited;
  Tables 1, 2A/2B, 3A/3B and `ClusterInNorm.zip` stay in the staging directory and are named in
  `si_expected`. `download()` symlinks Table 4 from the mirror after a sha256 check and has no
  network path.

### What Supplemental Table 4 holds (measured)

4,382 rows x 19 tab-separated columns: `Sysname`, `NAME`, then 17 peak columns `RPS19`, `arg1`,
`gshbiotin`, `NacOrn`, `LeuIleCit`, `GlnVal`, `MetPro`, `Thr`, `Ala`, `Ser`, `Asn-Tyr`, `Gly`,
`LysA`, `Orn`, `LysB`, `Glu`, `Asp`; `-` marks a missing cell. The headers are the authors'
template names, not Fig. 1's peak numbers. They match the 17 column labels of Fig. 4B one to one
(Biotin, Threonine, N-acetylOrnithine, Leu,Ile,Cit, Asn,Tyr, Lysine-related, LysineA, LysineB,
Gln,Val, Arginine, Met,Pro, Ornithine, Alanine, Glycine, Serine, Glu, Asp; caption: "Columns
represent each amino acid as labeled in B."), read off the mirrored image
`images/70fa0211a77f9bb993b8fd8fe4aec911d98d41634bbe9b4b075d122f5b6c833b.jpg`; Fig. 1
(`images/7fc3cb9ee368ca46dece6e0e27b69a139b19874e121080cccb866e086ddd9870.jpg`) confirms the
1-18 order the plan's scout guessed: 1 Lysine-related, 2 Arginine, 3 Citrulline, 4
Leucine+Isoleucine, 5 Glutamine+Valine, 6 Methionine+Proline, 7 Alanine, 8 Threonine, 9
Asparagine+Tyrosine, 10 Serine, 11 Glycine, 12 Lysine, 13 Ornithine, 14 Lysine, 15 NBDF-excess, 16
Background, 17 Glutamate, 18 Aspartate. The released table merges peaks 3 and 4 into `LeuIleCit`,
drops 15 and 16, and adds `gshbiotin` (Fig. 4B "Biotin"; glutathione and biotin were both spike-in
standards, paper.md L125, and nothing else in the paper names the peak).

`PEAK_KEYS` (header -> stored key): `RPS19 -> lysine_related_peak1`, `arg1 -> arginine`,
`gshbiotin -> gshbiotin`, `NacOrn -> n_acetylornithine`, `LeuIleCit ->
leucine+isoleucine+citrulline`, `GlnVal -> glutamine+valine`, `MetPro -> methionine+proline`,
`Thr -> threonine`, `Ala -> alanine`, `Ser -> serine`, `Asn-Tyr -> asparagine+tyrosine`, `Gly ->
glycine`, `LysA -> lysine_a`, `Orn -> ornithine`, `LysB -> lysine_b`, `Glu -> glutamate`, `Asp ->
aspartate`. `lysine_a`/`lysine_b` keep the released labels because which of Fig. 1's lysine peaks
12 and 14 is A is not stated. `process()` raises on any other header set.

Legend vs values: the deposited legend reads "Supplemental Table 4. Log2 transformed ratios
representing fold change compared to average for each identified amino acid in each of nearly
4500 samples." and the Methods describe a log2-transformed ratio to the plate average. The released
values are the LINEAR ratio: no negative value anywhere, exact zeros (201 in `RPS19`, 236 in
`NacOrn`, 74 in `Ser`, 58 in `Asp`, 36 in `Orn`, 35 in `gshbiotin`, 23 in `Thr`), maxima up to 83,
and column means (present cells) `arg1` 1.004, `LeuIleCit` 1.005, `GlnVal` 0.997, `MetPro` 1.004,
`Ala` 0.997, `Asn-Tyr` 1.006, `Gly` 1.004, `LysA` 1.016, `LysB` 1.024, which is what a ratio to the
plate mean gives. Five columns have means above 1 with no stated reason: `RPS19` 2.225 (n 2,034),
`Glu` 2.151, `Asp` 1.987, `gshbiotin` 1.530, `NacOrn` 1.298. Stored as released,
`measurement_type = "ce_lif_peak_area_ratio_to_plate_mean"`, reference 1.0 per key,
`reference_centered=False` in `METABOLITE_DATASETS`; a released 0 is a value.

Missingness per column: `RPS19` 2,348 of 4,382, `gshbiotin` 1,502, `NacOrn` 1,016, `Asp` 361,
`Thr` 338, `Ser` 322, `Orn` 285, `Glu` 37, `LysB` 21, `MetPro` 16, `Asn-Tyr` 9, `arg1` 8,
`LeuIleCit` 8, `Gly` 8, `LysA` 6, `Ala` 4, `GlnVal` 3. No row is empty; every record carries
exactly its present keys and its reference the same keys at 1.0 (158 distinct references in the
build, one per distinct key set).

### Decisions, and where they departed from the plan

- `n_replicates = 1` for every key (conservative lower end). Tables 3A/3B were inspected for the
  per-record count the plan hoped for: after identifier normalization 4,123 Table 4 rows have
  their identifier in both 3A and 3B, 99 only in 3A, 158 only in 3B, and 2 (`YAL043C-A`,
  `YAR043C-A`) in neither; the 46 duplicated identifiers have the same multiplicity in 3A/3B for
  43 of them but cannot be paired row to row; and 3A/3B membership counts collected traces, not
  traces that passed the 0.35 template cut. The join is not exact, so the tables are not consumed.
- Identifier normalization before the resolver, all five recorded in the ledger: `YML048WA- ->
  YML048W-A`, `YJL038C  -> YJL038C`, `YLR228 C -> YLR228C`, `YML009c -> YML009C`, `YMR062 C ->
  YMR062C`. `perturbed_gene_name` is the normalized source name; the verbatim cell is in the ledger.
- Duplicate identifiers (the plan said "distinct records keyed by row (strain_id)"): 46 identifiers
  occur twice in the file and one more pair appears after normalization (`YJL038C` and `YJL038C`),
  47 second rows. `KanMxDeletionPerturbation` carries no `strain_id` (only
  `MarkerDeletionPerturbation` and the SGA leaves do) and the metabolite verifier's L1
  `genotype_uniqueness` keys on `(systematic_gene_name, perturbation_type, perturbed_gene_name)`
  with no `allow_duplicate` switch, so two rows of one deletion cannot both be served records
  without editing `schema.py` or `verification/metabolite.py`, both out of scope. The FIRST row in
  file order is served; the second is written with all of its values to `duplicate_strain_rows`
  in `preprocess/dropped_records.json`, never averaged. Follow-up: a strain_id-aware L1 (as
  `verification/fitness.py` has) plus a discriminator slot, then re-admit the 47 rows.
- Essentiality: the SGD store (`data/torchcell/gene_essentiality_sgd`, 1,140 genes, every record
  `is_essential=True` by construction of `sgd.py`) lists 21 Table 4 ORFs: ATG1, ATG2, ATG3, ATG5,
  ATG7, ATG8, ATG9, ATG10, ATG12, ATG15, ATG16, ATG17, ATG18, ERG24, HHT2, PLC1, RPS28A, SAC1,
  SPC72, VPS30, YMR185W, all viable deletion strains; the deletion-only Mulleder 2016 set has 28
  hits of the same kind. No canonical essential gene (CDC28, ACT1, TUB2, RPB1, FAS1, FAS2, GLC7,
  CMD1, CDC42) is in Table 4, so the tet-promoter (YSC1182) data are not in it. The instruction to
  route store hits to `excluded_non_deletion` was therefore not followed: the 22 flagged rows (21
  ORFs plus one RENAMED row onto a flagged gene) are KEPT and listed in `essentiality_flagged`;
  `excluded_non_deletion` has the rule for a parent-strain or tet row and count 0.
- `gshbiotin` identity gap: `ProvenanceGap.field` must name a real field that is `None`, so the
  gap lives on `target_metabolite_ids` (reason `deferred_pending_source_review`; the note names
  gshbiotin when the key is present). `metabolite_level_se` is a gap on every phenotype;
  `temperature` a gap on every environment.
- Medium: `COOPER_SC = dropout(SC, L-alanine, L-asparagine, L-cysteine, L-glutamine, glycine,
  L-proline, uracil)`, 25 components, `base_medium == "SC"`, liquid, loader-local; shared-rule
  verdict `derived` (below). The 15 stated uM values stay in the quote.
- Strain `BY4741` with `PARENT_STRAIN_QUOTE` carrying the paper's `BY4742`; duration 16 h; no
  temperature. `Publication(pubmed_id="20610602", doi="10.1101/gr.105825.110")`.

### Build (2026-09-15, `build_dataset_lmdb`, dev tree)

`BUILT AminoAcidCooper2010Dataset: 4313 records ... gene_set size 4291; references 158`.
Ledger `preprocess/dropped_records.json`: `n_source_rows` 4382, `n_kept` 4313, `n_values_kept`
67114, `n_dropped_orfs` 22 (8 `retired`: YAR043C-A, YCL006C x2, YCL013W, YCL026C, YDL061W,
YHR034W, YHR096W, YKL147W; 14 `non_gene_feature`: pseudogenes YFL056C, YIL170W, YCL074W,
YCL075W; blocked reading frames YOR031W, YER109C, YIL167W, YDR134C, YIL168W, YIR043C, YIR044C,
YER108C; transposable-element gene YJR026W), `n_renamed` 47 (e.g. `YPR090W -> YPR089W`,
`YML048W-A -> YML047W-A`), `n_normalized` 5, `n_duplicate_rows` 47, `n_excluded_non_deletion` 0,
`n_essentiality_flagged` 22. A first build raised on the gshbiotin gap field name and was retired
with `scripts/deprecate.sh` to `/scratch/projects/torchcell-deprecated` before the rebuild.

### Verification (Cooper entry only, `runners.run_metabolite`, verbatim)

```
0 structural PASS: 4313 records validated
1 count PASS: observed 4313, expected 4313
1 genotype_uniqueness PASS: 4313 unique strains (deletion sets), one record each
2 value_fidelity PASS: 67114 values checked
2 se_nonnegative PASS: 0 values checked
3 reference_finite PASS: reference level finite + key-subset for all 67114 values
3 measurement_type_consistent PASS: single measurement_type: 'ce_lif_peak_area_ratio_to_plate_mean'
4 gene_containment_scmd_ohya2005 PASS: 0.987 of scmd_ohya2005's 4291 deletion genes are in Ohya (>= 0.9)
amino_acid_cooper2010: PASS
```

Shared rules run ad hoc (`shared_rule_results` with the R64 resolver and gene set), verbatim:

```
1 provenance_gaps True 12939 documented provenance gaps over 4313/4313 records; 1 deferred field(s): ['target_metabolite_ids']; 254467 undeclared None values over 3 carrier fields (top: Compound.inchi x138016, Compound.chebi_id x112138, Environment.duration_generations x4313)
1 canonical_gene_names True 4291 systematic names, one canonical spelling each (22 merged-ORF aliases), each current in the genome
2 uncertainty_sanity True 0 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
3 compound_identity True environment edits: 0 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
3 media_compound_identity True medium components: 107825 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
3 media_membership True 0 records on a shared MEDIA_LIBRARY medium, 4313 on a medium deriving from one (1 distinct media)
4 gene_containment_sgd True 1.000 of 4291 measured genes are S288C reference genes (>= 0.9)
4 current_genome_genes True every one of the 4291 measured systematic names is a gene of the current genome
```

### Admission dry run (copy of the served manifest, dev data root)

`Admission check: AminoAcidCooper2010Dataset  ->  BLOCKED`. Report
`value_surface_changed = []` (no `media.py` drift from this branch), `novel_symbols = []`,
`dev_lmdb_status = fresh`, `in_adapter_map = True`. The three block reasons name drift between the
served commit `513cbfa1` and current `main` `53c869e9`, none of it from this branch: (1) 36 served
datasets' closures changed on `DoseBasis` (Nadal-Ribelles also `EnvironmentPerturbation`); (2) graph
class `environment perturbation` changed; (3) adapter drift in `_environment_node`, `_media_node`,
`_temperature_node` and their edge/reference methods (36 served datasets). A control run for
`Bloom2019Dataset` (landed on `main`, never served) against the same copy is BLOCKED with the same
three reasons. The served store is behind `main`; Cooper joins the next full rebuild, or an
increment once the store is rebuilt. The production manifest was not touched (sha256
`7cfae7e4b3ea5c5ff94486cb8788a946ba09a4b9de240531c5f1cc3295d781ea` before and after).

### Mulleder join

Exact on `arginine`, `alanine`, `threonine`, `serine`, `glycine`, `glutamate`, `aspartate`
(4,051 shared ORFs measured on the deposit). `lysine_a`/`lysine_b` need a peak rule;
`leucine+isoleucine+citrulline`, `glutamine+valine`, `methionine+proline`, `asparagine+tyrosine`
need an aggregation on the Mulleder side (a sum of the members' mM is the obvious candidate;
the author decides); `lysine_related_peak1`, `gshbiotin`, `n_acetylornithine`, `ornithine` have no
Mulleder partner. Cooper is a linear ratio to a plate mean, Mulleder an absolute mM, so the join
is by rank or by log ratio to each dataset's own center, never by value.

- [x] deposit, loader, adapter, registration, build, L0-L4 PASS, admit dry run recorded
- [ ] strain_id-aware L1 for metabolite datasets, then re-admit the 47 duplicate rows
- [ ] `target_metabolite_ids` (keys -> Yeast9 s_NNNN) from YeastGEM
- [ ] migrate `COOPER_SC` into `media.py` in the next full-rebuild window

## 2026.09.29 - Showcase page (group 3)

Shown on `docs/source/datasets/scerevisiae/amino-acid-betaxanthin.md` ([[experiments.034-showcase-datasets.scripts.amino_acid_betaxanthin]]). The dev store's content sha256 equals release `2026.09.21-ab6d8c5d`'s. Measured there: Spearman between Mulleder mM and the Cooper ratio over the shared genes is -0.006 to 0.059 on the seven single-amino-acid keys, and between the kept and ledger rows of the duplicated Table 4 identifiers (39 to 47 pairs) -0.021 to 0.133. The duplicate strains do not reproduce each other, which bears on how much a Cooper value can say about a gene.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518; the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes (#518); PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no (symlink); refused deposit leaving a directory: partial (a refused second file left the first deposited).

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `TABLE4_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.12 - Every Table 4 row is a record under its released identifier

Owner rule of 2026-10-12 (the data is the data; the genome is a separate layer), applied through the genome-gating sweep ([[database.genome-gating-sweep.2026.10.12]], rows 172 and 197) on branch `fix/genome-gate-reversals-yeast-2`.

- The shared resolver is gone from the loader. `KeptRow.systematic_gene_name` and `perturbed_gene_name` are both the normalized `Sysname` cell, verbatim. The 22 rows whose released ORF R64-4-1 carries as a non-gene feature (13: YFL056C, YOR031W, YER109C, YIL170W, YIL167W, YCL074W, YCL075W, YDR134C, YIL168W, YIR043C, YIR044C, YJR026W, YER108C) or no longer carries (9: YAR043C-A, YCL006C x2, YCL013W, YCL026C, YDL061W, YHR034W, YHR096W, YKL147W) are records like every other row; the 47 names R64 renamed (YAL043C-A to YAL042C-A and the rest of the former `renamed_orfs` list) are stored under the released name, not the current one. Whether a released name joins a gene node is decided at modeling time.
- The 46 identifiers on two rows and the one on three (47 extra rows) are all records. The release carries no plate, well or clone key, so the records of one identifier differ only in their values; `preprocess/dropped_records.json` lists them under `duplicate_identifiers`, and the metabolite verifier's L1 `genotype_uniqueness` reports the collisions as a finding (`allow_duplicate_strains`, set in the runner entry) rather than refusing the store.
- The load-time join against the local `gene_essentiality_sgd` store is gone (a cross-reference the rule forbids). The 21 ORFs it used to flag (ATG1, ATG2, ATG3, ATG5, ATG7, ATG8, ATG9, ATG10, ATG12, ATG15, ATG16, ATG17, ATG18, ERG24, HHT2, PLC1, RPS28A, SAC1, SPC72, VPS30, YMR185W) are viable deletion-collection strains whose SGD "inviable" annotation is conditional; they are stored as the release reports them, a deletion each.
- Expected records: 4,382 (every row; was 4,313). Values kept: the sum over all rows of their present peaks. The dev store is rebuilt under slurm with `--retire-existing --verify` from a `git clone --shared` snapshot of the branch rebased onto `main` at `f85df69af5` (#1069 landed: genome rules report, never gate), slurm array 4451; the L0-L4 table is appended below when the job finishes.
- Tests: `test_cooper2010.py` drives `read_table4` / `retain_rows` with no fake genome; the pseudogene, the retired name and the renamed name are kept under their released names, the second GAL1 row is record 2, and the ledger has no `dropped_orfs`, `renamed_orfs` or `essentiality_flagged` lists.

## 2026.11.24 - Per-replicate audit: Tables 3A/3B are released but could not be retrieved

Per-replicate policy (tracking #1178, follow-up #1281; #1203 item 9). Branch `fix/per-replicate-cooper2010`. No record changes: the store keeps one record per Table 4 row.

### What a replicate is (sourced)

- Biological: "Strains were screened in duplicate, starting from fresh yeast colonies." (`paper.md` L26, sha256 `b7739b1e...`). Each screen is its own overnight culture, extract and CE trace.
- Table 4 is the authors' average: "An average of the two replicates was calculated (Supplemental Table 4). In the case where only one quality trace was collected, those data were used alone." (L149).
- The per-replicate release is Supplemental Tables 3A/3B. Legend (`Supplemental_Table_Legends.doc`, sha256 `ca4983cd...`): "Supplemental Tables 3A and 3B. Integrated peak information for first (3A) and second (3B) replicate data collection.  For each row, the number of peaks identified is listed in column 3. The next four sets of columns correspond to height, %total height, area and %total area of each identified peak." The Methods call these the 32Karat integration export (L129). A row is one trace's peak list with no amino-acid label; the assignment is the authors' dynamic programming plus manual adjustment (L145), and no assignment table is named in the paper or legends.
- No withdrawn sample is quoted anywhere.

### Retrieval log, 2026-11-26 (generic user agent unless stated)

| URL | response |
|---|---|
| `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1101/gr.105825.110&format=json` | 200: PMC2928507, PMID 20610602 |
| `https://doi.org/10.1101/gr.105825.110` | 302 to `http://genome.cshlp.org/lookup/doi/10.1101/gr.105825.110` |
| `https://genome.cshlp.org/content/20/9/1288` | 403, `cf-mitigated: challenge` (Cloudflare "Just a moment...") |
| `https://genome.cshlp.org/content/suppl/2010/07/07/gr.105825.110.DC1` (and the `http://` form, 301 to it) | 403, `cf-mitigated: challenge` |
| `https://genome.cshlp.org/content/20/9/1288/suppl/DC1` | 403, `cf-mitigated: challenge` |
| `.../gr.105825.110.DC1/SupplementalTable3A-peaks.txt`, `3B-peaks.txt`, `ClusterInNorm.zip`, `SupplementalTable4.txt` | 403, `cf-mitigated: challenge` each; 3A also with a browser user agent |
| `http://www.genome.org/content/suppl/.../SupplementalTable3A-peaks.txt`, `http://www.genome.org/cgi/content/full/gr.105825.110/DC1` | two redirects to cshlp, then 403 challenge |
| Wayback CDX for every DC1 file name (3A, 3B, ClusterInNorm, Table 4, legends, Table 1, 2A) | no capture of any file |
| Wayback CDX `genome.cshlp.org/content/20/9/1288/suppl/DC1` | one capture, 20170710141614: the supplement page itself |
| `https://web.archive.org/web/20170710141614id_/http://genome.cshlp.org/content/20/9/1288/suppl/DC1` | 200, 23,235 B, sha256 `9942f5ef212e31e07abd534d44459c1148606883ee785e9e963d4b9d59f147dc`; lists 11 files: `k562Suppl.pdf`, `SupplementalTable1.xls`, `SupplementalTable2A-raw.txt`, `SupplementalTable2B-raw.txt`, `SupplementalTable3A-peaks.txt`, `SupplementalTable3B-peaks.txt`, `SupplementalTable4.txt`, `Supplemental_Table_Legends.doc`, `Theberge017574_SuppFig1.jpg`, `Theberge017574_SuppFig2.jpg`, `ClusterInNorm.zip` |
| `https://pmc.ncbi.nlm.nih.gov/articles/PMC2928507/` | 200; no supplementary-material section, only the pointer to `http://www.genome.org` |
| `https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC2928507` (and `PMC2928507.1/`, `oa_comm/xml/all/`, `oa_noncomm/xml/all/`, `author_manuscript/xml/all/`, `metadata/`) | 200, KeyCount 0; the same listing returns 15 keys for PMC10832289 (Rachwalski 2024), so the route works and the article is not in the set |
| `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC2928507` | 404 (also 404 for PMC10832289: the endpoint is gone) |
| `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2928507/supplementaryFiles` | 200 errorBean "Article with id PMC2928507 is not open access one" |
| `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC2928507/fullTextXML` | 500 |
| Europe PMC core search `PMCID:PMC2928507` | isOpenAccess N, hasSuppl N |

The paper and legends name no other repository. Local copies: the user's 2026-09-15 browser download held the whole supplement, staged at `/tmp/screenshots/AA_data/` and in a session scratchpad (`.../c9adb0ba-.../scratchpad/cooper2010/deposit/`, plan note 2026.09.15). Both are gone (the first is empty, the second does not exist), and neither graveyard (`/scratch/projects/deprecated`, `/bulk/deprecated`, `/bulk/torchcell-deprecated`) nor `/bulk/torchcell-raw`, `/bulk/torchcell-library` or `/bulk/zotero-library` holds a 3A/3B, 2A or `ClusterInNorm` file.

Outcome: blocked. Nothing was deposited and nothing substituted. The manual route that produced Table 4 (a browser session past the challenge, institutional login if asked) would also reach both files; that is the owner's call under #1281.

### Corrections

- **The raw manifest said "not deposited" for Tables 2A/2B and 3A/3B. Wrong:** the publisher lists all four. `SI_EXPECTED` in the loader now names every file on the page and its fate, and the raw mirror's `manifest.json` `si_expected` was rewritten to it (files and retrieval records untouched; manifest sha256 `c98694f7...` -> `91855806...`).
- **The 3A/3B join statistics are withdrawn** (#1203 item 9). The 2026.09.15 section's "4,123 Table 4 rows ... in both 3A and 3B, 99 only in 3A, 158 only in 3B, and 2 ... in neither", and "the 46 duplicated identifiers have the same multiplicity in 3A/3B for 43 of them", were measured on the staged copy, which was never mirrored, has no retrieval record and no longer exists. They cannot be reproduced and are removed from the loader docstring and `REPLICATE_RULE`; the old section stays as history.
- **Duplicate identifiers:** 48 identifiers sit on exactly two rows each (96 rows, 48 second rows), not "46 twice and one three times". 46 are verbatim repeats; `YML009C`/`YML009c` and `YML048W-A`/`YML048WA-` pair up after normalization (`YJL038C ` has no partner). The docstring and the runner comment are corrected; the ledger always counted 48.

### Code

- `cooper2010.py`: docstring `REPLICATES` paragraph; `TABLE3A_NAME`, `TABLE3B_NAME`, `DC1_LISTING_CAPTURE(_SHA256)`, `RetrievalAttempt`, `TABLE3_ATTEMPTS` (8 attempts), `SI_EXPECTED`, `PER_REPLICATE_RELEASE`, `REPLICATE_TABLES_LEGEND`; two new paper quotes (`replicate_tables_content` L129, `peak_assignment_algorithm` L145) and one legend quote (`replicate_tables_legend`); the build ledger gains `per_replicate_release` and `per_replicate_release_attempts`.
- Tests: the ledger and sourced-values tests take the new keys; the legend test checks both legend quotes; the deposit test checks `si_expected`; new `test_replicate_tables_are_named_in_si_expected_and_every_attempt` and `@pytest.mark.data test_real_table4_counts_and_raw_mirror_manifest` (4,382 rows, 4,382 records, 5 normalizations, 48 duplicate identifiers, 4,334 distinct, 68,202 values, 0 excluded).

### Build and verification

Scratch build `/bulk/torchcell-scratch-overflow/per-replicate/per-replicate-cooper2010/amino_acid_cooper2010`: 4,382 records. Metabolite verifier PASS: L0 4,382 validated; L1 count 4,382 = expected; L1 genotype_uniqueness 48 deletion sets in more than one record (finding); L2 68,202 values; L3 reference finite, one measurement type; L4 0.981 of the 4,334 genes in Ohya. Against the dev store (built 2026-10-16, read only): 4,382 of 4,382 records identical once the `replicate_id: null` field added by PR #1181 is set aside; max absolute value difference 0.0.

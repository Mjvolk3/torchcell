---
id: hm8r4vc3noy3hl5yafevztu
title: cooper2010-amino-acid-metabolome
desc: ''
updated: 1789455346929
created: 1789455346929
---

## Context

Cooper 2010 (Genome Res 20:1288-1296, doi `10.1101/gr.105825.110`, PMID 20610602, PMCID PMC2928507) profiled free amino-acid pools of the yeast deletion collection by capillary electrophoresis with laser-induced fluorescence (CE-LIF) after NBD-F derivatization: 18 assigned peaks per trace, each a log2 ratio of the peak's quantity to its plate average, duplicates averaged, released as Genome Research Supplemental Table 4. Mirror key `cooperHighthroughputProfilingAmino2010`; `paper.pdf` sha256 `54bf1aa6795dcefe6b8a8510cab51f79587ce5201bdd4e9183a28f64b60f48ec`; MinerU `paper.md` sha256 `b7739b1eeaf68b609a3e13b870f28649c331fe36b3c930bbbfd8a4c4fd19e52a` (both re-hashed 2026-09-15).

It becomes the 51st dataset: the second amino-acid metabolome on the deletion collection after Mulleder 2016 (`torchcell/datasets/scerevisiae/mulleder2016.py`), on a different analytical platform, which is the cross-platform replicate the paper triage named (`notes/paper.north-star.dataset-triage.md` L136; `notes/experiments.database.expansion-100.md` L224-225, "Cooper vs Mulleder", a transfer test none has run).

Deliverable, PR only: loader `AminoAcidCooper2010Dataset` (root `data/torchcell/amino_acid_cooper2010`) writing `MetaboliteExperiment` / `MetabolitePhenotype` records per deletion strain, raw mirror `$DATA_ROOT/torchcell-raw/cooperHighthroughputProfilingAmino2010/` with `manifest.json`, L0-L4 via `METABOLITE_DATASETS`, adapter + conf yaml, every registration surface, candidate/supported tables regenerated, tests, dendron notes, weekly child note, and a recorded `kg_manifest admit` verdict. No KG increment `sbatch`; rebuild timing is the orchestrator's. `DATA_ROOT=/scratch/projects/torchcell-scratch`.

**Hard gate** (measured 2026-09-15 02:02): the per-strain table is behind the Genome Research login wall. Every scripted attempt in the scratchpad `/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/cooper2010/` failed (`dc1_retry.html` is a `login-redirect?prompt=login&openid_connect_destination=/content/20/9/1288/suppl/D` page; `epmc_supp.zip` is 164 bytes and not a zip; `epmc_full.xml` is empty). The user is downloading the supplemental tables by browser into that directory. File names, format (xls/xlsx/csv), the identifier column, whether a replicate-count column exists, and whether BY4742 or tet-promoter rows are present are UNKNOWN. Every column-dependent step below is conditional on inspecting the file.

### Paper facts the loader hardcodes

Each is a `SourcedValue` whose `quote` must be a substring of the mirrored `paper.md` (every quote below hits exactly once with `grep -c -F`, re-verified 2026-09-15; line numbers are `paper.md` lines; the number-with-unit forms that FAIL as plain prose are in Gotcha 1):

- Collection: `Open Biosystems (YSC1053)` citing `Winzeler et al. 1999` (L109). Tet-promoter collection `Open Biosystems (YSC1182)` (L109), doxycycline protocol at L121.
- Parent: `Compared with the parent strain, BY4742` (L48) and again in the spike-in methods (L125); the same L109 sentence calls the YSC1053 strains `MATa`.
- Medium: the sentence beginning `Yeast growth was in synthetic complete media (adenine` (L109) lists 15 supplements (adenine, arginine, aspartic acid, glutamic acid, histidine, isoleucine, leucine, lysine, methionine, phenylalanine, serine, threonine, tryptophan, tyrosine, valine) and no uracil; alanine, asparagine, cysteine, glutamine, glycine and proline are absent. Serine 3.64, threonine 1.6 and valine 1.19 uM are two to three orders below the rest and were never corrected, so no stated concentration is written as a typed `Concentration` (Decision 2).
- Growth: `cells were grown for ${ \sim } 1 6 \mathrm { h }$ in synthetic complete media in deepwell 96 well plates (VWR).` (L113). No growth temperature anywhere; the only degree values are GC-MS derivatization and oven settings (L157).
- Replicates: `Strains were screened in duplicate, starting from fresh yeast colonies.` and `An average of the two replicates was calculated (Supplemental Table 4). In the case where only one quality trace was collected, those data were used alone.` (L149).
- Normalization: `-transformed ratio of the amino acid quantity in the sample to the average for the plate was calculated. This ratio corrects for plate effects.` (L149, preceded by `$\log _ { 2 }$`).
- Quantification: `quantification of peak area correlating to relative amino acid concentration` (L145); NBD-F labeling and CE separation in the Fig. 1 caption (L34).
- QC: `we eliminated data for which correlation with the template was below 0.35.` (L145). Counts: `We analyzed 4382 samples with data meeting quality standards for at least one replicate.` (L42); median OD600 `0.370` (L26); `all 52 traces` aligned to a master template (L129).
- Peaks: `does not separate from valine` (glutamine, L62 region); `Asparagine + tyrosine` and `Glutamine + valine` are Table 1 row labels (L69, an HTML table). The full 18-peak assignment is in the Fig. 1 image under `images/`, not in text. The scout's reading of that image, UNCONFIRMED until the JPEG is read (Gotcha 2): 1 lysine-related, 2 arginine, 3 citrulline, 4 leucine+isoleucine, 5 glutamine+valine, 6 methionine+proline, 7 alanine, 8 threonine, 9 asparagine+tyrosine, 10 serine, 11 glycine, 12 lysine, 13 ornithine, 14 lysine, 15 NBDF-excess, 16 background, 17 glutamate, 18 aspartate.

## Relevant Files

| path | action | purpose | stance |
|---|---|---|---|
| `torchcell/datasets/scerevisiae/cooper2010.py` | NEW | `AminoAcidCooper2010Dataset`: mirror-linked `download`, Table 4 parse, resolver retention, loader-local derived SC medium, per-strain `MetaboliteExperiment`, drop/exclusion ledgers | n/a |
| `torchcell/adapters/cooper2010_adapter.py` | NEW | `AminoAcidCooper2010Adapter`, clone of `mulleder2016_adapter.py` L16-50 pointing at its own yaml | n/a |
| `torchcell/adapters/conf/amino_acid_cooper2010_adapter.yaml` | NEW | copy of `amino_acid_mulleder2016_adapter.yaml` (same node/edge enable-list; metabolite phenotype methods) | n/a |
| `tests/torchcell/datasets/scerevisiae/test_cooper2010.py` | NEW | quote-verbatim, medium-derivation, key-set, resolver-ledger, manifest-hash tests on a synthetic Table 4 | n/a |
| `tests/torchcell/adapters/test_cooper2010_adapter.py` | NEW | conf parses, every enabled method exists, adapter points at its own conf + dataset | n/a |
| `$DATA_ROOT/torchcell-raw/cooperHighthroughputProfilingAmino2010/` | NEW | `data/<Table 4 file>` + `manifest.json` (`manual_browser`, DC1 URL, user's date, sha256) | n/a |
| `notes/torchcell.datasets.scerevisiae.cooper2010.md` | NEW | loader note: provenance, key set as inspected, ledgers, admit verdict, Mulleder-join rule | n/a |
| `notes/torchcell.adapters.cooper2010_adapter.md` | NEW | adapter note (conf is the whole surface) | n/a |
| `notes/user.Mjvolk3.torchcell.tasks.weekly.2026.38.cooper2010-amino-acid-metabolome.md` | NEW | weekly child note (precedent `...2026.38.genomes-tier.md`; no 2026.38 parent exists) | n/a |
| `torchcell/datasets/scerevisiae/__init__.py` | MODIFY | import (near L39), `cooper_datasets = [...]` (near L96), `+ cooper_datasets` in `__all__` (near L150) | undocumented |
| `torchcell/adapters/__init__.py` | MODIFY | import (L60-62 shape), append to `proteome_metabolome_adapters` (L160-165) | undocumented |
| `torchcell/knowledge_graphs/dataset_adapter_map.py` | MODIFY | adapter import (L4), dataset import (L99), map entry beside `AminoAcidMulleder2016Dataset` (L161) | stable |
| `torchcell/verification/runners.py` | MODIFY | `METABOLITE_DATASETS` entry `amino_acid_cooper2010` after Mulleder (L460-470), `reference_centered=True`, `expected_count` set after deposit | stable |
| `experiments/database/scripts/build_supported_datasets_table.py` | MODIFY | `CuratedRow(section="Metabolite", ...)` beside Mulleder (L115) | stable |
| `experiments/database/scripts/build_candidate_datasets_table.py` | MODIFY | `BUILT_COUNT = 51` (L50); Cooper row L438-456 renamed CE-MS to CE-LIF, `status="built"`, `why` extended; band key L3909 and synergy key L4581 renamed to match | stable |
| `notes-tex/database-expansion-100/tables/*.tex` | MODIFY | regenerated by the candidate script (never hand-edited, `notes-tex/database-expansion-100/Makefile` L11-12) | n/a |
| `notes/torchcell.knowledge_graphs.dataset-admission-loop.md` | MODIFY | dated section recording the admit dry run, precedent "2026.09.12 - The 50th dataset through the gate" (L114) | stable |
| `notes/experiments.database.expansion-100.md` | MODIFY | dated section: Cooper moves from candidate to built; CE-MS label corrected | stable |
| `notes/torchcell.datasets.scerevisiae.mulleder2016.md` | MODIFY | dated cross-reference: the Cooper join and its aggregation rule | stable |
| `torchcell/datasets/scerevisiae/mulleder2016.py` | REFERENCE | `AMINO_ACIDS` L79-99 (lowercase names), `process` L152-222 (header-mismatch `RuntimeError` L160), regex-only ORF filter L176-186, `create_experiment` L223-276, inline `Media(name="SM", ...)` L241; no resolver, no raw mirror | stable |
| `torchcell/datasets/scerevisiae/hoepfner2014.py` | REFERENCE | `_KEPT_STATUSES` L318, `_sv` L348 + the sourced-value dict after it, `ProvenanceGap` gaps L505-535, `raw_mirror_dir` L539, `deposit_raw_mirror` L545-603, `load_manifest` L606, `DroppedOrf` L630-645, `__init__(genome=...)` L906-935 (`overwrite=False` comment L930), `download` symlink+sha256 L956-981, resolver retention loop L1248-1250 | stable |
| `torchcell/datasets/scerevisiae/costanzo2021.py` | REFERENCE | the only `manual_browser` precedent: `MANUAL_RECIPE` L490-496, `RetrievalRecord(method=manual_browser, retriever="manual", params={"retrieval_command": ...})` L541-546 | stable |
| `torchcell/datasets/scerevisiae/bloom2019.py` | REFERENCE | manifest with `data/`, `code/`, `paper/` subdirs L532-549, `zip_member` retriever L581, `manifest_sha256` L679 | stable |
| `torchcell/datasets/scerevisiae/cachera2023.py` | REFERENCE | population-centered reference `metabolite_level={...: 0.0}` L333-337 | stable |
| `torchcell/literature/manifest.py` | REFERENCE | `sha256_file` L37, `RetrievalMethod.manual_browser` L77, `RetrievalRecord` L90-110 (`extra="forbid"`), `ArtifactRecord` L126, `Manifest` L139-176 (`si_data_sources`, `si_expected`, `provenance_complete`) | stable |
| `torchcell/datamodels/media.py` | REFERENCE | `dropout` L203-261, `_SC_AMINO_ACIDS` L646-667 (canonical `L-` names), `SC` L726-782 (escaped-LaTeX quote L755), `_hm_dropout` L811-840, `MEDIA_LIBRARY` L1440; NOT edited | stable |
| `torchcell/verification/common.py` | REFERENCE | `SharedRecordRules` L234, `_media_verdict` L428-443, `shared_rule_results` L807-823 | stable |
| `torchcell/verification/metabolite.py` | REFERENCE | `_l3_reference_zero` L96, `_l3_measurement_type_consistent` L156, `verify_metabolite_dataset` L172-233 (`allow_nan=False` L206) | stable |
| `torchcell/verification/sourced.py` | REFERENCE | `SourcedValue` L43, `ProvenanceGapReason` L131, `ProvenanceGap` L147 | stable |
| `torchcell/datamodels/schema.py` | REFERENCE | `ReferenceGenome` L25, `KanMxDeletionPerturbation` L265, `Genotype` L953, `ProvenanceGapMixin` L1114, `Environment` L1542 (`temperature: Temperature or None, default None` L1556), `MetabolitePhenotype` L2613-2669 (key-set and `>= 1` validator L2653-2662); NOT edited | in-flux |
| `torchcell/database/build_dataset_lmdb.py` | REFERENCE | refuses an existing `processed/lmdb` L59-64; injects `genome` when `__init__` has the param L66-75 | stable |
| `database/slurm/scripts/gilahyper_build_dataset_lmdb.slurm` | REFERENCE | `sbatch --export=ALL,DATASET_CLASS=AminoAcidCooper2010Dataset ...` | stable |
| `torchcell/knowledge_graphs/kg_manifest.py` | REFERENCE | `VALUE_SURFACE_RELPATHS` L139 (media.py is in it), `admit` subcommand L1311-1327 | stable |
| `/scratch/projects/torchcell/database/kg_manifest.json` | REFERENCE | served manifest; admit runs against a COPY | n/a |

## Key Design Decisions

1. **The deposit is step 1 and the retrieval record is genuine `manual_browser`.** The DC1 URL `http://genome.cshlp.org/content/suppl/2010/07/07/gr.105825.110.DC1` redirects to a login page for any client (measured, `dc1_retry.html`), so no `torchcell.literature.retrieve` function can reproduce the bytes. Follow `costanzo2021.py` L490-496 and L541-546 exactly:
   - `RetrievalRecord(method=RetrievalMethod.manual_browser, source_url=<DC1 URL>, retriever="manual", params={"retrieval_command": <recipe naming the browser steps, the file, and its sha256>}, sha256=<measured>, retrieved_at=<the user's date>)`. `retrieved_at` dates the browser action, not the deposit; ask for it, because a wrong value cannot be corrected without a new manifest version.
   - `Manifest(si_data_sources=[DC1 URL], si_expected=["Supplemental Table 1", "Supplemental Tables 2A,B", "Supplemental Tables 3A,B", "Supplemental Table 4"], provenance_complete=True)`: the recipe is what a rebuild re-runs by hand.
   - Files go under `data/` (Bloom layout L532-549). Only Table 4 is recorded in `files` even if the user downloads every table; the others are named in `si_expected`. If the download is a bundle, deposit the member and record the container sha256 in `params` as Bloom's `zip_member` does.

2. **Derived SC medium built INSIDE the loader module, not in `media.py`.** `media.py` is in `VALUE_SURFACE_RELPATHS` (`kg_manifest.py` L139-140): editing it makes `admit` report value-surface drift and blocks incremental admission, and it trips the ontology-figure pre-commit gate (`.pre-commit-config.yaml` L56-63).
   - `_media_verdict` (`common.py` L428-443) accepts `base_medium` naming a library key as `derived:SC`, and `dropout()` (`media.py` L203-261) copies `base_medium` from `SC`, so a module-level `COOPER_SC = dropout(SC, "L-alanine", "L-asparagine", "L-cysteine", "L-glutamine", "glycine", "L-proline", "uracil", name="SC minus Ala/Asn/Cys/Gln/Gly/Pro/Ura (Cooper 2010 synthetic complete recipe)", provenance=[_sv(...)])` passes L3 and joins Mulleder's and Hillenmeyer's media at `SC`. State is `liquid` (inherited from `SC`; growth is in deep-well plates).
   - This is the first loader-local derived medium (Hillenmeyer's `_hm_dropout` L811-840 lives in `media.py`); the loader note records why, and the follow-up is to migrate it into `media.py` in the next full-rebuild window.
   - Rejected: an inline free-text `Media(name="SC", ...)` like Mulleder L241 (joins nothing, `free_text` verdict); a `media.py` constant (blocks admission). The 15 stated concentrations stay in the quote only, since three of them are uncorrected outliers (Context) and typing any would assert a recipe the paper does not support.

3. **Metabolite keys are what was measured: one key per assigned peak, Mulleder's lowercase spelling where a peak is a single amino acid.** Composite peaks are composite keys (`leucine+isoleucine`, `glutamine+valine`, `methionine+proline`, `asparagine+tyrosine`); the paper says glutamine `does not separate from valine`, and splitting would invent a measurement. Singles: `arginine`, `citrulline`, `alanine`, `threonine`, `serine`, `glycine`, `ornithine`, `glutamate`, `aspartate` (the last two match Mulleder L81-82). Lysine has three peaks (1 "lysine-related", 12, 14) and stays distinct by peak number (`lysine_related_peak1`, `lysine_peak12`, `lysine_peak14`) because the paper reports them separately (L62 correlates "lysine peak 1" and "peak 2" with OD600 independently). `NBDF-excess` (15) and `Background` (16) are not metabolites and are never keys.
   - The FINAL set is decided on Table 4's column headers: a module constant `PEAK_KEYS: dict[str, str]` maps each header verbatim to its key, and `process()` raises `RuntimeError` listing unmapped and missing headers on any mismatch (Mulleder L160 pattern).
   - Consequence for the promised Cooper-vs-Mulleder join: exact on the 7 shared singles (`arginine`, `alanine`, `threonine`, `serine`, `glycine`, `glutamate`, `aspartate`); lysine needs a peak-selection rule; the four composites need an aggregation rule on the Mulleder side (e.g. log2 of the summed mM of the two members) that the note states and the author decides; `citrulline` and `ornithine` have no Mulleder partner. `target_metabolite_ids=None`, deferred as Mulleder L249.

4. **Reference is identically 0.0 log2 per key; `reference_centered=True`.** The value is a log2 ratio to the plate mean, so the population center is the natural zero (Cachera L333-337) and `_l3_reference_zero` (`metabolite.py` L96) is the right L3. `measurement_type = "ce_lif_log2_ratio_to_plate_mean"`.
   - A BY4742 row, if Table 4 carries one, is NOT the reference (the reference is the plate mean by construction) and cannot be a `MetaboliteExperiment` in a deletion dataset (a `Genotype` is a list of gene perturbations, `schema.py` L953; a wild type has none), so it goes to the `excluded_non_deletion` ledger with its values, where the author can read it.
   - `n_replicates` per key is read from Table 4 if it carries a per-strain replicate count; otherwise the conservative lower end `1` (some strains had "only one quality trace"), never `2`.
   - `metabolite_level_se=None` plus `ProvenanceGap(field="metabolite_level_se", reason=not_reported_by_primary, note="duplicates were averaged; no per-strain spread is released")` on the phenotype (Hoepfner L505-535; `MetabolitePhenotype` inherits `ProvenanceGapMixin` through `Phenotype`).

5. **Resolver retention with a ledger; identifier column not assumed.** `__init__(..., genome=None)` as Hoepfner L906-935 so `build_dataset_lmdb.py` L66-75 injects the genome; `resolve_gene_name` on whatever the identifier column holds (ORF or standard name; the resolver accepts both). `CURRENT` kept; `RENAMED` kept under the current systematic name with `perturbed_gene_name` = source cell verbatim; `NON_GENE_FEATURE`, `RETIRED`, `AMBIGUOUS` to a local `DroppedOrf` ledger (defined in `cooper2010.py`; importing Hoepfner's would couple the loaders and its field description names Hoepfner's column) written to `preprocess/dropped_records.json` with counts logged. Rejected: Mulleder's regex-only filter L176-186 (keeps retired ORFs as if current).

6. **Tet-promoter (YSC1182) strains are out of scope.** They are essential-gene knockdowns under doxycycline, not KanMX deletions, and their protocol differs (L121). If Table 4 marks them (a collection or strain-type column, or a `tet`/`TH_` prefix), they go to `excluded_non_deletion` with the count logged; `expected_count` in `runners.py` is set to the kept count after the first build. If Table 4 does NOT distinguish them, that is an open question for the author, not a heuristic; the ad hoc diagnostic is the fraction of kept genes absent from Ohya's deletion set (L4 containment, `runners.py` L611), which tet strains would inflate.

7. **`ReferenceGenome(strain="BY4741")` per the deliberation, with the BY4742 sentence preserved as a flagged quote.** The paper names the parent `BY4742` twice (L48, L125) and says the YSC1053 strains are `MATa`; the loader stores a module `SourcedValue` `PARENT_STRAIN_QUOTE` whose `note` states the discrepancy so a reader sees it without re-reading the paper. `KanMxDeletionPerturbation` (`schema.py` L265) is justified by the Winzeler 1999 citation of the deletion project; the marker is not stated in this paper and the note says so.

8. **Temperature is a typed absence, never 30 C.** `Environment.temperature` is optional (`schema.py` L1556): `temperature=None, provenance_gaps=[ProvenanceGap(field="temperature", reason=not_reported_by_primary, note="growth temperature is not stated; the only degree values are GC-MS oven settings")]`. `duration_hours=16.0` with the `${ \sim } 1 6 \mathrm { h }$` quote and a note that the tilde is the source's.

9. **QC threshold and OD are note material, not filters.** The 0.35 template-correlation cut and the 0.370 median OD600 describe what the authors already removed or observed (L145, L149: Table 4 is the averaged post-QC output). There is no `qc_flags` field on any schema class (grep 2026-09-15), so they are `SourcedValue` constants in the module and paragraphs in the loader note; no record is dropped on them.

10. **No verifier, schema, media, `cell_adapter.py` or yaml edits, so `admit` can say ADMISSIBLE.** `verify_metabolite_dataset` has no resolver or `SharedRecordRules`; wiring them into `run_metabolite` is a follow-up, but `shared_rule_results` (`common.py` L807-823) is run ad hoc on Cooper's records and the output goes in the note.

11. **Candidate table: rename to CE-LIF, `status="built"`, `BUILT_COUNT = 51`.** "CE-MS" is wrong (detection is laser-induced fluorescence of NBD-F adducts; the GC-MS in the paper is a validation experiment). The name is a dict key in three places (row L439, band L3909, synergies L4581), so all three change together (Gotcha 7). Bloom's built row (L348-366) is the template for the `why` suffix.

## Approach

Execution order, gated.

1. **Deposit gate.** Wait for the file(s) in the scratchpad `cooper2010/` directory; do not proceed on the failed-retrieval HTML there now. Ask the user for the retrieval date and which link they clicked. `sha256sum` every downloaded file; the hash of the Table 4 file becomes `TABLE4_SHA256`.
2. **Table 4 inspection.** Read with pandas: `pd.read_excel` (openpyxl 3.1.5) for `.xlsx`; `engine="xlrd"` for legacy `.xls` only, since xlrd 2.0.2 reads nothing else; `read_csv` otherwise. Print `df.columns`, `df.shape`, `df.head()`, `dtypes`, the identifier column's regex profile (how many cells match `Y[A-P][LR]\d{3}[CW](-[A-Z])?$`), any column that looks like a replicate count or strain type, and whether `BY4742`, `wild`, `tet` or `TH` occur in any string column. Decide `PEAK_KEYS`, the identifier column, the replicate rule and the exclusion rule from that output; paste the printout into the loader note.
3. **Loader `cooper2010.py`.** Module constants (citation key, DOI, `PAPER_MD_SHA256`, `TABLE4_NAME`, `TABLE4_SHA256`, `DC1_URL`, `MANUAL_RECIPE`, `MEASUREMENT_TYPE`, `PEAK_KEYS`); the `_sv` helper and sourced-value dict (Hoepfner L348 onward); `COOPER_SC` via `dropout(SC, ...)`; `raw_mirror_dir` / `deposit_raw_mirror(source_path, retrieved_at, data_root)` / `load_manifest` (Hoepfner L539-612 with the costanzo2021 `manual_browser` record); `DroppedOrf` + `ExcludedRow` + `DroppedRecordReport` pydantic ledgers; the dataset class (`experiment_class`, `reference_class`, `raw_file_names=[TABLE4_NAME]`, `download` = symlink from the mirror after a sha256 check with NO network path, raising with `MANUAL_RECIPE` when the mirror is absent); `process()` under `@post_process` writing `preprocess/data.csv`, the ledger JSON and the LMDB (Mulleder L152-222 shape); `create_experiment(row)` in Mulleder's L223-276 shape with Decisions 2, 4 and 8 substituted: `Environment(media=COOPER_SC, temperature=None, duration_hours=16.0, provenance_gaps=[the temperature gap])`, a phenotype whose `metabolite_level` and `n_replicates` dicts are keyed by `PEAK_KEYS.values()` with the SE gap attached, a reference phenotype of 0.0 per key under the same `n_replicates` rule, `environment_reference=environment.model_copy()`, and `Publication` from the PMID/DOI in Context.
4. **Run.** `deposit_raw_mirror` once by hand, the build CLI, the verifier, the ad hoc shared rules, the admit dry run (commands in Verification). Set `expected_count` from the kept count and re-run the verifier.
5. **Surfaces.** Adapter + yaml, the three registrations, the two table scripts and regenerated `tables/*.tex`, tests, the loader and adapter notes, the admission-loop and expansion-100 dated sections, the Mulleder cross-reference, the weekly child note.

Out of scope: any edit to `media.py`, `schema.py`, `pydant.py`, `cell_adapter.py`, `verification/metabolite.py`, `verification/common.py`; the resolver/SharedRecordRules wiring of `run_metabolite`; `target_metabolite_ids` (Yeast9 mapping); the Cooper-vs-Mulleder transfer experiment itself; the `classes.tex` wording of the join (a follow-up for the author); the KG increment `sbatch`.

## Gotchas

1. **MinerU rewrote every number-with-unit into LaTeX, so most "verbatim" quotes from the scout brief are NOT substrings of `paper.md`.** Measured: `Yeast growth was in synthetic complete media (adenine 140` 0 hits, `log2-transformed ratio` 0 hits, `serine 3.64` 0 hits, `52 plates` 0 hits (the text says `all 52 traces`). The recipe sentence reads `synthetic complete media (adenine $1 4 0 ~ \mu \mathrm { M }$ , arginine $1 0 9 \ \mu \mathrm { M } ,$ ...` with inconsistent spacing per item. Sidestep: copy each quote from `paper.md` with the Read tool, keep the LaTeX (escape backslashes in Python as `SC`'s quote does, `"$2 \\%$"`, `media.py` L755), prefer the shortest unambiguous span that carries the fact (for the recipe, the prefix through `adenine $1 4 0 ~ \mu \mathrm { M }$` and a second `SourcedValue` for the tail `and valine $1 . 1 9 ~ \mu \mathrm { M } )$`), and pin every one with the Hoepfner-style test `test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper` (`test_hoepfner2014.py` L263-279). Do NOT source quotes from `pdftotext -layout` of `paper.pdf`: it renders the micro sign as `m` (`adenine 140 mM`), which would falsify the units.

2. **The 18-peak assignment is inside a figure image.** `Lysine-related` and `Citrulline` as peak labels are not in `paper.md` prose. Sidestep: anchor the composite keys to the Table 1 row-label quotes and the glutamine/valine sentence, anchor the peak numbering to the Fig. 1 caption sentence `(Lysine has two peaks as it has two reactive amine groups and therefore can be labeled once or twice.)` (L34), and read the Fig. 1 JPEG in `images/` with the Read tool to confirm the 1-18 order before fixing `PEAK_KEYS`. Table 4's own column headers override all of this and are the loud-failure check.

3. **`dropout()` raises on a name that is not a component of `SC`, and `SC`'s amino acids are `L-` prefixed (`_SC_AMINO_ACIDS` L646-667) except `glycine`.** `resolved_compound` maps a source spelling to the canonical name, but confirm at test time that `dropout(SC, "L-cysteine", ...)` and `dropout(SC, "cysteine", ...)` both resolve; a `ValueError` at import would break every registration import. Put the `COOPER_SC` construction under a test that also asserts `COOPER_SC.base_medium == "SC"` and `len(COOPER_SC.dropouts) == 7`.

4. **`build_dataset_lmdb.py` refuses to run if `processed/lmdb` exists** (L59-64) and the base class silently reuses it. A wrong first build (bad `PEAK_KEYS`) must be moved with `/deprecate` (`trash` is not installed on GilaHyper), never `rm`, before the rebuild.

5. **`SharedRecordRules` matches media by equality or by `base_medium`.** A loader-local `Media` is never equal to a library dump, so the verdict is `derived:SC`, the intended outcome; if the ad hoc run reports `free_text`, `base_medium` was lost (e.g. a `model_copy(update=...)` that dropped it) and the medium must be rebuilt through `dropout()`.

6. **`n_replicates` values must be >= 1 for every key and the key sets must match** (`MetabolitePhenotype` validator L2653-2662). If Table 4 has per-peak missing cells (a peak dropped by the template match for one strain), the record cannot carry that key with `NaN` (L2 forbids NaN, `metabolite.py` L206). Decide per Table 4: either drop the key from that record's dicts (per-record key sets may differ; `_l3_measurement_type_consistent` only checks the type string) or drop the strain; log the count either way and state the rule in the note.

7. **The candidate name is a dict key in three places** (Decision 11). Renaming only L439 leaves L3909 (band) and L4581 (synergies) orphaned, which the script reports as a missing band (`band_why` is required out of the scale band, L4997) or silently drops the synergy rows. Rename all three, regenerate, and diff `tables/swaps.tex`: the previous-pass ordering is computed in-file (`scale_key` L216-223), so a rename can appear as a removal plus an addition; if it does, say so in the note rather than hiding it.

8. **The adapter yaml must not gain methods the `CellAdapter` lacks** (test precedent `test_every_enabled_method_exists_in_the_cell_adapter_tables`, `test_hoepfner2014_adapter.py` L54). Copy Mulleder's yaml byte for byte; the environment carries no perturbation, so no `environment perturbation` methods.

9. **`admit` must run against a COPY of the served manifest with the DEV data root**, `--data-root /scratch/projects/torchcell-scratch`, because the LMDB is in the dev tree and the served `kg_manifest.json` is not to be touched by a dry run. A `[BLOCK]` naming `media.py` means someone edited the value surface in this branch; revert it.

10. **The `genome` parameter opens `SCerevisiaeGenome(overwrite=False)` at build time.** `overwrite=True` races any other process holding the gffutils database (Hoepfner L930 comment); keep `overwrite=False` and never build two genome-injected loaders concurrently on one machine.

## Verification

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/plan/cooper2010-amino-acid-metabolome
export PYTHONPATH=$PWD
# unit tests (synthetic Table 4; the quote test skips without the mirror)
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/datasets/scerevisiae/test_cooper2010.py -xvs
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/adapters/test_cooper2010_adapter.py -xvs
~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/adapters tests/torchcell/knowledge_graphs -x -q  # registration surfaces stay green (Mulleder has no tests of its own)
# lint + types (strict mypy is a pre-commit gate)
~/miniconda3/envs/torchcell/bin/ruff check torchcell/datasets/scerevisiae/cooper2010.py torchcell/adapters/cooper2010_adapter.py tests/torchcell/datasets/scerevisiae/test_cooper2010.py tests/torchcell/adapters/test_cooper2010_adapter.py
~/miniconda3/envs/torchcell/bin/mypy torchcell/datasets/scerevisiae/cooper2010.py torchcell/adapters/cooper2010_adapter.py
# deposit (once, by hand, after the user's download; record the printed sha256 in the note)
~/miniconda3/envs/torchcell/bin/python -c "from torchcell.datasets.scerevisiae.cooper2010 import deposit_raw_mirror; print(deposit_raw_mirror(source_path='<scratchpad>/cooper2010/<Table 4 file>', retrieved_at='<user date>'))"
sha256sum /scratch/projects/torchcell-scratch/torchcell-raw/cooperHighthroughputProfilingAmino2010/data/*
# build (dev tree; refuses if processed/lmdb exists)
~/miniconda3/envs/torchcell/bin/python -m torchcell.database.build_dataset_lmdb --dataset AminoAcidCooper2010Dataset
# or: sbatch --export=ALL,DATASET_CLASS=AminoAcidCooper2010Dataset database/slurm/scripts/gilahyper_build_dataset_lmdb.slurm
# verifier: METABOLITE_DATASETS entry, L0-L3 + L4 Ohya containment; report lands in preprocess/
~/miniconda3/envs/torchcell/bin/python -c "from torchcell.verification.runners import run_metabolite; import os; raise SystemExit(0 if run_metabolite(os.environ['DATA_ROOT']) else 1)"
# ad hoc shared rules (media verdict must be derived:SC; gene rules against the R64 gene set)
~/miniconda3/envs/torchcell/bin/python -c "
import os; from torchcell.verification.runners import load_records
from torchcell.verification.common import shared_rule_results
recs = load_records(os.path.join(os.environ['DATA_ROOT'], 'data/torchcell/amino_acid_cooper2010'))
for r in shared_rule_results(recs): print(r.level, r.name, r.passed, r.message)"
# admit dry run on a COPY of the served manifest
cp /scratch/projects/torchcell/database/kg_manifest.json /scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/kg_manifest.copy.json
~/miniconda3/envs/torchcell/bin/python -m torchcell.knowledge_graphs.kg_manifest --manifest /scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/kg_manifest.copy.json admit --dataset AminoAcidCooper2010Dataset --data-root /scratch/projects/torchcell-scratch --report /scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/cooper2010_admit.json
# tables
~/miniconda3/envs/torchcell/bin/python experiments/database/scripts/build_candidate_datasets_table.py
~/miniconda3/envs/torchcell/bin/python experiments/database/scripts/build_supported_datasets_table.py --max-gb 3
# registration smoke
~/miniconda3/envs/torchcell/bin/python -c "from torchcell.datasets.scerevisiae import AminoAcidCooper2010Dataset; from torchcell.adapters import AminoAcidCooper2010Adapter; from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map as m; print(m[AminoAcidCooper2010Dataset])"
```

Expected: every test green; ruff and mypy clean; the build logs `kept / dropped / excluded_non_deletion` counts and writes `preprocess/dropped_records.json`; the verifier report passes L0-L4 with `expected_count` equal to the kept count (set it in `runners.py` from the first build, then rebuild the report); the shared-rules media verdict is `derived:SC`; `admit` prints ADMISSIBLE with `value surface: unchanged`; `git diff --stat notes-tex/database-expansion-100/tables/` shows `counts.tex` at 51 built and `candidates.tex` without the Cooper candidate row; the ontology-figure gate does NOT fire (no `media.py`/`schema.py` in the diff). Record the admit verdict, the counts, and the shared-rules output in the loader note and the admission-loop note.

## Open Questions

1. **The deposit.** Which file(s), from which link, on which date, in which format; the sha256 of each. Nothing past Approach step 1 starts before this is answered.
2. **Table 4 layout.** Column headers (peak labels vs amino-acid names vs numbers 1-18), whether `NBDF-excess`/`Background` columns are present, the identifier column (ORF, standard name, or Open Biosystems clone id), and the row count against `4382`.
3. **Replicates.** Whether Table 4 carries a per-strain replicate count (or two replicate columns whose presence gives it). If not, `n_replicates = 1` everywhere with the conservative-lower-end rule stated in the note; the author may prefer a per-strain rule if the raw Tables 2/3 expose it.
4. **BY4742 and tet-promoter rows.** Whether either is present and how they are marked. If tet strains are present but unmarked, the author decides between keeping essential-gene rows as deletions (wrong) and dropping by an external essential-gene list (not in the mirror).
5. **Missing cells.** Per-record key dropping vs strain dropping (Gotcha 6); "quality standards for at least one replicate" suggests per-peak gaps exist.
6. **The Mulleder join rule** for lysine (which of three peaks) and the four composites; the `classes.tex` wording is the author's.
7. **Strain label.** `BY4741` per the deliberation vs `BY4742` per the paper; the flagged quote is in the loader either way, and a one-line change flips it.
8. **`expected_count`** is set after the first build; 4382 counts samples meeting QC before resolver drops and tet/BY4742 exclusions, so it is not guessable from the paper.

## 2026.09.15 - Deposit inspected: what Supplemental Table 4 actually holds

The user downloaded the whole supplement by browser (institutional login; every scripted
path returned 429/403/404) into `/tmp/screenshots/AA_data/` on 2026-09-15; it is staged
unchanged with `SHA256SUMS.txt` and `RETRIEVAL.txt` at
`/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/c9adb0ba-07ff-4030-910b-ed8ce6d42537/scratchpad/cooper2010/deposit/`.
Files: `SupplementalTable4.txt` (874,149 B, sha256
3c56cd492b4cab51249358e90c7e9731982960eb619e45b8850093e616a471fb), `Supplemental_Table_Legends.doc`
(ca4983cd...), `SupplementalTable1.xls`, `SupplementalTable2A-raw.txt` / `2B` (66 and 67 MB
raw traces), `SupplementalTable3A-peaks.txt` / `3B` (peak integrations), `ClusterInNorm.zip`
(three Cluster files), plus `Theberge017574_SuppFig1.jpg`, `SuppFig2.jpg` and `k562Suppl.pdf`,
which carry another manuscript's names and are NOT deposited into the raw mirror.

Measured on the file (pandas, tab-separated, `-` = missing):

- 4,382 rows x 19 columns: `Sysname`, `NAME`, then 17 peak columns named by the authors'
  template, not by Fig. 1: `RPS19`, `arg1`, `gshbiotin`, `NacOrn`, `LeuIleCit`, `GlnVal`,
  `MetPro`, `Thr`, `Ala`, `Ser`, `Asn-Tyr`, `Gly`, `LysA`, `Orn`, `LysB`, `Glu`, `Asp`. The
  key set in Decision 3 must follow these 17 columns. `RPS19` is the "lysine-related
  metabolite" that accumulates in rps19a/rps19b strains (paper.md L88, L95); `gshbiotin` is
  explained nowhere in the paper or legends and keeps its raw label with a typed gap;
  `NacOrn` reads as N-acetylornithine and `LeuIleCit` as leucine+isoleucine+citrulline, both
  to be confirmed against Fig. 1 and the legends before naming.
- 4,334 distinct ORFs; 46 rows repeat a `Sysname` with different values (e.g. `YBR020W GAL1`
  twice, `YBR009C HHF1` twice): replicate strains, kept as distinct records keyed by row
  (strain_id), never averaged. 5 identifiers are malformed by whitespace or case
  (`YJL038C`, `YLR228 C`, `YML009c`, `YMR062 C`, `YML048WA-`); normalize whitespace and case
  and the `WA-` suffix to `W-A` before the resolver, and record each normalization.
- The legend says "Log2 transformed ratios"; the values are NOT log2: per-column medians 0.9
  to 1.5, minima exactly 0 (201 zeros in `RPS19`, 236 in `NacOrn`, 74 in `Ser`), maxima up to
  83. These are linear ratios to the plate average. Store as released with
  `measurement_type` naming a linear ratio to the plate mean, reference 1.0 per key,
  `reference_centered=False` (L3 `reference_finite`), and record the legend contradiction as
  a sourced note; a zero is a released value, not a missing one.
- Missingness: `RPS19` 2,348 of 4,382 missing, `gshbiotin` 1,502, `NacOrn` 1,016, `Asp` 361,
  `Thr` 338, `Ser` 322, `Orn` 285; the major peaks miss fewer than 40. Each record's key set
  is therefore ragged; `n_replicates` keys must equal the record's present keys.
- Overlap: 1,313 ORFs shared with Kemmeren 2014, 4,051 with Mulleder 2016. 21 ORFs are in
  the SGD essentiality gene set; the implementer checks their essentiality value and
  routes true essentials (tet-promoter strains) to the `excluded_non_deletion` ledger.
- Tables 3A/3B carry per-replicate peak integrations, so whether a strain had one or two
  quality traces IS recoverable per record; `n_replicates` per record comes from them if the
  join is exact, else the conservative 1.

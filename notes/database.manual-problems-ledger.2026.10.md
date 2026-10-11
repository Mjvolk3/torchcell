---
id: f8ggw115pr8gnbm35aws6yh
title: Manual Problems Ledger 2026.10
desc: ''
updated: 1791693231988
created: 1791693231988
---

## 2026.10.10 - Every problem only the owner can solve, before KG 4.0

The owner's request: "I want to get to the end where we know all the manual problems that need to be solved. I will then try to solve them." This ledger is that list, one row per problem, each traced to the issue or PR line it comes from. Umbrella issue: #919 (`by-hand`), one checkbox per row.

**What counts as a row.** A problem an agent may not or cannot solve: a retrieval behind a wall that needs a browser, a Zotero write (agents never write to Zotero), a data decision a PR or issue explicitly left to the owner (not one taken by recommendation), a credential, or a build and disk step reserved for the owner. Decisions already taken by recommendation, and open issues an agent can fix, are listed at the end as checked and excluded, so the list can be audited for completeness.

**Sources read, all of them.** Every open issue labeled `by-hand`, `dataset` or `before-next-kg-build` (39, from `gh issue list --state open --label <l> --limit 200`); the 33 issues closed since 2026-10-08 and their comments (`gh issue list --state closed --search "closed:>=2026-10-08"`); the bodies of the 64 PRs closed since 2026-10-08 12:00 UTC (`gh pr list --state closed --limit 100`); the program memory `bacteria-expansion-program-state`; [[database.kg-4-pre-build-verification]]; the `blocked` rows and `EXCLUDED` list of `experiments/database/scripts/build_bacteria_candidate_datasets_table.py` (`ranked()` at `edd48664ae`; "candidate rank" below is a position in that order); [[plan.dataset-admission-pipeline.2026.10.10]]; and `python -m torchcell.candidates ledger --print` (two verdicts, both `admissible`, no blocked gate, so it contributes no row).

**Verification.** Every issue's state was read on 2026-10-10 between 23:00 and 23:45 CDT, and a solved one is not restated. Machine-side numbers (dev-store record counts, Zotero coverage of the 56 bacterial citation keys, deposits present, disk usage, PubChem reachability) come from `experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements.py` ([[experiments.036-dataset-fixes-before-kg-build.scripts.manual_problems_ledger_measurements]]), run at 23:43 and again at 23:45 CDT on `edd48664ae` with the same results. Record counts are LMDB entry counts of the dev stores; a count quoted from an issue or PR is marked by its quote. Fifteen decisions that lived only in PR bodies or closed issues were filed as #904 to #918 with the options exactly as their source stated them.

**The three that unblock the most records.**

1. #691, attach the Nichols 2011 PDF: a loader the candidate table estimates at 1,289,196 instances (a product of dimensions, not a count).
2. #726, the compound-identity curator (a go-ahead, plus a non-throttled PubChem host only if GilaHyper is throttled again): 94,640 Borchert 2024 records dropped on the `RCH2_defined_noCarbon` medium (#860), all 47,320 of Rand 2017's among them.
3. #910, file Teteneva 2024 in Zotero and then choose the W3110 annotation route: 66,162 reported RB-TnSeq fitness values.

#788 item 3 (Tong 2020's CarPE table) touches more records, 111,420, but fills a typed gap on records already stored rather than adding records.

### Count by kind

| kind | problems | largest unblock in the kind |
|---|---:|---|
| walled retrieval | 3 | #788 (item 3): fills the carbon-source concentration, a typed gap on all 111,420 stored records |
| Zotero curation | 6 | #691: the Nichols 2011 loader: the candidate table estimates 1,289,196 instances (a product of dimensions, not a count); Shiver 2016 already holds a re-scored Nichols batch-0 block of 835,337 measured values, deliberately not loaded |
| owner data decision | 21 | #726: 94,640 Borchert 2024 records dropped on the RCH2 medium, among them all 47,320 of Rand 2017's (#860); identities for the 276 bacterial gap labels counted in PR #838 |
| infrastructure/credential | 1 | #726: the same 94,640 records as the go-ahead row; the curator re-queries all 5,532 table rows |
| build/ops | 9 | #916: serves every mapped dataset, including all bacterial stores; resolves the four drift notices |
| **total** | **40** | |

### Walled retrieval (3)

| id | dataset or row | program | owner action | unblocks | agent step afterwards |
|---|---|---|---|---|---|
| #788 (item 3) | Tong 2020, E. coli carbon sources (`ecoli_carbon_source_tong2020`) | bacteria | Open CarPE's 'carbon conditions' tab in a browser, copy or export the table, deposit it with a `DEPOSIT.md` and `SHA256SUMS.txt` | fills the carbon-source concentration, a typed gap on all 111,420 stored records | consume the deposit as a `manual_browser` artifact, fill the concentration in `tong2020.py`, rebuild the store with `--verify` |
| #853 | Balakrishnan 2022 (`mrna_fraction_balakrishnan2022`) | bacteria | Download the six NIHMS1856867 supplement files from PMC in a browser and deposit them | Table S6 (about 2,700 genes' mRNA degradation rates, promoter on-rates, translation initiation rates; no counterpart on main); Table S4 proteomics (content not measured); the SI's media and Table S1 strain genotypes behind 28 stored records; likely settles #906 | record the deposit in the raw manifest, measure Tables S4 to S7, type the media and strains in `balakrishnan2022.py`, rebuild with `--verify` |
| #699 | Ishii 2007 (`metabolome_ishii2007`, `proteome_ishii2007`, `flux_ishii2007`) | bacteria | Download the Science supporting-online-material PDF in a browser and deposit it under the raw mirror's `si/` | fills the chemostat medium, temperature and flux-fitting procedure, typed gaps on 84 stored records (28 per store) | record the SOM in the raw manifest, OCR it (`lit_capture_si.py --ocr-recorded` pattern), replace the three typed gaps in `ishii2007.py`, rebuild the three stores with `--verify` |

#### #788 (item 3): Tong 2020, E. coli carbon sources (`ecoli_carbon_source_tong2020`)

Source (#788, item 3): "A Shiny app: a scripted GET returns only its "Please Wait" loader page, measured 2026-10-07 ... The concentration is a typed provenance gap on the stored records until it is mirrored."

https://edbrownlab.shinyapps.io/CarPE/

Deposit under `$DATA_ROOT/torchcell-raw/tongGeneDispensabilityEscherichia2020/data/carpe/` (the `data/dryad/` and `data/benchling/` precedent of 2026-10-09: the file as it arrived, `DEPOSIT.md` saying how, `SHA256SUMS.txt`). If the app offers no export, paste the rendered table as tab-separated text and say so in `DEPOSIT.md`, as was done for Yunus 2026.

#### #853: Balakrishnan 2022 (`mrna_fraction_balakrishnan2022`)

Source (#853, Recipe): "Open https://pmc.ncbi.nlm.nih.gov/articles/PMC9804519/ in a browser (the proof-of-work page clears itself), download the six supplement files, and deposit them under `$DATA_ROOT/torchcell-raw/balakrishnanPrinciplesGeneRegulation2022/` with `RetrievalMethod.manual_browser` and the sha256 of the bytes that arrive"

https://pmc.ncbi.nlm.nih.gov/articles/PMC9804519/

Files and targets, from #853: `si/NIHMS1856867-supplement-Supplementary_Information.pdf`; `data/NIHMS1856867-supplement-Table_S4.xlsx`, `..._Table_S5.xlsx`, `..._Table_S6.xlsx`, `..._Table_S7.xlsx`; `data/NIHMS1856867-supplement-Table_S3.xlsx` (to check byte identity against the GEO copy already mirrored). Measured 2026-10-10: the raw mirror holds only `GSE205717_Processed_data_Table_S3.xlsx` and the PMC article text.

#### #699: Ishii 2007 (`metabolome_ishii2007`, `proteome_ishii2007`, `flux_ishii2007`)

Source (`si_expected` of `$DATA_ROOT/torchcell-raw/ishiiMultipleHighThroughputAnalyses2007/manifest.json` (PR #814)): "MANUAL RECIPE: open https://www.science.org/doi/10.1126/science.1132067 in a browser, follow 'Supplementary Materials', download the supporting-online-material PDF, then deposit it under si/ with RetrievalMethod.manual_browser and the sha256 of the bytes that arrive. It is the only source of the chemostat medium, the temperature and the flux-fitting procedure"

https://www.science.org/doi/10.1126/science.1132067

https://www.science.org/doi/suppl/10.1126/science.1132067

Deposit to `$DATA_ROOT/torchcell-raw/ishiiMultipleHighThroughputAnalyses2007/si/`. Measured 2026-10-10: neither mirror holds an `si/` for this key. The other three papers #699 names need no download: Tian 2019 releases no SI at all (PR #815), and the Thompson 2020 and Schmidt 2022 SIs were marked unneeded because both rows are subsumed by Borchert 2024. Hypothesis (unmeasured): the aggregation-decomposition audit in flight on `docs/aggregation-decomposition-policy` may reopen those two, since it loads a compendium's original studies from their own releases.

### Zotero curation (6)

| id | dataset or row | program | owner action | unblocks | agent step afterwards |
|---|---|---|---|---|---|
| #691 | Nichols 2011 (candidate rank 10) | bacteria | Attach the article PDF (and SI PDFs if available) as a stored file on group item `REEL3RAD` in `database/Escherichia-coli` | the Nichols 2011 loader: the candidate table estimates 1,289,196 instances (a product of dimensions, not a count); Shiver 2016 already holds a re-scored Nichols batch-0 block of 835,337 measured values, deliberately not loaded | nightly `lit_sync` captures and OCRs it; then a loader plus the correlation check against Shiver's batch-0 block (`notes/plan.bacteria-si-phenotype-audit-ecoli.md`, rank 2) |
| #910 | Teteneva 2024, E. coli W3110 lake water (candidate rank 40) | bacteria | Add Teteneva et al. 2024 (PMC11188689) with its PDF to `database/Escherichia-coli`, or say 'close the row as deferred' | the Teteneva 2024 loader: 66,162 RB-TnSeq fitness values reported for Table S4 | `lit_sync` mirrors it; `pmc_cloud` fetches Tables S3/S4 into the raw mirror; then the annotation-route decision (row below) and the loader |
| #788 (item 4) | Banerjee 2024 isoprenol (candidate rank 79) and Kang 2026's reference titer | bacteria | Say go-ahead and add Banerjee 2024 (doi 10.1016/j.ymben.2024.02.004) with its PDF to `database/Pseudomonas-putida` | the Banerjee 2024 row (candidate estimate 171 instances) and Kang 2026's 7 fed-batch titer records, refused today for want of the reference titer Kang defers to Banerjee 2024 | `lit_sync` mirrors it; the Kang 2026 loader builds its reference through the deferral chain; a Banerjee 2024 loader |
| #911 | nine raw-mirror-only papers (Balakrishnan 2022, Brunk 2016, Li 2014, Mohiuddin 2022, Thompson 2019 valerolactam, Wang 2024, CeCaFDB, Rand 2017, Thompson 2019 lysine) | literature | Add each with its PDF to the group library, or decide raw-mirror-only is enough | no records (loaders read the raw mirror); gives each paper a `\cite` key, a `torchcell-library/` OCR copy and `tc-lit` access | `lit_sync` captures them; re-run `candidate-gate` on Balakrishnan 2022 and Li 2014 so their verdicts read `zotero_item: present` |
| #644 | group library item pair `8WK5B5MP` / `Y6J4Q887` | literature | Merge or re-key the two group items sharing citation key `chaoPredictingDynamicExpression2025` | no records; it made the SI re-OCR job (slurm 3241, #607) refuse at its Zotero precheck on 2026-10-02 | re-run the #607 SI re-OCR job |
| #624 | personal item with key `yun$On$ConnectionsAre2020` | literature | Regenerate or edit the Better BibTeX citation key so it has no `$` | no records; a `\cite` copied from Zotero resolves again | none; the nightly `lit_bib_store` warning stops |

#### #691: Nichols 2011 (candidate rank 10)

Source (#691): "**Owner, by hand:** attach the article PDF as a stored file on the group item (and the SI PDFs if available), then nothing else is needed."

https://doi.org/10.1016/j.cell.2010.11.052

Measured 2026-10-10: no `nicholsPhenotypicLandscapeBacterial2011` directory in `$DATA_ROOT/torchcell-library/`.

#### #910: Teteneva 2024, E. coli W3110 lake water (candidate rank 40)

Source (PR #841, Open decisions for the owner, 1): "**File Teteneva 2024 in Zotero, or close the row as deferred.** Everything else waits on this. It is the one thing an agent must not do."

https://pmc.ncbi.nlm.nih.gov/articles/PMC11188689/

#### #788 (item 4): Banerjee 2024 isoprenol (candidate rank 79) and Kang 2026's reference titer

Source (#788, comment of 2026-10-08 (correction to item 4)): "Banerjee 2024 is still not mirrored, and it is still the paper Kang 2026 defers to for the only baseline titer it states ... adding Banerjee 2024 is a **new curation decision**"

https://doi.org/10.1016/j.ymben.2024.02.004

#### #911: nine raw-mirror-only papers (Balakrishnan 2022, Brunk 2016, Li 2014, Mohiuddin 2022, Thompson 2019 valerolactam, Wang 2024, CeCaFDB, Rand 2017, Thompson 2019 lysine)

Source (PR #816 (Mohiuddin 2022); plan decision D5): "**Adding the item is the library owner's to do; nothing here writes to Zotero.**"

#### #644: group library item pair `8WK5B5MP` / `Y6J4Q887`

Source (#644): "Two top-level items in the TorchCell group library (6582362) carry the same Better BibTeX citation key `chaoPredictingDynamicExpression2025`, item keys `8WK5B5MP` and `Y6J4Q887`."

#### #624: personal item with key `yun$On$ConnectionsAre2020`

Source (#624; still logged by the 2026-10-10 03:49 CDT run (`/tmp/torchcell-lit-bib-store.log`)): "The fix is in Zotero (regenerate or edit the citation key of that item), which is the owner's by-hand work"

### Owner data decision (21)

| id | dataset or row | program | owner action | unblocks | agent step afterwards |
|---|---|---|---|---|---|
| #726 | compound-identity table rows for bacterial labels; Borchert 2024 `RCH2_defined_noCarbon` (#860) | bacteria | Say go-ahead for the curator to add table rows (iron(II) chloride and valerolactam, PR #876; the RCH2 salts and Wolfe's mixes, #860; pinene, D-alanine, L-malate, #726 comments), or build KG 4.0 with the typed InChIKey gaps | 94,640 Borchert 2024 records dropped on the RCH2 medium, among them all 47,320 of Rand 2017's (#860); identities for the 276 bacterial gap labels counted in PR #838 | run the curator, check `compound_identity_table_diff.py` shows no served renames, then #860's media entry and the affected dev-store rebuilds |
| #896 | Kuzmin 2018 trigenic interactions (`tmi_kuzmin2018`) | yeast | Choose: record the one-sided selection (Dataset node or provenance note) and keep the store as released, or source the unselected triples | 91,111 records gain a stated selection (91,110 negative tau, 0 positive) | annotate the dataset (or source the triples) and re-run the #889 verifier |
| #910 | Teteneva 2024 W3110 annotation route (`ecoli_K12_W3110_ASM1024v1`) | bacteria | After the paper is filed: read W3110 RefSeq-primary (`Y75_RS` tags) or add a symbol-plus-`ECK:JW:b` ingest route | the same 66,162 values (gene keys for every record) | read Table S4's identifier column, implement the chosen route in the genomes tier, write the loader |
| #913 | Borchert 2023 significance triples (`rbtnseq_borchert2023`) | bacteria | Open the fitness-contrast phenotype follow-up, or leave the triple refused | 64,853 (gene, comparison) contrasts carrying 259,412 significance values | measure first whether #880's p-value pair on `EnvironmentResponsePhenotype` expresses a contrast without misfiling; then the loader extension and store rebuild |
| #902, #903 | SynthLethDB yeast synthetic lethality and rescue (`syn_leth_db_yeast`, `syn_rescue_db_yeast`) | yeast | Choose the typed-gap carrier for 'medium not carried': `ProvenanceGapMixin` on `Media`, or `Environment.media` optional | 20,938 records (13,996 + 6,942) stop failing L3 `media_membership` | the schema change (touches every adapter that reads `media.state`), both stores rebuilt, the #889 verifier re-run |
| #879 | Wang 2018 Supplementary Data 5 (`crispri_guide_fitness_wang2018`) | bacteria | Choose: load SD5 with the FDR only and record Z and FPR as not stored, or add carriers for Z and FPR first | 20,158 gene-level rows (4,142 + 4,004 x 4) | source the FDR definition from the paper, then the loader arm (and the carriers, if chosen), store rebuild |
| #908 | Thompson 2019 lysine inside Borchert 2024 (`rbtnseq_borchert2024`) | bacteria | Choose: attach `thompsonMassivelyParallelFitness2019` as `SourceStudy` of the four samples or not; serve set6IT057's vessel as the compendium's microplate or the Methods' culture tubes | 18,928 records change publication (and possibly vessel) | edit Borchert 2024 attribution, rebuild the 1,372,280-record store under slurm |
| #909 | Royet 2025 Table S4 HMM essentiality | bacteria | Choose: load ES vs not-ES as a second `GeneEssentialityPhenotype` class (GD/GA grades discarded), or leave Table S4 unloaded | at most 11,458 records (5,729 genes x 2 growth stages, arithmetic on the PR's dimensions) | new class, adapter, verdict, pins, L0 to L4 |
| #901 | SGD gene essentiality (`gene_essentiality_sgd`) | yeast | Choose: keep only unconditional whole-gene null annotations, or carry allele and condition into genotype and environment; and a sourced medium or a typed gap in place of the guessed YEPD/30 C | 1,329 records (24 exact duplicates, 24 'essential' genes that grow as viable deletions in Costanzo 2016) | loader change, store rebuild, #889 verifier re-run |
| #912 | InhibitorBioscreenVolk2021 (private) in the KG 4.0 release snapshot | KG build | Choose before the snapshot is cut: add a visibility field (changes the release JSON schema), or list the private dataset like a public one | 977 private records' listing in the public `served_counts.md` | snapshot field, regenerated schema, flipped pinned test; or close |
| #887 | Xue 2025 in-house FFA strains (`ffa_xue2025`) | yeast | Name the allele the lab's RGR1, MED4 and TFC7 strains carry, or approve dropping those records with a logged reason | 112 of 176 records stop asserting an inviable full deletion; Xue's L4 containment (0.769) can pass | store the allele leaf (or the logged drop), rebuild, re-run its verifier |
| #904 | Brunk 2016 SRM proteome (`proteome_brunk2016`) | bacteria | Choose: DHSB keyed to b0724 on a trypsin cut-site assumption or still refused; FUMA and TKT1 keep their paralog-shared UniProt keys or are refused | 3 protein keys across the 81 stored records | peptide-route edit, store rebuild with `--verify` |
| #915 | Menasalvas 2025 pathway genes (`isoprenol_selection_menasalvas2025`) | bacteria | Choose: map the pIY670 part tokens to organisms from Sheet 4's UniProt names, make `source_organism` optional with a `ProvenanceGap`, or keep the `"unreported"` sentinel | 58 records carrying the five genes | schema change or loader mapping, store rebuild |
| #894, #895, #897 | Kuzmin 2018 DMF/TMF and Kuzmin 2020 DMF released SDs of exactly 0 | yeast | Choose: a typed gap on the uncertainty of these rows, or keep the SE of 0 | 42 records (4 + 5 + 33) stop claiming perfect precision | loader change in `kuzmin2018.py` / `kuzmin2020.py`, store rebuilds, #889 verifier re-run |
| #906 | Balakrishnan 2022 strains (`mrna_fraction_balakrishnan2022`) | bacteria | Wait for #853's Table S1, name a source, or leave the xylR cassettes and NQ393's GDH-null allele untyped | 28 records' genotypes (NQ1243 and NQ1390 split; NQ393 typed) | loader typing, store rebuild |
| #914 | Caglar 2017 Table S8 fold changes (`protein_fold_change_caglar2017`) | bacteria | Choose: keep the 8 records citing Caglar with the Houser 2015 share in the ledger, or refuse the family | 8 records holding 33,268 fold changes (kept today) | close, or remove the class and re-derive pins |
| #905 | Rousset 2018 S1 padj (`ecoli_crispri_rousset2018`) | bacteria | Choose: source the DESeq2 adjustment method anyway, attach padj to Cui 2018's records (refused by measurement in PR #883), or close | 0 Rousset records (none carries a growth-screen value); up to 54,326 Cui 2018 records under the attach option | mirror the DESeq2 reference, or a Cui loader change, or close |
| #907 | Tian 2019 isopentenol CRISPRi (candidate rank 17) | bacteria | Close as unloadable, or accept nothing | 0 records (no SI exists; every titer is a bar height; the 4 prose values are percent improvements) | record the choice in the expansion note and, once verdicts exist, the candidate store |
| #907 | Wang 2022 P. putida isoprenoids (candidate rank 19) | bacteria | Close as unloadable, or accept nothing | 0 records (titers are figures; 19 prose order statistics with no uncertainty; EDD answers 502) | as for Tian 2019 |
| #800 | Li 2021 mevalonate flux (candidate rank 50) | bacteria | Accept the refusal (keep the row as a candidate with a release defect) and close #800 | 0 records (the control's flux map is absent; one of three maps carries another strain's constraints) | none beyond closing; the next 13C-MFA row with a released control map (Ishii 2007 is loaded) carries the flux family |
| #918 | Supplementary figure `FigS-dataset-admission-loop`, panel c | paper | Go-ahead (or not) to redraw panel c and revise its caption to the Agent-tool orchestration | no records | `/drawio-diagram` or `/fig-swap`, caption through `/paper-edit`, rebuild `editing-nature-biotech.pdf` |

Source lines, verbatim:

- **#726** (PR #876, Not done; PR #838, Owner decision 2: "Build with those typed InChIKey gaps, or wait."): "`iron(II) chloride` and `valerolactam` rows in `compound_identity_table.json` (the issue's second gap): curation, left for an explicit go-ahead."
- **#896** (#896): "Owner decision: record the selection (e.g. on the Dataset node or as a provenance note) and keep the store as released, or source the unselected triples."
- **#910** (PR #841, Open decisions for the owner, 2): "Either the set is read RefSeq-primary with `Y75_RS` tags, or the ingest gains a symbol-plus-`ECK:JW:b` route. Decide against the identifiers Table S4 reports."
- **#913** (PR #836; `notes/torchcell.datasets.pputida.borchert2023.md` (2026.10.09)): "Recommendation recorded for the owner: open a follow-up for a fitness-contrast phenotype with a significance triple, and close #776's structural half against it."
- **#902, #903** (#902 and #903): "Fix direction (owner decision, open since #622): a typed gap on the medium (`ProvenanceGapMixin` on `Media`, or `Environment.media` optional), which touches every adapter that reads `media.state`."
- **#879** (#879): "So this needs an owner decision: load SD5 with the FDR only and record Z and FPR as not stored, or add carriers for them first."
- **#908** (PR #856, Owner decisions open (not taken)): "Should these four samples carry `SourceStudy` attribution to this paper in the Borchert loader? The change rewrites the publication on 18,928 records and needs a full rebuild of the store."
- **#909** (PR #865, Open): "`GeneEssentialityPhenotype` is boolean: ES vs not-ES would be honest but discards the GD/GA grades, and it would be a second dataset class. Owner call."
- **#901** (#901): "Fix direction (owner decision): keep only unconditional whole-gene null annotations as essentiality, or carry the allele and condition into the genotype and environment; replace the guessed medium with a sourced one or a typed gap."
- **#912** (#827, comment on PR #843): "whether the release snapshot should mark a private dataset (today it lists one exactly like a public one, which the public served-count docs page inherits)"
- **#887** (#887): "What allele was used for RGR1, MED4 and TFC7 in the Xue 2025 strains? ... If it cannot be recovered, should those 112 records be dropped with a logged reason rather than served as impossible deletions?"
- **#904** (#872, closing comment): "Two decisions remain for the owner before this closes: whether DHSB may key on a trypsin cut-site assumption (it would key to b0724), and whether FUMA and TKT1 keep their UniProt keys though a paralog (fumB, tktB) shares their peptides."
- **#915** (PR #844; PR #835 (source_organism left open)): "Acting on it would change every served `IsoprenolSelectionMenasalvas2025Dataset` record, which is a full-rebuild case needing an explicit go-ahead."
- **#894, #895, #897** (#894 and #895; #897's third rule is the same pattern): "Fix direction (owner decision): store a typed gap on the uncertainty for these rows rather than an SE of 0."
- **#906** (PR #884, Sourcing and the open decisions): "The xylR driver cassettes (Ptet in NQ1243, lacIq promoter in NQ1390). No mirrored source states their location ... NQ393's GDH-null allele. No source states whether it is a deletion or a point lesion."
- **#914** (PR #834): "**Owner decision wanted.** The alternative is to refuse the whole family on the grounds that a mixed-provenance denominator cannot be attributed, which would discard 33,268 fold changes a published table releases."
- **#905** (#878, closing comment): "Whether to source the DESeq2 adjustment method anyway, or attach padj to Cui's records, stays an owner decision."
- **#907** (#826): "| 17 | Tian 2019 | blocked: the paper has no SI, 96 values are bar heights, 4 prose values are percent improvements (commit 2aad5357b) | owner decision: close as unloadable, or accept nothing |"
- **#907** (#826): "| 19 | Wang 2022 | blocked: SI holds only media and strain tables, titers are figures, strains live in the JBEI ICE registry (commit ba31f240f) | owner decision as above |"
- **#800** (#826; its last comment lists Li 2021 among "Still owner decisions"): "| 50 | Li 2021 | refused (#800): the control's flux map is absent and the maps are model fits | nothing unless the authors release the control map |"
- **#918** (`notes/plan.dataset-admission-pipeline.2026.10.10.md`, decision 14): "the figure and caption change go through `/drawio-diagram` or `/fig-swap` after the author's go-ahead, because the backmatter carries status chips."

### Infrastructure/credential (1)

| id | dataset or row | program | owner action | unblocks | agent step afterwards |
|---|---|---|---|---|---|
| #726 | the compound-identity curator's PubChem host | bacteria | Only if PubChem answers 429 from GilaHyper again: open a Delta ssh session with Duo (or name another host) so the curator can run there | the same 94,640 records as the go-ahead row; the curator re-queries all 5,532 table rows | run the curator from GilaHyper first; fall back to Delta over the open Duo session |

Source lines, verbatim:

- **#726** (PR #838, Owner decision 2): "**#726, the compound-identity curator pass, is still blocked.** PubChem returned HTTP 200 once at 04:05:13 and 429 on the next four probes; 276 bacterial gap labels remain and 0 of them resolves to a row today."
  Measured 2026-10-10 at 23:26 CDT by hand and at 23:43 CDT by `manual_problems_ledger_measurements.py --probe-pubchem`: 6 of 6 sequential PubChem PUG REST name lookups from GilaHyper answered HTTP 200 both times. The wall may have lapsed at a low request rate; whether the curator's full pass stays under the throttle is not measured. Delta needs Duo for a fresh ssh, and BatchMode ssh from GilaHyper works only while the owner's Duo session is open (program memory `cluster-rules-igb-delta`, `025-graph-reg-round2-launch`).

### Build/ops (9)

| id | dataset or row | program | owner action | unblocks | agent step afterwards |
|---|---|---|---|---|---|
| #916 | KG 4.0 (135 mapped datasets on `edd48664ae`, 134 public + 1 private) | KG build | Decide the build point, cut the `DB(kg)` tag, submit `gilahyper_live_rebuild-slurm_docker.slurm` from a worktree at the tag | serves every mapped dataset, including all bacterial stores; resolves the four drift notices | before: array 3733's last task, `--list-stale --include-private` to 0, sweep re-runs, hand preflight; after: re-validate drifted queries, `scripts/kg_compat_page.py` |
| #840 | supported query `solid_growth_025` | KG build | After KG 4.0: re-validate against the new release (or deprecate) and close by hand | no records; restores the query's guarantee over 13 Costanzo/Kuzmin datasets, SGD essentiality and SynthLethDB | `python -m torchcell.knowledge_graphs.supported_queries validate solid_growth_025 --release <new>` |
| #848 | supported query `expression_proteome_morphology` | KG build | After KG 4.0: re-validate (or deprecate) and close by hand | no records; 5 datasets' query guarantee | `supported_queries validate expression_proteome_morphology --release <new>` |
| #847 | supported query `amino_acid_betaxanthin` | KG build | After KG 4.0: re-validate (or deprecate) and close by hand | no records; 3 datasets' query guarantee | `supported_queries validate amino_acid_betaxanthin --release <new>` |
| #842 | supported query `essentiality_smf` | KG build | After KG 4.0: re-validate (or deprecate) and close by hand | no records; 2 datasets' query guarantee | `supported_queries validate essentiality_smf --release <new>` |
| #917 (item 3) | retired dev-store copies under `$DATA_ROOT/data/torchcell/` and `/scratch/projects/torchcell-deprecated` | KG build | Purge by hand, September `deprecated-*` first, never while a build or rebuild reads the tree | no records; 869 G on `/scratch` (508 G `*.superseded.*` in 838 dirs, 293 G `deprecated-*` in 69, 68 G graveyard) | none |
| #917 (item 1) | `/db/database/biocypher-out/2026-10-09_07-20-33` (owned by root) | KG build | Delete the orphan CSV directory of failed job 3547 (as root or uid 7474) | no records; 170 G on `/db` | none |
| #917 (item 2) | `/db/database/data.superseded.2026-10-06_04-41-16` (KG 2.0, uid 7474) | KG build | Keep, or delete (backed up under `/bulk/kg-releases/2026.09.17-7715ee35`) | no records; 135 G on `/db` | none |
| #917 (item 4) | worktree `feat/ecoli-ishii2007` (one uncommitted test edit) and branch `backup/pre-rebase-836` | KG build | Discard or keep each | no records | none |

Source lines, verbatim:

- **#916** (owner, 2026-10-09 23:10 CDT (program memory)): "don't start rebuild yet because we want to add more datasets before we continue."
- **#840** (#840): "After the next KG build, re-validate with `python -m torchcell.knowledge_graphs.supported_queries validate solid_growth_025 --release <new release>` (or deprecate the query) and close this issue by hand."
- **#848** (#848): "After the next KG build, re-validate with `python -m torchcell.knowledge_graphs.supported_queries validate expression_proteome_morphology --release <new release>` (or deprecate the query) and close this issue by hand."
- **#847** (#847): "After the next KG build, re-validate with `python -m torchcell.knowledge_graphs.supported_queries validate amino_acid_betaxanthin --release <new release>` (or deprecate the query) and close this issue by hand."
- **#842** (#842): "After the next KG build, re-validate with `python -m torchcell.knowledge_graphs.supported_queries validate essentiality_smf --release <new release>` (or deprecate the query) and close this issue by hand."
- **#917 (item 3)** (PR #838, Owner decision 3; re-measured 2026-10-10 by `manual_problems_ledger_measurements.py --disk`): "**530 G of retired dev-store copies sit on `/scratch`** across 433 directories ... Graveyard contents are the owner's to purge; `trash` is not installed on GilaHyper."
- **#917 (item 1)** (`notes/database.kg-4-build-sizing.md`; program memory "delete the 170 G orphan"): "170 G of CSV for the 51 yeast adapters, read directly off `/db/database/biocypher-out/2026-10-09_07-20-33`."
- **#917 (item 2)** (PR #838, Owner decision 4): "`/db/database/data.superseded.2026-10-06_04-41-16` (135 G) is the KG 2.0 store, two generations old and backed up under `/bulk/kg-releases/2026.09.17-7715ee35`; freeing it is the one 135 G lever on `/db`."
- **#917 (item 4)** (program memory, 2026-10-09 17:15 CDT): "Stray worktree feat/ecoli-ishii2007 is dirty (1 test file) and belongs to an older session."

### Not yet reported

Four agents launched at about 23:20 CDT were still running at 23:45 CDT with no wall reported: `feat/ecoli-potel2018` (Potel 2018), `feat/ecoli-baba2006-gerdes2003` (Baba 2006 and Gerdes 2003), `feat/ecoli-wetmore2015-own-release` (Wetmore 2015 from its own release) and `docs/aggregation-decomposition-policy` (the aggregation-decomposition audit). None had opened a PR or filed an issue. A by-hand item any of them files joins #919 and this ledger as a dated section.

### Checked and excluded

Measured as already resolved, still open:

- #625 (personal `thesis` Zotero collection): the 2026-10-10 03:32 CDT `lit_sync` run logs `sync: tree 'thesis' -> 1 collections` and `personal:thesis-prelim/thesis: 0 items`; earlier runs logged "personal root 'thesis' not synced". Close by hand after confirming.
- #709 (`--ocr-recorded`): landed through PR #803 (`scripts/lit_capture_si.py` carries the flag); the issue was never closed.
- #734's residual hole, "left for your call" in PR #831: re-measured by the ledger script (`manifests_without_bindings`), the 7 store manifests with no binding closure nodes are now the four Costanzo dev subsets and three retired copies; no mapped store is affected.

Decisions their source left open and a later PR took:

- PR #849's open decision (should Butland 2008 store its own hypomorphs): taken by PR #852, 301,803 records with 5,413 SPA-tag hypomorphs.
- PR #845's Lim 2025 pairing decision: taken by PR #852 (the four `Proteome_*` contrasts).
- PR #868's four 48 h drops: resolved by PR #876 (left-censored titers).
- PR #838's decision 1 (let job 3547 serve KG 4.0 or cancel): moot, job 3547 failed and arrays 3565 and 3733 rebuilt the stale stores.
- #833's part 2 (rebuild every bacterial dev store from merged main): done by arrays 3565 and 3733.
- Yang 2019 (candidate rank 49): refused by measurement in PR #845 (no released reference titer for ELPP000); #826's last comment does not list it among the owner decisions.

Agent work, not owner-only, in the same labels: #524, #528, #533, #537, #541, #673, #682 (test-campaign loader quirks); #757 (yeastphenome, unmapped); #758 (quote audit gate); #860 (resumes after #726); #861 (Rand 2017 net-new tables); #889 (16 unverified stores, agent in flight); #893 and the ORF-renaming halves of #897 to #900 ("Fix direction (not done)", no owner wording); #903's score half. Price 2018's `strain_fit.tab` (PR #845, "retrieval-gated") is not walled: it sits in the same `bigfit/html/Keio/` directory the loader already retrieves by `direct_url`. PR #835's other left-open recommendation, a `ProvenanceGap` on a list-valued field for Choe 2019's MS56 background, is a schema follow-up with no owner wording. PR #850's "Left for a separate decision" (`kg_manifest admit` does not probe readability) is tooling.

Decisions taken by recommendation are reversible and not rows here: PRs #830, #834, #835, #836, #837, #839, #843, #844, #845, #850, #851, #852, #855, #859, #864, #865, #866, #868, #870, #873, #874, #876, #880, #881, #882, #884, #885 and #892 each carry such a section; read them if a recommendation should be revisited.

Outside the three labels: #271 (unlabeled, 2026-08-23) lists Zotero hygiene the owner would also do by hand (four duplicate works, 20 personal items that fail the DOI-and-PDF gate); it is not a bacterial or KG 4.0 item.

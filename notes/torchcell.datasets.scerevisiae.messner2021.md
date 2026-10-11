---
id: fyszaumqd9dyo7gfev6hvkm
title: Messner2021
desc: ''
updated: 1791747239493
created: 1791747239493
---

## 2026.10.11 - Loader, release measurement and L0-L4

`ProteomeMessner2021Dataset` (`torchcell/datasets/scerevisiae/messner2021.py`, dev store `$DATA_ROOT/data/torchcell/proteome_messner2021/`) serves the yeast drug screen of Messner et al. 2021 (Nat Biotechnol, doi 10.1038/s41587-021-00860-4, Zotero XTEKHQBM): one `ProteinAbundanceExperiment` per drug-treated Scanning SWATH run of prototrophic BY4741, genotype = the reference strain (no perturbation), environment = synthetic minimal medium at 30 C plus one compound at 10 uM, phenotype = per-protein DIA-NN MaxLFQ quantities, reference = the DMSO runs. It is row 4 of wave 3 of the yeast Zotero audit ([[database.yeast-zotero-collection-audit.2026.10.11]]), which counted 95 runs and no protein values.

### What the release carries (measured from the bytes)

- The paper's tables carry no protein values. Supplementary Table 4 lists the 16 drugs and their class; the Methods add folic acid and FIN56.
- PRIDE PXD023613 holds 336 files: 318 raw (`.wiff` / `.wiff.scan`), 10 DIA-NN reports, 4 libraries and index files, 3 FASTA and one community SDRF. The drug screen has ONE processed file, the DIA-NN 1.7.12 main report `20200914_sswath_yeast_drugscreen.tsv` (741,477,090 bytes, sha256 `0416225401808bfb17ede25471f047cd6fb17897fd8d5e29cefc8d19a00afdfe`), which the authors' own `diann_outputfileinfo.txt` names under "Yeast drug screen". It is a processed quantitative release, so the row is loadable rather than raw-only.
- The report: 1,296,112 precursor rows over 102 runs, 95 plate injections named `<injection>_plate_2_wt_<well>_<agent>_<number>` and 7 pooled-digest QC injections `NA_NA_NA_QC_<n>_QC_NA`. Per (run, protein group) it carries `Genes.MaxLFQ.Unique`, a linear MaxLFQ quantity from proteotypic precursors, constant across the group's precursor rows (0 of 278,010 proteotypic cells carry two values). Every proteotypic group is one UniProt accession (4,747 groups, 0 multi-protein).
- The plate (measured): injection `i` is well `96 + i`; the 96 injections are four blocks of 24, each the 18 agents then six DMSO, the paper's "blocked by replicates" design. Each block's agent order is Supplementary Table 4's row order once FIN56 and folic acid are set aside, which is what maps the misspelled file tokens (`Rovustatin`, `Pravastatine`, `Lovastatine`) to the table's compounds; `process` refuses a plate where this does not hold. Injection 2 (block 1's fluvastatin) is in neither the report nor the raw files, so fluvastatin has 3 runs.
- The trailing number of each run name (for example `0.57`, `-0.03`, `NA`) is described nowhere in the paper, the SI or PRIDE. It is kept verbatim in `preprocess/runs.csv` and never interpreted. The community SDRF's `biological replicate` column is that number times 100, not a replicate index, so the SDRF is not consumed.

### The paper's three rules, reproduced

The Methods sentence that defines the yeast proteome is applied in its own order, and two of the paper's stated numbers come back out of the bytes:

| rule | quote (paper.md) | measured effect |
|---|---|---|
| 1 | "Only proteins identified with proteotypic (that is, specific) peptides and 0.01 protein $q$ -value were considered." | `Proteotypic == 1` and `Protein.Q.Value <= 0.01`: 4,244 proteins, 232,493 (run, protein) cells |
| 2 | "proteins were considered only if detected in ${ > } 5 0 \%$ of the samples" | over the 102 injections: 2,200 proteins kept, 30,902 cells dropped |
| 3 | "samples were removed if they had ${ < } 8 0 \%$ of the maximum identification number across samples" | after rule 2 the maximum is 2,133 and the threshold 1,706.4: one run removed, `5_plate_2_wt_101_Pemetrexed_NA` (1,096 identifications under rule 1, 1,043 under rule 2) |

- The per-run mean after rules 1 and 2 is 1,976.4 over the 102 injections; the paper states "On average 1,980 unique proteins ( $1 \%$ FDR) (Supplementary Fig. 4a) were identified per run" over "103 injections ... (quality control and samples)". Rule 1 alone gives 2,279.3, so rule 2 is part of the stated proteome.
- With rule 3 as read here the 16 Table S4 drugs keep 62 runs, the "n = 62 for biological samples" of Supplementary Fig. 4b. Applied before rule 2 instead, rule 3 would remove 26 runs (23 with `PG.Q.Value`), which the paper's quadruplicates contradict.
- One protein is refused by name: P03872 (REP2, a 2-micron plasmid protein, quantified in 59 of 102 runs and 39 of the 70 record runs). No feature of the S288C R64 GFF names it, so it has no systematic ORF key. Every other kept accession maps through the SGD GFF cross-references (the Messner 2023 mapper); an unmapped one raises.

### Records

- 70 records: 95 plate runs minus 24 DMSO runs minus the 1 outlier. 4 runs per agent except fluvastatin sodium and pemetrexed disodium hemipentahydrate (3 each). 62 are the 16 Table S4 drugs, 8 are folic acid and FIN56.
- 138,662 protein values over 2,199 ORF keys, 1,793 to 2,106 per record (mean 1,980.9). One acquisition per record: `n_replicates = 1`, `protein_abundance_se = None`. `measurement_type = swath_ms_diann_maxlfq_proteotypic_quantity`.
- Reference: the paper's contrast is "drug-treated versus DMSO-treated", so each protein's reference is its mean over the 24 DMSO runs, its SE (SD / sqrt(n)) and n (5 to 24; 1,263 of 2,199 proteins in all 24), restricted to the proteins the record measured, as Messner 2023 does.
- Environment: `SM_UNSTATED_MESSNER2021`, a module-local medium with one `composition_deferred` component. The paper says only "synthetic minimal medium" and "synthetic minimal liquid medium" and cites no recipe; the same lab's 6.7 g/L YNB + 2% glucose (Mulleder 2016, Messner 2023) is plausibly the same, but that identification is not stated, so the grams are not copied in (the `SM_DEFERRED` treatment of Zelezniak 2018). Temperature 30 C. Each compound is a `SmallMoleculePerturbation` at 10 uM, `DoseBasis.fixed`, with a DMSO `Solvent` at an unstated fraction; the reference environment carries DMSO as `Concentration(basis=fixed)` with no value.
- Genotype: `ReferenceGenome(strain="BY4741")` and an empty genotype. The four prototrophy knock-ins ("rendered prototrophic by genomic knock-in of the missing genes") are not modeled, as in the Messner 2023 and Mulleder loaders, because `ProteinAbundanceExperimentReference` takes the base `ReferenceGenome`; filed as #1032 (shared with Messner 2023 and Mulleder 2016).

### Sourcing (verbatim, mirror sha256)

`paper.md` sha256 `07756a757d7378503f2b284c3d0ad2896446a9bee94435120ebf6715484b0e20`; `si/si1.md` sha256 `30bb324267d8c64057cfcd629fbb6bc57b278d6ff40be8a55f7c4329ce90792d` (MinerU OCR of `si/si1.pdf`, run for this key with `scripts/lit_capture_si.py messnerUltrafastProteomicsScanning2021 --ocr-recorded`, report `_sync_reports/si_ocr_20261011T164958Z.json`; the audit had captured the SI PDFs without OCR). Every quote below is a `SourcedValue` in `messner2021.SOURCED` and is re-read by the L3 audit.

| value | quote | file |
|---|---|---|
| strain | "The auxotrophic S. cerevisiae strain BY4741 (∆his3, ∆leu2, ∆ura3, ∆met15) was rendered prototrophic by genomic knock-in of the missing genes." | paper.md, Methods |
| medium, 30 C | "Subsequently, colonies were inoculated in synthetic minimal liquid medium $2 5 \mathrm { m l } )$ ) and incubated at $3 0 ^ { \circ } \mathrm { C }$ for 1 day." | paper.md, Methods |
| 10 uM | "drugs were added to achieve a working concentration of $1 0 \mu \mathrm { M }$ (1 ml total volume per well)." | paper.md, Methods |
| quadruplicate | "Samples were measured in quadruplicate and pooled samples were repeatedly injected for quality control." | paper.md, Results |
| blocks | "Randomization drug response measurements were blocked by replicates." | paper.md, Reporting Summary |
| value | "Protein quantities were obtained using the MaxLFQ algorithm84 as implemented in either DIA–NN (yeast drug screen)" | paper.md, Methods |
| reference | "The differential expression analysis (drug-treated versus DMSO-treated) for the yeast drug screen was done on the $\log _ { 2 }$ -transformed protein quantities" | paper.md, Methods |
| folic acid, FIN56 | "samples treated with folic acid and FIN56 were excluded from the analysis because these do not belong to the three studied drug classes." | paper.md, Methods |
| n = 62 | "$\scriptstyle { \mathtt { n } } = 6 2$ for biological samples" | si/si1.md, Supplementary Fig. 4 legend |
| drugs | the 16 cell pairs of Supplementary Table 4, e.g. `<td rowspan=1 colspan=1>Rosuvastatin Calcium</td><td rowspan=1 colspan=1>Statin</td>` | si/si1.md |

### Verification (L0-L4, `messner2021.verify_build`, dev store at commit of this note)

`PYTHONPATH=<wt> python -m torchcell.database.build_dataset_lmdb --dataset ProteomeMessner2021Dataset --retire-existing --verify`; registered in `runners.PROTEIN_DATASETS` as `proteome_messner2021` with `verify_build_module`, so `run_protein` dispatches to the loader's gate.

| level | check | result |
|---|---|---|
| L0 | structural | 70 records validated |
| L1 | count | 70 of 70 |
| L1 | wild_type_genotype | 70 of 70 carry no perturbation |
| L1 | runs_per_compound | 18 compounds; 4 runs each, fluvastatin sodium and pemetrexed 3 |
| L1 | release_layout | 102 runs (95 plate, 7 QC), absent injection [2], outlier [5_plate_2_wt_101_Pemetrexed_NA], 24 DMSO runs |
| L2 | value_fidelity | 138,662 finite values |
| L2 | values_match_report | 138,662 of 138,662 equal the report bytes (independent dict path) |
| L2 | reference_matches_report | 138,662 of 138,662 reference means re-derived from the 24 DMSO runs |
| L3 | reference_finite, measurement_type_consistent | pass |
| L3 | sourced_quotes_verbatim | 32 of 32 quotes verbatim in the mirror |
| L4 | paper_mean_identifications | 1,976.4 rounds to the paper's 1,980 |
| L4 | paper_biological_sample_count | 62 Table S4 records = Supplementary Fig. 4b's n = 62 |
| L4 | gene_containment_sgd | 2,199 of 2,199 keys are S288C systematic names |
| L4 | messner2023_reference_agreement | Pearson r 0.813 of log2 DMSO reference vs the Messner 2023 HIS3 WT reference over 1,581 shared ORFs (bar 0.6, set before measuring) |

### Raw mirror

`$DATA_ROOT/torchcell-raw/messnerUltrafastProteomicsScanning2021/`, written by `messner2021.deposit_raw_mirror` (`python -m torchcell.datasets.scerevisiae.messner2021 --deposit-from <dir>` after `curl -f -o <file> <url>`), retrieval `direct_url`, retrieved 2026-10-11:

- `data/pride/20200914_sswath_yeast_drugscreen.tsv`, sha256 `0416225401808bfb17ede25471f047cd6fb17897fd8d5e29cefc8d19a00afdfe`
- `data/pride/diann_outputfileinfo.txt`, sha256 `008924ee01c4151eb74453731680a722cbd67101f45c862c3ae926b30b8f148f`

Not deposited, recorded in the manifest's `si_expected`: the raw spectra (accession PXD023613, about 3.5 GB per `.wiff.scan`), the spectral library and FASTA behind the report, and the community SDRF.

### Compound table

Ten rows added as a pure insertion (5,533 to 5,543), curated from PubChem by name with `compound_identity_curate.py` on `compound_identity_inputs/messner2021.txt`: rosuvastatin calcium, fluvastatin sodium, pitavastatin calcium, pemetrexed disodium hemipentahydrate (PubChem's record for "Pemetrexed Disodium Hydrate", CID 135564527), pralatrexate, pravastatin sodium, miconazole nitrate, sodium methotrexate ("Methotrexate Disodium", CID 11329481), uniconazole, FIN56. These ten labels truly lacked a row; the other nine (pyrimethamine, clotrimazole, lovastatin, ketoconazole, atorvastatin, simvastatin, itraconazole, folic acid, DMSO) already resolved. No other loader passes any new lookup key, so no served node is renamed.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: "No schema contract changes vs origin/main." No new class or field; `has_gene_perturbations = False` on the loader.

### Owner decisions taken by recommendation

1. Per-run records, not per-drug means: each run is one acquisition and the replicate structure stays in the store (Messner 2023 and Muenzner 2024 do the same). A per-drug mean is derivable downstream.
2. The paper's rules 2 and 3 are applied, because they define the proteome the paper reports (the 1,980 and n = 62 back-solves). The alternative, serving the 2,044 proteins rule 2 drops (30,902 cells), is a one-line change.
3. Folic acid and FIN56 are served (8 records): the paper excluded them for scope ("do not belong to the three studied drug classes"), not quality, and the plate-wide 10 uM sentence covers them.
4. The vehicle is DMSO at an unstated fraction, because the control is "DMSO-treated"; the paper never says the drugs were dissolved in DMSO.
5. Salts keep the form Supplementary Table 4 names (the Hillenmeyer convention), so rosuvastatin calcium and the free-acid rosuvastatin row stay different compounds.

### Refused, with the measurement

- 1 run (`5_plate_2_wt_101_Pemetrexed_NA`) by the paper's rule 3.
- 1 protein (P03872, REP2) with no S288C systematic name; 39 record cells.
- 24 DMSO runs are the reference, not records; 7 QC runs are pooled digests, not records.

### Candidate verdict

`candidate-gate gate` has no table row for an off-schedule key, so `experiments/database/scripts/messner2021_candidate_verdict.py` ([[experiments.database.scripts.messner2021_candidate_verdict]]) re-runs the yeast audit's composition for this one entry with the loader's inputs: the raw mirror for G3 (42 files in library + raw, all intact), the module's own DOI for G4 (no key held by another module), and the 18 compound labels plus DMSO for G5 (all resolve). Outcome `admissible`, G1 to G5 pass; `candidate-gate validate` accepts the JSON and `candidate-gate audit` re-reads every evidence quote of the store, ok 82. G4-value was not run: `overlap` reads one scalar per record, and this phenotype is a per-protein map (the L2 `values_match_report` check is the byte-level comparison). `candidate-gate ledger` refused: "torchcell.candidates.ledger.md already has '## 2026.10.11 - Candidate gate ledger'; one ledger per day", so the verdict reaches the ledger with the next day's run.

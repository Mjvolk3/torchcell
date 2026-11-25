---
id: 50wlx3pcp36pvonadaqp60s
title: Potel2018
desc: ''
updated: 1791817491945
created: 1791817491945
---

## 2026.10.12 - Loader, the phosphosite-intensity leaf, and the L0-L4 table (#921)

Potel CM, Lin MH, Heck AJR, Lemeer S (2018) Nature Methods 15:187-190, doi:10.1038/nmeth.4580, PMID 29377012; ProteomeXchange PXD008369 (jPOST JPST000369). Loader: `torchcell/datasets/ecoli/potel2018.py` (`PhosphositeIntensityPotel2018Dataset`). Release inventory and mirror deposit: [[experiments.036-dataset-fixes-before-kg-build.scripts.potel2018_release_inventory]]. Schema: [[torchcell.datamodels.bacterial-perturbation-ontology]] (2026.10.12 section). Family gate: [[torchcell.verification.phosphosite]]. Proteome-family fingerprints and stale-store round trips: [[experiments.036-dataset-fixes-before-kg-build.scripts.potel2018_phosphosite_checks]].

### What is stored

One `BacterialPhosphositeIntensityExperiment` per released sample column of Supplementary Table 2 (`data/41592_2018_BFnmeth4580_MOESM4_ESM.xlsx`, sha256 `f69ca61a...`), 12 records, wild type W3110 (`Genotype(perturbations=[])`) against `ecoli_K12_W3110_ASM1024v1`. Every record carries all 2,129 sites exactly as the table gives them: the accession string, gene names, residue and positions verbatim, keyed `<Acc. No.>|<Amino acid>|<Positions>`. The 42 sites listed on more than one gene (10 on more than one accession) are one key each. No accession, gene name or position is looked up or resolved against any annotation (owner, 2026-10-12: "the data is the data!"; linking to gene nodes is the adapter's and the model's concern).

Counted at build time (`preprocess/build_accounting.json`, asserted against the release inventory's counts):

| measure | value |
|---|---|
| sites per record | 2,129 (S 1,220, T 501, Y 162, H 246) |
| multi-gene / multi-accession sites | 42 / 10 |
| diagnostic peak `+` | Y 82, H 31 (none on S or T) |
| cells | 25,548; 13,453 quantified, 12,095 `not_quantified` (the released `0`) |
| sites `not_quantified` in all 12 samples | 245, stored on every record |

| sample column | carbon | phase | rep | harvest OD600 | quantified sites | reference |
|---|---|---|---|---|---|---|
| glucose_exp1 | glucose | exponential | 1 | 0.6 | 989 | itself |
| glucose_exp2 | glucose | exponential | 2 | 0.6 | 1,002 | itself |
| glucose_exp3 | glucose | exponential | 3 | 0.6 | 897 | itself |
| glucose_stat1 | glucose | stationary | 1 | 1.2 | 1,200 | glucose_exp1 |
| glucose_stat2 | glucose | stationary | 2 | 1.2 | 1,241 | glucose_exp2 |
| glucose_stat3 | glucose | stationary | 3 | 1.2 | 1,136 | glucose_exp3 |
| glycerol_exp1 | glycerol | exponential | 1 | 0.5 | 1,064 | itself |
| glycerol_exp2 | glycerol | exponential | 2 | 0.5 | 1,087 | itself |
| glycerol_exp3 | glycerol | exponential | 3 | 0.5 | 1,054 | itself |
| glycerol_stat1 | glycerol | stationary | 1 | 1.0 | 1,263 | glycerol_exp1 |
| glycerol_stat2 | glycerol | stationary | 2 | 1.0 | 1,201 | glycerol_exp2 |
| glycerol_stat3 | glycerol | stationary | 3 | 1.0 | 1,319 | glycerol_exp3 |

Each column's carbon, phase and replicate are sourced from its own Table 2 legend line ("... in the 1st replicate of glucose-fed exponential E. coli cells."); the harvest OD600 from the dissertation's Chapter V ("mid-log phase (OD600= 0.6/0.5 for glucose / glycerol-fed" and "stationary phase (OD600= 1.2/1.0 for glucose/ glycerol-fed cells)").

### Environment and genotype, sourced from Chapter V of the dissertation

The published Online Methods are walled (#922); the first author's dissertation (raw mirror `paper/thesis/Potel.pdftotext.txt`, sha256 `05a50657...`) reprints the paper as Chapter V, and every Methods value is quoted from Chapter V of that rendering:

- Host: "E. coli strain W3110 was grown in M9 minimal medium, consisting of M9 salts (6    2" (pdftotext sets the subscripted Na2HPO4 out of line).
- Salts: 6 g/L Na2HPO4 (Chapter V's rendering is broken; the same sentence in Chapter VI renders "M9 salts (6 g/L\nNa2HPO4, 3 g/L KH2PO4, 0.5 g/L NaCl, 1 g/L NH4Cl)" and is quoted as the corroboration), "3 g/L KH2PO4, 0.5 g/L NaCl, 1 g/L 4NH Cl) supplemented with either additional 0.5% (w/v)" (`4NH Cl` is the rendering of NH4Cl), "glucose or 0.5% glycerol, 1 mM MgSO", ", 0.1 mM CaCl2".
- Carbon: 0.5% (w/v) D-glucose, or 0.5% glycerol stored as `ConcentrationUnit.percent`, because the sentence states the w/v basis for glucose only.
- 37 C, aerobic: "with vigorous shaking at 37°C. Cells".
- One `Media` per carbon source (base `M9`, defined in the loader, not `MEDIA_LIBRARY`: its 6 g/L Na2HPO4 differs from the library `M9` base's 6.78 g/L). `Environment.media` is stated on every record, never gapped.

### The replicate unit is a typed gap, not a pick

Two SI legends call the twelve samples "n=12 biological replicates" (Supplementary Figures 3 and 6); the reporting summary says "we made technical triplicates (after digestion, before enrichment) of each" condition and the dissertation's Chapter V says "Fe3+-IMAC was performed in technical triplicates for each biological groups". `replicate_unit` is None on every record with `ProvenanceGap(reason=deferred_pending_source_review, resolve_with=the published Online Methods)`, quoting both statements. Closing it is #922.

### Not stored

- Table 2's total `Intensity`: defined over "the identified  phosphopeptide sequence" with no sample, and not the sum of the twelve (equal within 1e-6 relative for 278 sites, larger for 1,130, smaller for 477, 0 for 244; build accounting). Hypothesis, not checked: it sums over all 18 raw files of the MaxQuant search, including the six protocol-comparison runs that have no Table 2 column.
- Supplementary Table 5: derived from Table 2 (L4 below).
- The peptide-level columns (`Number of Phospho (HSTY)`, `Phospho (HSTY) Probabilities`, `Charge`, `Mass error [ppm]`) and `Protein names`.

### Reference (a decision, stated for review)

The release names no denominator. Each record's reference is the exponential-phase sample of its own carbon source with the same replicate number: the release's one contrast (Supplementary Table 5) is exponential against stationary within a carbon source, and the replicate number only picks one real released sample deterministically. It does not pair cultures: the exponential and stationary injections are different batches (`OR13_A2_20161014` and `OR13_20160922` in jPOST).

### The host's genome class and the W3110 vocabulary

The records pin `ecoli_K12_W3110_ASM1024v1` (GCF_000010245.2), which needs W3110 in the schema's strain vocabulary and a genome class that reads the set (candidate gate G2). Both come from the Teteneva 2024 branch (`feat/ecoli-teteneva2024`, commit `b24d494b4`), carried into PR #1062 unchanged as its first commit, without that commit's `TransposonInsertionPerturbation.identifier_kind` change, which awaits the owner's sign-off on #910. Record: `database/schema_changes/2026-10-12-w3110-reference-strain-vocabulary.json`. No reader was written here.

### Dev store and L0-L4 verification

Built under slurm job 4398 at `8cb4e490e` with `build_dataset_lmdb --dataset PhosphositeIntensityPotel2018Dataset --retire-existing --verify` into `$DATA_ROOT/data/torchcell/phosphosite_intensity_potel2018`: 12 records in 2 s, 6 distinct references, gene set size 0 (no record perturbs a gene). The build manifest reads `torchcell_dirty: true` because of untracked files only (the worktree guard and these uncommitted notes); the job logged 0 modified tracked files. `--list-stale --include-private` does not name the store.

| level | check | result |
|---|---|---|
| L0 | every record validates against `ExperimentType` | pass, 12 records |
| L1 | record count | pass, 12 of 12 |
| L1 | one record per released sample | pass, 12 labels, 0 repeated |
| L1 | every record and reference carries one site key set | pass, 1 set of 2,129 keys |
| L1 | per-sample quantified sites are the release inventory's | pass, 989 to 1,319, every site on every record |
| L2 | stored intensities finite and positive | pass, 13,453 intensities |
| L2 | one identification record per site across the store | pass, 2,129 sites, 0 differ |
| L2 | every stored cell and site is the released one (re-read of the pinned workbook) | pass, 25,548 cells, 0 differ |
| L3 | one measurement type | pass, `fe3_imac_lc_msms_maxquant_summed_xic` |
| L3 | carbon, phase, replicate, OD600 and reference follow each column's legend | pass, 12 of 12 |
| L3 | the reference pins W3110 | pass, `ecoli_K12_W3110_ASM1024v1` |
| L3 | every quote is verbatim in its sha256-pinned mirror file | pass, 30 quotes |
| L4 | Supplementary Table 5's log2 values are log2 of the stored intensities | pass: 1,436 of 1,469 Table 5 sites join on the released key; 6,513 cells agree within 1e-3, 2,103 are Perseus imputations of `not_quantified` cells, 0 differ |

The L4 is a cross-table check inside the release, not a genome-reference rule: no annotation decides whether a site exists, how it is keyed, or what status it carries. The 33 Table 5 keys that do not join are sites Table 2 lists on rows with more than one position (9 multi-accession rows, 24 one-accession rows naming two genes, 22 of them `P0CE47`), which Table 5 writes with one position; they are counted, not compared, and nothing is dropped for them.

### Released count against stored count

| store | released | stored | difference |
|---|---|---|---|
| `phosphosite_intensity_potel2018` | 2,129 sites x 12 sample columns (25,548 cells) | 12 records x 2,129 sites; 13,453 intensities, 12,095 `not_quantified` | none: every site, every cell |

Not stored, each with its count: the total `Intensity` column (2,129 cells); Supplementary Table 5 (1,469 rows, derived); the peptide-level columns (`Number of Phospho (HSTY)`, `Phospho (HSTY) Probabilities`, `Charge`, `Mass error [ppm]`) and `Protein names` (2,129 rows each). Supplementary Tables 1 and 3 are standards and an antibody comparison, not this study's cells.

### Candidate verdict

`experiments/database/scripts/potel2018_candidate_verdict.py --write` at `8f69868dd`: **admissible**, G1 to G5 pass (G2: W3110 resolves and reads; G3: 11 raw-mirror files intact; G5: `PhosphositeIntensityPhenotype`). `database/candidates/potelWidespreadBacterialProtein2018.json`. No Zotero item: the publisher PDF is walled (#922), and the only mirrored PDF is the whole dissertation, which `scripts/zotero_add_paper.py --dry-run` would file under the article's DOI; filing it would mislabel the item, so filing waits for the published PDF.

### Open, for the owner

1. The replicate unit (#922): the published Online Methods decide whether the twelve samples are biological replicates or technical triplicates of four conditions.
2. The reference rule above (exponential sample of the same carbon source and replicate number) is a recommendation; the release names no denominator.

## 2026.11.24 - Per-replicate review (#1178, part of #1172)

### Verdict: already one record per released replicate; no record changes

The store holds 12 records, one per Supplementary Table 2 sample column, and each column is one raw file: the jPOST listing and `mqpar.xml` of JPST000369 map the twelve `OR13_..._Ecoli_<carbon>_<phase>_2mg_IMAC_R<n>.raw` files one to one onto the twelve MaxQuant experiments `glucose_exp1` ... `glycerol_stat3` (read 2026-11-24, not deposited, since no loader reads them). Counts per replicate number: 4, 4, 4; per condition (carbon x phase): 3 each; total 12. The authors combine nothing across a condition's three: the Chapter V t-test runs on the three log2 values, and Supplementary Table 5 releases them per replicate.

A rebuild of this branch into `$DATA_ROOT/tmp/scratch/per-replicate-potel2018/phosphosite_intensity_potel2018` is identical to the dev store (all 12 experiments and references equal as dicts), so grouping the records by genotype, medium and harvest phase and taking the per-site mean reproduces the old per-condition means with a max absolute difference of 0.0. Verifier: PASS, 38 quotes re-read.

### The replicate unit: the evidence on each side

On the biological side, two published SI legends (`si/41592_2018_BFnmeth4580_MOESM1_ESM.pdftotext.txt`, sha256 `4dee5c2a...`): "The same UV profiles were observed for the n=12 biological replicates." (Supplementary Figure 3, L36) and "profiles derive from the totality of the n=12 biological replicates." (Supplementary Figure 6, L87).

On the technical side:

- reporting summary (`si/..._MOESM2_ESM.pdftotext.txt`, sha256 `9951a1ac...`), item 1: "phosphorylation in 4 biological conditions. In order to compare those conditions" and "we made technical triplicates (after digestion, before enrichment) of each" (L19-20); item 3: "Results were reproduced under 4 biological conditions, each consisting of technical" (L30);
- published Supplementary Figure 11 legend (L173): "(each replicate corresponding to a different enrichment and different LC-MS/MS injection)";
- dissertation Chapter V (`paper/thesis/Potel.pdftotext.txt`, sha256 `05a50657...`): "Eluted peptides were split in 2 mg peptide aliquots and subsequently dried down" (L6040), "Fe3+-IMAC was performed in technical triplicates for each biological groups" (L6052), "The log2 scale of phosphosite intensity from technical" / "Fe3+-IMAC triplicates were subjected into a two tailed t-test" (L6147-6148), and its Figure S11 legend "peptides (pink) in 3 technical replicates were all above 0.93" (L6727);
- the dissertation's reprints of the Figure S3 and S6 legends (L6473-6484, L6583-6588) carry no "n=12 biological replicates" sentence; the string "n=12" does not occur in the dissertation;
- the Fig. 1 source data (`si/..._MOESM7_ESM.xlsx`, sha256 `61ff4b38...`) names the runs `OR13_20160922_CMP_Ecoli_glycerol_stat_2mg_IMAC_R1` to `_R3`, the replicate suffix on the IMAC step of one 2 mg sample.

All eleven text statements are `REPLICATE_UNIT_EVIDENCE` in the loader and are re-read against the mirror bytes at L3. No sample is withdrawn: the reporting summary's item 2 reads "No data was excluded from the analysis".

### Why the count is 12 under either reading

If the three are technical, policy 2 of #1178 combines inside the record only measurements "that the design requires to be combined to give one valid value". Each enrichment here gives its own released value that the authors test as a replicate, and no combined value is released; folding three into one would need a loader-computed combination over intensities with `not_quantified` cells, plus a phenotype field for the technical n, which `PhosphositeIntensityPhenotype` does not have. So the plan's "4 if technical" estimate would need both a schema change and a computed statistic, and is not done.

### Open, for the owner (#1172)

1. Label: set `replicate_unit = SampleUnit.technical_replicate` on all 12 records (two SI legends against nine statements and the dissertation's own reprint), or keep the `deferred_pending_source_review` gap until the published Online Methods are read (#922). The records keep their count either way; only the label and the gap change.
2. If the owner wants the technical triplicate folded into 4 records, that needs a schema shape first, for example a per-site `technical_intensities: dict[str, list[float | None]]` plus `technical_n` and `technical_unit` on `PhosphositeIntensityPhenotype`, with the per-enrichment values kept verbatim inside the record rather than averaged.
3. The reference rule (exponential sample of the same carbon source and replicate number) pairs replicate numbers across two injection batches; under a technical reading the numbers pair nothing. It is a single released sample, not an aggregate over the condition's replicates, and the phenotype requires `replicate >= 1`, so a policy-7 aggregate reference would also need a schema decision.

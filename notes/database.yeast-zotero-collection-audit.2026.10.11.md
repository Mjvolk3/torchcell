---
id: 9hkxjko11w2botbpfvipx0v
title: '11'
desc: ''
updated: 1791716131951
created: 1791716131951
---

## 2026.10.11 - Audit of the yeast Zotero collection for missed datasets

The owner's question: are there datasets in Zotero for yeast that we have not loaded, now that their papers are mirrored? The group collection `database/Saccharomyces-cerevisiae` (key `KCG3NWH7`) held 68 items when the driver measured it this morning and 73 by 11:00 UTC. The driver's scan against DOI and key mentions in `torchcell/datasets/**` left 30 items with no loader (schedule rows 1 to 5, being loaded in parallel, excluded). Every one of the 30 is classified below. Five more items with no loader are not in the driver's list because they are schedule rows 6 to 10 (Jackson 2020, Puddu 2019, N'Guessan 2025, Hackett 2020, Hu 2007), the next rows in order rather than missed datasets; their SI was captured by the driver at 10:48 UTC and `candidate-gate gate --row` passes G1 to G4 for the four with a DOI url (G5 not run; N'Guessan's row url is a PMC link and needs `--citation-key nguessanRefiningResolutionYeast2025`). They are not audited further here.

How it was measured. SI was captured for the 15 dataset-plausible keys with `scripts/lit_capture_si.py` (no OCR), all `captured` (report `$DATA_ROOT/torchcell-library/_sync_reports/si_20261011T104400Z.json`). Each paper's Data availability, Methods and SI were read from the mirror bytes and every record count below was counted from a released file (openpyxl / pandas, filters stated per key); where the values live only in an external repository, the listing was probed and the count is either measured from a small file or marked not measured with the reason. The judgments are held as data in `experiments/database/scripts/yeast_zotero_collection_audit.py` ([[experiments.database.scripts.yeast_zotero_collection_audit]]), which gates the 20 dataset candidates with the `torchcell.candidates` evaluators and writes `database/candidates/<key>.json`. `python -m torchcell.candidates audit` re-read all 79 quoted evidence slices of the store against the mirror: ok 79. The tables below are copied from the script's output (`experiments/database/results/yeast_zotero_collection_audit.md`); regenerate, never edit.

Categories: (a) loadable now, no schema change; (b) loadable with a named additive schema gap; (c) method, tool, protocol or database paper with no genotype x environment -> phenotype records; (d) genomes tier; (e) aggregation; (f) blocked at source; (g) wrong organism. No verdict is written for (c) and (g): the verdict's G1 vocabulary (`primary`, `aggregation`, `transcription`, `prediction`) has no kind for a paper that releases no records. A verdict outcome is the gate order's first blocker, so an (f) key can read `blocked:provision` when its host has no assembly set; the category column names the decisive blocker.

### Loadable now

| rank | key | Zotero item | schedule | records (measured) | genotype basis | phenotype class | overlap risk | verdict | mirror |
|---|---|---|---|---|---|---|---|---|---|
| 1 | piotrowskiFunctionalAnnotationChemical2017 | K8PYLHCJ | #73 | 3,438,421 strain x compound z-scores | 291 diagnostic kanMX deletions in the pdr1 pdr3 snq2 drug-sensitized SGA background (pool v2.0 lists 310; not the ~157 the schedule row assumed) | EnvironmentResponsePhenotype | none held: no Piotrowski store; vanacloig2022.py cites it for methods only; yp-data has no folder for PMID 28759014. Same sensitized lineage as env_chemgen_vanacloig2022 (different screens). | admissible | `$DATA_ROOT/torchcell-library/piotrowskiFunctionalAnnotationChemical2017/` |
| 2 | parsonsExploringModeofActionBioactive2006 | 3PRLBH7Y | #60 | 329,254 strain x compound log2(control/treated) ratios | MATa haploid YKO (Research Genetics; BY4741 inferred), pooled barcode microarray | EnvironmentResponsePhenotype | none held; latent: yp-data v1.0 curates the same Table S7 (PMID 16901791), so yeastphenome SCREENS must skip it (#1000). The ratio is control over treated: a loader negates or records the orientation. | admissible | `$DATA_ROOT/torchcell-library/parsonsExploringModeofActionBioactive2006/` |
| 3 | yoshikawaComprehensivePhenotypicAnalysis2009 | DCF6RHD6 | off-schedule | 18,916 strain x condition growth records (two replicate rates + a call each) | haploid MATalpha YKO (BY4742; some BY4739) | EnvironmentResponsePhenotype | none held; latent: yp-data v1.0 curates the same Table S1 (PMID 19054128). Off-schedule: schedule row 85 is Yoshikawa 2011 (G3, xylose), a different paper not in Zotero. | pending | `$DATA_ROOT/torchcell-library/yoshikawaComprehensivePhenotypicAnalysis2009/` |
| 4 | messnerUltrafastProteomicsScanning2021 | XTEKHQBM | off-schedule | 95 drug-screen proteome runs (protein x run values not counted) | wild type only (prototrophic BY4741) | ProteinAbundancePhenotype | none: proteome_messner2023 holds no drug or DMSO samples (yeast5k metadata grep 0) | admissible | `$DATA_ROOT/torchcell-library/messnerUltrafastProteomicsScanning2021/` |

### Schema-gated

| key | Zotero item | schedule | records (measured) | genotype basis | gap | issue | verdict | mirror |
|---|---|---|---|---|---|---|---|---|
| gaschGenomicExpressionPrograms2000 | GNF43TXA | #75 | 1,032,274 gene x array log2 ratios (173 arrays x 6,152 genes) | wild type (DBY7286, DBY8768) plus yap1 and msn2 msn4 deletions and MSN2/MSN4/YAP1 overexpression | MicroarrayExpressionPhenotype requires linear per-gene expression and per-gene n_replicates; every release carries log2 ratios against time zero only | #989 | blocked:provision | `$DATA_ROOT/torchcell-library/gaschGenomicExpressionPrograms2000/` |
| leutertRegulatoryLandscapeYeast2023 | 3ZCWNIHX | #39 | 529,400 phosphosite x treatment log2 fold changes | wild type only (BY4741; BY4743 for the diploid condition) | no PhosphositeFoldChangePhenotype | #921 | admissible_with_gaps | `$DATA_ROOT/torchcell-library/leutertRegulatoryLandscapeYeast2023/` |
| vaishnavEvolutionEvolvabilityEngineering2022 | CTSIU75E | #56 | 75,487 sequence x expression-level records (MolEvol set; 20,616,659 + 30,722,376 random sequences stated, not counted) | promoter library: an 80 bp variable insert in a YFP reporter promoter on a CEN plasmid (Y8205 and S288C::ura3) | PromoterActivityPhenotype exists, but no yeast genotype holds the variable promoter sequence, its well_id is a required str a pooled sort-seq library has none of, and ReporterReadout has no sort-seq member | #991 | admissible_with_gaps | `$DATA_ROOT/torchcell-library/vaishnavEvolutionEvolvabilityEngineering2022/` |
| khaiwalPredictingNaturalVariation2025 | 8HLUB5DG | off-schedule | 28,435 isolate x trait values for the 33 new phenotypes | natural isolates (the 1,011 panel) | no CellFeaturePhenotype | #992 | admissible_with_gaps | `$DATA_ROOT/torchcell-library/khaiwalPredictingNaturalVariation2025/` |
| zhangCombiningMechanisticMachine2020 | B46K3SBE | off-schedule | 341 promoter-combination strain records (GFP synthesis rate) + 6 HPLC titers | engineered strain: CEN.PK2-1C platform with five genes re-integrated under one of six promoters each (6^5 designs) | no yeast perturbation states which promoter drives a gene, and no class fits a biosensor-proxy readout without an environmental edit | #990 | blocked:provision | `$DATA_ROOT/torchcell-library/zhangCombiningMechanisticMachine2020/` |

### Aggregation and blocked at source

| key | Zotero item | schedule | category | records (measured) | why | issue | verdict | mirror |
|---|---|---|---|---|---|---|---|---|
| turcoGlobalAnalysisYeast2023 | KQWRRBY4 | #71 | aggregation | 296,777 strain x screen NPVs the loader keeps (83 screens, 47 studies) | the knockout phenome compendium; a loader exists (dev store yeastphenome, not in the adapter map) |  | admissible | `$DATA_ROOT/torchcell-library/turcoGlobalAnalysisYeast2023/` |
| leeMappingCellularResponse2014 | 6BWAYPY5 | #57 | blocked at source | not measured | the largest chemical-genetic matrix stays blocked: the FD matrices are in no release we hold | #994 | blocked:wall | `$DATA_ROOT/torchcell-library/leeMappingCellularResponse2014/` |
| guzmanUltrafastLabelfreeQuantification2024 | MU4HHLNG | off-schedule | blocked at source | 1,375,123 protein x knockout-run values (genotypes unreleased) | 1,375,123 values with no genotype key | #998 | blocked:wall | `$DATA_ROOT/torchcell-library/guzmanUltrafastLabelfreeQuantification2024/` |
| hoMassiveQTLAnalysis2021 | 7DQ7JSCP | off-schedule | blocked at source | 20,377 segregant x trait values (genotypes blocked) | phenotypes in hand, genotypes vanished with She 2018's Dropbox (#70) | #70 | blocked:wall | `$DATA_ROOT/torchcell-library/hoMassiveQTLAnalysis2021/` |
| sheMappingCausalVariants2018 | 6E5RS4AS | #90 | blocked at source | not measured | per-record data vanished with the authors' Dropbox; schedule row design is wrong | #70 | blocked:wall | `$DATA_ROOT/torchcell-library/sheMappingCausalVariants2018/` |
| zackrissonScanomaticHighResolutionMicrobial2016 | QZ9CKZQQ | #148 | blocked at source | not measured | a platform paper; its demonstration screen is figure-only | #999 | blocked:wall | `$DATA_ROOT/torchcell-library/zackrissonScanomaticHighResolutionMicrobial2016/` |
| lianCombinatorialMetabolicEngineering2017 | WFBK2TWS | #19 | blocked at source | not measured | figure-only values; CEN.PK also has no assembly set | #997 | blocked:provision | `$DATA_ROOT/torchcell-library/lianCombinatorialMetabolicEngineering2017/` |
| deloacheEnzymecoupledBiosensorEnables2015 | WV48S84B | off-schedule | blocked at source | not measured | a method paper with a 54-strain panel and figure-only values | #996 | blocked:wall | `$DATA_ROOT/torchcell-library/deloacheEnzymecoupledBiosensorEnables2015/` |
| trikkaIterativeCarotenogenicScreens2015 | 3WAM7IER | #105 | blocked at source | not measured | hit list only; EG60 has no assembly set | #995 | blocked:provision | `$DATA_ROOT/torchcell-library/trikkaIterativeCarotenogenicScreens2015/` |

### Not a dataset, wrong organism, or genomes tier

| key | Zotero item | schedule | category | why | verdict | mirror |
|---|---|---|---|---|---|---|
| odonnellTelomeretotelomereAssemblies1422023 | HSFSW32M | off-schedule | genomes tier | T2T assemblies (median 17 contigs vs 3,259 for Peter 2018), phased haplotypes, a 4,809-row SV list, Ty and telomere annotation; no phenotypes | blocked:provision | `$DATA_ROOT/torchcell-library/odonnellTelomeretotelomereAssemblies1422023/` |
| loeglerGenotypePhenotype10862025 | 2EYKDJAT | off-schedule | genomes tier | near-T2T genomes for 1,086 isolates, an SV VCF and graph pangenomes; phenotypes all re-served | blocked:provision | `$DATA_ROOT/torchcell-library/loeglerGenotypePhenotype10862025/` |
| brettnerUltraHighthroughputMassively2024 | VP8TSA3J | #30 | not a dataset | method (yeast SPLiT-seq); its value is as the chemistry a Perturb-seq proposal names, as schedule row 30 says | no verdict | `$DATA_ROOT/torchcell-library/brettnerUltraHighthroughputMassively2024/` |
| wagihSGAtoolsOnestopAnalysis2013 | I45F26KL | off-schedule | not a dataset | SGAtools, a web tool for scoring colony-size screens | no verdict | `$DATA_ROOT/torchcell-library/wagihSGAtoolsOnestopAnalysis2013/` |
| kuzminSyntheticGeneticArray2016 | WKGW6AJH | off-schedule | not a dataset | SGA protocol (Cold Spring Harb Protoc) | no verdict | `$DATA_ROOT/torchcell-library/kuzminSyntheticGeneticArray2016/` |
| yantongSyntheticGeneticArray2006 | ZXC7VLW7 | off-schedule | not a dataset | SGA methods chapter | no verdict | `$DATA_ROOT/torchcell-library/yantongSyntheticGeneticArray2006/` |
| engelNewDataCollaborations2022 | H47A584B | off-schedule | not a dataset | SGD release paper; nothing new to load, a citation for sgd.py | no verdict | `$DATA_ROOT/torchcell-library/engelNewDataCollaborations2022/` |
| wangSynLethDB20Webbased2022 | VI2GJEBI | off-schedule | not a dataset | the release paper of the served SynLethDB aggregation (not missed) | no verdict | `$DATA_ROOT/torchcell-library/wangSynLethDB20Webbased2022/` |
| longEnzymeEngineeringDatabase2026 | DKNER94S | off-schedule | not a dataset | enzyme sequence-function database across organisms; no cell genotype x environment records | no verdict | `$DATA_ROOT/torchcell-library/longEnzymeEngineeringDatabase2026/` |
| yuanOpenEnzymeDatabase2026 | 3IL3E6NI | off-schedule | not a dataset | enzyme kinetics repository; no cell genotype x environment records | no verdict | `$DATA_ROOT/torchcell-library/yuanOpenEnzymeDatabase2026/` |
| aranguizMachineLearningReveals2025 | KE773CUE | off-schedule | wrong organism | a cross-species screen, not an S. cerevisiae genotype panel; 15 S. cerevisiae values are not worth a loader | no verdict | `$DATA_ROOT/torchcell-library/aranguizMachineLearningReveals2025/` |
| suthersGenomescaleMetabolicReconstruction2020 | USUXQJR3 | off-schedule | wrong organism | Issatchenkia orientalis SD108 (Pichia kudriavzevii) model; likely mis-filed, the empty collection database/Pichia-kudriavzevii exists (moving it is the owner's call) | no verdict | `$DATA_ROOT/torchcell-library/suthersGenomescaleMetabolicReconstruction2020/` |

### The two genomes resources

- **O'Donnell 2023**: 197 nuclear assemblies of 142 strains (counted in Supp Table 1; 136 mitochondrial stated), ENA PRJEB59869, no phenotype records. 124 of its strains are 1,011-panel isolates, 123 of them already with a Peter 2018 short-read assembly in `peter2018_1011_assemblies`. The tier would gain telomere-to-telomere contiguity (median 17 contigs per assembly against Peter's 3,259), phased haplotypes for heterozygous diploids and polyploids, mitochondrial genomes, and a 4,809-row non-redundant SV list with inversions, translocations, Ty and tDNA positions. The current natural-isolate encoding (`caudal2024.py`) is gene-level, so intergenic SVs, inversions and Ty insertions have no perturbation representation; whether to re-derive isolate genotypes from T2T assemblies is a design decision, not measured here.
- **Loegler 2025**: 1,482 assemblies over 1,086 isolates (TableS2), Zenodo 15698884 (GenomeAssemblies.tar.gz 16.8 GB) and ENA PRJEB77686 / PRJEB81147. 995 isolates are in the 1,011 panel; 91 are new (90 `XTRA_*` plus JLL). The tier would gain near-T2T assemblies for the whole panel, an SV VCF (1,087 samples), and gene-based and graph pangenomes. Phenotypes: TableS11 lists 8,153 traits (232 growth, 5,816 transcript, 2,105 protein), every one re-served from Peter 2018, Caudal 2024, Muenzner 2024 and Teyssonniere 2024 ("This resource complements previous phenotypic data"); the per-isolate values sit only in a 79.6 MB Zenodo tarball (not measured). 65 of its 100 growth-condition stems (`SC*` and `YNB*`) are attributable to no mirrored paper. Load the originals; use Loegler as a cross-check.

### Turco 2023 under the aggregation-decomposition policy

Yeast Phenome is the knockout-phenome aggregation ([[torchcell.candidates.aggregation-decomposition]]). Measured: the v1.0 release (yp-data commit `83e2917b`, the Zenodo 7714347 freeze) has 517 PMID folders (the paper states 531 studies, 14,495 screens); 516 PMIDs resolve to a DOI; 17 of the 517 papers are in the mirror; 507 are net new against the dataset modules (510 by `gates.module_index()` alone, because `ohya2005.py`, `ozaydin2013.py` and `kemmeren2014.py` declare no DOI constant). Value origin is `derived_by_aggregator`: each paper's published values rescaled per screen to an NPV, whose formula is on `yeastphenome.org/about/project/` (root mean square deviation from a KDE mode, not the MAD statistic). The existing loader keeps 83 screens from 47 studies, 296,777 records, built in dev but not in the adapter map. Ten source studies are already their own datasets and are excluded. The per-source decomposition (each study's own release and an L4 cross-check) was not measured; a 10-study probe found 0 of 10 kept studies with a paper in the mirror. Under the policy, Parsons 2006 and Yoshikawa 2009 load from their own releases and must stay off `SCREENS` (#1000). Verdict: `admissible` as an aggregation (`n_source_studies` 517, `sources` None until decomposed).

### Corrections to schedule rows (measured; the table script is not edited here)

- She 2018 (rank 90): 1,125 F6 segregants of RM11-1a x YJM975 in 26 drug conditions, not "~500 isolates x ~40 conditions"; seq_basis is segregant-WGS.
- Piotrowski 2017 (rank 73): 291 strains in Data Set 5 (pool v2.0 lists 310), not ~157; 11,984 conditions.
- Parsons 2006 (rank 60): 4,111 strains in Table S7, not ~5,000.
- Vaishnav 2022 (rank 56): accessions confirmed, GEO GSE163045 and GSE163866 plus Zenodo 4436477.
- Leutert 2023 (rank 39): PXD034997 is the total proteome; the phosphoproteome is PXD035050.
- Lian 2017 (rank 19): the combinatorial library is the cellulase-display arm; beta-carotene is a five-arm rational design.
- Yoshikawa 2011 (rank 85) is a different paper (G3, DOI 10.1534/g3.111.000695) and is not in Zotero; the collection item is Yoshikawa 2009 (FEMS Yeast Res, ethanol stress), off-schedule.
- Rows naming no host (Lian 2017, Gasch 2000, Leutert 2023, She 2018) are refused by `gate --row` at G2 as `absent_unnamed`; the script supplies the host the paper names.

### Issues filed or updated

| issue | kind | keys |
|---|---|---|
| #989 | schema gap (`dataset`, `before-next-kg-build`) | Gasch 2000: ratio-only two-channel arrays |
| #990 | schema gap (`dataset`, `before-next-kg-build`) | Zhang 2020: yeast promoter assignment, biosensor-proxy readout, CEN.PK assembly |
| #991 | schema gap (`dataset`, `before-next-kg-build`) | Vaishnav 2022: promoter-sequence genotype, `well_id`, sort-seq readout |
| #992 | schema gap (`dataset`) | Khaiwal 2025: cytometry cell features, chronological lifespan |
| #921 (comment) | schema gap, existing | Leutert 2023 shape for the phosphosite leaf designed for Potel 2018 |
| #994 | blocked source (`dataset`, `by-hand`, `collect-paper`) | Lee 2014 FD matrices |
| #995 | blocked source | Trikka 2015 |
| #996 | blocked source | DeLoache 2015 |
| #997 | blocked source | Lian 2017 |
| #998 | blocked source | Guzman 2024 KO key |
| #999 | blocked source | Zackrisson 2016 |
| #70 (labels + comment) | blocked source, existing | She 2018 and Ho 2021 (the deleted Dropbox) |
| #1000 | loader findings (`dataset`) | yeastphenome: scope, hit calls labeled z_score, NPV formula, latent duplicates |

Not filed, because every dataset that needs them is blocked at source: no `ConcentrationUnit` for ppm or mg/L (Ho 2021), and no class for a surface-displayed enzyme activity (Lian 2017). Peter 2018's natural loss-of-function gaps are #987 and #988 (filed by the Peter loader).

### The duplicate Peter 2018 key (reported, not fixed)

The mirror holds the Peter 2018 DOI under two keys: `peterGenomeEvolution10112018` (group library 6582362, item `YBDGXVH6`, collection `database`, with `data/` and `si/`) and `peter1011GenomeEvolution2018` (personal library 9025336, item `L6GIM7C9`, collections `torchcell-main` and `torchcell-in`, paper only; its `paper.pdf` is byte-identical, sha256 `dba932c8...`). Every loader and tier record references the group key: `caudal2024.py` `SEQUENCE_SOURCE`, `scripts/migrate_genomes_tier.py`, the `peter2018_1011_assemblies` manifest, the 018 and embedding scripts. For loaders the duplicate is harmless, and #271 decided to keep group/personal pairs. It is not harmless for the candidate gate: no module declared the Peter DOI as its own when measured, so `gates.citation_key_for_doi` returns the first mirror manifest in scan order, the personal duplicate, and `gate --row "Peter 2018"` without `--citation-key` inventories it and reads G3 `unmeasured` (nothing under `si/` or `data/`); with `--citation-key peterGenomeEvolution10112018`, G3 passes on 77 files. Hypothesis (untested): the Peter loader that landed as PR #993 declares the DOI, which pins the lookup to the group key. O'Donnell 2023 has the same pair (`odonnell142TelomeretotelomereAssemblies2023`, paper only).

### Code change on this branch

`torchcell/literature/sync.py` gains `SCEREVISIAE_COLLECTION = "Saccharomyces-cerevisiae"` in `DEFAULT_COLLECTIONS` ([[torchcell.literature.sync]]), so the nightly sync captures papers filed only in the organism subcollection. `capture_si.py` has no default collection list (`--collection` is always explicit), so it is unchanged.

### Ledger

`python -m torchcell.candidates ledger` refused: `torchcell.candidates.ledger.md already has '## 2026.10.11 - Candidate gate ledger'; one ledger per day`. The ledger note forbids editing a generated section, so the 20 verdicts reach the ledger with the next day's run (one run after the wave lands records every branch's verdicts at once). `ledger --print` renders them now.

### Owner decisions open

1. Gasch 2000 (#989): a ratio-only sibling class, or an optional `expression` on the served class.
2. Whether the four low-yield author requests (#995 Trikka, #996 DeLoache, #997 Lian 2017, #999 Zackrisson) are worth sending.
3. Khaiwal 2025: the privacy review of the CEFIPRA antifungal data before any load.
4. Yoshikawa 2009: load Table S1 from the yp-data copy (scriptable, author file re-saved by a curator in 2013, with a 5% ethanol condition the paper never describes) or capture the publisher original by hand first.
5. Suthers 2020 (Issatchenkia orientalis) to `database/Pichia-kudriavzevii`: a curation move, not made.
6. Whether O'Donnell 2023 and Loegler 2025 become genomes-tier sets now, and whether isolate genotypes are re-derived from T2T assemblies.

### Per-key record

#### Piotrowski 2017 (`piotrowskiFunctionalAnnotationChemical2017`)

- category: a (loadable now)
- records: 3,438,421 strain x compound z-scores; si/si13.xlsx (Supplementary Data Set 5) sheet All_conditions: 291 strains x 11,984 conditions, numeric cells only (48,923 'NA' dropped); replicate-averaged over 3 technical PCR replicates
- genotype basis: 291 diagnostic kanMX deletions in the pdr1 pdr3 snq2 drug-sensitized SGA background (pool v2.0 lists 310; not the ~157 the schedule row assumed)
- overlap: none held: no Piotrowski store; vanacloig2022.py cites it for methods only; yp-data has no folder for PMID 28759014. Same sensitized lineage as env_chemgen_vanacloig2022 (different screens).
- released: si/si13.xlsx (NIHMS884855 Suppl_Dataset_5.xlsx) (mirror (PMC open-data bucket); scriptable; `96e07f956f7edeb07359fc8b1fa5c72a167bf9bb18d1e8a680c76362e97aaefd`)
- released: si/si11.xlsx (Data Set 4: condition metadata, SMILES) (mirror; scriptable; `eba460fe354bafd4e57c2a8438201cdc9399cf97582e3d86adf3e9362f821f87`)
- verdict: admissible (G1 pass, G2 pass, G3 pass, G4 pass, G5 pass)

#### Parsons 2006 (`parsonsExploringModeofActionBioactive2006`)

- category: a (loadable now)
- records: 329,254 strain x compound log2(control/treated) ratios; si/si8.xls (Table S7, mmc8): 4,111 ORFs x 82 conditions, numeric cells (7,848 blank noise-removed cells dropped); companion si/si5.xls (Table S4) holds 339,135 -log10 p over 4,232 ORFs
- genotype basis: MATa haploid YKO (Research Genetics; BY4741 inferred), pooled barcode microarray
- overlap: none held; latent: yp-data v1.0 curates the same Table S7 (PMID 16901791), so yeastphenome SCREENS must skip it (#1000). The ratio is control over treated: a loader negates or records the orientation.
- released: si/si8.xls (mmc8.xls, Table S7 raw matrix) (mirror (Elsevier mmc); scriptable; `564e94c963424687edc24ed23b6e3c6882b70c246a2524057cf1dc5f393c848f`)
- released: si/si5.xls (mmc5.xls, Table S4 p-values) (mirror; scriptable; `0446eae63b5b08af7a1a94fa3b8d42d70f1c75b55a60e0aa6e03fea189753a01`)
- verdict: admissible (G1 pass, G2 pass, G3 pass, G4 pass, G5 pass)

#### Yoshikawa 2009 (`yoshikawaComprehensivePhenotypicAnalysis2009`)

- category: a (loadable now)
- records: 18,916 strain x condition growth records (two replicate rates + a call each); FYR_456_sm_tableS1.xlsx (yp-data v1.0 raw_data copy, sha256 f8567d2c..., read in audit scratch): 4,729 ORFs x 4 conditions; the sensitive and tolerant counts reproduce the paper's Table 2; 14,187 without the 5% ethanol condition the paper never describes
- genotype basis: haploid MATalpha YKO (BY4742; some BY4739)
- overlap: none held; latent: yp-data v1.0 curates the same Table S1 (PMID 19054128). Off-schedule: schedule row 85 is Yoshikawa 2011 (G3, xylose), a different paper not in Zotero.
- released: Table S1 (publisher original) (Wiley/OUP supporting information; walled; bytes not read)
- released: Datasets/19054128/raw_data/FYR_456_sm_tableS1.xlsx (github.com/yeastphenome/yp-data at v1.0; scriptable; `f8567d2cbfe974a87cdb8629086884bc5e3c280513b99ab944065fb53d873871`)
- verdict: pending (G1 pass, G2 pass, G3 unmeasured, G4 pass, G5 pass)

#### Messner 2021 (`messnerUltrafastProteomicsScanning2021`)

- category: a (loadable now)
- records: 95 drug-screen proteome runs (protein x run values not counted); PRIDE PXD023613 file listing: 95 `*_plate_*.wiff` runs, 71 drug-treated over 18 agents + 24 DMSO; the protein values sit in the 741 MB DIA-NN report 20200914_sswath_yeast_drugscreen.tsv
- genotype basis: wild type only (prototrophic BY4741)
- overlap: none: proteome_messner2023 holds no drug or DMSO samples (yeast5k metadata grep 0)
- released: 20200914_sswath_yeast_drugscreen.tsv (DIA-NN report) (PRIDE PXD023613 (FTP); scriptable; bytes not read)
- verdict: admissible (G1 pass, G2 pass, G3 pass, G4 pass, G5 pass)

#### Leutert 2023 (`leutertRegulatoryLandscapeYeast2023`)

- category: b (schema gap)
- records: 529,400 phosphosite x treatment log2 fold changes; si/si9.xlsx (Table S6) sheet p_site_diff_reg read with keep_default_na=False: 5,294 sites x 100 treatments vs untreated, each with p and BH-adjusted p; 0 duplicated pairs
- genotype basis: wild type only (BY4741; BY4743 for the diploid condition)
- overlap: none: no phosphoproteome store; the total-proteome arm (Table S4) fits ProteinFoldChangePhenotype today. The schedule's PXD034997 is the total proteome; the phospho set is PXD035050.
- released: si/si9.xlsx (MOESM9, Table S6) (mirror (Springer ESM); scriptable; `12412d7697e36534699d53c0c75dbb1a138ea0f31519010afd673aea446acebe`)
- released: si/si7.xlsx (MOESM7, Table S4 total proteome, 30 treatments) (mirror; scriptable; `fc5304c701fe8b8afe7362cd1ff0962043dfafa6ab987391d86a87a686fa5cc2`)
- verdict: admissible_with_gaps (G1 pass, G2 pass, G3 pass, G4 pass, G5 gap)

#### Vaishnav 2022 (`vaishnavEvolutionEvolvabilityEngineering2022`)

- category: b (schema gap)
- records: 75,487 sequence x expression-level records (MolEvol set; 20,616,659 + 30,722,376 random sequences stated, not counted); GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz (sha256 9193507a..., read in audit scratch): 75,488 lines minus the header; the full random libraries are 0.67 to 3.59 GB files on GEO and Zenodo
- genotype basis: promoter library: an 80 bp variable insert in a YFP reporter promoter on a CEN plasmid (Y8205 and S288C::ura3)
- overlap: none in yeast; the defined-medium test set is labeled NBT_S288CdU_SCUra, which may be the de Boer 2020 set (inference, unmeasured)
- released: GSE163045 and GSE163866 supplementary files (GEO (ftp.ncbi.nlm.nih.gov listing; acc.cgi serves a reCAPTCHA); scriptable; bytes not read)
- released: record 4436477 (31 files) (Zenodo; scriptable; bytes not read)
- verdict: admissible_with_gaps (G1 pass, G2 pass, G3 pass, G4 pass, G5 gap)

#### Khaiwal 2025 (`khaiwalPredictingNaturalVariation2025`)

- category: b (schema gap)
- records: 28,435 isolate x trait values for the 33 new phenotypes; si/si3.xlsx (Dataset EV1): phenotypes annotated 'Not Published' (33), numeric cells over 1,011 strain rows: 22 antifungal traits 20,104, 8 cytometry features 6,616, 3 lifespan timepoints 1,715
- genotype basis: natural isolates (the 1,011 panel)
- overlap: the 190 re-served traits duplicate Peter 2018 (35), De Chiara 2022 (99), Galardini 2019 (47) and D'Angiolo 2023 (9): load those from the originals; the 33 new traits overlap no store. The antifungal data carry 'do not distribute' terms in their CEFIPRA README: privacy review first.
- released: si/si3.xlsx (MOESM3, Dataset EV1) (mirror (PMC open-data bucket); scriptable; `de4c39640239dcf76ac839cf2d143fc96d5b6b01907929ea0ec9f5d259722055`)
- verdict: admissible_with_gaps (G1 pass, G2 pass, G3 pass, G4 pass, G5 gap)

#### Gasch 2000 (`gaschGenomicExpressionPrograms2000`)

- category: b (schema gap)
- records: 1,032,274 gene x array log2 ratios (173 arrays x 6,152 genes); complete_dataset.txt (Wayback capture 20060110024849, sha256 3050664e..., read in audit scratch): 173 array columns, 155 wild type + 8 deletion + 3 overexpression + 7 DeRisi 1997
- genotype basis: wild type (DBY7286, DBY8768) plus yap1 and msn2 msn4 deletions and MSN2/MSN4/YAP1 overexpression
- overlap: none: no SPELL store and $DATA_ROOT/data/sgd/spell is empty (#982); Kemmeren and Sameith are deletion-vs-pool arrays
- released: complete_dataset.txt (the authors' file) (Wayback Machine copy of genome-www.stanford.edu/yeast_stress (original site gone); scriptable; `3050664e0deba976b70b954676675cce74d3288198d850abb6fb123bac58daa7`)
- released: Gasch_2000_PMID_11102521.zip (SGD-reprocessed, 138 columns, no mutant arrays) (sgd-archive.yeastgenome.org (SPELL); scriptable; `31f41d08b1f7e2bcded2933518eb99809b39c2a1a34f8c9e9b56f6b61ee8d5ab`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### Zhang 2020 (`zhangCombiningMechanisticMachine2020`)

- category: b (schema gap)
- records: 341 promoter-combination strain records (GFP synthesis rate) + 6 HPLC titers; raw mirror ice.csv (299 library strains with a full five-promoter genotype, 275 distinct) + recommendations.csv (42 strains, disjoint); edd.csv carries 75,026 GFP rows, 3 replicate lines per strain; si/si1.md holds the 6 titers
- genotype basis: engineered strain: CEN.PK2-1C platform with five genes re-integrated under one of six promoters each (6^5 designs)
- overlap: none found
- released: data/ice.csv.zip, data/edd.csv.zip, data/recommendations.csv, data/biosensor_val.csv (raw mirror (GitHub commit f95f7d10); manifest sha256 all reproduce; scriptable; `d1ea7cb4c6189d444ab0d0a24c5e5dbd48689b634cf52d6afc1435841e3c9081`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### O'Donnell 2023 (`odonnellTelomeretotelomereAssemblies1422023`)

- category: d (genomes tier)
- records: 197 nuclear assemblies (0 phenotype records); si/si4.xlsx sheet 'Supp Table 1': 197 rows (142 haploid or collapsed, 42 diploid, 6 triploid, 7 tetraploid haplotypes); 136 mitochondrial assemblies stated
- genotype basis: natural isolates and lab strains; 124 of the 142 strains are 1,011-panel isolates, 123 of them with a Peter 2018 assembly already in the tier
- overlap: genomes only; 71 of its assemblies are reused verbatim by Loegler 2025. A second mirror key, odonnell142TelomeretotelomereAssemblies2023, holds the same DOI (personal library; #271 keeps such pairs).
- released: assemblies PRJEB59413 (unphased), PRJEB59129 / PRJEB59231 / PRJEB59232 (haplotypes), PRJEB59230 (mitochondria) (ENA; scriptable; bytes not read)
- released: si/si4.xlsx (Supplementary Tables 1-17) (mirror (PMC open-data bucket); scriptable; `6e1ca79129743ee03ad6d7241d7636b3f8aaadc67f79cd4cd84dce61f765e20f`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### Loegler 2025 (`loeglerGenotypePhenotype10862025`)

- category: d (genomes tier)
- records: not measured; not measured: the per-isolate values are only in Zenodo Phenotypes_8391Traits.tar.gz (79.6 MB); measured instead: si/si3.xlsx TableS11 8,153 traits (232 growth, 5,816 transcript, 2,105 protein), TableS2 1,482 assemblies over 1,086 isolates
- genotype basis: natural isolates: 995 are 1,011-panel isolates, 91 are not (90 `XTRA_*` plus JLL)
- overlap: HIGH for phenotypes: growth duplicates Peter 2018 (landed as PR #993), transcripts Caudal 2024, proteins Muenzner 2024 (wave 1) and Teyssonniere 2024. Genomes: 994 isolates already have a Peter assembly; 71 assemblies are O'Donnell 2023's.
- released: GenomeAssemblies.tar.gz (16.8 GB), SVcalls.tar.gz, pangenomes, Phenotypes_8391Traits.tar.gz (Zenodo 15698884; scriptable; bytes not read)
- released: si/si3.xlsx (Supplementary Tables S1-S15) (mirror (PMC open-data bucket); scriptable; `ae46f4663ad1e9dcba901096623d598f0de4faaa6426015b3a9ed4c31402bd55`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### Turco 2023 (`turcoGlobalAnalysisYeast2023`)

- category: e (aggregation)
- records: 296,777 strain x screen NPVs the loader keeps (83 screens, 47 studies); the 47 dev-store `raw/*_valuez.txt` files (each equal to its SCREENS sha256 pin), the loader's column filter; equals verification_report.json L1. The release: 517 PMID folders at yp-data v1.0 (GitHub API listing, sha256 34fdc938...); the paper states 531 studies and 14,495 screens
- genotype basis: YKO collections: haploid BY4741 and BY4742, homozygous diploid BY4743 (het screens dropped)
- overlap: the yeastphenome dev store IS this loader (built, not served). 10 source studies are their own dev stores and are excluded; Parsons 2006 and Yoshikawa 2009 would duplicate if SCREENS widened (#1000). 517 sources, 17 mirrored, 507 net new; decomposition not measured.
- released: yp-data v1.0 (commit 83e2917b), Datasets/<PMID>/ (GitHub / Zenodo 7714347; scriptable; bytes not read)
- released: NPV bulk downloads (Zenodo 7714354; scriptable; bytes not read)
- verdict: admissible (G1 pass, G2 pass, G3 pass, G4 pass, G5 pass)

#### Lee 2014 (`leeMappingCellularResponse2014`)

- category: f (blocked at source)
- records: not measured; not measurable: every sheet of the ten workbooks was opened; the closest are 455 clearance scores for selected HIP hits (s2) and median FD per signature gene (s4)
- genotype basis: heterozygous (HIP) and homozygous (HOP) diploid YKO pools, BY4743
- overlap: env_chemgen_lee2014 holds raw/ only (no LMDB); yeastphenome has no Lee values
- released: HIPHOP portal download (FD matrices) (chemogenomics.pharmacy.ubc.ca; vanished; bytes not read)
- released: 1250217s2.xlsx (sha256 shown) and nine more Science workbooks (legacy dev-store raw dir (not in any mirror manifest); walled; `6a74854a3e7ef3802e7cc3ed355e98bc168882387f36bff4159b8ff8c262a0c7`)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### Guzman 2024 (`guzmanUltrafastLabelfreeQuantification2024`)

- category: f (blocked at source)
- records: 1,375,123 protein x knockout-run values (genotypes unreleased); si/si13.xlsx sheet 'KO PG collection ': 4,887 non-contaminant protein groups x 312 KO run columns, non-null cells
- genotype basis: 104 BY4743 knockout-collection strains, identities unreleased
- overlap: proteome_messner2023 may share genes (unverifiable without the key); different strains and samples
- released: si/si13.xlsx (MOESM13) (mirror (PMC open-data bucket); scriptable; `6772df03faa6c9feb1d12dfcf3230248895e0a755424403d1fe01c0f884db1b5`)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### Ho 2021 (`hoMassiveQTLAnalysis2021`)

- category: f (blocked at source)
- records: 20,377 segregant x trait values (genotypes blocked); si/si_data/13068_2021_2059_MOESM3_ESM.xlsx sheet 'For the mapping experiments': 1,152 rows x 20 trait columns, numeric cells (25,317 with the five NOIS ester columns)
- genotype basis: segregants (F6 advanced intercross RM11-1a x YJM975)
- overlap: none (bloom2019 crosses do not use YJM975); same segregants as She 2018
- released: si/si_data/13068_2021_2059_MOESM3_ESM.xlsx (mirror (Springer ESM); scriptable; `7857557eaeb21591edf16141cb9a13c6eb4219343c05564be67808fc80256e74`)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### She 2018 (`sheMappingCausalVariants2018`)

- category: f (blocked at source)
- records: not measured; not measurable: the Cell SI holds only Table S3 (727 causal-variant rows over 28 condition labels)
- genotype basis: segregants (1,125 F6 RM11-1a x YJM975), not the ~500 isolates the schedule row says
- overlap: none (bloom2019 crosses do not use YJM975)
- released: Dropbox 'additional data' (dropbox.com; vanished; bytes not read)
- released: PRJNA387489 reads (SRA; scriptable; bytes not read)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### Zackrisson 2016 (`zackrissonScanomaticHighResolutionMicrobial2016`)

- category: f (blocked at source)
- records: not measured; not measurable: no per-strain file in the release
- genotype basis: haploid YKO (BY4742)
- overlap: none held
- released: 21 SI files (figures, File S1, movies) (mirror (PMC open-data bucket); scriptable; bytes not read)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### Lian 2017 (`lianCombinatorialMetabolicEngineering2017`)

- category: f (blocked at source)
- records: not measured; not measurable: pdftotext of si/si1.pdf; no table carries a phenotype number
- genotype basis: engineered strain: CEN.PK2-1C with the CRISPR-AID cassette, combinatorial a/i/d guides
- overlap: none: crispr_magic_lian2019 is the same platform on BY4742 with guide-enrichment records
- released: si/si1.pdf (Supplementary Information); si/si2.pdf is the peer-review file (mirror (PMC open-data bucket); scriptable; `9f81bc16c67ef2a0fa1bad6345fdc742d0af879d59071118bf9683d5ac05d930`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### DeLoache 2015 (`deloacheEnzymecoupledBiosensorEnables2015`)

- category: f (blocked at source)
- records: not measured; not measurable: 54 strain rows in SI Table 1 (pdftotext), none with a value
- genotype basis: engineered strain: BY4741 with integrated heterologous BIA pathway constructs and CYP76AD1 point mutants
- overlap: none: betaxanthin_cachera2023 is a deletion-collection screen
- released: si/si1.pdf (Supplementary Text and Figures) (mirror (Zotero attachment); unknown; `fa5922dc23976e7c9edbf4cdce8e37141ff1074196191b80804760e7c902779b`)
- verdict: blocked:wall (G1 pass, G2 pass, G3 blocked:wall, G4 unmeasured, G5 unmeasured)

#### Trikka 2015 (`trikkaIterativeCarotenogenicScreens2015`)

- category: f (blocked at source)
- records: not measured; not measurable: si/si1.doc text extracted; 100 ORF cells, 0 numeric cells
- genotype basis: heterozygous EG60 x BY4741 hybrids (not BY4743)
- overlap: none: carotenoid_ozaydin2013 is a different screen on haploid BY4741
- released: si/si1.doc (Additional file 1) (mirror (Springer ESM); scriptable; `e5587f1eeb0148bca6d7c54bbc4038aaf0b495bbdfe6ad9d3a75c2c58e715b9f`)
- verdict: blocked:provision (G1 pass, G2 blocked:provision, G3 unmeasured, G4 unmeasured, G5 unmeasured)

#### Brettner 2024 (`brettnerUltraHighthroughputMassively2024`)

- category: c (not a dataset)
- records: not measured; not pursued: GEO GSE251966 holds unfiltered matrices of three technical validation runs
- genotype basis: wild-type S288C / BY4741 and two reporter strains, no perturbation axis
- overlap: none
- released: GSE251966 (GEO; scriptable; bytes not read)

#### Wagih 2013 (`wagihSGAtoolsOnestopAnalysis2013`)

- category: c (not a dataset)
- records: not measured; tool paper
- genotype basis: none
- overlap: none

#### Kuzmin 2016 (`kuzminSyntheticGeneticArray2016`)

- category: c (not a dataset)
- records: not measured; protocol
- genotype basis: none
- overlap: none

#### Tong 2006 (`yantongSyntheticGeneticArray2006`)

- category: c (not a dataset)
- records: not measured; protocol (methods chapter; the Zotero item has no DOI)
- genotype basis: none
- overlap: none

#### Engel 2022 (`engelNewDataCollaborations2022`)

- category: c (not a dataset)
- records: not measured; database update paper
- genotype basis: none
- overlap: SGD content is already consumed: GeneEssentialitySgdDataset (sgd.py, which cites no release paper) and the genomes-tier set sgd_S288C_R64-4-1_20230830

#### Wang 2022 (SynLethDB 2.0) (`wangSynLethDB20Webbased2022`)

- category: c (not a dataset)
- records: not measured; database paper of a served aggregation
- genotype basis: none
- overlap: already loaded: synth_leth_db.py names it SYNLETHDB_CITATION_KEY; dev stores synth_lethality_yeast_synth_leth_db and synth_rescue_yeast_synth_leth_db. Its aggregation verdict is the piece-3 backfill, not this audit

#### Long 2026 (EnzEngDB) (`longEnzymeEngineeringDatabase2026`)

- category: c (not a dataset)
- records: not measured; database paper
- genotype basis: none
- overlap: none

#### Yuan 2026 (OED) (`yuanOpenEnzymeDatabase2026`)

- category: c (not a dataset)
- records: not measured; database paper
- genotype basis: none
- overlap: none

#### Aranguiz 2025 (`aranguizMachineLearningReveals2025`)

- category: g (wrong organism)
- records: 15 S. cerevisiae strain x condition values; si/si12.xlsx source data: 5 S288C deletion strains x 3 oxidants; the screen itself is 284 species x 2 doses (si/si3.xlsx), one value per species
- genotype basis: a species panel (284 Saccharomycotina species); S. cerevisiae only as 5 deletion strains
- overlap: none

#### Suthers 2020 (`suthersGenomescaleMetabolicReconstruction2020`)

- category: g (wrong organism)
- records: not measured; genome-scale model paper
- genotype basis: none
- overlap: none

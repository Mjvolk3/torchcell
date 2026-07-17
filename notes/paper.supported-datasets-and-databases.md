---
id: y7lp8tw60a3vezhlczyrzf4
title: Supported Datasets and Databases
desc: ''
updated: 1783460208823
created: 1783460208823
---

## Supported datasets + databases (paper table)

<!-- GENERATED: raw data from experiments/database/scripts/build_supported_datasets_table.py
(-> results/pre-build/<date>/supported_datasets.json); this note + the paper LaTeX are
VIEWS rendered by render_supported_datasets_table.py. Do not edit tables by hand. -->

Pre-build inventory of the datasets schematized + L0-L4 verified as LMDBs (not yet a
versioned Neo4j DB build). For the reader to grasp the **diversity and scale** of the
training signal; the full backlog is `[[paper.north-star.dataset-triage]]`.

**Columns.** *Genotypes* = distinct perturbed strains/isolates (curated). *Env* = number
of environments. *Instances* = dataset length (total genotype×environment records).
*Shape* = shape of a single phenotype instance (`scalar` / `vector (D)`). *Graph role* =
where the label sits in the cell graph (`global` / `node` / `edge` / `hyperedge` /
`bipartite node`; a digenic interaction is an `edge`, a trigenic one a `hyperedge`).
*Signal (gzip, bytes)* = scientific-notation gzip size of the concatenated stored
instances (the **perturbation** + environment + phenotype of every record) -- a
Kolmogorov-complexity proxy. The perturbation counts each instance's edit off the S288C
reference: a single deletion (a few bytes) or a natural isolate's thousands of gene-presence
entries that amount to a new genome (sequence stays external, referenced by uri+sha256). The
shared reference genome is never counted. Instances, Shape, Graph role, and Signal are
**derived from the built LMDB**, not hand-typed.

### Fitness + genetic interaction

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Costanzo 2016 smf | 20,484 | 2 | 20,484 | single-mutant fitness | scalar | global | 6.8×10⁵ |
| Costanzo 2016 dmf | 20.7M | 2 | 20,705,612 | double-mutant fitness | scalar | global | 8.9×10⁸ |
| Costanzo 2016 dmi | 20.7M | 2 | 20,705,612 | digenic interaction | scalar | edge | 7.3×10⁸ |
| Kuzmin 2018 smf | 1,539 | 1 | 1,539 | single-mutant fitness | scalar | global | 4.3×10⁴ |
| Kuzmin 2018 dmf | 410,399 | 1 | 410,399 | double-mutant fitness | scalar | global | 1.9×10⁷ |
| Kuzmin 2018 tmf | 91,111 | 1 | 91,111 | triple-mutant fitness | scalar | global | 4.8×10⁶ |
| Kuzmin 2018 dmi | 410,399 | 1 | 410,399 | digenic interaction | scalar | edge | 1.6×10⁷ |
| Kuzmin 2018 tmi | 91,111 | 1 | 91,111 | trigenic interaction | scalar | hyperedge | 4.0×10⁶ |
| Kuzmin 2020 smf | 472 | 1 | 472 | single-mutant fitness | scalar | global | 1.1×10⁴ |
| Kuzmin 2020 dmf | 632,797 | 1 | 632,797 | double-mutant fitness | scalar | global | 2.9×10⁷ |
| Kuzmin 2020 tmf | 301,798 | 1 | 301,798 | triple-mutant fitness | scalar | global | 1.3×10⁷ |
| Kuzmin 2020 dmi | 632,797 | 1 | 632,797 | digenic interaction | scalar | edge | 2.4×10⁷ |
| Kuzmin 2020 tmi | 301,798 | 1 | 301,798 | trigenic interaction | scalar | hyperedge | 1.3×10⁷ |
| Baryshnikova 2010 (smf) | 5,993 | 1 | 5,993 | single-mutant fitness | scalar | global | 2.1×10⁵ |
| O'Duibhir 2014 (smf) | 1,312 | 1 | 1,312 | single-mutant fitness | scalar | global | 3.3×10⁴ |

### Environmental / chemogenomic

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Auesukaree 2009 (stress screen) | 333 | 6 | 525 | stress sensitivity (categorical) | scalar | global | 1.1×10⁴ |
| Mota 2024 (weak-acid screen) | 601 | 3 | 1,273 | weak-acid susceptibility (categorical) | scalar | global | 2.4×10⁴ |
| Vanacloig-Pedros 2022 | 3,647 | 45 | 164,115 | chemogenomic fitness (log2-ratio) | scalar | global | 1.0×10⁷ |
| Costanzo 2021 (condition-SGA) | 4,399 | 14 | 61,318 | differential mutant fitness | scalar | global | 1.1×10⁶ |
| Hillenmeyer 2008 het (FitDb HIP) | 5,814 | 514 | 2,921,078 | HIP fitness-defect log2-ratio | scalar | global | 1.0×10⁸ |
| Hillenmeyer 2008 hom (FitDb HOP) | 4,667 | 279 | 1,179,520 | HOP fitness-defect z-score | scalar | global | 4.3×10⁷ |
| Wildenhain 2015 (drug tolerance) | 256 | 5,178 | 428,573 | growth-inhibition z-score | scalar | global | 1.9×10⁷ |
| Hoepfner 2014 (HIP/HOP atlas) | 10,719 | 5,879 | 3,112,880 | HIP/HOP sensitivity score | scalar | global | 1.8×10⁸ |
| Smith 2006 (chemogenomic) | 4,721 | 3 | 14,163 | chemogenomic sensitivity (clear-zone ordinal) | scalar | global | 2.1×10⁵ |
| Lian 2019 (MAGIC CRISPR-AID) | 266,415 | 3 | 266,415 | furfural tolerance fitness (log2-ratio) | scalar | global | 1.6×10⁷ |
| Mormino 2022 (CRISPRi acetic-acid) | 12 | 1 | 12 | acetic-acid sensitivity (categorical) | scalar | global | 1.1×10³ |
| Smith 2016 (CRISPRi chem-genetic) | 1,035 | 26 | 14,463 | chemogenomic fitness (log2-ratio) | scalar | global | 6.0×10⁵ |

### Viability

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| SGD essentiality | 1,329 | 1 | 1,329 | gene essentiality | scalar | node | 1.5×10⁴ |
| SynLethDB (lethal) | 14,000 | 1 | 14,000 | synthetic lethality | scalar | edge | 2.7×10⁵ |
| SynLethDB (rescue) | 6,948 | 1 | 6,948 | synthetic rescue | scalar | edge | 1.4×10⁵ |

### Morphology

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Ohya 2005 (SCMD CalMorph) | 4,718 | 1 | 4,718 | cell morphology (CalMorph) | vector (281) | global | 1.9×10⁷ |
| Ohnuki 2018 (SCMD CalMorph) | 1,112 | 1 | 1,112 | cell morphology (CalMorph) | vector (281) | global | 6.0×10⁶ |
| Ohnuki 2022 (SCMD CalMorph) | 1,979 | 1 | 1,979 | cell morphology (CalMorph) | vector (281) | global | 1.1×10⁷ |

### Expression (microarray)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Kemmeren 2014 | 1,484 | 1 | 1,484 | mRNA log2(mut/wt) | vector (6169) | node | 4.0×10⁸ |
| Sameith 2015 sm | 82 | 1 | 82 | mRNA log2(mut/ref) | vector (6169) | node | 2.3×10⁷ |
| Sameith 2015 dm | 72 | 1 | 72 | mRNA log2(mut/ref) | vector (6169) | node | 2.1×10⁷ |

### Expression (RNA-seq)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Caudal 2024 (pan-transcriptome) | 943 | 1 | 943 | mRNA abundance (RNA-seq) | vector (6000) | node | 1.9×10⁸ |
| Nadal-Ribelles 2025 (Perturb-seq) | 3,150 | 2 | 6,188 | mRNA logFC (Perturb-seq) | vector (5639) | node | 2.8×10⁸ |

### Metabolite

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Cachera 2023 (CRI-SPA betaxanthin) | 4,735 | 1 | 4,735 | betaxanthin (product proxy) | scalar | bipartite node | 2.8×10⁵ |
| Mülleder 2016 (amino-acid metabolome) | 4,678 | 1 | 4,678 | amino-acid concentrations | vector (19) | bipartite node | 1.0×10⁶ |
| Zelezniak 2018 (metabolome) | 95 | 1 | 95 | metabolite levels | vector (25) | bipartite node | 3.8×10⁴ |
| Ozaydin 2013 (β-carotene screen) | 4,474 | 1 | 4,474 | β-carotene (colony-color visual score) | scalar | global | 1.3×10⁵ |
| da Silveira 2014 (lipidomics) | 127 | 1 | 127 | lipid-species relative abundance | vector (135) | bipartite node | 1.3×10⁵ |
| Yoshida 2012 (organic acids) | 17 | 1 | 17 | organic-acid titer | vector (6) | bipartite node | 1.7×10³ |
| Xue 2025 (free fatty acids, private) | 176 | 1 | 176 | free-fatty-acid titer | vector (5) | bipartite node | 2.2×10⁴ |
| Lopez 2024 (isobutanol screen, private) | 4,554 | 1 | 4,554 | isobutanol biosensor fold-change | scalar | bipartite node | 1.1×10⁵ |
| Lopez 2024 (isobutanol validated, private) | 224 | 1 | 224 | isobutanol biosensor fold-change (validated) | scalar | bipartite node | 9.3×10³ |

### Protein abundance

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Zelezniak 2018 (SWATH proteome) | 97 | 1 | 97 | protein abundance | vector (726) | node | 2.1×10⁶ |
| Messner 2023 (proteome) | 4,699 | 1 | 4,699 | protein abundance | vector (1830) | node | 1.4×10⁸ |

### Total

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (49 datasets)** |  |  | **52,540,300** |  |  |  | **3.2×10⁹** |

### In progress (not yet built/verified)

Ho 2009 (bioethanol tolerance -- genotype needs a WGS pipeline) · Trikka 2015 (sclareol titers -- figure-only data). See `[[paper.north-star.dataset-triage]]` for the full ~75-candidate backlog.

## Reference databases

Curated resources torchcell reads from or cross-references. Not per-strain datasets.

| Database | Type | What it provides | URL |
| :-- | :-- | :-- | :-- |
| SGD | genome / knowledgebase | reference genome, annotation, phenotype + literature curation | yeastgenome.org |
| **SPELL** | expression compendium (SGD) | search engine over 752 datasets / 15,475 arrays / 576 studies | spell.yeastgenome.org |
| YMDB 2.0 | metabolite database | curated yeast metabolite structures, concentrations, pathways | ymdb.ca |
| YeastNet v3 | functional gene network | probabilistic integrated gene-interaction network | inetbio.org/yeastnet |
| CYCLoPs / LoQAtE | localization + abundance | GFP-collection protein localization/abundance atlases | thecellvision / weizmann |
| TheCellMap.org | genetic-interaction portal | query/download for the Costanzo/Boone global GI network | thecellmap.org |
| Yeast9 GEM | genome-scale metabolic model | consensus stoichiometric reconstruction (metabolite-node IDs) | github SysBioChalmers/yeast-GEM |
| ScRAPdb | pan-omics assembly panel | 142-strain telomere-to-telomere reference panel omics | evomicslab.org/db/ScRAPdb |
| Yeast PeptideAtlas | proteome-observation DB | reprocessed MS peptide/protein observation confidence | peptideatlas.org/builds/yeast |

## Maintenance + provenance

- **Two steps, regenerated never hand-edited.** `build_supported_datasets_table.py` scans
  the LMDBs -> `results/pre-build/<date>/supported_datasets.json` (raw data);
  `render_supported_datasets_table.py` renders this note + the paper LaTeX + a preview PDF
  off that JSON; `plot_supported_datasets_signal.py` draws instances-vs-signal. Per-dataset
  spot check: `python -m torchcell.paper.signal <subpath>`.
- **Signal caveat.** The gzip number includes phenotype metadata (SE, n_replicates,
  measurement_type), so it is a *relative* proxy, comparable across datasets, not exact.
- Backlog + differentiation: `[[paper.north-star.dataset-triage]]`, `[[paper.north-star]]`.

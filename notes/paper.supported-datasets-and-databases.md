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
where the label sits in the cell graph (`global` / `node` / `hyperedge` / `bipartite
node`). *Signal (gzip, bytes)* = scientific-notation gzip size of the concatenated
serialized phenotypes -- a Kolmogorov-complexity proxy. Instances, Shape, Graph role, and
Signal are **derived from the built LMDB**, not hand-typed.

### Fitness + genetic interaction

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Costanzo 2016 smf | 20,484 | 2 | 20,484 | single-mutant fitness | scalar | global | 1.0×10⁵ |
| Costanzo 2016 dmf | 20.7M | 2 | 20,705,612 | double-mutant fitness | scalar | global | 1.3×10⁸ |
| Costanzo 2016 dmi | 20.7M | 2 | 20,705,612 | digenic interaction | scalar | hyperedge | 1.5×10⁸ |
| Kuzmin 2018 smf | 1,539 | 1 | 1,539 | single-mutant fitness | scalar | global | 5.1×10³ |
| Kuzmin 2018 dmf | 410,399 | 1 | 410,399 | double-mutant fitness | scalar | global | 2.6×10⁶ |
| Kuzmin 2018 tmf | 91,111 | 1 | 91,111 | triple-mutant fitness | scalar | global | 5.7×10⁵ |
| Kuzmin 2018 dmi | 410,399 | 1 | 410,399 | digenic interaction | scalar | hyperedge | 3.6×10⁶ |
| Kuzmin 2018 tmi | 91,111 | 1 | 91,111 | trigenic interaction | scalar | hyperedge | 7.4×10⁵ |
| Kuzmin 2020 smf | 472 | 1 | 472 | single-mutant fitness | scalar | global | 2.6×10³ |
| Kuzmin 2020 dmf | 632,797 | 1 | 632,797 | double-mutant fitness | scalar | global | 4.0×10⁶ |
| Kuzmin 2020 tmf | 301,798 | 1 | 301,798 | triple-mutant fitness | scalar | global | 1.9×10⁶ |
| Kuzmin 2020 dmi | 632,797 | 1 | 632,797 | digenic interaction | scalar | hyperedge | 5.4×10⁶ |
| Kuzmin 2020 tmi | 301,798 | 1 | 301,798 | trigenic interaction | scalar | hyperedge | 2.6×10⁶ |

### Viability

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| SGD essentiality | 1,329 | 1 | 1,329 | gene essentiality | scalar | node | 5.8×10² |
| SynLethDB (lethal) | 14,000 | 1 | 14,000 | synthetic lethality | scalar | hyperedge | 1.6×10⁴ |
| SynLethDB (rescue) | 6,948 | 1 | 6,948 | synthetic rescue | scalar | hyperedge | 5.7×10³ |

### Morphology

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Ohya 2005 (SCMD CalMorph) | 4,718 | 1 | 4,718 | cell morphology (CalMorph) | vector (281) | global | 1.9×10⁷ |

### Expression (microarray)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Kemmeren 2014 | 1,484 | 1 | 1,484 | mRNA log2(mut/wt) | vector (6169) | node | 4.0×10⁸ |
| Sameith 2015 sm | 82 | 1 | 82 | mRNA log2(mut/ref) | vector (6169) | node | 2.3×10⁷ |
| Sameith 2015 dm | 72 | 1 | 72 | mRNA log2(mut/ref) | vector (6169) | node | 2.1×10⁷ |

### Expression (RNA-seq)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Caudal 2024 (pan-transcriptome) | 943 | 1 | 943 | mRNA abundance (RNA-seq) | vector (6000) | node | 9.4×10⁷ |

### Metabolite

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Cachera 2023 (CRI-SPA betaxanthin) | 4,735 | 1 | 4,735 | betaxanthin (product proxy) | scalar | bipartite node | 1.2×10⁵ |
| Mülleder 2016 (amino-acid metabolome) | 4,678 | 1 | 4,678 | amino-acid concentrations | vector (19) | bipartite node | 9.1×10⁵ |
| Zelezniak 2018 (metabolome) | 95 | 1 | 95 | metabolite levels | vector (25) | bipartite node | 3.6×10⁴ |

### Protein abundance

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Zelezniak 2018 (SWATH proteome) | 97 | 1 | 97 | protein abundance | vector (726) | node | 2.0×10⁶ |

### Visual / product-proxy score

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bytes) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Ozaydin 2013 (β-carotene screen) | 4,474 | 1 | 4,474 | colony-color visual score | scalar | global | 2.2×10⁴ |

### In progress (not yet built/verified)

Baryshnikova 2010 (smf; liquid-growth assay -- MinerU the paper first) · Ohnuki 2018 / 2022 (morphology) · O'Duibhir 2014 (expression) · Wildenhain 2015 (drug tolerance, 195 × 4,915 conditions) · Lian 2017 (AID furfural tolerance) · FitDb (fitness across 1,144 conditions). See `[[paper.north-star.dataset-triage]]` for the full ~75-candidate backlog.

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

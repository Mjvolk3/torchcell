---
id: y7lp8tw60a3vezhlczyrzf4
title: Supported Datasets and Databases
desc: ''
updated: 1783460208823
created: 1783460208823
---

## Supported datasets + databases (paper table)

Readable, paper-facing inventory of the datasets currently schematized and built in the
torchcell database (all L0-L4 verified; "supported" is implied). Deliberately less
detailed than the full candidate backlog `[[paper.north-star.dataset-triage]]` -- this
table is for the reader to grasp the **diversity and scale** of the training signal, not
to plan ingestion.

**Columns.** *Genotypes* = distinct perturbed strains/combinations. *Env* = number of
environments/conditions. *Shape* = genotypes × label dimensionality (scalar, a k-vector,
or a graph edge/hyperedge). *Signal (gzip)* = gzip-compressed size of the serialized
per-strain phenotype values -- a Kolmogorov-complexity proxy for the total information in
the dataset (captures breadth × depth in one number). Numbers come **from the built LMDB**
and are refreshed when a dataset is (re)built; large fitness/interaction LMDBs (20M+
records) are marked *pending* rather than estimated.

### Fitness + genetic interaction

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Baryshnikova 2010 | 6,022 | 1 | single-mutant fitness | scalar | n×1 | pending |
| Costanzo 2016 smf | 20,484 | 2 | single-mutant fitness | scalar | n×1 | 101 KB |
| Costanzo 2016 dmf | 20.7M | 2 | double-mutant fitness | scalar | n×1 | pending (43 GB LMDB) |
| Costanzo 2016 dmi | 20.7M | 2 | digenic interaction | edge | n×1 | pending (45 GB LMDB) |
| Kuzmin 2018 smf/dmf/tmf | 1,539 / 410k / 91k | 1 | 1-/2-/3-mutant fitness | scalar | n×1 | smf 5 KB; dmf/tmf pending |
| Kuzmin 2018 dmi/tmi | 410k / 91k | 1 | di-/trigenic interaction | edge / hyperedge | n×1 | pending |
| Kuzmin 2020 smf…tmi | 480 – 538k | 1 | fitness + interaction | scalar / edge | n×1 | pending |

### Viability

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| SGD essentiality | 1,329 | 1 | gene essentiality | bool | n×1 | 0.6 KB |
| SynLethDB (lethal) | 14,000 | 1 | synthetic lethality | bool | n×1 | 15 KB |
| SynLethDB (rescue) | 6,948 | 1 | synthetic rescue | bool | n×1 | 5.5 KB |

### Morphology

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Ohya 2005 (SCMD CalMorph) | 4,718 | 1 | cell morphology | 254-vector | n×254 | pending |
| Ohnuki 2018 / 2022 | 1,112 / 1,982 | 1 | cell morphology | 254-vector | n×254 | pending |

### Expression (microarray)

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Kemmeren 2014 | 1,484 | 1 | mRNA log2(mut/wt) | ~6,000-gene vector | n×~6000 | pending |
| Sameith 2015 sm | 82 | 1 | mRNA log2(mut/ref) | 6,169-gene vector | 82×6169 | 22 MB |
| Sameith 2015 dm | 72 | 1 | mRNA log2(mut/ref) | 6,169-gene vector | 72×6169 | 20 MB |
| O'Duibhir 2014 | 1,312 | 1 | expression / fitness | gene vector | n×g | pending |

### Metabolite

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Cachera 2023 (CRI-SPA betaxanthin) | 4,735 | 1 | betaxanthin (product proxy) | scalar | n×1 | 115 KB |
| Mülleder 2016 (amino-acid metabolome) | 4,678 | 1 | 19 amino-acid concentrations | 19-vector | 4678×19 | 893 KB |
| Zelezniak 2018 (metabolome) | 97 | 1 | ~46 metabolite levels | ~46-vector | 97×~46 | not yet ingested |

### Protein abundance

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Zelezniak 2018 (SWATH proteome) | 97 | 1 | protein abundance | 726-protein vector | 97×726 | 1.9 MB |

### Visual / product-proxy score

| Dataset | Genotypes | Env | Phenotype | Label | Shape | Signal (gzip) |
| :-- | --: | --: | :-- | :-- | :-- | --: |
| Ozaydin 2013 (β-carotene screen) | 4,474 | 1 | colony-color visual score | ordinal scalar | n×1 | 20 KB |

### In progress (not yet built/verified)

Wildenhain 2015 (drug tolerance, 195 × 4,915 conditions) · Lian 2017 (AID furfural
tolerance) · FitDb (fitness across 1,144 conditions). See
`[[paper.north-star.dataset-triage]]` for the full ~75-candidate backlog.

## Reference databases

Curated resources torchcell reads from or cross-references (identity, annotation,
metabolic-model scaffold). Not per-strain perturbation datasets.

| Database | Type | What it provides | URL |
| :-- | :-- | :-- | :-- |
| SGD | genome / knowledgebase | reference genome, annotation, phenotype + literature curation | yeastgenome.org |
| **SPELL** | expression compendium (SGD) | search engine over 752 datasets / 15,475 arrays / 576 studies of yeast expression microarrays | spell.yeastgenome.org |
| YMDB 2.0 | metabolite database | curated yeast metabolite structures, concentrations, pathways | ymdb.ca |
| YeastNet v3 | functional gene network | probabilistic integrated gene-interaction network | inetbio.org/yeastnet |
| CYCLoPs / LoQAtE | localization + abundance | GFP-collection protein localization/abundance atlases | thecellvision / weizmann |
| TheCellMap.org | genetic-interaction portal | query/download for the Costanzo/Boone global GI network | thecellmap.org |
| Yeast9 GEM | genome-scale metabolic model | consensus stoichiometric reconstruction (metabolite-node IDs) | github SysBioChalmers/yeast-GEM |
| ScRAPdb | pan-omics assembly panel | 142-strain telomere-to-telomere reference panel omics | evomicslab.org/db/ScRAPdb |
| Yeast PeptideAtlas | proteome-observation DB | reprocessed MS peptide/protein observation confidence | peptideatlas.org/builds/yeast |

## Maintenance + provenance

- **Update on every build.** When a dataset is built/rebuilt, refresh its *Shape* +
  *Signal (gzip)* row from the built LMDB. Recipe (per dataset root):

  ```python
  # gzip-signal = compressibility of the serialized per-strain phenotype values
  import lmdb, pickle, json, gzip, os.path as osp
  env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
  blob = bytearray(); n = 0
  with env.begin() as txn:
      for _, v in txn.cursor():
          ph = pickle.loads(v)["experiment"]["phenotype"]
          blob += json.dumps(ph, sort_keys=True, default=str).encode(); n += 1
  signal = len(gzip.compress(bytes(blob), 6)); print(n, signal)
  ```

- **Signal caveat.** The gzip number includes phenotype metadata (SE, n_replicates,
  measurement_type) alongside the measured values, so it is a *relative* signal proxy,
  not an absolute information content -- comparable across datasets, not exact.
- The detailed candidate backlog + Qian-2026 cross-reference lives in
  `[[paper.north-star.dataset-triage]]`; the differentiation framing in `[[paper.north-star]]`.

# Datasets

A **dataset** in torchcell is one loader class that turns the files released with one
publication (or one database) into validated experiment records. Loaders live under
`torchcell/datasets/scerevisiae/`, and each registers itself in
`torchcell.datasets.dataset_registry.dataset_registry` with the `@register_dataset`
decorator, which is how build tools look a loader up by class name.

## How a dataset is built

Every loader subclasses `ExperimentDataset` (`torchcell/data/experiment_dataset.py`),
itself a PyTorch Geometric `Dataset`. A subclass declares `experiment_class`,
`reference_class`, `raw_file_names`, `download()`, `preprocess_raw()`,
`create_experiment()` and a `process()` decorated with `@post_process`. The on-disk
layout under the dataset's `root` is:

```text
<root>/
  raw/                           files as released by the publisher (or copied from the raw mirror)
  preprocess/
    data.csv                     the cleaned table the records are built from (when the loader writes one)
    gene_set.json                every systematic gene name perturbed in the dataset
    experiment_reference_index.json   records grouped by the reference they share
    build_manifest.json          the schema contract the LMDB was built against
  processed/
    lmdb/                        one pickled record per key "0", "1", ...
    interned/                    constant sub-objects stored once, referenced by "$ref"
```

`@post_process` runs after `process()` finishes. It computes and writes the gene set and
the reference index, asserts that every record is covered by a reference, and
writes `build_manifest.json` through
`torchcell.provenance.build_manifest.write_build_manifest`.

**`build_manifest.json`** is a `BuildManifest` pydantic record: the loader class and
module, the build time and host, the torchcell commit (recorded for provenance only), and
`closure`, the contract fingerprint of every schema symbol the loader depends on. A
fingerprint changes only when a symbol in the loader's closure changes, so a schema edit
elsewhere leaves the dataset fresh. To list which built datasets are stale against the
local schema:

```bash
python -m torchcell.provenance.build_manifest   # reads $DATA_ROOT; exit status 1 if any dataset is stale
```

A loader skips `process()` whenever `processed/` already exists, so a stale LMDB is
reused silently unless it is retired first. The build CLI refuses instead:

```bash
python -m torchcell.database.build_dataset_lmdb --dataset SmfCostanzo2016Dataset
```

It resolves the class from the registry, builds under
`$DATA_ROOT/<loader default root>` (for example `data/torchcell/smf_costanzo2016`),
passes the shared genome and gene graph to loaders whose constructors take them, and
exits with an error when `processed/lmdb` already exists, pointing at
`scripts/deprecate.sh` to move the old build aside.

### Two build trees

`$DATA_ROOT/data/torchcell/<dataset>/` is the development tree: what loaders write by
default and what the command above builds. `$DATA_ROOT/database/data/torchcell/<dataset>/`
is the knowledge-graph build tree, owned by the graph-build user and populated by the
graph build, not by a loader run. Rebuilding a loader in the development tree does not
change what the served graph holds; see [Knowledge graph](knowledge-graph.md).

## Raw files and their provenance

The stored file and its sha256 are the record of what a dataset was built from; the URL
it was fetched from is retrieval metadata. Loaders that follow the raw-mirror convention
read their inputs from `$DATA_ROOT/torchcell-raw/<citation_key>/`, where each file is
pinned by sha256 in a manifest together with its source URL and the command that
retrieved it. The Mota 2024 loader (`torchcell/datasets/scerevisiae/mota2024.py`), for
example, reads its three supplementary spreadsheets from the mirror and uses the network
only when the mirror is absent. The mirror keeps exactly
the files the loader consumed for its first successful build. Older loaders, such as
`SmfCostanzo2016Dataset`, still fetch their raw files from the publisher's URL in
`download()`.

Values a loader cannot read from a data column, such as a replicate count or the kind
of an uncertainty, are recorded as `SourcedValue` objects that carry a verbatim quote
from the paper and the sha256 of the file quoted (see [Data model](data-model.md)).

## Supported datasets

The tables below are copied verbatim from `notes/paper.supported-datasets-and-databases.md`
as committed on 2026-09-21 (commit `fd805a8fc`). That note is rendered by
`experiments/database/scripts/render_supported_datasets_table.py` from the JSON that
`experiments/database/scripts/build_supported_datasets_table.py` writes after scanning
every built LMDB under `$DATA_ROOT`. The generator needs the built LMDBs and writes into
`experiments/database/results/`, so it was not rerun for this page. Genotypes, Env and
Phenotype are curated in the generator; Instances, Shape, Graph role and Signal are read
from the built LMDBs.

**Columns.** *Genotypes*: distinct perturbed strains or isolates. *Env*: number of
environments. *Instances*: records in the dataset (genotype x environment). *Shape*: shape
of one phenotype instance. *Graph role*: where the label sits on the cell graph.
*Signal (gzip, bits)*: gzip size, in bits, of the concatenated stored instances, a
relative measure of information content that includes phenotype metadata.

### Fitness + genetic interaction

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Costanzo 2016 (smf) | 20,484 | 2 | 20,484 | single-mutant fitness | scalar | global | 5.5×10⁶ |
| Costanzo 2016 (dmf) | 20.7M | 2 | 20,705,612 | double-mutant fitness | scalar | global | 7.1×10⁹ |
| Costanzo 2016 (dmi) | 20.7M | 2 | 20,705,612 | digenic interaction | scalar | edge | 5.8×10⁹ |
| Kuzmin 2018 (smf) | 1,539 | 1 | 1,539 | single-mutant fitness | scalar | global | 3.5×10⁵ |
| Kuzmin 2018 (dmf) | 410,399 | 1 | 410,571 | double-mutant fitness | scalar | global | 1.5×10⁸ |
| Kuzmin 2018 (tmf) | 91,111 | 1 | 91,111 | triple-mutant fitness | scalar | global | 3.9×10⁷ |
| Kuzmin 2018 (dmi) | 410,399 | 1 | 410,399 | digenic interaction | scalar | edge | 1.3×10⁸ |
| Kuzmin 2018 (tmi) | 91,111 | 1 | 91,111 | trigenic interaction | scalar | hyperedge | 3.2×10⁷ |
| Kuzmin 2020 (smf) | 472 | 1 | 472 | single-mutant fitness | scalar | global | 9.2×10⁴ |
| Kuzmin 2020 (dmf) | 632,797 | 1 | 632,998 | double-mutant fitness | scalar | global | 2.3×10⁸ |
| Kuzmin 2020 (tmf) | 301,798 | 1 | 301,798 | triple-mutant fitness | scalar | global | 1.1×10⁸ |
| Kuzmin 2020 (dmi) | 632,797 | 1 | 632,797 | digenic interaction | scalar | edge | 2.0×10⁸ |
| Kuzmin 2020 (tmi) | 301,798 | 1 | 301,798 | trigenic interaction | scalar | hyperedge | 1.1×10⁸ |
| Baryshnikova 2010 (smf) | 5,993 | 1 | 5,993 | single-mutant fitness | scalar | global | 1.7×10⁶ |
| O'Duibhir 2014 (smf) | 1,312 | 1 | 1,312 | single-mutant fitness | scalar | global | 2.0×10⁶ |

### Environmental / chemogenomic

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Auesukaree 2009 (stress screen) | 333 | 6 | 525 | stress sensitivity (categorical: sensitive/no_change) | scalar | global | 2.5×10⁵ |
| Mota 2024 (weak-acid screen) | 601 | 3 | 1,270 | weak-acid susceptibility (ordinal: 0/1/2 grades) | scalar | global | 6.2×10⁵ |
| Vanacloig-Pedros 2022 (chemogenomic fitness) | 3,647 | 41 | 143,218 | chemogenomic fitness (log2-ratio) | scalar | global | 8.7×10⁷ |
| Costanzo 2021 (condition-SGA) | 4,399 | 14 | 61,430 | differential mutant fitness | scalar | global | 1.0×10⁸ |
| Hillenmeyer 2008 (FitDb HIP, het) | 5,814 | 514 | 2,698,797 | HIP fitness-defect log2-ratio | scalar | global | 1.4×10⁹ |
| Hillenmeyer 2008 (FitDb HOP, hom) | 4,667 | 279 | 1,088,620 | HOP fitness-defect z-score | scalar | global | 5.9×10⁸ |
| Wildenhain 2015 (drug tolerance) | 256 | 5,168 | 428,206 | growth-inhibition z-score | scalar | global | 2.0×10⁸ |
| Hoepfner 2014 (HIP/HOP atlas) | 10,779 | 563 | 3,124,319 | HIP/HOP sensitivity score | scalar | global | 1.8×10⁹ |
| Smith 2006 (chemogenomic) | 4,721 | 3 | 12,747 | chemogenomic sensitivity (clear-zone ordinal) | scalar | global | 1.4×10⁶ |
| Lian 2019 (MAGIC CRISPR-AID) | 266,415 | 3 | 266,304 | furfural tolerance fitness (log2-ratio) | scalar | global | 1.3×10⁸ |
| Mormino 2022 (CRISPRi acetic-acid) | 12 | 1 | 12 | acetic-acid sensitivity (categorical) | scalar | global | 1.2×10⁴ |
| Smith 2016 (CRISPRi chem-genetic) | 1,035 | 8 | 7,053 | chemogenomic fitness (log2-ratio) | scalar | global | 2.1×10⁶ |
| Bloom 2019 (16-cross segregant panel) | 13,950 | 38 | 530,100 | colony size residual / absolute | scalar | global | 2.5×10⁸ |

### Viability

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| SGD (essentiality) | 1,329 | 1 | 1,329 | gene essentiality | scalar | node | 1.4×10⁵ |
| SynLethDB (lethal) | 14,000 | 1 | 14,000 | synthetic lethality | scalar | edge | 2.2×10⁶ |
| SynLethDB (rescue) | 6,948 | 1 | 6,948 | synthetic rescue | scalar | edge | 1.1×10⁶ |

### Morphology

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Ohya 2005 (SCMD CalMorph) | 4,718 | 1 | 4,718 | cell morphology (CalMorph) | vector (281) | global | 1.5×10⁸ |
| Ohnuki 2018 (SCMD CalMorph) | 1,112 | 1 | 1,112 | cell morphology (CalMorph) | vector (281) | global | 4.8×10⁷ |
| Ohnuki 2022 (SCMD CalMorph) | 1,979 | 1 | 1,979 | cell morphology (CalMorph) | vector (281) | global | 8.8×10⁷ |

### Expression (microarray)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Kemmeren 2014 (deletion compendium) | 1,484 | 1 | 1,484 | mRNA log2(mut/wt) | vector (6169) | node | 3.2×10⁹ |
| Sameith 2015 (sm) | 82 | 1 | 82 | mRNA log2(mut/ref) | vector (6169) | node | 1.8×10⁸ |
| Sameith 2015 (dm) | 72 | 1 | 72 | mRNA log2(mut/ref) | vector (6169) | node | 1.7×10⁸ |

### Expression (RNA-seq)

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Caudal 2024 (pan-transcriptome) | 943 | 1 | 943 | mRNA abundance (RNA-seq) | vector (6000) | node | 1.5×10⁹ |
| Nadal-Ribelles 2025 (Perturb-seq) | 3,150 | 2 | 6,188 | mRNA logFC (Perturb-seq) | vector (5639) | node | 2.2×10⁹ |

### Metabolite

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Cachera 2023 (CRI-SPA betaxanthin) | 4,735 | 1 | 4,719 | betaxanthin (product proxy) | scalar | bipartite node | 2.2×10⁶ |
| Mülleder 2016 (amino-acid metabolome) | 4,678 | 1 | 4,678 | amino-acid concentrations | vector (19) | bipartite node | 8.1×10⁶ |
| Cooper 2010 (CE-LIF amino-acid metabolome) | 4,313 | 1 | 4,313 | amino-acid peak ratios to plate mean | vector (16) | bipartite node | 5.1×10⁶ |
| Zelezniak 2018 (metabolome) | 95 | 1 | 95 | metabolite levels | vector (25) | bipartite node | 3.1×10⁵ |
| Ozaydin 2013 (β-carotene screen) | 4,474 | 1 | 4,474 | β-carotene (colony-color visual score) | scalar | global | 1.0×10⁶ |
| da Silveira 2014 (lipidomics) | 127 | 1 | 127 | lipid-species relative abundance | vector (135) | bipartite node | 1.0×10⁶ |
| Yoshida 2012 (organic acids) | 17 | 1 | 17 | organic-acid titer | vector (6) | bipartite node | 1.4×10⁴ |

Datasets marked private in the generator's output are omitted from this public page; the Total row below still counts them (51 datasets), as the generator wrote it.

### Protein abundance

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| Zelezniak 2018 (SWATH proteome) | 97 | 1 | 97 | protein abundance | vector (726) | node | 1.7×10⁷ |
| Messner 2023 (proteome) | 4,699 | 1 | 4,699 | protein abundance | vector (1830) | node | 1.1×10⁹ |

### Total

| Dataset | Genotypes | Env | Instances | Phenotype | Shape | Graph role | Signal (gzip, bits) |
| :-- | --: | --: | --: | :-- | :-- | :-- | --: |
| **Total (51 datasets)** |  |  | **52,743,236** |  |  |  | **2.7×10¹⁰** |

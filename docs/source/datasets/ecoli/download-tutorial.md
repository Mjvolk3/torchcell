# Downloading E. coli datasets

Four built *Escherichia coli* K-12 datasets are served as archives by the `tc-data` endpoint ({doc}`../../guide/downloads`). A loader constructed with `TC_DATA_URL` set downloads its archive, checks the sha256, unpacks the store, and opens the records, so no build step and no source file is needed. Every code block below is a part of one script, {download}`ecoli_download_tutorial.py <../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py>`, and each block is followed by its printed output.

```{include} _generated/download-tutorial/datasets_table.md
```

Terms used below:

- **archive**: one `tar.xz` file holding one store, named `<slug>-<torchcell version>-<first eight hex digits of its sha256>.tar.xz`.
- **content hash**: the sha256 over the sorted experiment ids of a dataset. An experiment id is the sha256 of one record's `experiment`, and each knowledge-graph release records one content hash per dataset.
- **Hfr Cavalli donor**: the *E. coli* strain that transfers a marked deletion into a recipient by conjugation in the Babu et al. 2014 screen.
- **Keio collection**: the *E. coli* K-12 BW25113 single-gene deletion strains of Baba et al. 2006, in which a kanamycin-resistance cassette replaces each gene.
- **modified z-score**: the value Fuhrer et al. 2017 release for each strain and ion, summarized over the strain's replicates and standardized across strains.
- **record**: one stored measurement, a dict with three keys. `experiment` holds the genotype, the environment, and the phenotype. `reference` holds the genome and the level the value is compared with. `publication` holds the identifiers of the source paper.
- **S-score**: the signed genetic-interaction score Babu et al. 2014 compute from the colony sizes of a double mutant. A negative score is an aggravating interaction and a positive score is an alleviating one.
- **slug**: the directory name of a store, which is also the dataset's name on the endpoint.
- **store**: the built form of a dataset. Its `processed/` directory holds the records LMDB and the `interned` LMDB beside it, and its `preprocess/` directory holds the build manifest and the tables the loader wrote.

## Install

torchcell needs Python 3.13 or newer. Either installer works on a new virtual environment:

```bash
pip install torchcell
```

```bash
uv pip install torchcell
```

On Linux the default torch wheel depends on the CUDA libraries. On a machine with no GPU, both installers can select the CPU build of torch, which gives a smaller environment:

```bash
uv pip install --torch-backend cpu torchcell
pip install --extra-index-url https://download.pytorch.org/whl/cpu torchcell
```

A first install was timed on two machines, each arm in a new Python 3.13 virtual environment, with the package caches, the temporary directory, and the environments on one filesystem. An empty cache is the first install on a machine, and a filled cache is a later install into another environment.

```{include} _generated/download-tutorial/install_benchmark.md
```

The same installs were repeated for the next release. The table gives each one with the result of importing a loader in the new environment.

```{include} _generated/download-tutorial/install_release_check.md
```

## Set the endpoint and import the loaders

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [setup]"
:end-before: "# [/setup]"
```

`DEFAULT_URL` is the public endpoint. It takes no key and no account. A loader downloads only when `TC_DATA_URL` is set in the process environment. With the variable unset, the same constructor builds the store from the publisher's files, which needs the raw-file mirror of a development machine.

## See what the endpoint serves

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [index]"
:end-before: "# [/index]"
```

```{include} _generated/download-tutorial/output_index.md
```

`select` returns the newest archive whose `torchcell_version` shares the installed `major.minor`, and `None` when the endpoint has no such archive. An archive name includes the version it was packaged with, and every release on that `major.minor` line reads it.

## Growth rate (Campos 2018)

`GrowthRateCampos2018Dataset` holds one record per Keio strain. The label is the strain's maximal growth rate in M9 medium with casamino acids and glucose at 30 °C, stored as a ratio to the parent strain BW25113, so the reference level is 1.0.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth]"
:end-before: "# [/growth]"
```

```{include} _generated/download-tutorial/output_growth_rate.md
```

The constructor did the download. After it returns, `data/torchcell/ecoli_growth_rate_campos2018/` holds the archive under `artifacts/` and the store under `processed/` and `preprocess/`. A second construction with the same `root` opens the store without a request.

## The same record as typed objects

`dataset[i]` returns plain dicts. `transform_item` validates one record through the pydantic classes of the schema and returns the typed objects.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [typed]"
:end-before: "# [/typed]"
```

```{include} _generated/download-tutorial/output_typed_record.md
```

## Cell shape and cell cycle (Campos 2018)

`MorphologyCampos2018Dataset` covers the same strains from the same imaging screen. Each record stores the cell-shape means and the cell-cycle features under the symbols the paper gives them, and the coefficients of variation of the shape features in `morphology_coefficient_of_variation`.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [morphology]"
:end-before: "# [/morphology]"
```

```{include} _generated/download-tutorial/output_morphology.md
```

## Genetic interactions (Babu 2014)

`GeneInteractionBabu2014Dataset` holds one record per double mutant of the released high-confidence set. The genotype of a record lists two perturbations, one from an Hfr Cavalli donor and one from the recipient array, and the label is the pair's S-score.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [interaction]"
:end-before: "# [/interaction]"
```

```{include} _generated/download-tutorial/output_genetic_interaction.md
```

The donor is the first perturbation in some records and the second in others, so the `collection` field of each perturbation, not its position, says which side of the cross it came from.

## Metabolome (Fuhrer 2017)

`MetabolomeFuhrer2017Dataset` holds one record per Keio strain. The label is a dict from ion key to modified z-score, with the same ions in the same order in every record, and `preprocess/ions.csv` gives the mass-to-charge ratio of each key.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [metabolome]"
:end-before: "# [/metabolome]"
```

```{include} _generated/download-tutorial/output_metabolome.md
```

## Check the records against the release

The endpoint's index row holds the content hash the knowledge-graph release recorded for the dataset. Hashing the loaded records the same way reproduces it when the download holds exactly the records the graph serves.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [verify]"
:end-before: "# [/verify]"
```

```{include} _generated/download-tutorial/output_verify.md
```

## What a download costs

The seconds below run from the call of a loader's constructor on an empty directory to the opened store. They cover the index request, the download, the sha256 check, and the unpacking.

```{include} _generated/download-tutorial/timings.md
```

## The same output on two machines

The script was run on the machine the stores were built on, from a repository checkout, and on a second machine that held only a new virtual environment. The table names how torchcell was installed on each. Standard output was byte-identical on the two.

```{include} _generated/download-tutorial/runs.md
```

## Caveats

**torchcell 1.8.4 and earlier need `torch-scatter` before a loader imports.** A plain install of those releases raises `ModuleNotFoundError: No module named 'torch_scatter'` on the first loader import, and `torch-scatter` has to be installed from the PyG wheel index for the installed torch build. From 1.8.5 the loaders import without it.

**A loader builds instead of downloading when `TC_DATA_URL` is unset.** torchcell does not read a `.env` file by itself, and the built-in `DEFAULT_URL` does not switch a loader to the download. Set the variable in the shell or in the process before the first constructor, as the setup block does.

**An archive is tied to a `major.minor` line.** `select` returns `None` for an installed version on another line, and the loader then raises and names the slug. It does not fall back to building.

**Some imaging records hold fewer features than the others.** The morphology section prints how many records hold each number of features. A table built from all records holds `NaN` for the features a strain lacks.

**Some metabolome records pool two or three Keio strains.** The deposit labels such a column with one name that matches more than one Keio entry, and the record's genotype lists every one of them. The metabolome section prints how many records name more than one gene, and joins their gene names with `+` in the strain labels.

## Provenance

- Tutorial code: `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py`.
- Runs and page fragments: `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py`, which writes `experiments/034-showcase-datasets/results/ecoli_download_tutorial.<host>.json` per host and the fragments under `docs/source/datasets/ecoli/_generated/download-tutorial/`.
- Install timings: `experiments/034-showcase-datasets/scripts/install_benchmark.py`, which writes `experiments/034-showcase-datasets/results/install_benchmark.<host>.json`.
- Archives: packaged by `scripts/package_dataset_lmdb.py` from the stores the loaders built.

```{include} _generated/download-tutorial/archives.md
```

# Downloading the E. coli archives

The four *E. coli* datasets of {doc}`tutorial` are served as archives by the `tc-data` endpoint. A loader constructed with `TC_DATA_URL` set downloads its archive, checks the sha256, unpacks the store, and opens the records, so no build step and no source file is needed. This page covers what the endpoint serves, what a download leaves on disk, how long it takes, and how to check the records against the knowledge-graph release. The endpoint's routes, the other file tiers and the `curl` commands are in {doc}`../../guide/downloads`. The code blocks are parts of the tutorial script, {download}`ecoli_download_tutorial.py <../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py>`.

Terms used below:

- **archive**: one `tar.xz` file holding one store, named `<slug>-<torchcell version>-<first eight hex digits of its sha256>.tar.xz`.
- **content hash**: the sha256 over the sorted experiment ids of a dataset. An experiment id is the sha256 of one record's `experiment`, and each knowledge-graph release records one content hash per dataset.
- **slug**: the directory name of a store, which is also the dataset's name on the endpoint.
- **store**: the built form of a dataset. Its `processed/` directory holds the records LMDB and the `interned` LMDB beside it, and its `preprocess/` directory holds the build manifest and the tables the loader wrote.

## Set the endpoint

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

```{include} _generated/tutorial/output_index.md
```

`select` returns the newest archive whose `torchcell_version` shares the installed `major.minor`, and `None` when the endpoint has no such archive. An archive name includes the version it was packaged with, and every release on that `major.minor` line reads it.

## Download by constructing a loader

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth_load]"
:end-before: "# [/growth_load]"
```

```{include} _generated/tutorial/output_growth_load.md
```

The constructor did the download. After it returns, `data/torchcell/ecoli_growth_rate_campos2018/` holds the archive under `artifacts/` and the store under `processed/` and `preprocess/`. A second construction with the same `root` opens the store without a request. The other three loaders work the same way, each with its own slug as the directory name.

## What a download costs

The seconds below run from the call of a loader's constructor on an empty directory to the opened store. They cover the index request, the download, the sha256 check, and the unpacking.

```{include} _generated/tutorial/timings.md
```

## Check the records against the release

The endpoint's index row holds the content hash the knowledge-graph release recorded for the dataset. Hashing the loaded records the same way reproduces it when the download holds exactly the records the graph serves.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [verify]"
:end-before: "# [/verify]"
```

```{include} _generated/tutorial/output_verify.md
```

## The archives

```{include} _generated/tutorial/archives.md
```

The archives were packaged by `scripts/package_dataset_lmdb.py` from the stores the loaders built.

## Caveats

**A loader builds instead of downloading when `TC_DATA_URL` is unset.** torchcell does not read a `.env` file by itself, and the built-in `DEFAULT_URL` does not switch a loader to the download. Set the variable in the shell or in the process before the first constructor, as the setup block does.

**An archive is tied to a `major.minor` line.** `select` returns `None` for an installed version on another line, and the loader then raises and names the slug. It does not fall back to building.

**torchcell 1.8.4 and earlier need `torch-scatter` before a loader imports.** From 1.8.5 the loaders import without it ({doc}`../../guide/install-timings`).

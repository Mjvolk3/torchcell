---
id: p8gozp13tuorsmoeb9bptn6
title: Test_import_all
desc: ''
updated: 1790409872828
created: 1790409872828
---

## 2026.09.26 - Parametrized import of every torchcell module

One test case per module, the list taken from the filesystem so collection imports nothing (Decision 9 of [[plan.test-suite-buildout.2026.09.25]]). `NEVER_IMPORT` excludes modules whose import has a side effect (HuggingFace download, dataset build, `plt.show()`, cwd log files, SSL env mutation, and, found by the network guard on the first run, `graph/uniprot_api_ec.py` and `ncbi/ncbi.py`, which query a web API at import). `KNOWN_BROKEN` holds the modules that fail to import today as strict xfails with the reason, so a fix surfaces as an XPASS failure and the entry has to go. Every case runs under `monkeypatch.chdir(tmp_path)` because biocypher and the `knowledge_graphs` modules write log files into the working directory.

First run (2026.09.26, PR-0a tip 8617b5c6a): 401 modules collected after `NEVER_IMPORT`, 384 import, 17 fail, 10.7 s. The 17 (now `KNOWN_BROKEN`) fall into three groups: names that no longer exist in the package (`torchcell.data.Dataset`, `torchcell.datasets.CellDataset`, `torchcell.models.DCellLinear`, `torchcell.losses.WeightedMSELoss`, `NaNTolerantPearsonCorrCoef`, `torchcell.sgd`, `torchcell.data_prior`, `torchcell.datasets.fungal_utr_transformer`), third-party packages absent from the env (`rpy2`, `dask`, `gene_graph`, `pytorch_metric_learning`, `intermine` on Python 3.13), and the pydantic-v1 `trainers/utils.py`. All 17 are in the legacy or init-only partition of `scripts/legacy_partition.py`; the entries leave with the Phase 0d move (`^torchcell\.legacy\.` joins `NEVER_IMPORT`).

This file is deselected from the behavioral coverage run and measured separately as the import-only column (Decision 17). The second test, `test_no_hard_coded_machine_paths`, pins the four source lines that carry a developer machine path (`data/sgd_expression.py`, `datasets/scerevisiae/mechanisitc_aware.py`, `datasets/scerevisiae/spell.py`, `models/hetero_cell_bipartite_dango_diff_gi.py`) so a new one fails.

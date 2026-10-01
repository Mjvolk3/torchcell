<p align="center">
  <img src="https://raw.githubusercontent.com/Mjvolk3/torchcell/main/notes/assets/drawio/torchcell-logo.drawio.png" />
</p>

<p align="center">
  <a href="https://github.com/Mjvolk3/torchcell/actions/workflows/style.yaml"><img src="https://img.shields.io/github/check-runs/Mjvolk3/torchcell/main?nameFilter=ruff&label=lint%20(ruff)" alt="Lint (ruff)" /></a>
  <a href="https://github.com/Mjvolk3/torchcell/actions/workflows/test.yaml"><img src="https://img.shields.io/github/check-runs/Mjvolk3/torchcell/main?nameFilter=pytest-coverage&label=pytest" alt="Pytest with Coverage" /></a>
  <a href="https://github.com/Mjvolk3/torchcell/actions/workflows/mypy.yaml"><img src="https://img.shields.io/github/check-runs/Mjvolk3/torchcell/main?nameFilter=mypy-check&label=mypy" alt="Type check (mypy)" /></a>
  <a href="https://github.com/Mjvolk3/torchcell/actions/workflows/docs.yaml"><img src="https://img.shields.io/github/check-runs/Mjvolk3/torchcell/main?nameFilter=build&label=docs" alt="Build and Deploy Docs" /></a>
  <a href="https://codecov.io/gh/Mjvolk3/torchcell"><img src="https://codecov.io/gh/Mjvolk3/torchcell/branch/main/graph/badge.svg" alt="codecov" /></a>
  <a href="https://github.com/Mjvolk3/torchcell/releases/latest"><img src="https://img.shields.io/github/v/release/Mjvolk3/torchcell?sort=semver" alt="Latest release" /></a>
  <a href="https://github.com/Mjvolk3/torchcell/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python 3.13+" /></a>
</p>

<p align="center">
  <img src="./notes/assets/images/Fig1-torchcell-overview-abc.png" alt="TorchCell overview: literature and public databases feed an ontology and a knowledge graph (a); reference data and experiment data (b); persistent entities, the encoder, perturbation, and decoder path, and contingent observations (c)" />
</p>

Documentation: <https://mjvolk3.github.io/torchcell/>

Database: <https://torchcell-database.ncsa.illinois.edu:7473/browser/> (Neo4j Browser over the served TorchCell knowledge graph)

Database releases and compatibility: <https://mjvolk3.github.io/torchcell/database/compatibility.html>

Datasets: <https://mjvolk3.github.io/torchcell/datasets/index.html>

Dataset downloads (the tc-data API): <https://mjvolk3.github.io/torchcell/guide/downloads.html>

## Download a dataset

Built datasets are served as versioned archives (the records LMDB plus its build
manifest) by the `tc-data` endpoint on the database host, with Swagger at `/docs`. Each
page below is one supported query over the knowledge graph and bundles the datasets it
returns: what the experiments measured, one stored record each, the value distributions,
the query, and the download commands.

| Page | Datasets (records) |
| :-- | :-- |
| [Gene essentiality and single-mutant fitness](https://mjvolk3.github.io/torchcell/datasets/scerevisiae/essentiality-smf.html) | Gene essentiality, SGD (1,329); single-mutant fitness, Costanzo 2016 (20,484) |
| [Amino acids and betaxanthin](https://mjvolk3.github.io/torchcell/datasets/scerevisiae/amino-acid-betaxanthin.html) | Amino acids, Mulleder 2016 (4,678); amine peaks, Cooper 2010 (4,313); betaxanthin, Cachera 2023 (4,719) |

With an endpoint URL and a key (see [Downloading datasets](https://mjvolk3.github.io/torchcell/guide/downloads.html)),
a loader fetches its archive instead of building:

```bash
export TC_DATA_URL=http://torchcell-database.ncsa.illinois.edu:8724
export TC_DATA_API_KEY=<your key>
```

```python
from torchcell.datasets.scerevisiae.mulleder2016 import AminoAcidMulleder2016Dataset

dataset = AminoAcidMulleder2016Dataset(root="data/torchcell/amino_acid_mulleder2016")
```

The full collection served by the knowledge graph (51 datasets, 52.7 million
experiments at release 2026.09.21) is listed on the
[Datasets](https://mjvolk3.github.io/torchcell/datasets/index.html) page.

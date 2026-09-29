# Installation

torchcell requires Python 3.13 or newer (`requires-python = ">=3.13"` in
`pyproject.toml`). The package's runtime dependencies are read from
`env/requirements.txt` (`[tool.setuptools.dynamic]` in `pyproject.toml`), so the pip
requirement files under `env/` are the source of truth for what gets installed.

## Create the environment

The repository does not ship a conda environment file for the main package
(`env/tc-graph.yaml` is a separate Python 3.11 environment for the graph-build
tooling). Create a Python 3.13 environment and install into it in the order the CI
workflow (`.github/workflows/test.yaml`) uses:

```bash
conda create -n torchcell python=3.13
conda activate torchcell

# 1. torch first; choose the wheel that matches your platform and CUDA version
pip install "torch==2.11.0"

# 2. the runtime stack
pip install -r env/requirements.txt

# 3. torch-scatter, from the PyG wheel index matched to the installed torch build
#    (env/requirements_dependent.txt must be installed after torch)
pip install -r env/requirements_dependent.txt -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

# 4. test tooling (pytest, coverage, diff-cover)
pip install -r env/requirements_test.txt

# 5. torchcell itself, editable
pip install -e .
```

For a CUDA build, replace `+cpu` in the wheel index URL with the CUDA tag of your torch
build (for example `+cu128`), as the comment in `env/requirements_dependent.txt` shows.
Documentation dependencies live in `env/requirements_docs.txt`.

The `Makefile` does not create environments; its `test`, `test-ci`, `test-fast` and
coverage targets call `$(HOME)/miniconda3/envs/torchcell/bin/python` unless `PYTHON` is
overridden (`make test PYTHON=$(which python)`).

## Environment variables

torchcell reads its paths and connection settings from the process environment. Most
entry points call `python-dotenv`'s `load_dotenv()`, so a `.env` file at the repository
root works as well. The table lists names only; values are machine specific.

| Variable | Required | Read by |
| :-- | :-- | :-- |
| `DATA_ROOT` | For any data access | Dataset loaders, the genome (`$DATA_ROOT/torchcell-genomes/`), raw-data mirrors, build CLIs. Importing torchcell does not need it. |
| `ASSET_IMAGES_DIR` | Optional | Plotting code and experiment scripts that save figures. |
| `EXPERIMENT_ROOT` | Optional | Experiment scripts under `experiments/`. |
| `NEO4J_URI` | Optional | `torchcell.database.connection`; defaults to a local bolt instance. |
| `NEO4J_USER` | Optional | As above; has a default. |
| `NEO4J_PASSWORD` | Optional | As above; has a default. Set it for any server other than a local development instance. |
| `TORCHCELL_KG_VERSION` | Optional | Which knowledge-graph release a query opens; defaults to `latest`. See {ref}`kg-choosing-a-release`. |
| `TC_LIT_URL` | Optional | Literature tooling that pulls OCR'd paper artifacts from the `tc-lit` service (for example `scripts/lit_bib_pull.py`). |
| `TC_LIT_API_KEY` | Optional | API key for `tc-lit`, used with `TC_LIT_URL`. |

The Neo4j variables are resolved at call time, not import time, by
`torchcell.database.connection.neo4j_connection_settings()`, so a `load_dotenv()` issued
after `import torchcell` is still honored.

## Verify the install

The check below needs no `DATA_ROOT`, no network and no database. Importing
`torchcell.datasets.scerevisiae` registers every dataset loader class in
`torchcell.datasets.dataset_registry.dataset_registry`.

```python
import torchcell
import torchcell.datasets.scerevisiae  # importing the loaders fills the registry
from torchcell.datasets.dataset_registry import dataset_registry

print(torchcell.__version__)
print("SmfCostanzo2016Dataset" in dataset_registry)
print(dataset_registry["SmfCostanzo2016Dataset"])
```

Output (torchcell 1.2.1, run with `DATA_ROOT` unset):

```text
1.2.1
True
<class 'torchcell.datasets.scerevisiae.costanzo2016.SmfCostanzo2016Dataset'>
```

Some compiled dependencies print `DeprecationWarning: builtin type SwigPyPacked has no
__module__ attribute` (and similar) on stderr during this import; the import still
succeeds.

To run the hermetic test suite, see {ref}`contributing-tests`.

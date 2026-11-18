# Install timings and release checks

Measurements of installing torchcell, kept apart from the dataset pages: how long each installer takes on two machines, whether each release imports a loader after a plain install, and whether a new environment prints the same data as a repository checkout. {doc}`installation` gives the install commands.

Terms used below:

- **arm**: one install measured once, in its own new Python 3.13 virtual environment.
- **empty cache**: the first install on a machine, with nothing downloaded before.
- **filled cache**: a later install into another environment, with the packages already in the installer's cache.

## pip and uv

Each arm ran with the package caches, the temporary directory, and the environments on one filesystem. On Linux the default torch wheel depends on the CUDA libraries, and the two CPU arms select the CPU build of torch.

```{include} _generated/install-timings/install_benchmark.md
```

## The same installs on the next release

The same installs were repeated for torchcell 1.8.5. The table gives each one with the result of importing a loader in the new environment.

```{include} _generated/install-timings/install_release_check.md
```

torchcell 1.8.4 and earlier raise `ModuleNotFoundError: No module named 'torch_scatter'` on the first loader import after a plain install, and `torch-scatter` has to be installed from the PyG wheel index for the installed torch build. From 1.8.5 the loaders import without it.

## The same output on two machines

The script of {doc}`../datasets/ecoli/tutorial` was run on the machine the stores were built on, from a repository checkout, and on a second machine that held only a new virtual environment. The table names how torchcell was installed on each. Standard output was byte-identical on the two.

```{include} _generated/install-timings/runs.md
```

## Provenance

- Install timings: `experiments/034-showcase-datasets/scripts/install_benchmark.py`, which writes `experiments/034-showcase-datasets/results/install_benchmark.<host>.json` and, run on a later release, `install_release_check.<host>.json`.
- Runs of the tutorial script and the tables on this page: `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py`.

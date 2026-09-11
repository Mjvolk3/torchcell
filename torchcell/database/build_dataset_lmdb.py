# torchcell/database/build_dataset_lmdb.py
# [[torchcell.database.build_dataset_lmdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/database/build_dataset_lmdb
# Test file: tests/torchcell/database/test_build_dataset_lmdb.py
"""Build ONE registered dataset's LMDB in the dev tree, for knowledge-graph admission.

The knowledge-graph build reads every dataset from ``$DATA_ROOT/database/data/torchcell/
<slug>/processed/lmdb`` (the uid-7474 build tree). Those LMDBs are produced in the dev tree
(``$DATA_ROOT/data/torchcell/<slug>``) by the dataset loader and then staged across. This CLI
is the dev-tree half: it resolves a loader class from the dataset registry, injects the shared
genome / graph when the loader declares them (the same rule the KG builder uses), and runs the
loader's ``process()`` by instantiating it.

A stale LMDB is never silently reused. If ``processed/lmdb`` already exists the command
refuses and points at ``scripts/deprecate.sh``; the dataset base class would otherwise skip
``process()`` and hand the knowledge graph an LMDB built against an older schema.

Usage (dev tree, from a slurm job or a shell)::

    python -m torchcell.database.build_dataset_lmdb --dataset NadalRibellesPerturbSeq2025Dataset
"""

from __future__ import annotations

import argparse
import inspect
import os
import os.path as osp
import sys
import time
from typing import Any

from dotenv import load_dotenv


def dataset_default_root(dataset_class: type) -> str:
    """The loader's default ``root`` (relative to ``DATA_ROOT``), e.g. ``data/torchcell/x``."""
    params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]
    return str(params["root"].default)


def resolve_dataset_class(name: str) -> type:
    """Look a dataset class up by name in the registry (importing the loader package)."""
    import torchcell.datasets.scerevisiae  # noqa: F401  # populates the registry
    from torchcell.datasets.dataset_registry import dataset_registry

    if name not in dataset_registry:
        known = ", ".join(sorted(dataset_registry))
        raise KeyError(f"{name!r} is not a registered dataset; known: {known}")
    return dataset_registry[name]


def build_dataset(dataset_class: type, data_root: str, io_workers: int) -> Any:
    """Instantiate ``dataset_class`` under ``data_root`` so it builds its LMDB; return it."""
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    root = osp.join(data_root, dataset_default_root(dataset_class))
    lmdb_dir = osp.join(root, "processed", "lmdb")
    if osp.isdir(lmdb_dir):
        raise FileExistsError(
            f"{lmdb_dir} already exists; the loader would reuse it instead of rebuilding. "
            "Retire it first: DEPRECATED_DIR=<graveyard> scripts/deprecate.sh "
            f"{osp.join(root, 'processed')} '<reason>' (and the sibling preprocess/)."
        )
    params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]
    kwargs: dict[str, Any] = {"root": root, "io_workers": io_workers}
    if "genome" in params or "scerevisiae_graph" in params:
        genome = SCerevisiaeGenome(
            genome_root=osp.join(data_root, "data/sgd/genome"),
            go_root=osp.join(data_root, "data/go"),
            overwrite=False,
        )
        if "genome" in params:
            kwargs["genome"] = genome
        if "scerevisiae_graph" in params:
            kwargs["scerevisiae_graph"] = SCerevisiaeGraph(
                sgd_root=osp.join(data_root, "data/sgd/genome"),
                string_root=osp.join(data_root, "data/string"),
                tflink_root=osp.join(data_root, "data/tflink"),
                genome=genome,
            )
    return dataset_class(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """CLI: build one registered dataset's LMDB and report its build manifest."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.database.build_dataset_lmdb",
        description="Build one registered dataset's LMDB in the dev tree.",
    )
    parser.add_argument(
        "--dataset", required=True, help="registered dataset class name"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="dev-tree DATA_ROOT (default: $DATA_ROOT from the environment / .env)",
    )
    parser.add_argument("--io-workers", type=int, default=0)
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = args.data_root or os.environ["DATA_ROOT"]
    dataset_class = resolve_dataset_class(args.dataset)
    start = time.time()
    dataset = build_dataset(dataset_class, data_root, args.io_workers)
    n = len(dataset)
    print(
        f"BUILT {dataset_class.__name__}: {n} records at {dataset.root} "
        f"in {time.time() - start:.0f}s; gene_set size {len(dataset.gene_set)}; "
        f"references {len(dataset.experiment_reference_index)}"
    )
    manifest = osp.join(dataset.preprocess_dir, "build_manifest.json")
    if not osp.exists(manifest):
        print(f"ERROR: build manifest missing at {manifest}", file=sys.stderr)
        return 1
    print(f"build manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

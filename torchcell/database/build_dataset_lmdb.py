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

Genome injection is by parameter NAME: ``genome`` receives ``SCerevisiaeGenome`` and
``scerevisiae_graph`` the graph on it; ``ecoli_genome`` / ``pputida_genome`` receive the
bacterial genome of the loader's ``REFERENCE_STRAIN``
(``torchcell.datasets.bacteria_common.BacterialGenomeInjector``). Each genome is built only
when the loader names its parameter, so a yeast build never reads the bacterial tier and a
bacterial build never resolves names against S288C.

A stale LMDB is never silently reused. If ``processed/lmdb`` already exists the command
refuses and points at ``scripts/deprecate.sh``; the dataset base class would otherwise skip
``process()`` and hand the knowledge graph an LMDB built against an older schema.

Usage (dev tree, from a slurm job or a shell)::

    python -m torchcell.database.build_dataset_lmdb --dataset NadalRibellesPerturbSeq2025Dataset

A full knowledge-graph rebuild needs EVERY mapped dev store fresh under the current schema
(the live-rebuild slurm script refuses otherwise), and a shared-class change such as
``Publication`` makes all of them stale at once. ``--list-stale`` prints, one class per
line, every mapped dataset whose store is stale, missing, or without a manifest (the same
check the live rebuild's preflight runs; ``--include-private`` adds the private map), and
``--retire-existing`` moves an existing ``processed/`` and ``preprocess/`` aside as
``<name>.superseded.<timestamp>`` siblings before the build instead of refusing, which is
what ``database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm`` runs per array
task over that list. Nothing is deleted: a superseded store is a rename.
"""

from __future__ import annotations

import argparse
import inspect
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


def dataset_default_root(dataset_class: type) -> str:
    """The loader's default ``root`` (relative to ``DATA_ROOT``), e.g. ``data/torchcell/x``."""
    params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]
    return str(params["root"].default)


def resolve_dataset_class(name: str) -> type:
    """Look a dataset class up by name in the registry (importing the loader package)."""
    import torchcell.datasets.ecoli  # noqa: F401  # populates the registry
    import torchcell.datasets.pputida  # noqa: F401  # populates the registry
    import torchcell.datasets.private_torchcell  # noqa: F401  # populates the registry
    import torchcell.datasets.scerevisiae  # noqa: F401  # populates the registry
    from torchcell.datasets.dataset_registry import dataset_registry

    if name not in dataset_registry:
        known = ", ".join(sorted(dataset_registry))
        raise KeyError(f"{name!r} is not a registered dataset; known: {known}")
    return dataset_registry[name]


StoreState = Literal["fresh", "stale", "no_lmdb", "no_manifest"]


class StoreStatus(BaseModel):
    """One mapped dataset's dev store against the current schema surface."""

    model_config = ConfigDict(frozen=True)

    dataset_class: str
    root: str
    state: StoreState
    drift: list[str] = []

    @property
    def needs_rebuild(self) -> bool:
        """Everything but ``fresh`` is rebuilt before a full knowledge-graph build."""
        return self.state != "fresh"


def mapped_store_status(data_root: str, include_private: bool) -> list[StoreStatus]:
    """The state of every mapped dataset's dev store, in adapter-map order.

    The same check the live rebuild's preflight runs: an LMDB must exist, its build
    manifest must exist, and every fingerprint in the manifest's closure must equal the
    current schema surface's.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
    from torchcell.provenance.build_manifest import BuildManifest, check_manifest
    from torchcell.provenance.schema_deps import load_default_surface

    surface = load_default_surface()
    statuses: list[StoreStatus] = []
    for cls in build_adapter_map(include_private=include_private):
        root = osp.join(data_root, dataset_default_root(cls))
        preprocess_dir = osp.join(root, "preprocess")
        manifest_path = osp.join(preprocess_dir, "build_manifest.json")
        if not osp.isdir(osp.join(root, "processed", "lmdb")):
            statuses.append(
                StoreStatus(dataset_class=cls.__name__, root=root, state="no_lmdb")
            )
            continue
        if not osp.isfile(manifest_path):
            statuses.append(
                StoreStatus(dataset_class=cls.__name__, root=root, state="no_manifest")
            )
            continue
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = BuildManifest.model_validate_json(handle.read())
        result = check_manifest(manifest, surface, preprocess_dir)
        statuses.append(
            StoreStatus(
                dataset_class=cls.__name__,
                root=root,
                state="stale" if result.is_stale else "fresh",
                drift=sorted({d.symbol for d in result.drift}),
            )
        )
    return statuses


def retire_existing(root: str, stamp: str | None = None) -> list[str]:
    """Move ``root/processed`` and ``root/preprocess`` aside as ``.superseded.<stamp>``.

    A rename on the same filesystem, never a delete; returns the new paths. A sibling
    that already carries the stamp is refused rather than merged into.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    moved: list[str] = []
    for name in ("processed", "preprocess"):
        src = osp.join(root, name)
        if not osp.isdir(src):
            continue
        dest = f"{src}.superseded.{stamp}"
        if osp.exists(dest):
            raise FileExistsError(f"{dest} already exists; refusing to retire over it")
        os.rename(src, dest)
        moved.append(dest)
    return moved


def build_dataset(dataset_class: type, data_root: str, io_workers: int) -> Any:
    """Instantiate ``dataset_class`` under ``data_root`` so it builds its LMDB; return it.

    The genomes the loader's ``__init__`` names are injected by name (module docstring);
    a bacterial loader that also names the yeast ``genome`` is refused before any genome
    is built.
    """
    from torchcell.datasets.bacteria_common import BacterialGenomeInjector
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
    kwargs.update(BacterialGenomeInjector(data_root).genome_kwargs(dataset_class))
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
    parser.add_argument("--dataset", help="registered dataset class name")
    parser.add_argument(
        "--data-root",
        default=None,
        help="dev-tree DATA_ROOT (default: $DATA_ROOT from the environment / .env)",
    )
    parser.add_argument("--io-workers", type=int, default=0)
    parser.add_argument(
        "--list-stale",
        action="store_true",
        help="print every mapped dataset whose dev store is stale, missing or has no "
        "manifest, one class per line, and exit (no build)",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="with --list-stale: include PRIVATE_DATASET_ADAPTER_MAP",
    )
    parser.add_argument(
        "--retire-existing",
        action="store_true",
        help="move an existing processed/ and preprocess/ aside as "
        "*.superseded.<timestamp> before building instead of refusing",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = args.data_root or os.environ["DATA_ROOT"]
    if args.list_stale:
        for status in mapped_store_status(data_root, args.include_private):
            if status.needs_rebuild:
                print(status.dataset_class)
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --list-stale is given")
    dataset_class = resolve_dataset_class(args.dataset)
    if args.retire_existing:
        for moved in retire_existing(
            osp.join(data_root, dataset_default_root(dataset_class))
        ):
            print(f"retired {moved}")
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

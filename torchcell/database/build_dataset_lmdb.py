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
line, every mapped dataset whose store is stale, missing, without a manifest, or
UNREADABLE (the same :func:`torchcell.provenance.build_manifest.check_store` the live
rebuild's preflight runs; ``--include-private`` adds the private map), and
``--retire-existing`` moves an existing ``processed/`` and ``preprocess/`` aside as
``<name>.superseded.<timestamp>`` siblings before the build instead of refusing, which is
what ``database/slurm/scripts/gilahyper_build_dataset_lmdbs_array.slurm`` runs per array
task over that list. Nothing is deleted: a superseded store is a rename.

``--verify`` (what the array script passes, so a rebuilt store is verified in the same
task that built it) runs the verification that covers the dataset and writes its
``preprocess/verification_report.json``, which is where that dataset's own tests read it
from: the 105-store array rebuild of 2026-10-09 left those reports absent, because a
build writes a build manifest and nothing else. Resolution order is
:func:`resolve_verifier`; a verification that raises or fails retires the build manifest
(:func:`retire_manifest`), so an unverified store reads ``no_manifest`` rather than
``fresh`` and is rebuilt before the graph is built from it.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import os.path as osp
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.provenance.build_manifest import StoreFreshness, StoreState


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


class StoreStatus(BaseModel):
    """One mapped dataset's dev store against the current schema surface."""

    model_config = ConfigDict(frozen=True)

    dataset_class: str
    root: str
    state: StoreState
    drift: list[str] = []
    reason: str | None = None  # set only for ``unreadable``: "<ExcClass>: <message>"

    @property
    def needs_rebuild(self) -> bool:
        """Everything but ``fresh`` is rebuilt before a full knowledge-graph build."""
        return self.state != "fresh"

    def describe(self) -> str:
        """``<class>: <what is wrong>``, the line a preflight refuses with."""
        return f"{self.dataset_class}: {self.freshness.describe()}"

    @property
    def freshness(self) -> StoreFreshness:
        """The verdict this status was built from, for its wording."""
        return StoreFreshness(
            root=self.root, state=self.state, drift=self.drift, reason=self.reason
        )


def mapped_store_status(data_root: str, include_private: bool) -> list[StoreStatus]:
    """The state of every mapped dataset's dev store, in adapter-map order.

    One call to :func:`torchcell.provenance.build_manifest.check_store` per mapped
    dataset, which is also what the live rebuild's preflight and the
    ``build_manifest`` CLI call: an LMDB must exist, its build manifest must exist,
    every fingerprint in the manifest's closure must equal the current schema
    surface's, and the store's first record must deserialize under the local schema
    (issue #833 -- a record pickled under a class the schema no longer has is in no
    closure, so fingerprints alone read it as fresh).
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
    from torchcell.provenance.build_manifest import check_store
    from torchcell.provenance.schema_deps import load_default_surface

    surface = load_default_surface()
    statuses: list[StoreStatus] = []
    for cls in build_adapter_map(include_private=include_private):
        root = osp.join(data_root, dataset_default_root(cls))
        freshness = check_store(root, surface)
        statuses.append(
            StoreStatus(
                dataset_class=cls.__name__,
                root=root,
                state=freshness.state,
                drift=freshness.drift,
                reason=freshness.reason,
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


def dataset_slug(dataset_class: type) -> str:
    """The store's directory name, e.g. ``gene_interaction_babu2014``."""
    return osp.basename(osp.normpath(dataset_default_root(dataset_class)))


def _required_parameters(function: Any) -> list[str]:
    """The names of ``function``'s parameters that have no default."""
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


def _reports_of(returned: Any) -> list[Any]:
    """Every ``VerificationReport`` a verification entry point returned.

    The entry points return one report, a tuple of them, or a dict of them keyed by
    arm, so the caller is given a flat list and judges ``passed`` over all of it.
    """
    from torchcell.verification.report import VerificationReport

    candidates = (
        list(returned.values()) if isinstance(returned, dict) else list(returned)
        if isinstance(returned, tuple | list)
        else [returned]
    )
    return [item for item in candidates if isinstance(item, VerificationReport)]


def resolve_verifier(dataset_class: type) -> tuple[str, Callable[[str], list[Any]]]:
    """The verification entry point that covers ONE dataset, with the name to print.

    Resolution order, first match wins; ``LookupError`` when none matches:

    1. a bioproduction registry in :mod:`torchcell.verification.runners` holds the
       dataset's name (:func:`~torchcell.verification.runners.verify_bacterial_dataset`),
       which runs the loader's own gate AND the family's L4 containment;
    2. the loader module's ``run_verification`` with no required parameter, called with
       the data root (it writes its own report, as ``babu2014`` does);
    3. the loader module's ``verify_build`` whose only required parameter is
       ``dataset_root``, called with the store root and the data root; its report is
       written here, since in a family run the runner writes it.

    A module whose entry point needs an argument this CLI cannot supply (a ``family``,
    an ``arm``, a dataset ``name`` -- the modules that build several stores) does NOT
    match: those stores are either held by a registry, which is rule 1, or verified by
    their family runner, which reads every store in the family and is not a per-dataset
    build step.
    """
    from torchcell.verification import runners

    slug = dataset_slug(dataset_class)
    if slug in runners.bacterial_registry_names():
        return (
            f"runners.verify_bacterial_dataset({slug!r})",
            lambda data_root: [runners.verify_bacterial_dataset(slug, data_root)],
        )
    module = importlib.import_module(dataset_class.__module__)
    run_verification = getattr(module, "run_verification", None)
    if run_verification is not None and _required_parameters(run_verification) == []:
        return (
            f"{module.__name__}.run_verification",
            lambda data_root: _reports_of(run_verification(data_root)),
        )
    verify_build = getattr(module, "verify_build", None)
    if verify_build is not None and _required_parameters(verify_build) == [
        "dataset_root"
    ]:

        def run(data_root: str) -> list[Any]:
            root = osp.join(data_root, dataset_default_root(dataset_class))
            reports = _reports_of(verify_build(root, data_root=data_root))
            for report in reports:
                runners._write_report(report, osp.join(root, "preprocess"))
            return reports

        return f"{module.__name__}.verify_build", run
    raise LookupError(
        f"{dataset_class.__name__}: {dataset_class.__module__} declares no "
        "run_verification()/verify_build() this CLI can call and no bioproduction "
        "registry holds it; its family runner in torchcell.verification.runners is "
        "what covers it"
    )


def retire_manifest(root: str, stamp: str | None = None) -> str | None:
    """Rename ``preprocess/build_manifest.json`` aside so the store stops reading fresh.

    A rename to ``build_manifest.json.unverified.<stamp>``, never a delete, so the
    manifest a failed verification rejected is still on disk to read. With no manifest
    the store reads ``no_manifest``, which ``--list-stale`` names and the live
    rebuild's preflight refuses. ``None`` when there was no manifest to retire.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    src = osp.join(root, "preprocess", "build_manifest.json")
    if not osp.isfile(src):
        return None
    dest = f"{src}.unverified.{stamp}"
    if osp.exists(dest):
        raise FileExistsError(f"{dest} already exists; refusing to retire over it")
    os.rename(src, dest)
    return dest


def verify_after_build(dataset_class: type, data_root: str) -> bool:
    """Run the verification covering ``dataset_class``; True when its store stays fresh.

    What ``--verify`` does after a build. A dataset whose verification raises OR whose
    report does not pass has its build manifest retired (:func:`retire_manifest`), so
    an unverified store cannot read fresh to the knowledge-graph preflight -- the
    builder wrote the records, and nothing else re-checks them before they are
    serialized into the graph. A dataset with no per-dataset entry point is reported as
    such and keeps its manifest: its family runner is what covers it, and that runner
    reads every store in the family.

    ``except Exception`` is the point of the flag: a raising verifier is turned into a
    retired manifest and a non-zero exit, not into a silent pass.
    """
    root = osp.join(data_root, dataset_default_root(dataset_class))
    try:
        kind, run = resolve_verifier(dataset_class)
    except LookupError as error:
        print(f"NO PER-DATASET VERIFIER -- {error}")
        return True
    print(f"verifying {dataset_class.__name__} with {kind}")
    try:
        reports = run(data_root)
    except Exception as error:  # reported and acted on, never swallowed
        print(
            f"ERROR: verification of {dataset_class.__name__} raised "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        print(f"ERROR: build manifest retired to {retire_manifest(root)}", file=sys.stderr)
        return False
    for report in reports:
        print(report.summary())
    failed = [report.dataset_name for report in reports if not report.passed]
    if not reports:
        print(
            f"ERROR: {kind} returned no verification report for "
            f"{dataset_class.__name__}",
            file=sys.stderr,
        )
        print(f"ERROR: build manifest retired to {retire_manifest(root)}", file=sys.stderr)
        return False
    if failed:
        print(f"ERROR: verification FAILED for {sorted(failed)}", file=sys.stderr)
        print(f"ERROR: build manifest retired to {retire_manifest(root)}", file=sys.stderr)
        return False
    print(f"verification PASSED: {len(reports)} report(s) by {kind}")
    return True


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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after the build, run the verification that covers this dataset and "
        "write its report; a verification that raises or fails retires the build "
        "manifest, so the store does not read fresh",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = args.data_root or os.environ["DATA_ROOT"]
    if args.list_stale:
        # stdout is the class list an array job reads line by line; the reason a store
        # is unreadable goes to stderr so it is visible without corrupting that list.
        for status in mapped_store_status(data_root, args.include_private):
            if status.needs_rebuild:
                print(status.dataset_class)
            if status.state == "unreadable":
                print(status.describe(), file=sys.stderr)
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
    if args.verify and not verify_after_build(dataset_class, data_root):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

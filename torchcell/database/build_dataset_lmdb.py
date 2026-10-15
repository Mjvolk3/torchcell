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
rebuild's preflight runs; ``--include-private`` adds the private map, and
``--organisms scerevisiae`` narrows the walk to the datasets whose declared organism is
in the comma-separated set, the same set the build's ``--organisms`` takes, so a
yeast-only rebuild is not held up by a stale bacterial store), and
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
``fresh`` and is rebuilt before the graph is built from it. "Fails" is the report's
verdict, which only ``gate`` rules decide: a genome-reference finding
(``FindingClass.report``: containment, name currency, assembly pins) is printed under
the verdict and never retires a manifest. The verdict is typed (:class:`VerifyOutcome`)
and the build log names it: a dataset with no per-dataset entry point is
``no_per_dataset_verifier`` and keeps its manifest (its family runner covers it), never
a silent ``True`` that reads like a pass.

:func:`mapped_store_lanes` is the preflight's second walk, beside
:func:`mapped_store_status`: it reads ``perturbation_leaves`` off each mapped store's
build manifest and checks every leaf against the node lanes the dataset's adapter conf
enables (``torchcell.knowledge_graphs.perturbation_lanes``), so a store whose records
carry a leaf no enabled lane emits is refused before generation starts.

:func:`mapped_store_collisions` is the third walk (lane V, 2026-10-15): it reads
``content_id_collisions`` off the same manifests and refuses a store whose records
share an ``Experiment`` content id with each other beyond what the dataset declares in
:data:`EXPECTED_CONTENT_ID_COLLISIONS`. The sink keeps one node per id, so such a record
is in the LMDB and absent from the graph with no error anywhere; four served datasets
and 65 records were in that state when lane Q first asked.
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
from enum import StrEnum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.data.experiment_dataset import (
    ALL_ORGANISMS,
    Organism,
    format_organisms,
    parse_organisms,
)
from torchcell.knowledge_graphs.perturbation_lanes import LaneGaps
from torchcell.provenance.build_manifest import (
    MANIFEST_FILENAME,
    BuildManifest,
    StoreFreshness,
    StoreState,
)


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


def mapped_store_status(
    data_root: str,
    include_private: bool,
    organisms: frozenset[Organism] = ALL_ORGANISMS,
) -> list[StoreStatus]:
    """The state of every mapped dataset's dev store, in adapter-map order.

    The walk is ``build_adapter_map(include_private, organisms)``, the map the build
    itself selects, so the list is exactly the stores a build of that organism set reads.

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
    for cls in build_adapter_map(include_private=include_private, organisms=organisms):
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


class StoreLaneStatus(BaseModel):
    """One mapped store's measured perturbation leaves against its conf's lanes."""

    model_config = ConfigDict(frozen=True)

    dataset_class: str
    root: str
    has_leaves: bool  # the manifest exists and carries ``perturbation_leaves``
    gaps: LaneGaps = LaneGaps()

    @property
    def blocks(self) -> bool:
        """A store with no measured leaves, or with a leaf no lane serves, blocks a build."""
        return not self.has_leaves or not self.gaps.empty

    def describe(self) -> str:
        """``<class>: <what is wrong>``, the line the preflight refuses with."""
        if not self.has_leaves:
            return (
                f"{self.dataset_class}: build manifest carries no perturbation_leaves "
                "(built before the walk existed); rebuild it"
            )
        return f"{self.dataset_class}: {self.gaps.describe()}"


def mapped_store_lanes(
    data_root: str,
    include_private: bool,
    organisms: frozenset[Organism] = ALL_ORGANISMS,
) -> list[StoreLaneStatus]:
    """Every mapped store's ``perturbation_leaves`` checked against its conf's lanes.

    The same walk as :func:`mapped_store_status` (``build_adapter_map`` of the organism
    set), reading each store's build manifest rather than its records: the leaves were
    measured over every record when the manifest was written. A leaf is a gap when the
    conf enables the edge method that walks it and no enabled node method emits it
    (``torchcell.knowledge_graphs.perturbation_lanes.unserved_leaves``); a manifest
    without the field, or no manifest at all, blocks as well, since the check cannot
    run. The exhaustiveness test runs the same arithmetic on each loader's direct schema
    imports; this runs it on what the store actually holds.
    """
    import torchcell
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
    from torchcell.knowledge_graphs.kg_manifest import dataset_conf_methods
    from torchcell.knowledge_graphs.perturbation_lanes import unserved_leaves

    repo_root = Path(torchcell.__file__).resolve().parent.parent
    statuses: list[StoreLaneStatus] = []
    for cls in build_adapter_map(include_private=include_private, organisms=organisms):
        root = osp.join(data_root, dataset_default_root(cls))
        manifest_path = osp.join(root, "preprocess", MANIFEST_FILENAME)
        leaves = None
        if osp.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                leaves = BuildManifest.model_validate_json(
                    handle.read()
                ).perturbation_leaves
        if leaves is None:
            statuses.append(
                StoreLaneStatus(dataset_class=cls.__name__, root=root, has_leaves=False)
            )
            continue
        gaps = unserved_leaves(
            leaves.genotype, leaves.environment, dataset_conf_methods(cls, repo_root)
        )
        statuses.append(
            StoreLaneStatus(
                dataset_class=cls.__name__, root=root, has_leaves=True, gaps=gaps
            )
        )
    return statuses


#: The records each dataset's SOURCE released more than once, byte-identically, so that
#: two stored records dump to one ``Experiment`` content id and the graph holds one node.
#: The number is ``n_records_lost`` of the store's manifest, and a dataset is in this
#: table only with the released evidence written down in its loader module: a value here
#: says "the source printed one measurement twice", never "we accept losing a record".
#: Everything absent declares 0, and :func:`mapped_store_collisions` refuses the rebuild
#: of any store whose measured count is not the declared one.
EXPECTED_CONTENT_ID_COLLISIONS: dict[str, int] = {
    # Mota 2024 lists RNR4 twice in Table S1 and twice in Table S2. Measured cell by
    # cell (lane V, 2026-10-15): the two rows of each pair are EQUAL in every released
    # column but the gene-name cell, which differs by a trailing non-breaking space that
    # 19, 21 and 23 rows of the three sheets carry as formatting noise. One measurement
    # printed twice, so the store keeps both released rows and the graph holds one node
    # for each pair. Table S3's two RNR4 rows release different grades (+ and ++) and
    # are two records; they never collided.
    "EnvChemgenMota2024Dataset": 2
}


class StoreCollisionStatus(BaseModel):
    """One mapped store's measured content-id collisions against its declaration."""

    model_config = ConfigDict(frozen=True)

    dataset_class: str
    root: str
    has_count: bool  # the manifest exists and carries ``content_id_collisions``
    measured: int = 0
    declared: int = 0
    example_groups: list[list[str]] = []

    @property
    def blocks(self) -> bool:
        """A store with no measured count, or one off its declaration, blocks a build."""
        return not self.has_count or self.measured != self.declared

    def describe(self) -> str:
        """``<class>: <what is wrong>``, the line the preflight refuses with."""
        if not self.has_count:
            return (
                f"{self.dataset_class}: build manifest carries no content_id_collisions "
                "(built before the count existed); rebuild it"
            )
        return (
            f"{self.dataset_class}: {self.measured} records share an Experiment content "
            f"id with another record of the same store and will be absent from the "
            f"graph; {self.declared} are declared in EXPECTED_CONTENT_ID_COLLISIONS. "
            f"Example groups (LMDB keys): {self.example_groups}"
        )


def mapped_store_collisions(
    data_root: str,
    include_private: bool,
    organisms: frozenset[Organism] = ALL_ORGANISMS,
) -> list[StoreCollisionStatus]:
    """Every mapped store's content-id collision count against its declaration.

    The third walk beside :func:`mapped_store_status` and :func:`mapped_store_lanes`,
    reading each store's build manifest rather than its records: the collisions were
    measured over every record when the manifest was written
    (``torchcell.provenance.build_manifest.walk_store``). A nonzero count means the
    sink will write one ``Experiment`` node for several records and the extra
    measurements will be in the LMDB and absent from the graph, which is what happened
    to four served datasets and 65 records before lane Q measured it.

    It REFUSES rather than reports, and the reason is that reporting was the status quo:
    the number was computable from the served store the whole time and nobody computed
    it, so 18 SGD essentiality annotations, 31 Ozaydin measurements and 2 Mota rows went
    into three KG releases unnoticed. A refusal has one escape and it is a written one:
    an entry in :data:`EXPECTED_CONTENT_ID_COLLISIONS` with the source evidence in the
    loader module, for a source that printed one measurement twice. The build manifest
    itself still RECORDS the count whatever it is; only the rebuild is gated.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    statuses: list[StoreCollisionStatus] = []
    for cls in build_adapter_map(include_private=include_private, organisms=organisms):
        root = osp.join(data_root, dataset_default_root(cls))
        manifest_path = osp.join(root, "preprocess", MANIFEST_FILENAME)
        collisions = None
        if osp.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                collisions = BuildManifest.model_validate_json(
                    handle.read()
                ).content_id_collisions
        if collisions is None:
            statuses.append(
                StoreCollisionStatus(
                    dataset_class=cls.__name__, root=root, has_count=False
                )
            )
            continue
        statuses.append(
            StoreCollisionStatus(
                dataset_class=cls.__name__,
                root=root,
                has_count=True,
                measured=collisions.n_records_lost,
                declared=EXPECTED_CONTENT_ID_COLLISIONS.get(cls.__name__, 0),
                example_groups=collisions.example_groups,
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

    if isinstance(returned, dict):
        candidates = list(returned.values())
    elif isinstance(returned, tuple | list):
        candidates = list(returned)
    else:
        candidates = [returned]
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
    an ``arm``, a dataset ``name``) does NOT match: those stores are either held by a
    registry, which is rule 1, or verified by their family runner, which reads every
    store in the family and is not a per-dataset build step.

    Rule 3 resolves by MODULE, so every store a module builds reaches the same
    ``verify_build``. A module that builds several stores (``campos2018``,
    ``dechiara2022``, ``muenzner2024``) therefore picks the gate from the store root's
    basename and refuses a root that is none of its stores; one that ignored the root
    would run one store's gate on another's records.
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


class VerifyOutcome(StrEnum):
    """What ``--verify`` found, named in the build log as ``VERIFY OUTCOME: <value>``.

    ``passed``: a per-dataset verifier ran and every report passed; the manifest stays.
    ``no_per_dataset_verifier``: :func:`resolve_verifier` found no entry point; the
    manifest stays because the dataset's FAMILY runner covers it, but the log says so
    rather than reporting a pass that never ran (review 3 finding 7, 2026-10-15).
    ``failed``: the verifier raised, returned no report, or a report did not pass; the
    manifest is retired and the build exits non-zero.
    """

    passed = "passed"
    no_per_dataset_verifier = "no_per_dataset_verifier"
    failed = "failed"

    @property
    def keeps_manifest(self) -> bool:
        """Only ``failed`` retires the build manifest."""
        return self is not VerifyOutcome.failed


def verify_after_build(dataset_class: type, data_root: str) -> VerifyOutcome:
    """Run the verification covering ``dataset_class``; the typed outcome.

    What ``--verify`` does after a build. A dataset whose verification raises OR whose
    report does not pass has its build manifest retired (:func:`retire_manifest`), so
    an unverified store cannot read fresh to the knowledge-graph preflight -- the
    builder wrote the records, and nothing else re-checks them before they are
    serialized into the graph. A dataset with no per-dataset entry point is
    ``no_per_dataset_verifier`` and keeps its manifest: its family runner is what
    covers it, and that runner reads every store in the family. Which of the three
    happened is printed as ``VERIFY OUTCOME: <value>`` so a build log never reads a
    skipped verification as a pass.

    The verdict is :attr:`VerificationReport.passed`, which only ``gate`` rules decide;
    each report's genome-reference findings are printed in its summary and counted on
    the PASS line, and none of them retires a manifest.

    ``except Exception`` is the point of the flag: a raising verifier is turned into a
    retired manifest and a non-zero exit, not into a silent pass.
    """
    root = osp.join(data_root, dataset_default_root(dataset_class))
    try:
        kind, run = resolve_verifier(dataset_class)
    except LookupError as error:
        print(f"NO PER-DATASET VERIFIER -- {error}")
        print(
            f"VERIFY OUTCOME: {VerifyOutcome.no_per_dataset_verifier} "
            "(manifest kept; the family runner covers this store)"
        )
        return VerifyOutcome.no_per_dataset_verifier
    print(f"verifying {dataset_class.__name__} with {kind}")

    def refuse(message: str) -> VerifyOutcome:
        """Report ``message``, retire the manifest, and answer ``failed``."""
        print(f"ERROR: {message}", file=sys.stderr)
        print(
            f"ERROR: build manifest retired to {retire_manifest(root)}", file=sys.stderr
        )
        print(f"VERIFY OUTCOME: {VerifyOutcome.failed}")
        return VerifyOutcome.failed

    try:
        reports = run(data_root)
    except Exception as error:  # reported and acted on, never swallowed
        return refuse(
            f"verification of {dataset_class.__name__} raised "
            f"{type(error).__name__}: {error}"
        )
    for report in reports:
        print(report.summary())
    if not reports:
        return refuse(
            f"{kind} returned no verification report for {dataset_class.__name__}"
        )
    failed = sorted(report.dataset_name for report in reports if not report.passed)
    if failed:
        return refuse(f"verification FAILED for {failed}")
    n_findings = sum(len(report.open_findings) for report in reports)
    print(
        f"verification PASSED: {len(reports)} report(s) by {kind}; "
        f"{n_findings} genome-reference finding(s) reported, not gated"
    )
    print(f"VERIFY OUTCOME: {VerifyOutcome.passed}")
    return VerifyOutcome.passed


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
        "--organisms",
        default=format_organisms(ALL_ORGANISMS),
        help="with --list-stale: walk only the mapped datasets whose declared organism "
        "is in this comma-separated set (default: every organism, "
        f"{format_organisms(ALL_ORGANISMS)})",
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
        organisms = parse_organisms(args.organisms)
        for status in mapped_store_status(data_root, args.include_private, organisms):
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
    if (
        args.verify
        and verify_after_build(dataset_class, data_root) is VerifyOutcome.failed
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

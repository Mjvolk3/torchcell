# torchcell/knowledge_graphs/kg_manifest.py
# [[torchcell.knowledge_graphs.kg_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/kg_manifest
# Test file: tests/torchcell/knowledge_graphs/test_kg_manifest.py
"""The served knowledge graph's build manifest, and the admission check for adding a
dataset to it without rebuilding the others.

A served Neo4j store is a function of (a) the schema contract every served dataset was
serialized under, (b) the BioCypher graph schema (which node labels carry which
properties), and (c) the adapter code that turns records into content-addressed nodes.
``KgBuildManifest`` records all three at build time, per dataset, so a later change can
be judged against what is actually in the store rather than against ``main``.

Admission rule (``check_admission``): a dataset may be added INCREMENTALLY only when
nothing already served would change. Concretely, every blocker below is empty:

- **served schema drift** -- a served dataset's closure fingerprints (the same
  content-addressed fingerprints ``torchcell.provenance`` uses for LMDB staleness)
  differ between the manifest and the working tree. Those records would serialize
  differently today, so the served nodes for that dataset are stale and incremental
  import, which cannot update or delete nodes, cannot repair them: FULL REBUILD.
- **graph schema drift** -- an existing node/edge class in
  ``torchcell_schema_config.yaml`` changed. New classes are fine (additive); a changed
  existing class means served nodes lack or carry properties the code now expects: FULL
  REBUILD.
- **value surface drift** -- the three checks above fingerprint CODE and SCHEMA, which
  leaves the shared VALUES that node ids are built from unwatched. A medium in
  ``torchcell/datamodels/media.py`` and a row of ``compound_identity_table.json`` are data:
  editing YPD's component list, or filling a compound's InChIKey, changes the content the
  adapter serializes for a medium or a compound node without changing one line of adapter
  code or one schema fingerprint. The served node keeps its old id, the dataset being
  admitted writes a node with a new one, and the graph ends up with two YPDs that no query
  joins. The surface is hashed file by file and compared; a change blocks with the same
  written-acknowledgment mechanism adapter drift has (``--ack-value-drift``).
- **adapter drift** -- node ids are sha256 of what ``CellAdapter`` serializes, so an
  adapter change CAN move ids silently. The check is method-level: a changed
  ``CellAdapter`` method blocks only if a SERVED adapter conf enables it, or if it is
  plumbing (not in the node/edge method table); a changed per-dataset adapter file or
  conf blocks only for a served dataset. New methods and new files are additive. What
  does block can be acknowledged with a written reason, which is recorded.
- **the new dataset** -- its dev-tree LMDB must exist and be fresh against the
  working-tree schema (its ``build_manifest.json``), it must be in
  ``dataset_adapter_map``, and its adapter may only enable phenotype node methods the
  graph schema declares (BioCypher drops undeclared classes silently).
- **a dataset that is already served** may be re-admitted only as a SUPERSET: every
  experiment id the live store holds under its Dataset node must still be produced by
  the dev-tree LMDB, and the LMDB must produce at least one id the store lacks. Node
  ids are content-addressed, so this is exactly the condition under which the
  incremental import matches every served node (``--skip-duplicate-nodes``) and adds
  only the new records; a served id the LMDB no longer produces would be left behind
  as a stale node beside its replacement, which is the full-rebuild case. The proof
  reads the served ids from the live store (``admit --neo4j-uri``), never from the
  manifest, and the dev ids through the adapter's own path (``transform_item`` then
  ``experiment_node_id``). A loader fix that only emits rows it used to drop passes;
  a loader fix that changes any existing record blocks.

Several datasets can be admitted in ONE increment (``check_batch_admission``): each
member is checked in isolation against the SERVED manifest and the batch is admissible
only when every member is. Isolation is what makes the batch a plain extension rather
than a second rule: a member is either absent from the store or a proven superset of
what the store holds for it, so nothing a member ADDS (a schema symbol, a graph class,
an adapter method, a record) can block another member, and a symbol two members both
introduce is additive and only reported.

The manifest lives beside the store (a machine-local file under the build tree), never
in git: it describes one physical database.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import re
import socket
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field

from torchcell.provenance.build_manifest import (
    MANIFEST_FILENAME,
    BuildManifest,
    _git_info,
    check_manifest,
)
from torchcell.provenance.schema_deps import (
    SchemaSurface,
    load_surface,
    load_surface_from_sources,
    loader_closure,
    loader_closure_from_source,
)

__all__ = [
    "KG_MANIFEST_SCHEMA_VERSION",
    "GraphSchemaEntry",
    "KgDatasetEntry",
    "KgEvent",
    "KgBuildManifest",
    "ServedDrift",
    "AdapterDrift",
    "SupersetCheck",
    "SupersetLineage",
    "AdmissionReport",
    "BatchAdmissionReport",
    "experiment_node_id",
    "dev_experiment_ids",
    "live_experiment_ids",
    "superset_check",
    "SCHEMA_CONFIG_RELPATH",
    "CELL_ADAPTER_RELPATH",
    "SURFACE_RELPATHS",
    "VALUE_SURFACE_RELPATHS",
    "graph_schema_from_yaml",
    "value_surface_from_sources",
    "value_surface_in_worktree",
    "value_surface_at_ref",
    "value_surface_drift",
    "cell_adapter_surface",
    "adapter_file_relpaths",
    "loader_relpath",
    "closure_at_ref",
    "closure_in_worktree",
    "bootstrap_manifest",
    "check_admission",
    "check_batch_admission",
    "batch_report_from_members",
    "record_admission",
    "record_batch_admission",
    "load_manifest",
    "save_manifest",
    "load_report",
    "format_report",
    "format_batch_report",
    "split_dataset_args",
    "parse_n_experiments",
]

KG_MANIFEST_SCHEMA_VERSION = 1
SCHEMA_CONFIG_RELPATH = "biocypher/config/torchcell_schema_config.yaml"
CELL_ADAPTER_RELPATH = "torchcell/adapters/cell_adapter.py"
ADAPTER_DIR_RELPATH = "torchcell/adapters"
SURFACE_RELPATHS = ("torchcell/datamodels/schema.py", "torchcell/datamodels/pydant.py")
# Shared VALUES (not code shapes) that served node ids are built from: the recipes every
# dataset's Media resolves to, the curated compound identity rows, and the resolver that
# turns a name into a Compound. A change to any of them moves the content-addressed id of
# a media or compound node, which no schema or adapter fingerprint would notice.
VALUE_SURFACE_RELPATHS = (
    "torchcell/datamodels/media.py",
    "torchcell/datamodels/compound_identity.py",
    "torchcell/datamodels/compound_identity_table.json",
)


class GraphSchemaEntry(BaseModel):
    """One BioCypher class: what the graph stores for it."""

    kind: Literal["node", "edge"]
    properties: list[str] = Field(default_factory=list)  # sorted; nodes only
    source: list[str] = Field(default_factory=list)  # edges only
    target: list[str] = Field(default_factory=list)  # edges only

    def served_nodes_unchanged_by(self, current: GraphSchemaEntry | None) -> bool:
        """True if nodes/edges written under ``self`` are still what ``current`` means.

        A node class must keep exactly its property set (served nodes would otherwise
        lack, or carry, properties the code now expects). An edge class may GAIN source
        or target labels (a new phenotype family joining ``phenotype member of`` is
        additive); losing one, or changing kind, is a change to what is served.
        """
        if current is None or current.kind != self.kind:
            return False
        if self.kind == "node":
            return current.properties == self.properties
        return set(self.source) <= set(current.source) and set(self.target) <= set(
            current.target
        )


class SupersetLineage(BaseModel):
    """The entry a served dataset grew from when it was re-admitted as a superset.

    The store then holds records from two imports: every record of the previous entry
    (matched by id, untouched) plus ``n_added`` new ones from the increment.
    """

    biocypher_out: str
    import_mode: Literal["full", "incremental"]
    admitted_at: str
    n_experiments: int | None
    n_added: int


class KgDatasetEntry(BaseModel):
    """A dataset as it exists in the served store."""

    dataset_class: str
    loader_relpath: str  # repo-relative loader module path
    adapter_files: list[str]  # repo-relative adapter module + conf yaml
    closure: dict[str, str]  # schema symbol -> contract fingerprint at admission
    n_experiments: int | None = None
    biocypher_out: str  # the biocypher-out/<timestamp> that produced its CSVs
    import_mode: Literal["full", "incremental"]
    admitted_at: str
    torchcell_commit: str | None = None
    # sha256 of the dataset's sorted experiment node ids (``releases.content_sha256``):
    # equal across two releases means byte-identical serialized records. None for a
    # manifest written before releases were stamped.
    content_sha256: str | None = None
    # Set when this entry was admitted as a superset of an entry already served: the
    # previous entry's identity and how many records the increment added to it.
    superset_of: SupersetLineage | None = None


class KgEvent(BaseModel):
    """One change to the served store."""

    kind: Literal[
        "bootstrap", "full_build", "incremental_admission", "superset_admission"
    ]
    at: str
    torchcell_commit: str | None
    datasets: list[str]
    biocypher_out: str | None = None
    note: str | None = None
    acknowledged_adapter_drift: list[str] = Field(default_factory=list)
    acknowledged_value_drift: list[str] = Field(default_factory=list)


class KgBuildManifest(BaseModel):
    """What the served knowledge graph was built from, dataset by dataset."""

    manifest_schema_version: int = KG_MANIFEST_SCHEMA_VERSION
    database: str
    store_host: str
    neo4j_version: str
    biocypher_version: str
    torchcell_commit: str | None  # commit of the FULL build the store descends from
    # Release identity (``torchcell.knowledge_graphs.releases``): ``version`` is
    # <major>.<minor> (full build bumps major, admission bumps minor); ``release`` is
    # <build date>-<commit[:8]>, the immutable id a dump or backup carries. None before
    # the store was stamped.
    version: str | None = None
    release: str | None = None
    graph_schema: dict[str, GraphSchemaEntry]
    cell_adapter_methods: dict[str, str]  # CellAdapter function -> source fingerprint
    cell_adapter_table: dict[str, str]  # conf method name -> CellAdapter function
    adapter_files: dict[str, str]  # repo-relative adapter/conf path -> sha256
    # Shared VALUE files -> sha256 of their content. Empty for a manifest written before
    # the value surface was recorded: the check then reports "value surface not recorded"
    # instead of blocking, since there is no baseline to compare against.
    value_surface: dict[str, str] = Field(default_factory=dict)
    datasets: dict[str, KgDatasetEntry]
    events: list[KgEvent]
    created_at: str
    updated_at: str


class ServedDrift(BaseModel):
    """A served dataset whose schema closure no longer matches the store."""

    dataset_class: str
    changed_symbols: list[str]


class AdapterDrift(BaseModel):
    """Adapter code changes that could move ids or properties of SERVED nodes."""

    plumbing_methods: list[str] = Field(default_factory=list)
    served_methods: dict[str, list[str]] = Field(default_factory=dict)  # fn -> datasets
    served_files: dict[str, list[str]] = Field(default_factory=dict)  # path -> datasets

    @property
    def is_empty(self) -> bool:
        """True when no change touches a served dataset."""
        return not (self.plumbing_methods or self.served_methods or self.served_files)

    def describe(self) -> str:
        """One-line summary for reports and the manifest event log."""
        parts: list[str] = []
        if self.plumbing_methods:
            parts.append("plumbing: " + ", ".join(self.plumbing_methods))
        for fn, datasets in sorted(self.served_methods.items()):
            parts.append(f"{fn} (used by {len(datasets)} served datasets)")
        for path, datasets in sorted(self.served_files.items()):
            parts.append(f"{path} ({', '.join(datasets)})")
        return "; ".join(parts)


class SupersetCheck(BaseModel):
    """Proof that a served dataset's records are all still produced by its dev LMDB.

    ``n_missing`` must be 0 for the re-admission to be a superset, and ``n_added`` must
    be positive for it to be worth an import.
    """

    n_served: int  # experiment ids under the Dataset node in the live store
    n_dev: int  # experiment ids the dev-tree LMDB produces
    n_missing: int  # served ids the dev LMDB no longer produces
    n_added: int  # dev ids the store lacks: what the increment would add
    missing_sample: list[str] = Field(
        default_factory=list
    )  # up to 5 of the missing ids
    served_source: str  # where the served ids were read from (the bolt URI)


class AdmissionReport(BaseModel):
    """Verdict on adding one dataset to the served store incrementally."""

    dataset_class: str
    checked_at: str
    torchcell_commit: str | None
    torchcell_dirty: bool | None
    served_commit: str | None
    verdict: Literal["admissible", "blocked"]
    reasons: list[str]
    new_dataset_closure: dict[str, str]
    novel_symbols: list[str]  # in the new dataset's closure, in no served closure
    shared_symbols: dict[str, list[str]]  # symbol -> served datasets that share it
    stale_served: list[ServedDrift]
    graph_schema_changed: list[str]  # existing classes whose definition changed
    graph_schema_added: list[str]
    adapter_drift: AdapterDrift
    adapter_methods_added: list[str]
    adapter_drift_acknowledged: str | None
    # Shared value files whose content changed since the build (a blocker unless
    # acknowledged), files added to the surface since then (additive: nothing served was
    # built from them), and whether the manifest recorded a surface at all.
    value_surface_changed: list[str] = Field(default_factory=list)
    value_surface_added: list[str] = Field(default_factory=list)
    value_surface_recorded: bool = True
    value_drift_acknowledged: str | None = None
    dev_lmdb_status: Literal["fresh", "stale", "unmanifested", "missing"]
    dev_lmdb_root: str
    in_adapter_map: bool
    undeclared_phenotype_methods: list[str]
    # True when the dataset is already in the served store; the admission is then a
    # superset re-admission and ``superset`` carries the proof (None when the proof
    # could not be run, which is itself a blocker).
    served: bool = False
    superset: SupersetCheck | None = None


class BatchAdmissionReport(BaseModel):
    """Verdict on adding SEVERAL datasets to the served store in ONE increment.

    Every member carries its own ``AdmissionReport``, checked against the SERVED
    manifest alone; the batch is admissible only when every one of them is. Two derived
    views make the batch readable:

    - ``co_introduced_symbols`` -- a schema symbol that more than one member introduces
      (novel to the served store). Purely informational: a symbol no served dataset
      holds cannot make served nodes stale, so members sharing one is additive.
    - ``changed_symbol_importers`` -- for a symbol that DID change relative to the
      served manifest, the served datasets that import it. This is the inverse of the
      members' ``stale_served``, so a block reads as "served datasets X, Y import Z".
    """

    dataset_classes: list[str]
    checked_at: str
    torchcell_commit: str | None
    torchcell_dirty: bool | None
    served_commit: str | None
    verdict: Literal["admissible", "blocked"]
    reasons: list[str]
    members: list[AdmissionReport]
    co_introduced_symbols: dict[str, list[str]]  # symbol -> members introducing it
    changed_symbol_importers: dict[str, list[str]]  # symbol -> served datasets


# --------------------------------------------------------------------------- helpers


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_show(repo_root: Path, ref: str, relpath: str) -> str:
    """Contents of ``relpath`` at ``ref``; raises if the path does not exist there."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{ref}:{relpath}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"{relpath} does not exist at {ref}: {result.stderr}")
    return result.stdout


def _git_ls(repo_root: Path, ref: str, prefix: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "-r",
            "--name-only",
            ref,
            "--",
            prefix,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def graph_schema_from_yaml(text: str) -> dict[str, GraphSchemaEntry]:
    """Parse ``torchcell_schema_config.yaml`` into per-class entries."""
    raw = yaml.safe_load(text) or {}
    entries: dict[str, GraphSchemaEntry] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "represented_as" not in spec:
            continue
        if spec["represented_as"] == "node":
            entries[name] = GraphSchemaEntry(
                kind="node", properties=sorted((spec.get("properties") or {}).keys())
            )
        else:
            source = spec.get("source", [])
            target = spec.get("target", [])
            entries[name] = GraphSchemaEntry(
                kind="edge",
                source=sorted(source if isinstance(source, list) else [source]),
                target=sorted(target if isinstance(target, list) else [target]),
            )
    return entries


def _is_table_entry(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == 2
        and isinstance(node.elts[0], ast.Constant)
        and isinstance(node.elts[0].value, str)
        and isinstance(node.elts[1], ast.Attribute)
    )


class _StripMethodTables(ast.NodeTransformer):
    """Drop ``("conf name", self._fn)`` entries from list literals (the method tables).

    The tables are fingerprinted separately (``cell_adapter_table``), so registering a
    NEW method must not make ``__init__`` look changed for the served datasets.
    """

    def visit_List(self, node: ast.List) -> ast.AST:
        self.generic_visit(node)
        node.elts = [elt for elt in node.elts if not _is_table_entry(elt)]
        return node


def cell_adapter_surface(source: str) -> tuple[dict[str, str], dict[str, str]]:
    """``(method fingerprints, conf-name -> method)`` of ``CellAdapter`` from its source.

    A method fingerprint hashes the function's normalized source (docstring stripped,
    method-table entries stripped from ``__init__``), so a comment or docstring edit, or
    registering an additional method, does not count as drift. The table is read from
    the ``("<conf method name>", self._fn)`` tuples in ``__init__``.
    """
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CellAdapter"
    )
    methods: dict[str, str] = {}
    table: dict[str, str] = {}
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if node.name == "__init__":
            for sub in ast.walk(node):
                if _is_table_entry(sub):
                    assert isinstance(sub, ast.Tuple)
                    assert isinstance(sub.elts[0], ast.Constant)
                    assert isinstance(sub.elts[0].value, str)
                    assert isinstance(sub.elts[1], ast.Attribute)
                    table[sub.elts[0].value] = sub.elts[1].attr
        clone = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=body or [ast.Pass()],
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_params=getattr(node, "type_params", []),
        )
        if node.name == "__init__":
            clone = _StripMethodTables().visit(copy.deepcopy(clone))
        methods[node.name] = _sha256(ast.unparse(ast.fix_missing_locations(clone)))
    return methods, table


def adapter_file_relpaths(repo_root: Path) -> list[str]:
    """Every per-dataset adapter module and conf yaml in the working tree."""
    paths = sorted((repo_root / ADAPTER_DIR_RELPATH).glob("*_adapter.py"))
    paths += sorted((repo_root / ADAPTER_DIR_RELPATH / "conf").glob("*.yaml"))
    return [str(p.relative_to(repo_root)) for p in paths if p.name != "cell_adapter.py"]


_CONF_RE = re.compile(r'"([A-Za-z0-9_]+_adapter\.yaml)"')


def _adapter_conf_name(adapter_source: str) -> str:
    match = _CONF_RE.search(adapter_source)
    if match is None:
        raise ValueError("adapter module does not name its conf yaml")
    return match.group(1)


def dataset_adapter_files(dataset_class: type, repo_root: Path) -> list[str]:
    """Repo-relative adapter module + conf yaml serving ``dataset_class``."""
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    adapter_class = dataset_adapter_map[cast(Any, dataset_class)]
    adapter_file = Path(inspect.getsourcefile(adapter_class) or "").resolve()
    rel = str(adapter_file.relative_to(repo_root.resolve()))
    conf = _adapter_conf_name(adapter_file.read_text(encoding="utf-8"))
    return [rel, f"{ADAPTER_DIR_RELPATH}/conf/{conf}"]


def dataset_conf_methods(dataset_class: type, repo_root: Path) -> list[str]:
    """Conf method names (node + edge) the dataset's adapter enables."""
    _, conf_rel = dataset_adapter_files(dataset_class, repo_root)
    conf = yaml.safe_load((repo_root / conf_rel).read_text(encoding="utf-8"))
    return [
        m["method_name"]
        for key in ("node_methods", "edge_methods")
        for m in conf["cell_adapter"][key]
    ]


def loader_relpath(dataset_class: type, repo_root: Path) -> str:
    """Repo-relative path of the module that defines ``dataset_class``."""
    module_file = sys.modules[dataset_class.__module__].__file__
    assert module_file is not None, dataset_class
    return str(Path(module_file).resolve().relative_to(repo_root.resolve()))


def surface_at_ref(repo_root: Path, ref: str) -> SchemaSurface:
    return load_surface_from_sources(
        {relpath: _git_show(repo_root, ref, relpath) for relpath in SURFACE_RELPATHS}
    )


def surface_in_worktree(repo_root: Path) -> SchemaSurface:
    return load_surface([repo_root / relpath for relpath in SURFACE_RELPATHS])


def value_surface_from_sources(sources: Mapping[str, str]) -> dict[str, str]:
    """``relpath -> sha256`` of shared value-file CONTENT.

    Content, not a parse: a medium's components, a compound row's InChIKey and the
    resolver's normalization rules all feed the node id, and any edit to the file is
    therefore a candidate for moving one. A false positive (a comment edit) is answered by
    the acknowledgment, which is recorded; a false negative would silently split a node.
    """
    return {relpath: _sha256(text) for relpath, text in sorted(sources.items())}


def value_surface_in_worktree(repo_root: Path) -> dict[str, str]:
    """The value surface of the working tree (files that exist)."""
    return value_surface_from_sources(
        {
            relpath: (repo_root / relpath).read_text(encoding="utf-8")
            for relpath in VALUE_SURFACE_RELPATHS
            if (repo_root / relpath).exists()
        }
    )


def value_surface_at_ref(repo_root: Path, ref: str) -> dict[str, str]:
    """The value surface as it was at ``ref`` (files that existed at that commit)."""
    present = {
        relpath
        for relpath in VALUE_SURFACE_RELPATHS
        if _git_ls(repo_root, ref, relpath) == [relpath]
    }
    return value_surface_from_sources(
        {
            relpath: _git_show(repo_root, ref, relpath)
            for relpath in VALUE_SURFACE_RELPATHS
            if relpath in present
        }
    )


def value_surface_drift(
    stored: Mapping[str, str], current: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """``(changed, added)`` between a recorded value surface and the current one.

    CHANGED covers a recorded file whose content differs and a recorded file that is now
    gone; both mean the values the served nodes were built from are not the values a new
    node would be built from. ADDED is a file that joined the surface after the build:
    nothing served was built from it, so it is additive and only reported.
    """
    changed = sorted(
        relpath for relpath, digest in stored.items() if current.get(relpath) != digest
    )
    added = sorted(set(current) - set(stored))
    return changed, added


def closure_at_ref(
    repo_root: Path, ref: str, loader_rel: str, surface: SchemaSurface
) -> dict[str, str]:
    """Symbol -> fingerprint for a loader's closure, both taken at ``ref``."""
    closure = loader_closure_from_source(_git_show(repo_root, ref, loader_rel), surface)
    return {name: surface.fingerprints[name] for name in sorted(closure)}


def closure_in_worktree(
    repo_root: Path, loader_rel: str, surface: SchemaSurface
) -> dict[str, str]:
    """Symbol -> fingerprint for a loader's closure, from the working tree."""
    closure = loader_closure(repo_root / loader_rel, surface)
    return {name: surface.fingerprints[name] for name in sorted(closure)}


def _dataset_class(name: str) -> type:
    import torchcell.datasets.scerevisiae  # noqa: F401  # populates the registry
    from torchcell.datasets.dataset_registry import dataset_registry

    return dataset_registry[name]


def _dataset_default_root(dataset_class: type) -> str:
    params = inspect.signature(dataset_class.__init__).parameters  # type: ignore[misc]
    return str(params["root"].default)


# --------------------------------------------------------------------------- manifest


def load_manifest(path: Path) -> KgBuildManifest:
    """Read a manifest JSON file."""
    return KgBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: KgBuildManifest, path: Path) -> None:
    """Write the manifest, stamping ``updated_at``."""
    manifest.updated_at = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def bootstrap_manifest(
    *,
    repo_root: Path,
    commit: str,
    dataset_classes: list[str],
    n_experiments: dict[str, int],
    database: str,
    store_host: str,
    neo4j_version: str,
    biocypher_version: str,
    biocypher_out: str,
    built_at: str,
) -> KgBuildManifest:
    """Reconstruct the manifest of a store built by a FULL build at ``commit``.

    Every fingerprint is computed from the sources AT THAT COMMIT (``git show``), so the
    manifest describes what was serialized into the store, not the working tree. The
    adapter files of each dataset are resolved through the working tree's
    ``dataset_adapter_map`` and then hashed at the commit.
    """
    surface = surface_at_ref(repo_root, commit)
    datasets: dict[str, KgDatasetEntry] = {}
    for name in sorted(dataset_classes):
        cls = _dataset_class(name)
        rel = loader_relpath(cls, repo_root)
        datasets[name] = KgDatasetEntry(
            dataset_class=name,
            loader_relpath=rel,
            adapter_files=dataset_adapter_files(cls, repo_root),
            closure=closure_at_ref(repo_root, commit, rel, surface),
            n_experiments=n_experiments.get(name),
            biocypher_out=biocypher_out,
            import_mode="full",
            admitted_at=built_at,
            torchcell_commit=commit,
        )
    methods, table = cell_adapter_surface(
        _git_show(repo_root, commit, CELL_ADAPTER_RELPATH)
    )
    adapter_files = {
        rel: _sha256(_git_show(repo_root, commit, rel))
        for rel in _git_ls(repo_root, commit, ADAPTER_DIR_RELPATH)
        if rel.endswith("_adapter.py") or rel.endswith(".yaml")
    }
    now = _now()
    return KgBuildManifest(
        database=database,
        store_host=store_host,
        neo4j_version=neo4j_version,
        biocypher_version=biocypher_version,
        torchcell_commit=commit,
        graph_schema=graph_schema_from_yaml(
            _git_show(repo_root, commit, SCHEMA_CONFIG_RELPATH)
        ),
        cell_adapter_methods=methods,
        cell_adapter_table=table,
        adapter_files=adapter_files,
        value_surface=value_surface_at_ref(repo_root, commit),
        datasets=datasets,
        events=[
            KgEvent(
                kind="bootstrap",
                at=now,
                torchcell_commit=commit,
                datasets=sorted(dataset_classes),
                biocypher_out=biocypher_out,
                note=f"reconstructed from the full build at {commit}; built_at {built_at}",
            )
        ],
        created_at=now,
        updated_at=now,
    )


# --------------------------------------------------------------------------- admission


def _dev_lmdb_status(
    dataset_class: type, data_root: Path, surface: SchemaSurface
) -> tuple[str, str]:
    root = data_root / _dataset_default_root(dataset_class)
    if not (root / "processed" / "lmdb").is_dir():
        return "missing", str(root)
    manifest_path = root / "preprocess" / MANIFEST_FILENAME
    if not manifest_path.exists():
        return "unmanifested", str(root)
    manifest = BuildManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = check_manifest(manifest, surface, str(root / "preprocess"))
    return ("stale" if result.is_stale else "fresh"), str(root)


def experiment_node_id(experiment: Any) -> str:
    """The content-addressed id ``CellAdapter._experiment_node`` gives an experiment.

    Mirrors the adapter byte for byte (sha256 of the json-dumped ``model_dump``, dict
    order as the model emits it) so a dev record can be matched against a served node
    without running the adapter. The adapter-drift check guards the mirror: a change to
    ``_experiment_node`` blocks every served dataset, and the test file pins the two.
    """
    return hashlib.sha256(
        json.dumps(experiment.model_dump()).encode("utf-8")
    ).hexdigest()


def dev_experiment_ids(dataset_class: type, data_root: Path) -> list[str]:
    """Experiment node ids the dev-tree LMDB of ``dataset_class`` produces, in order.

    Walks the adapter's own path: the raw item from the store, ``transform_item`` into
    the loader's experiment class, then ``experiment_node_id``.
    """
    dataset = dataset_class(root=str(data_root / _dataset_default_root(dataset_class)))
    ids = [
        experiment_node_id(dataset.transform_item(dataset[i])["experiment"])
        for i in range(len(dataset))
    ]
    dataset.close_lmdb()
    return ids


def superset_check(
    served_ids: Iterable[str], dev_ids: Iterable[str], served_source: str
) -> SupersetCheck:
    """Compare the ids a store holds for a dataset with the ids its dev LMDB produces."""
    served = set(served_ids)
    dev = set(dev_ids)
    missing = sorted(served - dev)
    return SupersetCheck(
        n_served=len(served),
        n_dev=len(dev),
        n_missing=len(missing),
        n_added=len(dev - served),
        missing_sample=missing[:5],
        served_source=served_source,
    )


def live_experiment_ids(
    uri: str, user: str, password: str, database: str, dataset_class_name: str
) -> list[str]:
    """Experiment node ids under one Dataset node of a running store.

    Reached through the Dataset node's ``ExperimentMemberOf`` edges, so it needs no
    property index on ``Experiment.id`` (the served store carries none between
    increments) and streams one dataset at a time, as ``kg_content_hashes.sh`` does.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (d:Dataset {id: $name})<-[:ExperimentMemberOf]-(e:Experiment) "
            "RETURN e.id AS id",
            name=dataset_class_name,
        )
        ids = [str(record["id"]) for record in result]
    driver.close()
    return ids


def adapter_drift_against(
    manifest: KgBuildManifest, repo_root: Path
) -> tuple[AdapterDrift, list[str]]:
    """Adapter changes since the build that touch served datasets; plus new methods."""
    methods_now, table_now = cell_adapter_surface(
        (repo_root / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    changed_fns = {
        fn
        for fn, digest in manifest.cell_adapter_methods.items()
        if methods_now.get(fn) != digest
    }
    added = sorted(set(methods_now) - set(manifest.cell_adapter_methods))
    table_fns = set(manifest.cell_adapter_table.values())
    plumbing = sorted(fn for fn in changed_fns if fn not in table_fns)

    served_methods: dict[str, list[str]] = {}
    served_files: dict[str, list[str]] = {}
    for name, entry in sorted(manifest.datasets.items()):
        cls = _dataset_class(name)
        for conf_name in dataset_conf_methods(cls, repo_root):
            fn = manifest.cell_adapter_table.get(conf_name)
            # a conf name unknown to the served table, or re-pointed to another
            # function, is drift for this dataset as well
            if fn is None or table_now.get(conf_name) != fn:
                served_methods.setdefault(f"table[{conf_name}]", []).append(name)
            elif fn in changed_fns:
                served_methods.setdefault(fn, []).append(name)
        for rel in entry.adapter_files:
            stored = manifest.adapter_files.get(rel)
            path = repo_root / rel
            current = (
                _sha256(path.read_text(encoding="utf-8")) if path.exists() else None
            )
            if stored is None or current != stored:
                served_files.setdefault(rel, []).append(name)
    return (
        AdapterDrift(
            plumbing_methods=plumbing,
            served_methods=served_methods,
            served_files=served_files,
        ),
        added,
    )


def check_admission(
    manifest: KgBuildManifest,
    repo_root: Path,
    dataset_class_name: str,
    data_root: Path,
    ack_adapter_drift: str | None = None,
    ack_value_drift: str | None = None,
    served_experiment_ids: Callable[[str], Iterable[str]] | None = None,
    served_source: str = "live store",
) -> AdmissionReport:
    """Decide whether ``dataset_class_name`` can be added to the store incrementally.

    ``served_experiment_ids`` reads the experiment ids the live store holds under a
    Dataset node (``live_experiment_ids`` bound to a connection in production, named by
    ``served_source`` in the report). It is called only when the dataset is already
    served, to run the superset proof; without it a served dataset blocks.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    commit, dirty = _git_info(repo_root)
    surface = surface_in_worktree(repo_root)
    reasons: list[str] = []

    # 1. served datasets: closure drift against the working-tree schema
    stale_served: list[ServedDrift] = []
    for name, entry in sorted(manifest.datasets.items()):
        current = closure_in_worktree(repo_root, entry.loader_relpath, surface)
        changed = sorted(
            symbol
            for symbol, stored in entry.closure.items()
            if current.get(symbol) != stored
        )
        changed.extend(sorted(set(current) - set(entry.closure)))
        if changed:
            stale_served.append(
                ServedDrift(dataset_class=name, changed_symbols=changed)
            )
    if stale_served:
        reasons.append(
            "served datasets whose schema closure changed since the build "
            "(full rebuild required): "
            + ", ".join(
                f"{d.dataset_class}[{', '.join(d.changed_symbols)}]"
                for d in stale_served
            )
        )

    # 2. graph schema: existing classes must be unchanged; new classes are additive
    current_schema = graph_schema_from_yaml(
        (repo_root / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )
    graph_schema_changed = sorted(
        name
        for name, entry in manifest.graph_schema.items()
        if not entry.served_nodes_unchanged_by(current_schema.get(name))
    )
    graph_schema_added = sorted(set(current_schema) - set(manifest.graph_schema))
    if graph_schema_changed:
        reasons.append(
            "graph schema classes present in the served store changed "
            f"(full rebuild required): {', '.join(graph_schema_changed)}"
        )

    # 3. adapter drift touching served datasets (blocks unless acknowledged)
    drift, methods_added = adapter_drift_against(manifest, repo_root)
    if not drift.is_empty and not ack_adapter_drift:
        reasons.append(
            "adapter code serving existing datasets changed since the build; their node "
            "ids may have moved. Review the diff, then re-run with --ack-adapter-drift "
            f"'<why served ids are unchanged>'. Drift: {drift.describe()}"
        )

    # 4. shared VALUE files: a changed recipe or identity row moves media/compound ids
    value_changed, value_added = value_surface_drift(
        manifest.value_surface, value_surface_in_worktree(repo_root)
    )
    if value_changed and not ack_value_drift:
        reasons.append(
            f"VALUE SURFACE CHANGED: {value_changed}. These files hold the shared VALUES "
            "served media and compound node ids are content-addressed from, so a served "
            "node and a node the new dataset writes for the same substance or medium will "
            "not share an id. Review the diff, then re-run with --ack-value-drift "
            "'<why served ids are unchanged>'."
        )

    # 5. the new dataset itself
    dataset_class = _dataset_class(dataset_class_name)
    in_map = dataset_class in dataset_adapter_map
    if not in_map:
        reasons.append(f"{dataset_class_name} is not in dataset_adapter_map")
    served = dataset_class_name in manifest.datasets
    rel = loader_relpath(dataset_class, repo_root)
    new_closure = closure_in_worktree(repo_root, rel, surface)
    shared: dict[str, list[str]] = {}
    for name, entry in sorted(manifest.datasets.items()):
        for symbol in entry.closure:
            if symbol in new_closure:
                shared.setdefault(symbol, []).append(name)
    novel = sorted(set(new_closure) - set(shared))
    lmdb_status, lmdb_root = _dev_lmdb_status(dataset_class, data_root, surface)
    if lmdb_status != "fresh":
        reasons.append(
            f"dev-tree LMDB for {dataset_class_name} is {lmdb_status} at {lmdb_root}; "
            "build it with python -m torchcell.database.build_dataset_lmdb"
        )
    undeclared: list[str] = []
    if in_map:
        for conf_name in dataset_conf_methods(dataset_class, repo_root):
            if conf_name.endswith("phenotype (chunked)"):
                label = conf_name[: -len(" (chunked)")]
                if label not in current_schema:
                    undeclared.append(label)
    if undeclared:
        reasons.append(
            "adapter enables phenotype node methods the graph schema does not declare "
            f"(BioCypher would drop them silently): {', '.join(undeclared)}"
        )

    # 6. already served: only a proven superset may be re-admitted
    superset: SupersetCheck | None = None
    if served and served_experiment_ids is None:
        reasons.append(
            f"{dataset_class_name} is already in the served store; re-admitting it needs "
            "the superset proof, which reads the served experiment ids from the live "
            "store (admit --neo4j-uri)"
        )
    elif served and lmdb_status == "fresh":
        assert served_experiment_ids is not None
        superset = superset_check(
            served_experiment_ids(dataset_class_name),
            dev_experiment_ids(dataset_class, data_root),
            served_source,
        )
        if superset.n_missing:
            reasons.append(
                f"{dataset_class_name} is already in the served store and its dev LMDB "
                f"no longer produces {superset.n_missing} of the {superset.n_served} "
                "served experiment ids (full rebuild required: incremental import would "
                "leave those nodes beside their replacements). First missing ids: "
                + ", ".join(superset.missing_sample)
            )
        elif superset.n_added == 0:
            reasons.append(
                f"{dataset_class_name} is already in the served store and its dev LMDB "
                f"produces exactly the {superset.n_served} served experiment ids; "
                "nothing to add"
            )

    return AdmissionReport(
        dataset_class=dataset_class_name,
        checked_at=_now(),
        torchcell_commit=commit,
        torchcell_dirty=dirty,
        served_commit=manifest.torchcell_commit,
        verdict="blocked" if reasons else "admissible",
        reasons=reasons,
        new_dataset_closure=new_closure,
        novel_symbols=novel,
        shared_symbols=shared,
        stale_served=stale_served,
        graph_schema_changed=graph_schema_changed,
        graph_schema_added=graph_schema_added,
        adapter_drift=drift,
        adapter_methods_added=methods_added,
        adapter_drift_acknowledged=ack_adapter_drift if not drift.is_empty else None,
        value_surface_changed=value_changed,
        value_surface_added=value_added,
        value_surface_recorded=bool(manifest.value_surface),
        value_drift_acknowledged=ack_value_drift if value_changed else None,
        dev_lmdb_status=lmdb_status,  # type: ignore[arg-type]
        dev_lmdb_root=lmdb_root,
        in_adapter_map=in_map,
        undeclared_phenotype_methods=undeclared,
        served=served,
        superset=superset,
    )


def batch_report_from_members(members: list[AdmissionReport]) -> BatchAdmissionReport:
    """Aggregate per-dataset admission reports into one batch verdict.

    Pure aggregation: the members were each decided against the served manifest, and the
    batch adds no rule of its own beyond "every member is admissible".
    """
    if not members:
        raise ValueError("a batch admission needs at least one dataset")
    names = [member.dataset_class for member in members]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"dataset named more than once in the batch: {repeated}")

    introduced: dict[str, list[str]] = {}
    for member in members:
        for symbol in member.novel_symbols:
            introduced.setdefault(symbol, []).append(member.dataset_class)
    co_introduced = {
        symbol: sharers
        for symbol, sharers in sorted(introduced.items())
        if len(sharers) > 1
    }

    # the inverse of stale_served: which SERVED datasets import each changed symbol
    importers: dict[str, set[str]] = {}
    for member in members:
        for drift in member.stale_served:
            for symbol in drift.changed_symbols:
                importers.setdefault(symbol, set()).add(drift.dataset_class)

    reasons = [
        f"{member.dataset_class}: {reason}"
        for member in members
        for reason in member.reasons
    ]
    first = members[0]
    return BatchAdmissionReport(
        dataset_classes=names,
        checked_at=_now(),
        torchcell_commit=first.torchcell_commit,
        torchcell_dirty=first.torchcell_dirty,
        served_commit=first.served_commit,
        verdict="blocked" if reasons else "admissible",
        reasons=reasons,
        members=members,
        co_introduced_symbols=co_introduced,
        changed_symbol_importers={
            symbol: sorted(datasets) for symbol, datasets in sorted(importers.items())
        },
    )


def check_batch_admission(
    manifest: KgBuildManifest,
    repo_root: Path,
    dataset_class_names: list[str],
    data_root: Path,
    ack_adapter_drift: str | None = None,
    ack_value_drift: str | None = None,
    served_experiment_ids: Callable[[str], Iterable[str]] | None = None,
    served_source: str = "live store",
) -> BatchAdmissionReport:
    """Decide whether every named dataset can be added in ONE incremental import."""
    return batch_report_from_members(
        [
            check_admission(
                manifest,
                repo_root,
                name,
                data_root,
                ack_adapter_drift,
                ack_value_drift,
                served_experiment_ids,
                served_source,
            )
            for name in dataset_class_names
        ]
    )


def _dataset_entry(
    report: AdmissionReport,
    *,
    biocypher_out: str,
    n_experiments: int | None,
    repo_root: Path,
    at: str,
    previous: KgDatasetEntry | None,
) -> KgDatasetEntry:
    """The manifest entry for one admitted dataset.

    ``previous`` is the entry the store already held for it, present exactly when the
    admission was a superset; the new entry records that lineage.
    """
    if report.served != (previous is not None):
        raise ValueError(
            f"{report.dataset_class}: report says served={report.served} but the "
            f"manifest {'has' if previous else 'has no'} entry for it"
        )
    if previous is not None and report.superset is None:
        raise ValueError(
            f"{report.dataset_class} is served but the report carries no superset proof"
        )
    dataset_class = _dataset_class(report.dataset_class)
    lineage = None
    if previous is not None:
        assert report.superset is not None
        lineage = SupersetLineage(
            biocypher_out=previous.biocypher_out,
            import_mode=previous.import_mode,
            admitted_at=previous.admitted_at,
            n_experiments=previous.n_experiments,
            n_added=report.superset.n_added,
        )
    return KgDatasetEntry(
        dataset_class=report.dataset_class,
        loader_relpath=loader_relpath(dataset_class, repo_root),
        adapter_files=dataset_adapter_files(dataset_class, repo_root),
        closure=report.new_dataset_closure,
        n_experiments=n_experiments,
        biocypher_out=biocypher_out,
        import_mode="incremental",
        admitted_at=at,
        torchcell_commit=report.torchcell_commit,
        superset_of=lineage,
    )


def _event_kind(
    members: list[AdmissionReport],
) -> Literal["incremental_admission", "superset_admission"]:
    """A superset admission when any member re-admits a served dataset."""
    if any(member.served for member in members):
        return "superset_admission"
    return "incremental_admission"


def _superset_note(members: list[AdmissionReport]) -> str | None:
    """``Dataset +N`` for every member that grew a served dataset."""
    grown = [
        f"{member.dataset_class} +{member.superset.n_added} "
        f"(served {member.superset.n_served})"
        for member in members
        if member.superset is not None
    ]
    return "superset of served: " + "; ".join(grown) if grown else None


def _adopt_current_surfaces(manifest: KgBuildManifest, repo_root: Path) -> None:
    """Make the graph schema and adapter surface now in force the manifest's reference.

    The store contains nodes produced by them once the import has succeeded.
    """
    manifest.graph_schema = graph_schema_from_yaml(
        (repo_root / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )
    methods, table = cell_adapter_surface(
        (repo_root / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    manifest.cell_adapter_methods = methods
    manifest.cell_adapter_table = table
    manifest.adapter_files = {
        rel: _sha256((repo_root / rel).read_text(encoding="utf-8"))
        for rel in adapter_file_relpaths(repo_root)
    }
    manifest.value_surface = value_surface_in_worktree(repo_root)


def _acknowledged_drift(report: AdmissionReport) -> list[str]:
    if not report.adapter_drift_acknowledged:
        return []
    return [f"{report.adapter_drift.describe()}: {report.adapter_drift_acknowledged}"]


def _acknowledged_value_drift(report: AdmissionReport) -> list[str]:
    if not report.value_drift_acknowledged:
        return []
    return [
        f"{', '.join(report.value_surface_changed)}: {report.value_drift_acknowledged}"
    ]


def record_admission(
    manifest: KgBuildManifest,
    report: AdmissionReport,
    *,
    biocypher_out: str,
    n_experiments: int | None,
    repo_root: Path,
) -> KgBuildManifest:
    """Append an admitted dataset to the manifest (call after the import succeeded).

    The graph schema and adapter surface now in force become the manifest's reference,
    since the store now contains nodes produced by them.
    """
    if report.verdict != "admissible":
        raise ValueError(f"cannot record a blocked admission: {report.reasons}")
    now = _now()
    manifest.datasets[report.dataset_class] = _dataset_entry(
        report,
        biocypher_out=biocypher_out,
        n_experiments=n_experiments,
        repo_root=repo_root,
        at=now,
        previous=manifest.datasets.get(report.dataset_class),
    )
    _adopt_current_surfaces(manifest, repo_root)
    manifest.events.append(
        KgEvent(
            kind=_event_kind([report]),
            at=now,
            torchcell_commit=report.torchcell_commit,
            datasets=[report.dataset_class],
            biocypher_out=biocypher_out,
            note=_superset_note([report]),
            acknowledged_adapter_drift=_acknowledged_drift(report),
            acknowledged_value_drift=_acknowledged_value_drift(report),
        )
    )
    return manifest


def record_batch_admission(
    manifest: KgBuildManifest,
    report: BatchAdmissionReport,
    *,
    biocypher_out: str,
    n_experiments: dict[str, int],
    repo_root: Path,
) -> KgBuildManifest:
    """Record every member of an admitted batch under ONE event.

    The members were imported by a single ``neo4j-admin database import incremental``
    call from a single BioCypher output directory, so they share ``biocypher_out`` and
    one event; the per-dataset experiment counts are the live ones the runner verified.
    """
    if report.verdict != "admissible":
        raise ValueError(f"cannot record a blocked batch admission: {report.reasons}")
    missing = sorted(set(report.dataset_classes) - set(n_experiments))
    if missing:
        raise ValueError(f"no experiment count given for {missing}")
    unexpected = sorted(set(n_experiments) - set(report.dataset_classes))
    if unexpected:
        raise ValueError(
            f"experiment counts for datasets not in the batch: {unexpected}"
        )
    now = _now()
    for member in report.members:
        manifest.datasets[member.dataset_class] = _dataset_entry(
            member,
            biocypher_out=biocypher_out,
            n_experiments=n_experiments[member.dataset_class],
            repo_root=repo_root,
            at=now,
            previous=manifest.datasets.get(member.dataset_class),
        )
    _adopt_current_surfaces(manifest, repo_root)
    acknowledged: list[str] = []
    acknowledged_values: list[str] = []
    for member in report.members:
        # the drift is measured against the same manifest for every member, so the
        # acknowledgment text repeats; record it once
        acknowledged.extend(
            line for line in _acknowledged_drift(member) if line not in acknowledged
        )
        acknowledged_values.extend(
            line
            for line in _acknowledged_value_drift(member)
            if line not in acknowledged_values
        )
    manifest.events.append(
        KgEvent(
            kind=_event_kind(report.members),
            at=now,
            torchcell_commit=report.torchcell_commit,
            datasets=list(report.dataset_classes),
            biocypher_out=biocypher_out,
            note=_superset_note(report.members),
            acknowledged_adapter_drift=acknowledged,
            acknowledged_value_drift=acknowledged_values,
        )
    )
    return manifest


# --------------------------------------------------------------------------- live store


def live_dataset_counts(
    uri: str, user: str, password: str, database: str
) -> dict[str, int]:
    """``Dataset.id -> number of ExperimentMemberOf edges`` from a running store."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        rows = session.run(
            "MATCH (d:Dataset) OPTIONAL MATCH (d)<-[:ExperimentMemberOf]-(e:Experiment) "
            "RETURN d.id AS id, count(e) AS n ORDER BY id"
        ).values()
    driver.close()
    return {row[0]: int(row[1]) for row in rows}


def live_neo4j_version(uri: str, user: str, password: str) -> str:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database="system") as session:
        row = session.run(
            "CALL dbms.components() YIELD versions RETURN versions"
        ).single()
    driver.close()
    return str(row["versions"][0])


# --------------------------------------------------------------------------- CLI


def _format_value_surface(report: AdmissionReport) -> str:
    """The value-surface line: not recorded, unchanged, or the files that changed."""
    if not report.value_surface_recorded:
        return (
            "not recorded (manifest predates the value surface; nothing to compare "
            "against, so this does not block)"
        )
    added = (
        f"; added since the build (additive): {', '.join(report.value_surface_added)}"
        if report.value_surface_added
        else ""
    )
    if not report.value_surface_changed:
        return f"unchanged ({len(VALUE_SURFACE_RELPATHS)} files){added}"
    acknowledged = (
        f" (acknowledged: {report.value_drift_acknowledged})"
        if report.value_drift_acknowledged
        else ""
    )
    return f"CHANGED: {', '.join(report.value_surface_changed)}{acknowledged}{added}"


def format_report(report: AdmissionReport) -> str:
    """One dataset's verdict, with the evidence each blocker is decided from."""
    lines = [
        f"Admission check: {report.dataset_class}  ->  {report.verdict.upper()}",
        f"  working tree {report.torchcell_commit} (dirty={report.torchcell_dirty}); "
        f"served store built at {report.served_commit}",
        f"  dev LMDB: {report.dev_lmdb_status} ({report.dev_lmdb_root})",
        f"  closure: {len(report.new_dataset_closure)} symbols, "
        f"{len(report.novel_symbols)} novel ({', '.join(report.novel_symbols) or '-'}), "
        f"{len(report.shared_symbols)} shared with served datasets",
        f"  served datasets with schema drift: {len(report.stale_served)}",
        f"  graph schema: {len(report.graph_schema_changed)} changed, "
        f"{len(report.graph_schema_added)} added ({', '.join(report.graph_schema_added) or '-'})",
        f"  adapter methods added: {', '.join(report.adapter_methods_added) or '-'}",
        f"  adapter drift touching served datasets: {report.adapter_drift.describe() or 'none'}"
        + (
            f" (acknowledged: {report.adapter_drift_acknowledged})"
            if report.adapter_drift_acknowledged
            else ""
        ),
        f"  value surface: {_format_value_surface(report)}",
        f"  served: {_format_served(report)}",
    ]
    for reason in report.reasons:
        lines.append(f"  [BLOCK] {reason}")
    return "\n".join(lines)


def _format_served(report: AdmissionReport) -> str:
    """Whether the dataset is already served and, if so, the superset proof."""
    if not report.served:
        return "no (new dataset)"
    if report.superset is None:
        return "yes; superset proof not run"
    s = report.superset
    return (
        f"yes; superset proof from {s.served_source}: {s.n_served} served ids, "
        f"{s.n_dev} in the dev LMDB, {s.n_missing} served ids missing from it, "
        f"{s.n_added} to add"
    )


def format_batch_report(report: BatchAdmissionReport) -> str:
    """The batch verdict, every member's report, and the two cross-member views."""
    lines = [
        f"Batch admission check: {', '.join(report.dataset_classes)}  ->  "
        f"{report.verdict.upper()}",
        f"  working tree {report.torchcell_commit} (dirty={report.torchcell_dirty}); "
        f"served store built at {report.served_commit}",
    ]
    for member in report.members:
        lines.extend(f"  {line}" for line in format_report(member).splitlines())
    lines.append(
        "  symbols introduced by more than one batch member (additive): "
        + (
            ", ".join(
                f"{symbol} ({', '.join(sharers)})"
                for symbol, sharers in report.co_introduced_symbols.items()
            )
            or "none"
        )
    )
    if report.changed_symbol_importers:
        lines.append("  changed symbols and the SERVED datasets that import them:")
        for symbol, datasets in report.changed_symbol_importers.items():
            in_batch = sorted(
                member.dataset_class
                for member in report.members
                if symbol in member.new_dataset_closure
            )
            lines.append(
                f"    {symbol}: served {', '.join(datasets)}"
                + (
                    f"; batch members importing it: {', '.join(in_batch)}"
                    if in_batch
                    else ""
                )
            )
    for reason in report.reasons:
        lines.append(f"  [BLOCK] {reason}")
    return "\n".join(lines)


def load_report(path: Path) -> AdmissionReport | BatchAdmissionReport:
    """Read an admission report; a BATCH report is the one carrying ``members``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "members" in data:
        return BatchAdmissionReport.model_validate(data)
    return AdmissionReport.model_validate(data)


def split_dataset_args(values: list[str]) -> list[str]:
    """Dataset names from repeated ``--dataset`` values, each optionally a comma list."""
    names: list[str] = []
    for value in values:
        for piece in value.split(","):
            name = piece.strip()
            if not name:
                raise ValueError(f"empty dataset name in --dataset {value!r}")
            if name in names:
                raise ValueError(f"dataset named more than once: {name}")
            names.append(name)
    return names


def parse_n_experiments(
    values: list[str], dataset_classes: list[str]
) -> dict[str, int]:
    """``--n-experiments`` values as ``dataset -> count``.

    A single dataset takes a bare count (``--n-experiments 6188``); a batch takes one
    ``NAME=COUNT`` per member, and the names must be exactly the batch's members.
    """
    counts: dict[str, int] = {}
    for value in values:
        if "=" in value:
            name, _, raw = value.partition("=")
        elif len(dataset_classes) == 1:
            name, raw = dataset_classes[0], value
        else:
            raise ValueError(
                f"--n-experiments {value!r} needs the NAME=COUNT form for a batch of "
                f"{len(dataset_classes)} datasets"
            )
        if name in counts:
            raise ValueError(f"--n-experiments given twice for {name}")
        counts[name] = int(raw)
    missing = sorted(set(dataset_classes) - set(counts))
    if missing:
        raise ValueError(f"--n-experiments missing for {missing}")
    unexpected = sorted(set(counts) - set(dataset_classes))
    if unexpected:
        raise ValueError(
            f"--n-experiments names datasets not in the report: {unexpected}"
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.kg_manifest",
        description="Served knowledge-graph manifest: bootstrap, admission check, record.",
    )
    parser.add_argument("--manifest", required=True, help="path of kg_manifest.json")
    parser.add_argument(
        "--repo", default=None, help="torchcell checkout (default: cwd)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser(
        "bootstrap", help="reconstruct the manifest of a full build"
    )
    p_boot.add_argument("--commit", required=True)
    p_boot.add_argument("--biocypher-out", required=True)
    p_boot.add_argument("--built-at", required=True, help="ISO timestamp of the build")
    p_boot.add_argument("--biocypher-version", required=True)
    p_boot.add_argument("--database", default="torchcell")
    p_boot.add_argument("--store-host", default=socket.gethostname())
    p_boot.add_argument(
        "--neo4j-uri", default=None, help="query datasets + counts live"
    )

    p_admit = sub.add_parser("admit", help="check whether a dataset can be added")
    p_admit.add_argument(
        "--dataset",
        required=True,
        action="append",
        help="dataset class name; repeat the flag, or pass a comma-separated list, to "
        "check a BATCH admitted in one incremental import",
    )
    p_admit.add_argument("--data-root", required=True, help="dev tree holding the LMDB")
    p_admit.add_argument("--ack-adapter-drift", default=None)
    p_admit.add_argument(
        "--ack-value-drift",
        default=None,
        help="why a changed shared VALUE file (media.py, compound_identity*) leaves the "
        "served media/compound node ids unchanged; recorded in the manifest event",
    )
    p_admit.add_argument("--report", default=None, help="write the JSON report here")
    p_admit.add_argument(
        "--neo4j-uri",
        default=None,
        help="bolt URI of the served store (default: NEO4J_URI or the connection "
        "default); read only when a named dataset is already served, to prove the "
        "re-admission is a superset of what the store holds",
    )
    p_admit.add_argument("--database", default="torchcell")

    p_rec = sub.add_parser("record", help="record a completed incremental admission")
    p_rec.add_argument("--report", required=True, help="the admission report JSON")
    p_rec.add_argument("--biocypher-out", required=True)
    p_rec.add_argument(
        "--n-experiments",
        action="append",
        required=True,
        help="live experiment count: COUNT for a single dataset, or NAME=COUNT "
        "repeated, one per member, for a batch report",
    )

    sub.add_parser("show", help="print the manifest summary")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo or ".").resolve()
    manifest_path = Path(args.manifest)

    if args.command == "bootstrap":
        from torchcell.database.connection import neo4j_connection_settings

        settings = neo4j_connection_settings()
        uri = args.neo4j_uri or settings.uri
        counts = live_dataset_counts(
            uri, settings.username, settings.password, args.database
        )
        manifest = bootstrap_manifest(
            repo_root=repo_root,
            commit=args.commit,
            dataset_classes=sorted(counts),
            n_experiments=counts,
            database=args.database,
            store_host=args.store_host,
            neo4j_version=live_neo4j_version(uri, settings.username, settings.password),
            biocypher_version=args.biocypher_version,
            biocypher_out=args.biocypher_out,
            built_at=args.built_at,
        )
        save_manifest(manifest, manifest_path)
        print(f"bootstrapped {len(manifest.datasets)} datasets -> {manifest_path}")
        return 0

    manifest = load_manifest(manifest_path)
    if args.command == "show":
        print(
            f"{manifest.database} on {manifest.store_host}: neo4j {manifest.neo4j_version}, "
            f"biocypher {manifest.biocypher_version}, full build {manifest.torchcell_commit}, "
            f"{len(manifest.datasets)} datasets, {len(manifest.events)} events"
        )
        for name, entry in sorted(manifest.datasets.items()):
            print(
                f"  {name}: {entry.n_experiments} experiments "
                f"({entry.import_mode}, {entry.biocypher_out})"
            )
        return 0
    if args.command == "admit":
        from torchcell.database.connection import neo4j_connection_settings

        names = split_dataset_args(args.dataset)
        settings = neo4j_connection_settings()
        uri = args.neo4j_uri or settings.uri

        def served_ids(name: str) -> list[str]:
            return live_experiment_ids(
                uri, settings.username, settings.password, args.database, name
            )

        if len(names) == 1:
            report = check_admission(
                manifest,
                repo_root,
                names[0],
                Path(args.data_root),
                args.ack_adapter_drift,
                args.ack_value_drift,
                served_ids,
                uri,
            )
            print(format_report(report))
            if args.report:
                Path(args.report).write_text(
                    report.model_dump_json(indent=2), encoding="utf-8"
                )
            return 0 if report.verdict == "admissible" else 1
        batch = check_batch_admission(
            manifest,
            repo_root,
            names,
            Path(args.data_root),
            args.ack_adapter_drift,
            args.ack_value_drift,
            served_ids,
            uri,
        )
        print(format_batch_report(batch))
        if args.report:
            Path(args.report).write_text(
                batch.model_dump_json(indent=2), encoding="utf-8"
            )
        return 0 if batch.verdict == "admissible" else 1
    if args.command == "record":
        loaded = load_report(Path(args.report))
        if isinstance(loaded, BatchAdmissionReport):
            record_batch_admission(
                manifest,
                loaded,
                biocypher_out=args.biocypher_out,
                n_experiments=parse_n_experiments(
                    args.n_experiments, loaded.dataset_classes
                ),
                repo_root=repo_root,
            )
            save_manifest(manifest, manifest_path)
            print(f"recorded {', '.join(loaded.dataset_classes)} -> {manifest_path}")
            return 0
        counts = parse_n_experiments(args.n_experiments, [loaded.dataset_class])
        record_admission(
            manifest,
            loaded,
            biocypher_out=args.biocypher_out,
            n_experiments=counts[loaded.dataset_class],
            repo_root=repo_root,
        )
        save_manifest(manifest, manifest_path)
        print(f"recorded {loaded.dataset_class} -> {manifest_path}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

# torchcell/knowledge_graphs/supported_queries/check.py
# [[torchcell.knowledge_graphs.supported_queries.check]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/supported_queries/check.py
# Test file: tests/torchcell/knowledge_graphs/supported_queries/test_check.py
"""The supported-query drift check: does each registered query still hold on a release?

The check compares every registered query with a committed release snapshot
(``database/releases/<release>.json`` and its ``.closures.json``, written by ``stamp``)
and with the checkout's schema surface. It needs no store, no GilaHyper manifest and no
torch: pydantic, PyYAML and python-dotenv are its whole import closure, which is what the
``query-drift`` CI job installs.

Drift kinds (:class:`QueryDrift`):

- ``file_missing``: the ``.cql`` file is not in the checkout.
- ``missing_node_label``: a label the query matches is not a node label of the release.
  A class ``foo bar`` of the snapshot's ``graph_schema`` is the label ``FooBar`` (BioCypher's
  sentence-to-Pascal rule); an ancestor declared by ``is_a`` in the checkout's
  ``biocypher/config/torchcell_schema_config.yaml`` is a label too (``PhenotypicFeature``
  on every ``... phenotype`` node). The snapshot does not record ``is_a``, so the checkout's
  config supplies the hierarchy; Biolink ancestors above a declared ``is_a`` are not known.
- ``missing_relationship_type``: a relationship type that is not an ``edge`` class.
- ``missing_property``: a ``Label.prop`` read (see ``cypher_deps``) whose label is a node
  label of the release but whose class does not carry ``prop``. ``id`` and
  ``preferred_id``, which BioCypher writes on every node, always count. A label that is
  only an ancestor must have ``prop`` on EVERY class under it: a class without it would
  drop out of the query silently.
- ``missing_dataset``: a ``dataset.id`` literal the release does not serve.
- ``contract_changed``: a class of ``phenotype_classes``, or a surface class it references
  transitively, whose contract fingerprint in the checkout differs from the closure the
  release recorded for a selected dataset that serves it (the records would serialize
  differently now); also a listed class that no selected dataset's closure contains.
- ``composite_changed``: the recorded ``dataset_composite`` differs from the composite of
  the selected datasets in the release, i.e. the records the query returns changed (or the
  query was never validated).
- ``converter_missing``: the converter's module is not in the checkout or does not define
  the class at top level (read from source with ``ast``; nothing is imported).

A ``supported`` query with any drift makes the check exit 1; a ``deprecated`` one is
reported and never fails. ``validate`` records a release as the one a query holds on.
"""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from torchcell.knowledge_graphs.kg_manifest import GraphSchemaEntry, surface_in_worktree
from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    composite_sha256,
    load_closures,
    load_snapshot,
    load_snapshots,
    snapshot_paths,
)
from torchcell.knowledge_graphs.supported_queries.cypher_deps import (
    extract_dependencies,
)
from torchcell.knowledge_graphs.supported_queries.registry import (
    KG_PACKAGE_RELPATH,
    QueryDependencies,
    QueryRegistry,
    SupportedQuery,
)
from torchcell.provenance.schema_deps import SchemaSurface, forward_closure

__all__ = [
    "DriftKind",
    "SCHEMA_CONFIG_RELPATH",
    "IMPLICIT_NODE_PROPERTIES",
    "QueryDrift",
    "QueryResult",
    "CheckReport",
    "GraphLabels",
    "pascal_label",
    "schema_is_a",
    "graph_labels",
    "selected_composite",
    "query_drifts",
    "run_check",
    "checkout_commit",
    "resolve_snapshot",
    "check_repo",
    "validate_query",
]

DriftKind = Literal[
    "file_missing",
    "missing_node_label",
    "missing_relationship_type",
    "missing_property",
    "missing_dataset",
    "contract_changed",
    "composite_changed",
    "converter_missing",
]

SCHEMA_CONFIG_RELPATH = "biocypher/config/torchcell_schema_config.yaml"
# BioCypher writes these on every node whatever the schema config lists.
IMPLICIT_NODE_PROPERTIES = frozenset({"id", "preferred_id"})


class QueryDrift(BaseModel):
    """One way a registered query no longer holds on a release."""

    query_id: str
    kind: DriftKind
    detail: str


class QueryResult(BaseModel):
    """One query's verdict, with the issue the CI job files when it fails."""

    query_id: str
    title: str
    status: Literal["supported", "deprecated"]
    cql_path: str
    validated_release: str | None
    drifts: list[QueryDrift]
    issue_title: str
    issue_body: str

    @property
    def failing(self) -> bool:
        """True for a supported query with at least one drift."""
        return self.status == "supported" and bool(self.drifts)


class CheckReport(BaseModel):
    """The check over a whole registry against one release."""

    release: str
    release_version: str
    release_commit: str
    checkout_commit: str
    exit_code: int
    results: list[QueryResult]
    report_markdown: str


class GraphLabels(BaseModel):
    """Node labels and relationship types a release carries, with node properties.

    ``node_properties[label]`` is the set of properties every node carrying ``label``
    has: the class's own list (plus the implicit ones) for a concrete class, the
    intersection over its classes for an ancestor-only label.
    """

    node_properties: dict[str, list[str]]
    relationship_types: list[str]


def pascal_label(name: str) -> str:
    """BioCypher's label for a schema class: ``fitness phenotype`` -> ``FitnessPhenotype``."""
    return "".join(word[:1].upper() + word[1:] for word in name.split())


def schema_is_a(config_text: str) -> dict[str, list[str]]:
    """``class -> [parent, ...]`` from the ``is_a`` entries of a BioCypher schema config."""
    config = yaml.safe_load(config_text)
    parents: dict[str, list[str]] = {}
    for name, entry in config.items():
        if not isinstance(entry, dict) or "is_a" not in entry:
            continue
        value = entry["is_a"]
        parents[name] = [value] if isinstance(value, str) else list(value)
    return parents


def _ancestors(name: str, is_a: Mapping[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(is_a.get(name, []))
    while stack:
        parent = stack.pop()
        if parent not in seen:
            seen.add(parent)
            stack.extend(is_a.get(parent, []))
    return seen


def graph_labels(
    graph_schema: Mapping[str, GraphSchemaEntry], is_a: Mapping[str, list[str]]
) -> GraphLabels:
    """The labels and types of a release's graph schema under the ``is_a`` hierarchy."""
    concrete: dict[str, set[str]] = {}
    under_ancestor: dict[str, list[set[str]]] = {}
    relationship_types: set[str] = set()
    for name, entry in graph_schema.items():
        if entry.kind == "edge":
            relationship_types.add(pascal_label(name))
            continue
        props = set(entry.properties) | IMPLICIT_NODE_PROPERTIES
        concrete[pascal_label(name)] = props
        for ancestor in _ancestors(name, is_a):
            under_ancestor.setdefault(pascal_label(ancestor), []).append(props)
    node_properties = {label: sorted(props) for label, props in concrete.items()}
    for label, prop_sets in under_ancestor.items():
        if label not in concrete:
            node_properties[label] = sorted(set.intersection(*prop_sets))
    return GraphLabels(
        node_properties=dict(sorted(node_properties.items())),
        relationship_types=sorted(relationship_types),
    )


def selected_composite(snapshot: KgReleaseSnapshot, dataset_ids: list[str]) -> str:
    """``composite_sha256`` over the selected datasets the release serves."""
    return composite_sha256(
        {
            name: snapshot.datasets[name]
            for name in dataset_ids
            if name in snapshot.datasets
        }
    )


def _converter_defined(converter: str, repo_root: Path) -> str | None:
    """None when the converter class is defined in the checkout, else why not."""
    module, _, class_name = converter.rpartition(".")
    path = repo_root / (module.replace(".", "/") + ".py")
    if not path.exists():
        return f"converter module {module} is not in the checkout ({path.name} absent)"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if class_name not in names:
        return f"{module} defines no top-level class {class_name}"
    return None


def _contract_drifts(
    query: SupportedQuery,
    selected: list[str],
    closures: Mapping[str, Mapping[str, str]],
    surface: SchemaSurface,
    release: str,
) -> list[QueryDrift]:
    drifts: list[QueryDrift] = []
    for phenotype in query.phenotype_classes:
        serving = [name for name in selected if phenotype in closures.get(name, {})]
        if not serving:
            drifts.append(
                QueryDrift(
                    query_id=query.id,
                    kind="contract_changed",
                    detail=(
                        f"{phenotype} is in the recorded closure of none of the selected "
                        f"datasets in {release}"
                    ),
                )
            )
            continue
        symbols = (
            forward_closure({phenotype}, surface.ref_graph)
            if phenotype in surface.names
            else {phenotype}
        )
        changed: set[str] = set()
        datasets: list[str] = []
        for name in serving:
            recorded = closures[name]
            moved = {
                symbol
                for symbol in symbols | {phenotype}
                if symbol in recorded
                and surface.fingerprints.get(symbol) != recorded[symbol]
            }
            if moved:
                changed |= moved
                datasets.append(name)
        if changed:
            gone = sorted(s for s in changed if s not in surface.fingerprints)
            detail = (
                f"{phenotype}: contract of {', '.join(sorted(changed))} differs from the "
                f"closure {release} recorded for {', '.join(datasets)}"
            )
            if gone:
                detail += f" ({', '.join(gone)} absent from the checkout's schema)"
            drifts.append(
                QueryDrift(query_id=query.id, kind="contract_changed", detail=detail)
            )
    return drifts


def query_drifts(
    query: SupportedQuery,
    deps: QueryDependencies | None,
    snapshot: KgReleaseSnapshot,
    closures: Mapping[str, Mapping[str, str]],
    surface: SchemaSurface,
    labels: GraphLabels,
    repo_root: Path,
) -> list[QueryDrift]:
    """Every drift of ``query`` against ``snapshot``; ``deps`` None means no file."""

    def drift(kind: DriftKind, detail: str) -> QueryDrift:
        return QueryDrift(query_id=query.id, kind=kind, detail=detail)

    release = snapshot.release
    if deps is None:
        return [
            drift(
                "file_missing",
                f"{KG_PACKAGE_RELPATH}/{query.cql_path} is not in the checkout",
            )
        ]
    drifts: list[QueryDrift] = []
    for label in deps.node_labels:
        if label not in labels.node_properties:
            drifts.append(
                drift("missing_node_label", f"{label} is not a node label of {release}")
            )
    for rel_type in deps.relationship_types:
        if rel_type not in labels.relationship_types:
            drifts.append(
                drift(
                    "missing_relationship_type",
                    f"{rel_type} is not a relationship type of {release}",
                )
            )
    for read in deps.label_properties:
        label, _, prop = read.partition(".")
        carried = labels.node_properties.get(label)
        if carried is not None and prop not in carried:
            drifts.append(
                drift(
                    "missing_property",
                    f"{label}.{prop}: {label} nodes of {release} do not all carry {prop}",
                )
            )
    for name in deps.dataset_ids:
        if name not in snapshot.datasets:
            drifts.append(
                drift("missing_dataset", f"{name} is not served by {release}")
            )
    drifts.extend(
        _contract_drifts(
            query,
            [name for name in deps.dataset_ids if name in snapshot.datasets],
            closures,
            surface,
            release,
        )
    )
    composite = selected_composite(snapshot, deps.dataset_ids)
    if query.dataset_composite != composite:
        recorded = (
            f"recorded {query.dataset_composite} (validated on {query.validated_release})"
            if query.dataset_composite is not None
            else "never validated"
        )
        drifts.append(
            drift(
                "composite_changed",
                f"selected datasets compose to {composite} in {release}; {recorded}",
            )
        )
    if query.converter is not None:
        reason = _converter_defined(query.converter, repo_root)
        if reason is not None:
            drifts.append(drift("converter_missing", reason))
    return drifts


def _issue_body(
    query: SupportedQuery,
    drifts: list[QueryDrift],
    snapshot: KgReleaseSnapshot,
    commit: str,
) -> str:
    lines = [
        f"Supported query `{query.id}` ({query.title}) drifts against KG release "
        f"`{snapshot.release}` (version {snapshot.version}, built from "
        f"`{snapshot.torchcell_commit[:8]}`).",
        "",
        f"- Query: `{KG_PACKAGE_RELPATH}/{query.cql_path}`, status `{query.status}`",
        f"- Last validated on: `{query.validated_release or 'never'}`",
        f"- Checkout commit: `{commit}`",
        "",
        "Drift:",
        "",
    ]
    lines += [f"- `{d.kind}`: {d.detail}" for d in drifts]
    lines += [
        "",
        "After the next KG build, re-validate with "
        f"`python -m torchcell.knowledge_graphs.supported_queries validate {query.id} "
        "--release <new release>` (or deprecate the query) and close this issue by hand.",
    ]
    return "\n".join(lines) + "\n"


def _report_markdown(
    results: list[QueryResult], snapshot: KgReleaseSnapshot, commit: str
) -> str:
    drifted = [r for r in results if r.drifts]
    lines = [
        f"## Supported queries against `{snapshot.release}` (KG {snapshot.version})",
        "",
        f"Checkout `{commit}`: {len(results)} queries, {len(drifted)} drifted, "
        f"{sum(r.failing for r in results)} failing.",
        "",
    ]
    for result in results:
        verdict = "ok" if not result.drifts else f"{len(result.drifts)} drift(s)"
        lines.append(f"- `{result.query_id}` ({result.status}): {verdict}")
    for result in drifted:
        lines += ["", f"### {result.issue_title}", "", result.issue_body.rstrip("\n")]
    return "\n".join(lines) + "\n"


def run_check(
    registry: QueryRegistry,
    snapshot: KgReleaseSnapshot,
    closures: Mapping[str, Mapping[str, str]],
    surface: SchemaSurface,
    schema_config_text: str,
    repo_root: Path,
    commit: str,
) -> CheckReport:
    """Check every registered query; exit code 1 when a supported query drifts."""
    labels = graph_labels(snapshot.graph_schema, schema_is_a(schema_config_text))
    results: list[QueryResult] = []
    for query in sorted(registry.queries, key=lambda q: q.id):
        path = query.cql_file(repo_root)
        deps = (
            extract_dependencies(path.read_text(encoding="utf-8"))
            if path.exists()
            else None
        )
        drifts = query_drifts(
            query, deps, snapshot, closures, surface, labels, repo_root
        )
        results.append(
            QueryResult(
                query_id=query.id,
                title=query.title,
                status=query.status,
                cql_path=query.cql_path,
                validated_release=query.validated_release,
                drifts=drifts,
                issue_title=(
                    f"Before the next KG build: supported query {query.id} drifts "
                    f"against {snapshot.release}"
                ),
                issue_body=_issue_body(query, drifts, snapshot, commit),
            )
        )
    return CheckReport(
        release=snapshot.release,
        release_version=snapshot.version,
        release_commit=snapshot.torchcell_commit,
        checkout_commit=commit,
        exit_code=1 if any(r.failing for r in results) else 0,
        results=results,
        report_markdown=_report_markdown(results, snapshot, commit),
    )


def checkout_commit(repo_root: Path) -> str:
    """``git rev-parse HEAD`` of the checkout at ``repo_root``."""
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def resolve_snapshot(repo_root: Path, release: str | None) -> KgReleaseSnapshot:
    """The named release's snapshot, or the newest committed one."""
    if release is not None:
        return load_snapshot(snapshot_paths(repo_root, release)[0])
    snapshots = load_snapshots(repo_root)
    if not snapshots:
        raise FileNotFoundError(
            f"no release snapshots under {repo_root}/database/releases"
        )
    return snapshots[-1]


def check_repo(
    repo_root: Path, registry: QueryRegistry, release: str | None
) -> CheckReport:
    """:func:`run_check` with every input read from the checkout at ``repo_root``."""
    snapshot = resolve_snapshot(repo_root, release)
    return run_check(
        registry,
        snapshot,
        load_closures(repo_root, snapshot.release),
        surface_in_worktree(repo_root),
        (repo_root / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8"),
        repo_root,
        checkout_commit(repo_root),
    )


def validate_query(
    registry: QueryRegistry, query_id: str, repo_root: Path, release: str
) -> tuple[QueryRegistry, list[QueryDrift]]:
    """Record ``release`` as the one ``query_id`` holds on.

    Returns the updated registry and no drifts, or the unchanged registry and the drifts
    that refuse the validation. Only ``composite_changed`` is resolved by validating (it
    is what a new release is expected to change); any other drift is a query or schema
    problem that must be fixed first. Sets ``since_kg_version`` when it is unset.
    """
    query = registry.get(query_id)
    report = check_repo(repo_root, QueryRegistry(queries=[query]), release)
    (result,) = report.results
    blocking = [d for d in result.drifts if d.kind != "composite_changed"]
    if blocking:
        return registry, blocking
    snapshot = resolve_snapshot(repo_root, release)
    deps = extract_dependencies(query.cql_file(repo_root).read_text(encoding="utf-8"))
    updated = query.model_copy(
        update={
            "validated_release": snapshot.release,
            "dataset_composite": selected_composite(snapshot, deps.dataset_ids),
            "since_kg_version": query.since_kg_version or snapshot.version,
        }
    )
    return registry.replace(SupportedQuery.model_validate(updated.model_dump())), []

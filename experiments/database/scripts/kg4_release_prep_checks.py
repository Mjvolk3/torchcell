# experiments/database/scripts/kg4_release_prep_checks.py
# [[experiments.database.scripts.kg4_release_prep_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/kg4_release_prep_checks.py
"""The two measurements KG 4.0 release preparation needs, before the build exists.

Read-only on the checkout; it opens no store, no bolt connection and no dev LMDB.

``queries``
    Every registered supported query (``torchcell/knowledge_graphs/supported_queries/
    registry.json``) is parsed with the registry's own dependency extractor and its
    labels, relationship types and ``Label.prop`` reads are checked against THIS
    checkout's graph schema (``biocypher/config/torchcell_schema_config.yaml`` through
    ``kg_manifest.graph_schema_from_yaml``), which is the schema the next build will
    serve. A query naming a label the branch retired shows up here as
    ``missing_node_label``. The drift kinds against a committed release come from
    ``supported_queries.check``, unchanged.

    ``dataset_composite`` is deliberately NOT predicted: it is
    ``composite_sha256`` over the served datasets' ``content_sha256`` values
    (``release_snapshot``), which are the swept stores' content hashes recorded by
    ``stamp`` at the end of the build. Before the 4.0 manifest is stamped there is no
    value to compute, and ``supported_queries validate <id> --release <new release>``
    is the step that records it.

``compat``
    For each package tag, the closure check ``releases.closure_compatibility`` against a
    committed release's recorded closures, with the schema surface read AS IT WAS AT
    THAT TAG (``git show <tag>:torchcell/datamodels/schema.py`` and ``pydant.py``), the
    same inputs ``scripts/kg_compat_page.py`` renders the compatibility page from. This
    answers what a given client version can read, without regenerating the page (which
    can only be regenerated once the release snapshot is committed).

Usage::

    PYTHONPATH=. python experiments/database/scripts/kg4_release_prep_checks.py queries
    PYTHONPATH=. python experiments/database/scripts/kg4_release_prep_checks.py compat \
        --tag v1.6.2 --tag v1.7.5 --worktree
    PYTHONPATH=. python experiments/database/scripts/kg4_release_prep_checks.py all --markdown
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from torchcell.knowledge_graphs.kg_manifest import (
    SURFACE_RELPATHS,
    graph_schema_from_yaml,
    surface_in_worktree,
)
from torchcell.knowledge_graphs.release_snapshot import load_closures, load_snapshots
from torchcell.knowledge_graphs.releases import closure_compatibility
from torchcell.knowledge_graphs.supported_queries.check import (
    SCHEMA_CONFIG_RELPATH,
    check_repo,
    graph_labels,
    schema_is_a,
)
from torchcell.knowledge_graphs.supported_queries.cypher_deps import (
    extract_dependencies,
)
from torchcell.knowledge_graphs.supported_queries.registry import (
    QueryRegistry,
    registry_path,
)
from torchcell.provenance.schema_deps import load_surface_from_sources

RESULT_RELPATH = "experiments/database/results/kg4_release_prep_checks.json"


def query_rows(repo_root: Path, release: str | None) -> list[dict[str, Any]]:
    """One row per registered query: does it parse, and does it hold on this schema."""
    registry = QueryRegistry.load(registry_path(repo_root))
    config_text = (repo_root / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    labels = graph_labels(graph_schema_from_yaml(config_text), schema_is_a(config_text))
    report = check_repo(repo_root, registry, release)
    by_id = {result.query_id: result for result in report.results}
    rows: list[dict[str, Any]] = []
    for query in sorted(registry.queries, key=lambda q: q.id):
        path = query.cql_file(repo_root)
        if not path.exists():
            rows.append({"query_id": query.id, "parses": False, "cql": query.cql_path})
            continue
        deps = extract_dependencies(path.read_text(encoding="utf-8"))
        missing_props = []
        for read in deps.label_properties:
            label, _, prop = read.partition(".")
            carried = labels.node_properties.get(label)
            if carried is not None and prop not in carried:
                missing_props.append(read)
        result = by_id[query.id]
        rows.append(
            {
                "query_id": query.id,
                "parses": True,
                "cql": query.cql_path,
                "status": query.status,
                "validated_release": query.validated_release,
                "recorded_composite": query.dataset_composite,
                "node_labels": sorted(deps.node_labels),
                "relationship_types": sorted(deps.relationship_types),
                "dataset_ids": sorted(deps.dataset_ids),
                "missing_node_labels": sorted(
                    label
                    for label in deps.node_labels
                    if label not in labels.node_properties
                ),
                "missing_relationship_types": sorted(
                    rel
                    for rel in deps.relationship_types
                    if rel not in labels.relationship_types
                ),
                "missing_properties": sorted(missing_props),
                "drifts_against_release": [
                    {"kind": drift.kind, "detail": drift.detail}
                    for drift in result.drifts
                ],
            }
        )
    return rows


def _surface_at_tag(repo_root: Path, tag: str) -> Any:
    sources = {}
    for relpath in SURFACE_RELPATHS:
        sources[relpath] = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{tag}:{relpath}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    return load_surface_from_sources(sources)


def compat_rows(
    repo_root: Path, tags: list[str], release: str | None, worktree: bool
) -> list[dict[str, Any]]:
    """One row per tag: how many of a release's served closures that surface reproduces."""
    snapshots = load_snapshots(repo_root)
    snapshot = (
        next(s for s in snapshots if s.release == release) if release else snapshots[-1]
    )
    closures = load_closures(repo_root, snapshot.release)
    names = sorted(snapshot.datasets)
    rows: list[dict[str, Any]] = []
    surfaces: list[tuple[str, Any]] = [
        (tag, _surface_at_tag(repo_root, tag)) for tag in tags
    ]
    if worktree:
        surfaces.append(("worktree", surface_in_worktree(repo_root)))
    for name, surface in surfaces:
        report = closure_compatibility(
            snapshot.release, snapshot.torchcell_commit, names, closures, surface
        )
        rows.append(
            {
                "surface": name,
                "release": snapshot.release,
                "release_version": snapshot.version,
                "paired_package": snapshot.torchcell_tag,
                "paired": report.paired,
                "n_served": len(names),
                "n_drifted": len(report.drifted),
                "n_unchecked": len(report.unchecked),
            }
        )
    return rows


def queries_markdown(rows: list[dict[str, Any]], release: str) -> str:
    """The supported-query table of the release note."""
    lines = [
        f"| query | parses | labels | relationship types | datasets | retired label "
        f"named | drift vs {release} |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if not row["parses"]:
            lines.append(f"| `{row['query_id']}` | no | - | - | - | - | file absent |")
            continue
        retired = (
            ", ".join(row["missing_node_labels"] + row["missing_relationship_types"])
            or "none"
        )
        kinds = sorted({drift["kind"] for drift in row["drifts_against_release"]})
        drift = (
            f"{len(row['drifts_against_release'])} x {', '.join(kinds)}"
            if kinds
            else "none"
        )
        lines.append(
            f"| `{row['query_id']}` | yes | {len(row['node_labels'])} | "
            f"{len(row['relationship_types'])} | {len(row['dataset_ids'])} | "
            f"{retired} | {drift} |"
        )
    return "\n".join(lines)


def compat_markdown(rows: list[dict[str, Any]]) -> str:
    """The client-compatibility table of the release note."""
    lines = [
        "| schema surface | verdict | drifted | unverified | of served |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        verdict = "compatible" if row["paired"] else "incompatible"
        lines.append(
            f"| {row['surface']} | {verdict} | {row['n_drifted']} | "
            f"{row['n_unchecked']} | {row['n_served']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["queries", "compat", "all"])
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--release", default=None, help="default: the newest snapshot")
    parser.add_argument(
        "--tag", action="append", default=[], help="a package tag, repeatable"
    )
    parser.add_argument(
        "--worktree",
        action="store_true",
        help="also read the working tree's schema surface",
    )
    parser.add_argument("--markdown", action="store_true", help="print the tables")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    out: dict[str, Any] = {}
    if args.command in ("queries", "all"):
        out["queries"] = query_rows(repo_root, args.release)
    if args.command in ("compat", "all"):
        tags = args.tag or ["v1.6.2", "v1.7.5"]
        out["compat"] = compat_rows(repo_root, tags, args.release, args.worktree)

    result_path = repo_root / RESULT_RELPATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        json.loads(result_path.read_text(encoding="utf-8"))
        if result_path.exists()
        else {}
    )
    existing.update(out)
    result_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULT_RELPATH}")

    if args.markdown:
        snapshots = load_snapshots(repo_root)
        release = args.release or snapshots[-1].release
        if "queries" in out:
            print()
            print(queries_markdown(out["queries"], release))
        if "compat" in out:
            print()
            print(compat_markdown(out["compat"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

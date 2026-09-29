# torchcell/knowledge_graphs/supported_queries/__main__.py
# [[torchcell.knowledge_graphs.supported_queries.__main__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/supported_queries/__main__.py
"""CLI for the supported-query registry.

    python -m torchcell.knowledge_graphs.supported_queries check [--release R] [--json]
    python -m torchcell.knowledge_graphs.supported_queries list
    python -m torchcell.knowledge_graphs.supported_queries validate <id> --release R

``--repo-root`` defaults to the checkout ``torchcell`` was imported from, so the
pre-commit hook (which exports ``PYTHONPATH`` to the worktree) checks the worktree.
``check`` exits 1 when a supported query drifts, 0 otherwise; ``validate`` exits 1 and
changes nothing when the query drifts in a way validation cannot resolve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torchcell.knowledge_graphs.releases import package_checkout
from torchcell.knowledge_graphs.supported_queries.check import (
    check_repo,
    validate_query,
)
from torchcell.knowledge_graphs.supported_queries.cypher_deps import (
    extract_dependencies,
)
from torchcell.knowledge_graphs.supported_queries.registry import (
    QueryRegistry,
    registry_path,
)


def _list(registry: QueryRegistry, repo_root: Path) -> str:
    """One line per query; a query whose ``.cql`` is absent says so instead of failing."""
    lines = []
    for query in sorted(registry.queries, key=lambda q: q.id):
        path = query.cql_file(repo_root)
        if path.exists():
            deps = extract_dependencies(path.read_text(encoding="utf-8"))
            shape = (
                f"{len(deps.dataset_ids)} datasets, {len(deps.node_labels)} labels, "
                f"{len(deps.relationship_types)} relationship types"
            )
        else:
            shape = "file absent"
        composite = (query.dataset_composite or "-")[:12]
        lines.append(
            f"{query.id}  {query.status}  since {query.since_kg_version or '-'}  "
            f"validated {query.validated_release or '-'} ({composite})  "
            f"{query.cql_path}: {shape}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="supported_queries", description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="checkout to read (default: the one torchcell was imported from)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="drift of every registered query")
    check.add_argument("--release", default=None, help="default: newest snapshot")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    sub.add_parser("list", help="registered queries")
    validate = sub.add_parser("validate", help="record the release a query holds on")
    validate.add_argument("query_id")
    validate.add_argument("--release", required=True)
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or package_checkout()).resolve()
    path = registry_path(repo_root)
    registry = QueryRegistry.load(path)
    if args.command == "list":
        print(_list(registry, repo_root))
        return 0
    if args.command == "check":
        report = check_repo(repo_root, registry, args.release)
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(report.report_markdown, end="")
        return report.exit_code
    updated, blocking = validate_query(registry, args.query_id, repo_root, args.release)
    if blocking:
        print(f"{args.query_id} does not hold on {args.release}; nothing recorded:")
        for drift in blocking:
            print(f"- {drift.kind}: {drift.detail}")
        return 1
    updated.save(path)
    query = updated.get(args.query_id)
    print(
        f"{query.id}: validated on {query.validated_release}, dataset_composite "
        f"{query.dataset_composite}, since_kg_version {query.since_kg_version}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

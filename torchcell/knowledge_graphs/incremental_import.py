# torchcell/knowledge_graphs/incremental_import.py
# [[torchcell.knowledge_graphs.incremental_import]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/incremental_import
# Test file: tests/torchcell/knowledge_graphs/test_incremental_import.py
"""Turn one BioCypher output directory into a Neo4j *incremental* import.

BioCypher only knows how to emit ``neo4j-admin database import full`` (which wipes the
store). Adding one dataset to the SERVED graph instead uses Neo4j Enterprise's
``neo4j-admin database import incremental``, which matches incoming nodes to existing
ones through a node-property uniqueness constraint and attaches new relationships to
either. Three things have to be prepared for that, all derived from the CSV files
BioCypher already wrote:

1. **Headers.** Incremental import matches an incoming node to an existing one through
   ONE uniqueness constraint per id space. BioCypher writes every node into the single
   global id space (``:ID`` with no group; edges use bare ``:START_ID``/``:END_ID``), so
   the constraint must key a label EVERY node carries: ``Entity``, the root of the
   BioLink hierarchy that BioCypher stacks onto each node's ``:LABEL`` column. The
   incremental header becomes ``id:ID{label:Entity}`` with the redundant ``id`` column
   ignored, so the imported ``id`` property is exactly the one the served graph already
   carries on every node (the content-addressed sha256). One constraint per label would
   NOT work: the importer aborts with "Multiple different indexes for group global id
   space" (measured on 5.26.28), and per-label id groups are impossible because a
   BioCypher edge file mixes endpoint labels row by row.
2. **Constraints.** ``CREATE CONSTRAINT ... FOR (n:Entity) REQUIRE n.id IS UNIQUE``. The
   served store was built by a full import and has none; it must exist BEFORE the
   incremental import runs, created through Cypher on the online database
   (``neo4j-admin database import incremental --schema`` refuses: "Applying schema
   commands during incremental import is not currently supported", measured on 5.26.28).
   The runner applies the emitted file with ``cypher-shell -f``. Global uniqueness of
   ``id`` holds in the served store by construction: the full import ran one global id
   space with ``--skip-duplicate-nodes``.
3. **Reference analysis.** Every relationship endpoint that is NOT a node of this
   increment must already exist in the served graph; a relationship whose BOTH endpoints
   are external would be re-created on top of an existing one (incremental import cannot
   dedup relationships), so it is reported as a blocker rather than silently duplicated.
4. **Existing-edge filter (against the live store).** The increment re-emits the shared
   context nodes its records use (genome, medium, temperature, an environment another
   dataset already produced) together with THEIR relationships, e.g. medium -> environment.
   Those nodes are matched to the existing ones, but the relationships would be created
   again (measured: ``MediaMemberOf`` and ``TemperatureMemberOf`` into a shared control
   environment doubled in a dress rehearsal). ``filter_existing_edges`` looks every
   relationship row up in the online store (through the ``Entity.id`` index) and rewrites
   the edge part files without the rows that already exist, keeping the originals under
   ``unfiltered/``. Run it after the constraint exists and before the database is
   stopped.

The import call mirrors the flags of the full build (tab delimiter, ``|`` array
delimiter, single-quote quoting) with the two safety flags flipped: bad relationships
ABORT the import (``--skip-bad-relationships=false --strict=true``) instead of being
dropped, because a dangling endpoint means the analysis above was wrong. Duplicate node
ids stay skipped (``--skip-duplicate-nodes=true``): a shared genome / media / temperature
node that already exists is expected and must be left as is.
"""

from __future__ import annotations

import csv
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "CsvGroup",
    "ReferenceAnalysis",
    "IncrementalImportPlan",
    "discover_csv_groups",
    "incremental_node_header",
    "write_incremental_headers",
    "constraints_cypher",
    "analyze_references",
    "incremental_import_call",
    "prepare_incremental_import",
    "INCREMENTAL_CALL_FILENAME",
    "CONSTRAINTS_FILENAME",
    "REFERENCE_ANALYSIS_FILENAME",
    "EXISTING_EDGES_FILENAME",
    "ExistingEdgeFilter",
    "query_existing_edges",
    "filter_existing_edges",
]

INCREMENTAL_CALL_FILENAME = "neo4j-admin-incremental-import-call.sh"
CONSTRAINTS_FILENAME = "incremental-constraints.cypher"
REFERENCE_ANALYSIS_FILENAME = "incremental-reference-analysis.json"
INCREMENTAL_HEADER_SUFFIX = "-header.incremental.csv"
ID_LABEL = "Entity"
"""The label whose ``id`` uniqueness constraint keys the global id space."""

_DELIMITER = "\t"
_QUOTE = "'"
_HEADER_RE = re.compile(r"^(?P<label>.+)-header\.csv$")
_PART_RE = re.compile(r"^(?P<label>.+)-part\d+\.csv$")


class CsvGroup(BaseModel):
    """One BioCypher label's files: a header plus its ``-partNNN.csv`` data files."""

    label: str  # PascalCase as BioCypher names files, e.g. "Experiment"
    kind: str  # "node" | "edge"
    header_path: str
    part_paths: list[str]
    columns: list[str]


class ReferenceAnalysis(BaseModel):
    """What this increment's relationships point at, split by where the nodes live."""

    node_counts: dict[str, int]  # label -> rows in this increment
    edge_counts: dict[str, int]  # type -> rows in this increment
    n_node_ids: int  # distinct node ids in this increment
    n_external_ids: int  # endpoint ids that are NOT nodes of this increment
    external_ids_sample: list[str] = Field(default_factory=list)
    n_edges_between_external: int  # both endpoints external -> would duplicate
    edges_between_external_sample: list[list[str]] = Field(default_factory=list)

    @property
    def has_duplicate_edge_risk(self) -> bool:
        """True when a relationship joins two nodes that both already exist."""
        return self.n_edges_between_external > 0


class IncrementalImportPlan(BaseModel):
    """Everything written into a BioCypher output directory to make it incremental."""

    out_dir: str
    database: str
    call_script: str
    constraints_file: str
    reference_analysis_file: str
    node_labels: list[str]
    edge_types: list[str]
    analysis: ReferenceAnalysis


def _read_header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").rstrip("\n").split(_DELIMITER)


def discover_csv_groups(out_dir: Path) -> list[CsvGroup]:
    """Group a BioCypher output directory's files by label, sorted by label.

    A header with a ``:ID`` column is a node file; one with ``:START_ID`` is an edge
    file. A header with no part files is an error (BioCypher never writes one without
    the other), as is a part file with no header.
    """
    headers: dict[str, Path] = {}
    parts: dict[str, list[Path]] = {}
    for path in sorted(out_dir.iterdir()):
        if path.name.endswith(INCREMENTAL_HEADER_SUFFIX):
            continue
        if (m := _HEADER_RE.match(path.name)) is not None:
            headers[m.group("label")] = path
        elif (m := _PART_RE.match(path.name)) is not None:
            parts.setdefault(m.group("label"), []).append(path)
    orphan_parts = sorted(set(parts) - set(headers))
    if orphan_parts:
        raise ValueError(f"part files without a header: {orphan_parts}")
    groups: list[CsvGroup] = []
    for label, header_path in sorted(headers.items()):
        if label not in parts:
            raise ValueError(f"header without part files: {header_path}")
        columns = _read_header(header_path)
        if ":ID" in columns or any(c.endswith(":ID") for c in columns):
            kind = "node"
        elif ":START_ID" in columns:
            kind = "edge"
        else:
            raise ValueError(f"header is neither node nor edge: {header_path}")
        groups.append(
            CsvGroup(
                label=label,
                kind=kind,
                header_path=str(header_path),
                part_paths=[str(p) for p in parts[label]],
                columns=columns,
            )
        )
    return groups


def incremental_node_header(columns: list[str], label: str) -> list[str]:
    """Rewrite a BioCypher node header for incremental import.

    ``:ID`` becomes ``id:ID{label:Entity}`` (the id is stored as property ``id`` and
    matched through the global ``Entity.id`` uniqueness constraint; ``label`` names the
    file only for error messages), and the separate ``id`` column becomes ``id:IGNORE``
    so the property is not declared twice.
    """
    if columns.count(":ID") != 1:
        raise ValueError(f"{label}: expected exactly one ':ID' column, got {columns}")
    if columns.count("id") != 1:
        raise ValueError(f"{label}: expected exactly one 'id' column, got {columns}")
    out = []
    for column in columns:
        if column == ":ID":
            out.append(f"id:ID{{label:{ID_LABEL}}}")
        elif column == "id":
            out.append("id:IGNORE")
        else:
            out.append(column)
    return out


def write_incremental_headers(groups: list[CsvGroup]) -> dict[str, Path]:
    """Write ``<Label>-header.incremental.csv`` beside every node header; return them."""
    written: dict[str, Path] = {}
    for group in groups:
        if group.kind != "node":
            continue
        header = Path(group.header_path)
        target = header.with_name(f"{group.label}{INCREMENTAL_HEADER_SUFFIX}")
        target.write_text(
            _DELIMITER.join(incremental_node_header(group.columns, group.label)) + "\n",
            encoding="utf-8",
        )
        written[group.label] = target
    return written


def _constraint_name(label: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()
    return f"{snake}_id_unique"


def constraints_cypher(id_label: str = ID_LABEL) -> str:
    """Cypher creating THE uniqueness constraint keying the global id space."""
    return (
        f"CREATE CONSTRAINT {_constraint_name(id_label)} IF NOT EXISTS "
        f"FOR (n:{id_label}) REQUIRE n.id IS UNIQUE;\n"
    )


def _assert_every_node_carries(groups: list[CsvGroup], label: str) -> None:
    """Every node row's ``:LABEL`` list must include ``label`` (the constraint's label)."""
    for group in groups:
        if group.kind != "node":
            continue
        label_index = group.columns.index(":LABEL")
        for row in _iter_rows(group.part_paths):
            labels = _unquote(row[label_index]).split("|")
            if label not in labels:
                raise ValueError(
                    f"{group.label} row {row[0]!r} lacks the {label} label ({labels}); "
                    "it could not be matched through the global id constraint"
                )


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == _QUOTE and value[-1] == _QUOTE:
        return value[1:-1]
    return value


def _iter_rows(part_paths: list[str]) -> Iterator[list[str]]:
    """Stream the rows of a label's part files (a field can be a multi-MB JSON blob)."""
    # An Experiment's serialized_data is the whole record (a pseudobulk expression
    # record is ~1 MB), far above csv's 128 KiB default field limit.
    csv.field_size_limit(sys.maxsize)
    for part in part_paths:
        with open(part, encoding="utf-8", newline="") as handle:
            reader = csv.reader(
                handle, delimiter=_DELIMITER, quotechar=_QUOTE, strict=True
            )
            yield from reader


def analyze_references(groups: list[CsvGroup], sample: int = 20) -> ReferenceAnalysis:
    """Count nodes/edges and find endpoints that lie outside this increment."""
    node_ids: set[str] = set()
    node_counts: dict[str, int] = {}
    for group in groups:
        if group.kind != "node":
            continue
        id_index = group.columns.index(":ID")
        n_rows = 0
        for row in _iter_rows(group.part_paths):
            n_rows += 1
            node_ids.add(_unquote(row[id_index]))
        node_counts[group.label] = n_rows

    edge_counts: dict[str, int] = {}
    external: set[str] = set()
    between_external: list[list[str]] = []
    n_between_external = 0
    for group in groups:
        if group.kind != "edge":
            continue
        start_index = group.columns.index(":START_ID")
        end_index = group.columns.index(":END_ID")
        n_rows = 0
        for row in _iter_rows(group.part_paths):
            n_rows += 1
            start, end = _unquote(row[start_index]), _unquote(row[end_index])
            start_ext, end_ext = start not in node_ids, end not in node_ids
            if start_ext:
                external.add(start)
            if end_ext:
                external.add(end)
            if start_ext and end_ext:
                n_between_external += 1
                if len(between_external) < sample:
                    between_external.append([group.label, start, end])
        edge_counts[group.label] = n_rows
    return ReferenceAnalysis(
        node_counts=node_counts,
        edge_counts=edge_counts,
        n_node_ids=len(node_ids),
        n_external_ids=len(external),
        external_ids_sample=sorted(external)[:sample],
        n_edges_between_external=n_between_external,
        edges_between_external_sample=between_external,
    )


def incremental_import_call(
    groups: list[CsvGroup],
    headers: dict[str, Path],
    database: str,
    *,
    bin_prefix: str = "/var/lib/neo4j/bin/",
    report_file: str | None = None,
    threads: int = 8,
    max_off_heap_memory: str = "16G",
) -> str:
    """The ``neo4j-admin database import incremental`` command for this increment.

    Paths are used verbatim: the output directory is generated inside the container at
    the same path the import runs from (``/var/lib/neo4j/biocypher-out/<ts>``). The
    uniqueness constraints are NOT passed here (``--schema`` is unsupported for
    incremental import); they are created on the online database beforehand. Threads
    and off-heap memory are bounded because the import runs INSIDE the serving
    container next to the (stopped-database, still running) Neo4j server.
    """
    # The database is the positional argument and goes FIRST: --nodes/--relationships
    # take one-or-more files, so a trailing positional is swallowed as a file
    # ("File 'torchcell' doesn't exist", measured).
    parts: list[str] = [
        f"{bin_prefix}neo4j-admin database import incremental {database}",
        "--force",
        "--verbose",
        '--delimiter="\\t"',
        '--array-delimiter="|"',
        '--quote="\'"',
        "--skip-duplicate-nodes=true",
        "--skip-bad-relationships=false",
        "--strict=true",
        f"--threads={threads}",
        f"--max-off-heap-memory={max_off_heap_memory}",
    ]
    if report_file is not None:
        parts.append(f"--report-file={report_file}")
    for group in groups:
        directory = Path(group.header_path).parent
        if group.kind == "node":
            header = headers[group.label]
            parts.append(f'--nodes="{header},{directory}/{group.label}-part.*"')
        else:
            parts.append(
                f'--relationships="{group.header_path},{directory}/{group.label}-part.*"'
            )
    return " \\\n    ".join(parts) + "\n"


def prepare_incremental_import(
    out_dir: Path, database: str, *, bin_prefix: str = "/var/lib/neo4j/bin/"
) -> IncrementalImportPlan:
    """Write headers, constraints, the call script and the analysis into ``out_dir``."""
    groups = discover_csv_groups(out_dir)
    _assert_every_node_carries(groups, ID_LABEL)
    headers = write_incremental_headers(groups)
    node_labels = [g.label for g in groups if g.kind == "node"]
    edge_types = [g.label for g in groups if g.kind == "edge"]
    constraints_file = out_dir / CONSTRAINTS_FILENAME
    constraints_file.write_text(constraints_cypher(), encoding="utf-8")
    analysis = analyze_references(groups)
    analysis_file = out_dir / REFERENCE_ANALYSIS_FILENAME
    analysis_file.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    call = incremental_import_call(
        groups,
        headers,
        database,
        bin_prefix=bin_prefix,
        report_file=str(out_dir / "incremental-import.report"),
    )
    call_script = out_dir / INCREMENTAL_CALL_FILENAME
    call_script.write_text("#!/bin/bash\nset -euo pipefail\n" + call, encoding="utf-8")
    call_script.chmod(0o755)
    return IncrementalImportPlan(
        out_dir=str(out_dir),
        database=database,
        call_script=str(call_script),
        constraints_file=str(constraints_file),
        reference_analysis_file=str(analysis_file),
        node_labels=node_labels,
        edge_types=edge_types,
        analysis=analysis,
    )


# --------------------------------------------------------------- existing-edge filter

EXISTING_EDGES_FILENAME = "incremental-existing-edges.json"
UNFILTERED_DIRNAME = "unfiltered"


class ExistingEdgeFilter(BaseModel):
    """Outcome of dropping relationship rows that the served store already holds."""

    checked: dict[str, int]  # type -> rows looked up
    existing: dict[str, int]  # type -> rows dropped because they already exist
    sample: list[list[str]] = Field(default_factory=list)  # [type, start, end]

    @property
    def n_existing(self) -> int:
        """Total relationship rows dropped."""
        return sum(self.existing.values())


def _edge_rows(group: CsvGroup) -> Iterator[tuple[str, str, str]]:
    """``(start, end, raw row)`` for every row of an edge group's part files."""
    start_index = group.columns.index(":START_ID")
    end_index = group.columns.index(":END_ID")
    for part in group.part_paths:
        with open(part, encoding="utf-8", newline="") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(_DELIMITER)
                yield _unquote(fields[start_index]), _unquote(fields[end_index]), line


def query_existing_edges(
    group: CsvGroup, rows: list[tuple[str, str]], session: Any, batch_size: int = 5000
) -> set[tuple[str, str]]:
    """Which ``(start, end)`` pairs already carry a relationship of this group's type.

    Uses the ``Entity.id`` uniqueness index on both endpoints, so a batch of a few
    thousand lookups takes well under a second on the served store.
    """
    existing: set[tuple[str, str]] = set()
    cypher = (
        "UNWIND $rows AS r "
        "MATCH (a:Entity {id: r.s})-[x]->(b:Entity {id: r.e}) "
        "WHERE type(x) = $t "
        "RETURN DISTINCT r.s AS s, r.e AS e"
    )
    for offset in range(0, len(rows), batch_size):
        batch = [{"s": s, "e": e} for s, e in rows[offset : offset + batch_size]]
        result = session.run(cypher, rows=batch, t=group.label)
        existing.update((record["s"], record["e"]) for record in result)
    return existing


def filter_existing_edges(
    out_dir: Path,
    lookup: Callable[[CsvGroup, list[tuple[str, str]]], set[tuple[str, str]]],
) -> ExistingEdgeFilter:
    """Rewrite every edge part file without the rows ``lookup`` reports as existing.

    ``lookup`` receives an edge group and its distinct ``(start, end)`` pairs and returns
    the subset that already exists in the store (``query_existing_edges`` bound to a
    session in production; any callable in tests). Originals are kept under an
    ``unfiltered/`` subdirectory, and a JSON summary is written.
    """
    groups = discover_csv_groups(out_dir)
    checked: dict[str, int] = {}
    dropped: dict[str, int] = {}
    sample: list[list[str]] = []
    for group in groups:
        if group.kind != "edge":
            continue
        pairs = sorted({(s, e) for s, e, _ in _edge_rows(group)})
        checked[group.label] = len(pairs)
        existing = lookup(group, pairs) if pairs else set()
        dropped[group.label] = 0
        if not existing:
            continue
        for part in group.part_paths:
            original = Path(part)
            # originals go to a SUBDIRECTORY: neo4j-admin globs `<Type>-part.*` as a
            # regex, so a sibling `...csv.unfiltered` would be imported as well
            backup = original.parent / UNFILTERED_DIRNAME / original.name
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                original.rename(backup)
            kept: list[str] = []
            start_index = group.columns.index(":START_ID")
            end_index = group.columns.index(":END_ID")
            with open(backup, encoding="utf-8", newline="") as handle:
                for line in handle:
                    fields = line.rstrip("\n").split(_DELIMITER)
                    pair = (_unquote(fields[start_index]), _unquote(fields[end_index]))
                    if pair in existing:
                        dropped[group.label] += 1
                        if len(sample) < 20:
                            sample.append([group.label, pair[0], pair[1]])
                        continue
                    kept.append(line)
            original.write_text("".join(kept), encoding="utf-8")
    summary = ExistingEdgeFilter(checked=checked, existing=dropped, sample=sample)
    (out_dir / EXISTING_EDGES_FILENAME).write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI: ``filter-existing-edges`` against a live store (run before STOP DATABASE)."""
    import argparse

    from neo4j import GraphDatabase

    from torchcell.database.connection import neo4j_connection_settings

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.incremental_import"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_filter = sub.add_parser(
        "filter-existing-edges",
        help="drop relationship rows the served store already holds",
    )
    p_filter.add_argument("--out-dir", required=True)
    p_filter.add_argument("--database", default="torchcell")
    p_filter.add_argument("--uri", default=None, help="bolt URI (default: NEO4J_URI)")
    args = parser.parse_args(argv)

    settings = neo4j_connection_settings()
    driver = GraphDatabase.driver(
        args.uri or settings.uri, auth=(settings.username, settings.password)
    )
    with driver.session(database=args.database) as session:
        summary = filter_existing_edges(
            Path(args.out_dir),
            lambda group, pairs: query_existing_edges(group, pairs, session),
        )
    driver.close()
    print(
        f"existing-edge filter: checked {sum(summary.checked.values())} distinct "
        f"relationships, dropped {summary.n_existing} already in the store "
        f"({', '.join(f'{k}={v}' for k, v in summary.existing.items() if v) or 'none'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

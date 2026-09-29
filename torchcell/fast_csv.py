# torchcell/fast_csv
# [[torchcell.fast_csv]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/fast_csv
# Test file: tests/torchcell/test_fast_csv.py
"""Render neo4j-admin import rows in the chunk workers; the parent only dedups and appends.

Why. BioCypher's batch writer formats every row in the build's main process: label
ancestry, PascalCase conversion, quoting, a global deduplicator, and CSV flushes, all
under one GIL, while the executor's management thread unpickles millions of
BioCypherNode objects into the same process. On the tcdb-002 ladder (r3, job 2859) the
main process sat at 1.0 core for the whole Costanzo pass while 43 workers averaged 9
cores, and the in-flight results backlog was the 166 GB peak.

What. :func:`build_row_specs` reads, from a live BioCypher instance, exactly what its
writer would use per label (schema property order and types, the ``:LABEL`` ancestry
string, delimiter, quote, array delimiter) and freezes it into picklable specs. Workers
call :func:`render_node` / :func:`render_edge` to produce the byte-identical row a
BioCypher writer would produce, and ship compact (key, line) lists. The parent's
:class:`FastCsvSink` deduplicates and appends to ``<Pascal>-part000.csv`` files in
BioCypher's own output directory, then hands the property dicts back to BioCypher so
its ``write_import_call`` writes the headers and the neo4j-admin call unchanged.

Dedup semantics match BioCypher's: node ids are unique globally; an edge is unique per
type by ``relationship_id`` or ``f"{source}_{target}"``. The sink keeps the full node id
set. For edges it keeps a full key set only where a duplicate is possible: an edge type
with an Experiment endpoint can only repeat when that experiment node itself repeated
(the experiment id is the record's content hash), so the sink tracks the set of
duplicated experiment ids (small) and checks the full key only for edges touching one.
Edge types between shared entities (perturbation to genotype, media to environment,
...) keep a full key set, which is bounded by the number of distinct shared entities
rather than by the number of records. BioCypher keeps a ``src_tgt`` string for every
edge of every type, about 95 GB for the 525M-edge full build.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher.output.write._batch_writer import parse_label
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

NUMERIC_TYPES = frozenset({"int", "integer", "long", "float", "double", "dbl"})
BOOL_TYPES = frozenset({"bool", "boolean"})
EXPERIMENT_LABEL = "experiment"

# Edge labels the adapters emit that are Biolink predicates rather than schema-config
# entries (the schema declares ``publication mentions experiment: is_a: mentions`` but
# the adapter emits ``mentions`` itself, publication -> experiment). BioCypher takes
# such an edge's properties from the first edge it sees (none here), keeps the id
# column, and labels it by its ontology ancestry (Leaves order for Neo4j). The value
# is which endpoint is the Experiment node, for the sink's dedup rule.
SCHEMA_FREE_EDGES: dict[str, str | None] = {"mentions": "target"}


class NodeRowSpec(BaseModel):
    """Everything needed to render one node label's rows without BioCypher."""

    label: str = Field(description="Sentence-case label as the adapter emits it")
    pascal: str = Field(description="PascalCase label used in file names")
    prop_types: dict[str, str] = Field(
        description="Schema property -> type, in header order, including id and preferred_id"
    )
    labels: str = Field(description="The rendered :LABEL column (ancestry, quoted)")


class EdgeRowSpec(BaseModel):
    """Everything needed to render one edge type's rows without BioCypher."""

    label: str
    pascal: str
    prop_types: dict[str, str]
    skip_id: bool = Field(description="True when the schema sets use_id false")
    labels: str = Field(description="The rendered :TYPE column")
    experiment_endpoint: str | None = Field(
        default=None,
        description="'source' or 'target' when that endpoint is an Experiment node; "
        "None when the edge joins two shared entities",
    )


class RowSpecs(BaseModel):
    """Per-label render specs plus the writer's delimiters."""

    delim: str
    quote: str
    adelim: str
    nodes: dict[str, NodeRowSpec]
    edges: dict[str, EdgeRowSpec]

    def quote_string(self, value: str) -> str:
        """BioCypher's `_quote_string`: wrap in the quote, doubling embedded quotes."""
        return f"{self.quote}{value.replace(self.quote, self.quote * 2)}{self.quote}"

    def format_value(self, value: Any, type_name: str) -> str:
        """BioCypher's per-property formatting, in the same branch order."""
        if value is None:
            return ""
        if type_name in BOOL_TYPES:
            return str(value).lower()
        if type_name in NUMERIC_TYPES:
            return str(value)
        if isinstance(value, list):
            return self.quote_string(self.adelim.join(str(x) for x in value))
        return self.quote_string(str(value))


def build_row_specs(bc: Any) -> RowSpecs:
    """Freeze the writer's per-label formatting decisions from a live BioCypher."""
    if bc._writer is None:
        bc._initialize_writer()
    writer = bc._writer
    ontology = writer.translator.ontology
    schema: dict[str, Any] = ontology.mapping.extended_schema
    nodes: dict[str, NodeRowSpec] = {}
    edges: dict[str, EdgeRowSpec] = {}
    for label, entry in schema.items():
        if not isinstance(entry, dict):
            continue
        pascal = writer.translator.name_sentence_to_pascal(parse_label(label))
        props = dict(entry.get("properties") or {})
        if writer.strict_mode:
            props.update({"source": "str", "version": "str", "licence": "str"})
        if entry.get("represented_as") == "node":
            props["id"] = "str"
            props["preferred_id"] = "str"
            nodes[label] = NodeRowSpec(
                label=label,
                pascal=pascal,
                prop_types=props,
                labels=writer._get_all_labels(label, writer.node_labels_order, False),
            )
        elif entry.get("represented_as") == "edge":
            sources = _as_list(entry.get("source"))
            targets = _as_list(entry.get("target"))
            endpoint = (
                "source"
                if EXPERIMENT_LABEL in sources
                else "target"
                if EXPERIMENT_LABEL in targets
                else None
            )
            edges[label] = EdgeRowSpec(
                label=label,
                pascal=pascal,
                prop_types=props,
                skip_id=entry.get("use_id") is False,
                labels=writer._get_all_labels(label, writer.edge_labels_order),
                experiment_endpoint=endpoint,
            )
    for label, endpoint in SCHEMA_FREE_EDGES.items():
        edges[label] = EdgeRowSpec(
            label=label,
            pascal=writer.translator.name_sentence_to_pascal(parse_label(label)),
            prop_types={},
            skip_id=False,
            labels=writer._get_all_labels(label, writer.edge_labels_order),
            experiment_endpoint=endpoint,
        )
    return RowSpecs(
        delim=writer.delim,
        quote=writer.quote,
        adelim=writer.adelim,
        nodes=nodes,
        edges=edges,
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def render_node(node: BioCypherNode, specs: RowSpecs) -> tuple[str, str, str]:
    """Return ``(pascal_label, node_id, line)`` for one node, as BioCypher would write it."""
    spec = specs.nodes.get(node.node_label)
    if spec is None:
        raise KeyError(
            f"node label {node.node_label!r} is not a node in the schema config"
        )
    props = node.properties
    if props.keys() != spec.prop_types.keys():
        raise ValueError(
            f"node {node.node_id!r} of {node.node_label!r} carries properties "
            f"{sorted(props)} but the schema declares {sorted(spec.prop_types)}"
        )
    delim = specs.delim
    plist = delim.join(
        specs.format_value(props.get(k), t) for k, t in spec.prop_types.items()
    )
    line = f"{node.node_id}{delim}{plist}{delim}{spec.labels}\n"
    return spec.pascal, node.node_id, line


def render_edge(edge: BioCypherEdge, specs: RowSpecs) -> tuple[str, str, str, str]:
    """Return ``(pascal_label, source_id, target_id, line)`` for one edge."""
    spec = specs.edges.get(edge.relationship_label)
    if spec is None:
        raise KeyError(
            f"edge label {edge.relationship_label!r} is not an edge in the schema config"
        )
    props = edge.properties
    if props.keys() != spec.prop_types.keys():
        raise ValueError(
            f"edge {edge.source_id!r}->{edge.target_id!r} of {edge.relationship_label!r} "
            f"carries properties {sorted(props)} but the schema declares "
            f"{sorted(spec.prop_types)}"
        )
    delim = specs.delim
    entries = [edge.source_id]
    if not spec.skip_id:
        entries.append(edge.relationship_id or "")
    if spec.prop_types:
        entries.append(
            delim.join(
                specs.format_value(props.get(k), t) for k, t in spec.prop_types.items()
            )
        )
    entries.append(edge.target_id)
    entries.append(spec.labels)
    return spec.pascal, edge.source_id, edge.target_id, delim.join(entries) + "\n"


class RenderedChunk:
    """One chunk's rows, deduplicated within the chunk, as compact lists.

    ``nodes[pascal] = (ids, lines)`` and ``edges[pascal] = (sources, targets, lines)``.
    Plain lists of str pickle as fast as the bytes they hold, which is the point: the
    parent unpickles rows, not object graphs.
    """

    __slots__ = ("edges", "nodes")

    def __init__(self) -> None:
        """Start empty; fill through :meth:`from_rows`."""
        self.nodes: dict[str, tuple[list[str], list[str]]] = {}
        self.edges: dict[str, tuple[list[str], list[str], list[str]]] = {}

    @classmethod
    def from_rows(cls, rows: Iterable[Any], specs: RowSpecs) -> RenderedChunk:
        """Render a chunk's BioCypher objects into one compact, chunk-deduplicated bundle."""
        chunk = cls()
        seen_nodes: set[str] = set()
        seen_edges: dict[str, set[tuple[str, str]]] = {}
        for item in rows:
            if isinstance(item, BioCypherNode):
                # Dedup before rendering: a constant node (environment, interned
                # constant) repeats on every record of a chunk, and its line is KBs.
                if item.node_id in seen_nodes:
                    continue
                pascal, node_id, line = render_node(item, specs)
                seen_nodes.add(node_id)
                bucket = chunk.nodes.setdefault(pascal, ([], []))
                bucket[0].append(node_id)
                bucket[1].append(line)
            elif isinstance(item, BioCypherEdge):
                pascal, src, tgt, line = render_edge(item, specs)
                key = (
                    (src, tgt)
                    if item.relationship_id is None
                    else (item.relationship_id, "")
                )
                seen = seen_edges.setdefault(pascal, set())
                if key in seen:
                    continue
                seen.add(key)
                bucket_e = chunk.edges.setdefault(pascal, ([], [], []))
                bucket_e[0].append(src)
                bucket_e[1].append(tgt)
                bucket_e[2].append(line)
            else:
                raise TypeError(f"cannot render {type(item).__name__}")
        return chunk


class FastCsvSink:
    """The parent's side: global dedup and append into BioCypher's output directory."""

    def __init__(self, bc: Any, specs: RowSpecs) -> None:
        """Bind to a BioCypher instance whose writer owns the output directory."""
        if bc._writer is None:
            bc._initialize_writer()
        self.bc = bc
        self.specs = specs
        self.outdir = Path(bc._writer.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Any] = {}
        self.seen_nodes: set[str] = set()
        self.dup_experiments: set[str] = set()
        self.seen_edges: dict[str, set[tuple[str, str]]] = {}
        self.node_rows: dict[str, int] = {}
        self.edge_rows: dict[str, int] = {}
        self.node_dups = 0
        self.edge_dups = 0
        self._node_pascal_to_label = {s.pascal: s.label for s in specs.nodes.values()}
        self._edge_pascal_to_label = {s.pascal: s.label for s in specs.edges.values()}
        self._edge_endpoint = {
            s.pascal: s.experiment_endpoint for s in specs.edges.values()
        }
        self._experiment_pascal = specs.nodes[EXPERIMENT_LABEL].pascal

    def _file(self, pascal: str) -> Any:
        handle = self._files.get(pascal)
        if handle is None:
            path = self.outdir / f"{pascal}-part000.csv"
            if path.exists():
                raise FileExistsError(f"{path} exists; the sink writes each label once")
            handle = path.open("w", encoding="utf-8", buffering=1 << 20)
            self._files[pascal] = handle
        return handle

    def write_nodes(self, items: Iterator[Any]) -> int:
        """Consume rendered chunks or raw BioCypherNodes; return rows written."""
        written = 0
        for item in items:
            if isinstance(item, RenderedChunk):
                for pascal, (ids, lines) in item.nodes.items():
                    written += self._add_nodes(pascal, ids, lines)
            elif isinstance(item, BioCypherNode):
                pascal, node_id, line = render_node(item, self.specs)
                written += self._add_nodes(pascal, [node_id], [line])
            else:
                raise TypeError(f"write_nodes cannot take {type(item).__name__}")
        return written

    def write_edges(self, items: Iterator[Any]) -> int:
        """Consume rendered chunks or raw BioCypherEdges; return rows written."""
        written = 0
        for item in items:
            if isinstance(item, RenderedChunk):
                for pascal, (srcs, tgts, lines) in item.edges.items():
                    written += self._add_edges(pascal, srcs, tgts, lines)
            elif isinstance(item, BioCypherEdge):
                pascal, src, tgt, line = render_edge(item, self.specs)
                written += self._add_edges(pascal, [src], [tgt], [line])
            else:
                raise TypeError(f"write_edges cannot take {type(item).__name__}")
        return written

    def _add_nodes(self, pascal: str, ids: list[str], lines: list[str]) -> int:
        seen = self.seen_nodes
        is_experiment = pascal == self._experiment_pascal
        out: list[str] = []
        for node_id, line in zip(ids, lines, strict=True):
            if node_id in seen:
                self.node_dups += 1
                if is_experiment:
                    self.dup_experiments.add(node_id)
                continue
            seen.add(node_id)
            out.append(line)
        if out:
            self._file(pascal).write("".join(out))
            self.node_rows[pascal] = self.node_rows.get(pascal, 0) + len(out)
        return len(out)

    def _add_edges(
        self, pascal: str, srcs: list[str], tgts: list[str], lines: list[str]
    ) -> int:
        endpoint = self._edge_endpoint[pascal]
        dups = self.dup_experiments
        seen = self.seen_edges.setdefault(pascal, set())
        out: list[str] = []
        for src, tgt, line in zip(srcs, tgts, lines, strict=True):
            if endpoint is None:
                key = (src, tgt)
                if key in seen:
                    self.edge_dups += 1
                    continue
                seen.add(key)
            else:
                experiment_id = src if endpoint == "source" else tgt
                if experiment_id in dups:
                    key = (src, tgt)
                    if key in seen:
                        self.edge_dups += 1
                        continue
                    seen.add(key)
            out.append(line)
        if out:
            self._file(pascal).write("".join(out))
            self.edge_rows[pascal] = self.edge_rows.get(pascal, 0) + len(out)
        return len(out)

    def finish(self) -> None:
        """Close the part files and register what was written with BioCypher.

        After this, ``bc.write_schema_info(as_node=True)`` and
        ``bc.write_import_call()`` behave as if BioCypher had written the rows itself:
        the writer's property dicts drive the header files and the import call's
        ``<Pascal>-part.*`` globs, and the deduplicator's type sets drive schema info.
        """
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        writer = self.bc._writer
        dedup = self.bc._get_deduplicator()
        for pascal in self.node_rows:
            label = self._node_pascal_to_label[pascal]
            writer.node_property_dict[label] = dict(self.specs.nodes[label].prop_types)
            dedup.entity_types.add(label)
        for pascal in self.edge_rows:
            label = self._edge_pascal_to_label[pascal]
            writer.edge_property_dict[label] = dict(self.specs.edges[label].prop_types)
            dedup.seen_relationships.setdefault(label, set())
        # BioCypher writes the header files (and registers them for the import call)
        # at the end of its own write_nodes / write_edges, which the sink bypassed.
        if not writer._write_node_headers():
            raise RuntimeError("BioCypher refused to write the node headers")
        if not writer._write_edge_headers():
            raise RuntimeError("BioCypher refused to write the edge headers")
        log.info(
            "fast csv sink: %d node rows (%d duplicates dropped), %d edge rows "
            "(%d duplicates dropped), %d duplicated experiment ids",
            sum(self.node_rows.values()),
            self.node_dups,
            sum(self.edge_rows.values()),
            self.edge_dups,
            len(self.dup_experiments),
        )

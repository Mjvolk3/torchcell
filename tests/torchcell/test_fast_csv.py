# tests/torchcell/test_fast_csv.py
# [[tests.torchcell.test_fast_csv]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/test_fast_csv.py
r'''Tests for ``torchcell.fast_csv`` against a fake BioCypher writer.

``FakeBioCypher`` exposes only what ``build_row_specs`` and ``FastCsvSink`` read: a lazily
initialized ``_writer`` (translator + extended schema, ``strict_mode``, delimiters, label
orders, ``_get_all_labels``, ``outdir``, the property dicts and the two header writers,
which follow BioCypher 0.15.2's ``_Neo4jBatchWriter`` format) and ``_get_deduplicator``.
The writer uses the production delimiters (``biocypher/config/*_config.yaml``): tab
between columns, ``|`` inside arrays, ``"`` as the quote.

Derived rows (``\t`` is a tab). Node columns are the id, then the schema properties in
schema order followed by ``id`` and ``preferred_id`` (``BioCypherNode`` sets
``preferred_id`` to ``"id"``), then the quoted ``Ascending`` ancestry:

- ``exp1`` with ``dataset_name='Kuzmin "2018"'``: the embedded quotes double, the float
  prints bare: ``exp1\t"Kuzmin ""2018"""\t0.5\t"exp1"\t"id"\t"Experiment|NamedThing"``.
- ``gp1``: the ``str[]`` list joins on ``|`` inside one quote, ``True`` lowers to
  ``true``, ``None`` is an empty field:
  ``gp1\t"YAL001C"\t"TFC3|FUN24"\ttrue\t\t"gp1"\t"id"\t"GenePerturbation|Perturbation|NamedThing"``.

Edge columns are the source, the relationship id unless ``use_id: false``, the properties
(only when the type has any), the target, then the ``Leaves`` label (first ancestor):
``gp1\t1.5\tgt1\t"PerturbationMemberOf"`` (``use_id: false``),
``ref1\tr1\texp1\t"ExperimentReferenceOf"`` and, with no relationship id,
``ref2\t\texp1\t"ExperimentReferenceOf"``.

Sink counts: nodes ``[chunk(exp1, gp1), chunk(exp1, exp2), gp1]`` write 2 + 1 + 0 = 3
rows with 2 duplicates, and only ``exp1`` is a duplicated Experiment. Edges write 5 rows
with 3 duplicates (see ``test_sink_counts_files_and_headers``).
'''

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._translate import Translator

from torchcell.fast_csv import (
    EdgeRowSpec,
    FastCsvSink,
    NodeRowSpec,
    RenderedChunk,
    RowSpecs,
    build_row_specs,
)

SCHEMA: dict[str, Any] = {
    "schema_version": "1.0",
    "phenotype": {"is_a": "named thing"},
    "experiment": {
        "represented_as": "node",
        "properties": {"dataset_name": "str", "score": "float"},
    },
    "gene perturbation": {
        "represented_as": "node",
        "properties": {
            "systematic_gene_name": "str",
            "aliases": "str[]",
            "essential": "bool",
            "count": "int",
        },
    },
    "perturbation member of": {
        "represented_as": "edge",
        "source": "gene perturbation",
        "target": "genotype",
        "use_id": False,
        "properties": {"weight": "float"},
    },
    "experiment reference of": {
        "represented_as": "edge",
        "source": "experiment reference",
        "target": ["experiment"],
    },
    "experiment measures phenotype": {
        "represented_as": "edge",
        "source": "experiment",
        "target": "phenotype",
    },
}
ANCESTRY: dict[str, list[str]] = {
    "experiment": ["experiment", "named thing"],
    "gene perturbation": ["gene perturbation", "perturbation", "named thing"],
    "perturbation member of": ["perturbation member of", "related to"],
    "experiment reference of": ["experiment reference of", "related to"],
    "experiment measures phenotype": ["experiment measures phenotype", "related to"],
    "mentions": ["mentions", "related to at instance level", "related to"],
}
HEADER_TYPES = {
    "float": "double",
    "int": "long",
    "bool": "boolean",
    "str[]": "string[]",
}

EXP1 = 'exp1\t"Kuzmin ""2018"""\t0.5\t"exp1"\t"id"\t"Experiment|NamedThing"\n'
EXP2 = 'exp2\t"Costanzo"\t-1.25\t"exp2"\t"id"\t"Experiment|NamedThing"\n'
GP1 = 'gp1\t"YAL001C"\t"TFC3|FUN24"\ttrue\t\t"gp1"\t"id"\t"GenePerturbation|Perturbation|NamedThing"\n'
MEMBER = 'gp1\t1.5\tgt1\t"PerturbationMemberOf"\n'
REF_R1 = 'ref1\tr1\texp1\t"ExperimentReferenceOf"\n'
REF2 = 'ref2\t\texp1\t"ExperimentReferenceOf"\n'
REF_EXP2 = 'ref1\t\texp2\t"ExperimentReferenceOf"\n'
MEASURES = 'exp1\t\tph1\t"ExperimentMeasuresPhenotype"\n'
MENTIONS = 'pub1\t\texp1\t"Mentions"\n'


class FakeTranslator:
    """``translator.ontology.mapping.extended_schema`` plus BioCypher's own PascalCase."""

    def __init__(self) -> None:  # noqa: D107
        self.ontology = SimpleNamespace(mapping=SimpleNamespace(extended_schema=SCHEMA))

    @staticmethod
    def name_sentence_to_pascal(name: str) -> str:  # noqa: D102
        return str(Translator.name_sentence_to_pascal(name))


class FakeWriter:
    """The writer attributes ``fast_csv`` reads, with BioCypher's header format."""

    def __init__(self, outdir: Path, strict_mode: bool) -> None:  # noqa: D107
        self.translator = FakeTranslator()
        self.strict_mode = strict_mode
        self.delim, self.quote, self.adelim = "\t", '"', "|"
        self.node_labels_order, self.edge_labels_order = "Ascending", "Leaves"
        self.outdir = str(outdir)
        self.node_property_dict: dict[str, dict[str, str]] = {}
        self.edge_property_dict: dict[str, dict[str, str]] = {}

    def _get_all_labels(self, label: str, order: str, force: bool = False) -> str:
        names = [Translator.name_sentence_to_pascal(a) for a in ANCESTRY[label]]
        kept = names[:1] if order == "Leaves" else names
        return f'"{"|".join(kept)}"'

    def _header(self, label: str, cols: list[str]) -> None:
        pascal = Translator.name_sentence_to_pascal(label)
        Path(self.outdir, f"{pascal}-header.csv").write_text("\t".join(cols))

    @staticmethod
    def _props(props: dict[str, str]) -> list[str]:
        return [
            f"{k}:{HEADER_TYPES[t]}" if t in HEADER_TYPES else k
            for k, t in props.items()
        ]

    def _write_node_headers(self) -> bool:
        for label, props in self.node_property_dict.items():
            self._header(label, [":ID", *self._props(props), ":LABEL"])
        return bool(self.node_property_dict)

    def _write_edge_headers(self) -> bool:
        for label, props in self.edge_property_dict.items():
            skip_id = SCHEMA.get(label, {}).get("use_id") is False
            id_col = [] if skip_id else ["id"]
            self._header(
                label, [":START_ID", *id_col, *self._props(props), ":END_ID", ":TYPE"]
            )
        return bool(self.edge_property_dict)


class FakeDeduplicator:
    """The two sets ``FastCsvSink.finish`` registers labels in."""

    def __init__(self) -> None:  # noqa: D107
        self.entity_types: set[str] = set()
        self.seen_relationships: dict[str, set[str]] = {}


class FakeBioCypher:
    """A writer created on demand (counted) and a deduplicator."""

    def __init__(self, outdir: Path, strict_mode: bool = False) -> None:  # noqa: D107
        self._writer: FakeWriter | None = None
        self.outdir, self.strict_mode = outdir, strict_mode
        self.initialized = 0
        self.dedup = FakeDeduplicator()

    def _initialize_writer(self) -> None:
        self.initialized += 1
        self._writer = FakeWriter(self.outdir, self.strict_mode)

    def _get_deduplicator(self) -> FakeDeduplicator:
        return self.dedup


def exp1(**props: Any) -> Any:
    return BioCypherNode(
        "exp1",
        "experiment",
        properties=props or {"dataset_name": 'Kuzmin "2018"', "score": 0.5},
    )


def exp2() -> Any:
    return BioCypherNode(
        "exp2", "experiment", properties={"dataset_name": "Costanzo", "score": -1.25}
    )


def gp1() -> Any:
    props = {
        "systematic_gene_name": "YAL001C",
        "aliases": ["TFC3", "FUN24"],
        "essential": True,
        "count": None,
    }
    return BioCypherNode("gp1", "gene perturbation", properties=props)


def edge(src: str, tgt: str, label: str, rid: str | None = None, **props: Any) -> Any:
    return BioCypherEdge(src, tgt, label, relationship_id=rid, properties=props)


@pytest.fixture
def bc(tmp_path: Path) -> FakeBioCypher:
    return FakeBioCypher(tmp_path / "out" / "nested")


def test_build_row_specs_reads_the_writer_layout(bc: FakeBioCypher) -> None:
    specs = build_row_specs(bc)
    assert bc.initialized == 1
    exp_props = {
        "dataset_name": "str",
        "score": "float",
        "id": "str",
        "preferred_id": "str",
    }
    gp_props = {"systematic_gene_name": "str", "aliases": "str[]", "essential": "bool"}
    gp_props |= {"count": "int", "id": "str", "preferred_id": "str"}

    def node(
        label: str, pascal: str, props: dict[str, str], labels: str
    ) -> NodeRowSpec:
        return NodeRowSpec(label=label, pascal=pascal, prop_types=props, labels=labels)

    def edge_spec(
        label: str, pascal: str, props: dict[str, str], skip_id: bool, end: str | None
    ) -> EdgeRowSpec:
        return EdgeRowSpec(
            label=label,
            pascal=pascal,
            prop_types=props,
            skip_id=skip_id,
            labels=f'"{pascal}"',
            experiment_endpoint=end,
        )

    assert specs == RowSpecs(
        delim="\t",
        quote='"',
        adelim="|",
        nodes={
            "experiment": node(
                "experiment", "Experiment", exp_props, '"Experiment|NamedThing"'
            ),
            "gene perturbation": node(
                "gene perturbation",
                "GenePerturbation",
                gp_props,
                '"GenePerturbation|Perturbation|NamedThing"',
            ),
        },
        edges={
            "perturbation member of": edge_spec(
                "perturbation member of",
                "PerturbationMemberOf",
                {"weight": "float"},
                True,
                None,
            ),
            "experiment reference of": edge_spec(
                "experiment reference of", "ExperimentReferenceOf", {}, False, "target"
            ),
            "experiment measures phenotype": edge_spec(
                "experiment measures phenotype",
                "ExperimentMeasuresPhenotype",
                {},
                False,
                "source",
            ),
            "mentions": edge_spec("mentions", "Mentions", {}, False, "target"),
        },
    )
    # Pydantic equality ignores dict order; the header column order is load-bearing.
    assert list(specs.nodes["experiment"].prop_types) == list(exp_props)
    assert list(specs.nodes["gene perturbation"].prop_types) == list(gp_props)
    build_row_specs(bc)
    assert bc.initialized == 1


def test_build_row_specs_strict_mode_adds_provenance_columns(tmp_path: Path) -> None:
    specs = build_row_specs(FakeBioCypher(tmp_path, strict_mode=True))
    strict = ["source", "version", "licence"]
    assert list(specs.nodes["experiment"].prop_types) == [
        "dataset_name",
        "score",
        *strict,
        "id",
        "preferred_id",
    ]
    assert list(specs.edges["perturbation member of"].prop_types) == ["weight", *strict]
    assert specs.edges["mentions"].prop_types == {}


def test_chunk_renders_exact_lines_and_dedups_before_rendering(
    bc: FakeBioCypher,
) -> None:
    specs = build_row_specs(bc)
    rows = [
        exp1(),
        gp1(),
        # Same id with a property set the schema rejects: rendering it would raise, so
        # this passing proves the seen-id check runs before render_node.
        exp1(unexpected="x"),
        edge("gp1", "gt1", "perturbation member of", weight=1.5),
        edge("gp1", "gt1", "perturbation member of", weight=2.0),
        edge("ref1", "exp1", "experiment reference of", rid="r1"),
        edge("ref2", "exp1", "experiment reference of", rid="r1"),
        edge("ref2", "exp1", "experiment reference of"),
        edge("pub1", "exp1", "mentions"),
    ]
    chunk = RenderedChunk.from_rows(rows, specs)
    assert chunk.nodes == {
        "Experiment": (["exp1"], [EXP1]),
        "GenePerturbation": (["gp1"], [GP1]),
    }
    assert chunk.edges == {
        "PerturbationMemberOf": (["gp1"], ["gt1"], [MEMBER]),
        "ExperimentReferenceOf": (["ref1", "ref2"], ["exp1", "exp1"], [REF_R1, REF2]),
        "Mentions": (["pub1"], ["exp1"], [MENTIONS]),
    }


def test_render_errors(bc: FakeBioCypher) -> None:
    specs = build_row_specs(bc)
    with pytest.raises(
        KeyError,
        match=re.escape("node label 'genotype' is not a node in the schema config"),
    ):
        RenderedChunk.from_rows([BioCypherNode("g1", "genotype")], specs)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "node 'exp1' of 'experiment' carries properties ['id', 'preferred_id', 'unexpected'] "
            "but the schema declares ['dataset_name', 'id', 'preferred_id', 'score']"
        ),
    ):
        RenderedChunk.from_rows([exp1(unexpected="x")], specs)
    with pytest.raises(
        KeyError,
        match=re.escape("edge label 'binds' is not an edge in the schema config"),
    ):
        RenderedChunk.from_rows([edge("a", "b", "binds")], specs)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "edge 'gp1'->'gt1' of 'perturbation member of' carries properties [] "
            "but the schema declares ['weight']"
        ),
    ):
        RenderedChunk.from_rows([edge("gp1", "gt1", "perturbation member of")], specs)
    with pytest.raises(TypeError, match="^cannot render str$"):
        RenderedChunk.from_rows(["exp1"], specs)


def test_sink_counts_files_and_headers(bc: FakeBioCypher) -> None:
    specs = build_row_specs(bc)
    sink = FastCsvSink(bc, specs)
    outdir = bc.outdir
    assert outdir.is_dir()

    chunks = [
        RenderedChunk.from_rows([exp1(), gp1()], specs),
        RenderedChunk.from_rows([exp1(), exp2()], specs),
    ]
    assert sink.write_nodes(iter([*chunks, gp1()])) == 3
    assert (sink.node_dups, sink.dup_experiments) == (2, {"exp1"})
    assert sink.node_rows == {"Experiment": 2, "GenePerturbation": 1}

    edge_chunk = RenderedChunk.from_rows(
        [
            edge("gp1", "gt1", "perturbation member of", weight=1.5),
            edge("ref1", "exp1", "experiment reference of", rid="r1"),
            edge("ref1", "exp2", "experiment reference of"),
        ],
        specs,
    )
    raw = [
        edge(
            "gp1", "gt1", "perturbation member of", weight=1.5
        ),  # shared pair: dropped
        edge(
            "ref1", "exp1", "experiment reference of", rid="r1"
        ),  # exp1 repeated: dropped
        edge("ref1", "exp2", "experiment reference of"),  # exp2 never repeated: kept
        edge("exp1", "ph1", "experiment measures phenotype"),  # first: kept
        edge(
            "exp1", "ph1", "experiment measures phenotype"
        ),  # source exp1 repeated: dropped
    ]
    assert sink.write_edges(iter([edge_chunk, *raw])) == 5
    assert sink.edge_dups == 3
    assert sink.edge_rows == {
        "PerturbationMemberOf": 1,
        "ExperimentReferenceOf": 3,
        "ExperimentMeasuresPhenotype": 1,
    }

    sink.finish()
    files = {p.name: p.read_text() for p in outdir.iterdir()}
    assert files == {
        "Experiment-part000.csv": EXP1 + EXP2,
        "GenePerturbation-part000.csv": GP1,
        "PerturbationMemberOf-part000.csv": MEMBER,
        "ExperimentReferenceOf-part000.csv": REF_R1 + REF_EXP2 + REF_EXP2,
        "ExperimentMeasuresPhenotype-part000.csv": MEASURES,
        "Experiment-header.csv": ":ID\tdataset_name\tscore:double\tid\tpreferred_id\t:LABEL",
        "GenePerturbation-header.csv": (
            ":ID\tsystematic_gene_name\taliases:string[]\tessential:boolean\tcount:long\tid\tpreferred_id\t:LABEL"
        ),
        "PerturbationMemberOf-header.csv": ":START_ID\tweight:double\t:END_ID\t:TYPE",
        "ExperimentReferenceOf-header.csv": ":START_ID\tid\t:END_ID\t:TYPE",
        "ExperimentMeasuresPhenotype-header.csv": ":START_ID\tid\t:END_ID\t:TYPE",
    }
    assert sink._files == {}
    assert bc._writer is not None
    assert list(bc._writer.node_property_dict) == ["experiment", "gene perturbation"]
    assert bc.dedup.entity_types == {"experiment", "gene perturbation"}
    assert bc.dedup.seen_relationships == {
        "perturbation member of": set(),
        "experiment reference of": set(),
        "experiment measures phenotype": set(),
    }


def test_sink_errors(bc: FakeBioCypher) -> None:
    specs = build_row_specs(bc)
    sink = FastCsvSink(bc, specs)
    with pytest.raises(TypeError, match="^write_nodes cannot take str$"):
        sink.write_nodes(iter(["exp1"]))
    with pytest.raises(TypeError, match="^write_edges cannot take int$"):
        sink.write_edges(iter([1]))
    with pytest.raises(
        RuntimeError, match="^BioCypher refused to write the node headers$"
    ):
        sink.finish()

    sink = FastCsvSink(bc, specs)
    assert sink.write_nodes(iter([gp1()])) == 1
    with pytest.raises(
        RuntimeError, match="^BioCypher refused to write the edge headers$"
    ):
        sink.finish()

    blocked = bc.outdir / "Experiment-part000.csv"
    blocked.write_text("")
    with pytest.raises(
        FileExistsError,
        match=re.escape(f"{blocked} exists; the sink writes each label once"),
    ):
        FastCsvSink(bc, specs).write_nodes(iter([exp1()]))

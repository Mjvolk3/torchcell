# tests/torchcell/datamodels/test_ontology_checks.py
# [[tests.torchcell.datamodels.test_ontology_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_ontology_checks.py
"""Each ontology check's verdict and message on a hand-built tree, pass and failure.

``test_ontology_coherence.py`` runs the checks against the live schema, where every
check passes, so the failure branches never execute there. Here the checks run on
inputs built to fail in one known way each (2026.09.30, Phase 16).

The schema-walking checks read the module global ``s``; the ``fake_schema`` fixture
swaps it for ``tc_fake_schema``, a module exec'd from ``_FAKE_SCHEMA`` whose models are
plain ``BaseModel`` classes (never subclasses of a real schema class, so the live
ontology tree walked by ``test_ontology_all_trees`` is untouched). Its shape and the
expected verdicts:

- ``Experiment(ExperimentBase)`` composes ``Genotype`` (inherited field),
  ``Environment`` and ``Phenotype``; ``ExperimentReference`` composes ``Environment``.
  ``Environment`` holds ``Temperature | None`` and ``Optional[str]``.
- Phenotype leaves: ``LeftPhenotype`` (``left``), ``RightPhenotype`` (``right``), the
  diamond ``BothPhenotype(LeftPhenotype, RightPhenotype)`` (``both``) and
  ``LoopPhenotype`` (``back: ExperimentReference | None``, ``evidence:
  list[SourcedValue]``).
- ``Orphan`` (``value: SourcedValue | None``) and ``SegregantGenotype`` are composed by
  nothing an experiment reaches, so ``orphan_models`` is ``["Orphan",
  "SegregantGenotype"]``.
- ``LoopPhenotype -> ExperimentReference`` is the one back edge, so the phenotype lane
  also absorbs ``ExperimentReference``, ``Environment`` and ``Temperature``.
- ``Aliased`` has ``first = second = "x"`` and ``third = "y"``: one collision
  ``["first", "second"]``; ``Distinct`` has none.

The adapter checks read source text, the media checks a patched ``MEDIA_LIBRARY`` of
real ``Media`` records, the collision check a real ``Environment``, and the census
plain mappings; each test's docstring gives its expected values.
"""

import sys
import types
from typing import Any

import pytest

from torchcell.datamodels import ontology_checks as oc
from torchcell.datamodels import schema as s
from torchcell.knowledge_graphs.kg_manifest import GraphSchemaEntry
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

_FAKE_SCHEMA = """
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class Aliased(str, Enum):
    first = "x"
    second = "x"
    third = "y"


class Distinct(str, Enum):
    one = "1"
    two = "2"


class Genotype(BaseModel):
    name: str


class SegregantGenotype(BaseModel):
    parent: str


class Temperature(BaseModel):
    value: float


class Environment(BaseModel):
    temperature: Temperature | None = None
    note: Optional[str] = None


class Phenotype(BaseModel):
    label: str


class LeftPhenotype(Phenotype):
    left: float


class RightPhenotype(Phenotype):
    right: float


class BothPhenotype(LeftPhenotype, RightPhenotype):
    both: float


class ExperimentReference(BaseModel):
    environment: Environment


class LoopPhenotype(Phenotype):
    back: ExperimentReference | None = None
    evidence: list[SourcedValue] = []


class ExperimentBase(BaseModel):
    genotype: Genotype


class Experiment(ExperimentBase):
    environment: Environment
    phenotype: Phenotype


class Orphan(BaseModel):
    value: SourcedValue | None = None


PhenotypeType = Union[
    Phenotype, LeftPhenotype, RightPhenotype, BothPhenotype, LoopPhenotype
]
"""


@pytest.fixture
def fake_schema(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """``oc.s`` replaced by the exec'd fake module for the duration of one test."""
    module = types.ModuleType("tc_fake_schema")
    module.__dict__["SourcedValue"] = SourcedValue
    monkeypatch.setitem(sys.modules, "tc_fake_schema", module)
    exec(compile(_FAKE_SCHEMA, "tc_fake_schema", "exec"), module.__dict__)
    monkeypatch.setattr(oc, "s", module)
    return module


def _names(models: Any) -> list[str]:
    return sorted(m.__name__ for m in models)


def test_schema_models_and_enums_are_those_defined_in_the_module(
    fake_schema: types.ModuleType,
) -> None:
    """Imported names (``BaseModel``, ``Enum``, ``SourcedValue``) are excluded."""
    assert _names(oc.schema_models()) == [
        "BothPhenotype",
        "Environment",
        "Experiment",
        "ExperimentBase",
        "ExperimentReference",
        "Genotype",
        "LeftPhenotype",
        "LoopPhenotype",
        "Orphan",
        "Phenotype",
        "RightPhenotype",
        "SegregantGenotype",
        "Temperature",
    ]
    assert _names(oc.schema_enums()) == ["Aliased", "Distinct"]


def test_experiment_closure_and_orphans(fake_schema: types.ModuleType) -> None:
    """Composition, bases (``ExperimentBase``) and subclasses (the phenotype leaves) are
    reached; ``Orphan`` and ``SegregantGenotype`` are not.
    """
    assert _names(oc.experiment_closure()) == [
        "BothPhenotype",
        "Environment",
        "Experiment",
        "ExperimentBase",
        "ExperimentReference",
        "Genotype",
        "LeftPhenotype",
        "LoopPhenotype",
        "Phenotype",
        "RightPhenotype",
        "Temperature",
    ]
    assert oc.orphan_models() == ["Orphan", "SegregantGenotype"]
    assert _names(oc.composition_targets(fake_schema.Experiment)) == [
        "Environment",
        "Genotype",
        "Phenotype",
    ]


def test_a_back_edge_is_reported_and_pulls_the_environment_into_the_phenotype_lane(
    fake_schema: types.ModuleType,
) -> None:
    """``phenotype: LoopPhenotype -> ExperimentReference`` is the one back edge; through
    it the phenotype lane contains ``Environment`` and ``Temperature``, so it overlaps
    the environment lane. The diamond ``BothPhenotype`` is listed once.
    """
    lanes = {k: _names(v) for k, v in oc.lane_membership().items()}
    assert lanes == {
        "genotype": ["Genotype", "SegregantGenotype"],
        "environment": ["Environment", "Temperature"],
        "phenotype": [
            "BothPhenotype",
            "Environment",
            "ExperimentReference",
            "LeftPhenotype",
            "LoopPhenotype",
            "Phenotype",
            "RightPhenotype",
            "Temperature",
        ],
    }
    assert oc.lane_back_edges() == ["phenotype: LoopPhenotype -> ExperimentReference"]


def test_sourced_value_fields_and_enum_collisions(
    fake_schema: types.ModuleType,
) -> None:
    """Both ``SourcedValue`` carriers with their rendered annotations; ``Aliased``'s
    two names for ``"x"`` collide, ``third`` and ``Distinct`` do not.
    """
    assert oc.sourced_value_fields() == {
        "LoopPhenotype.evidence": "list[torchcell.verification.sourced.SourcedValue]",
        "Orphan.value": "torchcell.verification.sourced.SourcedValue | None",
    }
    assert oc.enum_value_collisions() == {"Aliased": ["first", "second"]}


def _entry(kind: str, **kwargs: list[str]) -> GraphSchemaEntry:
    return GraphSchemaEntry.model_validate({"kind": kind, **kwargs})


def test_phenotype_label_map_matches_ambiguous_unmatched_and_unmapped(
    fake_schema: types.ModuleType,
) -> None:
    """``both phenotype`` ({both}) matches BothPhenotype alone; ``left phenotype``
    ({left}) fits LeftPhenotype and its subclass BothPhenotype; ``ghost phenotype`` fits
    nothing; ``envelope phenotype`` carries only envelope properties and so has no
    discriminating set; a non-phenotype node and an edge are ignored. Left, Right and
    Loop stay unclaimed, so the map is not bijective.
    """
    schema = {
        "both phenotype": _entry("node", properties=["both", "graph_level"]),
        "left phenotype": _entry("node", properties=["label_name", "left"]),
        "ghost phenotype": _entry("node", properties=["nothing"]),
        "envelope phenotype": _entry(
            "node", properties=["graph_level", "serialized_data"]
        ),
        "gene": _entry("node", properties=["left"]),
        "edge phenotype": _entry("edge", source=["gene"], target=["gene"]),
    }
    result = oc.phenotype_label_map(schema)
    assert result.matched == {"both phenotype": "BothPhenotype"}
    assert result.ambiguous == {"left phenotype": ["BothPhenotype", "LeftPhenotype"]}
    assert result.unmatched_labels == ["envelope phenotype", "ghost phenotype"]
    assert result.unmapped_classes == [
        "LeftPhenotype",
        "LoopPhenotype",
        "RightPhenotype",
    ]
    assert result.is_bijective is False


def test_is_bijective_only_without_ambiguity_or_leftovers() -> None:
    """An all-matched map is bijective; any one leftover list makes it not."""
    assert oc.PhenotypeLabelMap(matched={"a phenotype": "A"}).is_bijective is True
    assert oc.PhenotypeLabelMap(unmapped_classes=["A"]).is_bijective is False
    assert oc.PhenotypeLabelMap(unmatched_labels=["a phenotype"]).is_bijective is False
    assert oc.PhenotypeLabelMap(ambiguous={"a": ["A", "B"]}).is_bijective is False


def test_isolated_graph_node_classes() -> None:
    """``c`` is on no edge; ``a`` (source) and ``b`` (target) are connected."""
    schema = {
        "a": _entry("node"),
        "b": _entry("node"),
        "c": _entry("node"),
        "a to b": _entry("edge", source=["a"], target=["b"]),
    }
    assert oc.isolated_graph_node_classes(schema) == ["c"]


_ADAPTER = """
def build_props():
    return {"z": 1, "a": 2}


class CellAdapter:
    def literal(self):
        yield BioCypherNode(node_id="1", node_label="gene", properties={"b": 1, "a": 2})

    def local(self):
        props = {"y": 1, "x": 2}
        yield BioCypherNode(node_label="media", properties=props)

    def parameter(self, props):
        yield BioCypherNode(node_label="param", properties=props)

    def helper(self):
        yield BioCypherNode(node_label="helper", properties=self.build_props())

    def missing_helper(self):
        yield BioCypherNode(node_label="gone", properties=self.nothing())

    def computed_key(self, k):
        yield BioCypherNode(node_label="computed", properties={k: 1, "a": 2})

    def dynamic_label(self, label):
        yield BioCypherNode(node_label=label, properties={"a": 1})

    def bare(self):
        yield BioCypherNode(node_label="bare")

    def called(self):
        yield BioCypherNode(node_label="called", properties=dict(a=1))

    def edge(self):
        yield BioCypherEdge(source_id="1", target_id="2")
"""


def _sites() -> dict[str, oc.AdapterNodeSite]:
    return {site.function: site for site in oc.adapter_node_sites(_ADAPTER)}


def test_adapter_node_sites_read_each_properties_form() -> None:
    """A dict literal and a local dict give sorted keys; a helper call resolves through
    the module-wide table of dict-returning functions; a parameter, an unknown helper
    and a ``dict(...)`` call are unreadable (None); no ``properties`` gives ``[]``; a
    non-literal label is None; the edge is not a node site.
    """
    sites = oc.adapter_node_sites(_ADAPTER)
    assert [site.lineno for site in sites] == sorted(site.lineno for site in sites)
    by_function = {
        site.function: (site.node_label, site.property_keys) for site in sites
    }
    assert by_function == {
        "literal": ("gene", ["a", "b"]),
        "local": ("media", ["x", "y"]),
        "parameter": ("param", None),
        "helper": ("helper", ["a", "z"]),
        "missing_helper": ("gone", None),
        "computed_key": ("computed", []),
        "dynamic_label": (None, ["a"]),
        "bare": ("bare", []),
        "called": ("called", None),
    }
    assert _sites()["literal"].lineno == 8


def test_a_computed_key_reads_as_no_properties_and_yields_a_false_mismatch() -> None:
    """Finding: ``_dict_keys`` returns ``[]`` for a dict with a non-literal key
    (``ontology_checks.py:503-504``), not None, so the site is read as emitting nothing
    instead of being skipped as unreadable, and every declared property of its class is
    reported as ``declared_not_emitted``. Pinned until the reader returns None there.
    """
    site = _sites()["computed_key"]
    mismatches = oc.adapter_property_mismatches(
        [site], {"computed": _entry("node", properties=["a", "k"])}
    )
    assert [m.model_dump() for m in mismatches] == [
        {
            "node_label": "computed",
            "function": "computed_key",
            "lineno": site.lineno,
            "emitted_not_declared": [],
            "declared_not_emitted": ["a", "k"],
        }
    ]


def test_adapter_property_mismatches_both_ways_undeclared_and_skipped() -> None:
    """``gene`` emits {a, b} against declared {a, c}; ``media`` is not declared at all;
    ``helper`` matches exactly; the dynamic label and the unreadable sites are skipped.
    """
    sites = _sites()
    schema = {
        "gene": _entry("node", properties=["a", "c"]),
        "helper": _entry("node", properties=["a", "z"]),
        "param": _entry("node", properties=["q"]),
    }
    mismatches = oc.adapter_property_mismatches(
        [
            sites["literal"],
            sites["local"],
            sites["helper"],
            sites["dynamic_label"],
            sites["parameter"],
        ],
        schema,
    )
    assert [
        (m.node_label, m.emitted_not_declared, m.declared_not_emitted)
        for m in mismatches
    ] == [("gene", ["b"], ["c"]), ("media", ["x", "y"], [])]


_TRAVERSALS = """
def emit(data, other):
    a = data['experiment'].environment.temperature.value
    b = data['experiment'].environment.temperature
    c = data['experiment'].environment.unknown.value
    d = data['experiment'].phenotype.label.upper
    e = data.reference.environment.note.strip
    f = other.environment.temperature.value
    g = data.reference
    h = data['experiment'].environment.temperature.value
"""


def test_adapter_optional_traversals_flag_only_walks_through_a_nullable_hop(
    fake_schema: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Line 3 walks through ``Environment.temperature`` (``X | None``) and line 7
    through ``Environment.note`` (``Optional[str]``); stopping AT the optional field
    (line 4), an unknown field (line 5), a non-model hop (line 6), a foreign root
    (line 8) and a bare root (line 9) are not flagged; the repeat on line 10 is
    deduplicated to line 3.
    """
    monkeypatch.setattr(
        oc,
        "ADAPTER_ROOTS",
        {
            "data['experiment']": fake_schema.Experiment,
            "data.reference": fake_schema.ExperimentReference,
        },
    )
    found = oc.adapter_optional_traversals(_TRAVERSALS)
    assert [t.model_dump() for t in found] == [
        {
            "lineno": 3,
            "chain": "data['experiment'].environment.temperature.value",
            "optional_hop": "Environment.temperature",
        },
        {
            "lineno": 7,
            "chain": "data.reference.environment.note.strip",
            "optional_hop": "Environment.note",
        },
    ]


_GLUCOSE_KEY = "WQZGKKKJIJFFOK-GASJEMHNSA-N"


def _component(
    name: str,
    role: s.MediaComponentRole = s.MediaComponentRole.other,
    definition: s.ComponentDefinition = s.ComponentDefinition.defined,
    **identity: Any,
) -> s.MediaComponent:
    return s.MediaComponent(
        compound=s.Compound(name=name, **identity), role=role, definition=definition
    )


def _gapped(name: str) -> s.Compound:
    return s.Compound(
        name=name,
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey", reason=ProvenanceGapReason.not_carried_by_curation
            )
        ],
    )


def _library() -> dict[str, s.Media]:
    base = s.Media(
        name="A",
        state="liquid",
        is_synthetic=True,
        components=[
            _component("mystery"),
            _component(
                "glucose", s.MediaComponentRole.carbon_source, inchikey=_GLUCOSE_KEY
            ),
            _component("YNB", definition=s.ComponentDefinition.composition_deferred),
            s.MediaComponent(
                compound=_gapped("gapped"), role=s.MediaComponentRole.other
            ),
        ],
        dropouts=[
            s.Compound(name="uracil"),
            s.Compound(name="histidine", chebi_id="CHEBI:1"),
        ],
    )
    derived = s.Media(
        name="C",
        state="liquid",
        is_synthetic=True,
        base_medium="A",
        components=[
            _component("glucose", s.MediaComponentRole.other, inchikey=_GLUCOSE_KEY)
        ],
        dropouts=[s.Compound(name="mystery", smiles="C")],
    )
    faithful = s.Media(
        name="E",
        state="solid",
        is_synthetic=True,
        base_medium="A",
        components=base.components,
    )
    return {
        "A": base,
        "B": s.Media(name="B", state="liquid", is_synthetic=True, base_medium="NOPE"),
        "C": derived,
        "D": s.Media(name="D", state="liquid", is_synthetic=True, base_medium="D"),
        "E": faithful,
    }


def test_media_library_compound_issues_name_unidentified_defined_and_dropouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In A, ``mystery`` (defined, no identifier, no gap) and the dropout ``uracil`` are
    issues; glucose (InChIKey), ``gapped`` (typed gap), the deferred YNB and the
    ChEBI-identified dropout are not. E repeats A's components, so ``mystery`` again.
    """
    monkeypatch.setattr(oc, "MEDIA_LIBRARY", _library())
    issues = [i.model_dump() for i in oc.media_library_compound_issues()]
    assert issues == [
        {
            "context": "A.components",
            "compound_name": "mystery",
            "definition": "defined",
        },
        {"context": "A.dropouts", "compound_name": "uracil", "definition": "dropout"},
        {
            "context": "E.components",
            "compound_name": "mystery",
            "definition": "defined",
        },
    ]


def test_media_base_issues_name_a_base_outside_the_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only B's ``NOPE`` resolves to nothing; A has no base, C and E name A, D itself."""
    monkeypatch.setattr(oc, "MEDIA_LIBRARY", _library())
    assert [i.model_dump() for i in oc.media_base_issues()] == [
        {"media_key": "B", "base_medium": "NOPE"}
    ]


def test_media_derivation_issues_count_a_role_change_as_a_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C keeps glucose under role ``other`` instead of ``carbon_source``, drops
    ``mystery`` by declaration and omits YNB and ``gapped``: missing is
    [YNB, gapped, glucose]. E repeats A exactly; B, D and A itself are skipped.
    """
    monkeypatch.setattr(oc, "MEDIA_LIBRARY", _library())
    assert [i.model_dump() for i in oc.media_derivation_issues()] == [
        {
            "media_key": "C",
            "base_medium": "A",
            "missing_components": ["YNB", "gapped", "glucose"],
        }
    ]


def test_environment_compound_collisions_by_name_and_by_inchikey() -> None:
    """The component ``ethanol`` (no InChIKey) collides with the perturbation
    `` Ethanol`` by normalized name; the glucose component collides with a physical
    perturbation's agent ``D-glucose`` by InChIKey. A physical perturbation with no
    agent, a biologic (no compound attribute), the deferred YNB and an unrelated
    compound do not collide.
    """
    dose = s.Concentration(value=1.0, unit=s.ConcentrationUnit.millimolar)
    environment = s.Environment(
        media=s.Media(
            name="M",
            state="liquid",
            is_synthetic=True,
            components=[
                _component("ethanol"),
                _component("glucose", inchikey=_GLUCOSE_KEY),
                _component(
                    "ethanol", definition=s.ComponentDefinition.composition_deferred
                ),
            ],
        ),
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(name=" Ethanol"), concentration=dose
            ),
            s.SmallMoleculePerturbation(
                compound=s.Compound(name="caffeine"), concentration=dose
            ),
            s.EnvironmentPhysicalPerturbation(
                factor=s.PhysicalFactor.carbon_source,
                agent=s.Compound(name="D-glucose", inchikey=_GLUCOSE_KEY),
            ),
            s.EnvironmentPhysicalPerturbation(factor=s.PhysicalFactor.ph),
            s.BiologicPerturbation(
                agent_class=s.BiologicAgentClass.peptide,
                name="defensin",
                concentration=dose,
            ),
        ],
    )
    assert [
        c.model_dump() for c in oc.environment_compound_collisions(environment)
    ] == [
        {
            "identity": "ethanol",
            "component_name": "ethanol",
            "perturbation_name": " Ethanol",
        },
        {
            "identity": _GLUCOSE_KEY,
            "component_name": "glucose",
            "perturbation_name": "D-glucose",
        },
    ]


def test_join_key_audit_counts_each_partial_record_exactly() -> None:
    """Four records. r1 carries every key and two identified compounds (a component by
    InChIKey, a perturbation by ChEBI). r2 has a species without a strain, only
    unnamed perturbations, a base without a name, a temperature gap, a deferred
    component, an unidentified ``agent`` and a perturbation with no compound. r3 names a
    medium without a base and has neither temperature nor gap. r4 is two empty mappings.

    Finding: ``n_with_every_compound_identified`` counts a record with no compounds at
    all (``ontology_checks.py:889-898`` starts from ``resolved = True``), so r3 and the
    empty r4 raise it to 3 of 4. Pinned until the census separates "no compounds" from
    "all identified".
    """
    r1 = (
        {
            "genotype": {"perturbations": [{"systematic_gene_name": "YAL001C"}]},
            "environment": {
                "media": {
                    "name": "SC",
                    "base_medium": "SC",
                    "components": [
                        {
                            "definition": "defined",
                            "compound": {"name": "glucose", "inchikey": _GLUCOSE_KEY},
                        }
                    ],
                },
                "temperature": {"value": 30.0},
                "perturbations": [{"compound": {"name": "x", "chebi_id": "CHEBI:7"}}],
            },
        },
        {
            "genome_reference": {
                "species": "S. cerevisiae",
                "strain": "S288C",
                "ploidy": "haploid",
            }
        },
    )
    r2 = (
        {
            "genotype": {"perturbations": [{"systematic_gene_name": None}]},
            "environment": {
                "media": {
                    "base_medium": "YPD",
                    "components": [
                        {
                            "definition": "composition_deferred",
                            "compound": {"name": "YNB"},
                        }
                    ],
                },
                "temperature": None,
                "provenance_gaps": [{"field": "temperature"}],
                "perturbations": [{"agent": {"name": "mystery"}}, {"factor": "pH"}],
            },
        },
        {"genome_reference": {"species": "S. cerevisiae"}},
    )
    r3: tuple[dict[str, Any], dict[str, Any]] = (
        {"environment": {"media": {"name": "YPD"}}},
        {},
    )
    census = oc.join_key_audit([r1, r2, r3, ({}, {})])
    assert census.model_dump() == {
        "n_records": 4,
        "n_with_genome": 1,
        "n_with_systematic_gene_names": 1,
        "n_with_media_name": 2,
        "n_with_media_base": 2,
        "n_with_temperature": 1,
        "n_with_temperature_gap": 1,
        "n_with_every_compound_identified": 3,
        "distinct_genomes": ["S. cerevisiae|S288C|haploid"],
        "distinct_media_bases": ["SC", "YPD"],
        "distinct_compound_identities": [
            "chebi_id:CHEBI:7",
            f"inchikey:{_GLUCOSE_KEY}",
        ],
        "unidentified_compound_names": ["mystery"],
    }

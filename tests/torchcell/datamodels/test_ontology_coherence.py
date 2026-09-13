# tests/torchcell/datamodels/test_ontology_coherence
# [[tests.torchcell.datamodels.test_ontology_coherence]]
"""Coherence of the ontology ACROSS its artifacts, and of the identity keys joins need.

The existing modules each guard one artifact: ``test_ontology_invariants`` and
``test_ontology_all_trees`` guard the pydantic trees, ``test_schema_invariants`` the
registries, ``test_adapter_schema_consistency`` the enabled adapter conf against the
graph schema. Nothing guarded the SEAMS: a phenotype class with no node class, a node
class whose declared properties drifted off the model, an adapter chain that walks
through a field the schema allows to be None, a compound the ontology cannot resolve
to a structure, a medium whose base label names nothing.

Four families here, matching the four properties in
``torchcell/datamodels/ontology_checks.py``:

- **Representation** -- lanes stay disjoint and the composition DAG has no back-edge
  into the experiment lane; experiment/reference pairs agree on type tag, phenotype
  class and environment class; nothing is defined but unreachable.
- **Additive expansion** -- concrete phenotype classes, graph node classes and
  adapter node methods are in bijection, and every emitted property is declared (and
  every declared property emitted).
- **Joins** -- ``Compound`` identity, shared ``Media`` bases and derivation,
  ``Temperature``, systematic gene names; plus the census a verifier runs over a
  built dataset.
- **Provenance honesty** -- the ``ProvenanceGapMixin`` rule holds for EVERY subclass,
  and ``SourcedValue`` fields keep one name.

One test is ``xfail(strict=True)``: it names a real defect still open, and flips to a
hard failure the moment the defect is fixed, so the xfail cannot rot. The two media
defects it used to sit beside (a name-only shared library, a ``base_medium`` naming
nothing) are fixed and their tests are ordinary passing tests.
Findings and rejected checks: [[torchcell.datamodels.ontology-checks]].
"""

from __future__ import annotations

import itertools
import typing
from typing import Any, cast

import pytest
from pydantic import BaseModel

from torchcell.datamodels import ontology_checks as oc
from torchcell.datamodels import schema as s
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

GRAPH_SCHEMA = oc.graph_schema()
ADAPTER_SOURCE = oc.cell_adapter_source()
ADAPTER_SITES = oc.adapter_node_sites(ADAPTER_SOURCE)

# Models that exist in schema.py but that no experiment record reaches. Both are
# deliberate states rather than accidents, and BOTH are named here so a NEW orphan
# (a class added and then never wired into a record) fails this module.
#
# - ``Publication`` is a record the DATASET carries, not the experiment: the adapter
#   reads it from ``data["publication"]`` and emits a ``publication`` node class.
# - ``SOTerm`` is dead. The gene-perturbation leaves carry ``mechanism_so_id`` /
#   ``mechanism_so_name`` as plain strings, so nothing composes ``SOTerm`` and
#   nothing emits it; only tests and the figure's lane table mention it.
DOCUMENTED_ORPHANS: frozenset[str] = frozenset({"Publication", "SOTerm"})


def _cls_id(cls: object) -> str:
    """Stable parametrize id for a class value."""
    return getattr(cls, "__name__", str(cls))


CONCRETE_PHENOTYPES: list[type[s.Phenotype]] = sorted(
    (m for m in typing.get_args(s.PhenotypeType) if m is not s.Phenotype), key=_cls_id
)
# The registries are heterogeneous class-object dicts, so mypy joins their value
# type up to ModelMetaclass; the cast restores the type the schema guarantees (and
# test_schema_invariants already proves every value subclasses the declared base).
EXPERIMENT_TYPES: dict[str, type[s.Experiment]] = cast(
    "dict[str, type[s.Experiment]]", s.EXPERIMENT_TYPE_MAP
)
EXPERIMENT_REFERENCE_TYPES: dict[str, type[s.ExperimentReference]] = cast(
    "dict[str, type[s.ExperimentReference]]", s.EXPERIMENT_REFERENCE_TYPE_MAP
)
EXPERIMENT_KINDS: list[str] = sorted(EXPERIMENT_TYPES)


# --------------------------------------------------------------------------- #
# Representation: lanes, direction, reachability.
# --------------------------------------------------------------------------- #
def test_record_lanes_are_disjoint() -> None:
    """Genotype, environment and phenotype share no class.

    Disjointness is what keeps a persistent entity separable from a contingent
    observation. If a ``Media`` were also reachable from a ``Phenotype``, "what the
    cell was grown in" and "what was measured" would be one object, and a query for
    every record on YPD would depend on which phenotype family was measured.
    """
    lanes = oc.lane_membership()
    overlaps = {
        f"{a}&{b}": sorted(c.__name__ for c in lanes[a] & lanes[b])
        for a, b in itertools.combinations(sorted(lanes), 2)
        if lanes[a] & lanes[b]
    }
    assert overlaps == {}


def test_no_lane_composes_an_experiment() -> None:
    """The composition DAG points DOWN: no lane class holds an Experiment.

    Node ids are the sha256 of a model dump, so a phenotype composing its experiment
    would make each id depend on the other and no id could be computed at all.
    """
    assert oc.lane_back_edges() == []


def test_every_schema_model_is_reachable_or_documented() -> None:
    """No class is defined and then unreachable from any experiment record."""
    assert set(oc.orphan_models()) == set(DOCUMENTED_ORPHANS)


@pytest.mark.xfail(
    reason="SOTerm (torchcell/datamodels/schema.py:61) is dead: no field composes it "
    "and no adapter method emits it -- the gene leaves carry mechanism_so_id / "
    "mechanism_so_name as plain strings instead. Either type those fields on SOTerm "
    "or delete the class; until then it is an unreachable class inside the served "
    "schema surface that kg_manifest still fingerprints.",
    strict=True,
)
def test_every_orphan_model_is_at_least_emitted_to_the_graph() -> None:
    """An unreachable model earns its place only by being a graph node class."""
    node_classes = {
        name for name, entry in GRAPH_SCHEMA.items() if entry.kind == "node"
    }
    emitted = {name.replace(" ", "").lower() for name in node_classes}
    unusable = [name for name in oc.orphan_models() if name.lower() not in emitted]
    assert unusable == []


@pytest.mark.parametrize("kind", EXPERIMENT_KINDS)
def test_experiment_and_reference_agree_on_their_type_tag(kind: str) -> None:
    """The registry key, the experiment tag and the reference tag are one string.

    The three are written independently in schema.py, and the adapter + the
    reconstruction path key on them. A drift between any two silently splits one
    phenotype family into two.
    """
    experiment = EXPERIMENT_TYPES[kind]
    reference = EXPERIMENT_REFERENCE_TYPES[kind]
    assert experiment.model_fields["experiment_type"].default == kind
    assert reference.model_fields["experiment_reference_type"].default == kind


@pytest.mark.parametrize("kind", EXPERIMENT_KINDS)
def test_experiment_and_reference_share_a_phenotype_class(kind: str) -> None:
    """A record and its control measure the SAME phenotype class.

    Otherwise the control is not a control: the reference value would be a different
    quantity from the one it is subtracted from.
    """
    experiment = EXPERIMENT_TYPES[kind]
    reference = EXPERIMENT_REFERENCE_TYPES[kind]
    assert (
        experiment.model_fields["phenotype"].annotation
        is reference.model_fields["phenotype_reference"].annotation
    )


@pytest.mark.parametrize("kind", EXPERIMENT_KINDS)
def test_every_experiment_family_uses_the_shared_environment_class(kind: str) -> None:
    """Environment is never narrowed per family, on either side of the pair.

    A family-specific environment class would put the medium behind a per-family
    type, and the cross-dataset aggregate at the YPD level would stop being
    expressible.
    """
    experiment = EXPERIMENT_TYPES[kind]
    reference = EXPERIMENT_REFERENCE_TYPES[kind]
    assert experiment.model_fields["environment"].annotation is s.Environment
    assert reference.model_fields["environment_reference"].annotation is s.Environment
    assert reference.model_fields["genome_reference"].annotation is s.ReferenceGenome


# --------------------------------------------------------------------------- #
# Additive expansion: pydantic <-> graph schema <-> adapter.
# --------------------------------------------------------------------------- #
def test_every_adapter_node_site_is_statically_readable() -> None:
    """Every ``BioCypherNode(...)`` construction exposes its label and properties.

    The property checks below are only as complete as this reader. A site written in
    a form the reader does not understand must FAIL here rather than being skipped,
    or the coherence guarantee quietly shrinks as the adapter grows.
    """
    unreadable = [
        f"{site.function}:{site.lineno}"
        for site in ADAPTER_SITES
        if site.node_label is None or site.property_keys is None
    ]
    assert unreadable == []


def test_adapter_emits_exactly_the_declared_properties() -> None:
    """Emitted property set == declared property set, for every node class.

    BioCypher drops an undeclared label or property with no error, so an emitted
    property the schema omits is data lost silently. The reverse is the mirror
    failure: a declared property nothing emits is a column that is always null, and
    a query written against it returns nothing rather than erroring.

    The existing ``test_every_node_method_property_is_declared`` pins two labels by
    hand; this covers all of them, in both directions, from the AST.
    """
    mismatches = oc.adapter_property_mismatches(ADAPTER_SITES, GRAPH_SCHEMA)
    assert [m.model_dump() for m in mismatches] == []


def test_phenotype_classes_and_node_classes_are_in_bijection() -> None:
    """Each concrete phenotype has exactly one node class, and conversely.

    The mapping is derived from the property sets, not from a naming convention, so
    it holds for ``RNASeqExpressionPhenotype`` <-> ``rnaseq expression phenotype``
    and ``CalMorphPhenotype`` <-> ``calmorph phenotype`` without an alias table, and
    a field renamed on one side without the other breaks the match.
    """
    mapping = oc.phenotype_label_map(GRAPH_SCHEMA)
    assert mapping.ambiguous == {}
    assert mapping.unmatched_labels == []
    assert mapping.unmapped_classes == []
    assert len(mapping.matched) == len(CONCRETE_PHENOTYPES)


def test_phenotype_member_of_sources_are_exactly_the_phenotype_node_classes() -> None:
    """``phenotype member of`` lists every phenotype node class and nothing else.

    ``test_every_declared_phenotype_is_a_phenotype_member_of_source`` checks the
    subset direction; a source that is NOT a phenotype node class is the other way a
    new family gets mis-wired, and BioCypher accepts it silently.
    """
    sources = set(GRAPH_SCHEMA["phenotype member of"].source)
    phenotypes = {
        name
        for name, entry in GRAPH_SCHEMA.items()
        if entry.kind == "node" and name.endswith("phenotype")
    }
    assert sources == phenotypes


def test_no_graph_node_class_is_isolated() -> None:
    """Every node class participates in at least one edge class.

    An isolated node class is written to the store and then unreachable from any
    traversal starting at an experiment, so nothing can ever join to it.
    """
    assert oc.isolated_graph_node_classes(GRAPH_SCHEMA) == []


def test_every_declared_node_class_is_emitted_by_the_adapter() -> None:
    """No node class is declared in the yaml that no adapter method ever writes."""
    declared = {name for name, entry in GRAPH_SCHEMA.items() if entry.kind == "node"}
    emitted = {site.node_label for site in ADAPTER_SITES if site.node_label}
    assert sorted(declared - emitted) == []


def test_adapter_never_walks_through_an_optional_schema_field() -> None:
    """No adapter chain dereferences past a field the schema allows to be None.

    ``Environment.temperature`` is optional because a curation layer that never carried
    a temperature records a typed gap rather than a guess, and ``CellAdapter`` used to
    walk straight through it in six chains: a record that legally gaps temperature
    raised AttributeError during the KG build. Every one of those reads is now bound to
    a local and guarded, so a gapped temperature yields no temperature node and no
    ``temperature member of`` edge. The check is generic, so it also holds the line for
    the next optional field the schema grows.
    """
    traversals = oc.adapter_optional_traversals(ADAPTER_SOURCE)
    assert [t.model_dump() for t in traversals] == []


# --------------------------------------------------------------------------- #
# Joins: compound identity, media bases, temperature, gene names.
# --------------------------------------------------------------------------- #
def test_compound_identity_predicates_discriminate() -> None:
    """The identity predicates separate identified, gapped and name-only compounds.

    This is the rule every other compound check is built on, so it is exercised
    directly rather than only through a sweep.
    """
    identified = s.Compound(name="furfural", inchikey="HYBBIBNJHNGZAN-UHFFFAOYSA-N")
    assert oc.compound_has_identity(identified)
    assert not oc.compound_has_identity_gap(identified)

    gapped = s.Compound(
        name="tunicamycin",
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey",
                reason=ProvenanceGapReason.deferred_pending_source_review,
            )
        ],
    )
    assert not oc.compound_has_identity(gapped)
    assert oc.compound_has_identity_gap(gapped)

    name_only = s.Compound(name="something nobody resolved")
    assert not oc.compound_has_identity(name_only)
    assert not oc.compound_has_identity_gap(name_only)


def test_small_molecule_perturbation_compound_is_checkable() -> None:
    """A chemical edit's compound is subject to the same identity rule as a component.

    A name-only compound on a perturbation is the case the review brief rules out
    outright: the records depending on it are dropped rather than served, so the
    predicate must see them.
    """
    perturbation = s.SmallMoleculePerturbation(
        compound=s.Compound(name="unresolvable code CMB0000"),
        concentration=s.Concentration(value=1.0, unit=s.ConcentrationUnit.micromolar),
    )
    assert not oc.compound_has_identity(perturbation.compound)
    assert not oc.compound_has_identity_gap(perturbation.compound)


def test_shared_media_compounds_are_identified_or_gapped() -> None:
    """Every single-substance compound in MEDIA_LIBRARY resolves, or says it cannot.

    Was a strict xfail: the library built every compound as a bare
    ``Compound(name=...)``, so 40 distinct substances carried neither a structure
    identifier nor a ``ProvenanceGap`` and the object the cross-dataset environment
    join keys on could not itself join. Every single-substance component and every
    dropout now goes through ``resolved_compound``.
    """
    issues = oc.media_library_compound_issues()
    assert [i.model_dump() for i in issues] == []


def test_every_media_base_resolves_to_a_library_member() -> None:
    """A derived medium names a base that EXISTS as a shared object.

    Was a strict xfail: ``SD_MINIMAL`` declared base ``'SD'`` and the two SGA
    selection media declared ``'SD_MSG'``, neither of which was a ``MEDIA_LIBRARY``
    key, so "aggregate every record on an SD/MSG base" joined nothing. ``SD_MSG`` is
    now a first-class base object and the minimal medium is registered under ``SD``.
    """
    issues = oc.media_base_issues()
    assert [i.model_dump() for i in issues] == []


def test_resolvable_media_bases_survive_into_their_derivatives() -> None:
    """A derived medium keeps every base component, or declares it as a dropout.

    This is what makes SC-Ura a typed edit of SC rather than a lookalike recipe: the
    single missing component is exactly the declared uracil dropout.
    """
    issues = oc.media_derivation_issues()
    assert [i.model_dump() for i in issues] == []


def test_media_base_chain_is_acyclic() -> None:
    """Following ``base_medium`` never revisits a medium.

    A root names itself (``YPD.base_medium == 'YPD'``), which is a fixed point, not a
    cycle. A genuine cycle (A derives from B derives from A) would make "the base of
    this medium" undefined and would hang any resolver that walks the chain. Whether
    a base label resolves at all is a separate property, checked above.
    """
    for key in sorted(MEDIA_LIBRARY):
        visited = [key]
        cursor = MEDIA_LIBRARY[key]
        while (
            cursor.base_medium is not None
            and cursor.base_medium in MEDIA_LIBRARY
            and cursor.base_medium != visited[-1]
        ):
            assert cursor.base_medium not in visited, visited
            visited.append(cursor.base_medium)
            cursor = MEDIA_LIBRARY[cursor.base_medium]


def test_library_media_names_are_unique() -> None:
    """No two library media share a ``name`` (the human-facing join handle)."""
    names = [media.name for media in MEDIA_LIBRARY.values()]
    assert sorted(names) == sorted(set(names))


def _ypd_environment(**overrides: Any) -> s.Environment:
    fields: dict[str, Any] = dict(
        media=MEDIA_LIBRARY["YPD"], temperature=s.Temperature(value=30.0)
    )
    fields.update(overrides)
    return s.Environment(**fields)


def test_a_component_compound_may_not_also_be_a_perturbation() -> None:
    """The same species cannot be both a constant ingredient and the studied edit.

    Encoding it twice double-counts the dose: an aggregate over "records dosed with
    glucose" would disagree with the medium's own composition for the same records.
    """
    glucose = s.Compound(name="D-glucose")
    medium = s.Media(
        name="YPD (test)",
        state="liquid",
        is_synthetic=False,
        base_medium="YPD",
        components=[
            s.MediaComponent(
                compound=glucose,
                role=s.MediaComponentRole.carbon_source,
                concentration=s.Concentration(
                    value=2.0, unit=s.ConcentrationUnit.percent_w_v
                ),
            )
        ],
    )
    collided = _ypd_environment(
        media=medium,
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(name="d-glucose"),
                concentration=s.Concentration(
                    value=1.0, unit=s.ConcentrationUnit.percent_w_v
                ),
            )
        ],
    )
    collisions = oc.environment_compound_collisions(collided)
    assert [c.identity for c in collisions] == ["d-glucose"]

    clean = _ypd_environment(
        media=medium,
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(name="hydrogen peroxide"),
                concentration=s.Concentration(
                    value=1.5, unit=s.ConcentrationUnit.millimolar
                ),
            )
        ],
    )
    assert oc.environment_compound_collisions(clean) == []


def test_shipped_library_media_carry_no_self_collision() -> None:
    """Every shared medium, used unperturbed, is collision-free by construction."""
    for key, media in sorted(MEDIA_LIBRARY.items()):
        environment = s.Environment(media=media, temperature=s.Temperature(value=30.0))
        assert oc.environment_compound_collisions(environment) == [], key


def _record_pair(environment: s.Environment) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = s.FitnessExperiment(
        dataset_name="test",
        genotype=s.Genotype(
            perturbations=[
                s.KanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
                )
            ]
        ),
        environment=environment,
        phenotype=s.FitnessPhenotype(fitness=0.9),
    )
    reference = s.FitnessExperimentReference(
        dataset_name="test",
        genome_reference=s.ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=environment,
        phenotype_reference=s.FitnessPhenotype(fitness=1.0),
    )
    return experiment.model_dump(mode="json"), reference.model_dump(mode="json")


def test_join_key_audit_counts_every_identity_key() -> None:
    """The census a verifier runs over a built LMDB reports each join level.

    It reads stored mappings rather than model objects, so a verifier can run it on
    any dataset without importing that dataset's loader.
    """
    census = oc.join_key_audit([_record_pair(_ypd_environment())])
    assert census.n_records == 1
    assert census.n_with_genome == 1
    assert census.n_with_systematic_gene_names == 1
    assert census.n_with_media_name == 1
    assert census.n_with_media_base == 1
    assert census.distinct_media_bases == ["YPD"]
    assert census.n_with_temperature == 1
    assert census.n_with_temperature_gap == 0
    # YPD's glucose used to be name-only, which made the record un-joinable on
    # compounds; the shared library now resolves it, so the census counts it.
    assert census.n_with_every_compound_identified == 1
    assert census.unidentified_compound_names == []


def test_join_key_audit_separates_a_missing_temperature_from_a_gapped_one() -> None:
    """A typed gap is counted as documented, a silent None is not counted at all."""
    gapped = _ypd_environment(
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature", reason=ProvenanceGapReason.not_carried_by_curation
            )
        ],
    )
    silent = _ypd_environment(temperature=None)
    census = oc.join_key_audit([_record_pair(gapped), _record_pair(silent)])
    assert census.n_records == 2
    assert census.n_with_temperature == 0
    assert census.n_with_temperature_gap == 1


def test_join_key_audit_sees_an_identified_perturbation_compound() -> None:
    """A fully identified environment counts as compound-joinable."""
    medium = s.Media(
        name="defined test medium",
        state="liquid",
        is_synthetic=True,
        base_medium="YPD",
        components=[
            s.MediaComponent(
                compound=s.Compound(
                    name="D-glucose", inchikey="WQZGKKKJIJFFOK-GASJEMHNSA-N"
                ),
                role=s.MediaComponentRole.carbon_source,
                concentration=s.Concentration(
                    value=2.0, unit=s.ConcentrationUnit.percent_w_v
                ),
            )
        ],
    )
    environment = _ypd_environment(
        media=medium,
        perturbations=[
            s.SmallMoleculePerturbation(
                compound=s.Compound(
                    name="hydrogen peroxide", inchikey="MHAJPDPJQMAIIY-UHFFFAOYSA-N"
                ),
                concentration=s.Concentration(
                    value=1.5, unit=s.ConcentrationUnit.millimolar
                ),
            )
        ],
    )
    census = oc.join_key_audit([_record_pair(environment)])
    assert census.n_with_every_compound_identified == 1
    assert census.unidentified_compound_names == []
    assert len(census.distinct_compound_identities) == 2


# --------------------------------------------------------------------------- #
# Provenance honesty: the gap mixin rule, and SourcedValue field naming.
# --------------------------------------------------------------------------- #
def _gap_mixin_subclasses() -> list[type[BaseModel]]:
    out: set[type[BaseModel]] = set()
    stack: list[type[BaseModel]] = [s.ProvenanceGapMixin]
    while stack:
        node = stack.pop()
        for child in node.__subclasses__():
            if child not in out:
                out.add(child)
                stack.append(child)
    return sorted(out, key=_cls_id)


GAP_MIXIN_SUBCLASSES = _gap_mixin_subclasses()


@pytest.mark.parametrize("cls", GAP_MIXIN_SUBCLASSES, ids=_cls_id)
def test_gap_mixin_rule_is_never_weakened_by_a_subclass(cls: type[BaseModel]) -> None:
    """No subclass shadows the gap validator or redeclares ``provenance_gaps``.

    The two honesty rules (a gap names a real field; a gapped field is None) live in
    one inherited validator. A subclass that redefined either the field or the
    validator would keep the attribute name while dropping the guarantee, and every
    downstream reader would still treat its gaps as audited.
    """
    assert "validate_provenance_gaps" not in cls.__dict__
    assert "provenance_gaps" not in (getattr(cls, "__annotations__", {}) or {})
    assert "provenance_gaps" in cls.model_fields


def test_gap_on_a_populated_field_is_rejected_on_each_mixin_family() -> None:
    """A value and a declared absence for the same field cannot coexist.

    Exercised on all three families that mix the behavior in (compound, environment,
    phenotype), since each reaches the validator through a different inheritance
    path.
    """
    with pytest.raises(ValueError):
        s.Compound(
            name="furfural",
            inchikey="HYBBIBNJHNGZAN-UHFFFAOYSA-N",
            provenance_gaps=[
                ProvenanceGap(
                    field="inchikey", reason=ProvenanceGapReason.not_reported_by_primary
                )
            ],
        )
    with pytest.raises(ValueError):
        s.Environment(
            media=MEDIA_LIBRARY["YPD"],
            temperature=s.Temperature(value=30.0),
            provenance_gaps=[
                ProvenanceGap(
                    field="temperature",
                    reason=ProvenanceGapReason.not_carried_by_curation,
                )
            ],
        )
    with pytest.raises(ValueError):
        s.FitnessPhenotype(
            fitness=0.9,
            fitness_std=0.1,
            provenance_gaps=[
                ProvenanceGap(
                    field="fitness_std",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_gap_on_an_unknown_field_is_rejected() -> None:
    """A gap must name a real field, so the gap census is a usable worklist."""
    with pytest.raises(ValueError):
        s.Compound(
            name="furfural",
            provenance_gaps=[
                ProvenanceGap(
                    field="not_a_field",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_sourced_value_fields_keep_one_name() -> None:
    """Every field carrying ``SourcedValue`` is called ``provenance``.

    One name is what lets a generic auditor walk any model and find the quotes
    without a per-class table; a second spelling would make the audit silently
    partial.
    """
    fields = oc.sourced_value_fields()
    assert fields
    misnamed = [key for key in fields if key.split(".")[-1] != "provenance"]
    assert misnamed == []


def test_no_schema_enum_aliases_two_members_to_one_value() -> None:
    """Two vocabulary terms cannot collapse into one through a shared value.

    ``a = "x"`` then ``b = "x"`` makes ``b`` an alias of ``a`` with no error, so the
    ontology would lose a term and every record written with it would deserialize as
    the other.
    """
    assert oc.enum_value_collisions() == {}

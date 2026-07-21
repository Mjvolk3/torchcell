"""Tier-1 ontology-integrity invariants across ALL trees + the semantic / grounding
rules. Companion to ``test_ontology_invariants.py`` (which covers the
gene-perturbation tree in depth).

This module implements the rule specs in
``notes/paper.database.ontological-enforcement.md`` (§"Detailed test specifications"),
scoped to what Tier 1 (branch ``ws15-env-chemogenomic``) can enforce WITHOUT
re-verifying the landed genetic datasets (Costanzo / Kuzmin / Kemmeren / Caudal):

Structural (Family I), ported to env / phenotype / experiment / experiment-reference:
  - S3 closed-world completeness (union == concrete leaves, grandfathering a permissive
    root member on the phenotype/experiment/reference unions -- a documented Tier-2 item).
  - S4 Liskov substitutability.
  - S5 discriminator uniqueness (trees that carry a discriminator tag).
  - S6 unique-parse round-trip (enforced here for the ENV tree; the phenotype /
    experiment trees' round-trips are exercised empirically by every built dataset's
    L0-L4 verification).
  - S7 abstract exclusion.

Semantic (Family II):
  - M1 edit != consequence (banned-vocab lint; enforced on the ENV tree, ALLOWLIST empty;
    the gene-tree suppressor/ts case is an xfail deferred to Tier 2).
  - M2 canonical form (``copy_number > 0`` on every dosage leaf; absence has ONE encoding).
  - M3 orthogonal factorization (provenance is a free field, not folded into a name).
  - M4 derived-not-stored (interaction order is genotype arity, never a stored field).

Grounding (Family III):
  - G1 identifier well-formedness (SO id+name pinned pairs; InChIKey / ChEBI / CID shapes).
  - G2 unit typing (Concentration / Temperature units are typed enums, not free strings).
  - G3 phenotype measurement contract (a phenotype that declares ``measurement_type``
    types it as an enum and declares ``units``).
"""

import re
import typing
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from torchcell.datamodels import schema as s
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    CopyNumberVariantPerturbation,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
    EnvironmentPerturbation,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    ExperimentReferenceType,
    ExperimentType,
    GeneInteractionPhenotype,
    GenePerturbation,
    GenePerturbationType,
    Genotype,
    MeasurementType,
    Phenotype,
    PhenotypeType,
    SmallMoleculePerturbation,
    Temperature,
    TemperatureUnit,
)


# --------------------------------------------------------------------------- #
# Generic leaf discovery: BFS over __subclasses__ from a root, minus the abstract
# bases. Mirrors the gene-tree helper so a newly added CONCRETE leaf is discovered
# automatically and MUST appear in the union or S3 fails.
# --------------------------------------------------------------------------- #
def _discover_leaves(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every (transitive) subclass of ``root``."""
    seen: set[type[BaseModel]] = set()
    stack: list[type[BaseModel]] = [root]
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _union_members(union: object) -> tuple[type[BaseModel], ...]:
    return typing.cast(tuple[type[BaseModel], ...], typing.get_args(union))


def _cls_id(c: object) -> str:
    """Stable parametrize id for a class-or-object value (pytest ids callable)."""
    return getattr(c, "__name__", str(c))


class TreeSpec:
    """One (root, union) tree + the config the generic invariants need."""

    def __init__(
        self,
        name: str,
        root: type[BaseModel],
        union: object,
        tag_field: str | None,
        root_in_union: bool,
    ) -> None:
        """Discover the tree's concrete leaves + union members from (root, union)."""
        self.name = name
        self.root = root
        self.union = union
        self.tag_field = tag_field
        # phenotype/experiment/reference unions permissively include their own root
        # as a generic fallback member (a documented Tier-2 review item); the env
        # tree does not. That root is the ONLY abstract class a union may contain.
        self.root_in_union = root_in_union
        self.leaves = frozenset(_discover_leaves(root))
        self.members = frozenset(_union_members(union))

    def __repr__(self) -> str:
        """Stable parametrize id (the tree name)."""
        return self.name


TREES: list[TreeSpec] = [
    TreeSpec(
        "env",
        EnvironmentPerturbation,
        EnvironmentPerturbationType,
        "perturbation_type",
        root_in_union=False,
    ),
    TreeSpec("phenotype", Phenotype, PhenotypeType, None, root_in_union=True),
    TreeSpec(
        "experiment", Experiment, ExperimentType, "experiment_type", root_in_union=True
    ),
    TreeSpec(
        "experiment_reference",
        ExperimentReference,
        ExperimentReferenceType,
        "experiment_reference_type",
        root_in_union=True,
    ),
]
_TREE_IDS = [t.name for t in TREES]


# --------------------------------------------------------------------------- #
# S3 closed-world completeness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tree", TREES, ids=_TREE_IDS)
def test_s3_union_equals_concrete_leaves(tree: TreeSpec) -> None:
    """Union == the independently discovered concrete leaves (root grandfathered).

    The important half: EVERY discovered leaf is in the union, so a new leaf can't
    silently drift out and fail to reconstruct from the KG.
    """
    grandfathered = {tree.root} if tree.root_in_union else set()
    assert tree.members - grandfathered == tree.leaves, (
        f"{tree.name}: union (minus grandfathered root) != discovered leaves; "
        f"missing={tree.leaves - tree.members}, extra={tree.members - grandfathered - tree.leaves}"
    )


# --------------------------------------------------------------------------- #
# S4 Liskov substitutability.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tree", TREES, ids=_TREE_IDS)
def test_s4_union_members_subclass_root(tree: TreeSpec) -> None:
    for member in tree.members:
        assert issubclass(member, tree.root), f"{member.__name__} !<= {tree.name} root"


# --------------------------------------------------------------------------- #
# S5 discriminator uniqueness (trees that carry a tag; phenotype is structural).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tree",
    [t for t in TREES if t.tag_field],
    ids=[t.name for t in TREES if t.tag_field],
)
def test_s5_discriminator_tags_unique(tree: TreeSpec) -> None:
    assert tree.tag_field is not None
    tags = [
        c.model_fields[tree.tag_field].default
        for c in tree.leaves
        if tree.tag_field in c.model_fields
    ]
    concrete_tags = [t for t in tags if isinstance(t, str)]
    assert len(concrete_tags) == len(set(concrete_tags)), (
        f"{tree.name}: duplicate {tree.tag_field} tags: {concrete_tags}"
    )


# --------------------------------------------------------------------------- #
# S7 abstract exclusion: the ONLY abstract class a union may contain is a
# grandfathered root; no other non-leaf base leaks in.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tree", TREES, ids=_TREE_IDS)
def test_s7_no_unexpected_abstract_in_union(tree: TreeSpec) -> None:
    allowed_abstract = {tree.root} if tree.root_in_union else set()
    abstract_in_union = tree.members - tree.leaves
    assert abstract_in_union == allowed_abstract, (
        f"{tree.name}: unexpected abstract classes in union: "
        f"{abstract_in_union - allowed_abstract}"
    )


# --------------------------------------------------------------------------- #
# S6 unique-parse round-trip -- ENFORCED for the ENV tree here (small, clean).
# The phenotype / experiment round-trips are exercised by every built dataset's
# L0-L4 verification, which serializes real records to LMDB and back.
# --------------------------------------------------------------------------- #
_ENV_FACTORY: dict[type[EnvironmentPerturbation], dict[str, Any]] = {
    SmallMoleculePerturbation: dict(
        compound=Compound(name="isobutanol"),
        concentration=Concentration(value=1.0, unit=ConcentrationUnit.millimolar),
    ),
    EnvironmentPhysicalPerturbation: dict(
        factor=s.PhysicalFactor.osmolarity,
        magnitude=Concentration(value=1.0, unit=ConcentrationUnit.molar),
        agent=Compound(name="sodium chloride"),
    ),
    s.BiologicPerturbation: dict(
        agent_class=s.BiologicAgentClass.protein,
        name="nisin",
        concentration=Concentration(value=1.0, unit=ConcentrationUnit.millimolar),
    ),
}
_ENV_ADAPTER: TypeAdapter[EnvironmentPerturbation] = TypeAdapter(
    EnvironmentPerturbationType
)


def test_env_factory_covers_every_leaf() -> None:
    env_tree = next(t for t in TREES if t.name == "env")
    assert set(_ENV_FACTORY) == set(env_tree.leaves)


@pytest.mark.parametrize("cls", sorted(_ENV_FACTORY, key=_cls_id), ids=_cls_id)
def test_s6_env_round_trip(cls: type[EnvironmentPerturbation]) -> None:
    inst = cls(**_ENV_FACTORY[cls])
    back = _ENV_ADAPTER.validate_python(inst.model_dump())
    assert type(back) is cls
    assert back.model_dump() == inst.model_dump()
    back_json = _ENV_ADAPTER.validate_json(inst.model_dump_json())
    assert type(back_json) is cls


# --------------------------------------------------------------------------- #
# M1 edit != consequence: no perturbation NAME (class name, field name, or the
# names/values of enums typed on its fields) uses a phenotypic-consequence word.
# --------------------------------------------------------------------------- #
BANNED_VOCAB: frozenset[str] = frozenset(
    {
        "stress",
        "sensitive",
        "tolerant",
        "resistant",
        "essential",
        "lethal",
        "damaging",
        "toxic",
        "inhibitory",
        "suppressor",
    }
)
# dose PROVENANCE endpoints are not consequences (IC30 target != "the strain is
# sensitive"); exempt so a DoseBasis enum does not trip the lint.
ALLOWED_VALUES: frozenset[str] = frozenset({"ic30", "ic50", "mic", "fixed"})


def _enums_in(annotation: object) -> list[type[Enum]]:
    """StrEnum classes referenced anywhere inside a (nested) type annotation."""
    out: list[type] = []

    def walk(ann: object) -> None:
        if isinstance(ann, type) and issubclass(ann, Enum):
            out.append(ann)
            return
        for arg in typing.get_args(ann):
            walk(arg)

    walk(annotation)
    return out


def _name_tokens(cls: type[BaseModel]) -> set[str]:
    """Tokens from the class name + own field names + enum member names/values."""
    toks: set[str] = set()

    def add(text: str) -> None:
        for part in re.split(r"[_\-]", text):
            if part:
                toks.add(part.lower())

    add(cls.__name__)
    for fname, field in cls.model_fields.items():
        add(fname)
        for enum_cls in _enums_in(field.annotation):
            for member in enum_cls.__members__.values():
                add(member.name)
                add(str(member.value))
    return toks - ALLOWED_VALUES


_ENV_LEAVES = sorted(
    _discover_leaves(EnvironmentPerturbation), key=lambda c: c.__name__
)


@pytest.mark.parametrize("cls", _ENV_LEAVES, ids=_cls_id)
def test_m1_env_perturbation_is_edit_only(cls: type[EnvironmentPerturbation]) -> None:
    """Every environment-perturbation leaf names an EDIT, never a consequence.

    ALLOWLIST is EMPTY for the env tree (the ``stress_type`` / ``stress_category``
    category error is what motivated this rule).
    """
    offending = _name_tokens(cls) & BANNED_VOCAB
    assert not offending, f"{cls.__name__} uses consequence vocab: {sorted(offending)}"


@pytest.mark.xfail(
    reason="Tier 2: gene-side suppressor/ts allowlist-vs-demote decision pending "
    "(SuppressorAllele* names carry consequence vocab as identity handles).",
    strict=True,
)
def test_m1_gene_tree_currently_has_consequence_names() -> None:
    """DOCUMENTS the deferred gene-tree M1 debt (xfail until Tier 2 resolves it)."""
    for cls in _discover_leaves(GenePerturbation):
        offending = _name_tokens(cls) & BANNED_VOCAB
        assert not offending, f"{cls.__name__}: {sorted(offending)}"


# --------------------------------------------------------------------------- #
# M2 canonical form: a dosage leaf's copy_number is strictly > 0, so "gene absent"
# has exactly ONE encoding (the presence/absence absence leaf), never CNV(0).
# --------------------------------------------------------------------------- #
_DOSAGE_LEAVES = [
    CopyNumberVariantPerturbation,
    EngineeredCopyNumberPerturbation,
    s.NaturalGenePresencePerturbation,
]


@pytest.mark.parametrize("cls", _DOSAGE_LEAVES, ids=_cls_id)
def test_m2_copy_number_must_be_positive(cls: type[GenePerturbation]) -> None:
    """copy_number == 0 is rejected on every dosage-bearing leaf (M2)."""
    base: dict[str, Any] = dict(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
    )
    if cls is EngineeredCopyNumberPerturbation:
        base |= dict(reference_copy_number=2.0)
    else:  # natural CNV / accessory-presence leaves are pangenome-keyed
        base |= dict(
            systematic_gene_name="pangenome1011:orf1",
            perturbed_gene_name="orf1",
            strain_id="AAB",
        )
    with pytest.raises(ValueError):
        cls(**{**base, "copy_number": 0.0})
    # a genuine positive dosage is accepted
    ok = cls(**{**base, "copy_number": 2.0})
    assert getattr(ok, "copy_number") == 2.0


def test_m2_absence_has_single_encoding() -> None:
    """Gene absent is ONLY the presence/absence absence leaf, never a CNV at 0.

    NaturalGeneAbsencePerturbation carries NO copy_number field (absence is a state,
    not a dosage), and no CNV leaf can represent 0 copies (validated above).
    """
    assert "copy_number" not in s.NaturalGeneAbsencePerturbation.model_fields
    absent = s.NaturalGeneAbsencePerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3", strain_id="AAB"
    )
    assert absent.state == "absent"


# --------------------------------------------------------------------------- #
# M3 orthogonal factorization: provenance is a free field on the gene root (not
# folded into a class name); the dosage axis keeps mechanism separate from state.
# --------------------------------------------------------------------------- #
def test_m3_provenance_is_a_free_field() -> None:
    assert "provenance" in GenePerturbation.model_fields
    provenances = {
        c.model_fields["provenance"].default
        for c in _discover_leaves(GenePerturbation)
        if "provenance" in c.model_fields
    }
    assert {"engineered", "natural"} <= provenances


def test_m3_dosage_axis_separates_mechanism_from_value() -> None:
    fields = EngineeredCopyNumberPerturbation.model_fields
    assert "mechanism_so_id" in fields and "copy_number" in fields
    assert "state" in fields  # engineered CNV keeps the (present) state distinct


# --------------------------------------------------------------------------- #
# M4 derived-not-stored: interaction order = genotype arity, never a stored field.
# --------------------------------------------------------------------------- #
def test_m4_interaction_order_is_not_stored() -> None:
    for banned in ("interaction_kind", "order", "interaction_order"):
        assert banned not in GeneInteractionPhenotype.model_fields, (
            f"GeneInteractionPhenotype must not store '{banned}' (derive from arity)"
        )


def test_m4_order_is_derivable_from_arity() -> None:
    def interaction_order(genotype: Genotype) -> int:
        return len(genotype.perturbations)

    digenic = Genotype(
        perturbations=[
            s.KanMxDeletionPerturbation(
                systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
            ),
            s.KanMxDeletionPerturbation(
                systematic_gene_name="YAL002W", perturbed_gene_name="VPS8"
            ),
        ]
    )
    assert interaction_order(digenic) == 2  # epsilon (k=2), computed not stored


# --------------------------------------------------------------------------- #
# G1 identifier well-formedness (membership, not just shape).
# --------------------------------------------------------------------------- #
SO_ALLOWED: dict[str, str] = {
    "SO:0000159": "deletion",
    "SO:0000667": "insertion",
    "SO:0001060": "sequence_variant",
    "SO:0001483": "SNV",
    "SO:0001019": "copy_number_variation",
    "SO:0001998": "sgRNA",
}


@pytest.mark.parametrize(
    "cls", sorted(_union_members(GenePerturbationType), key=_cls_id), ids=_cls_id
)
def test_g1_so_id_name_pairs_are_pinned(cls: type[GenePerturbation]) -> None:
    """Every gene leaf's (mechanism_so_id, mechanism_so_name) is a pinned SO pair."""
    fields = cls.model_fields
    if "mechanism_so_id" not in fields:
        pytest.skip(f"{cls.__name__} carries no SO mechanism")
    so_id = fields["mechanism_so_id"].default
    so_name = fields["mechanism_so_name"].default
    assert so_id in SO_ALLOWED, f"{cls.__name__}: unpinned SO id {so_id!r}"
    assert SO_ALLOWED[so_id] == so_name, (
        f"{cls.__name__}: SO id/name desync {so_id!r} != {so_name!r}"
    )


def test_g1_inchikey_shape_enforced() -> None:
    Compound(name="water", inchikey="XLYOFNOQVPJJNP-UHFFFAOYSA-N")  # valid
    for bad in ("not-a-key", "XLYOFNOQVPJJNP_UHFFFAOYSA_N", "ABC-DEF-G"):
        with pytest.raises(ValueError):
            Compound(name="x", inchikey=bad)


def test_g1_chebi_and_cid_shape_enforced() -> None:
    Compound(name="glucose", chebi_id="CHEBI:17234", pubchem_cid=5793)
    with pytest.raises(ValueError):
        Compound(name="x", chebi_id="17234")  # missing CHEBI: prefix
    with pytest.raises(ValueError):
        Compound(name="x", pubchem_cid=0)  # not a positive integer


# --------------------------------------------------------------------------- #
# G2 unit typing: dose / temperature units are typed enums, never free strings.
# --------------------------------------------------------------------------- #
def test_g2_concentration_unit_is_enum_typed() -> None:
    assert ConcentrationUnit in _enums_in(Concentration.model_fields["unit"].annotation)
    assert DoseBasis in _enums_in(Concentration.model_fields["basis"].annotation)
    with pytest.raises(ValueError):
        Concentration(value=1.0, unit="micromolar")  # type: ignore[arg-type]  # free string rejected


def test_g2_temperature_unit_is_enum_typed() -> None:
    assert Temperature.model_fields["unit"].annotation is TemperatureUnit
    with pytest.raises(ValueError):
        Temperature(value=30.0, unit="C")  # type: ignore[arg-type]  # free string rejected


# --------------------------------------------------------------------------- #
# G3 phenotype measurement contract: a phenotype that declares measurement_type
# types it as an enum and declares units.
#
# Tier-1 scope: ENFORCED where measurement_type is ALREADY enum-typed (the env
# tree). The landed metabolite / protein / rnaseq phenotypes carry a str-typed
# measurement_type; converting them to a typed enum rebuilds Zelezniak / Caudal, so
# it is Tier 2 -- tracked by the xfail below (fails until the backfill lands).
# --------------------------------------------------------------------------- #
_PHENO_WITH_MEASUREMENT = sorted(
    (c for c in _discover_leaves(Phenotype) if "measurement_type" in c.model_fields),
    key=lambda c: c.__name__,
)
_PHENO_ENUM_MEASUREMENT = [
    c
    for c in _PHENO_WITH_MEASUREMENT
    if _enums_in(c.model_fields["measurement_type"].annotation)
]


@pytest.mark.parametrize("cls", _PHENO_ENUM_MEASUREMENT, ids=_cls_id)
def test_g3_measurement_contract(cls: type[Phenotype]) -> None:
    ann = cls.model_fields["measurement_type"].annotation
    assert MeasurementType in _enums_in(ann), (
        f"{cls.__name__}: measurement_type must be enum-typed"
    )
    assert "units" in cls.model_fields, f"{cls.__name__}: must declare units"


def test_g3_env_response_participates() -> None:
    """Sanity: the env-response phenotype (Tier-1 work) is under the contract."""
    assert s.EnvironmentResponsePhenotype in set(_PHENO_ENUM_MEASUREMENT)


@pytest.mark.xfail(
    reason="Tier 2: metabolite/protein/rnaseq measurement_type is str-typed; typing it "
    "as a MeasurementType enum rebuilds Zelezniak/Caudal.",
    strict=True,
)
def test_g3_all_measurement_types_enum_typed() -> None:
    """DOCUMENTS the deferred phenotype-typing backfill (xfail until Tier 2)."""
    for cls in _PHENO_WITH_MEASUREMENT:
        assert _enums_in(cls.model_fields["measurement_type"].annotation), (
            f"{cls.__name__}: measurement_type still str-typed"
        )

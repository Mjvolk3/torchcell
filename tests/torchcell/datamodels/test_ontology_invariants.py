"""Ontology-integrity invariants for the perturbation hierarchy.

``schema.py`` is not just a schema -- it is the torchcell ONTOLOGY. As it grows
(more datasets, BioCypher adapters) we must PROVE it keeps good structural
properties. These checks are GENERIC: they walk the class tree / enumerate the
``GenePerturbationType`` union, so they keep holding as leaves are added.

The eight invariants (mirroring ``plan.torchcell-perturbation-ontology``):

1. Single-root + acyclic hierarchy (inheritance AND model-composition).
2. Liskov substitution across the union.
3. Union <-> concrete-leaf-set consistency.
4. Discriminator (``perturbation_type``) uniqueness.
5. Round-trip / serialization fidelity through the union.
6. SO-annotation well-formedness.
7. Provenance completeness.
8. Identity well-formedness (CURIE-or-systematic; a sequence_ref round-trips as tc://).
"""

import re
import typing
from typing import Any, cast

import pytest
from pydantic import BaseModel, TypeAdapter

from torchcell.datamodels import schema
from torchcell.datamodels.schema import (
    AllelePerturbation,
    ArtifactRef,
    BacterialCrisprActivationPerturbation,
    BacterialCrisprInterferencePerturbation,
    BacterialDegronPerturbation,
    BacterialDeletionPerturbation,
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialSpanDeletionPerturbation,
    BacterialVariantCall,
    BacterialVariantPerturbation,
    BacterialVariantType,
    BacterialMarkedAllelePerturbation,
    BarcodedKanMxDeletionPerturbation,
    ConditionalAllelePerturbation,
    CopyNumberVariantPerturbation,
    CrisprActivationPerturbation,
    CrisprConstruct,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    DampPerturbation,
    DeletionPerturbation,
    EngineeredCopyNumberPerturbation,
    ExpressionModulationPerturbation,
    GeneAdditionPerturbation,
    GenePerturbation,
    GenePerturbationType,
    HeterologousPathwayPerturbation,
    HeterozygousDeletionPerturbation,
    KanMxDeletionPerturbation,
    MarkerDeletionPerturbation,
    MeanDeletionPerturbation,
    NatMxDeletionPerturbation,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    PresenceAbsencePerturbation,
    PromoterReplacementPerturbation,
    SequencePerturbation,
    SequenceVariantPerturbation,
    SgaAllelePerturbation,
    SgaDampPerturbation,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    SuppressorAllelePerturbation,
    TransposonInsertionPerturbation,
    TsAllelePerturbation,
    VariantCallMode,
    VariantFrequencyBasis,
    VariantSiteKind,
)

SO_ID_RE = re.compile(r"^SO:\d{7}$")

# A CURIE ``prefix:local``, a bare S288C systematic gene name, or a bacterial locus
# tag of one of the three deposited namespaces. The bacterial forms are the patterns
# ``schema.BACTERIAL_LOCUS_TAG_PATTERNS`` validates against, restated here so this
# module checks the identity rule independently of the validator that enforces it --
# a leaf whose identifier matched neither family would fail here even if its own
# validator let it through.
_CURIE_RE = re.compile(r"^[A-Za-z0-9_.]+:[A-Za-z0-9_.\-]+$")
_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
_BACTERIAL_TAG_RE = re.compile(
    r"^(b\d{4}|BW25113_\d{4}"
    r"|PP_(?:\d{4}|(?:5|16|23)S[A-G]|t\d{2}|tm\d{2}|mr\d{2}|r\d{2}))$"
)


def _members(union: object) -> tuple[type[GenePerturbation], ...]:
    """Concrete classes in a (possibly nested) ``X | Y | ...`` union annotation."""
    return cast(tuple[type[GenePerturbation], ...], typing.get_args(union))


def _by_name(cls: type[GenePerturbation]) -> str:
    """Sort key: class name (keeps parametrize ids stable + mypy-typed)."""
    return cls.__name__


UNION_MEMBERS: frozenset[type[GenePerturbation]] = frozenset(
    _members(GenePerturbationType)
)

# ABSTRACT = classes that are NEVER materialized as a stored record. These are the
# three axis ABCs + the root, PLUS the intermediate "family base" classes that only
# ever serve as parents for the marker / Sga leaves (no loader constructs them
# directly -- confirmed by grepping every ``*Perturbation(`` call site). Defining
# abstractness explicitly (rather than "has a perturbation_type default") is what
# lets the union<->leaf test below be exact: a newly added CONCRETE leaf is auto
# discovered via ``__subclasses__`` and, unless it is added here, MUST appear in the
# union or test_union_equals_concrete_leaves fails.
ABSTRACT: frozenset[type[GenePerturbation]] = frozenset(
    {
        GenePerturbation,
        PresenceAbsencePerturbation,
        SequencePerturbation,
        ExpressionModulationPerturbation,
        DeletionPerturbation,
        DampPerturbation,
        TsAllelePerturbation,
        AllelePerturbation,
        SuppressorAllelePerturbation,
        # The called-variant base (#731) defines what a call is and which host it
        # is written against; its three leaves below say WHERE the call sits, and
        # only they are ever stored.
        BacterialVariantPerturbation,
    }
)


def _all_perturbation_subclasses() -> set[type[GenePerturbation]]:
    """Every (transitive) subclass of ``GenePerturbation``."""
    seen: set[type[GenePerturbation]] = set()
    stack: list[type[GenePerturbation]] = [GenePerturbation]
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child not in seen and issubclass(child, GenePerturbation):
                seen.add(child)
                stack.append(child)
    return seen


CONCRETE_LEAVES: frozenset[type[GenePerturbation]] = frozenset(
    _all_perturbation_subclasses() - ABSTRACT
)

_LEAVES_SORTED: list[type[GenePerturbation]] = sorted(CONCRETE_LEAVES, key=_by_name)
_UNION_SORTED: list[type[GenePerturbation]] = sorted(UNION_MEMBERS, key=_by_name)


# --------------------------------------------------------------------------- #
# Minimal valid instance for every concrete leaf. Placeholder-but-valid values
# for required fields (strain_id, source_organism, ...); identities are CURIE or
# systematic so the identity invariant holds, and sequence types carry a sha256-pinned
# ArtifactRef so the sequence-pointer invariant is actually exercised.
# --------------------------------------------------------------------------- #
_SYS = dict(systematic_gene_name="YAL001C", perturbed_gene_name="TFC3")
_BNUM = dict(
    systematic_gene_name="b0002",
    perturbed_gene_name="thrA",
    gene_namespace="ecoli_k12_mg1655_bnumber",
)
_URI = dict(
    sequence_ref=ArtifactRef(
        tier="genomes",
        key="peter2018_1011_assemblies",
        path="allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        member="YAL001C.fasta#AAB",
        sha256="a" * 64,
    )
)

_SNV_CALL = BacterialVariantCall(
    variant_type=BacterialVariantType.snv,
    type_statement="SNP",
    reference_sequence="AE015451",
    position_start=3866001,
    position_end=3866001,
    sequence_change="G\u2192A",
    annotation="P293S (CCA\u2192TCA)",
    call_mode=VariantCallMode.clone,
    frequency_statement="1",
    frequency=1.0,
    frequency_basis=VariantFrequencyBasis.fraction,
    caller="breseq 0.33.1",
)
_INTERGENIC_CALL = BacterialVariantCall(
    variant_type=BacterialVariantType.insertion,
    type_statement="INS",
    reference_sequence="AE015451",
    position_start=4586057,
    position_end=4586057,
    sequence_change="+C",
    annotation="intergenic (+140/+75)",
    call_mode=VariantCallMode.clone,
    frequency_statement="1",
    frequency=1.0,
    frequency_basis=VariantFrequencyBasis.fraction,
)
_SPAN_CALL = BacterialVariantCall(
    variant_type=BacterialVariantType.deletion,
    type_statement="DEL",
    reference_sequence="AE015451",
    position_start=4588139,
    position_end=4593691,
    sequence_change="\u03945,553 bp",
    call_mode=VariantCallMode.clone,
    frequency_statement="1",
    frequency=1.0,
    frequency_basis=VariantFrequencyBasis.fraction,
)

FACTORY: dict[type[GenePerturbation], dict[str, Any]] = {
    KanMxDeletionPerturbation: {**_SYS},
    BarcodedKanMxDeletionPerturbation: {
        **_SYS,
        "barcode": "ACGTACGTACGTACGTACGT",
        "collection": "Euroscarf MATa deletion set",
    },
    HeterozygousDeletionPerturbation: {**_SYS, "cassette": "kanMX4"},
    ConditionalAllelePerturbation: {
        **_SYS,
        "allele_class": "temperature_sensitive",
        "allele_name": "tfc3-1",
    },
    NatMxDeletionPerturbation: {**_SYS},
    MeanDeletionPerturbation: {**_SYS, "num_duplicates": 2},
    MarkerDeletionPerturbation: {**_SYS, "marker": "KlURA3"},
    SgaKanMxDeletionPerturbation: {**_SYS, "strain_id": "AAA"},
    SgaNatMxDeletionPerturbation: {**_SYS, "strain_id": "AAA"},
    SgaDampPerturbation: {**_SYS, "strain_id": "AAA"},
    SgaTsAllelePerturbation: {**_SYS, "strain_id": "AAA"},
    SgaAllelePerturbation: {**_SYS, "strain_id": "AAA"},
    SgaSuppressorAllelePerturbation: {**_SYS, "strain_id": "AAA"},
    GeneAdditionPerturbation: dict(
        systematic_gene_name="Xden:crtYB",
        perturbed_gene_name="crtYB",
        source_organism="Xanthophyllomyces dendrorhous",
        is_heterologous=True,
        localization="episomal_2micron",
    ),
    NaturalGeneAbsencePerturbation: {**_SYS, "strain_id": "AAB", **_URI},
    NaturalGenePresencePerturbation: dict(
        systematic_gene_name="pangenome1011:orf1",
        perturbed_gene_name="orf1",
        strain_id="AAB",
        **_URI,
    ),
    SequenceVariantPerturbation: {**_SYS, "strain_id": "AAB", **_URI},
    CopyNumberVariantPerturbation: dict(
        systematic_gene_name="pangenome1011:orf1",
        perturbed_gene_name="orf1",
        copy_number=2.0,
        strain_id="AAB",
        **_URI,
    ),
    EngineeredCopyNumberPerturbation: {
        **_SYS,
        "copy_number": 1.0,
        "reference_copy_number": 2.0,
        "marker": "KanMX",
    },
    CrisprDeletionPerturbation: {
        **_SYS,
        "crispr": CrisprConstruct(
            effector="SaCas9", guide_sequence="TGGGATGAACACCATCAAGT"
        ),
    },
    CrisprActivationPerturbation: {
        **_SYS,
        "crispr": CrisprConstruct(
            effector="dLbCas12a-VP", guide_sequence="CCACGGCATGTCAACAGGTGAGT"
        ),
    },
    CrisprInterferencePerturbation: {
        **_SYS,
        "crispr": CrisprConstruct(
            effector="dSpCas9-RD1152", guide_sequence="CGTACTACCAGATAACCTAA"
        ),
    },
    # Bacterial leaves: the identity is a locus tag of the namespace the record
    # declares, so each one names a REAL tag of its host (measured from that host's
    # deposited GenBank file), never a placeholder.
    BacterialDeletionPerturbation: {
        **_BNUM,
        "collection": "Keio collection",
        "cassette": "FRT-kan-FRT",
    },
    TransposonInsertionPerturbation: {
        "systematic_gene_name": "PP_0002",
        "perturbed_gene_name": "PP_0002",
        "gene_namespace": "pputida_kt2440_locus_tag",
        "barcode": "ACGTACGTACGTACGTACGT",
        "insertion_position": 12345,
        "insertion_strand": "+",
        "transposon": "mariner Himar1",
    },
    BacterialCrisprInterferencePerturbation: {
        "systematic_gene_name": "BW25113_0344",
        "perturbed_gene_name": "lacZ",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "crispr": CrisprConstruct(
            effector="dCas9-Mxi1", guide_sequence="CGTACTACCAGATAACCTAA"
        ),
    },
    PromoterReplacementPerturbation: {
        **_BNUM,
        "expression_direction": "increased",
        "promoter_name": "Ptac",
        "is_inducible": True,
    },
    # The called-variant leaves (#731). Every value is a row of the Lim 2025 released
    # matrix (the PP_3415 P293S SNP, the PP_4061/PP_4063 intergenic insertion, the
    # 5,553 bp deletion at 4,588,139), so the instances carry real coordinates.
    BacterialSequenceVariantPerturbation: dict(
        systematic_gene_name="PP_3415",
        perturbed_gene_name="PP_3415",
        gene_namespace="pputida_kt2440_locus_tag",
        call=_SNV_CALL,
    ),
    BacterialSiteVariantPerturbation: dict(
        systematic_gene_name="AE015451:4586057",
        perturbed_gene_name="PP_4061, PP_4063",
        gene_namespace="pputida_kt2440_locus_tag",
        call=_INTERGENIC_CALL,
        site_kind=VariantSiteKind.intergenic,
        flanking_systematic_gene_names=("PP_4061", "PP_4063"),
        flanking_gene_statement="PP_4061, PP_4063",
    ),
    BacterialSpanDeletionPerturbation: dict(
        systematic_gene_name="PP_4061",
        perturbed_gene_name="PP_4061",
        gene_namespace="pputida_kt2440_locus_tag",
        call=_SPAN_CALL,
        span_designation="AE015451:4588139 \u03945,553 bp",
        span_systematic_gene_names=("PP_4061", "PP_4063"),
    ),
    # Round-2 bacterial leaves (#749, #792, #799). Each identity is a REAL tag of the
    # host it declares, as the five above are.
    BacterialMarkedAllelePerturbation: {
        "systematic_gene_name": "BW25113_0344",
        "perturbed_gene_name": "lacZ-SPA",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "cassette": "kan",
        "insertion_site": "3'-UTR",
        "tag": "SPA",
        "terminus": "C",
        "allele_effect": "hypomorphic",
        "collection": "SPA-tag essential",
    },
    BacterialDegronPerturbation: {
        "systematic_gene_name": "BW25113_0344",
        "perturbed_gene_name": "lacZ-DAS",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "degron": "DAS+4",
        "terminus": "C",
        "protease": "ClpXP",
        "adaptor": "SspB",
    },
    BacterialCrisprActivationPerturbation: {
        "systematic_gene_name": "BW25113_0344",
        "perturbed_gene_name": "lacZ",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "crispr": CrisprConstruct(
            effector="dCas9*-MCPSoxS", guide_sequence="CGTACTACCAGATAACCTAA"
        ),
    },
    HeterologousPathwayPerturbation: dict(
        systematic_gene_name="Efa:mvaE",
        perturbed_gene_name="mvaE",
        gene_namespace="pputida_kt2440_locus_tag",
        source_organism="Enterococcus faecalis",
        is_heterologous=True,
        localization="chromosomal_integration",
        pathway_name="isoprenol via mevalonate",
        copy_number=2.0,
    ),
}

_ADAPTER: TypeAdapter[GenePerturbation] = TypeAdapter(GenePerturbationType)


def _instance(cls: type[GenePerturbation]) -> GenePerturbation:
    """A minimal valid instance of a concrete leaf."""
    inst = cls(**FACTORY[cls])
    assert isinstance(inst, GenePerturbation)
    return inst


def _reparse(inst: GenePerturbation) -> GenePerturbation:
    """Round-trip an instance through the ``GenePerturbationType`` union."""
    return _ADAPTER.validate_python(inst.model_dump())


def test_factory_covers_every_concrete_leaf() -> None:
    """The instance factory stays in lockstep with the concrete leaves."""
    assert set(FACTORY) == CONCRETE_LEAVES


# --------------------------------------------------------------------------- #
# 1. Single-root + acyclic hierarchy (inheritance AND model composition).
# --------------------------------------------------------------------------- #
def test_hierarchy_is_single_rooted() -> None:
    """Every perturbation subclass reaches the single root ``GenePerturbation``."""
    for cls in _all_perturbation_subclasses():
        assert issubclass(cls, GenePerturbation)
        assert GenePerturbation in cls.__mro__


def _schema_models() -> set[type[BaseModel]]:
    """All pydantic models DEFINED in schema.py."""
    return {
        obj
        for obj in vars(schema).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == schema.__name__
    }


def _referenced_models(annotation: object) -> set[type[BaseModel]]:
    """BaseModel classes referenced anywhere inside a (nested) type annotation."""
    found: set[type[BaseModel]] = set()

    def walk(ann: object) -> None:
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            found.add(ann)
            return
        for arg in typing.get_args(ann):
            walk(arg)

    walk(annotation)
    return found


def test_model_composition_is_acyclic() -> None:
    """The "model contains model" graph over schema.py has no cycle (topo-sortable)."""
    models = _schema_models()
    edges: dict[type[BaseModel], set[type[BaseModel]]] = {}
    for model in models:
        refs: set[type[BaseModel]] = set()
        for field in model.model_fields.values():
            refs |= _referenced_models(field.annotation)
        edges[model] = {r for r in refs if r in models and r is not model}

    # DFS three-colour cycle detection.
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {m: WHITE for m in models}

    def visit(node: type[BaseModel], path: list[str]) -> None:
        colour[node] = GREY
        for nxt in edges[node]:
            if colour[nxt] == GREY:
                cycle = " -> ".join([*path, node.__name__, nxt.__name__])
                raise AssertionError(f"composition cycle: {cycle}")
            if colour[nxt] == WHITE:
                visit(nxt, [*path, node.__name__])
        colour[node] = BLACK

    for model in models:
        if colour[model] == WHITE:
            visit(model, [])


# --------------------------------------------------------------------------- #
# 2. Liskov substitution: a concrete instance IS every base, and substitutes for
#    a parent-typed slot (round-trips through the union preserving its class).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_liskov_instance_is_every_base(cls: type[GenePerturbation]) -> None:
    inst = _instance(cls)
    for base in cls.__mro__:
        if isinstance(base, type) and issubclass(base, GenePerturbation):
            assert isinstance(inst, base), f"{cls.__name__} not a {base.__name__}"


@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_liskov_substitutes_in_union_slot(cls: type[GenePerturbation]) -> None:
    inst = _instance(cls)
    back = _reparse(inst)
    assert type(back) is cls


# --------------------------------------------------------------------------- #
# 3. Union <-> concrete-leaf-set consistency.
# --------------------------------------------------------------------------- #
def test_union_equals_concrete_leaves() -> None:
    assert UNION_MEMBERS == CONCRETE_LEAVES


def test_no_abstract_base_in_union() -> None:
    assert not (UNION_MEMBERS & ABSTRACT)


@pytest.mark.parametrize("member", _UNION_SORTED)
def test_union_members_are_gene_perturbations(member: type[GenePerturbation]) -> None:
    assert issubclass(member, GenePerturbation)


# --------------------------------------------------------------------------- #
# 4. Discriminator uniqueness: perturbation_type is a unique tag per leaf.
# --------------------------------------------------------------------------- #
def test_perturbation_type_defaults_are_unique() -> None:
    tags = [m.model_fields["perturbation_type"].default for m in UNION_MEMBERS]
    assert len(tags) == len(set(tags)), f"duplicate perturbation_type: {tags}"


# --------------------------------------------------------------------------- #
# 5. Round-trip / serialization fidelity through the union.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_round_trip_fidelity(cls: type[GenePerturbation]) -> None:
    inst = _instance(cls)
    dump = inst.model_dump()
    back = _reparse(inst)
    assert type(back) is cls
    assert back.model_dump() == dump
    # JSON path too.
    back_json = _ADAPTER.validate_json(inst.model_dump_json())
    assert type(back_json) is cls


# --------------------------------------------------------------------------- #
# 6. SO-annotation well-formedness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_mechanism_so_id_well_formed(cls: type[GenePerturbation]) -> None:
    inst = _instance(cls)
    so_id = getattr(inst, "mechanism_so_id", None)
    if so_id is not None:
        assert SO_ID_RE.match(so_id), f"{cls.__name__}: bad SO id {so_id!r}"


# --------------------------------------------------------------------------- #
# 7. Provenance completeness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_provenance_is_declared(cls: type[GenePerturbation]) -> None:
    inst = _instance(cls)
    assert inst.provenance in {"engineered", "natural"}


# --------------------------------------------------------------------------- #
# 8. Identity well-formedness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_identity_is_curie_or_systematic(cls: type[GenePerturbation]) -> None:
    name = _instance(cls).systematic_gene_name
    assert (
        _SYSTEMATIC_RE.match(name)
        or _BACTERIAL_TAG_RE.match(name)
        or _CURIE_RE.match(name)
    ), name


def test_the_three_identity_families_do_not_overlap() -> None:
    """A yeast name, a bacterial tag and a CURIE are three disjoint identifier spaces.

    This is what makes the namespace field load-bearing rather than decorative:
    ``Genotype`` content identity is the sorted systematic names, so if a b-number
    could also read as a yeast ORF name the two hosts' records would share an id
    space. Checked on the real forms rather than asserted.
    """
    yeast = ["YAL001C", "YAL001C-A", "Q0010", "YNCA0001W"]
    bacterial = ["b0002", "BW25113_0002", "PP_0002", "PP_16SA", "PP_t01", "PP_mr01"]
    for name in yeast:
        assert _SYSTEMATIC_RE.match(name) and not _BACTERIAL_TAG_RE.match(name), name
    for name in bacterial:
        assert _BACTERIAL_TAG_RE.match(name) and not _SYSTEMATIC_RE.match(name), name
        assert not _CURIE_RE.match(name), name


@pytest.mark.parametrize("cls", _LEAVES_SORTED)
def test_sequence_ref_round_trips_through_its_string_form(
    cls: type[GenePerturbation],
) -> None:
    """A set ``sequence_ref`` is an ArtifactRef whose tc:// string parses back to it."""
    ref = getattr(_instance(cls), "sequence_ref", None)
    if ref is not None:
        assert isinstance(ref, ArtifactRef)
        assert ArtifactRef.parse(str(ref), sha256=ref.sha256, bytes=ref.bytes) == ref


@pytest.mark.parametrize(
    "cls",
    [
        CopyNumberVariantPerturbation,
        NaturalGeneAbsencePerturbation,
        NaturalGenePresencePerturbation,
        SequenceVariantPerturbation,
    ],
)
def test_sequence_pointer_is_one_ref_field(cls: type[GenePerturbation]) -> None:
    """The natural-variation leaves point at sequence by ``sequence_ref`` alone."""
    fields = cls.model_fields
    assert "sequence_ref" in fields
    assert fields["sequence_ref"].default is None
    assert "sequence_source" in fields
    assert "sequence_uri" not in fields
    assert "sequence_sha256" not in fields


# --------------------------------------------------------------------------- #
# SOTerm well-formedness (the reusable SO record used by the ontology).
# --------------------------------------------------------------------------- #
def test_soterm_accepts_valid_id() -> None:
    term = schema.SOTerm(so_id="SO:0000159", name="deletion")
    assert term.so_id == "SO:0000159"


def test_soterm_rejects_malformed_id() -> None:
    with pytest.raises(ValueError):
        schema.SOTerm(so_id="SO:159", name="deletion")


# --------------------------------------------------------------------------- #
# EngineeredCopyNumberPerturbation + ploidy: HIP/HOP chemogenomics support.
# --------------------------------------------------------------------------- #
def test_engineered_cnv_discriminator_and_defaults() -> None:
    """The engineered CNV leaf carries its own unique tag + engineered provenance."""
    pert = EngineeredCopyNumberPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        copy_number=1.0,
        reference_copy_number=2.0,
        marker="KanMX",
    )
    assert pert.perturbation_type == "engineered_copy_number"
    assert pert.provenance == "engineered"
    assert pert.state == "present"
    assert pert.mechanism_so_id == "SO:0001019"
    assert pert.mechanism_so_name == "copy_number_variation"


def test_engineered_cnv_uses_real_systematic_validator() -> None:
    """These are reference genes, not pangenome ORFs -- the strict validator applies."""
    with pytest.raises(ValueError):
        EngineeredCopyNumberPerturbation(
            systematic_gene_name="pangenome1011:orf1",
            perturbed_gene_name="orf1",
            copy_number=1.0,
            reference_copy_number=2.0,
        )


def test_engineered_cnv_marker_is_optional() -> None:
    """``marker`` defaults to None (an unmarked dosage change)."""
    pert = EngineeredCopyNumberPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        copy_number=3.0,
        reference_copy_number=1.0,
    )
    assert pert.marker is None


def test_reference_genome_ploidy_default_and_values() -> None:
    """Ploidy defaults to haploid (backward-compatible) and accepts diploid."""
    assert schema.ReferenceGenome(species="s", strain="w").ploidy == "haploid"
    assert (
        schema.ReferenceGenome(species="s", strain="w", ploidy="diploid").ploidy
        == "diploid"
    )
    with pytest.raises(ValueError):
        schema.ReferenceGenome(species="s", strain="w", ploidy="triploid")  # type: ignore[arg-type]


def test_hip_style_record_round_trips() -> None:
    """HIP: diploid ReferenceGenome + heterozygous deletion as copy 1 of 2."""
    ref = schema.ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
    )
    geno = schema.Genotype(
        perturbations=[
            EngineeredCopyNumberPerturbation(
                systematic_gene_name="YAL001C",
                perturbed_gene_name="TFC3",
                copy_number=1.0,
                reference_copy_number=2.0,
                marker="KanMX",
            )
        ]
    )
    assert ref.ploidy == "diploid"
    back = schema.Genotype.model_validate(geno.model_dump())
    assert isinstance(back.perturbations[0], EngineeredCopyNumberPerturbation)
    assert back.perturbations[0].copy_number == 1.0
    assert back.perturbations[0].reference_copy_number == 2.0


def test_hop_style_record_round_trips() -> None:
    """HOP: diploid ReferenceGenome + homozygous deletion stays absence (Deletion).

    Uses the concrete KanMX deletion leaf (the abstract ``DeletionPerturbation`` base is
    not a union member); it remains an AXIS-1 absence, NOT an engineered CNV.
    """
    ref = schema.ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
    )
    geno = schema.Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
            )
        ]
    )
    assert ref.ploidy == "diploid"
    back = schema.Genotype.model_validate(geno.model_dump())
    assert isinstance(back.perturbations[0], KanMxDeletionPerturbation)
    assert back.perturbations[0].state == "absent"
    assert not isinstance(back.perturbations[0], EngineeredCopyNumberPerturbation)


# --------------------------------------------------------------------------- #
# CRISPR perturbations: expression-modulation axis + CRISPR-deletion mechanism.
# --------------------------------------------------------------------------- #
def test_crispr_deletion_is_a_deletion() -> None:
    """CRISPRd is a THIRD deletion MECHANISM: it IS-A DeletionPerturbation (so any
    "all absences" filter catches it) yet carries the guide the paper released.
    """
    inst = CrisprDeletionPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        crispr=CrisprConstruct(
            effector="SaCas9", guide_sequence="TGGGATGAACACCATCAAGT"
        ),
    )
    assert isinstance(inst, DeletionPerturbation)
    assert isinstance(inst, PresenceAbsencePerturbation)
    assert inst.state == "absent"
    assert inst.mechanism_so_id == "SO:0000159"
    assert inst.provenance == "engineered"
    assert inst.crispr.effector == "SaCas9"
    # Not an expression modulation -- different axis, same tool.
    assert not isinstance(inst, ExpressionModulationPerturbation)


@pytest.mark.parametrize(
    "cls,direction",
    [
        (CrisprActivationPerturbation, "increased"),
        (CrisprInterferencePerturbation, "decreased"),
    ],
)
def test_expression_modulation_direction_defaults(
    cls: type[ExpressionModulationPerturbation], direction: str
) -> None:
    """Activation increases, interference decreases; the gene stays present, mechanism sgRNA."""
    inst = cls(**FACTORY[cls])
    assert isinstance(inst, ExpressionModulationPerturbation)
    assert inst.expression_direction == direction
    assert inst.state == "present"
    assert inst.mechanism_so_id == "SO:0001998"
    assert inst.provenance == "engineered"


def test_expression_direction_validator_rejects_bad_value() -> None:
    """expression_direction is constrained to increased | decreased."""
    with pytest.raises(ValueError):
        CrisprInterferencePerturbation(
            systematic_gene_name="YAL001C",
            perturbed_gene_name="TFC3",
            expression_direction="sideways",
            crispr=CrisprConstruct(effector="dCas9-Mxi1"),
        )


def test_crispr_construct_effector_is_required() -> None:
    """The effector must be sourced -- no default (provenance discipline)."""
    with pytest.raises(ValueError):
        CrisprConstruct()  # type: ignore[call-arg]


def test_crispr_construct_guide_is_optional_for_defer() -> None:
    """A screen that released only target genes (Mormino) scaffolds with guide_sequence=None."""
    c = CrisprConstruct(effector="dCas9-Mxi1", n_guides=3)
    assert c.guide_sequence is None
    assert c.n_guides == 3


def test_crispr_construct_plasmid_pointer_is_a_content_addressed_ref() -> None:
    """A plasmid pointer is an ArtifactRef, so it cannot exist without its sha256."""
    assert "effector_plasmid_uri" not in CrisprConstruct.model_fields
    assert "effector_plasmid_sha256" not in CrisprConstruct.model_fields
    assert CrisprConstruct(effector="dSpCas9-RD1152").effector_plasmid_ref is None
    with pytest.raises(ValueError):
        CrisprConstruct.model_validate(
            {
                "effector": "dSpCas9-RD1152",
                "effector_plasmid_ref": {
                    "tier": "objects",
                    "key": "k",
                    "path": "plasmid.gb",
                },
            }
        )
    ref = ArtifactRef(
        tier="objects", key="k", path="plasmid.gb", member="pMAGIC", sha256="a" * 64
    )
    ok = CrisprConstruct(effector="dSpCas9-RD1152", effector_plasmid_ref=ref)
    assert ok.effector_plasmid_ref == ref
    assert CrisprConstruct.model_validate(ok.model_dump()) == ok


def test_crispr_leaves_round_trip_through_genotype() -> None:
    """All three CRISPR leaves survive a Genotype round-trip preserving their class + guide."""
    geno = schema.Genotype(
        perturbations=[
            CrisprActivationPerturbation(
                systematic_gene_name="YAL001C",
                perturbed_gene_name="TFC3",
                crispr=CrisprConstruct(
                    effector="dLbCas12a-VP", guide_sequence="CCACGGCATGTCAACAGGTGAGT"
                ),
            ),
            CrisprInterferencePerturbation(
                systematic_gene_name="YAL002W",
                perturbed_gene_name="VPS8",
                crispr=CrisprConstruct(
                    effector="dSpCas9-RD1152", guide_sequence="CGTACTACCAGATAACCTAA"
                ),
            ),
            CrisprDeletionPerturbation(
                systematic_gene_name="YAL003W",
                perturbed_gene_name="EFB1",
                crispr=CrisprConstruct(
                    effector="SaCas9", guide_sequence="TGGGATGAACACCATCAAGT"
                ),
            ),
        ]
    )
    back = schema.Genotype.model_validate(geno.model_dump())
    by_type = {p.perturbation_type: p for p in back.perturbations}
    act = by_type["crispr_activation"]
    ci = by_type["crispr_interference"]
    cd = by_type["crispr_deletion"]
    assert isinstance(act, CrisprActivationPerturbation)
    assert isinstance(ci, CrisprInterferencePerturbation)
    assert isinstance(cd, CrisprDeletionPerturbation)
    assert ci.crispr.guide_sequence == "CGTACTACCAGATAACCTAA"

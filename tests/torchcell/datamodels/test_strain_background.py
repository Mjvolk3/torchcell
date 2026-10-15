# tests/torchcell/datamodels/test_strain_background.py
# [[tests.torchcell.datamodels.test_strain_background]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_strain_background.py
"""Strain background, chemogenomic deletion leaves and culture protocol (issue #507).

Pins the schema decision the four chemogenomic loaders (Vanacloig #500, Wildenhain
#504, Hillenmeyer #505, Hoepfner #506) will build on: every background element is
sourced or a typed gap, per-locus dosage respects the background, the new leaves
round-trip through ``GenePerturbationType`` and stay hashable, the reference carrying
a background is large enough to be interned, and the new environment fields leave
every pre-#507 environment id where it was.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import pytest
from pydantic import TypeAdapter

from torchcell.data.experiment_dataset import (
    INTERN_MIN_BYTES,
    ExperimentDataset,
    canonical_json,
    resolve_interned,
)
from torchcell.datamodels import schema as s
from torchcell.datamodels.identity import (
    BACKGROUND_ALLELE_IDENTITY_FIELDS,
    BACKGROUND_ALLELE_NON_IDENTITY_FIELDS,
    CULTURE_FORMAT_IDENTITY_FIELDS,
    ENVIRONMENT_IDENTITY_FIELDS,
    ENVIRONMENT_OPTIONAL_IDENTITY_FIELDS,
    EUKARYOTIC_STRAIN_BACKGROUND_IDENTITY_FIELDS,
    INTEGRATED_CASSETTE_IDENTITY_FIELDS,
    INTEGRATED_CASSETTE_NON_IDENTITY_FIELDS,
    PRE_CULTURE_IDENTITY_FIELDS,
    PROJECTED_BACKGROUND_CLASSES,
    STRAIN_BACKGROUND_IDENTITY_FIELDS,
    STRAIN_BACKGROUND_NON_IDENTITY_FIELDS,
    environment_identity,
    identity_sha256,
    strain_background_identity,
)
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.strain_background import (
    BAID_CONSTRUCTION,
    BAID_GENOTYPE,
    BAID_STRAIN,
    BRACHMANN_1998,
    KANMX4_CASSETTE,
    STANDARD_ALLELES,
    STANDARD_BY_GENOTYPES,
    baid_background,
    pending_source_review,
    standard_allele,
    standard_background,
)
from torchcell.sequence.construct import ConstructSpan, FeatureKey
from torchcell.sequence.genome.registry import (
    BW25113_BACKGROUND_GENOTYPE,
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

SPECIES = "Saccharomyces cerevisiae"
_PERTURBATIONS: TypeAdapter[Any] = TypeAdapter(s.GenePerturbationType)

# Wildenhain 2016 Sci Data, mirrored (paper.md sha256 89ff4d9b...); the quote the
# Wildenhain loader will cite for BY4741.
WILDENHAIN_BY4741 = SourcedValue(
    value="BY4741",
    provenance=Provenance(
        source_uri="paper.md",
        citation_key="wildenhainSystematicChemicalgeneticChemicalchemical2016",
        sha256="89ff4d9bf1d31719ab15c18ab7aca0b7caf10f55c7c239c1b021908c95439e33",
    ),
    quote="isogenic to BY4741, which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0 "
    "ura3Δ0",
)


def _gap(field: str) -> ProvenanceGap:
    return ProvenanceGap(
        field=field, reason=ProvenanceGapReason.deferred_pending_source_review
    )


#: A deposited construct contig and the one feature on it a cassette can point at, in
#: the shape ``test_construct.py`` pins; the bytes are the Cachera pBTX002 map.
#:
#: The two spans below are built with ``ConstructSpan.one(start, end, orientation)``,
#: the single-interval constructor the span-parts shape gives: ``ConstructSpan`` holds
#: ``parts`` with a composition, and ``start`` and ``end`` are properties that raise on
#: a compound span. The switch from the base shape's keyword form was the one measured
#: interaction between this change and the span-parts change, and it is the only line
#: either of them moves in the other.
_CONTIG = "63f89c1eaa13f888610ed781c4de12c8ac34f4e20512b4eb2c4f32d31af096da"
_CONTIG_REF = s.ArtifactRef(
    tier="library",
    key="cacheraCRISPAHighthroughputMethod2023",
    path="si/plasmids/pBTX002.gb",
    sha256=_CONTIG,
    bytes=21901,
)


def _placement(**overrides: Any) -> s.ConstructPlacement:
    """The pointer an integrated cassette carries: a contig, a feature, a host site."""
    base: dict[str, Any] = {
        "contig_sha256": _CONTIG,
        "contig_ref": _CONTIG_REF,
        "feature_key": FeatureKey(
            label="KanMX",
            so_id="SO:0000316",
            span=ConstructSpan.one(3001, 4113, "inline"),
        ),
        "host_placement": s.HostPlacement(
            assembly_set=SGD_S288C_R64,
            site_label="Delta",
            insert_span=ConstructSpan.one(2003, 10105, "inline"),
            provenance_gaps=[
                ProvenanceGap(
                    field="span", reason=ProvenanceGapReason.not_reported_by_primary
                )
            ],
        ),
    }
    base.update(overrides)
    return s.ConstructPlacement(**base)


def _his3(zygosity: s.Zygosity | None = s.Zygosity.haploid, **kw: Any) -> Any:
    base: dict[str, Any] = dict(
        systematic_gene_name="YOR202W",
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        gene_name="HIS3",
        allele_name="his3Δ1",
        edit=s.AlleleEdit.partial_deletion,
        functional=False,
        zygosity=zygosity,
        provenance=[WILDENHAIN_BY4741],
    )
    base.update(kw)
    return s.BackgroundAllele(**base)


def _by4743() -> s.EukaryoticStrainBackground:
    return standard_background("BY4743", resolve_with=BRACHMANN_1998)


def _euk(**kwargs: Any) -> s.EukaryoticStrainBackground:
    """A eukaryotic background carrying the S288C pin every yeast record states."""
    base: dict[str, Any] = {"reference_strain": "S288C", "assembly_set": SGD_S288C_R64}
    base.update(kwargs)
    return s.EukaryoticStrainBackground(**base)


def _bacterial(**kwargs: Any) -> s.StrainBackground:
    """A host-agnostic background: BW25113, which states no ploidy (the retired
    ``BacterialStrainBackground``'s shape, now the base's).
    """
    base: dict[str, Any] = {
        "name": "BW25113",
        "reference_strain": "BW25113",
        "assembly_set": "ecoli_K12_BW25113_ASM75055v1",
        "genotype_statement": BW25113_BACKGROUND_GENOTYPE,
        "provenance": [WILDENHAIN_BY4741],
    }
    base.update(kwargs)
    return s.StrainBackground(**base)


def _rph() -> s.BackgroundAllele:
    """One BW25113 lesion as a typed allele: a locus tag in its own namespace."""
    return s.BackgroundAllele(
        systematic_gene_name="BW25113_3643",
        gene_namespace="ecoli_k12_bw25113_locus_tag",
        gene_name="rph",
        allele_name="rph-1",
        edit=s.AlleleEdit.sequence_variant,
        functional=False,
        provenance=[WILDENHAIN_BY4741],
    )


# ------------------------------------------------------------------ BackgroundAllele


def test_allele_sourced_by_quote_is_sourced() -> None:
    allele = _his3()
    assert allele.is_sourced
    assert allele.mechanism_so == ("SO:0000159", "deletion")


def test_allele_with_neither_quote_nor_gap_is_rejected() -> None:
    with pytest.raises(ValueError, match="carries no ProvenanceGap"):
        _his3(provenance=None)


def test_allele_with_empty_quote_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        _his3(provenance=[])


def test_allele_asserted_with_pending_review_gap() -> None:
    allele = _his3(provenance=None, provenance_gaps=[_gap("provenance")])
    assert not allele.is_sourced
    assert allele.allele_name == "his3Δ1"


def test_allele_unknown_zygosity_must_be_gapped_by_a_eukaryotic_background() -> None:
    """P5 moved the rule to the only class that can judge it, and nothing weakened.

    An allele on its own states no ploidy, so it cannot know whether a copy count is a
    fact; the eukaryotic background does, and refuses a silent ``None`` there exactly as
    the allele used to. The host-agnostic base refuses a copy count outright, because a
    single-copy genome has one possible value.
    """
    assert _his3(zygosity=None).zygosity is None
    with pytest.raises(ValueError, match="zygosity is unset and carries no"):
        _euk(
            name="BY4741",
            mating_type=s.MatingType.a,
            ploidy="haploid",
            alleles=[_his3(zygosity=None)],
            provenance=[WILDENHAIN_BY4741],
        )
    gapped = _euk(
        name="BY4741",
        mating_type=s.MatingType.a,
        ploidy="haploid",
        alleles=[_his3(zygosity=None, provenance_gaps=[_gap("zygosity")])],
        provenance=[WILDENHAIN_BY4741],
    )
    assert gapped.alleles[0].zygosity is None
    with pytest.raises(ValueError, match="states no ploidy"):
        _bacterial(alleles=[_rph().model_copy(update={"zygosity": s.Zygosity.haploid})])


def test_cassette_iff_cassette_replacement() -> None:
    with pytest.raises(ValueError, match="cassette"):
        _his3(cassette="kanMX4")
    with pytest.raises(ValueError, match="cassette"):
        _his3(edit=s.AlleleEdit.cassette_replacement)
    ok = _his3(edit=s.AlleleEdit.cassette_replacement, cassette="natMX")
    assert ok.cassette == "natMX"


def test_allele_systematic_name_is_validated() -> None:
    with pytest.raises(ValueError, match="systematic"):
        _his3(systematic_gene_name="HIS3")


def test_genomic_span_is_ordered_and_one_based() -> None:
    span = s.GenomicSpan(
        chromosome="chrXV",
        start=721946,
        end=722608,
        assembly="sgd_S288C_R64-4-1_20230830",
    )
    assert span.end - span.start == 662
    for start, end in ((0, 5), (10, 9)):
        with pytest.raises(ValueError):
            s.GenomicSpan(
                chromosome="chrXV",
                start=start,
                end=end,
                assembly="sgd_S288C_R64-4-1_20230830",
            )


# ------------------------------------------------------------------ StrainBackground


def test_background_mating_type_must_be_set_or_gapped() -> None:
    kwargs: dict[str, Any] = dict(
        name="BY4741", ploidy="haploid", provenance=[WILDENHAIN_BY4741]
    )
    with pytest.raises(ValueError, match="mating_type"):
        _euk(mating_type=None, **kwargs)
    gapped = _euk(mating_type=None, provenance_gaps=[_gap("mating_type")], **kwargs)
    assert gapped.mating_type is None
    assert not gapped.is_fully_sourced


def test_haploid_background_cannot_be_a_alpha() -> None:
    with pytest.raises(ValueError, match="haploid"):
        _euk(
            name="x",
            mating_type=s.MatingType.a_alpha,
            ploidy="haploid",
            provenance=[WILDENHAIN_BY4741],
        )


def test_zygosity_must_fit_ploidy() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        _euk(
            name="BY4743",
            mating_type=s.MatingType.a_alpha,
            ploidy="diploid",
            alleles=[_his3(s.Zygosity.haploid)],
            provenance=[WILDENHAIN_BY4741],
        )


def test_one_entry_per_locus_unless_compound_heterozygote() -> None:
    het_a = _his3(s.Zygosity.heterozygous)
    het_b = _his3(s.Zygosity.heterozygous, allele_name="his3Δ200")
    compound = _euk(
        name="x",
        mating_type=s.MatingType.a_alpha,
        ploidy="diploid",
        alleles=[het_a, het_b],
        provenance=[WILDENHAIN_BY4741],
    )
    assert compound.functional_copies("YOR202W") == 0
    with pytest.raises(ValueError, match="compound heterozygote"):
        _euk(
            name="x",
            mating_type=s.MatingType.a_alpha,
            ploidy="diploid",
            alleles=[_his3(s.Zygosity.homozygous), het_b],
            provenance=[WILDENHAIN_BY4741],
        )


def test_by4743_functional_copies_respect_the_background() -> None:
    """The #505/#506 marker-locus case: his3Δ1/his3Δ1 has 0 working copies, LYS2/lys2Δ0
    and MET15/met15Δ0 have 1, an untouched autosomal gene has 2.
    """
    by4743 = _by4743()
    assert by4743.functional_copies("YOR202W") == 0  # HIS3
    assert by4743.functional_copies("YBR115C") == 1  # LYS2
    assert by4743.functional_copies("YLR303W") == 1  # MET17 (met15Δ0)
    assert by4743.functional_copies("YCL018W") == 0  # LEU2
    assert by4743.functional_copies("YAL001C") == 2


def test_by4741_functional_copies() -> None:
    by4741 = standard_background("BY4741", provenance=[WILDENHAIN_BY4741])
    assert by4741.is_fully_sourced
    assert by4741.mating_type is s.MatingType.a
    assert by4741.functional_copies("YLR303W") == 0
    assert by4741.functional_copies("YBR115C") == 1


def test_functional_copies_refuses_a_gapped_zygosity() -> None:
    background = _euk(
        name="x",
        mating_type=s.MatingType.a_alpha,
        ploidy="diploid",
        alleles=[_his3(zygosity=None, provenance_gaps=[_gap("zygosity")])],
        provenance=[WILDENHAIN_BY4741],
    )
    with pytest.raises(ValueError, match="undetermined"):
        background.functional_copies("YOR202W")


# ------------------------------------------------------- StrainReferenceGenome


def _strain_reference_genome() -> s.StrainReferenceGenome:
    return s.StrainReferenceGenome(
        species=SPECIES,
        strain="BY4743",
        ploidy="diploid",
        background=_by4743(),
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )


def _culture_environment(**kw: Any) -> s.CultureEnvironment:
    return s.CultureEnvironment(
        media=MEDIA_LIBRARY["YPD"], temperature=s.Temperature(value=30.0), **kw
    )


def _strain_reference() -> s.StrainEnvironmentResponseExperimentReference:
    return s.StrainEnvironmentResponseExperimentReference(
        dataset_name="Toy",
        genome_reference=_strain_reference_genome(),
        environment_reference=_culture_environment(),
        phenotype_reference=s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.log2_ratio, environment_response=0.0
        ),
    )


def test_plain_reference_genome_carries_the_pin_and_no_background() -> None:
    """The base class every served dataset uses now carries the assembly pin (P2).

    #507 kept this dump byte-identical so no served genome node id moved; P2 moves
    every one of them on purpose (the pin is a fact of every record, and a genome node
    id is the sha256 of this dump), which its record states. The background still
    lives on ``StrainReferenceGenome`` alone.
    """
    reference = s.ReferenceGenome(
        species=SPECIES,
        strain="S288C",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )
    assert json.dumps(reference.model_dump()) == (
        '{"provenance_gaps": [], "species": "Saccharomyces cerevisiae", '
        '"strain": "S288C", "ploidy": "haploid", '
        '"assembly_set": "sgd_S288C_R64-4-1_20230830", '
        '"assembly_accession": "GCA_000146045.2"}'
    )
    assert "background" not in s.ReferenceGenome.model_fields
    assert "culture_format" not in s.Environment.model_fields


def test_strain_reference_genome_requires_a_background() -> None:
    with pytest.raises(ValueError):
        s.StrainReferenceGenome(
            species=SPECIES,
            strain="BY4743",
            ploidy="diploid",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )  # type: ignore[call-arg]  # the missing field is the point


def test_strain_reference_genome_must_agree_on_name_and_ploidy() -> None:
    by4743 = _by4743()
    with pytest.raises(ValueError, match="name"):
        s.StrainReferenceGenome(
            species=SPECIES,
            strain="BY4741",
            ploidy="diploid",
            background=by4743,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
    with pytest.raises(ValueError, match="ploidy"):
        s.StrainReferenceGenome(
            species=SPECIES,
            strain="BY4743",
            ploidy="haploid",
            background=by4743,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )


def test_strain_reference_genome_round_trips() -> None:
    reference = _strain_reference_genome()
    back = s.StrainReferenceGenome.model_validate_json(reference.model_dump_json())
    assert back == reference
    assert {a.allele_name for a in back.background.alleles} == {
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "met15Δ0",
        "ura3Δ0",
    }


def test_strain_record_round_trips_through_the_type_maps() -> None:
    """The reconstruction path (``EXPERIMENT_TYPE_MAP[experiment_type]``, as the
    neo4j query loader and the deduplicator use it) resolves the new tag to the
    classes that DECLARE the subclass fields, so nothing is dropped.
    """
    experiment = s.StrainEnvironmentResponseExperiment(
        dataset_name="Toy",
        genotype=s.Genotype(
            perturbations=[
                s.HeterozygousDeletionPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="TFC3",
                    cassette="kanMX4",
                )
            ]
        ),
        environment=_culture_environment(
            pre_culture=s.PreCulture(
                source=s.PreCultureSource.frozen_stock, source_label="-5gen"
            )
        ),
        phenotype=s.EnvironmentResponsePhenotype(
            measurement_type=s.MeasurementType.log2_ratio, environment_response=-1.5
        ),
    )
    reference = _strain_reference()
    dump = json.loads(experiment.model_dump_json())
    assert dump["environment"]["pre_culture"]["source"] == "frozen_stock"
    kind = dump["experiment_type"]
    assert kind == "strain_environment_response"
    experiment_cls = s.EXPERIMENT_TYPE_MAP[kind]
    assert isinstance(experiment_cls, type) and issubclass(experiment_cls, s.Experiment)
    assert experiment_cls.model_validate(dump) == experiment
    ref_dump = json.loads(reference.model_dump_json())
    assert ref_dump["genome_reference"]["background"]["name"] == "BY4743"
    reference_cls = s.EXPERIMENT_REFERENCE_TYPE_MAP[kind]
    assert isinstance(reference_cls, type)
    assert issubclass(reference_cls, s.ExperimentReference)
    assert reference_cls.model_validate(ref_dump) == reference
    assert isinstance(experiment, s.EnvironmentResponseExperiment)


def test_strain_reference_is_interned_in_the_lmdb() -> None:
    """The background rides on the reference, which the LMDB interns whole once its
    canonical JSON reaches INTERN_MIN_BYTES; a BY4743 background takes it past the
    floor, so the record stores a pointer and the read path splices it back.
    """
    reference = _strain_reference()
    assert len(canonical_json(reference)) >= INTERN_MIN_BYTES

    class _Txn:
        def __init__(self) -> None:
            self.rows: dict[bytes, bytes] = {}

        def get(self, key: bytes) -> bytes | None:
            return self.rows.get(key)

        def put(self, key: bytes, value: bytes) -> None:
            self.rows[key] = value

    txn = _Txn()
    record: dict[str, Any] = {"reference": reference.model_dump()}
    ExperimentDataset._maybe_intern(
        None,  # type: ignore[arg-type]  # the method reads no instance state
        record,
        "reference",
        reference,
        reference.dataset_name,
        txn,
    )
    assert set(record["reference"]) == {"$ref", "name"}
    interned = {k.decode(): pickle.loads(v) for k, v in txn.rows.items()}
    restored = resolve_interned(record, interned)
    assert (
        s.StrainEnvironmentResponseExperimentReference.model_validate(
            restored["reference"]
        )
        == reference
    )


# ------------------------------------------------------------- strain_background lib


def test_standard_background_needs_exactly_one_sourcing() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        standard_background("BY4741")
    with pytest.raises(ValueError, match="exactly one"):
        standard_background(
            "BY4741", provenance=[WILDENHAIN_BY4741], resolve_with=BRACHMANN_1998
        )
    with pytest.raises(ValueError, match="non-empty"):
        standard_background("BY4741", provenance=[])


def test_pending_background_gaps_every_element_with_its_resolver() -> None:
    by4743 = _by4743()
    assert not by4743.is_fully_sourced
    gaps = [
        *by4743.provenance_gaps,
        *(g for a in by4743.alleles for g in a.provenance_gaps),
    ]
    assert len(gaps) == 1 + len(STANDARD_BY_GENOTYPES["BY4743"].alleles)
    assert {g.field for g in gaps} == {"provenance"}
    assert {g.reason for g in gaps} == {
        ProvenanceGapReason.deferred_pending_source_review
    }
    assert all(g.resolve_with == BRACHMANN_1998 for g in gaps)


def test_standard_allele_reads_the_spec() -> None:
    can1 = standard_allele(
        "can1Δ::STE2pr-Sp_his5", s.Zygosity.haploid, provenance=[WILDENHAIN_BY4741]
    )
    assert can1.systematic_gene_name == "YEL063C"
    assert can1.cassette == "STE2pr-Sp_his5"
    assert STANDARD_ALLELES["met15Δ0"].gene_name == "MET17"
    with pytest.raises(KeyError):
        standard_allele("ade2-1", s.Zygosity.haploid, resolve_with=BRACHMANN_1998)


def test_pending_source_review_gap_shape() -> None:
    gap = pending_source_review("barcode", BRACHMANN_1998, note="table not mirrored")
    assert gap.reason is ProvenanceGapReason.deferred_pending_source_review
    assert gap.resolve_with == BRACHMANN_1998
    assert KANMX4_CASSETTE.value == "kanMX4"


# ------------------------------------------------------- HeterozygousDeletion leaf


def _het(gene: str = "YAL001C", name: str = "TFC3", **kw: Any) -> Any:
    return s.HeterozygousDeletionPerturbation(
        systematic_gene_name=gene, perturbed_gene_name=name, cassette="kanMX4", **kw
    )


def test_heterozygous_deletion_is_present_and_not_a_knockout() -> None:
    het = _het()
    assert het.state == "present"
    assert het.perturbation_type == "heterozygous_deletion"
    assert (het.mechanism_so_id, het.mechanism_so_name) == ("SO:0000159", "deletion")
    assert not isinstance(het, s.DeletionPerturbation)


def test_heterozygous_deletion_round_trips_and_hashes() -> None:
    het = _het(
        barcode="ACGTACGTACGTACGTACGT",
        collection="YSC1055 OpenBiosystems",
        construction=s.StrainConstruction(lab="Lab 14", batch="3", plate="12"),
        provenance_gaps=[_gap("downtag_barcode")],
    )
    back = _PERTURBATIONS.validate_python(het.model_dump())
    assert type(back) is s.HeterozygousDeletionPerturbation
    assert back == het and hash(back) == hash(het)
    genotype = s.Genotype(perturbations=[het])
    assert genotype == s.Genotype.model_validate_json(genotype.model_dump_json())


def test_heterozygous_deletion_functional_copies_respect_the_background() -> None:
    by4743 = _by4743()
    copies = s.heterozygous_deletion_functional_copies
    assert copies(by4743, _het()) == 1
    assert copies(by4743, _het("YOR202W", "HIS3")) == 0
    assert copies(by4743, _het("YBR115C", "LYS2")) is None
    assert copies(by4743, _het("YBR115C", "LYS2", replaced_allele="lys2Δ0")) == 1
    assert copies(by4743, _het("YBR115C", "LYS2", replaced_allele="LYS2")) == 0
    by4741 = standard_background("BY4741", provenance=[WILDENHAIN_BY4741])
    with pytest.raises(ValueError, match="diploid"):
        copies(by4741, _het())


# ------------------------------------------------------- ConditionalAllele leaf


def test_conditional_allele_class_is_set_or_gapped() -> None:
    kwargs: dict[str, Any] = dict(
        systematic_gene_name="YBR160W", perturbed_gene_name="CDC28"
    )
    with pytest.raises(ValueError, match="allele_class"):
        s.ConditionalAllelePerturbation(allele_class=None, **kwargs)
    unknown = s.ConditionalAllelePerturbation(
        allele_class=None,
        provenance_gaps=[
            pending_source_review(
                "allele_class",
                Provenance(
                    source_uri="Wildenhain 2016 Sci Data Table 1 (not mirrored)"
                ),
            ),
            _gap("allele_name"),
        ],
        **kwargs,
    )
    assert unknown.mechanism_so_id == "SO:0001060"
    back = _PERTURBATIONS.validate_json(unknown.model_dump_json())
    assert type(back) is s.ConditionalAllelePerturbation and back == unknown
    assert len({unknown, back}) == 1


# --------------------------------------------------------- BarcodedKanMx additions


def test_barcoded_deletion_new_fields_default_none_and_gap() -> None:
    plain = s.BarcodedKanMxDeletionPerturbation(
        systematic_gene_name="YAL001C", perturbed_gene_name="TFC3"
    )
    assert plain.cassette is None and plain.construction is None
    assert plain.provenance_gaps == []
    gapped = s.BarcodedKanMxDeletionPerturbation(
        systematic_gene_name="YAL001C",
        perturbed_gene_name="TFC3",
        cassette="kanMX4",
        provenance_gaps=[_gap("barcode")],
    )
    assert gapped.barcode is None
    with pytest.raises(ValueError, match="not None"):
        s.BarcodedKanMxDeletionPerturbation(
            systematic_gene_name="YAL001C",
            perturbed_gene_name="TFC3",
            barcode="ACGT",
            provenance_gaps=[_gap("barcode")],
        )


def test_constructed_orf_keeps_distinct_strains_distinct() -> None:
    """Two physical strains served on one current gene (YAR042W <- YAR044W) are two
    perturbations, never one averaged record.
    """
    own = s.BarcodedKanMxDeletionPerturbation(
        systematic_gene_name="YAR042W", perturbed_gene_name="SWH1", cassette="kanMX4"
    )
    merged = s.BarcodedKanMxDeletionPerturbation(
        systematic_gene_name="YAR042W",
        perturbed_gene_name="SWH1",
        cassette="kanMX4",
        constructed_orf=s.ConstructedOrf(
            source_systematic_name="YAR044W",
            relation=s.OrfHistoryRelation.merged,
            deleted_span=None,
            provenance_gaps=[_gap("deleted_span")],
        ),
    )
    assert len({own, merged}) == 2
    with pytest.raises(ValueError, match="relation"):
        s.ConstructedOrf(
            source_systematic_name="YAR044W", relation=None, deleted_span=None
        )


def test_strain_construction_needs_a_field() -> None:
    with pytest.raises(ValueError, match="at least one"):
        s.StrainConstruction()


# ---------------------------------------------------------------- Environment


def test_culture_environment_without_protocol_joins_the_plain_environment() -> None:
    """A ``CultureEnvironment`` that states nothing extra has the plain environment's
    identity, so the chemogenomic records still join every other YPD record.
    """
    plain = s.Environment(
        media=MEDIA_LIBRARY["YPD"], temperature=s.Temperature(value=30.0)
    )
    culture = _culture_environment()
    assert culture.culture_format is None and culture.pre_culture is None
    assert culture.auxotroph_supplements is None
    assert environment_identity(culture) == environment_identity(plain)
    assert set(environment_identity(plain)) == set(ENVIRONMENT_IDENTITY_FIELDS)


def test_culture_protocol_is_part_of_identity_when_set() -> None:
    static = _culture_environment(
        culture_format=s.CultureFormat(
            vessel="96-well plate",
            working_volume_ul=100.0,
            shaking_rpm=0.0,
            inoculum_cells=50000.0,
            endpoint=s.EndpointRule.until_control_saturation,
        )
    )
    shaken = _culture_environment(
        culture_format=s.CultureFormat(
            vessel="96-well plate",
            working_volume_ul=100.0,
            shaking_rpm=770.0,
            inoculum_cells=50000.0,
            endpoint=s.EndpointRule.until_control_saturation,
        )
    )
    frozen = _culture_environment(
        pre_culture=s.PreCulture(
            source=s.PreCultureSource.frozen_stock, source_label="-5gen"
        )
    )
    assert "culture_format" in environment_identity(static)
    ids = {identity_sha256(environment_identity(e)) for e in (static, shaken, frozen)}
    assert len(ids) == 3
    assert identity_sha256(environment_identity(_culture_environment())) not in ids


def test_pre_culture_rules() -> None:
    with pytest.raises(ValueError, match="frozen_stock"):
        s.PreCulture(
            source=s.PreCultureSource.frozen_stock, medium=MEDIA_LIBRARY["YPD"]
        )
    log_phase = s.PreCulture(
        source=s.PreCultureSource.log_phase_culture,
        medium=MEDIA_LIBRARY["YPD"],
        od600_at_transfer=2.0,
        generations=None,
        provenance_gaps=[_gap("generations")],
        source_label="5gen",
    )
    assert s.PreCulture.model_validate_json(log_phase.model_dump_json()) == log_phase
    with pytest.raises(ValueError, match="non-negative"):
        s.CultureFormat(working_volume_ul=-1.0)


def test_supplements_and_vehicle_gaps() -> None:
    """An unnamed auxotroph supplement and an unstated vehicle are typed gaps; the
    vehicle gap sits on ``SmallMoleculePerturbation.solvent`` (``Solvent`` itself is
    shared by every small-molecule dataset and is unchanged).
    """
    gapped = _culture_environment(
        auxotroph_supplements=None, provenance_gaps=[_gap("auxotroph_supplements")]
    )
    assert gapped.auxotroph_supplements is None
    high_dose = s.SmallMoleculePerturbation(
        compound=s.Compound(name="sodium chloride"),
        concentration=s.Concentration(value=320.0, unit=s.ConcentrationUnit.millimolar),
        solvent=None,
        provenance_gaps=[_gap("solvent")],
    )
    assert high_dose.solvent is None
    assert "provenance_gaps" not in s.Solvent.model_fields


# -------------------------------------------------------------- identity tuples


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (s.CultureEnvironment, ENVIRONMENT_OPTIONAL_IDENTITY_FIELDS),
        (s.CultureFormat, CULTURE_FORMAT_IDENTITY_FIELDS),
        (s.PreCulture, PRE_CULTURE_IDENTITY_FIELDS),
        (s.BackgroundAllele, BACKGROUND_ALLELE_IDENTITY_FIELDS),
        (s.StrainBackground, STRAIN_BACKGROUND_IDENTITY_FIELDS),
        (s.EukaryoticStrainBackground, EUKARYOTIC_STRAIN_BACKGROUND_IDENTITY_FIELDS),
        (s.IntegratedCassette, INTEGRATED_CASSETTE_IDENTITY_FIELDS),
    ],
)
def test_identity_tuples_name_real_fields(model: Any, fields: tuple[str, ...]) -> None:
    for field in fields:
        assert field in model.model_fields, f"{model.__name__}.{field}"


@pytest.mark.parametrize(
    ("model", "identity", "non_identity"),
    [
        (
            s.BackgroundAllele,
            BACKGROUND_ALLELE_IDENTITY_FIELDS,
            BACKGROUND_ALLELE_NON_IDENTITY_FIELDS,
        ),
        (
            s.StrainBackground,
            STRAIN_BACKGROUND_IDENTITY_FIELDS,
            STRAIN_BACKGROUND_NON_IDENTITY_FIELDS,
        ),
        (
            s.EukaryoticStrainBackground,
            EUKARYOTIC_STRAIN_BACKGROUND_IDENTITY_FIELDS,
            STRAIN_BACKGROUND_NON_IDENTITY_FIELDS,
        ),
        (
            s.IntegratedCassette,
            INTEGRATED_CASSETTE_IDENTITY_FIELDS,
            INTEGRATED_CASSETTE_NON_IDENTITY_FIELDS,
        ),
    ],
)
def test_every_field_is_declared_identity_or_declared_not_identity(
    model: Any, identity: tuple[str, ...], non_identity: tuple[str, ...]
) -> None:
    """The partition is what would have caught the two fields P5 found unprojected.

    ``gene_namespace`` landed on every ``GenePerturbation`` in P1 and
    ``BACKGROUND_ALLELE_IDENTITY_FIELDS`` was never updated, and a review had to find
    it by reading. A field in neither tuple fails here, so the next one cannot land
    silently: it is projected, or it is written down as not identity with its reason.
    """
    assert sorted((*identity, *non_identity)) == sorted(model.model_fields)
    assert not set(identity) & set(non_identity)


def test_the_base_projection_omits_the_three_fields_the_base_does_not_have() -> None:
    """Before P5 the projection read ploidy and mating type with no branch at all.

    A background that states neither could not be projected, and the honest shapes a
    wrong fix takes are both defects: reading them off the base raises, and projecting
    a eukaryote through the base's tuple drops them and MERGES two different strains.
    """
    bacterial = _bacterial(alleles=[_rph()])
    projection = strain_background_identity(bacterial)
    assert set(projection) == set(STRAIN_BACKGROUND_IDENTITY_FIELDS)
    assert "ploidy" not in projection and "mating_type" not in projection
    assert projection["assembly_set"] == "ecoli_K12_BW25113_ASM75055v1"
    assert projection["alleles"][0]["gene_namespace"] == "ecoli_k12_bw25113_locus_tag"


def test_a_eukaryote_and_a_base_background_of_one_name_are_different_keys() -> None:
    """The merge the base's tuple would cause, stated as a test."""
    haploid = _euk(
        name="x",
        mating_type=s.MatingType.a,
        ploidy="haploid",
        provenance=[WILDENHAIN_BY4741],
    )
    bare = s.StrainBackground(
        name="x",
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        provenance=[WILDENHAIN_BY4741],
    )
    assert strain_background_identity(haploid) != strain_background_identity(bare)
    assert identity_sha256(strain_background_identity(haploid)) != identity_sha256(
        strain_background_identity(bare)
    )


def test_mating_type_and_ploidy_each_change_the_key_on_their_own() -> None:
    """BY4741 and BY4742 differ in mating type; a haploid and a diploid of one name are
    not one strain. Dropping either from the projection merges records that are not the
    same thing, which is why they are read unconditionally ON THE SUBCLASS.
    """
    common = dict(name="x", provenance=[WILDENHAIN_BY4741])
    mata = _euk(mating_type=s.MatingType.a, ploidy="haploid", **common)
    matalpha = _euk(mating_type=s.MatingType.alpha, ploidy="haploid", **common)
    diploid = _euk(mating_type=s.MatingType.a_alpha, ploidy="diploid", **common)
    assert strain_background_identity(mata) != strain_background_identity(matalpha)
    assert strain_background_identity(mata) != strain_background_identity(diploid)


def test_an_allele_namespace_change_changes_the_key() -> None:
    """``gene_namespace`` is half of a locus: a name alone does not say whose genome it
    is a name in, which is exactly why P1 put the field on every perturbation.
    """
    yeast = _euk(
        name="x",
        mating_type=s.MatingType.a,
        ploidy="haploid",
        alleles=[_his3()],
        provenance=[WILDENHAIN_BY4741],
    )
    renamespaced = yeast.model_copy(
        update={
            "alleles": [
                _his3().model_copy(
                    update={
                        "systematic_gene_name": "BW25113_3643",
                        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
                    }
                )
            ]
        }
    )
    first = strain_background_identity(yeast)["alleles"][0]
    second = strain_background_identity(renamespaced)["alleles"][0]
    assert first["gene_namespace"] != second["gene_namespace"]
    assert identity_sha256(strain_background_identity(yeast)) != identity_sha256(
        strain_background_identity(renamespaced)
    )


def test_every_concrete_background_class_has_a_projection() -> None:
    """The ``PROJECTED_PERTURBATION_CLASSES`` guard, on the background side.

    A third subclass that landed without a branch would be projected as its base, which
    drops the fields it adds and merges two strains onto one key. It raises instead.
    """

    def subclasses(cls: type) -> set[type]:
        found = {cls}
        for child in cls.__subclasses__():
            found |= subclasses(child)
        return found

    assert subclasses(s.StrainBackground) == set(PROJECTED_BACKGROUND_CLASSES)


def test_an_unprojectable_background_raises_rather_than_falling_back() -> None:
    class _Mosaic(s.StrainBackground):
        aneuploid_chromosomes: list[str] = []

    with pytest.raises(TypeError, match="no identity projection for"):
        strain_background_identity(
            _Mosaic(
                name="x",
                reference_strain="S288C",
                assembly_set=SGD_S288C_R64,
                aneuploid_chromosomes=["chrIII"],
                provenance=[WILDENHAIN_BY4741],
            )
        )


# ------------------------------------------------ the shared base (P5, 2026-10-15)


def test_the_base_states_what_both_families_state() -> None:
    """The host-agnostic background: reference strain, assembly, genotype, alleles."""
    background = _bacterial(alleles=[_rph()])
    assert background.reference_strain == "BW25113"
    assert background.assembly_set == "ecoli_K12_BW25113_ASM75055v1"
    assert background.genotype_statement == BW25113_BACKGROUND_GENOTYPE
    assert background.functional_copies("BW25113_3643") == 0
    assert background.functional_copies("BW25113_0344") == 1
    assert [a.allele_name for a in background.alleles_at("BW25113_3643")] == ["rph-1"]
    assert background.is_fully_sourced


def test_the_assembly_pin_must_be_the_reference_strain_s_own_set() -> None:
    """One cross-check for every family, where there were two one-family branches."""
    with pytest.raises(ValueError, match="is assembly set"):
        _bacterial(assembly_set="ecoli_K12_MG1655_ASM584v2")
    with pytest.raises(ValueError, match="is assembly set"):
        _euk(
            name="BY4741",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            mating_type=s.MatingType.a,
            ploidy="haploid",
            provenance=[WILDENHAIN_BY4741],
        )
    assert s.REFERENCE_STRAIN_ASSEMBLY_SETS["S288C"] == s.SGD_S288C_ASSEMBLY_SET


def test_the_reference_strain_vocabulary_is_one_member_per_deposited_set() -> None:
    """Why ``assembly_set`` is required OUTRIGHT here and value-or-gap on the genome.

    ``ReferenceGenome.strain`` is a free string and may name a strain the tier has not
    deposited, so that field needs a typed gap. Here the companion field is closed and
    the map is total over it, so a background that names its reference has named a
    deposited set and a gap branch would be unreachable.
    """
    from typing import get_args

    assert set(get_args(s.ReferenceStrain)) == set(s.REFERENCE_STRAIN_ASSEMBLY_SETS)
    # Totality in the direction the argument needs: every reference strain names a set
    # the vocabulary admits. The converse is deliberately NOT asserted, because a
    # deposited set need not be some strain's own reference; which registry sets are
    # assemblies at all is pinned by ``AssemblySet`` plus ``NON_ASSEMBLY_REGISTRY_SETS``.
    assert set(s.REFERENCE_STRAIN_ASSEMBLY_SETS.values()) <= set(
        get_args(s.AssemblySet)
    )
    assert s.StrainBackground.model_fields["assembly_set"].is_required()
    assert "assembly_set" not in str(
        s.StrainBackground.model_fields["assembly_set"].annotation
    ).replace("AssemblySet", "")
    with pytest.raises(ValueError, match="assembly_set"):
        s.StrainBackground.model_validate(
            {
                "name": "x",
                "reference_strain": "S288C",
                "assembly_set": None,
                "provenance": [WILDENHAIN_BY4741.model_dump()],
                "provenance_gaps": [_gap("assembly_set").model_dump()],
            }
        )


def test_one_allele_entry_per_locus_on_a_single_copy_background() -> None:
    """The retired bacterial class's rule, now the base's."""
    with pytest.raises(ValueError, match="one allele entry per locus"):
        _bacterial(alleles=[_rph(), _rph()])


def test_a_strain_reference_genome_admits_both_families_through_one_field() -> None:
    """One annotation, one cross-check, and the subclass survives a JSON round trip."""
    bacterial = s.StrainReferenceGenome(
        species="Escherichia coli",
        strain="BW25113",
        assembly_set="ecoli_K12_BW25113_ASM75055v1",
        assembly_accession="GCA_000750555.1",
        background=_bacterial(alleles=[_rph()]),
    )
    assert type(bacterial.background) is s.StrainBackground
    restored = s.StrainReferenceGenome.model_validate(bacterial.model_dump())
    assert type(restored.background) is s.StrainBackground
    assert restored == bacterial
    eukaryotic = s.StrainReferenceGenome(
        species=SPECIES,
        strain="BY4743",
        ploidy="diploid",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        background=_by4743(),
    )
    reloaded = s.StrainReferenceGenome.model_validate(eukaryotic.model_dump())
    assert type(reloaded.background) is s.EukaryoticStrainBackground
    assert reloaded == eukaryotic
    with pytest.raises(ValueError, match="background pins assembly set"):
        s.StrainReferenceGenome(
            species="Escherichia coli",
            strain="BW25113",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            assembly_accession="GCA_000005845.2",
            background=_bacterial(),
        )


def test_an_integrated_cassette_points_at_the_same_construct_a_gene_addition_does() -> (
    None
):
    """Review 2 finding 9: bAID's effectors at Delta were a bare label while the Btx
    cassette at XII-5 had a real pointer. One placement type serves both, required or
    typed-gapped as P4 made it on the addition leaf.
    """
    cassette = _cassette()
    assert cassette.placement is not None
    assert cassette.placement.localization is s.Localization.chromosomal_integration
    assert cassette.placement.host_placement is not None
    assert cassette.placement.host_placement.site_label == "Delta"
    assert (
        s.IntegratedCassette.model_fields["placement"].annotation
        == s.GeneAdditionPerturbation.model_fields["placement"].annotation
    )
    with pytest.raises(ValueError, match="placement is unset and carries no"):
        _cassette(placement=None)
    gapped = _cassette(
        placement=None,
        provenance_gaps=[
            ProvenanceGap(
                field="placement", reason=ProvenanceGapReason.not_released_by_primary
            )
        ],
    )
    assert gapped.placement is None and not gapped.is_sourced


def test_a_cassette_refuses_an_episomal_placement() -> None:
    """An integration is at a host site; a replicon of its own is not an integration."""
    with pytest.raises(ValueError, match="which is\s+an episomal replicon"):
        _cassette(placement=_placement(host_placement=None))


def test_strain_background_identity_ignores_who_stated_it() -> None:
    quoted = standard_background("BY4741", provenance=[WILDENHAIN_BY4741])
    pending = standard_background("BY4741", resolve_with=BRACHMANN_1998)
    assert quoted.model_dump() != pending.model_dump()
    assert strain_background_identity(quoted) == strain_background_identity(pending)
    assert set(strain_background_identity(quoted)) == set(
        EUKARYOTIC_STRAIN_BACKGROUND_IDENTITY_FIELDS
    )
    by4742 = standard_background("BY4742", resolve_with=BRACHMANN_1998)
    assert strain_background_identity(by4742) != strain_background_identity(quoted)


# ------------------------------------------------- integrated cassettes (bAID, #507)


def _cassette(**overrides: Any) -> s.IntegratedCassette:
    """BAID's CRISPR-AID cassette, sourced, with fields overridable."""
    kwargs: dict[str, Any] = {
        "name": "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
        "locus": "Delta",
        "elements": ["KanMX", "dLbCpf1-VP", "Csy4", "dSpCas9-RD1152", "SaCas9"],
        "marker": "KanMX",
        "zygosity": s.Zygosity.haploid,
        "placement": _placement(),
        "provenance": [WILDENHAIN_BY4741],
    }
    kwargs.update(overrides)
    return s.IntegratedCassette(**kwargs)


def test_an_integrated_cassette_states_its_site_elements_and_marker() -> None:
    """A cassette is an INSERTION at a named site, with its parts verbatim in order."""
    cassette = _cassette()
    assert cassette.locus == "Delta"
    assert cassette.locus_systematic_gene_name is None
    assert cassette.elements[0] == "KanMX"
    assert cassette.marker == "KanMX"
    assert cassette.mechanism_so == ("SO:0000667", "insertion")
    assert cassette.mechanism_so == s.CASSETTE_INTEGRATION_SO
    assert cassette.is_sourced


def test_a_cassette_provenance_and_zygosity_are_sourced_or_typed_gaps() -> None:
    """The ``BackgroundAllele`` contract, on the insertion axis: never a silent None."""
    with pytest.raises(ValueError, match="provenance is unset"):
        _cassette(provenance=None)
    with pytest.raises(ValueError, match="zygosity is unset"):
        _cassette(zygosity=None)
    gapped = _cassette(
        provenance=None,
        zygosity=None,
        provenance_gaps=[_gap("provenance"), _gap("zygosity")],
    )
    assert gapped.gapped_fields() == {"provenance", "zygosity"}
    assert not gapped.is_sourced
    with pytest.raises(ValueError, match="is not None"):
        _cassette(provenance_gaps=[_gap("provenance")])


def test_a_cassette_refuses_an_empty_name_locus_or_element_list() -> None:
    """Each of the three stated facts must actually be stated."""
    with pytest.raises(ValueError, match="name cannot be empty"):
        _cassette(name="  ")
    with pytest.raises(ValueError, match="locus cannot be empty"):
        _cassette(locus="")
    with pytest.raises(ValueError, match="at least one non-empty element"):
        _cassette(elements=[])
    with pytest.raises(ValueError, match="at least one non-empty element"):
        _cassette(elements=["KanMX", " "])


def test_a_cassette_locus_given_as_an_orf_is_regex_validated() -> None:
    """A landing pad stays a free string; an ORF site is checked like any other name."""
    assert _cassette(locus="X4").locus_systematic_gene_name is None
    assert (
        _cassette(
            locus="NTH1", locus_systematic_gene_name="YDR001C"
        ).locus_systematic_gene_name
        == "YDR001C"
    )
    with pytest.raises(ValueError, match="Invalid systematic gene name"):
        _cassette(locus="NTH1", locus_systematic_gene_name="X4")


def test_a_background_carries_integrations_and_checks_their_zygosity() -> None:
    """Integrations default to empty and obey the ploidy check the alleles obey."""
    plain = standard_background("BY4742", resolve_with=BRACHMANN_1998)
    assert plain.integrations == []
    assert plain.integrations_at("Delta") == []
    with_cassette = _euk(
        name="bAID",
        parents=["BY4742"],
        mating_type=s.MatingType.alpha,
        ploidy="haploid",
        alleles=plain.alleles,
        integrations=[_cassette()],
        provenance=[WILDENHAIN_BY4741],
    )
    assert [c.name for c in with_cassette.integrations_at("Delta")] == [
        "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]"
    ]
    assert not with_cassette.is_fully_sourced  # the BY alleles are pending review
    with pytest.raises(ValueError, match="does not fit a haploid background"):
        _euk(
            name="bAID",
            mating_type=s.MatingType.alpha,
            ploidy="haploid",
            integrations=[_cassette(zygosity=s.Zygosity.homozygous)],
            provenance=[WILDENHAIN_BY4741],
        )


def test_is_fully_sourced_reads_the_integrations_too() -> None:
    """A gapped cassette makes the background not fully sourced, like a gapped allele."""
    sourced = _euk(
        name="bAID",
        mating_type=s.MatingType.alpha,
        ploidy="haploid",
        integrations=[_cassette()],
        provenance=[WILDENHAIN_BY4741],
    )
    assert sourced.is_fully_sourced
    gapped = _euk(
        name="bAID",
        mating_type=s.MatingType.alpha,
        ploidy="haploid",
        integrations=[_cassette(provenance=None, provenance_gaps=[_gap("provenance")])],
        provenance=[WILDENHAIN_BY4741],
    )
    assert not gapped.is_fully_sourced


def test_two_strains_differing_only_in_an_integration_are_not_the_same_strain() -> None:
    """The cross-dataset identity projection reads integrations as genome content."""
    base = _euk(
        name="bAID",
        mating_type=s.MatingType.alpha,
        ploidy="haploid",
        integrations=[_cassette()],
        provenance=[WILDENHAIN_BY4741],
    )
    extra = _cassette(name="X3::SIZ1i", locus="X3", elements=["SIZ1i"])
    engineered = base.model_copy(update={"integrations": [*base.integrations, extra]})
    assert set(strain_background_identity(base)) == set(
        EUKARYOTIC_STRAIN_BACKGROUND_IDENTITY_FIELDS
    )
    assert strain_background_identity(base) != strain_background_identity(engineered)
    # Who stated it is still dropped: a gapped cassette and a quoted one project equal.
    gapped = base.model_copy(
        update={
            "integrations": [
                _cassette(provenance=None, provenance_gaps=[_gap("provenance")])
            ]
        }
    )
    assert strain_background_identity(base) == strain_background_identity(gapped)


# ------------------------------------------------------- baid_background (shared host)


def test_baid_background_is_by4742_plus_the_delta_site_crispr_aid_cassette() -> None:
    """The shared bAID host: four BY4742 alleles and one integration at Delta.

    One helper, because two datasets were run in this strain: Lian 2019's MAGIC screen
    and the in-house Bioscreen dataset (thesis strain BY4742-iAID6). They join on this
    object, so its name, parents, alleles and cassette are pinned here.
    """
    background = baid_background()
    assert background.name == BAID_STRAIN == "bAID"
    assert background.parents == ["BY4742"]
    assert background.reference_strain == "S288C"
    assert background.ploidy == "haploid"
    assert background.mating_type is s.MatingType.alpha
    assert background.construction == BAID_CONSTRUCTION.quote
    assert {allele.allele_name for allele in background.alleles} == {
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "ura3Δ0",
    }
    assert len(background.alleles) == 4
    # Which alleles BY4742 carries is sourced (the Lian SI strain table); how each was
    # made is only in Brachmann 1998, which is not mirrored, so each allele is asserted
    # with a pending-review gap on provenance.
    for allele in background.alleles:
        assert allele.zygosity is s.Zygosity.haploid
        assert allele.gapped_fields() == {"provenance"}
    assert len(background.integrations) == 1
    cassette = background.integrations[0]
    assert cassette.name == "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]"
    assert cassette.locus == "Delta"
    assert cassette.locus_systematic_gene_name is None
    assert cassette.elements == [
        "KanMX",
        "dLbCpf1-VP",
        "Csy4",
        "dSpCas9-RD1152",
        "SaCas9",
    ]
    assert cassette.marker == "KanMX"
    assert cassette.zygosity is s.Zygosity.haploid
    # Lian 2019 names pAID6 and deposits no map of it, so the pointer is a typed gap.
    assert cassette.placement is None
    assert cassette.gapped_fields() == {"placement"}
    assert not cassette.is_sourced
    assert cassette.provenance == [BAID_GENOTYPE, BAID_CONSTRUCTION]


def test_baid_background_round_trips_through_json_unchanged() -> None:
    """It is embedded in stored records, so it has to survive the JSON round trip."""
    background = baid_background()
    restored = s.EukaryoticStrainBackground.model_validate_json(
        background.model_dump_json()
    )
    assert restored == background
    assert restored.model_dump() == background.model_dump()
    assert json.loads(background.model_dump_json()) == json.loads(
        restored.model_dump_json()
    )

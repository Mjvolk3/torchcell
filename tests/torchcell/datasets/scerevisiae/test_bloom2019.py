# tests/torchcell/datasets/scerevisiae/test_bloom2019.py
"""Bloom 2019 loader: marker parsing, block encoding round trip, the 38-condition table.

Synthetic-marker tests need no data. The one mirror-backed test (cross A end to end)
runs only when the raw mirror is on disk, and is the same check the L2 verifier makes.
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pytest

from torchcell.datamodels.schema import (
    ArtifactRef,
    EnvironmentPhysicalPerturbation,
    MeasurementType,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import bloom2019 as b


def test_parse_marker_handles_multiallelic_and_indels() -> None:
    m = b.parse_marker("chrI_693_T_A,G_7", 0)
    assert (m.chromosome, m.position, m.ref, m.alt, m.index) == (
        "chrI",
        693,
        "T",
        "A,G",
        7,
    )
    indel = b.parse_marker("chrI_770_TA_T_5", 3)
    assert (indel.ref, indel.alt, indel.column) == ("TA", "T", 3)
    with pytest.raises(ValueError):
        b.parse_marker("chrI_693_T", 0)


def test_sorted_markers_is_a_permutation_by_chromosome_then_position() -> None:
    header = ["chrII_10_A_T_1", "chrI_900_A_T_2", "chrI_100_A_T_3", "chrI_500_A_T_4"]
    order = b.sorted_markers(header)
    assert [m.column for m in order] == [2, 3, 1, 0]
    assert [m.position for m in order] == [100, 500, 900, 10]


def test_encode_expand_round_trip_with_chromosome_breaks() -> None:
    header = [
        "chrI_100_A_T_1", "chrI_200_A_T_2", "chrI_300_A_T_3", "chrI_400_A_T_4",
        "chrII_50_A_T_5", "chrII_60_A_T_6", "chrII_70_A_T_7",
    ]  # fmt: skip
    markers = b.sorted_markers(header)
    calls = np.array([1, 1, 2, 2, 2, 1, 1], dtype=np.int8)
    blocks = b.encode_blocks(calls, markers)
    assert [(x.chromosome, x.start, x.end, x.parent, x.n_markers) for x in blocks] == [
        ("chrI", 100, 200, 1, 2),
        ("chrI", 300, 400, 2, 2),
        ("chrII", 50, 50, 2, 1),
        ("chrII", 60, 70, 1, 2),
    ]
    assert np.array_equal(b.expand_blocks(blocks, markers), calls)
    # a same-parent run across a chromosome boundary is still two blocks
    calls2 = np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int8)
    assert len(b.encode_blocks(calls2, markers)) == 2
    # a tampered block fails the round trip loudly
    broken = list(blocks)
    broken[0] = broken[0].model_copy(update={"n_markers": 3})
    with pytest.raises(ValueError):
        b.expand_blocks(broken, markers)


def test_condition_table_partition_and_controls() -> None:
    conditions = b.build_conditions()
    assert len(conditions) == 38
    residual = [
        c
        for c, s in conditions.items()
        if s.measurement_type is MeasurementType.control_regression_residual
    ]
    absolute = [
        c
        for c, s in conditions.items()
        if s.measurement_type is MeasurementType.colony_size
    ]
    assert len(residual) == 36 and sorted(absolute) == ["YNB;;1", "YPD;;1"]
    # the batch digit selects the regressor plate
    assert conditions["Caffeine;15mM;2"].control_column == "YPD;;2"
    assert conditions["Zeocin;25ug/mL;3"].control_column == "YPD;;3"
    assert conditions["YNB;ph3;1"].control_column == "YNB;;1"
    # temperature is on the environment, never a perturbation
    assert (
        conditions["YPD;37;1"].temperature_c == 37.0
        and not conditions["YPD;37;1"].perturbations
    )
    ph = conditions["YNB;ph8;1"].perturbations[0]
    assert isinstance(ph, EnvironmentPhysicalPerturbation) and ph.factor.value == "pH"
    # carbon sources are media, not perturbations
    assert conditions["Galactose;;1"].media.name == "YP + 2% galactose"
    assert not conditions["Galactose;;1"].perturbations
    sorbitol = conditions["Sorbitol;1.5M;2"].perturbations[0]
    assert isinstance(sorbitol, SmallMoleculePerturbation)
    assert sorbitol.compound.name == "sorbitol"
    assert b.NOT_SERVED == {"YPD;;2", "YPD;;3"}


def test_every_stress_compound_resolves_to_a_structure_identifier() -> None:
    """No stress compound is name-only any more; ``resolved_compound`` fills or gaps it.

    Every one of the 20 dosed compounds is in the curated identity table, so the L3
    ``compound_identity`` rule (a compound with no identifier and no typed gap is
    unencodable) can no longer drop 47% of the records. Tunicamycin is the one
    ``RESOLVED_MIXTURE``: at least ten homologues, so ChEBI identifies the substance and
    no single-molecule InChIKey exists to fill.
    """
    compounds = {
        p.compound.name: p.compound
        for spec in b.build_conditions().values()
        for p in spec.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    }
    assert len(compounds) == 20
    unencodable = [
        name
        for name, c in compounds.items()
        if c.inchikey is None
        and c.chebi_id is None
        and c.pubchem_cid is None
        and not c.provenance_gaps
    ]
    assert not unencodable
    name_only = sorted(name for name, c in compounds.items() if c.inchikey is None)
    assert name_only == ["tunicamycin"]
    assert compounds["tunicamycin"].chebi_id == "CHEBI:29699"
    assert [g.field for g in compounds["tunicamycin"].provenance_gaps] == ["inchikey"]
    # a canonical name, never the loader's source label, is what reaches the graph
    assert compounds["fluconazole"].inchikey == "RFHAOTPXVQNOHP-UHFFFAOYSA-N"
    assert compounds["sodium dodecyl sulfate"].inchikey is not None


def test_every_condition_medium_is_a_shared_library_object() -> None:
    """The 14 media are library objects, so the L3 media-membership rule joins them."""
    from torchcell.datamodels.media import MEDIA_LIBRARY

    library = {m.name for m in MEDIA_LIBRARY.values()}
    used = {spec.media.name for spec in b.build_conditions().values()}
    assert len(used) == 14
    assert not used - library


def test_parent_ids_cover_every_readme_label() -> None:
    assert set(b.PARENT_PETER_ID) == set(b.PARENT_XLS_PREFIX)
    assert (
        b.PARENT_PETER_ID["273614xa"] == "MAA"
    )  # SACE_MAA in the xls, MAA in the index
    assert b.PARENT_PETER_ID["BYa"] is None


def _mirror_root() -> str | None:
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    if not data_root:
        return None
    root = osp.join(data_root, b.RAW_DIR_REL)
    return root if osp.exists(osp.join(root, "manifest.json")) else None


@pytest.mark.skipif(_mirror_root() is None, reason="Bloom 2019 raw mirror not on disk")
def test_cross_a_round_trip_against_the_release() -> None:
    root = _mirror_root()
    assert root is not None
    frame = b.read_marker_matrix(osp.join(root, "data", "genotype_A.tsv.gz"))
    markers = b.sorted_markers([str(c) for c in frame.columns])
    values = frame.to_numpy(dtype=np.int8)[:, [m.column for m in markers]]
    assert values.shape == (951, 49707)
    for calls in values[:25]:
        blocks = b.encode_blocks(calls, markers)
        assert np.array_equal(b.expand_blocks(blocks, markers), calls)
        assert sum(x.n_markers for x in blocks) == len(markers)
    info = b.read_cross_table(
        osp.join(root, "data", b.XLS_NAME), osp.join(root, "data", b.README_NAME)
    )
    assert info["375"].parent_1 == "BYa" and info["375"].parent_1_genotype.startswith(
        "BY "
    )
    assert {c: i.n_segregants_xls for c, i in info.items()} == b.EXPECTED_SEGREGANTS


# Phase 24: the condition-partition and block-coverage guards


def test_condition_partition_guard_refuses_38_residual_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_spec`` is patched to drop ``measurement_type``, so every column takes the
    residual default: 38 residual and 0 absolute, which trips the 36/2 partition guard
    (the 38-column count guard before it still passes).
    """
    real = b._spec

    def all_residual(*args: object, **kwargs: object) -> b.ConditionSpec:
        kwargs.pop("measurement_type", None)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "_spec", all_residual)
    with pytest.raises(
        ValueError, match="^expected exactly 36 residual and 2 absolute conditions$"
    ):
        b.build_conditions()


def test_expand_blocks_refuses_blocks_that_stop_short_of_the_marker_list() -> None:
    """Two chrI markers and one aligned block spanning only the first (n_markers 1):
    every block aligns, but coverage ends at 1 of 2 markers.
    """
    markers = b.sorted_markers(["chrI_100_A_T_1", "chrI_200_A_T_2"])
    blocks = b.encode_blocks(np.array([1, 2], dtype=np.int8), markers)[:1]
    with pytest.raises(ValueError, match="^blocks cover 1 of 2 markers$"):
        b.expand_blocks(blocks, markers)


# --------------------------------------------------------------------------- #
# 2026.10.07: parents point at their assemblies by ArtifactRef.
# --------------------------------------------------------------------------- #
_TARBALL = ArtifactRef(
    tier="genomes",
    key="peter2018_1011_assemblies",
    path="1011Assemblies.tar.gz",
    sha256="5" * 64,
    bytes=3_999_623_201,
)
_S288C = ArtifactRef(
    tier="genomes",
    key="sgd_S288C_R64-4-1_20230830",
    path="S288C_reference_sequence_R64-4-1_20230830.fsa",
    sha256="d" * 64,
    bytes=12_361_395,
)
_ASSEMBLIES = b.ParentAssemblyRefs(
    index={"AAA": "GENOMES_ASSEMBLED/AAA_6.re.fa"}, tarball=_TARBALL, s288c=_S288C
)


def test_a_peter_parent_serializes_the_tarball_member_ref_and_round_trips() -> None:
    """RM (Peter AAA) gets the 1011 tarball narrowed to its index member: the release's
    ``tarball::member`` form as one ref, whose tc:// string parses back given the sha256.
    """
    parent = b.build_parent("RMx", "RM genotype", _ASSEMBLIES)
    ref = parent.assembly_ref
    assert str(ref) == (
        "tc://genomes/peter2018_1011_assemblies/1011Assemblies.tar.gz"
        "#GENOMES_ASSEMBLED/AAA_6.re.fa"
    )
    assert parent.model_dump()["assembly_ref"] == {
        "tier": "genomes",
        "key": "peter2018_1011_assemblies",
        "path": "1011Assemblies.tar.gz",
        "member": "GENOMES_ASSEMBLED/AAA_6.re.fa",
        "sha256": "5" * 64,
        "bytes": 3_999_623_201,
        "media_type": None,
    }
    assert ArtifactRef.parse(str(ref), sha256=ref.sha256, bytes=ref.bytes) == ref


def test_the_by_parent_points_at_the_s288c_reference_sequence() -> None:
    parent = b.build_parent("BYa", "BY genotype", _ASSEMBLIES)
    assert parent.peter_strain_id is None
    assert parent.assembly_ref == _S288C
    assert str(parent.assembly_ref) == (
        "tc://genomes/sgd_S288C_R64-4-1_20230830/"
        "S288C_reference_sequence_R64-4-1_20230830.fsa"
    )


_GENOMES = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-genomes")


@pytest.mark.data
@pytest.mark.skipif(
    not (
        osp.isfile(osp.join(_GENOMES, "peter2018_1011_assemblies", "manifest.json"))
        and osp.isfile(
            osp.join(_GENOMES, "sgd_S288C_R64-4-1_20230830", "manifest.json")
        )
    ),
    reason="requires the Peter and SGD genomes tiers",
)
def test_parent_assembly_refs_find_both_files_in_the_tier_manifests() -> None:
    """The lookup finds the 1011 tarball at the digest the served Bloom records carried
    as ``assembly_sha256`` before 2026.10.07, and the SGD reference sequence file.
    """
    refs = b.ParentAssemblyRefs.from_tier()
    assert (refs.tarball.path, refs.tarball.sha256) == (
        "1011Assemblies.tar.gz",
        "53540d095958ae8c32509c04485f0d2d0948069c7647f828698d611899a9b4da",
    )
    assert (refs.s288c.path, refs.s288c.sha256) == (
        "S288C_reference_sequence_R64-4-1_20230830.fsa",
        "dbf065ffc3f5bbaa7554ef53c6861399eecc597fe058df5f81ba557944e3f86b",
    )
    assert set(b.PARENT_PETER_ID.values()) - {None} <= set(refs.index)

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

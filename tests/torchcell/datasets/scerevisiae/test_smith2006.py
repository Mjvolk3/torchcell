# tests/torchcell/datasets/scerevisiae/test_smith2006.py
"""Smith 2006 loader: the YEPD growth-control drop, the media identity, the ordinal map.

Everything but the mirror audit runs on a synthetic score table and a fake genome, so no
build tree and no network are needed. The audit test binds each module-level
``SourcedValue`` to its verbatim quote in the sha256-pinned mirror and skips when the
mirror is not mounted.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import cast

import pandas as pd
import pytest

from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    MeasurementType,
    PhysicalFactor,
    ResponseCategory,
)
from torchcell.datasets.scerevisiae import smith2006 as s
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

_GENES = {"YAL001C", "YAL002W", "YBR001C", "YBR002W"}
_STANDARD = {"TFC3": ["YAL001C"], "VPS8": ["YAL002W"], "SPO23": ["YBR001C"]}


class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` this loader's gene-name policy reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}
    gene_attribute_table = pd.DataFrame({"ID": sorted(_GENES)})
    alias_to_systematic: dict[str, list[str]] = {"YAL003W": ["YAL002W"]}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        for standard, ids in _STANDARD.items():
            if upper == standard:
                return _Resolution("renamed", ids[0])
        return _Resolution("retired", None)


def _table() -> pd.DataFrame:
    """Four strains: one clean, one NG, one LG, one unresolvable systematic name."""
    return pd.DataFrame(
        {
            s.SYSTEMATIC_COL: ["YAL001C", "YAL002W", "YBR001C", "YZZ999W"],
            s.STANDARD_COL: ["TFC3", "VPS8", "SPO23", "GHOST"],
            s.YEPD_QC_COL: [float("nan"), "NG", "LG", float("nan")],
            "Oleate (YPBO)": [3, 1, 1, 4],
            "Myristae (YPBM)": [4, 1, 2, 3],
            "Acetate (YPBA)": [3.0, 1.0, 2.5, 2.0],
        }
    )


def test_yepd_flagged_strains_are_dropped_and_counted() -> None:
    strains = s.resolve_strains(_table(), cast(SCerevisiaeGenome, _FakeGenome()))
    assert strains.orf == ["YAL001C"]
    assert strains.dropped_yepd_qc == ["YAL002W", "YBR001C"]
    assert strains.dropped_unresolved == ["YZZ999W"]


def test_common_name_comes_from_the_genome_not_the_source_column() -> None:
    strains = s.resolve_strains(_table(), cast(SCerevisiaeGenome, _FakeGenome()))
    assert strains.common == ["TFC3"]
    # ...and it round-trips through the resolver, which is the L1 canonical-name rule.
    assert _FakeGenome().resolve_gene_name("TFC3").systematic_name == "YAL001C"


def test_alias_resolution_that_would_collide_is_dropped_not_relabelled() -> None:
    table = pd.DataFrame(
        {
            s.SYSTEMATIC_COL: ["YAL002W", "YAL003W"],
            s.STANDARD_COL: ["VPS8", "OLD"],
            s.YEPD_QC_COL: [float("nan"), float("nan")],
            "Oleate (YPBO)": [3, 3],
            "Myristae (YPBM)": [3, 3],
            "Acetate (YPBA)": [3.0, 3.0],
        }
    )
    strains = s.resolve_strains(table, cast(SCerevisiaeGenome, _FakeGenome()))
    assert strains.orf == ["YAL002W"]
    assert strains.dropped_alias_collision == ["YAL003W"]


def test_every_condition_uses_a_shared_media_library_object() -> None:
    for spec in s.CONDITION_SPECS:
        assert spec["media"] in MEDIA_LIBRARY.values()
        assert spec["media"].base_medium in MEDIA_LIBRARY


def test_the_carbon_source_is_a_typed_factor_not_a_stress_compound() -> None:
    dataset = s.FattyAcidSmith2006Dataset.__new__(s.FattyAcidSmith2006Dataset)
    for spec in s.CONDITION_SPECS:
        environment = s.FattyAcidSmith2006Dataset._environment(dataset, spec)
        (perturbation,) = environment.perturbations
        assert isinstance(perturbation, EnvironmentPhysicalPerturbation)
        assert perturbation.factor is PhysicalFactor.carbon_source
        assert perturbation.magnitude is not None
        assert perturbation.magnitude.unit is ConcentrationUnit.percent_w_v
        assert perturbation.magnitude.value == spec["carbon_percent"]
        # the compound identity is resolved, never a bare name
        assert perturbation.agent is not None
        assert perturbation.agent.inchikey is not None
        # ...and the medium's own recipe already carries that carbon source
        carbon = [
            component
            for component in spec["media"].components
            if component.compound.name == perturbation.agent.name
        ]
        assert len(carbon) == 1


def test_duration_is_the_conservative_end_of_the_published_range() -> None:
    assert s.DURATION_HOURS.value == 72.0
    assert "3-4 days" in s.DURATION_HOURS.quote


def test_ordinal_scores_map_onto_the_shared_response_category_axis() -> None:
    assert s.CLEAR_ZONE_CATEGORY[4.0] == (ResponseCategory.enhanced, "enhanced")
    assert s.CLEAR_ZONE_CATEGORY[3.0] == (ResponseCategory.no_change, "wild_type")
    assert s.CLEAR_ZONE_CATEGORY[2.0] == (ResponseCategory.reduced, "reduced")
    assert s.CLEAR_ZONE_CATEGORY[1.0] == (
        ResponseCategory.severely_reduced,
        "defective",
    )
    assert s.GROWTH_CATEGORY[2.5] == (ResponseCategory.mildly_reduced, "intermediate")
    assert s.GROWTH_CATEGORY[2.0] == (ResponseCategory.reduced, "moderate")
    assert s.GROWTH_CATEGORY[1.0] == (ResponseCategory.severely_reduced, "poor")


def test_phenotype_keeps_the_verbatim_ordinal_and_the_source_word() -> None:
    dataset = s.FattyAcidSmith2006Dataset.__new__(s.FattyAcidSmith2006Dataset)
    spec = s.CONDITION_SPECS[2]  # acetate
    phenotype = s.FattyAcidSmith2006Dataset._phenotype(dataset, spec, 2.5)
    assert phenotype.measurement_type is MeasurementType.ordinal
    assert phenotype.environment_response == 2.5
    assert phenotype.category is ResponseCategory.mildly_reduced
    assert phenotype.category_label == "intermediate"
    assert phenotype.assay_type is AssayType.colony_size_array
    assert phenotype.n_samples == 3


def test_clear_zone_and_acetate_carry_different_assay_types() -> None:
    by_column = {spec["column"]: spec["assay_type"] for spec in s.CONDITION_SPECS}
    assert by_column["Oleate (YPBO)"] is AssayType.halo_zone
    assert by_column["Myristae (YPBM)"] is AssayType.halo_zone
    assert by_column["Acetate (YPBA)"] is AssayType.colony_size_array


def test_an_unmapped_score_raises_instead_of_being_guessed() -> None:
    dataset = s.FattyAcidSmith2006Dataset.__new__(s.FattyAcidSmith2006Dataset)
    with pytest.raises(RuntimeError, match="unmapped ordinal score"):
        s.FattyAcidSmith2006Dataset._phenotype(dataset, s.CONDITION_SPECS[0], 5.0)


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    values = [value for value in vars(s).values() if isinstance(value, SourcedValue)]
    assert values, "the loader must carry module-level SourcedValues"
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"

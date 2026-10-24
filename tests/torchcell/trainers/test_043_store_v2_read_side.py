# tests/torchcell/trainers/test_043_store_v2_read_side.py
"""The 043 read side of store v2: released symbols resolve, off-index genes are gaps.

Store v2 admits every served record (no gene-set clause in the query). Two record shapes
the v1 query dropped reach ``dataset_043`` and ``representation_043`` now: a deletion
stored as ``ReleasedIdentifierPerturbation`` (the released symbol in
``systematic_gene_name``) and a deletion of an ORF outside the 6,607-gene index. The
first must resolve to its ORF through the standard-name map; the second must become an
``off_index_gene`` token with gap ``NO_S288C_KEY``, never a drop and never a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "043-expression-proteome-morphology"
    / "scripts"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dataset_043 as d043  # type: ignore[import-not-found]  # noqa: E402
import representation_043 as rep  # type: ignore[import-not-found]  # noqa: E402
import store_version_043 as sv  # type: ignore[import-not-found]  # noqa: E402

from torchcell.data.representation_043 import (  # noqa: E402
    ContextTables,
    GapKind,
    GenePositions,
    RepresentationConfig,
    RepresentationResolver,
)
from torchcell.models.genotype_tokens_043 import (  # noqa: E402
    CHANGE_TYPE_INDEX,
    ChangeType,
)

GENES = ["YAL001C", "YDR040C", "YGR098C"]
DEL = CHANGE_TYPE_INDEX[ChangeType.deletion]


def _dataset() -> d043.Raw043Dataset:
    """A dataset with only the attributes ``_gene_idx`` and ``_mark_perturbations`` read."""
    ds = d043.Raw043Dataset.__new__(d043.Raw043Dataset)
    ds.config = d043.Dataset043Config(data_root="/unused")
    ds.genes = GENES
    ds.gene_to_idx = {g: i for i, g in enumerate(GENES)}
    ds.name2orf = {"ENA1": "YDR040C", "ESP1": "YGR098C"}
    return ds


def _resolver() -> RepresentationResolver:
    g = len(GENES)
    key_ids = np.arange(g * 4, dtype=np.int32).reshape(1, g, 4)
    tables = ContextTables(
        context_ids=("S288C",),
        key_ids=key_ids,
        key_gap=np.zeros((1, g, 4), dtype=np.int8),
        gene_copy=np.ones((1, g), dtype=np.float32),
        state=np.zeros((1, g), dtype=np.int8),
        accessory={},
    )
    positions = GenePositions.build(
        [("chrI", 10, 20), ("chrIV", 30, 40), ("chrVII", 5, 15)],
        {"chrI": 100, "chrIV": 100, "chrVII": 100},
    )
    return RepresentationResolver(
        RepresentationConfig(route="A"), positions, tables, None, ["BY", "RM"]
    )


def _released(name: str) -> dict[str, object]:
    return {
        "perturbation_type": "released_identifier",
        "systematic_gene_name": name,
        "identifier_kind": "gene symbol",
        "released_kind": "deletion",
    }


def test_released_symbol_resolves_to_its_orf() -> None:
    ds = _dataset()
    gene_index, pert_type, outside, accessory, promoter = ds._mark_perturbations(
        [_released("ENA1")], is_isolate=False
    )
    assert (gene_index, pert_type, outside, accessory, promoter) == (
        [1],
        [int(d043.PertType.DELETION)],
        0,
        0,
        None,
    )
    r = rep.resolve_genotype(
        _resolver(), {"perturbations": [_released("ENA1")]}, ds._gene_idx
    )
    assert r.genotype_class == "deletion"
    assert r.marked.tolist() == [1]
    assert r.change_type[1] == DEL
    assert r.extra_tokens == () and r.n_marked_outside_index == 0


def test_released_kind_without_a_change_type_raises() -> None:
    ds = _dataset()
    tagged = {**_released("ENA1"), "released_kind": "affinity_tag"}
    with pytest.raises(ValueError, match="affinity_tag"):
        ds._mark_perturbations([tagged], is_isolate=False)
    with pytest.raises(ValueError, match="affinity_tag"):
        rep.resolve_genotype(_resolver(), {"perturbations": [tagged]}, ds._gene_idx)


@pytest.mark.parametrize(
    "perturbation",
    [
        # an ORF outside the index (the v1 Kemmeren / Hu / Ohya / Messner / Nadal shape)
        {"perturbation_type": "kanmx_deletion", "systematic_gene_name": "YER109C"},
        # a released symbol with no standard-name owner in the index
        _released("NOTAGENE1"),
    ],
)
def test_off_index_gene_is_a_typed_gap_not_a_drop(
    perturbation: dict[str, object],
) -> None:
    ds = _dataset()
    name = perturbation["systematic_gene_name"]
    genotype = {
        "perturbations": [
            perturbation,
            {"perturbation_type": "kanmx_deletion", "systematic_gene_name": "YAL001C"},
        ]
    }
    gene_index, _, outside, _, _ = ds._mark_perturbations(
        genotype["perturbations"], is_isolate=False
    )
    assert gene_index == [0] and outside == 1
    r = rep.resolve_genotype(_resolver(), genotype, ds._gene_idx)
    assert r.genotype_class == "deletion"
    assert r.marked.tolist() == [0]
    assert r.n_marked_outside_index == 1
    (token,) = r.extra_tokens
    assert token.kind == "off_index_gene"
    assert token.feature_id == name
    assert token.change_type == DEL
    assert token.key_ids == (-1, -1, -1, -1)
    assert token.gaps == (int(GapKind.NO_S288C_KEY),) * 4


def test_single_off_index_deletion_keeps_the_record() -> None:
    ds = _dataset()
    genotype = {
        "perturbations": [
            {"perturbation_type": "marker_deletion", "systematic_gene_name": "YKL200C"}
        ]
    }
    r = rep.resolve_genotype(_resolver(), genotype, ds._gene_idx)
    assert r.genotype_class == "deletion"
    assert not r.operator_mask.any()
    assert [t.feature_id for t in r.extra_tokens] == ["YKL200C"]


def test_store_versions_name_disjoint_paths() -> None:
    v1, v2 = sv.store_paths("v1"), sv.store_paths("v2")
    assert (v1.raw_tag, v1.representation_tag, v1.split_version, v1.prep_version) == (
        "raw_043",
        "representation_043",
        "global_split_v1",
        "prep_v2",
    )
    assert (v2.raw_tag, v2.representation_tag, v2.split_version, v2.prep_version) == (
        "raw_043_v2",
        "representation_043_v2",
        "global_split_v2",
        "prep_v3",
    )
    assert v1.result_name("linear_evals_043") == "linear_evals_043.json"
    assert v2.result_name("linear_evals_043") == "linear_evals_043_v2.json"
    assert v2.result_name("verify_representation_043", "md") == (
        "verify_representation_043_v2.md"
    )
    with pytest.raises(ValueError, match="v3"):
        sv.store_paths("v3")


def test_active_store_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sv.ENV_VAR, raising=False)
    assert sv.active().version == "v1"
    monkeypatch.setenv(sv.ENV_VAR, "v2")
    assert sv.active().raw_tag == "raw_043_v2"
    assert d043.Dataset043Config(data_root="/d").store_root == (
        "/d/data/torchcell/experiments/043-expression-proteome-morphology/raw_043_v2"
    )
    assert str(rep.rep_root("/d")).endswith("representation_043_v2")

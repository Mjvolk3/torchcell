# tests/torchcell/datasets/scerevisiae/test_sgd.py
# [[tests.torchcell.datasets.scerevisiae.test_sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_sgd.py
"""The SGD essentiality provenance record: the decision, and the measurement behind it.

Synthetic tests run everywhere. They write per-gene JSON files from scratch in the
shape SGD's API serves, including a counterfactual annotation that DOES state a medium
with its physical state, so the storability rule is shown to separate the two rather
than merely to return 0 on the real bytes.

The ``@pytest.mark.data`` tests read the pinned dump: they re-derive its digest and
re-measure every count the decision rests on.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest

import torchcell.datasets.scerevisiae.sgd as m

DATA_ROOT = os.environ.get("DATA_ROOT")

#: What the pinned dump holds, re-measured by the ``data`` tests.
REAL_ANNOTATIONS = 1329
REAL_WITHOUT_CONDITION = 1299
REAL_WHOLE_GENE_DELETION = 1287
REAL_OTHER_ALLELE = 42
REAL_COLLAPSE_KEYS = 9
REAL_COLLAPSE_EXTRA_RECORDS = 24
REAL_CONDITION_PAIRS = {
    "media: 2% glucose": 1,
    "media: YPD with 2% glucose": 1,
    "media: nitrogen starvation": 27,
    "temperature: elevated temperature, 37 °C": 1,
}
REAL_EXPERIMENT_TYPES = {"classical genetics": 208, "systematic mutation set": 1121}


# --------------------------------------------------------------------------- #
# Synthetic SGD per-gene JSON
# --------------------------------------------------------------------------- #
def _annotation(
    annotation_id: int,
    locus: str,
    allele: str,
    pubmed_id: str,
    *,
    conditions: tuple[tuple[str, str], ...] = (),
    mutant_type: str = "null",
    strain: str = "S288C",
    phenotype: str = "inviable",
    experiment: str = "systematic mutation set",
    note: str | None = None,
) -> dict[str, Any]:
    properties: list[dict[str, Any]] = [
        {
            "class_type": "BIOITEM",
            "role": "Allele",
            "bioitem": {"display_name": allele},
            "note": None,
        }
    ]
    properties += [
        {"class_type": kind, "role": None, "note": text} for kind, text in conditions
    ]
    return {
        "id": annotation_id,
        "mutant_type": mutant_type,
        "strain": {"display_name": strain},
        "phenotype": {"display_name": phenotype},
        "locus": {"display_name": locus},
        "reference": {"pubmed_id": pubmed_id},
        "experiment": {"display_name": experiment},
        "properties": properties,
        "note": note,
    }


def _write_gene(genes_dir: Path, orf: str, details: list[dict[str, Any]]) -> None:
    genes_dir.mkdir(parents=True, exist_ok=True)
    (genes_dir / f"{orf}.json").write_text(
        json.dumps({"phenotype_details": details}), encoding="utf-8"
    )


@pytest.fixture
def genes_dir(tmp_path: Path) -> Path:
    """Four genes: a plain deletion, two SAC1 alleles under one paper, a point
    allele, and a viable annotation the selector must not read.
    """
    out = tmp_path / "genes"
    _write_gene(out, "YAL001C", [_annotation(1, "TFC3", "tfc3-Δ", "111")])
    _write_gene(
        out,
        "YKL212W",
        [
            _annotation(
                2,
                "SAC1",
                "sac1-Δ",
                "222",
                conditions=(("media", "nitrogen starvation"),),
            ),
            _annotation(
                3,
                "SAC1",
                "sac1Δ114-503",
                "222",
                conditions=(("media", "nitrogen starvation"),),
            ),
        ],
    )
    _write_gene(out, "YBR001C", [_annotation(4, "NTH2", "nth2-111", "333")])
    _write_gene(
        out, "YCR001W", [_annotation(5, "DAN4", "dan4-Δ", "444", phenotype="viable")]
    )
    return out


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_the_module_registers_no_dataset() -> None:
    """No dataset class: the source states no medium, so no record can be built."""
    from torchcell.datasets.dataset_registry import dataset_registry

    assert not any(cls.__module__ == m.__name__ for cls in dataset_registry.values())
    assert "GeneEssentialitySgdDataset" not in dataset_registry


def test_the_removed_class_is_not_importable_from_the_dataset_package() -> None:
    """The served class is gone from the package namespace, not merely unregistered."""
    import torchcell.datasets as datasets

    assert not hasattr(datasets, "GeneEssentialitySgdDataset")
    assert "GeneEssentialitySgdDataset" not in datasets.__all__


def test_the_class_is_not_in_the_public_adapter_map() -> None:
    """Nothing in the KG build can reach the dataset by class or by adapter."""
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    names = {cls.__name__ for cls in dataset_adapter_map}
    assert "GeneEssentialitySgdDataset" not in names
    assert not any(
        adapter.__name__ == "GeneEssentialitySgdAdapter"
        for adapter in dataset_adapter_map.values()
    )


def test_the_condition_pairs_are_the_four_the_dump_carries() -> None:
    """Each recorded pair says what it is, and none of them is a carryable medium."""
    assert set(m.SGD_CONDITION_PAIRS) == {
        ("media", "nitrogen starvation"),
        ("media", "2% glucose"),
        ("media", "YPD with 2% glucose"),
        ("temperature", "elevated temperature, 37 °C"),
    }
    assert all(reason for reason in m.SGD_CONDITION_PAIRS.values())


def test_the_dump_provenance_pins_the_digest_it_was_measured_on() -> None:
    """Every count here is stated of these bytes, so the pin travels with them."""
    assert m.SGD_DUMP.sha256 == m.SGD_GENES_DIGEST
    assert m.SGD_DUMP.citation_key == "cherrySGDSaccharomycesGenome1998"
    assert m.ANNOTATION_SELECTOR == {
        "mutant_type": "null",
        "strain": "S288C",
        "phenotype": "inviable",
    }


def test_the_two_disagreement_groups_are_disjoint_and_add_to_the_measurement() -> None:
    """16 conditional plus 7 unconditional are the 23 that survive the allele rule."""
    conditional = set(m.CONDITIONAL_DISAGREEMENTS)
    unconditional = set(m.UNCONDITIONAL_DISAGREEMENTS)
    assert not conditional & unconditional
    assert len(conditional) == 16
    assert len(unconditional) == 7
    assert m.VIABLE_FULL_DELETIONS_SMF_COSTANZO2016 == 4581


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def test_only_the_selected_annotations_are_read(genes_dir: Path) -> None:
    """A viable phenotype is not an essentiality call and is not read."""
    annotations = m.inviable_null_annotations(str(genes_dir), progress=False)
    assert [a.sgd_annotation_id for a in annotations] == [1, 4, 2, 3]
    assert [a.systematic_name for a in annotations] == [
        "YAL001C",
        "YBR001C",
        "YKL212W",
        "YKL212W",
    ]


def test_an_annotation_with_two_alleles_is_refused(tmp_path: Path) -> None:
    """One annotation names one allele; two is not a record with a chosen one."""
    detail = _annotation(9, "TFC3", "tfc3-Δ", "111")
    detail["properties"].append(
        {
            "class_type": "BIOITEM",
            "role": "Allele",
            "bioitem": {"display_name": "tfc3-1"},
            "note": None,
        }
    )
    _write_gene(tmp_path / "genes", "YAL001C", [detail])
    with pytest.raises(m.SgdAlleleError, match="2 alleles"):
        m.inviable_null_annotations(str(tmp_path / "genes"), progress=False)


@pytest.mark.parametrize(
    ("allele", "locus", "expected"),
    [
        ("tfc3-Δ", "TFC3", True),
        ("TFC3-Δ", "tfc3", True),
        ("yil102c-A-∆", "YIL102C-A", True),
        ("sac1Δ114-503", "SAC1", False),
        ("cdc42-111", "CDC42", False),
        ("Nog1-(1-426)", "NOG1", False),
        ("cdc24p-∆CH", "CDC24", False),
    ],
)
def test_whole_gene_deletion_folds_both_signs_and_case(
    allele: str, locus: str, expected: bool
) -> None:
    """U+2206 and U+0394 are the same name; anything after the sign is not."""
    assert m.is_whole_gene_deletion(allele, locus) is expected


def test_no_annotation_states_a_medium_the_schema_can_carry(genes_dir: Path) -> None:
    """Clause 1 on synthetic bytes: a nutrient edit is not a medium."""
    annotations = m.inviable_null_annotations(str(genes_dir), progress=False)
    inventory = m.condition_inventory(annotations)
    assert inventory.n_annotations == 4
    assert inventory.n_without_condition == 2
    assert inventory.pairs == {"media: nitrogen starvation": 2}
    assert inventory.n_naming_a_medium == 0
    assert inventory.n_stating_a_physical_state == 0
    assert inventory.n_storable == 0


def test_a_medium_with_a_state_would_be_counted_storable(tmp_path: Path) -> None:
    """The counterfactual: the rule is not hard-wired to 0."""
    _write_gene(
        tmp_path / "genes",
        "YAL001C",
        [
            _annotation(
                1, "TFC3", "tfc3-Δ", "111", conditions=(("media", "YPD solid plates"),)
            )
        ],
    )
    annotations = m.inviable_null_annotations(str(tmp_path / "genes"), progress=False)
    inventory = m.condition_inventory(annotations)
    assert inventory.n_naming_a_medium == 1
    assert inventory.n_stating_a_physical_state == 1
    assert inventory.n_storable == 1


def test_the_one_real_medium_names_no_physical_state() -> None:
    """The one real medium names no state Media.state could be read from.

    SGD writes it ``YPD with 2% glucose``: a medium name and a carbon
    concentration, and nothing about whether the cells were on a plate or in
    broth.
    """
    conditions = (("media", "YPD with 2% glucose"),)
    assert m.names_a_medium(conditions) is True
    assert m.states_a_physical_state(conditions) is False


def test_the_allele_inventory_counts_the_collapse_the_served_store_held(
    genes_dir: Path,
) -> None:
    """Clause 3 on synthetic bytes: two SAC1 alleles under one paper were one record."""
    annotations = m.inviable_null_annotations(str(genes_dir), progress=False)
    inventory = m.allele_inventory(annotations)
    assert inventory.n_annotations == 4
    assert inventory.n_whole_gene_deletion == 2
    assert inventory.n_other_allele == 2
    assert inventory.collapse_keys == {"YKL212W PMID 222": ["sac1-Δ", "sac1Δ114-503"]}
    assert inventory.n_collapse_extra_records == 1


def test_the_experiment_inventory_is_the_primary_layer(genes_dir: Path) -> None:
    """The shape of what a per-study loader would have to read instead."""
    annotations = m.inviable_null_annotations(str(genes_dir), progress=False)
    assert m.experiment_type_inventory(annotations) == {"systematic mutation set": 4}


def test_a_missing_dump_is_refused_by_name(tmp_path: Path) -> None:
    """No download path: a fresh fetch is a new release with its own digest."""
    with pytest.raises(m.SgdDumpDigestError, match="does not exist"):
        m.verify_sgd_dump(str(tmp_path / "absent"))


def test_bytes_off_the_pin_are_refused(genes_dir: Path) -> None:
    """The measurements are stated of the pinned bytes, so other bytes raise."""
    with pytest.raises(m.SgdDumpDigestError, match="is not the pinned"):
        m.verify_sgd_dump(str(genes_dir))


def test_the_digest_changes_with_any_gene_file(genes_dir: Path) -> None:
    """One sha256 over the sorted per-file lines, so any edit moves it."""
    before = m.sgd_genes_digest(str(genes_dir))
    _write_gene(genes_dir, "YZZ999W", [])
    assert m.sgd_genes_digest(str(genes_dir)) != before


# --------------------------------------------------------------------------- #
# The pinned dump
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT unset")
def test_the_real_dump_matches_the_pin_and_every_recorded_count() -> None:
    """Every number of the decision, re-measured on the pinned dump."""
    assert DATA_ROOT is not None
    real = osp.join(DATA_ROOT, m.SGD_GENES_RELPATH)
    if not osp.isdir(real):
        pytest.skip(f"{real} not on this machine")
    m.verify_sgd_dump(real)
    annotations = m.inviable_null_annotations(real, progress=False)
    assert len(annotations) == REAL_ANNOTATIONS

    conditions = m.condition_inventory(annotations)
    assert conditions.n_without_condition == REAL_WITHOUT_CONDITION
    assert conditions.pairs == REAL_CONDITION_PAIRS
    assert conditions.n_naming_a_medium == 1
    assert conditions.n_stating_a_physical_state == 0
    assert conditions.n_storable == 0

    alleles = m.allele_inventory(annotations)
    assert alleles.n_whole_gene_deletion == REAL_WHOLE_GENE_DELETION
    assert alleles.n_other_allele == REAL_OTHER_ALLELE
    assert len(alleles.collapse_keys) == REAL_COLLAPSE_KEYS
    assert alleles.n_collapse_extra_records == REAL_COLLAPSE_EXTRA_RECORDS

    assert m.experiment_type_inventory(annotations) == REAL_EXPERIMENT_TYPES

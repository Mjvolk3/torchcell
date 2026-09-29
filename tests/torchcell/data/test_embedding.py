# tests/torchcell/data/test_embedding.py
# [[tests.torchcell.data.test_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_embedding.py
"""``BaseEmbeddingDataset``: model-name validation, id lookup, and dataset addition.

Fixture: ``_Toy``, a concrete subclass whose ``MODEL_TO_WINDOW`` names two models and
whose ``process`` collates a class-level item list into ``<root>/processed/<model>.pt``.
Items are ``Data(id, dna_windows={model: window}, embeddings={model: (1, 2) tensor})``:
dataset "toy" holds g1 ("ACGT", [1, 2]) and g2 ("TTTT", [3, 4]); dataset "other" holds
g2 ("GGGG", [5, 6]) and g1 ("CCCC", [7, 8]); dataset "third" holds g2 and g3.
Embeddings load onto ``self.device`` (CUDA when present), so values are compared
through ``.cpu().tolist()``.

Addition merges per gene id: ``toy + other`` is a ``CombinedEmbedding`` at ``toy``'s
root, ids in ``toy``'s order [g1, g2], with g1's windows {toy: ACGT, other: CCCC} and
embeddings {toy: [[1, 2]], other: [[7, 8]]}. ``sum([...])`` reaches the same through
``__radd__(0)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from torchcell.data.embedding import BaseEmbeddingDataset, CombinedEmbedding


class _Toy(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {"toy": 4, "other": 4}
    items: list[Data] = []

    def initialize_model(self) -> None:
        return None

    def process(self) -> None:
        data, slices = self.collate(type(self).items)
        torch.save((data, slices), self.processed_paths[0])


def _item(gene: str, model: str, window: str, values: list[float]) -> Data:
    return Data(
        id=gene, dna_windows={model: window}, embeddings={model: torch.tensor([values])}
    )


def _make(root: Path, model: str, items: list[Data]) -> _Toy:
    _Toy.items = items
    return _Toy(str(root), model_name=model)


def _values(data: Any) -> dict[str, list[list[float]]]:
    return {key: value.cpu().tolist() for key, value in data.embeddings.items()}


@pytest.fixture
def toy(tmp_path: Path) -> _Toy:
    return _make(
        tmp_path / "toy",
        "toy",
        [
            _item("g1", "toy", "ACGT", [1.0, 2.0]),
            _item("g2", "toy", "TTTT", [3.0, 4.0]),
        ],
    )


@pytest.fixture
def other(tmp_path: Path) -> _Toy:
    return _make(
        tmp_path / "other",
        "other",
        [
            _item("g2", "other", "GGGG", [5.0, 6.0]),
            _item("g1", "other", "CCCC", [7.0, 8.0]),
        ],
    )


def test_invalid_model_name_is_refused_before_any_directory_is_made(
    tmp_path: Path,
) -> None:
    """Finding: the two message literals (embedding.py lines 30-31) join with no space,
    so it reads "Invalid model_name 'bad'.Valid options are: toy, other". The root is not
    created, since the check precedes ``InMemoryDataset.__init__``.
    """
    root = tmp_path / "never"
    message = "Invalid model_name 'bad'.Valid options are: toy, other"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _Toy(str(root), model_name="bad")
    assert not root.exists()


def test_model_dataset_processes_once_and_indexes_by_position_or_gene_id(
    toy: _Toy, tmp_path: Path
) -> None:
    """The processed file is ``<root>/processed/toy.pt``; there are no raw files; ids
    are [g1, g2]; integer indexing returns the stored item, string indexing returns a
    fresh ``Data`` with that gene's window and a (1, 2) embedding slice; an unknown gene
    is a ``KeyError`` naming it. A second dataset on the same root reuses the processed
    file: with the item list emptied, ``process`` (which would collate nothing) is not
    run again and the ids are still [g1, g2].
    """
    _Toy.items = []
    again = _Toy(str(tmp_path / "toy"), model_name="toy")
    assert again._data.id == ["g1", "g2"]
    assert _values(again[1]) == {"toy": [[3.0, 4.0]]}
    assert toy.processed_file_names == "toy.pt"
    assert toy.raw_file_names == []
    assert (tmp_path / "toy" / "processed" / "toy.pt").is_file()
    assert toy.model_name == "toy"
    assert len(toy) == 2
    assert toy._data.id == ["g1", "g2"]
    assert [data.id for data in toy.get_data_list()] == ["g1", "g2"]
    assert toy[1].id == "g2"
    assert _values(toy[0]) == {"toy": [[1.0, 2.0]]}
    by_id = toy["g2"]
    assert by_id.id == "g2"
    assert dict(by_id.dna_windows) == {"toy": "TTTT"}
    assert _values(by_id) == {"toy": [[3.0, 4.0]]}
    assert tuple(by_id.embeddings["toy"].shape) == (1, 2)
    with pytest.raises(KeyError, match="Gene nope not found in the dataset."):
        toy["nope"]


def test_no_model_name_means_an_empty_dataset(tmp_path: Path) -> None:
    """Finding: ``processed_file_names`` formats the absent name as "None.pt"
    (embedding.py line 60), so the base class looks for ``processed/None.pt``, finds
    nothing, runs the no-op ``process`` and leaves ``data``/``slices`` None. The
    ``CombinedEmbedding`` hooks all return None.
    """
    root = tmp_path / "combined"
    combined = CombinedEmbedding(root=str(root), model_name=None)
    assert combined.processed_file_names == "None.pt"
    assert (combined._data, combined.slices, combined.model_name) == (None, None, None)
    combined.initialize_model()
    combined.process()
    combined.download()
    assert (root / "processed").is_dir()
    assert not (root / "processed" / "None.pt").exists()


def test_add_merges_disjoint_keys_per_gene_into_a_combined_dataset(
    toy: _Toy, other: _Toy
) -> None:
    """``toy + other`` is a ``CombinedEmbedding`` at toy's root with model_name None,
    ids in toy's order, each gene carrying both windows and both embeddings.

    Finding: the left operand is mutated. ``__add__`` writes the right operand's keys
    into the items it read from ``self`` (embedding.py lines 131 and 140), and those
    items are ``InMemoryDataset``'s cached separated ``Data`` objects, so afterwards
    ``toy[0].dna_windows`` also holds ``other`` while the collated store
    ``toy._data.dna_windows`` does not, and repeating ``toy + other`` fails with the
    duplicate-key error. The right operand is untouched.
    """
    combined = toy + other
    assert type(combined) is CombinedEmbedding
    assert combined.root == toy.root
    assert combined.model_name is None
    assert len(combined) == 2
    assert combined._data.id == ["g1", "g2"]
    assert dict(combined["g1"].dna_windows) == {"toy": "ACGT", "other": "CCCC"}
    assert _values(combined["g1"]) == {"toy": [[1.0, 2.0]], "other": [[7.0, 8.0]]}
    assert dict(combined["g2"].dna_windows) == {"toy": "TTTT", "other": "GGGG"}
    assert _values(combined["g2"]) == {"toy": [[3.0, 4.0]], "other": [[5.0, 6.0]]}
    assert dict(other[0].dna_windows) == {"other": "GGGG"}
    assert dict(toy[0].dna_windows) == {"toy": "ACGT", "other": "CCCC"}
    assert toy._data.dna_windows == {"toy": ["ACGT", "TTTT"]}
    with pytest.raises(
        ValueError, match=r"^Duplicate keys found in dna_windows:other, other$"
    ):
        toy + other


def test_sum_and_the_zero_identity_reach_the_same_combination(
    toy: _Toy, other: _Toy
) -> None:
    """``sum`` starts from 0, which ``__radd__`` and ``__add__`` both treat as the
    identity, so ``sum([toy, other])`` is the combined dataset with toy's id order.
    """
    assert toy.__radd__(0) is toy
    assert toy + 0 is toy
    summed = sum([toy, other])
    assert type(summed) is CombinedEmbedding
    assert summed._data.id == ["g1", "g2"]
    assert dict(summed["g2"].dna_windows) == {"toy": "TTTT", "other": "GGGG"}
    assert _values(summed["g1"]) == {"toy": [[1.0, 2.0]], "other": [[7.0, 8.0]]}


def test_add_refuses_duplicate_keys_and_foreign_operands(toy: _Toy) -> None:
    """Finding: the duplicate-key message (embedding.py lines 148-151) has no space
    after the colon and repeats the key once per record that carried it, so adding a
    two-record dataset to itself reads "Duplicate keys found in dna_windows:toy, toy".
    A non-dataset operand, through ``+`` or ``__radd__``, is refused by type.
    """
    with pytest.raises(
        ValueError, match=r"^Duplicate keys found in dna_windows:toy, toy$"
    ):
        toy + toy
    with pytest.raises(ValueError, match=r"^Can only add datasets of the same type\.$"):
        toy + 3
    with pytest.raises(ValueError, match=r"^Can only add datasets of the same type\.$"):
        toy.__radd__(1)


def test_add_reports_embedding_duplicates_only_when_windows_are_clean(
    toy: _Toy, tmp_path: Path
) -> None:
    """A right operand whose window key is new but whose embedding key is ``toy`` again
    passes the window check and fails the embedding check, once per record:
    "Duplicate keys found in embeddings:toy, toy" (no space after the colon).
    """
    _Toy.items = [
        Data(
            id="g1",
            dna_windows={"w": "GGGG"},
            embeddings={"toy": torch.tensor([[5.0]])},
        ),
        Data(
            id="g2",
            dna_windows={"w": "CCCC"},
            embeddings={"toy": torch.tensor([[6.0]])},
        ),
    ]
    clashing = _Toy(str(tmp_path / "clash"), model_name="other")
    with pytest.raises(
        ValueError, match=r"^Duplicate keys found in embeddings:toy, toy$"
    ):
        toy + clashing


def test_add_with_a_gene_missing_from_one_side_fails_in_collate(
    toy: _Toy, tmp_path: Path
) -> None:
    """Finding: the docstring says ``__add__`` merges another dataset, but a gene present
    on only one side (g3 here) is appended with only its own keys, and collating items
    whose ``dna_windows`` dicts have different key sets raises ``KeyError`` from PyG
    (embedding.py line 162), so datasets over different gene sets cannot be added.
    """
    third = _make(
        tmp_path / "third",
        "other",
        [
            _item("g2", "other", "GGGG", [5.0, 6.0]),
            _item("g3", "other", "AAAA", [9.0, 9.0]),
        ],
    )
    with pytest.raises(KeyError, match=r"^'other'$"):
        toy + third

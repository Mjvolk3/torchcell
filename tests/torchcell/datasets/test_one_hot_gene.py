# tests/torchcell/datasets/test_one_hot_gene.py
# [[tests.torchcell.datasets.test_one_hot_gene]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_one_hot_gene.py
"""``OneHotGeneDataset`` built in ``tmp_path`` on the ``embedding_genome`` stub.

2026.10.06, Phase 21. Genome (``tests/torchcell/conftest.py``): gene set
``YAL001W``, ``YAL002C``, ``YAL003W`` (sorted ``GeneSet`` order), so the index map is
``{YAL001W: 0, YAL002C: 1, YAL003W: 2}`` and gene ``i`` stores the ``[1, 3]`` row with a
single 1 at column ``i``: the identity matrix row by row. ``dna_windows`` is empty, the
store is ``processed/one_hot_gene.pt``, and the built object keeps only a
``ParsedGenome`` (the gffutils-backed genome would not pickle).
"""

import os
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from torchcell.datasets.one_hot_gene import OneHotGeneDataset
from torchcell.sequence import ParsedGenome

GENES = ["YAL001W", "YAL002C", "YAL003W"]


@pytest.fixture
def dataset(embedding_genome: Any, tmp_path: Path) -> OneHotGeneDataset:
    """A dataset processed from scratch under ``tmp_path``."""
    return OneHotGeneDataset(root=str(tmp_path), genome=embedding_genome)


def test_each_gene_stores_its_identity_row(
    dataset: OneHotGeneDataset, tmp_path: Path
) -> None:
    """Integer index ``i`` holds gene ``GENES[i]`` with one-hot row ``eye(3)[i]``."""
    assert dataset.gene_to_index == {"YAL001W": 0, "YAL002C": 1, "YAL003W": 2}
    assert len(dataset) == 3
    eye = torch.eye(3)
    for i, gene in enumerate(GENES):
        item = dataset[i]
        assert item.id == gene
        emb = item.embeddings["one_hot_gene"]
        assert emb.dtype == torch.float32
        assert torch.equal(emb, eye[i : i + 1])
        assert item.dna_windows == {}
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "one_hot_gene.pt",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_gene_id_lookup_returns_the_same_row(dataset: OneHotGeneDataset) -> None:
    """``ds["YAL003W"]`` is the ``[1, 3]`` row ``[0, 0, 1]``; an unknown id is refused."""
    item = dataset["YAL003W"]
    assert torch.equal(item.embeddings["one_hot_gene"], torch.tensor([[0.0, 0.0, 1.0]]))
    with pytest.raises(
        KeyError, match=re.escape("Gene YBR999W not found in the dataset.")
    ):
        dataset["YBR999W"]


def test_built_dataset_keeps_a_parsed_genome_only(dataset: OneHotGeneDataset) -> None:
    """After construction ``genome`` is a ``ParsedGenome`` with the same gene set."""
    assert isinstance(dataset.genome, ParsedGenome)
    assert list(dataset.genome.gene_set) == GENES
    assert OneHotGeneDataset.parse_genome(None) is None


def test_one_hot_encode_gene_exact_and_refuses_unknown(
    dataset: OneHotGeneDataset,
) -> None:
    """``YAL002C`` encodes to ``[0, 1, 0]``; a gene outside the set is a ``KeyError``."""
    assert torch.equal(
        dataset.one_hot_encode_gene("YAL002C"), torch.tensor([0.0, 1.0, 0.0])
    )
    with pytest.raises(KeyError, match="^'YBR999W'$"):
        dataset.one_hot_encode_gene("YBR999W")


def test_pre_transform_runs_on_every_record_before_saving(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """A pre-transform doubling the row is applied once per gene and saved.

    The stored rows are ``2 * eye(3)``, and the call log lists the genes in order.
    """
    seen: list[str] = []

    def double(data: Data) -> Data:
        seen.append(data.id)
        data.embeddings = {"one_hot_gene": 2 * data.embeddings["one_hot_gene"]}
        return data

    ds = OneHotGeneDataset(
        root=str(tmp_path), genome=embedding_genome, pre_transform=double
    )
    assert seen == GENES
    for i in range(3):
        assert torch.equal(
            ds[i].embeddings["one_hot_gene"], 2 * torch.eye(3)[i : i + 1]
        )


def test_cache_hit_does_not_reprocess(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second construction on the same root loads the store and never calls
    ``process``; the rows are the identity rows written by the first build.
    """
    OneHotGeneDataset(root=str(tmp_path), genome=embedding_genome)

    def explode(self: OneHotGeneDataset) -> None:
        raise AssertionError("process ran on a cache hit")

    monkeypatch.setattr(OneHotGeneDataset, "process", explode)
    ds = OneHotGeneDataset(root=str(tmp_path), genome=embedding_genome)
    assert torch.equal(
        torch.cat([ds[i].embeddings["one_hot_gene"] for i in range(3)]), torch.eye(3)
    )


def test_process_without_a_model_name_writes_nothing(
    dataset: OneHotGeneDataset, tmp_path: Path
) -> None:
    """The ``if not self.model_name: return`` guard: no file is written or changed."""
    store = tmp_path / "processed" / "one_hot_gene.pt"
    before = store.read_bytes()
    dataset.model_name = ""
    dataset.process()
    assert store.read_bytes() == before
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "one_hot_gene.pt",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_missing_store_branch_names_a_method_that_does_not_exist(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: the post-init rebuild branch calls ``initialize_transformer``, which no
    class in the hierarchy defines.

    one_hot_gene.py:52-54 runs ``self.initialize_transformer()`` when the store is
    absent after ``super().__init__``. PyG's ``Dataset.__init__`` has already run
    ``process`` and written the store by then (the tests above), so the branch is
    unreachable today; if it ever ran it would raise ``AttributeError``. A sentinel
    patched onto the class records any call: a fresh build on an empty root completes
    without one. Pinned until the dead branch is removed.
    """
    assert not hasattr(OneHotGeneDataset, "initialize_transformer")
    assert hasattr(OneHotGeneDataset, "initialize_model")
    calls: list[str] = []

    def sentinel(self: OneHotGeneDataset) -> None:
        calls.append("initialize_transformer")
        raise AssertionError("initialize_transformer was called")

    monkeypatch.setattr(
        OneHotGeneDataset, "initialize_transformer", sentinel, raising=False
    )
    ds = OneHotGeneDataset(root=str(tmp_path), genome=embedding_genome)
    assert calls == []
    assert (tmp_path / "processed" / "one_hot_gene.pt").is_file()
    assert torch.equal(ds[1].embeddings["one_hot_gene"], torch.eye(3)[1:2])

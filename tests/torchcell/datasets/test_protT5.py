# tests/torchcell/datasets/test_protT5.py
# [[tests.torchcell.datasets.test_protT5]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_protT5.py
"""``ProtT5Dataset`` on the ``embedding_genome`` stub with the ProtT5 backbone faked.

2026.09.30, Phase 18. The module-level name ``torchcell.datasets.protT5.ProtT5`` is
replaced by ``_FakeProtT5``, so no tokenizer or weights are ever loaded; CUDA is
reported absent and ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second
guard. Every store is written under ``tmp_path``.

Genome (``tests/torchcell/conftest.py``): ``YAL001W`` Verified, protein ``MKPG*``;
``YAL002C`` Dubious, ``MSK*``; ``YAL003W`` Uncharacterized, ``MKKS*``.

Stand-in forward: features ``f = [len, #M, #K, #*]``, then the fixed lower-triangular
ones map (a cumulative sum), returned with the real wrapper's ``mean_embedding`` shape
``[batch, hidden]`` = ``[1, 4]``:

* ``MKPG*``: f = [5, 1, 1, 1], embedding [[5, 6, 7, 8]];
* ``MSK*``: f = [4, 1, 1, 1], embedding [[4, 5, 6, 7]];
* ``MKKS*``: f = [5, 1, 2, 1], embedding [[5, 6, 8, 9]].

The dataset converts an embedded vector to a numpy array (``protT5.py`` line 110) but
stores an excluded gene as a torch ``zeros(1, 1024)`` (line 105), so the collate keeps a
Python list of mixed types (Findings below).
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import torchcell.datasets.protT5 as prot_t5_module
from torchcell.datasets.protT5 import ProtT5Dataset

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
EMBEDDED = {
    "YAL001W": [[5.0, 6.0, 7.0, 8.0]],
    "YAL002C": [[4.0, 5.0, 6.0, 7.0]],
    "YAL003W": [[5.0, 6.0, 8.0, 9.0]],
}
PROTEIN = {"YAL001W": "MKPG*", "YAL002C": "MSK*", "YAL003W": "MKKS*"}


class _FakeProtT5:
    """Records its checkpoint name and every embed call; forward is a fixed map."""

    inits: list[str] = []
    calls: list[tuple[list[str], bool]] = []

    def __init__(self, model_name: str) -> None:
        _FakeProtT5.inits.append(model_name)

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        """Cumulative sum of ``[len, #M, #K, #*]``, shaped ``[batch, 4]``."""
        _FakeProtT5.calls.append((list(sequences), mean_embedding))
        return torch.stack(
            [
                torch.tensor(
                    [len(s), s.count("M"), s.count("K"), s.count("*")],
                    dtype=torch.float32,
                ).cumsum(0)
                for s in sequences
            ]
        )


class _ExplodingProtT5:
    """A backbone that must never be built (the store is already on disk)."""

    def __init__(self, model_name: str) -> None:
        raise AssertionError(f"backbone {model_name} built on a cache hit")


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in the stand-in, clear its logs, report CUDA absent, stay offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeProtT5, "inits", [])
    monkeypatch.setattr(_FakeProtT5, "calls", [])
    monkeypatch.setattr(prot_t5_module, "ProtT5", _FakeProtT5)


def test_no_dubious_stores_numpy_vectors_and_a_1024_wide_torch_zero_row(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """The stored embedding type depends on whether the gene was excluded.

    Pins: one backbone built, named ``prot_t5_xl_uniref50``; ``embed`` gets
    ``[protein]`` with ``mean_embedding=True`` for YAL001W and YAL003W only; their
    stored values are float32 numpy arrays equal to the stand-in's ``[1, 4]`` output;
    the Dubious YAL002C is a float32 torch tensor of zeros with shape ``[1, 1024]``;
    ``dna_windows`` holds the protein string; the store is ``processed/<name>.pt``.

    Finding: the stored type is mixed (numpy for embedded genes, torch for excluded
    ones), the collate therefore keeps a Python list rather than a tensor, and the
    zero row's width is the hard-coded 1024 rather than the backbone's width.
    Pinned until the dataset stores one tensor type of the backbone's width.
    """
    name = "prot_t5_xl_uniref50_no_dubious"
    ds = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert _FakeProtT5.inits == ["prot_t5_xl_uniref50"]
    assert _FakeProtT5.calls == [(["MKPG*"], True), (["MKKS*"], True)]
    stored = ds._data.embeddings[name]
    assert [type(v).__name__ for v in stored] == ["ndarray", "Tensor", "ndarray"]
    assert stored[0].dtype == np.float32
    assert stored[0].tolist() == EMBEDDED["YAL001W"]
    assert stored[2].tolist() == EMBEDDED["YAL003W"]
    assert stored[1].dtype == torch.float32
    assert torch.equal(stored[1], torch.zeros(1, 1024))
    assert [ds[i].dna_windows[name] for i in range(3)] == [PROTEIN[g] for g in GENE_IDS]
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "pre_filter.pt",
        "pre_transform.pt",
        f"{name}.pt",
    ]


@pytest.mark.parametrize(
    ("model_name", "embedded"),
    [
        ("prot_t5_xl_uniref50_all", GENE_IDS),
        ("prot_t5_xl_uniref50_no_uncharacterized", ["YAL001W", "YAL002C"]),
        ("prot_t5_xl_uniref50_no_dubious_uncharacterized", GENE_IDS),
    ],
)
def test_exclusion_list_selects_which_genes_reach_the_backbone(
    embedding_genome: Any, tmp_path: Path, model_name: str, embedded: list[str]
) -> None:
    """Only non-excluded genes are embedded; the rest are 1024-wide zeros.

    Finding: ``prot_t5_xl_uniref50_no_dubious_uncharacterized`` lists ``"dubious"``
    and ``"uncharacterized"`` in lowercase (``protT5.py`` lines 30 to 33), which never
    match SGD's ``Dubious`` / ``Uncharacterized``, so it embeds all three genes like
    ``_all``. Pinned until the list is capitalized.
    """
    ds = ProtT5Dataset(
        root=str(tmp_path), genome=embedding_genome, model_name=model_name
    )

    assert _FakeProtT5.calls == [([PROTEIN[g]], True) for g in embedded]
    stored = dict(zip(ds._data.id, ds._data.embeddings[model_name], strict=True))
    for gene in GENE_IDS:
        if gene in embedded:
            assert stored[gene].tolist() == EMBEDDED[gene]
        else:
            assert torch.equal(stored[gene], torch.zeros(1, 1024))


def test_lookup_by_gene_id_returns_a_one_element_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: ``ds["<gene>"].embeddings`` holds a list, not a ``[1, D]`` tensor.

    ``BaseEmbeddingDataset.__getitem__`` slices ``value[index : index + 1]``, which on
    the collated Python list returns a one-element list around the numpy array.
    Pinned until the dataset stores tensors.
    """
    name = "prot_t5_xl_uniref50_all"
    ds = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    got = ds["YAL003W"].embeddings[name]
    assert type(got) is list
    assert [v.tolist() for v in got] == [EMBEDDED["YAL003W"]]
    assert ds["YAL003W"].dna_windows == {name: "MKKS*"}


def test_second_construction_reads_store_without_building_backbone(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is; the backbone class is never instantiated."""
    name = "prot_t5_xl_uniref50_no_dubious"
    ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)
    monkeypatch.setattr(prot_t5_module, "ProtT5", _ExplodingProtT5)

    again = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    stored = again._data.embeddings[name]
    assert list(again._data.id) == GENE_IDS
    assert stored[0].tolist() == EMBEDDED["YAL001W"]
    assert torch.equal(stored[1], torch.zeros(1, 1024))
    assert stored[2].tolist() == EMBEDDED["YAL003W"]


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory or backbone is created.

    Finding: the two message fragments are joined without a space
    (``"...'prot_t5_bogus'.Valid options are: ..."``, ``embedding.py`` lines 30 to
    31). Pinned until a space is added.
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        ProtT5Dataset(
            root=str(root), genome=embedding_genome, model_name="prot_t5_bogus"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'prot_t5_bogus'.Valid options are: "
        "prot_t5_xl_uniref50_all, prot_t5_xl_uniref50_no_dubious_uncharacterized, "
        "prot_t5_xl_uniref50_no_dubious, prot_t5_xl_uniref50_no_uncharacterized"
    )
    assert _FakeProtT5.inits == []
    assert not root.exists()


def test_no_model_name_still_builds_the_backbone_every_construction(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: ``model_name=None`` loads the backbone and then stores nothing.

    ``process`` calls ``initialize_model()`` before its ``if not self.model_name``
    return (``protT5.py`` lines 85 to 87), and with no ``None.pt`` on disk PyG runs
    ``process`` on every construction, so two constructions build two backbones (two
    full ProtT5 downloads or loads in production). Only PyG's two marker files are
    written and the data stays ``None``. Pinned until the early return comes first.
    """
    first = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)
    ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)

    assert _FakeProtT5.inits == ["prot_t5_xl_uniref50", "prot_t5_xl_uniref50"]
    assert _FakeProtT5.calls == []
    assert first._data is None
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

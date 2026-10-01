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
``[batch, hidden]`` = ``[1, 4]``; ``model.config.hidden_size = 4``:

* ``MKPG*``: f = [5, 1, 1, 1], embedding [[5, 6, 7, 8]];
* ``MSK*``: f = [4, 1, 1, 1], embedding [[4, 5, 6, 7]];
* ``MKKS*``: f = [5, 1, 2, 1], embedding [[5, 6, 8, 9]].

Every gene is stored as a float32 CPU torch ``[1, 4]`` row (issue #543), an excluded
gene as ``zeros(1, hidden_size)``, so the collate is one ``[3, 4]`` tensor.
"""

import os
from pathlib import Path
from typing import Any

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
        config = type("Config", (), {"hidden_size": 4})()
        self.model = type("Model", (), {"config": config})()

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


def test_no_dubious_stores_one_tensor_of_the_backbone_width(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Embedded and excluded genes are both float32 torch rows of the backbone width.

    Pins: one backbone built, named ``prot_t5_xl_uniref50``; ``embed`` gets
    ``[protein]`` with ``mean_embedding=True`` for YAL001W and YAL003W only; the
    collate is one ``[3, 4]`` float32 tensor whose rows are the stand-in's output and,
    for the Dubious YAL002C, ``zeros(1, 4)`` (the width read from
    ``model.config.hidden_size``, issue #543); ``dna_windows`` holds the protein string;
    the store is ``processed/<name>.pt``.
    """
    name = "prot_t5_xl_uniref50_no_dubious"
    ds = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert _FakeProtT5.inits == ["prot_t5_xl_uniref50"]
    assert _FakeProtT5.calls == [(["MKPG*"], True), (["MKKS*"], True)]
    stored = ds._data.embeddings[name]
    assert type(stored) is torch.Tensor
    assert stored.dtype == torch.float32
    assert stored.tolist() == [
        EMBEDDED["YAL001W"][0],
        [0.0, 0.0, 0.0, 0.0],
        EMBEDDED["YAL003W"][0],
    ]
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
        ("prot_t5_xl_uniref50_no_dubious_uncharacterized", ["YAL001W"]),
    ],
)
def test_exclusion_list_selects_which_genes_reach_the_backbone(
    embedding_genome: Any, tmp_path: Path, model_name: str, embedded: list[str]
) -> None:
    """Only non-excluded genes are embedded; the rest are ``zeros(1, 4)``.

    ``prot_t5_xl_uniref50_no_dubious_uncharacterized`` lists SGD's capitalized
    ``Dubious`` / ``Uncharacterized`` (issue #543), so it zeroes YAL002C and YAL003W.
    """
    ds = ProtT5Dataset(
        root=str(tmp_path), genome=embedding_genome, model_name=model_name
    )

    assert _FakeProtT5.calls == [([PROTEIN[g]], True) for g in embedded]
    for gene in GENE_IDS:
        expected = EMBEDDED[gene] if gene in embedded else [[0.0, 0.0, 0.0, 0.0]]
        assert ds[gene].embeddings[model_name].tolist() == expected


def test_lookup_by_gene_id_and_by_index_return_the_same_row(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``ds["<gene>"]`` and ``ds[i]`` both return the gene's ``[1, 4]`` tensor row."""
    name = "prot_t5_xl_uniref50_all"
    ds = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    for i, gene in enumerate(GENE_IDS):
        by_id = ds[gene].embeddings[name]
        assert type(by_id) is torch.Tensor
        assert by_id.tolist() == EMBEDDED[gene]
        assert torch.equal(by_id, ds[i].embeddings[name])
    assert ds["YAL003W"].dna_windows == {name: "MKKS*"}


def test_second_construction_reads_store_without_building_backbone(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is; the backbone class is never instantiated."""
    name = "prot_t5_xl_uniref50_no_dubious"
    ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)
    monkeypatch.setattr(prot_t5_module, "ProtT5", _ExplodingProtT5)

    again = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert list(again._data.id) == GENE_IDS
    assert again._data.embeddings[name].tolist() == [
        EMBEDDED["YAL001W"][0],
        [0.0, 0.0, 0.0, 0.0],
        EMBEDDED["YAL003W"][0],
    ]


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory or backbone is created.

    The message names the bad name, then the valid names after a space (issue #543).
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        ProtT5Dataset(
            root=str(root), genome=embedding_genome, model_name="prot_t5_bogus"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'prot_t5_bogus'. Valid options are: "
        "prot_t5_xl_uniref50_all, prot_t5_xl_uniref50_no_dubious_uncharacterized, "
        "prot_t5_xl_uniref50_no_dubious, prot_t5_xl_uniref50_no_uncharacterized"
    )
    assert _FakeProtT5.inits == []
    assert not root.exists()


def test_no_model_name_builds_no_backbone(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``model_name=None`` returns from ``process`` before building a backbone.

    PyG still runs ``process`` on every construction (there is no ``None.pt``), but
    two constructions build no backbone (issue #543), only PyG's two marker files are
    written, and the data stays ``None``.
    """
    first = ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)
    ProtT5Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)

    assert _FakeProtT5.inits == []
    assert _FakeProtT5.calls == []
    assert first._data is None
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

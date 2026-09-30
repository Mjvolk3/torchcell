# tests/torchcell/datasets/test_esm2.py
# [[tests.torchcell.datasets.test_esm2]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_esm2.py
"""``Esm2Dataset`` on the ``embedding_genome`` stub with the ESM2 backbone faked.

2026.09.30, Phase 18. The module-level name ``torchcell.datasets.esm2.Esm2`` is
replaced by ``_FakeEsm2``, so no tokenizer or weights are ever loaded; CUDA is reported
absent and ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard.
Every store is written under ``tmp_path``.

Genome (``tests/torchcell/conftest.py``): ``YAL001W`` Verified, protein ``MKPG*``;
``YAL002C`` Dubious, ``MSK*``; ``YAL003W`` Uncharacterized, ``MKKS*``. The dataset feeds
the protein FASTA record verbatim (stop symbol included), one sequence per call.

Stand-in forward: features ``f = [len, #M, #K, #*]`` of the sequence, then the fixed
lower-triangular ones map (a cumulative sum), returned with the real wrapper's
``mean_embedding`` shape ``[1, batch, hidden]``; ``model.config.hidden_size = 4``.

* ``MKPG*``: f = [5, 1, 1, 1], embedding [5, 6, 7, 8];
* ``MSK*``: f = [4, 1, 1, 1], embedding [4, 5, 6, 7];
* ``MKKS*``: f = [5, 1, 2, 1], embedding [5, 6, 8, 9].

The dataset squeezes each to shape ``[4]`` float32, and an excluded gene gets
``zeros(4)`` without a backbone call. The PyG collate concatenates the three ``[4]``
vectors into one flat ``[12]`` tensor, which breaks lookup by gene id (Finding below).
"""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

import torchcell.datasets.esm2 as esm2_module
from torchcell.datasets.esm2 import Esm2Dataset
from torchcell.sequence import ParsedGenome

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
EMBEDDED = {
    "YAL001W": [5.0, 6.0, 7.0, 8.0],
    "YAL002C": [4.0, 5.0, 6.0, 7.0],
    "YAL003W": [5.0, 6.0, 8.0, 9.0],
}
PROTEIN = {"YAL001W": "MKPG*", "YAL002C": "MSK*", "YAL003W": "MKKS*"}


class _FakeEsm2:
    """Records its checkpoint name and every embed call; forward is a fixed map."""

    inits: list[str] = []
    calls: list[tuple[list[str], bool]] = []

    def __init__(self, model_name: str) -> None:
        _FakeEsm2.inits.append(model_name)
        config = type("Config", (), {"hidden_size": 4})()
        self.model = type("Model", (), {"config": config})()

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        """Cumulative sum of ``[len, #M, #K, #*]``, shaped ``[1, batch, 4]``."""
        _FakeEsm2.calls.append((list(sequences), mean_embedding))
        rows = [
            torch.tensor(
                [len(s), s.count("M"), s.count("K"), s.count("*")], dtype=torch.float32
            ).cumsum(0)
            for s in sequences
        ]
        return torch.stack(rows).unsqueeze(0)


class _ExplodingEsm2:
    """A backbone that must never be built (the store is already on disk)."""

    def __init__(self, model_name: str) -> None:
        raise AssertionError(f"backbone {model_name} built on a cache hit")


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in the stand-in, clear its logs, report CUDA absent, stay offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeEsm2, "inits", [])
    monkeypatch.setattr(_FakeEsm2, "calls", [])
    monkeypatch.setattr(esm2_module, "Esm2", _FakeEsm2)


def _per_gene(ds: Esm2Dataset, name: str) -> dict[str, list[float]]:
    """Gene id to its stored embedding, read by integer index."""
    out = {}
    for i in range(len(ds)):
        item = ds[i]
        emb = item.embeddings[name]
        assert emb.dtype == torch.float32
        assert emb.shape == (4,)
        out[item.id] = emb.tolist()
    return out


@pytest.mark.parametrize(
    ("model_name", "checkpoint", "zeroed"),
    [
        ("esm2_t6_8M_UR50D_all", "esm2_t6_8M_UR50D", set()),
        ("esm2_t12_35M_UR50D_no_dubious", "esm2_t12_35M_UR50D", {"YAL002C"}),
        ("esm2_t33_650M_UR50D_no_uncharacterized", "esm2_t33_650M_UR50D", {"YAL003W"}),
        ("esm2_t48_15B_UR50D_no_dubious_uncharacterized", "esm2_t48_15B_UR50D", set()),
    ],
)
def test_build_embeds_proteins_and_zeroes_excluded_classes(
    embedding_genome: Any,
    tmp_path: Path,
    model_name: str,
    checkpoint: str,
    zeroed: set[str],
) -> None:
    """Each gene's protein is embedded once, except excluded classes, which get zeros.

    Pins: the backbone is built exactly once with the checkpoint prefix of the dataset
    name; ``embed`` receives ``[protein]`` with ``mean_embedding=True`` for every
    non-excluded gene in gene-set order and is never called for an excluded one; the
    stored vector is the stand-in's value (or ``zeros(4)``), shape ``[4]`` float32;
    ``dna_windows`` holds the protein string; the store is ``processed/<name>.pt``.

    Finding: the ``_no_dubious_uncharacterized`` variants list ``"dubious"`` and
    ``"uncharacterized"`` in lowercase (``esm2.py`` lines 29 to 32), while SGD writes
    ``Dubious`` / ``Uncharacterized`` (the other variants use the capitalized form), so
    that variant excludes nothing and equals ``_all``. Pinned until the list is fixed.
    """
    ds = Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=model_name)

    assert _FakeEsm2.inits == [checkpoint]
    assert _FakeEsm2.calls == [
        ([PROTEIN[g]], True) for g in GENE_IDS if g not in zeroed
    ]
    expected = {
        g: [0.0, 0.0, 0.0, 0.0] if g in zeroed else EMBEDDED[g] for g in GENE_IDS
    }
    assert _per_gene(ds, model_name) == expected
    assert [ds[i].dna_windows[model_name] for i in range(3)] == [
        PROTEIN[g] for g in GENE_IDS
    ]
    assert sorted(os.listdir(tmp_path / "processed")) == [
        f"{model_name}.pt",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_second_construction_reads_store_without_building_backbone(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is; the backbone class is never instantiated.

    The second construction swaps in a backbone that raises when built and still
    returns the first build's embeddings, including the zeroed Dubious gene.
    """
    name = "esm2_t6_8M_UR50D_no_dubious"
    Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)
    monkeypatch.setattr(esm2_module, "Esm2", _ExplodingEsm2)

    again = Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert _per_gene(again, name) == {
        "YAL001W": EMBEDDED["YAL001W"],
        "YAL002C": [0.0, 0.0, 0.0, 0.0],
        "YAL003W": EMBEDDED["YAL003W"],
    }
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_pre_transform_is_applied_before_the_store_is_written(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cached tensors are the pre-transformed ones (here: embeddings times 10).

    A later construction without ``pre_transform`` reads the same scaled values, so
    the transform is baked into the store rather than applied at read time.
    """
    name = "esm2_t6_8M_UR50D_all"

    def times_ten(data: Data) -> Data:
        data.embeddings = {k: v * 10 for k, v in data.embeddings.items()}
        return data

    Esm2Dataset(
        root=str(tmp_path),
        genome=embedding_genome,
        model_name=name,
        pre_transform=times_ten,
    )
    monkeypatch.setattr(esm2_module, "Esm2", _ExplodingEsm2)
    with pytest.warns(UserWarning, match="The `pre_transform` argument differs"):
        again = Esm2Dataset(
            root=str(tmp_path), genome=embedding_genome, model_name=name
        )

    assert _per_gene(again, name) == {
        g: [10 * x for x in EMBEDDED[g]] for g in GENE_IDS
    }


def test_lookup_by_gene_id_slices_the_flattened_collate(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: ``ds["<gene>"]`` returns one scalar of the flat store, not the vector.

    The squeeze at ``esm2.py`` line 165 leaves each embedding 1-D, so the collate
    concatenates them into ``[5, 6, 7, 8, 4, 5, 6, 7, 5, 6, 8, 9]`` (``_all``), and
    ``BaseEmbeddingDataset.__getitem__`` slices ``value[index : index + 1]``: gene 0
    gives ``[5.]``, gene 1 ``[6.]`` (YAL001W's second component) and gene 2 ``[7.]``.
    Pinned until the stored embeddings keep a leading axis.
    """
    name = "esm2_t6_8M_UR50D_all"
    ds = Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert ds._data.embeddings[name].tolist() == [
        5.0,
        6.0,
        7.0,
        8.0,
        4.0,
        5.0,
        6.0,
        7.0,
        5.0,
        6.0,
        8.0,
        9.0,
    ]
    assert {g: ds[g].embeddings[name].tolist() for g in GENE_IDS} == {
        "YAL001W": [5.0],
        "YAL002C": [6.0],
        "YAL003W": [7.0],
    }
    assert ds["YAL002C"].dna_windows == {name: "MSK*"}


@pytest.mark.parametrize("model_name", ["esm2_t6_8M_UR50D_bogus", None])
def test_unknown_model_name_is_a_key_error_before_anything_is_written(
    embedding_genome: Any, tmp_path: Path, model_name: str | None
) -> None:
    """Finding: an unknown (or absent) name raises ``KeyError`` from ``MODEL_TO_WINDOW``.

    ``esm2.py`` line 102 indexes the table before ``BaseEmbeddingDataset`` can raise its
    ``ValueError("Invalid model_name ...")``, so this dataset refuses with the bare key,
    and ``model_name=None`` (the signature default) cannot construct at all. Nothing
    is built and nothing is written. Pinned until the lookup moves after the base check.
    """
    root = tmp_path / "store"
    with pytest.raises(KeyError) as excinfo:
        Esm2Dataset(root=str(root), genome=embedding_genome, model_name=model_name)

    assert excinfo.value.args == (model_name,)
    assert _FakeEsm2.inits == []
    assert not root.exists()

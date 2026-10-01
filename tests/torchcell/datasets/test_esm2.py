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

The dataset stores each as a ``[1, 4]`` float32 row (issue #543), and an excluded gene
gets ``zeros(1, 4)`` without a backbone call. The PyG collate stacks the rows into
``[3, 4]``, so integer and gene-id lookup both return the gene's ``[1, 4]`` row.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import Data

import torchcell.datasets.esm2 as esm2_module
from torchcell.datasets.cell import CellDataset
from torchcell.datasets.cell import ParsedGenome as CellParsedGenome
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
        assert emb.shape == (1, 4)
        out[item.id] = emb[0].tolist()
    return out


@pytest.mark.parametrize(
    ("model_name", "checkpoint", "zeroed"),
    [
        ("esm2_t6_8M_UR50D_all", "esm2_t6_8M_UR50D", set()),
        ("esm2_t12_35M_UR50D_no_dubious", "esm2_t12_35M_UR50D", {"YAL002C"}),
        ("esm2_t33_650M_UR50D_no_uncharacterized", "esm2_t33_650M_UR50D", {"YAL003W"}),
        (
            "esm2_t48_15B_UR50D_no_dubious_uncharacterized",
            "esm2_t48_15B_UR50D",
            {"YAL002C", "YAL003W"},
        ),
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
    stored row is the stand-in's value (or zeros), shape ``[1, 4]`` float32;
    ``dna_windows`` holds the protein string; the store is ``processed/<name>.pt``.

    The ``_no_dubious_uncharacterized`` variants list SGD's capitalized ``Dubious`` /
    ``Uncharacterized`` (issue #543), so the Dubious YAL002C and the Uncharacterized
    YAL003W are both zeroed and only YAL001W reaches the backbone.
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


def test_lookup_by_gene_id_and_by_index_return_the_same_row(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``ds["<gene>"]`` and ``ds[i]`` return the gene's exact ``[1, 4]`` row (issue #543).

    The collate is ``[3, 4]`` with one row per gene in gene-set order. The node
    feature ``CellDataset.create_embedding_graph`` builds (``squeeze(0)`` of each row)
    is the gene's ``[4]`` vector, as it was for the old flat layout.
    """
    name = "esm2_t6_8M_UR50D_all"
    ds = Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=name)

    assert ds._data.embeddings[name].tolist() == [EMBEDDED[g] for g in GENE_IDS]
    for i, gene in enumerate(GENE_IDS):
        by_id = ds[gene].embeddings[name]
        by_index = ds[i].embeddings[name]
        assert by_id.tolist() == [EMBEDDED[gene]]
        assert torch.equal(by_id, by_index)
    assert ds["YAL002C"].dna_windows == {name: "MSK*"}

    parsed = ds.genome
    assert isinstance(parsed, ParsedGenome)
    graph = CellDataset.create_embedding_graph(
        CellParsedGenome(gene_set=parsed.gene_set), ds
    )
    assert {g: graph.nodes[g]["embedding"].tolist() for g in graph.nodes} == EMBEDDED


@pytest.mark.parametrize("model_name", ["esm2_t6_8M_UR50D_bogus", "esm2_t6_8M"])
def test_unknown_model_name_is_a_value_error_naming_the_valid_names(
    embedding_genome: Any, tmp_path: Path, model_name: str
) -> None:
    """An unknown name is refused by ``BaseEmbeddingDataset`` with a ``ValueError``.

    The message names the bad name, then every valid name in table order (issue
    #543; it used to be a bare ``KeyError``). Nothing is built and nothing is written.
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        Esm2Dataset(root=str(root), genome=embedding_genome, model_name=model_name)

    valid = ", ".join(Esm2Dataset.MODEL_TO_WINDOW)
    assert str(excinfo.value) == (
        f"Invalid model_name '{model_name}'. Valid options are: {valid}"
    )
    assert valid.startswith("esm2_t6_8M_UR50D_all, esm2_t6_8M_UR50D_no_dubious_unch")
    assert len(Esm2Dataset.MODEL_TO_WINDOW) == 24
    assert _FakeEsm2.inits == []
    assert not root.exists()


def test_no_model_name_builds_no_backbone_and_no_store(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``model_name=None`` (the signature default) constructs empty, without a backbone.

    ``process`` returns before building one, so two constructions build none, and
    only PyG's two marker files are written.
    """
    first = Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)
    Esm2Dataset(root=str(tmp_path), genome=embedding_genome, model_name=None)

    assert _FakeEsm2.inits == []
    assert first._data is None
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "pre_filter.pt",
        "pre_transform.pt",
    ]

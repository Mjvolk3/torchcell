# tests/torchcell/datasets/test_node_embedding_builder.py
# [[tests.torchcell.datasets.test_node_embedding_builder]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_node_embedding_builder.py
"""``NodeEmbeddingBuilder.build`` on recording stand-ins for every dataset class.

2026.10.06, Phase 21. ``EMBEDDING_CONFIGS`` is the contract between an experiment
config's ``node_embeddings`` list and the on-disk stores: each name picks a dataset
class, a root under ``data_root`` and the ``model_name`` that selects the store file.
``TABLE`` below restates it by hand (class, root, model name, needs genome, needs
graph). The tests compare the real table against it, then swap every class for a
recorder (``_Fake``, which stores its kwargs and never builds anything) and check the
exact kwargs ``build`` passes. ``data_root`` is the literal ``/root``; nothing is
written.
"""

import re
from typing import Any

import pytest

from torchcell.datasets.codon_frequency import CodonFrequencyDataset
from torchcell.datasets.codon_language_model import CalmDataset
from torchcell.datasets.esm2 import Esm2Dataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.one_hot_gene import OneHotGeneDataset
from torchcell.datasets.protT5 import ProtT5Dataset
from torchcell.datasets.random_embedding import RandomEmbeddingDataset
from torchcell.datasets.sgd_gene_graph import GraphEmbeddingDataset

S = "data/scerevisiae/"
NT = S + "nucleotide_transformer_embedding"
ESM = S + "esm2_embedding"
RND = S + "random_embedding"
GG = S + "sgd_gene_graph_hot"
# name: (class, root_path, model_name, requires_genome, requires_graph)
TABLE: dict[str, tuple[type[Any], str, str | None, bool, bool]] = {
    "one_hot_gene": (OneHotGeneDataset, S + "one_hot_gene_embedding", None, True, False),
    "codon_frequency": (CodonFrequencyDataset, S + "codon_frequency_embedding", None, True, False),
    "calm": (CalmDataset, S + "calm_embedding", "calm", True, False),
    "fudt_downstream": (FungalUpDownTransformerDataset, S + "fudt_embedding", "species_downstream", True, False),
    "fudt_upstream": (FungalUpDownTransformerDataset, S + "fudt_embedding", "species_upstream", True, False),
    "nt_window_5979": (NucleotideTransformerDataset, NT, "nt_window_5979", True, False),
    "nt_window_5979_max": (NucleotideTransformerDataset, NT, "nt_window_5979_max", True, False),
    "nt_window_three_prime_5979": (NucleotideTransformerDataset, NT, "nt_window_three_prime_5979", True, False),
    "nt_window_five_prime_5979": (NucleotideTransformerDataset, NT, "nt_window_five_prime_5979", True, False),
    "nt_window_three_prime_300": (NucleotideTransformerDataset, NT, "nt_window_three_prime_300", True, False),
    "nt_window_five_prime_1003": (NucleotideTransformerDataset, NT, "nt_window_five_prime_1003", True, False),
    "prot_T5_all": (ProtT5Dataset, S + "protT5_embedding", "prot_t5_xl_uniref50_all", True, False),
    "prot_T5_no_dubious": (ProtT5Dataset, S + "protT5_embedding", "prot_t5_xl_uniref50_no_dubious", True, False),
    "esm2_t33_650M_UR50D_all": (Esm2Dataset, ESM, "esm2_t33_650M_UR50D_all", True, False),
    "esm2_t33_650M_UR50D_no_dubious": (Esm2Dataset, ESM, "esm2_t33_650M_UR50D_no_dubious", True, False),
    "esm2_t33_650M_UR50D_no_dubious_uncharacterized": (Esm2Dataset, ESM, "esm2_t33_650M_UR50D_no_dubious_uncharacterized", True, False),
    "esm2_t33_650M_UR50D_no_uncharacterized": (Esm2Dataset, ESM, "esm2_t33_650M_UR50D_no_uncharacterized", True, False),
    "normalized_chrom_pathways": (GraphEmbeddingDataset, GG, "normalized_chrom_pathways", False, True),
    "chrom_pathways": (GraphEmbeddingDataset, GG, "chrom_pathways", False, True),
    "random_1000": (RandomEmbeddingDataset, RND, "random_1000", True, False),
    "random_100": (RandomEmbeddingDataset, RND, "random_100", True, False),
    "random_10": (RandomEmbeddingDataset, RND, "random_10", True, False),
    "random_1": (RandomEmbeddingDataset, RND, "random_1", True, False),
}  # fmt: skip

GENOME = object()
G_GENE = object()


class _Graph:
    """Exposes the one attribute ``build`` reads from a graph."""

    G_gene = G_GENE


class _Fake:
    """Records the class it replaces and the kwargs it was built with."""

    built: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, replaces: str, **kwargs: Any) -> None:
        _Fake.built.append((replaces, kwargs))
        self.replaces = replaces
        self.kwargs = kwargs


def _fake_class(name: str) -> Any:
    def make(**kwargs: Any) -> _Fake:
        return _Fake(name, **kwargs)

    return make


@pytest.fixture
def faked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every config's class becomes a recorder tagged with the real class's name."""
    monkeypatch.setattr(_Fake, "built", [])
    configs = {
        name: {**cfg, "class": _fake_class(str(getattr(cfg["class"], "__name__")))}
        for name, cfg in NodeEmbeddingBuilder.EMBEDDING_CONFIGS.items()
    }
    monkeypatch.setattr(NodeEmbeddingBuilder, "EMBEDDING_CONFIGS", configs)


def test_the_config_table_is_exactly_the_contract() -> None:
    """Names (in order), classes, roots, model names and dependency flags.

    A ``requires_*`` key that is absent counts as False, as ``build`` reads it with
    ``config.get``.
    """
    configs = NodeEmbeddingBuilder.EMBEDDING_CONFIGS
    assert list(configs) == list(TABLE)
    for name, (cls, root, model_name, needs_genome, needs_graph) in TABLE.items():
        cfg = configs[name]
        assert cfg["class"] is cls, name
        assert cfg["root_path"] == root, name
        assert cfg["model_name"] == model_name, name
        assert bool(cfg.get("requires_genome")) is needs_genome, name
        assert bool(cfg.get("requires_graph")) is needs_graph, name


def test_build_passes_exact_kwargs_for_every_name(faked: None) -> None:
    """Root joined under ``data_root``; genome for genome datasets; ``graph.G_gene``
    (not the graph) for graph datasets; ``model_name`` only when the table has one.
    """
    out = NodeEmbeddingBuilder.build(list(TABLE), "/root", GENOME, graph=_Graph())
    assert list(out) == list(TABLE)
    expected = []
    for cls, root, model_name, needs_genome, needs_graph in TABLE.values():
        kwargs: dict[str, Any] = {"root": f"/root/{root}"}
        if needs_genome:
            kwargs["genome"] = GENOME
        if needs_graph:
            kwargs["graph"] = G_GENE
        if model_name:
            kwargs["model_name"] = model_name
        expected.append((cls.__name__, kwargs))
    assert _Fake.built == expected
    assert [(o.replaces, o.kwargs) for o in out.values()] == expected


def test_learnable_is_skipped_and_detected(faked: None) -> None:
    """``"learnable"`` builds nothing and is reported by ``check_learnable_embedding``."""
    out = NodeEmbeddingBuilder.build(["learnable", "random_10"], "/root", GENOME)
    assert list(out) == ["random_10"]
    assert _Fake.built == [
        (
            "RandomEmbeddingDataset",
            {"root": f"/root/{RND}", "genome": GENOME, "model_name": "random_10"},
        )
    ]
    assert NodeEmbeddingBuilder.check_learnable_embedding(["calm", "learnable"]) is True
    assert NodeEmbeddingBuilder.check_learnable_embedding(["calm"]) is False


def test_unknown_name_is_refused_with_the_available_list(faked: None) -> None:
    """The refusal lists every configured name; names before it are already built."""
    message = f"Unknown embedding name: esm2_all. Available embeddings: {list(TABLE)}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NodeEmbeddingBuilder.build(["random_1", "esm2_all"], "/root", GENOME)
    assert [name for name, _ in _Fake.built] == ["RandomEmbeddingDataset"]


def test_graph_embedding_without_a_graph_is_refused(faked: None) -> None:
    """``chrom_pathways`` with ``graph=None`` raises before building anything."""
    message = (
        "Embedding 'chrom_pathways' requires a graph instance, but none was provided"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        NodeEmbeddingBuilder.build(["chrom_pathways"], "/root", GENOME)
    assert _Fake.built == []

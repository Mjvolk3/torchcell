# tests/torchcell/datasets/test_datasets_fungal_up_down_transformer.py
# [[tests.torchcell.datasets.test_datasets_fungal_up_down_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_datasets_fungal_up_down_transformer.py
"""``FungalUpDownTransformerDataset`` on the ``embedding_genome`` stub, backbone faked.

2026.09.30, Phase 18. The module-level name
``torchcell.datasets.fungal_up_down_transformer.FungalUpDownTransformer`` is replaced by
``_FakeSpeciesLM``, which copies the real class's ``VALID_MODEL_NAMES`` (the dataset
asserts against it) and never loads a tokenizer or weights; CUDA is reported absent
and ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard. Every store
is written under ``tmp_path``.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)``; ``YAL002C`` ``-`` ``[20, 32)``;
``YAL003W`` ``+`` ``[2000, 5100)``. A ``-`` window is the reverse complement.

Stand-in forward: ``[len, #A, #C, #G, #T]`` of each sequence through the identity map,
in the real wrapper's ``mean_embedding`` shape ``[batch, 5]``, so each gene stores
``[1, 5]``.

Windows (``s288c.py`` ``window_five_prime`` / ``window_three_prime`` with the codon flag
``True`` and ``allow_undersize=True``):

* ``species_upstream`` (1003 nt ending 3 nt into the CDS, so the start codon is
  included): YAL001W ``[103 - 1003, 103)`` clips to ``[0, 103)``; YAL003W
  ``[1000, 2003)``; YAL002C (``-``) from ``32 - 3 = 29``: ``[29, 1032)``.
* ``species_downstream`` (300 nt starting 3 nt before the CDS end, so the stop codon
  is included): YAL001W ``[109, 409)``; YAL003W ``[5097, 5397)``; YAL002C (``-``)
  ``[23 - 300, 23)`` clips to ``[0, 23)``.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from Bio.Seq import Seq

import torchcell.datasets.fungal_up_down_transformer as fudt_module
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.models.fungal_up_down_transformer import FungalUpDownTransformer
from torchcell.sequence import ParsedGenome

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
STRAND = {"YAL001W": "+", "YAL002C": "-", "YAL003W": "+"}


def _features(seq: str) -> list[float]:
    """``[len, #A, #C, #G, #T]`` as floats."""
    return [float(len(seq))] + [float(seq.count(b)) for b in "ACGT"]


class _FakeSpeciesLM:
    """Records its model name and every embed call; forward is a fixed map."""

    VALID_MODEL_NAMES = FungalUpDownTransformer.VALID_MODEL_NAMES
    inits: list[str] = []
    calls: list[tuple[list[str], bool]] = []

    def __init__(self, model_name: str) -> None:
        _FakeSpeciesLM.inits.append(model_name)

    def embed(self, sequences: list[str], mean_embedding: bool = True) -> torch.Tensor:
        """``[len, #A, #C, #G, #T]`` per sequence, shaped ``[batch, 5]``."""
        _FakeSpeciesLM.calls.append((list(sequences), mean_embedding))
        return torch.tensor([_features(s) for s in sequences])


class _ExplodingSpeciesLM:
    """A backbone that must never be built (the store is already on disk)."""

    VALID_MODEL_NAMES = FungalUpDownTransformer.VALID_MODEL_NAMES

    def __init__(self, model_name: str) -> None:
        raise AssertionError(f"backbone {model_name} built on a cache hit")


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in the stand-in, clear its logs, report CUDA absent, stay offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeSpeciesLM, "inits", [])
    monkeypatch.setattr(_FakeSpeciesLM, "calls", [])
    monkeypatch.setattr(fudt_module, "FungalUpDownTransformer", _FakeSpeciesLM)


@pytest.mark.parametrize(
    ("model_name", "backbone", "windows"),
    [
        (
            "species_upstream",
            "upstream_species_lm",
            {"YAL001W": (0, 103), "YAL002C": (29, 1032), "YAL003W": (1000, 2003)},
        ),
        (
            "species_downstream",
            "downstream_species_lm",
            {"YAL001W": (109, 409), "YAL002C": (0, 23), "YAL003W": (5097, 5397)},
        ),
    ],
)
def test_build_embeds_the_codon_inclusive_flank_of_each_gene(
    embedding_genome: Any,
    tmp_path: Path,
    model_name: str,
    backbone: str,
    windows: dict[str, tuple[int, int]],
) -> None:
    """Each gene's flank (bounds in the module docstring) is embedded and stored.

    Pins: the dataset name maps to the SpeciesLM revision (``species_upstream`` to
    ``upstream_species_lm``, ``species_downstream`` to ``downstream_species_lm``) and
    the backbone is built once, inside ``process``; ``embed`` receives one
    ``[window]`` per gene in gene-set order with ``mean_embedding=True``; the stored
    embedding is the stand-in's ``[1, 5]`` row; ``dna_windows`` stores the
    ``DnaWindowResult`` itself (not its ``seq`` string, unlike the Nucleotide
    Transformer dataset) with the exact bounds and strand.
    """
    ds = FungalUpDownTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=model_name
    )
    chromosome = embedding_genome.chromosome
    expected = {}
    for gene, (lo, hi) in windows.items():
        piece = chromosome[lo:hi]
        expected[gene] = (
            str(Seq(piece).reverse_complement()) if STRAND[gene] == "-" else piece
        )

    assert _FakeSpeciesLM.inits == [backbone]
    assert _FakeSpeciesLM.calls == [([expected[g]], True) for g in GENE_IDS]
    assert ds._data.embeddings[model_name].shape == (3, 5)
    for gene in GENE_IDS:
        item = ds[gene]
        window = item.dna_windows[model_name]
        assert (window.id, window.strand, window.start_window, window.end_window) == (
            gene,
            STRAND[gene],
            *windows[gene],
        )
        assert window.seq == expected[gene]
        assert item.embeddings[model_name].tolist() == [_features(expected[gene])]


def test_second_construction_reads_store_and_leaves_backbone_unbuilt(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lazy guard: a store on disk is loaded and ``transformer`` stays ``None``.

    The second construction swaps in a backbone that raises when built, so an
    offline node can read an existing store (the reason for the lazy init in
    ``__init__``, lines 53 to 58).
    """
    name = "species_downstream"
    first = FungalUpDownTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=name
    )
    monkeypatch.setattr(fudt_module, "FungalUpDownTransformer", _ExplodingSpeciesLM)

    again = FungalUpDownTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=name
    )

    assert again.transformer is None
    assert torch.equal(again._data.embeddings[name], first._data.embeddings[name])
    assert list(again._data.id) == GENE_IDS
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_no_model_name_builds_nothing(embedding_genome: Any, tmp_path: Path) -> None:
    """``model_name=None`` constructs with no backbone, no data and no store file.

    ``initialize_model`` also returns ``None`` for it; PyG writes only its two marker
    files.
    """
    ds = FungalUpDownTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=None
    )

    assert ds.transformer is None
    assert ds.initialize_model() is None
    assert ds._data is None
    assert _FakeSpeciesLM.inits == []
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "pre_filter.pt",
        "pre_transform.pt",
    ]


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory is created.

    The message separates its two sentences with a space (issue #543).
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        FungalUpDownTransformerDataset(
            root=str(root), genome=embedding_genome, model_name="agnostic_upstream"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'agnostic_upstream'. Valid options are: "
        "species_downstream, species_upstream"
    )
    assert _FakeSpeciesLM.inits == []
    assert not root.exists()


# Phase 24: parse_genome on a merged dataset's None genome


def test_parse_genome_maps_none_to_none_and_a_genome_to_its_gene_set(
    embedding_genome: Any,
) -> None:
    """A dataset merged with ``+`` carries ``genome=None``, and the static parser
    returns None for it; a real genome becomes a ``ParsedGenome`` holding exactly its
    three gene ids.
    """
    assert FungalUpDownTransformerDataset.parse_genome(None) is None
    parsed = FungalUpDownTransformerDataset.parse_genome(embedding_genome)
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS

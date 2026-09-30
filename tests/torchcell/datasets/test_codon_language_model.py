# tests/torchcell/datasets/test_codon_language_model.py
# [[tests.torchcell.datasets.test_codon_language_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_codon_language_model.py
"""``CalmDataset`` on the ``embedding_genome`` stub with the CaLM backbone faked.

2026.09.30, Phase 18. ``CalmDataset.initialize_model`` imports ``calm`` lazily (CaLM is
an optional git-only dependency), so the tests put a stand-in module under
``sys.modules["calm"]`` whose ``CaLM`` records constructions and embed inputs; the real
package is never imported and no weights are loaded. Every store is written under
``tmp_path``.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)`` with CDS = the locus (12 nt); ``YAL002C``
``-`` ``[20, 32)`` with a spliced 9 nt CDS ``ATGGCCTAA``; ``YAL003W`` ``+``
``[2000, 5100)`` (3,100 nt, longer than the 3,072 nt window).

Stand-in forward: ``[len, #A, #C, #G, #T]`` through the identity map, returned in the
real ``embed_sequence`` shape ``[1, 5]``. ``ATGGCCTAA`` gives ``[9, 3, 2, 2, 2]``.

Branches (``codon_language_model.py`` lines 94 to 104):

* a gene of at most 3,072 nt embeds its CDS FASTA record, and the stored window is
  ``window(len(cds))`` on the locus: YAL001W ``[100, 112)``; YAL002C ``[32 - 9, 32)`` =
  ``[23, 32)``, 9 nt of genomic sequence that is NOT the CDS that was embedded;
* a longer gene embeds ``window(3072)`` with the default ``is_max_size=True``, which
  for a window shorter than the gene keeps the 5' end: YAL003W ``[2000, 5072)``.
"""

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from Bio.Seq import Seq

from torchcell.datasets.codon_language_model import CalmDataset
from torchcell.sequence import ParsedGenome

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]


def _features(seq: str) -> list[float]:
    """``[len, #A, #C, #G, #T]`` as floats."""
    return [float(len(seq))] + [float(seq.count(b)) for b in "ACGT"]


class _FakeCaLM:
    """Records constructions and every sequence embedded; forward is a fixed map."""

    inits = 0
    calls: list[str] = []

    def __init__(self) -> None:
        _FakeCaLM.inits += 1

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """``[len, #A, #C, #G, #T]``, shaped ``[1, 5]``."""
        _FakeCaLM.calls.append(sequence)
        return torch.tensor([_features(sequence)])


class _ExplodingCaLM:
    """A backbone that must never be built (the store is already on disk)."""

    def __init__(self) -> None:
        raise AssertionError("CaLM built on a cache hit")


def _calm_module(cls: type) -> types.ModuleType:
    """A stand-in ``calm`` module exposing ``cls`` as ``CaLM``."""
    module = types.ModuleType("calm")
    module.__dict__["CaLM"] = cls
    return module


@pytest.fixture(autouse=True)
def _fake_calm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the stand-in ``calm`` module and clear its logs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeCaLM, "inits", 0)
    monkeypatch.setattr(_FakeCaLM, "calls", [])
    monkeypatch.setitem(sys.modules, "calm", _calm_module(_FakeCaLM))


def test_build_embeds_cds_for_short_genes_and_a_5prime_window_for_long_ones(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Pins the embedded input, the stored window and the stored row for each branch.

    One CaLM is built; ``embed_sequence`` receives YAL001W's CDS, ``ATGGCCTAA`` for
    YAL002C, and ``chrI[2000:5072]`` for YAL003W, in gene-set order; each stored
    embedding is the stand-in's ``[1, 5]`` row of that input.

    Findings: (1) for a spliced gene the stored ``dna_windows`` entry (YAL002C,
    revcomp of ``chrI[23:32]``) is not the sequence that was embedded; (2)
    ``MODEL_TO_WINDOW["calm"]`` carries ``is_max_size=False`` (line 34), which
    ``process`` unpacks and never passes, so YAL003W gets the 5'-anchored
    ``[2000, 5072)`` rather than the symmetric ``[2014, 5086)``. Pinned until the
    window is taken from the embedded sequence and the flag is passed.
    """
    ds = CalmDataset(root=str(tmp_path), genome=embedding_genome)
    chromosome = embedding_genome.chromosome
    embedded = {
        "YAL001W": chromosome[100:112],
        "YAL002C": "ATGGCCTAA",
        "YAL003W": chromosome[2000:5072],
    }
    windows = {
        "YAL001W": ("+", 100, 112, chromosome[100:112]),
        "YAL002C": ("-", 23, 32, str(Seq(chromosome[23:32]).reverse_complement())),
        "YAL003W": ("+", 2000, 5072, chromosome[2000:5072]),
    }

    assert _FakeCaLM.inits == 1
    assert _FakeCaLM.calls == [embedded[g] for g in GENE_IDS]
    assert ds._data.embeddings["calm"].shape == (3, 5)
    assert ds["YAL002C"].embeddings["calm"].tolist() == [[9.0, 3.0, 2.0, 2.0, 2.0]]
    for gene in GENE_IDS:
        item = ds[gene]
        window = item.dna_windows["calm"]
        assert (window.strand, window.start_window, window.end_window, window.seq) == (
            windows[gene]
        )
        assert item.embeddings["calm"].tolist() == [_features(embedded[gene])]
    assert windows["YAL002C"][3] != embedded["YAL002C"]


def test_chunked_saves_merge_to_the_same_store(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``batch_size=1`` writes and re-reads a partial list after every gene.

    The final collated store equals the single-chunk build: same ids in gene-set
    order and the same ``[3, 5]`` embedding tensor.
    """
    whole = CalmDataset(root=str(tmp_path / "whole"), genome=embedding_genome)
    chunked = CalmDataset(
        root=str(tmp_path / "chunked"), genome=embedding_genome, batch_size=1
    )

    assert list(chunked._data.id) == GENE_IDS
    assert torch.equal(chunked._data.embeddings["calm"], whole._data.embeddings["calm"])
    assert [chunked[i].dna_windows["calm"].seq for i in range(3)] == [
        whole[i].dna_windows["calm"].seq for i in range(3)
    ]


def test_second_construction_reads_store_without_building_calm(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is: ``model`` stays ``None`` and CaLM is not built."""
    first = CalmDataset(root=str(tmp_path), genome=embedding_genome)
    monkeypatch.setitem(sys.modules, "calm", _calm_module(_ExplodingCaLM))

    again = CalmDataset(root=str(tmp_path), genome=embedding_genome)

    assert again.model is None
    assert torch.equal(again._data.embeddings["calm"], first._data.embeddings["calm"])
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory or backbone is created."""
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        CalmDataset(root=str(root), genome=embedding_genome, model_name="calm2")

    assert str(excinfo.value) == "Invalid model_name 'calm2'.Valid options are: calm"
    assert _FakeCaLM.inits == 0
    assert not root.exists()

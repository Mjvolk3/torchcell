# tests/torchcell/datasets/test_random_embedding.py
# [[tests.torchcell.datasets.test_random_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_random_embedding.py
"""``RandomEmbeddingDataset`` on the ``embedding_genome`` stub (no backbone at all).

2026.09.30, Phase 18. Every store is written under ``tmp_path``.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)`` with a 12 nt CDS; ``YAL002C`` ``-``
``[20, 32)`` with a 9 nt CDS; ``YAL003W`` ``+`` ``[2000, 5100)`` with a 3,100 nt CDS.

``process`` seeds the GLOBAL torch generator with 42 and draws ``torch.rand(1, W)`` once
per gene in gene-set order, so the three rows are the first ``3 * W`` draws of the
seed-42 stream. For ``W = 10`` the first row is ``[0.8823, 0.9150, 0.3829, 0.9593,
0.3904, 0.6009, 0.2566, 0.7936, 0.9408, 0.1332]`` (torch's CPU generator; these are
generator outputs, not derivable by hand, and are checked against a fresh
``torch.Generator().manual_seed(42)``).

Windows (``random_embedding.py`` lines 94 to 102): a gene no longer than ``W`` gets
``window(len(cds))``, a longer one ``window(W)``; ``window`` keeps the 5' end when the
window is shorter than the locus:

* ``random_1``: every gene is longer than 1 nt, so YAL001W ``[100, 101)``, YAL002C
  (``-``, 5' end is the right end) ``[31, 32)``, YAL003W ``[2000, 2001)``;
* ``random_6579``: every gene fits, so YAL001W ``[100, 112)``, YAL002C
  ``[32 - 9, 32)`` = ``[23, 32)`` (CDS length on the locus), YAL003W ``[2000, 5100)``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from Bio.Seq import Seq

from torchcell.datasets.random_embedding import RandomEmbeddingDataset
from torchcell.sequence import ParsedGenome

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
STRAND = {"YAL001W": "+", "YAL002C": "-", "YAL003W": "+"}
FIRST_ROW_SEED_42 = [
    0.8823,
    0.9150,
    0.3829,
    0.9593,
    0.3904,
    0.6009,
    0.2566,
    0.7936,
    0.9408,
    0.1332,
]


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Put the global torch generator back; ``process`` reseeds it (Finding below)."""
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def _seed_42_stream(rows: int, width: int) -> torch.Tensor:
    """The first ``rows * width`` draws of a fresh seed-42 CPU generator."""
    return torch.rand(rows, width, generator=torch.Generator().manual_seed(42))


def test_rows_are_the_seed_42_stream_regardless_of_the_caller_rng(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Pins the exact first row, the three-row stream, shape and dtype per gene.

    The caller seeds the global generator with 7 first; the stored rows are still the
    seed-42 draws, because ``process`` reseeds. ``ds["<gene>"]`` returns the gene's row
    as ``[1, 10]`` float32.
    """
    torch.manual_seed(7)
    ds = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name="random_10"
    )

    stored = ds._data.embeddings["random_10"]
    assert stored.shape == (3, 10)
    assert stored.dtype == torch.float32
    assert torch.equal(stored, _seed_42_stream(3, 10))
    first = ds["YAL001W"].embeddings["random_10"]
    assert first.shape == (1, 10)
    assert torch.allclose(first, torch.tensor([FIRST_ROW_SEED_42]), atol=5e-5)
    assert torch.equal(ds["YAL003W"].embeddings["random_10"], stored[2:3])


def test_construction_resets_the_global_torch_generator(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: building a store leaves the global generator at seed 42 plus draws.

    ``torch.manual_seed(42)`` at line 89 reseeds the process-wide generator, so after
    a ``random_1`` build (three draws) the caller's next ``torch.rand(1)`` is the
    fourth seed-42 draw, 0.9593, whatever the caller seeded before. Pinned until the
    dataset uses a private ``torch.Generator``.
    """
    torch.manual_seed(123)
    ds = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name="random_1"
    )

    next_draw = torch.rand(1)
    assert torch.equal(ds._data.embeddings["random_1"], _seed_42_stream(3, 1))
    assert torch.equal(next_draw, _seed_42_stream(1, 4)[0, 3:4])
    assert torch.allclose(next_draw, torch.tensor([0.9593]), atol=5e-5)


@pytest.mark.parametrize(
    ("model_name", "width", "windows"),
    [
        (
            "random_1",
            1,
            {"YAL001W": (100, 101), "YAL002C": (31, 32), "YAL003W": (2000, 2001)},
        ),
        (
            "random_6579",
            6579,
            {"YAL001W": (100, 112), "YAL002C": (23, 32), "YAL003W": (2000, 5100)},
        ),
    ],
)
def test_window_and_width_follow_the_model_size(
    embedding_genome: Any,
    tmp_path: Path,
    model_name: str,
    width: int,
    windows: dict[str, tuple[int, int]],
) -> None:
    """The stored window bounds and sequence, and a ``[1, W]`` row, per gene."""
    ds = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=model_name
    )
    chromosome = embedding_genome.chromosome

    assert torch.equal(ds._data.embeddings[model_name], _seed_42_stream(3, width))
    for gene in GENE_IDS:
        lo, hi = windows[gene]
        piece = chromosome[lo:hi]
        seq = str(Seq(piece).reverse_complement()) if STRAND[gene] == "-" else piece
        window = ds[gene].dna_windows[model_name]
        assert (window.strand, window.start_window, window.end_window, window.seq) == (
            STRAND[gene],
            lo,
            hi,
            seq,
        )
        assert ds[gene].embeddings[model_name].shape == (1, width)


def test_chunked_saves_merge_to_the_same_store(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``batch_size=2`` saves a partial list after gene 2 and merges it at gene 3.

    The final store equals the single-chunk build (same ids, same seed-42 rows).
    """
    whole = RandomEmbeddingDataset(
        root=str(tmp_path / "whole"), genome=embedding_genome, model_name="random_10"
    )
    chunked = RandomEmbeddingDataset(
        root=str(tmp_path / "chunked"),
        genome=embedding_genome,
        model_name="random_10",
        batch_size=2,
    )

    assert list(chunked._data.id) == GENE_IDS
    assert torch.equal(
        chunked._data.embeddings["random_10"], whole._data.embeddings["random_10"]
    )


def test_second_construction_reads_store_without_reprocessing(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """A store on disk is loaded as is, and ``process`` (which reseeds) is not run.

    The caller seeds 7 before the second construction; the next ``torch.rand(1)``
    is seed 7's first draw, which proves no reseed to 42 happened, and the stored
    rows are unchanged.
    """
    first = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name="random_10"
    )
    torch.manual_seed(7)
    again = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name="random_10"
    )

    assert torch.equal(
        torch.rand(1), torch.rand(1, generator=torch.Generator().manual_seed(7))
    )
    assert torch.equal(
        again._data.embeddings["random_10"], first._data.embeddings["random_10"]
    )
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_an_interrupted_chunk_file_blocks_the_next_construction(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: a partial ``{"data_list": [...]}`` store is never resumed.

    ``process`` saves intermediate chunks under the final store path (line 126). If a
    build dies between chunks, the next construction sees the path, skips
    ``process``, and ``BaseEmbeddingDataset.__init__`` unpacks the one-key dict into
    ``data, slices``, raising ``ValueError``. The "load existing data" branch
    (lines 116 to 121) therefore only ever merges chunks of the same run. Pinned until
    chunks go to a separate path.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    torch.save({"data_list": []}, processed / "random_10.pt")

    with pytest.raises(ValueError) as excinfo:
        RandomEmbeddingDataset(
            root=str(tmp_path), genome=embedding_genome, model_name="random_10"
        )

    assert str(excinfo.value) == "not enough values to unpack (expected 2, got 1)"


def test_pre_transform_is_baked_into_the_store(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``pre_transform`` runs on each gene's Data before the store is written (lines
    106 to 107), so a transform that scales the embeddings by 10 leaves the stored rows
    at ten times the seed-42 stream, and a second construction without the transform
    reads those scaled rows back (PyG warns that the argument differs).
    """

    def times_ten(data: Any) -> Any:
        data.embeddings = {k: v * 10 for k, v in data.embeddings.items()}
        return data

    ds = RandomEmbeddingDataset(
        root=str(tmp_path),
        genome=embedding_genome,
        model_name="random_10",
        pre_transform=times_ten,
    )
    assert torch.equal(ds._data.embeddings["random_10"], 10 * _seed_42_stream(3, 10))
    with pytest.warns(UserWarning, match="The `pre_transform` argument differs"):
        again = RandomEmbeddingDataset(
            root=str(tmp_path), genome=embedding_genome, model_name="random_10"
        )
    assert torch.equal(again._data.embeddings["random_10"], 10 * _seed_42_stream(3, 10))


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory is created."""
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        RandomEmbeddingDataset(
            root=str(root), genome=embedding_genome, model_name="random_2"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'random_2'.Valid options are: "
        "random_6579, random_1000, random_100, random_10, random_1"
    )
    assert not root.exists()

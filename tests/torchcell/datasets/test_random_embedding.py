# tests/torchcell/datasets/test_random_embedding.py
# [[tests.torchcell.datasets.test_random_embedding]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_random_embedding.py
"""``RandomEmbeddingDataset`` on the ``embedding_genome`` stub (no backbone at all).

2026.09.30, Phase 18. Every store is written under ``tmp_path``.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)`` with a 12 nt CDS; ``YAL002C`` ``-``
``[20, 32)`` with a 9 nt CDS; ``YAL003W`` ``+`` ``[2000, 5100)`` with a 3,100 nt CDS.

``process`` draws ``torch.rand(1, W)`` from a private ``torch.Generator`` seeded with 42
(issue #543; the global generator is untouched) once per gene in gene-set order, so the three rows are the first ``3 * W`` draws of the
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

import logging
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
    """Put the global torch generator back after tests that seed it."""
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
    seed-42 draws, because ``process`` uses its own seeded generator. ``ds["<gene>"]`` returns the gene's row
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


def test_construction_leaves_the_global_torch_generator_alone(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """A build draws from a private generator, so the caller's stream continues.

    The caller seeds 123; after a ``random_1`` build the next ``torch.rand(1)`` is
    seed 123's FIRST draw (issue #543; it used to be seed 42's fourth, 0.9593).
    """
    torch.manual_seed(123)
    ds = RandomEmbeddingDataset(
        root=str(tmp_path), genome=embedding_genome, model_name="random_1"
    )

    next_draw = torch.rand(1)
    assert torch.equal(ds._data.embeddings["random_1"], _seed_42_stream(3, 1))
    assert torch.equal(
        next_draw, torch.rand(1, generator=torch.Generator().manual_seed(123))
    )
    assert not torch.equal(next_draw, _seed_42_stream(1, 4)[0, 3:4])


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

    The final store equals the single-chunk build (same ids, same seed-42 rows), and
    the chunk file ``random_10.partial.pt`` is consumed.
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
    assert sorted(p.name for p in (tmp_path / "chunked" / "processed").iterdir()) == [
        "pre_filter.pt",
        "pre_transform.pt",
        "random_10.pt",
    ]


def test_second_construction_reads_store_without_reprocessing(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """A store on disk is loaded as is: the stored rows are unchanged.

    The caller seeds 7 before the second construction; the next ``torch.rand(1)``
    is seed 7's first draw.
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


def test_an_interrupted_chunk_file_is_removed_with_a_warning(
    embedding_genome: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A chunk file left by an interrupted build no longer blocks the next one.

    Chunks go to ``processed/<name>.partial.pt`` (issue #543), never the store path.
    A stale chunk holding a foreign entry is removed with the exact warning below,
    the build restarts at the first gene, and the store is the seed-42 stream of the
    three genes only.
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    partial = processed / "random_10.partial.pt"
    torch.save({"data_list": ["stale"]}, partial)

    with caplog.at_level(logging.WARNING, logger="torchcell.datasets.random_embedding"):
        ds = RandomEmbeddingDataset(
            root=str(tmp_path), genome=embedding_genome, model_name="random_10"
        )

    assert [r.getMessage() for r in caplog.records] == [
        f"Removing partial chunk file {partial} left by an interrupted build; "
        "rebuilding from the first gene."
    ]
    assert list(ds._data.id) == GENE_IDS
    assert torch.equal(ds._data.embeddings["random_10"], _seed_42_stream(3, 10))
    assert not partial.exists()


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
        "Invalid model_name 'random_2'. Valid options are: "
        "random_6579, random_1000, random_100, random_10, random_1"
    )
    assert not root.exists()

# tests/torchcell/datasets/test_datasets_nucleotide_transformer.py
# [[tests.torchcell.datasets.test_datasets_nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_datasets_nucleotide_transformer.py
"""``NucleotideTransformerDataset`` on the ``embedding_genome`` stub, backbone faked.

2026.09.30, Phase 18. The module-level name
``torchcell.datasets.nucleotide_transformer.NucleotideTransformer`` is replaced by
``_FakeNT`` (no tokenizer or weights are loaded); CUDA is reported absent and
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard. Every store is
written under ``tmp_path``.

2026.09.30, issue #543. ``process`` builds the backbone itself (PyG calls it from
inside ``super().__init__`` only when the store is absent), so a fresh build runs
unaided and a store on disk is read without building a backbone.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)``; ``YAL002C`` ``-`` ``[20, 32)``;
``YAL003W`` ``+`` ``[2000, 5100)``. A ``-`` window is the reverse complement.

Stand-in forward: ``[len, #A, #C, #G, #T]`` of each sequence through the identity map,
in the real wrapper's ``mean_embedding`` shape ``[batch, 5]``; the dataset embeds one
sequence per call and ``torch.cat`` gives ``[3, 5]``, and each gene stores its ``[1, 5]``
row.

Window bounds, derived from ``torchcell.sequence.data``:

* ``nt_window_5979`` (``is_max_size=False``, symmetric): YAL001W flank
  ``(5979 - 12) // 2 = 2983`` clips at 0, symmetric flank ``min(100, 2983) = 100`` gives
  ``[0, 212)``; YAL002C ``min(20, 2983) = 20`` gives ``[0, 52)``; YAL003W flank
  ``(5979 - 3100) // 2 = 1439``, right end clips at 6100, ``min(1439, 1000) = 1000`` gives
  ``[1000, 6100)``.
* ``nt_window_5979_max`` (``is_max_size=True``): YAL001W and YAL002C shift to
  ``[0, 5979)``; YAL003W ``[561, 6539)`` clips right and shifts to ``[121, 6100)``.
* ``nt_window_three_prime_300`` (``include_stop_codon=True``): a ``+`` window starts
  at the stop codon, 3 before the 3' end: YAL001W ``[109, 409)``, YAL003W
  ``[5097, 5397)``; YAL002C (``-``) ``[23 - 300, 23)`` clips to ``[0, 23)``.
* ``nt_window_five_prime_1003`` (``include_start_codon=True``): a ``+`` window ends
  after the start codon: YAL001W ``[103 - 1003, 103)`` clips to ``[0, 103)``, YAL003W
  ``[1000, 2003)``; YAL002C (``-``) ``[32 - 3, 29 + 1003)`` = ``[29, 1032)``.
"""

import os
import sys
import types
from pathlib import Path
from typing import Any

import dotenv
import pytest
import torch
from Bio.Seq import Seq

import torchcell.datasets.nucleotide_transformer as nt_dataset_module
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset, main
from torchcell.sequence import ParsedGenome

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
STRAND = {"YAL001W": "+", "YAL002C": "-", "YAL003W": "+"}
WINDOWS = {
    "nt_window_5979": {
        "YAL001W": (0, 212),
        "YAL002C": (0, 52),
        "YAL003W": (1000, 6100),
    },
    "nt_window_5979_max": {
        "YAL001W": (0, 5979),
        "YAL002C": (0, 5979),
        "YAL003W": (121, 6100),
    },
    "nt_window_three_prime_300": {
        "YAL001W": (109, 409),
        "YAL002C": (0, 23),
        "YAL003W": (5097, 5397),
    },
    "nt_window_five_prime_1003": {
        "YAL001W": (0, 103),
        "YAL002C": (29, 1032),
        "YAL003W": (1000, 2003),
    },
}


def _features(seq: str) -> list[float]:
    """``[len, #A, #C, #G, #T]`` as floats."""
    return [float(len(seq))] + [float(seq.count(b)) for b in "ACGT"]


def _window(chromosome: str, gene: str, bounds: tuple[int, int]) -> str:
    """The chromosome slice, reverse complemented for the ``-`` strand gene."""
    piece = chromosome[bounds[0] : bounds[1]]
    return str(Seq(piece).reverse_complement()) if STRAND[gene] == "-" else piece


class _FakeNT:
    """Records constructions and every embed call; forward is a fixed map."""

    inits = 0
    calls: list[tuple[list[str], bool]] = []

    def __init__(self) -> None:
        _FakeNT.inits += 1

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        """``[len, #A, #C, #G, #T]`` per sequence, shaped ``[batch, 5]``."""
        _FakeNT.calls.append((list(sequences), mean_embedding))
        return torch.tensor([_features(s) for s in sequences])


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in the stand-in class, clear its logs, report CUDA absent, stay offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeNT, "inits", 0)
    monkeypatch.setattr(_FakeNT, "calls", [])
    monkeypatch.setattr(nt_dataset_module, "NucleotideTransformer", _FakeNT)


class _ExplodingNT:
    """A backbone that must never be built (the store is already on disk)."""

    def __init__(self) -> None:
        raise AssertionError("backbone built on a cache hit")


def _build(root: Path, genome: Any, name: str) -> NucleotideTransformerDataset:
    """Build a store from scratch with the stand-in backbone."""
    return NucleotideTransformerDataset(root=str(root), genome=genome, model_name=name)


def test_fresh_build_creates_the_backbone_once_and_writes_the_store(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """With no store on disk, construction builds one backbone inside ``process``.

    Contract (issue #543): the backbone is set to ``None`` before ``super().__init__``
    and built by ``process`` on demand, so a fresh build embeds all three genes with a
    single ``NucleotideTransformer()`` and writes ``processed/<name>.pt`` beside PyG's
    two marker files; the instance keeps that backbone.
    """
    ds = _build(tmp_path, embedding_genome, "nt_window_5979")

    assert _FakeNT.inits == 1
    assert len(_FakeNT.calls) == 3
    assert isinstance(ds.transformer, _FakeNT)
    assert sorted(os.listdir(tmp_path / "processed")) == [
        "nt_window_5979.pt",
        "pre_filter.pt",
        "pre_transform.pt",
    ]


@pytest.mark.parametrize("model_name", list(WINDOWS))
def test_build_embeds_the_exact_window_of_each_gene(
    embedding_genome: Any, tmp_path: Path, model_name: str
) -> None:
    """Each gene's window (bounds in the module docstring) is embedded and stored.

    Pins: ``embed`` receives one ``[window]`` per gene in gene-set order with
    ``mean_embedding=True``; ``dna_windows`` stores the window string; the stored
    embedding is the stand-in's ``[1, 5]`` row of that window; the collated store is
    ``[3, 5]``.

    Prime windows (issue #543): the ``has_special_codon`` flag of ``MODEL_TO_WINDOW``
    is passed on as ``include_stop_codon`` / ``include_start_codon``, so YAL001W's 3'
    window starts at its stop codon (109, not 112) and its 5' window ends after its
    start codon (103). The exact window strings for YAL001W are asserted as literals
    of the chromosome as well, so a shifted slice cannot pass.
    """
    ds = _build(tmp_path, embedding_genome, model_name)
    chromosome = embedding_genome.chromosome
    expected = {g: _window(chromosome, g, WINDOWS[model_name][g]) for g in GENE_IDS}

    assert _FakeNT.calls == [([expected[g]], True) for g in GENE_IDS]
    assert ds._data.embeddings[model_name].shape == (3, 5)
    for i, gene in enumerate(GENE_IDS):
        item = ds[gene]
        assert item.dna_windows == {model_name: expected[gene]}
        assert item.embeddings[model_name].tolist() == [_features(expected[gene])]
        assert torch.equal(ds[i].embeddings[model_name], item.embeddings[model_name])


def test_prime_windows_carry_the_codon_and_no_other_gene_base(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """YAL001W (``+``, CDS ``chrI[100:112]``): the exact 3' and 5' window strings.

    The 3' window is the stop codon ``CDS[9:12]`` followed by the 297 downstream
    bases ``chrI[112:409]``; the 5' window is all 100 upstream bases ``chrI[0:100]``
    followed by the start codon ``CDS[0:3]``, and nothing else of the gene.
    """
    cds = str(embedding_genome["YAL001W"].cds.seq)
    chromosome = embedding_genome.chromosome
    assert cds == chromosome[100:112]
    three = _build(tmp_path / "three", embedding_genome, "nt_window_three_prime_300")
    five = _build(tmp_path / "five", embedding_genome, "nt_window_five_prime_1003")

    three_window = three["YAL001W"].dna_windows["nt_window_three_prime_300"]
    five_window = five["YAL001W"].dna_windows["nt_window_five_prime_1003"]
    assert three_window == cds[9:12] + chromosome[112:409]
    assert five_window == chromosome[0:100] + cds[0:3]
    assert (len(three_window), len(five_window)) == (300, 103)


def test_second_construction_reads_store_without_building_backbone(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is; no backbone is built and ``process`` is not run.

    The second construction swaps in a backbone that raises when built; the load
    succeeds with ``transformer`` still ``None`` and the rows match the first build.
    """
    name = "nt_window_three_prime_300"
    first = _build(tmp_path, embedding_genome, name)
    monkeypatch.setattr(nt_dataset_module, "NucleotideTransformer", _ExplodingNT)

    again = NucleotideTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=name
    )

    assert again.transformer is None
    assert _FakeNT.inits == 1
    assert torch.equal(again._data.embeddings[name], first._data.embeddings[name])
    assert list(again._data.id) == GENE_IDS
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_initialize_model_builds_the_backbone_only_for_a_named_model(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``initialize_model`` returns a backbone for a named dataset, ``None`` otherwise.

    ``model_name=None`` constructs without a backbone and without data (``process``
    returns first); the named dataset's ``initialize_model`` constructs exactly one
    ``NucleotideTransformer`` (no arguments).
    """
    unnamed = NucleotideTransformerDataset(
        root=str(tmp_path / "none"), genome=embedding_genome, model_name=None
    )
    assert unnamed.initialize_model() is None
    assert unnamed._data is None
    assert _FakeNT.inits == 0

    named = _build(tmp_path / "named", embedding_genome, "nt_window_5979")
    backbone = named.initialize_model()
    assert isinstance(backbone, _FakeNT)
    assert _FakeNT.inits == 2  # the one process built, then this one


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory is created, naming the
    bad name and then the valid names after a space.
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        NucleotideTransformerDataset(
            root=str(root), genome=embedding_genome, model_name="nt_window_1"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'nt_window_1'. Valid options are: nt_window_5979_max, "
        "nt_window_5979, nt_window_three_prime_5979, nt_window_five_prime_5979, "
        "nt_window_three_prime_300, nt_window_five_prime_1003"
    )
    assert not root.exists()


def test_main_builds_the_six_window_datasets_in_order(  # test-quality: allow main() returns None; its effects are asserted on the recorders
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``main`` logs to wandb, loads ``.env``, then builds each window dataset in turn.

    ``wandb`` (a ``sys.modules`` stand-in), ``dotenv.load_dotenv``, the genome class
    and the dataset class are all recorders, and ``DATA_ROOT`` is ``tmp_path``, so
    nothing is read or written. Pins the call order, the genome root
    ``<DATA_ROOT>/data/sgd/genome``, the shared store root
    ``<DATA_ROOT>/data/scerevisiae/nucleotide_transformer_embed``, the six names in
    build order, and one ``{"event": i}`` log before each build.
    """
    log: list[tuple[Any, ...]] = []
    fake_wandb = types.ModuleType("wandb")
    setattr(fake_wandb, "init", lambda **kw: log.append(("wandb.init", kw)))
    setattr(fake_wandb, "log", lambda payload: log.append(("wandb.log", payload)))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: log.append(("load_dotenv",)))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    def fake_genome(data_root: str) -> str:
        log.append(("genome", data_root))
        return "GENOME"

    monkeypatch.setattr(nt_dataset_module, "SCerevisiaeGenome", fake_genome)
    monkeypatch.setattr(
        nt_dataset_module,
        "NucleotideTransformerDataset",
        lambda root, genome, model_name: log.append(
            ("dataset", root, genome, model_name)
        ),
    )

    main()

    root = str(tmp_path / "data/scerevisiae/nucleotide_transformer_embed")
    names = [
        "nt_window_5979",
        "nt_window_5979_max",
        "nt_window_three_prime_5979",
        "nt_window_five_prime_5979",
        "nt_window_three_prime_300",
        "nt_window_five_prime_1003",
    ]
    expected: list[tuple[Any, ...]] = [
        ("wandb.init", {"mode": "online", "project": "torchcell_embeddings"}),
        ("load_dotenv",),
        ("genome", str(tmp_path / "data/sgd/genome")),
    ]
    for i, name in enumerate(names):
        expected += [("wandb.log", {"event": i}), ("dataset", root, "GENOME", name)]
    assert log == expected

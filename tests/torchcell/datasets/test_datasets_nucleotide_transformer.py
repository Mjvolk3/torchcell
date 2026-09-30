# tests/torchcell/datasets/test_datasets_nucleotide_transformer.py
# [[tests.torchcell.datasets.test_datasets_nucleotide_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_datasets_nucleotide_transformer.py
"""``NucleotideTransformerDataset`` on the ``embedding_genome`` stub, backbone faked.

2026.09.30, Phase 18. The module-level name
``torchcell.datasets.nucleotide_transformer.NucleotideTransformer`` is replaced by
``_FakeNT`` (no tokenizer or weights are loaded); CUDA is reported absent and
``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` are set as a second guard. Every store is
written under ``tmp_path``.

A fresh build cannot run as shipped: ``process`` reads ``self.transformer``, which
``__init__`` only assigns after ``super().__init__`` has already called ``process``
(Finding, first test). The window tests therefore put a stand-in instance on the CLASS
attribute ``transformer`` (``monkeypatch.setattr(..., raising=False)``, no subclass), so
``process`` runs the production window code unchanged.

Genome (``tests/torchcell/conftest.py``, chromosome I of 6,100 nt, 0-based half-open
slices): ``YAL001W`` ``+`` gene ``[100, 112)``; ``YAL002C`` ``-`` ``[20, 32)``;
``YAL003W`` ``+`` ``[2000, 5100)``. A ``-`` window is the reverse complement.

Stand-in forward: ``[len, #A, #C, #G, #T]`` of each sequence through the identity map,
in the real wrapper's ``mean_embedding`` shape ``[1, batch, 5]``; the dataset embeds one
sequence per call and ``torch.cat`` gives ``[3, 1, 5]``, so each gene stores ``[1, 5]``.

Window bounds, derived from ``torchcell.sequence.data``:

* ``nt_window_5979`` (``is_max_size=False``, symmetric): YAL001W flank
  ``(5979 - 12) // 2 = 2983`` clips at 0, symmetric flank ``min(100, 2983) = 100`` gives
  ``[0, 212)``; YAL002C ``min(20, 2983) = 20`` gives ``[0, 52)``; YAL003W flank
  ``(5979 - 3100) // 2 = 1439``, right end clips at 6100, ``min(1439, 1000) = 1000`` gives
  ``[1000, 6100)``.
* ``nt_window_5979_max`` (``is_max_size=True``): YAL001W and YAL002C shift to
  ``[0, 5979)``; YAL003W ``[561, 6539)`` clips right and shifts to ``[121, 6100)``.
* ``nt_window_three_prime_300``: ``+`` genes start at their 3' end, YAL001W
  ``[112, 412)``, YAL003W ``[5100, 5400)``; YAL002C ``[20 - 300, 20)`` clips to
  ``[0, 20)``.
* ``nt_window_five_prime_1003``: YAL001W ``[101 - 1003, 101)`` clips to ``[0, 101)``;
  YAL003W ``[998, 2001)``; YAL002C ``[32, 1035)``.
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
        "YAL001W": (112, 412),
        "YAL002C": (0, 20),
        "YAL003W": (5100, 5400),
    },
    "nt_window_five_prime_1003": {
        "YAL001W": (0, 101),
        "YAL002C": (32, 1035),
        "YAL003W": (998, 2001),
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
        """``[len, #A, #C, #G, #T]`` per sequence, shaped ``[1, batch, 5]``."""
        _FakeNT.calls.append((list(sequences), mean_embedding))
        return torch.tensor([_features(s) for s in sequences]).unsqueeze(0)


@pytest.fixture(autouse=True)
def _fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in the stand-in class, clear its logs, report CUDA absent, stay offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_FakeNT, "inits", 0)
    monkeypatch.setattr(_FakeNT, "calls", [])
    monkeypatch.setattr(nt_dataset_module, "NucleotideTransformer", _FakeNT)


def _build(
    monkeypatch: pytest.MonkeyPatch, root: Path, genome: Any, name: str
) -> NucleotideTransformerDataset:
    """Build a store with a class-level stand-in, then remove it again."""
    monkeypatch.setattr(
        NucleotideTransformerDataset, "transformer", _FakeNT(), raising=False
    )
    ds = NucleotideTransformerDataset(root=str(root), genome=genome, model_name=name)
    monkeypatch.delattr(NucleotideTransformerDataset, "transformer")
    return ds


def test_fresh_build_fails_before_the_backbone_is_created(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """Finding: with no store on disk, construction raises ``AttributeError``.

    PyG's ``InMemoryDataset.__init__`` calls ``process`` from inside
    ``super().__init__`` (``nucleotide_transformer.py`` line 53), and ``process`` reads
    ``self.transformer`` (line 112), which is only assigned at line 60, after that
    call. The backbone class is never constructed and no store is written. The sibling
    ``fungal_up_down_transformer.py`` fixed the same ordering by building the model
    inside ``process``. Pinned until this dataset does the same.
    """
    with pytest.raises(AttributeError) as excinfo:
        NucleotideTransformerDataset(
            root=str(tmp_path), genome=embedding_genome, model_name="nt_window_5979"
        )

    assert str(excinfo.value) == (
        "'NucleotideTransformerDataset' object has no attribute 'transformer'"
    )
    assert _FakeNT.inits == 0
    assert os.listdir(tmp_path / "processed") == []


@pytest.mark.parametrize("model_name", list(WINDOWS))
def test_build_embeds_the_exact_window_of_each_gene(
    embedding_genome: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """Each gene's window (bounds in the module docstring) is embedded and stored.

    Pins: ``embed`` receives one ``[window]`` per gene in gene-set order with
    ``mean_embedding=True``; ``dna_windows`` stores the window string; the stored
    embedding is the stand-in's ``[1, 5]`` row of that window; the collated store is
    ``[3, 5]``.

    Findings on the prime windows: ``MODEL_TO_WINDOW`` carries a ``has_special_codon``
    flag (``True``) that ``process`` unpacks and never passes (line 117 to 119), so the
    3' window starts after the stop codon (YAL001W at 112, not 109) and the 5' window
    omits the start-codon extension; and on the ``+`` strand the 5' window ends at the
    1-based start (``s288c.py`` line 287), so YAL001W's window ``[0, 101)`` includes
    the gene's first base ``chrI[100]``. Pinned until the flag is passed and the bound
    is ``start - 1``.
    """
    ds = _build(monkeypatch, tmp_path, embedding_genome, model_name)
    chromosome = embedding_genome.chromosome
    expected = {g: _window(chromosome, g, WINDOWS[model_name][g]) for g in GENE_IDS}

    assert _FakeNT.calls == [([expected[g]], True) for g in GENE_IDS]
    assert ds._data.embeddings[model_name].shape == (3, 5)
    for gene in GENE_IDS:
        item = ds[gene]
        assert item.dna_windows == {model_name: expected[gene]}
        assert item.embeddings[model_name].tolist() == [_features(expected[gene])]


def test_second_construction_reads_store_without_building_backbone(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store on disk is loaded as is; neither the class nor ``process`` is used.

    After the first build the class-level stand-in is removed, so a second
    ``process`` would raise ``AttributeError``; the load succeeds, the backbone count
    stays at the one instance ``_build`` made, and the rows match the first build.
    """
    name = "nt_window_three_prime_300"
    first = _build(monkeypatch, tmp_path, embedding_genome, name)

    again = NucleotideTransformerDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=name
    )

    assert _FakeNT.inits == 1
    assert torch.equal(again._data.embeddings[name], first._data.embeddings[name])
    assert list(again._data.id) == GENE_IDS
    parsed = again.genome
    assert isinstance(parsed, ParsedGenome)
    assert list(parsed.gene_set) == GENE_IDS


def test_initialize_model_builds_the_backbone_only_for_a_named_model(
    embedding_genome: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    named = _build(monkeypatch, tmp_path / "named", embedding_genome, "nt_window_5979")
    backbone = named.initialize_model()
    assert isinstance(backbone, _FakeNT)
    assert _FakeNT.inits == 2  # the class-level instance in _build, then this one


def test_unknown_model_name_is_refused_with_the_valid_list(
    embedding_genome: Any, tmp_path: Path
) -> None:
    """``BaseEmbeddingDataset`` raises before any directory is created (no space
    after the first sentence, the ``embedding.py`` Finding).
    """
    root = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        NucleotideTransformerDataset(
            root=str(root), genome=embedding_genome, model_name="nt_window_1"
        )

    assert str(excinfo.value) == (
        "Invalid model_name 'nt_window_1'.Valid options are: nt_window_5979_max, "
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

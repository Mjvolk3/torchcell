# tests/torchcell/datasets/test_genslm.py
# [[tests.torchcell.datasets.test_genslm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_genslm.py
"""``GenSLMDataset`` on the ``embedding_genome`` stub with the GenSLM backbone faked.

``GenSLMDataset.initialize_model`` is patched to return a stand-in whose
``embed(sequences, mean_embedding=True)`` maps each sequence to
``[len, #A, #C, #G, #T]``, so no checkpoint is read. Every store is written under
``tmp_path``.

Genome (``tests/torchcell/conftest.py``): ``YAL001W`` CDS = ``chrI[100:112]`` (12 nt);
``YAL002C`` spliced CDS ``ATGGCCTAA``; ``YAL003W`` CDS = ``chrI[2000:5100]`` (3,100 nt,
shorter than the 6,144 nt window, so embedded whole).

Contract: every gene with a CDS embeds its spliced CDS truncated to the first
6,144 nt in frame from the start codon, ``dna_windows`` stores exactly that string,
and a gene whose ``cds`` is ``None`` is left out of the store and listed in
``processed/<model_name>.no_cds.json``.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from torchcell.datasets.genslm import GenSLMDataset
from torchcell.sequence import ParsedGenome
from torchcell.sequence.data import GeneSet

GENE_IDS = ["YAL001W", "YAL002C", "YAL003W"]
MODEL = "genslm_25M_patric"


def _features(seq: str) -> list[float]:
    return [float(len(seq))] + [float(seq.count(b)) for b in "ACGT"]


class _FakeGenSLM:
    """Records every batch embedded; forward is a fixed per-sequence map."""

    inits = 0
    batches: list[list[str]] = []

    def __init__(self) -> None:
        _FakeGenSLM.inits += 1

    def embed(self, sequences: list[str], mean_embedding: bool = False) -> torch.Tensor:
        assert mean_embedding
        _FakeGenSLM.batches.append(list(sequences))
        return torch.tensor([_features(s) for s in sequences])


@pytest.fixture(autouse=True)
def fake_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGenSLM.inits = 0
    _FakeGenSLM.batches = []
    monkeypatch.setattr(GenSLMDataset, "initialize_model", lambda self: _FakeGenSLM())


class _Gene:
    def __init__(self, cds: str | None) -> None:
        self.cds = None if cds is None else SeqRecord(Seq(cds), id="x")


class _RnaGenome:
    """Two coding genes and one rRNA gene (no CDS), plus one gene past the window."""

    def __init__(self) -> None:
        self.genes = {
            "PP_0001": _Gene("ATGGCCGTCAAGTAA"),
            "PP_16SA": _Gene(None),
            "PP_0002": _Gene("ATG" + "GCC" * 2100 + "TAA"),
        }
        self.gene_set = GeneSet(self.genes)

    def __getitem__(self, gene_id: str) -> Any:
        return self.genes[gene_id]


def test_store_holds_every_cds_gene_with_its_window(
    tmp_path: Path, embedding_genome: Any
) -> None:
    dataset = GenSLMDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=MODEL
    )
    assert _FakeGenSLM.inits == 1
    assert len(dataset) == 3
    assert isinstance(dataset.genome, ParsedGenome)
    assert list(dataset.genome.gene_set) == GENE_IDS
    chromosome = embedding_genome.chromosome
    expected = {
        "YAL001W": chromosome[100:112],
        "YAL002C": "ATGGCCTAA",
        "YAL003W": chromosome[2000:5100],
    }
    for gene_id, window in expected.items():
        item = dataset[gene_id]
        assert item.dna_windows[MODEL] == window
        assert item.embeddings[MODEL].shape == (1, 5)
        assert item.embeddings[MODEL][0].tolist() == _features(window)
    assert (
        json.loads((tmp_path / "processed" / f"{MODEL}.no_cds.json").read_text()) == []
    )


def test_batches_follow_gene_set_order_and_batch_size(
    tmp_path: Path, embedding_genome: Any
) -> None:
    GenSLMDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=MODEL, batch_size=2
    )
    chromosome = embedding_genome.chromosome
    assert _FakeGenSLM.batches == [
        [chromosome[100:112], "ATGGCCTAA"],
        [chromosome[2000:5100]],
    ]


def test_rna_genes_are_left_out_and_recorded_and_long_cds_truncated(
    tmp_path: Path,
) -> None:
    genome = _RnaGenome()
    dataset = GenSLMDataset(root=str(tmp_path), genome=genome, model_name=MODEL)
    assert len(dataset) == 2
    assert json.loads(
        dataset.no_cds_path and Path(dataset.no_cds_path).read_text()
    ) == ["PP_16SA"]
    with pytest.raises(KeyError, match="PP_16SA"):
        dataset["PP_16SA"]
    window = dataset["PP_0002"].dna_windows[MODEL]
    assert len(window) == 6144
    assert window == ("ATG" + "GCC" * 2100 + "TAA")[:6144]


def test_existing_store_loads_without_building_the_backbone(
    tmp_path: Path, embedding_genome: Any
) -> None:
    GenSLMDataset(root=str(tmp_path), genome=embedding_genome, model_name=MODEL)
    assert _FakeGenSLM.inits == 1
    reloaded = GenSLMDataset(
        root=str(tmp_path), genome=embedding_genome, model_name=MODEL
    )
    assert _FakeGenSLM.inits == 1
    assert len(reloaded) == 3


def test_unknown_model_name_is_refused(tmp_path: Path, embedding_genome: Any) -> None:
    with pytest.raises(ValueError, match="Invalid model_name 'genslm_nope'"):
        GenSLMDataset(
            root=str(tmp_path), genome=embedding_genome, model_name="genslm_nope"
        )

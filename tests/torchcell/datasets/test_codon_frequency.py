# tests/torchcell/datasets/test_codon_frequency.py
# [[tests.torchcell.datasets.test_codon_frequency]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_codon_frequency.py
"""``CodonFrequencyDataset`` on a three-gene stub genome, built under ``tmp_path``.

2026.09.30, issue #570. The stub exposes only what ``process`` reads: a ``GeneSet``
and ``genome[gene_id].cds.seq``. Genes: ``YAA001W`` with CDS ``ATGAAATAA`` (three
codons ATG, AAA, TAA, each 1/3), ``YAA002W`` with an empty CDS, ``YAA003W`` with CDS
``ATGATG`` (ATG = 1). ``compute_codon_frequency`` refuses the empty CDS with
``Empty CDS string; a codon frequency needs at least one codon.``

Contract: the refused gene is excluded from the dataset; the skip is logged at WARNING
as ``skipping YAA002W: <message>`` and the total once at the end as ``skipped 1
gene(s) whose CDS has no codon frequency``. A genome with no refused gene logs no
warning.
"""

import logging
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchcell.datasets.codon_frequency import CodonFrequencyDataset
from torchcell.sequence import GeneSet, ParsedGenome


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the store on the CPU whatever the host has."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


CODONS = ["".join(c) for c in product("ATGC", repeat=3)]


class _StubGenome:
    """``gene_set`` plus ``genome[gene_id].cds.seq``, nothing else."""

    def __init__(self, cds: dict[str, str]) -> None:
        self.gene_set = GeneSet(cds)
        self._cds = cds

    def __getitem__(self, gene_id: str) -> SimpleNamespace:
        return SimpleNamespace(cds=SimpleNamespace(seq=self._cds[gene_id]))


def _vector(freqs: dict[str, float]) -> list[float]:
    """The 64-codon vector in ``SortedDict`` (alphabetical) order."""
    return [freqs.get(codon, 0.0) for codon in sorted(CODONS)]


def test_empty_cds_gene_is_logged_counted_and_excluded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """YAA002W is skipped with the exact refusal; the other two genes are stored."""
    genome = _StubGenome({"YAA001W": "ATGAAATAA", "YAA002W": "", "YAA003W": "ATGATG"})
    with caplog.at_level(logging.WARNING, logger="torchcell.datasets.codon_frequency"):
        dataset = CodonFrequencyDataset(root=str(tmp_path), genome=genome)  # type: ignore[arg-type]
    assert [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if r.name == "torchcell.datasets.codon_frequency"
    ] == [
        (
            "WARNING",
            "skipping YAA002W: Empty CDS string; a codon frequency needs at least "
            "one codon.",
        ),
        ("WARNING", "skipped 1 gene(s) whose CDS has no codon frequency"),
    ]
    assert len(dataset) == 2
    assert [dataset[i].id for i in range(2)] == ["YAA001W", "YAA003W"]
    with pytest.raises(KeyError, match="Gene YAA002W not found in the dataset."):
        dataset["YAA002W"]
    third = 1.0 / 3.0
    torch.testing.assert_close(
        dataset["YAA001W"].embeddings["cds_codon_frequency"],
        torch.tensor([_vector({"ATG": third, "AAA": third, "TAA": third})]),
    )
    torch.testing.assert_close(
        dataset["YAA003W"].embeddings["cds_codon_frequency"],
        torch.tensor([_vector({"ATG": 1.0})]),
    )


def test_no_refused_gene_logs_no_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With every CDS valid, nothing is skipped and no warning is emitted."""
    genome = _StubGenome({"YAA001W": "ATGAAATAA"})
    with caplog.at_level(logging.WARNING, logger="torchcell.datasets.codon_frequency"):
        dataset = CodonFrequencyDataset(root=str(tmp_path), genome=genome)  # type: ignore[arg-type]
    assert [
        r for r in caplog.records if r.name == "torchcell.datasets.codon_frequency"
    ] == []
    assert [dataset[i].id for i in range(len(dataset))] == ["YAA001W"]


# Phase 24: parse_genome(None), initialize_model and pre_transform


def test_parse_genome_none_and_a_stub_genome_give_none_and_its_gene_set() -> None:
    """``parse_genome(None)`` is None and a stub genome gives a ``ParsedGenome`` holding
    exactly its gene set. (``initialize_model`` is a bare ``return None`` with nothing
    to pin.)
    """
    genome = _StubGenome({"YAA001W": "ATGAAATAA"})
    assert CodonFrequencyDataset.parse_genome(None) is None
    assert CodonFrequencyDataset.parse_genome(genome) == ParsedGenome(  # type: ignore[arg-type]
        gene_set=GeneSet(["YAA001W"])
    )


def test_pre_transform_is_applied_before_the_store_is_written(tmp_path: Path) -> None:
    """A pre-transform that stamps ``tag = id + '!'`` and scales the vector by 2: every
    stored record carries the stamp, and ``ATGATG`` (ATG = 1) is stored as 2.0.
    """

    def stamp(data: Any) -> Any:
        data.tag = data.id + "!"
        data.embeddings = {k: 2 * v for k, v in data.embeddings.items()}
        return data

    genome = _StubGenome({"YAA001W": "ATGAAATAA", "YAA003W": "ATGATG"})
    dataset = CodonFrequencyDataset(
        root=str(tmp_path),
        genome=genome,  # type: ignore[arg-type]
        pre_transform=stamp,
    )
    assert [dataset[i].tag for i in range(2)] == ["YAA001W!", "YAA003W!"]
    torch.testing.assert_close(
        dataset["YAA003W"].embeddings["cds_codon_frequency"],
        torch.tensor([_vector({"ATG": 2.0})]),
    )

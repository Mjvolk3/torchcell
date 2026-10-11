# tests/torchcell/datasets/_sga_genome_stub.py
# [[tests.torchcell.datasets._sga_genome_stub]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/_sga_genome_stub.py
"""A genome stub on which every SGA systematic-name pin and refusal holds.

The Costanzo 2016 and Kuzmin 2020 loaders re-check all 16 rename pins and 35 refusals
of ``systematic_name_currency`` against the injected genome before reading a row, so a
hermetic build needs a genome that shows exactly what those tables state, plus the live
genes of its fixture. :class:`SgaGenomeStub` derives that genome from the tables
themselves: each renamed name is an alias of its ORF alone, each merged name resolves to
its locus (a gene, or one of the four non-gene loci), each retired name and every name
not given as live resolves as retired.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torchcell.datasets.gene_alias_resolution import GeneNameRefusalReason
from torchcell.datasets.systematic_name_currency import (
    SGA_SYSTEMATIC_REFUSALS,
    SGA_SYSTEMATIC_RENAMES,
)
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus

#: The four loci the merged names fold into that are not ``gene`` features in R64-4-1.
NON_GENE_LOCI = frozenset({"YER109C", "YFL056C", "YIL167W", "YIR043C"})


class SgaGenomeStub:
    """``gene_set`` / ``alias_to_systematic`` / ``feature_index`` / ``resolve_gene_name``."""

    def __init__(self, live: Iterable[str]) -> None:
        """Live genes: ``live`` plus every gene a pin or a merge names."""
        merged = [
            r
            for r in SGA_SYSTEMATIC_REFUSALS.values()
            if r.reason is GeneNameRefusalReason.MERGED_INTO_LOCUS
        ]
        loci = {r.locus for r in merged if r.locus is not None}
        self.gene_set = (
            set(live)
            | {pin.systematic_name for pin in SGA_SYSTEMATIC_RENAMES.values()}
            | (loci - NON_GENE_LOCI)
        )
        self.alias_to_systematic: dict[str, list[str]] = {
            alias: [pin.systematic_name]
            for alias, pin in SGA_SYSTEMATIC_RENAMES.items()
        }
        self.feature_index: dict[str, Any] = {"standard_to_ids": {}}
        self._aliases = {
            **{a: p.systematic_name for a, p in SGA_SYSTEMATIC_RENAMES.items()},
            **{r.name: r.locus for r in merged if r.locus is not None},
        }

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """The R64 layers, over the stub's tables."""
        if name in self.gene_set:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.CURRENT, systematic_name=name
            )
        if name in NON_GENE_LOCI:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=name,
            )
        if name in self._aliases:
            locus = self._aliases[name]
            status = (
                GeneNameStatus.NON_GENE_FEATURE
                if locus in NON_GENE_LOCI
                else GeneNameStatus.RENAMED
            )
            return GeneNameResolution(
                input_name=name, status=status, systematic_name=locus
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=name
        )

# torchcell/sequence/plasmid
# [[torchcell.sequence.plasmid]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/plasmid
# Test file: tests/torchcell/sequence/test_plasmid.py
"""Plasmid / vector sequence store (WS10 total-genomic-content-in-cell).

We model the DNA present in (or used to construct) a strain as sequence contigs so that
downstream embedding datasets can extract a gene's sequence and its flanking context.
This module parses GenBank plasmid maps into typed, provenance-linked records.

PRINCIPLE (total genomic content in the cell during phenotype collection):
    - A plasmid is captured as PRESENT CONTENT only if it is physically in the cell
      while the phenotype is measured (e.g. an episomal 2-micron plasmid retained under
      selection). Such a plasmid is a genuine additional contig of the cell's genome.
    - A plasmid used only to CONSTRUCT a strain (an integrative delivery vector, a
      CRISPR/sgRNA vector) is NOT present content. Its retained effect is a chromosomal
      sequence MANIPULATION: the integrated segment (between the integration-site
      homology arms) is inserted into the chromosome, and that edit is what a
      ``GeneAddition`` perturbation applies to the sequence. The construction plasmid is
      the SOURCE we extract the inserted segment from, not a stored cell contig.

This store holds parsed plasmid sequences regardless of role; whether a given plasmid is
present-content vs a construction-source is a property of how a perturbation references
it (``GeneAddition.localization``), not of the sequence itself. See
``[[torchcell.sequence.plasmid-and-genomic-content-design]]``.
"""

from __future__ import annotations

import hashlib
import os.path as osp

from Bio import SeqIO
from Bio.Seq import Seq

from torchcell.datamodels.pydant import ModelStrict


class PlasmidFeature(ModelStrict):
    """One annotated feature on a plasmid (0-based half-open coordinates)."""

    feature_type: str
    label: str
    start: int
    end: int
    strand: int


class PlasmidProvenance(ModelStrict):
    """Where a parsed plasmid came from (the hash-pinned mirrored GenBank)."""

    source_file: str
    sha256: str
    citation_key: str


class PlasmidSequence(ModelStrict):
    """A parsed plasmid/vector: full sequence + annotated features + provenance."""

    plasmid_id: str
    name: str
    topology: str
    length: int
    sequence: str
    features: list[PlasmidFeature]
    provenance: PlasmidProvenance

    def get_feature(self, label: str) -> PlasmidFeature:
        """Return the single feature with ``label`` (raise if 0 or >1 match)."""
        hits = [f for f in self.features if f.label == label]
        if len(hits) != 1:
            raise KeyError(f"expected exactly one feature '{label}', found {len(hits)}")
        return hits[0]

    def _slice(self, start: int, end: int) -> str:
        """Slice ``[start, end)`` honouring circular wrap for a circular plasmid."""
        if self.topology == "circular":
            return "".join(self.sequence[i % self.length] for i in range(start, end))
        s, e = max(0, start), min(self.length, end)
        return self.sequence[s:e]

    def feature_sequence(self, label: str, flank: int = 0) -> str:
        """Sequence of ``label`` extended by ``flank`` bp on each side.

        Returns the coding-strand sequence: reverse-complemented when the feature is on
        the minus strand, so the returned string reads 5'->3' for the feature. ``flank``
        adds upstream/downstream context (e.g. promoter/terminator), wrapping around a
        circular plasmid as needed.
        """
        feat = self.get_feature(label)
        window = self._slice(feat.start - flank, feat.end + flank)
        if feat.strand == -1:
            return str(Seq(window).reverse_complement())  # type: ignore[no-untyped-call]  # Bio.Seq is untyped
        return window


def _sha256(path: str) -> str:
    """Streamed sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_plasmid_genbank(path: str, citation_key: str) -> PlasmidSequence:
    """Parse one GenBank plasmid map into a ``PlasmidSequence`` with provenance."""
    record = SeqIO.read(path, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
    features: list[PlasmidFeature] = []
    for f in record.features:
        if f.type == "source":
            continue
        label = f.qualifiers.get("label", f.qualifiers.get("gene", [""]))[0]
        features.append(
            PlasmidFeature(
                feature_type=f.type,
                label=str(label),
                start=int(f.location.start),
                end=int(f.location.end),
                strand=1 if f.location.strand is None else int(f.location.strand),
            )
        )
    return PlasmidSequence(
        plasmid_id=str(record.name),
        name=str(record.name),
        topology=str(record.annotations.get("topology", "linear")),
        length=len(record.seq),
        sequence=str(record.seq),
        features=features,
        provenance=PlasmidProvenance(
            source_file=osp.basename(path),
            sha256=_sha256(path),
            citation_key=citation_key,
        ),
    )

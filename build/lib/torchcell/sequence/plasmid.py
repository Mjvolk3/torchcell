# torchcell/sequence/plasmid
# [[torchcell.sequence.plasmid]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/plasmid
# Test file: tests/torchcell/sequence/test_plasmid.py
"""Plasmid / vector sequence store, SBOL-aligned (WS10 total-genomic-content-in-cell).

We model the DNA present in (or used to construct) a strain as owned pydantic objects so
downstream embedding datasets can extract a gene's sequence + flanking context, and so
strain/plasmid DESIGN can be represented (plasmid design is foundational to future work).
The representation is aligned to the **Synthetic Biology Open Language (SBOL)** and the
**Sequence Ontology (SO)** so we interoperate with the standards while owning a SUPERSET
of them (see ``[[torchcell.sequence.plasmid-and-genomic-content-design]]``,
``[[paper.north-star]]``):

- ``Component`` == an SBOL DNA Component: a design (a whole plasmid, a cassette, or a
  single part) carrying a sequence, SO-typed ``roles``, and located ``features``.
- ``Feature`` == an SBOL SequenceFeature / SubComponent: a located, SO-role-annotated
  region, hierarchically composable via ``sub_features`` (part -> cassette -> plasmid).
- ``Location`` == an SBOL Range: half-open ``[start, end)`` coordinates + orientation.
- ``SORole`` == a Sequence Ontology term (``SO:NNNNNNN`` + name) for interoperability.

PRINCIPLE (total genomic content in the cell during phenotype collection): a plasmid is
captured as PRESENT CONTENT only if it is physically in the cell while the phenotype is
measured; a plasmid used only to CONSTRUCT a strain is a SOURCE whose integrated segment
becomes a chromosomal edit (a ``GeneAddition`` perturbation), not stored cell content.
Whether a Component is present-content vs construction-source is a property of how a
perturbation references it, not of the sequence itself.

We INGEST GenBank first (BioPython), and the owned model can EMIT GenBank/GFF3/SBOL for
interoperability (writers are future). ``feature_sequence`` + ``subcomponent`` provide the
extraction and design-composition operations the store is built for.
"""

from __future__ import annotations

import hashlib
import os.path as osp
from typing import Literal

from Bio import SeqIO
from Bio.Seq import Seq

from torchcell.datamodels.pydant import ModelStrict


class SORole(ModelStrict):
    """A Sequence Ontology term identifying a part/feature role."""

    so_id: str
    name: str


# GenBank feature type -> Sequence Ontology role (the roles we encounter on plasmid maps).
_SO_UNKNOWN = SORole(so_id="SO:0000110", name="sequence_feature")
_GENBANK_TYPE_TO_SO: dict[str, SORole] = {
    "gene": SORole(so_id="SO:0000704", name="gene"),
    "CDS": SORole(so_id="SO:0000316", name="CDS"),
    "promoter": SORole(so_id="SO:0000167", name="promoter"),
    "terminator": SORole(so_id="SO:0000141", name="terminator"),
    "rep_origin": SORole(so_id="SO:0000296", name="origin_of_replication"),
    "primer": SORole(so_id="SO:0005850", name="primer_binding_site"),
    "primer_bind": SORole(so_id="SO:0005850", name="primer_binding_site"),
    "RBS": SORole(so_id="SO:0000139", name="ribosome_entry_site"),
    "misc_feature": SORole(so_id="SO:0000804", name="engineered_region"),
    "protein_bind": SORole(so_id="SO:0000410", name="protein_binding_site"),
}

# SO role for a whole plasmid replicon and for a carved-out engineered construct.
_SO_PLASMID = SORole(so_id="SO:0000155", name="plasmid_vector")
_SO_ENGINEERED_REGION = SORole(so_id="SO:0000804", name="engineered_region")


class Location(ModelStrict):
    """An SBOL Range on the parent Component's sequence (0-based, half-open)."""

    start: int
    end: int
    orientation: Literal["inline", "reverse_complement"]


class Feature(ModelStrict):
    """An SBOL SequenceFeature / SubComponent: a located, SO-role-annotated region.

    Hierarchically composable: a transcription unit / cassette Feature may hold the
    promoter, CDS and terminator part Features as ``sub_features``.
    """

    name: str
    roles: list[SORole]
    location: Location
    sub_features: list[Feature] = []


class SequenceProvenance(ModelStrict):
    """Where a parsed sequence came from (the hash-pinned mirrored artifact)."""

    source_file: str
    sha256: str
    citation_key: str


class Component(ModelStrict):
    """An SBOL DNA Component: a design (plasmid, cassette or part) with a sequence.

    A plasmid, a cassette and a single part are all Components -- differing only by their
    SO ``roles`` and their place in a composition hierarchy.
    """

    identity: str
    roles: list[SORole]
    topology: str
    length: int
    sequence: str
    features: list[Feature]
    provenance: SequenceProvenance

    def get_feature(self, name: str) -> Feature:
        """Return the single top-level feature named ``name`` (raise on 0 or >1)."""
        hits = [f for f in self.features if f.name == name]
        if len(hits) != 1:
            raise KeyError(f"expected exactly one feature '{name}', found {len(hits)}")
        return hits[0]

    def features_by_role(self, so_id: str) -> list[Feature]:
        """Top-level features carrying the Sequence Ontology role ``so_id``."""
        return [f for f in self.features if any(r.so_id == so_id for r in f.roles)]

    def _slice(self, start: int, end: int) -> str:
        """Slice ``[start, end)`` honouring circular wrap for a circular replicon."""
        if self.topology == "circular":
            return "".join(self.sequence[i % self.length] for i in range(start, end))
        s, e = max(0, start), min(self.length, end)
        return self.sequence[s:e]

    def feature_sequence(self, name: str, flank: int = 0) -> str:
        """Coding-strand sequence of feature ``name`` extended by ``flank`` bp each side.

        Reverse-complemented when the feature is on the reverse strand, so the returned
        string reads 5'->3' for the feature; ``flank`` adds upstream/downstream context
        (e.g. promoter/terminator), wrapping a circular plasmid as needed.
        """
        feat = self.get_feature(name)
        window = self._slice(feat.location.start - flank, feat.location.end + flank)
        if feat.location.orientation == "reverse_complement":
            return str(Seq(window).reverse_complement())  # type: ignore[no-untyped-call]  # Bio.Seq is untyped
        return window

    def subcomponent(
        self, start: int, end: int, identity: str, roles: list[SORole] | None = None
    ) -> Component:
        """Carve ``[start, end)`` out as its own Component (SBOL design composition).

        Used to represent an engineered construct as a reusable design -- e.g. the
        chromosomal-integration insert between a plasmid's integration homology arms.
        Features fully inside the window are retained with coordinates rebased to the new
        Component; the new Component is linear (a construct, not a replicon).
        """
        seq = self._slice(start, end)
        kept: list[Feature] = []
        for f in self.features:
            if start <= f.location.start and f.location.end <= end:
                kept.append(
                    f.model_copy(
                        update={
                            "location": f.location.model_copy(
                                update={
                                    "start": f.location.start - start,
                                    "end": f.location.end - start,
                                }
                            )
                        }
                    )
                )
        return Component(
            identity=identity,
            roles=roles or [_SO_ENGINEERED_REGION],
            topology="linear",
            length=len(seq),
            sequence=seq,
            features=kept,
            provenance=self.provenance,
        )


def _sha256(path: str) -> str:
    """Streamed sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_genbank_component(path: str, citation_key: str) -> Component:
    """Parse one GenBank plasmid map into an SBOL-aligned ``Component`` with provenance."""
    record = SeqIO.read(path, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
    features: list[Feature] = []
    for f in record.features:
        if f.type == "source":
            continue
        label = f.qualifiers.get("label", f.qualifiers.get("gene", [""]))[0]
        strand = 1 if f.location.strand is None else int(f.location.strand)
        features.append(
            Feature(
                name=str(label),
                roles=[_GENBANK_TYPE_TO_SO.get(f.type, _SO_UNKNOWN)],
                location=Location(
                    start=int(f.location.start),
                    end=int(f.location.end),
                    orientation="reverse_complement" if strand == -1 else "inline",
                ),
            )
        )
    return Component(
        identity=str(record.name),
        roles=[_SO_PLASMID],
        topology=str(record.annotations.get("topology", "linear")),
        length=len(record.seq),
        sequence=str(record.seq),
        features=features,
        provenance=SequenceProvenance(
            source_file=osp.basename(path),
            sha256=_sha256(path),
            citation_key=citation_key,
        ),
    )

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
  It holds ``locations``, SBOL's own 0..* Locations, plus the ``SpanComposition`` saying
  what claim they make, so a feature across the origin of a circular replicon and an
  INSDC ``join`` are held as the ranges they are and never as their envelope.
- ``Location`` == an SBOL Range: half-open ``[start, end)`` coordinates + orientation.
  One range, as an SBOL Range is.
- ``SORole`` is ``torchcell.datamodels.schema.SOTerm``, re-exported under the name this
  module has always used. ONE Sequence Ontology map for the repo (principle 6): the
  former local duplicate asserted the same fact as a second class, which is a defect,
  and ``SOTerm`` additionally validates the ``SO:NNNNNNN`` id shape.

A ``Component`` IS A PROJECTION OF THE CANONICAL BYTES, never the canonical thing
(principle 11). The sha256-pinned GenBank file is canonical; this model keeps what the
projection needs and drops the rest, measured on the 17 mirrored Cachera maps: every
qualifier but ``/label`` and ``/gene`` is dropped (``/translation``, ``/note``,
``/codon_start``), the ``source`` feature is dropped, and a GenBank type with no entry in
``_GENBANK_TYPE_TO_SO`` falls to ``sequence_feature`` (``mRNA`` 11, ``misc_RNA`` 3,
``oriT`` 1 of 570 features). GenBank -> Component -> GenBank is therefore NOT identity,
and any GenBank/GFF3/SBOL emission is regenerated from the bytes rather than written out
of a Component. What the projection does carry is exact: a coordinate, an orientation and
an SO role, which is what an extraction window and a placement pointer need.

PRINCIPLE (total genomic content in the cell during phenotype collection): a plasmid is
captured as PRESENT CONTENT only if it is physically in the cell while the phenotype is
measured; a plasmid used only to CONSTRUCT a strain is a SOURCE whose integrated segment
becomes a chromosomal edit (a ``GeneAddition`` perturbation), not stored cell content.
Whether a Component is present-content vs construction-source is a property of how a
perturbation references it, not of the sequence itself.

We INGEST GenBank first (BioPython), and the owned model can EMIT GenBank/GFF3/SBOL for
interoperability (writers are future). ``feature_sequence`` + ``subcomponent`` provide the
extraction and design-composition operations the store is built for.

The INSDC location grammar is the vocabulary for what a feature's coordinates can say,
and this module admits exactly the three forms deposited DNA puts in front of it: one
interval, one interval across the origin of a circular replicon, and a ``join``. Every
other form is refused with its own typed reason (:class:`LocationRefusal`), named file
and feature, because a refusal is honest and a flattened envelope is not: of the 17
mirrored Cachera maps, 11 carry one origin-crossing feature each, and reading it as the
envelope made ``pBTX002.gb``'s 666 bp ``pUC`` origin the whole 12,096 bp plasmid.
"""

from __future__ import annotations

import hashlib
import os.path as osp
import warnings
from enum import StrEnum
from typing import Literal

from Bio import SeqIO
from Bio.Seq import Seq
from pydantic import model_validator

from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import SOTerm, SpanComposition

#: A Sequence Ontology term identifying a part/feature role. The repo has ONE SO map,
#: ``schema.SOTerm``; this name is kept because the module's readers and tests use it.
SORole = SOTerm


class LocationRefusal(StrEnum):
    r"""Why one GenBank location is or is not representable, as a closed vocabulary.

    The construct index names a map it could not read by one of these, so the set's
    coverage is countable and the work to unblock it is named. The members are the cases
    of the INSDC feature-table location grammar, each measured against the 17 mirrored
    Cachera maps before being admitted or refused.

    Representable since 2026-10-15 (``ConstructSpan`` holds parts): a simple interval,
    an interval across the origin of a circular replicon, and an INSDC ``join``. The
    first two members below record what was refused BEFORE that and are retained only so
    an index deposited earlier still loads; nothing produces them now.

    Still refused, and each for a reason, not for want of a field:

    - ``ordered_location`` -- INSDC ``order`` says the elements "can be found in the
      specified order" while "nothing is implied about the reasonableness about joining
      them", which is a different claim from ``join``'s "joined (placed end-to-end) to
      form one contiguous sequence". 0 of 570 features use it, so there is nothing to
      measure a representation against.
    - ``fuzzy_position`` -- ``<345..500`` says the feature extends past the recorded
      bound. A ``SpanPart`` is two exact coordinates, and storing ``345`` for ``<345``
      would assert a boundary the source declined to give.
    - ``between_bases_site`` -- ``123^124`` is a site between two bases, which Biopython
      hands back as a zero-length interval. A zero-length interval and a site are
      different claims.
    - ``mixed_strand_location`` -- the location's ranges are not all on one of the two
      strands: ``join(complement(1..10),20..30)`` is a trans-spliced claim that one
      orientation per feature cannot hold, and a location that states no strand at all
      is not a located feature either. The GenBank parser always assigns ``+1`` or
      ``-1`` to a single range, so in practice this is the mixed-strand join; the former
      code defaulted a missing strand to ``+``, which is the kind of fallback this
      module exists to refuse.
    - ``remote_reference_location`` -- ``J00194.1:100..202`` points at another
      accession's bases, which are not this contig's bytes.
    - ``unordered_join`` -- a ``join`` whose parts are not ascending and is not the
      two-part origin crossing. Reordering it would be a warn-and-fix of exactly the
      kind this module exists to refuse.
    - ``origin_wrap_on_linear_topology`` -- a location that crosses the origin of a
      replicon the file's own LOCUS line calls linear. The two statements of the file
      disagree and neither is privileged, so the file is reported rather than read. (A
      file writing the COMPACT form on a linear record lands in
      ``unparsable_location`` instead, because Biopython applies its wrap rewrite only
      to circular records and sets the location to ``None``.)
    - ``unparsable_location`` -- Biopython could not parse it and set the location to
      ``None`` (``3.6``, ``?``). There is nothing to store.
    """

    #: RETIRED 2026-10-15: an origin-wrapping location, now held as ``origin_spanning``.
    origin_wrapping_location = "origin_wrapping_location"
    #: RETIRED 2026-10-15: a compound location, now held as parts.
    compound_location = "compound_location"
    ordered_location = "ordered_location"
    fuzzy_position = "fuzzy_position"
    between_bases_site = "between_bases_site"
    mixed_strand_location = "mixed_strand_location"
    remote_reference_location = "remote_reference_location"
    unordered_join = "unordered_join"
    origin_wrap_on_linear_topology = "origin_wrap_on_linear_topology"
    unparsable_location = "unparsable_location"


#: The members no parse can produce any more; kept so an index deposited before
#: 2026-10-15 still loads, and asserted unreachable by the parser's own tests.
RETIRED_LOCATION_REFUSALS: frozenset[LocationRefusal] = frozenset(
    {LocationRefusal.origin_wrapping_location, LocationRefusal.compound_location}
)


class GenBankLocationError(ValueError):
    r"""A GenBank feature's location is one this model cannot hold without lying.

    Since 2026-10-15 the two cases that blocked 11 of the 17 mirrored Cachera maps are
    HELD rather than refused: an interval across the origin of a circular replicon (the
    compact ``11559..128`` form Biopython rewrites) and an INSDC ``join``. What remains
    refused is listed in :class:`LocationRefusal`, and every refusal is loud: the message
    names the file, the feature and the location verbatim, and ``refusal`` carries the
    typed reason so a blocked map is recorded by name with the right one rather than one
    hardcoded at the call site.
    """

    def __init__(self, message: str, refusal: LocationRefusal) -> None:
        """Carry the typed reason beside the message a reader sees."""
        super().__init__(message)
        self.refusal = refusal


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
    """An SBOL Range on the parent Component's sequence (0-based, half-open).

    One range, as SBOL's Range is. A feature whose location is several ranges holds
    several of these, because SBOL gives a SequenceFeature 0..* Locations; the envelope
    of those ranges is never a Location of its own.
    """

    start: int
    end: int
    orientation: Literal["inline", "reverse_complement"]


class Feature(ModelStrict):
    r"""An SBOL SequenceFeature / SubComponent: a located, SO-role-annotated region.

    ``locations`` is the feature's ordered ranges, in the order the source's location
    string writes them, and ``composition`` says what claim they make
    (``schema.SpanComposition``, grounded in the INSDC ``join`` / ``order`` grammar). An
    interval across the origin of a circular Component is ONE interval whose coordinates
    wrapped and is held as ``origin_spanning`` with the two ranges it wrapped into; a
    ``join`` is several ranges with sequence between them the feature omits. They are
    never the same shape: ``pBTX002.gb``'s ``pUC\ori`` is a 666 bp wrap, and reading it
    as the envelope made it the whole 12,096 bp plasmid.

    Every range of one feature carries the same orientation, which IS the refusal of a
    mixed-strand ``join``: one feature reads in one direction, so a location whose parts
    disagree is not one feature in this model and the parser refuses it.

    Hierarchically composable: a transcription unit / cassette Feature may hold the
    promoter, CDS and terminator part Features as ``sub_features``.
    """

    name: str
    roles: list[SORole]
    locations: list[Location]
    composition: SpanComposition = SpanComposition.simple
    sub_features: list[Feature] = []

    @model_validator(mode="after")
    def _check_locations(self) -> Feature:
        """One orientation throughout, and ranges that make the claim ``composition`` names."""
        if not self.locations:
            raise ValueError(f"feature {self.name!r} needs at least one location")
        for location in self.locations:
            if location.end < location.start:
                raise ValueError(
                    f"feature {self.name!r} has the inverted range "
                    f"[{location.start}, {location.end})"
                )
        orientations = {location.orientation for location in self.locations}
        if len(orientations) != 1:
            raise ValueError(
                f"feature {self.name!r} has ranges on both strands {sorted(orientations)}; "
                "one feature reads in one direction"
            )
        if self.composition is SpanComposition.simple:
            if len(self.locations) != 1:
                raise ValueError(
                    f"feature {self.name!r} is simple but has {len(self.locations)} "
                    "ranges; a multi-range feature is joined or origin_spanning"
                )
            return self
        if self.composition is SpanComposition.origin_spanning:
            if len(self.locations) != 2:
                raise ValueError(
                    f"feature {self.name!r} crosses the origin, which is one interval "
                    f"in two ranges, but has {len(self.locations)}"
                )
            if self.locations[1].start != 0:
                raise ValueError(
                    f"feature {self.name!r} crosses the origin, so its second range "
                    f"begins at 0, not {self.locations[1].start}"
                )
            if self.locations[0].start < self.locations[1].end:
                raise ValueError(
                    f"feature {self.name!r} has overlapping wrap ranges "
                    f"[{self.locations[0].start}, {self.locations[0].end}) and "
                    f"[{self.locations[1].start}, {self.locations[1].end})"
                )
            return self
        if len(self.locations) < 2:
            raise ValueError(
                f"feature {self.name!r} is a join with {len(self.locations)} range(s)"
            )
        for earlier, later in zip(self.locations, self.locations[1:], strict=False):
            if later.start < earlier.end:
                raise ValueError(
                    f"feature {self.name!r} has a join whose ranges are not ascending: "
                    f"[{earlier.start}, {earlier.end}) then "
                    f"[{later.start}, {later.end})"
                )
        return self

    @property
    def orientation(self) -> Literal["inline", "reverse_complement"]:
        """The strand the whole feature reads on (every range shares it)."""
        return self.locations[0].orientation

    @property
    def location(self) -> Location:
        """The one range of a single-range feature; RAISES for a multi-range one.

        A caller written for one range must fail loudly rather than silently receive the
        envelope, which for an origin-crossing feature is the whole replicon. The
        multi-range cases read ``locations``.
        """
        if self.composition is not SpanComposition.simple:
            raise ValueError(
                f"feature {self.name!r} is {self.composition} with "
                f"{len(self.locations)} ranges and has no single location; read "
                "`locations` (the envelope is what this model refuses to store)"
            )
        return self.locations[0]

    @property
    def length(self) -> int:
        """Bases the feature covers: the sum of its ranges, never the envelope."""
        return sum(location.end - location.start for location in self.locations)


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
        r"""Coding-strand sequence of feature ``name`` extended by ``flank`` bp each side.

        Every range of the feature is sliced and the pieces are concatenated in the
        order the source wrote them, which is what INSDC's ``join`` means: "joined
        (placed end-to-end) to form one contiguous sequence". For an origin-crossing
        feature the two ranges are adjacent in the molecule, so the concatenation is the
        666 bp ``pUC\ori`` rather than the 12,096 bp plasmid the envelope would give.
        Reverse-complemented when the feature is on the reverse strand, so the returned
        string reads 5'->3' for the feature; ``flank`` adds context at the two OUTER
        ends only, wrapping a circular plasmid as needed.
        """
        feat = self.get_feature(name)
        last = len(feat.locations) - 1
        pieces = [
            self._slice(
                location.start - (flank if index == 0 else 0),
                location.end + (flank if index == last else 0),
            )
            for index, location in enumerate(feat.locations)
        ]
        window = "".join(pieces)
        if feat.orientation == "reverse_complement":
            return str(Seq(window).reverse_complement())  # type: ignore[no-untyped-call]  # Bio.Seq is untyped
        return window

    def subcomponent(
        self, start: int, end: int, identity: str, roles: list[SORole] | None = None
    ) -> Component:
        """Carve ``[start, end)`` out as its own Component (SBOL design composition).

        Used to represent an engineered construct as a reusable design -- e.g. the
        chromosomal-integration insert between a plasmid's integration homology arms.
        A feature is retained only when EVERY one of its ranges is inside the window, so
        a feature that crosses the origin is never half-carved, and the retained ranges
        are rebased to the new Component; the new Component is linear (a construct, not a
        replicon).
        """
        seq = self._slice(start, end)
        kept: list[Feature] = []
        for f in self.features:
            if all(
                start <= location.start and location.end <= end
                for location in f.locations
            ):
                kept.append(
                    f.model_copy(
                        update={
                            "locations": [
                                location.model_copy(
                                    update={
                                        "start": location.start - start,
                                        "end": location.end - start,
                                    }
                                )
                                for location in f.locations
                            ]
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


#: The substring of Biopython's location warning that means it rewrote a compact
#: origin-wrapping location into the two parts it wraps into (``11559..128`` ->
#: ``join(11559..12096,1..128)``). Matched rather than parsed: the message carries the
#: offending location verbatim, which is how the compact form the source wrote is
#: recovered for the round trip. Biopython's own wording calls the compact form invalid
#: ("Attempting to fix invalid location '11559..128' as it looks like incorrect origin
#: wrapping. Please fix input file, this could have unintended behavior."), which is the
#: evidence that the canonical INSDC spelling of the same claim is the ``join``.
_ORIGIN_WRAP_WARNING = "incorrect origin wrapping"

#: The substring of the warning Biopython emits when it could not parse a location at
#: all and set the feature's location to ``None`` (``3.6``, ``?``).
_UNPARSED_LOCATION_WARNING = "Could not parse feature location"


def _refuse(
    path: str, label: object, location: object, why: str, refusal: LocationRefusal
) -> GenBankLocationError:
    """One refusal, naming the file, the feature and the location verbatim."""
    return GenBankLocationError(
        f"{osp.basename(path)}: feature {str(label)!r} has the location {location}, "
        f"which is refused: {why}",
        refusal,
    )


def _exact(position: object) -> bool:
    """Whether a Biopython position is exact (not ``<``, ``>``, within or unknown)."""
    return type(position).__name__ == "ExactPosition"


def _feature_locations(
    path: str, feature: object, label: object, replicon_length: int, topology: str
) -> tuple[list[Location], SpanComposition]:
    """One GenBank feature's ranges and the claim they make, or a typed refusal.

    Biopython hands a compound location's parts back in FEATURE order (5'->3'), so for a
    reverse-strand feature it reverses what the location string wrote; the parts are
    re-reversed here so the stored order is the source's written order and the location
    string regenerates exactly. Measured: ``complement(18..2)`` on a 20 bp circular map
    parses to parts ``[(0, 2), (17, 20)]``, and the written order is ``18..20`` then
    ``1..2``.
    """
    location = getattr(feature, "location", None)
    if location is None:
        raise _refuse(
            path,
            label,
            None,
            "Biopython could not parse it",
            LocationRefusal.unparsable_location,
        )
    operator = getattr(location, "operator", None)
    if operator == "order":
        raise _refuse(
            path,
            label,
            location,
            "INSDC `order` says nothing is implied about joining the elements, which is "
            "not the `join` claim a span of parts makes",
            LocationRefusal.ordered_location,
        )
    parts = list(location.parts)
    strands = {part.strand for part in parts}
    if strands - {1, -1} or len(strands) != 1:
        raise _refuse(
            path,
            label,
            location,
            f"its ranges are on strands {sorted(str(s) for s in strands)}, which is "
            "not one of + or -; one feature reads in one direction and a location that "
            "states no strand is not a located feature",
            LocationRefusal.mixed_strand_location,
        )
    reverse = strands == {-1}
    if reverse:
        parts = list(reversed(parts))
    for part in parts:
        if getattr(part, "ref", None) is not None:
            raise _refuse(
                path,
                label,
                location,
                f"it points at another accession ({part.ref}), whose bases are not this "
                "contig's bytes",
                LocationRefusal.remote_reference_location,
            )
        if not (_exact(part.start) and _exact(part.end)):
            raise _refuse(
                path,
                label,
                location,
                "it carries a fuzzy boundary, and a SpanPart is two exact coordinates",
                LocationRefusal.fuzzy_position,
            )
        if int(part.end) <= int(part.start):
            raise _refuse(
                path,
                label,
                location,
                "it is a zero-length interval, which is how Biopython returns a "
                "between-bases site (`123^124`); a site is not an interval",
                LocationRefusal.between_bases_site,
            )
    orientation: Literal["inline", "reverse_complement"] = (
        "reverse_complement" if reverse else "inline"
    )
    ranges = [
        Location(start=int(part.start), end=int(part.end), orientation=orientation)
        for part in parts
    ]
    if len(ranges) == 1:
        return ranges, SpanComposition.simple
    crosses_origin = (
        len(ranges) == 2
        and ranges[0].end == replicon_length
        and ranges[1].start == 0
        and ranges[0].start >= ranges[1].end
    )
    if crosses_origin:
        if topology != "circular":
            raise _refuse(
                path,
                label,
                location,
                f"it crosses the origin of a {topology!r} replicon, which has no origin "
                "to cross; the file's topology and its location disagree",
                LocationRefusal.origin_wrap_on_linear_topology,
            )
        return ranges, SpanComposition.origin_spanning
    for earlier, later in zip(ranges, ranges[1:], strict=False):
        if later.start < earlier.end:
            raise _refuse(
                path,
                label,
                location,
                "its join runs backwards or overlaps and is not the two-range origin "
                "crossing; reordering it would be a warn-and-fix",
                LocationRefusal.unordered_join,
            )
    return ranges, SpanComposition.joined


def parse_genbank_component(path: str, citation_key: str) -> Component:
    r"""Parse one GenBank plasmid map into an SBOL-aligned ``Component`` with provenance.

    Since 2026-10-15 a feature's location may be a simple interval, an interval across
    the origin of a CIRCULAR replicon, or an INSDC ``join``; each is held as the ranges
    it is made of, with the claim named in ``Feature.composition``. An envelope is never
    stored. What is still refused is loud and typed (:class:`LocationRefusal`), and the
    message names the file, the feature and the location verbatim.

    Biopython rewrites the compact ``11559..128`` form into the two parts it wraps into
    and says so in a warning, so the parse runs under ``catch_warnings(record=True)``
    with ``simplefilter("always")``: the default ``once`` registry suppresses a repeat of
    the same message from the same line, which is why re-parsing the mirrored maps in one
    process used to look clean (measured: 7 of 17 maps warn under the default registry,
    11 of 17 under ``always``; the 11 are the truth). The warnings are kept rather than
    raised on, because an unparsable location arrives the same way and the two need
    different refusals.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = SeqIO.read(path, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
        messages = [str(entry.message) for entry in caught]
    unparsed = [
        message for message in messages if _UNPARSED_LOCATION_WARNING in message
    ]
    if unparsed:
        raise GenBankLocationError(
            f"{osp.basename(path)}: Biopython could not parse a feature location and "
            f"set it to None, so there is nothing to store ({'; '.join(unparsed)})",
            LocationRefusal.unparsable_location,
        )
    topology = str(record.annotations.get("topology", "linear"))
    replicon_length = len(record.seq)
    features: list[Feature] = []
    for f in record.features:
        if f.type == "source":
            continue
        label = f.qualifiers.get("label", f.qualifiers.get("gene", [""]))[0]
        ranges, composition = _feature_locations(
            path, f, label, replicon_length, topology
        )
        features.append(
            Feature(
                name=str(label),
                roles=[_GENBANK_TYPE_TO_SO.get(f.type, _SO_UNKNOWN)],
                locations=ranges,
                composition=composition,
            )
        )
    return Component(
        identity=str(record.name),
        roles=[_SO_PLASMID],
        topology=topology,
        length=replicon_length,
        sequence=str(record.seq),
        features=features,
        provenance=SequenceProvenance(
            source_file=osp.basename(path),
            sha256=_sha256(path),
            citation_key=citation_key,
        ),
    )

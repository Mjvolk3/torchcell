#!/usr/bin/env python
# experiments/database/scripts/teteneva2024_symbol_annotation.py
# [[experiments.database.scripts.teteneva2024_symbol_annotation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/teteneva2024_symbol_annotation.py
# Test file: tests/torchcell/datasets/ecoli/test_teteneva2024_symbol_annotation.py
"""How Teteneva 2024's Table S4 gene symbols land on the W3110 annotations: a DOWNSTREAM
measurement, not loader logic (#910).

The loader (``torchcell/datasets/ecoli/teteneva2024.py``) stores every Table S4 value
under the gene symbol exactly as the release gives it (owner, 2026-10-12: "the data is
the data!"). Mapping those symbols onto an annotation of the W3110 assembly set is a
separate artifact, measured here and written to
``experiments/database/results/teteneva2024_symbol_annotation.json``:

* :func:`annotation_routes`: the GenBank member (GCA_000010245.1) carries 4,444 gene
  features and no ``locus_tag``, so ``read_genbank`` refuses it; the RefSeq member
  (GCF_000010245.2) parses, 4,531 ``Y75_RS`` loci;
* :func:`symbol_routes`: by ``/gene`` alone, 3,036 of the 3,691 symbols land on exactly
  one GenBank gene feature and 3,607 on exactly one RefSeq gene feature;
* :func:`resolve_symbols`: three ordered steps onto one RefSeq locus (the RefSeq
  ``/gene``; the GenBank ``/gene`` then its stop codon on RefSeq; an MG1655 name, its
  b-number on a GenBank ``ECK:JW:b`` note, then that CDS's stop codon on RefSeq), two
  symbols on one locus settled by step precedence, with two cross-checks
  (:func:`genbank_route_conflicts`, :func:`step_agreement`).

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``) with ``DATA_ROOT`` set::

    python experiments/database/scripts/teteneva2024_symbol_annotation.py
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Final, Literal

from Bio import SeqIO
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datasets.bacteria_common import bacterial_genome
from torchcell.datasets.ecoli import teteneva2024 as te
from torchcell.sequence.genome.bacterial import GenBankLocus, read_genbank
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12MG1655Genome
from torchcell.sequence.genome.ecoli.w3110 import EcoliK12W3110Genome
from torchcell.sequence.genome.registry import ECOLI_K12_W3110, resolve

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
RESULT: Final = (
    REPO / "experiments/database/results/teteneva2024_symbol_annotation.json"
)


# --------------------------------------------------------------------------- #
# The annotation route: which member can be read, measured (#910)
# --------------------------------------------------------------------------- #
class AnnotationRoute(BaseModel):
    """What one annotation member of the set offers a GenBank-first ingest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member: str
    gene_features: int = Field(description="'gene' features in the flat file")
    gene_features_with_locus_tag: int
    read_genbank_error: str | None = Field(
        description="read_genbank's message, or None when the member parses"
    )
    loci: int | None = Field(
        description="loci read, or None when the member is refused"
    )
    replicon: str | None


#: The ``ECK:JW:b`` crosswalk the GenBank CDS features carry instead of a locus tag.
ECK_JW_B_NOTE = re.compile(r'/note="(ECK\d+):(JW\d+):(b\d+)"')

#: Measured on the deposited bytes on 2026-10-09 by :func:`annotation_routes`. The
#: GenBank member is refused by name; the RefSeq member parses. The GenBank CDS notes
#: carry 3,730 ``ECK:JW:b`` triples, 3,730 distinct JW numbers and 3,726 distinct
#: b-numbers, so the file crosswalks W3110 to MG1655 without naming a W3110 locus tag.
GENBANK_GENE_FEATURES: Final = 4_444
GENBANK_GENE_FEATURES_WITH_LOCUS_TAG: Final = 0
GENBANK_READ_ERROR: Final = (
    "GCA_000010245.1_ASM1024v1_genomic.gbff.gz: a gene feature at [189:255](+) has no "
    "locus_tag"
)
ECK_JW_B_TRIPLES: Final = 3_730
ECK_JW_B_DISTINCT_JW: Final = 3_730
ECK_JW_B_DISTINCT_BNUMBER: Final = 3_726
REFSEQ_GENE_FEATURES: Final = 4_531
REFSEQ_LOCI: Final = 4_531
REFSEQ_LOCUS_TAG_PATTERN: Final = r"Y75_RS\d{5}"
REFSEQ_OLD_LOCUS_TAG_PATTERN: Final = r"Y75_p\d{4}"
REFSEQ_OLD_LOCUS_TAGS: Final = 4_254
REFSEQ_ONTOLOGY_TERM_ROWS: Final = 2_298

GENBANK_MEMBER: Final = "GCA_000010245.1_ASM1024v1_genomic.gbff.gz"
REFSEQ_MEMBER: Final = "GCF_000010245.2_ASM1024v1_genomic.gbff.gz"


def _gene_feature_counts(path: str) -> tuple[int, int]:
    """``(gene features, gene features carrying a locus_tag)`` of a flat file."""
    genes = tagged = 0
    in_features = in_gene = False
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN"):
                in_features = False
            if not in_features:
                continue
            if re.match(r"^     gene +", line):
                genes += 1
                in_gene = True
                continue
            if re.match(r"^     \S", line):
                in_gene = False
            if in_gene and "/locus_tag=" in line:
                tagged += 1
    return genes, tagged


def annotation_routes(data_root: str | None = None) -> tuple[AnnotationRoute, ...]:
    """Measure both annotation members of the deposited set, GenBank first.

    Returns one :class:`AnnotationRoute` per member in deposit order. The GenBank
    member's ``read_genbank_error`` is the parser's own message, so the finding that
    the tier's GenBank-first ingest does not reach W3110 is re-derived rather than
    asserted.
    """
    routes = []
    for member in (GENBANK_MEMBER, REFSEQ_MEMBER):
        path = resolve(ECOLI_K12_W3110, member, data_root=data_root)
        genes, tagged = _gene_feature_counts(path)
        try:
            annotation, _ = read_genbank(path, member)
        except ValueError as error:
            routes.append(
                AnnotationRoute(
                    member=member,
                    gene_features=genes,
                    gene_features_with_locus_tag=tagged,
                    read_genbank_error=str(error),
                    loci=None,
                    replicon=None,
                )
            )
            continue
        routes.append(
            AnnotationRoute(
                member=member,
                gene_features=genes,
                gene_features_with_locus_tag=tagged,
                read_genbank_error=None,
                loci=len(annotation.loci),
                replicon=annotation.replicons[0].accession,
            )
        )
    return tuple(routes)


def eck_jw_b_notes(data_root: str | None = None) -> tuple[tuple[str, str, str], ...]:
    """Every ``(ECK, JW, b-number)`` triple the GenBank member's CDS notes carry.

    This is the only W3110-to-MG1655 crosswalk the GenBank deposit publishes, and it is
    a ``/note``, not an identifier: no feature of the file carries a W3110 locus tag.
    """
    path = resolve(ECOLI_K12_W3110, GENBANK_MEMBER, data_root=data_root)
    with gzip.open(path, "rt") as handle:
        return tuple(ECK_JW_B_NOTE.findall(handle.read()))


def symbol_index(member: str, data_root: str | None = None) -> dict[str, set[str]]:
    """``/gene`` symbol -> the gene features carrying it, for one annotation member.

    A feature is keyed by its ``locus_tag`` where it has one (RefSeq) and by its
    location otherwise (the GenBank member has none), so two features sharing a symbol
    count as two.
    """
    path = resolve(ECOLI_K12_W3110, member, data_root=data_root)
    index: dict[str, set[str]] = {}
    with gzip.open(path, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features:
                if feature.type != "gene":
                    continue
                key = feature.qualifiers.get("locus_tag", [str(feature.location)])[0]
                for symbol in feature.qualifiers.get("gene", []):
                    index.setdefault(symbol, set()).add(key)
    return index


def symbol_routes(
    symbols: tuple[str, ...], data_root: str | None = None
) -> dict[str, dict[str, int]]:
    """How Table S4's symbols land on each annotation member, by ``/gene`` only.

    ``unique``: one gene feature carries the symbol; ``ambiguous``: two or more do;
    ``none``: no gene feature's ``/gene`` is the symbol (a synonym route is not
    counted, because which synonyms a route may follow is part of the decision).
    """
    out: dict[str, dict[str, int]] = {}
    for member in (GENBANK_MEMBER, REFSEQ_MEMBER):
        index = symbol_index(member, data_root)
        counts = Counter(
            "none"
            if symbol not in index
            else "unique"
            if len(index[symbol]) == 1
            else "ambiguous"
            for symbol in symbols
        )
        out[member] = {k: counts.get(k, 0) for k in ("unique", "ambiguous", "none")}
    return out


#: Measured on 2026-10-12 by :func:`symbol_routes` over the 3,691 Table S4 symbols.
SYMBOL_ROUTES: Final = {
    GENBANK_MEMBER: {"unique": 3_036, "ambiguous": 0, "none": 655},
    REFSEQ_MEMBER: {"unique": 3_607, "ambiguous": 44, "none": 40},
}


# --------------------------------------------------------------------------- #
# Gene resolution: Table S4's symbols onto W3110's RefSeq loci (#910)
# --------------------------------------------------------------------------- #
ResolutionStep = Literal[
    "refseq_symbol", "genbank_symbol_eck_jw_b", "mg1655_symbol_eck_jw_b"
]
STEPS: Final[tuple[ResolutionStep, ...]] = (
    "refseq_symbol",
    "genbank_symbol_eck_jw_b",
    "mg1655_symbol_eck_jw_b",
)
StepResult = Literal[
    "resolved",
    "no_feature",
    "several_features",
    "no_refseq_stop_match",
    "several_refseq_stop_matches",
    "split_pseudogene",
    "mg1655_not_one_gene",
    "no_genbank_cds",
    "several_genbank_cds",
    "locus_held_by_another_symbol",
]
Strand = Literal["+", "-"]


class StepOutcome(BaseModel):
    """What one resolution step did with one symbol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: ResolutionStep
    result: StepResult
    candidates: tuple[str, ...] = Field(
        default=(), description="the loci or features the step found when not one"
    )


class GeneResolution(BaseModel):
    """A Table S4 symbol, resolved to one W3110 RefSeq locus or kept unresolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(description="the released 'Gene name', verbatim")
    status: Literal["resolved", "unresolved"]
    step: ResolutionStep | None = Field(description="the step that resolved it")
    locus_tag: str | None
    refseq_symbol: str | None = Field(description="the RefSeq locus's own /gene")
    refseq_pseudo: bool | None = Field(description="the RefSeq locus is a pseudogene")
    eck_jw_b: tuple[str, str, str] | None = Field(
        description="the GenBank CDS crosswalk a step 2 or 3 resolution went through"
    )
    attempts: tuple[StepOutcome, ...]


class GenbankFeature(BaseModel):
    """A gene or CDS feature of the W3110 GenBank member (it has no locus tags)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["gene", "CDS"]
    symbol: str | None
    start: int = Field(description="1-based first base")
    end: int
    strand: Strand
    eck_jw_b: tuple[str, str, str] | None = Field(
        description="the CDS /note crosswalk, None on a gene feature or a CDS without one"
    )

    @property
    def stop(self) -> int:
        """The 3' end: ``end`` on the plus strand, ``start`` on the minus strand."""
        return self.end if self.strand == "+" else self.start


class GenbankMember(BaseModel):
    """The W3110 GenBank member's gene and CDS features, the ECK:JW:b route's input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    genes: tuple[GenbankFeature, ...]
    cds: tuple[GenbankFeature, ...]


_ECK_JW_B: Final = re.compile(r"^(ECK\d+):(JW\d+):(b\d+)$")


def _strand(value: int | None) -> Strand:
    if value == 1:
        return "+"
    if value == -1:
        return "-"
    raise ValueError(f"a GenBank feature has strand {value!r}")


def read_genbank_member(path: str | Path) -> GenbankMember:
    """The gene and CDS features of the W3110 GenBank flat file (GCA_000010245.1).

    A CDS ``/note`` that is an ``ECK:JW:b`` triple is kept as the feature's crosswalk; a
    CDS with two such notes is refused.
    """
    genes: list[GenbankFeature] = []
    cds: list[GenbankFeature] = []
    with gzip.open(path, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features:
                if feature.type not in ("gene", "CDS"):
                    continue
                triples = [
                    (match.group(1), match.group(2), match.group(3))
                    for note in feature.qualifiers.get("note", [])
                    if (match := _ECK_JW_B.match(note)) is not None
                ]
                if len(triples) > 1:
                    raise ValueError(f"a CDS at {feature.location} has {triples}")
                symbols = feature.qualifiers.get("gene", [])
                item = GenbankFeature(
                    kind=feature.type,
                    symbol=symbols[0] if symbols else None,
                    start=int(feature.location.start) + 1,
                    end=int(feature.location.end),
                    strand=_strand(feature.location.strand),
                    eck_jw_b=triples[0] if triples else None,
                )
                (genes if feature.type == "gene" else cds).append(item)
    return GenbankMember(genes=tuple(genes), cds=tuple(cds))


class _RefseqIndex:
    """RefSeq loci by ``/gene`` symbol and by stop codon, with the pseudogene spans."""

    def __init__(self, loci: Mapping[str, GenBankLocus]) -> None:
        self.loci = loci
        self.by_symbol: dict[str, list[str]] = {}
        self.by_stop: dict[tuple[Strand, int], list[str]] = {}
        for tag, locus in loci.items():
            if locus.symbol is not None:
                self.by_symbol.setdefault(locus.symbol, []).append(tag)
            stop = locus.end if locus.strand == "+" else locus.start
            self.by_stop.setdefault((locus.strand, stop), []).append(tag)

    def pseudogenes_overlapping(self, feature: GenbankFeature) -> list[str]:
        """RefSeq pseudogene loci on the feature's strand that overlap its span."""
        return sorted(
            tag
            for tag, locus in self.loci.items()
            if locus.pseudo
            and locus.strand == feature.strand
            and max(locus.start, feature.start) <= min(locus.end, feature.end)
        )

    def by_stop_codon(
        self, step: ResolutionStep, feature: GenbankFeature
    ) -> tuple[StepOutcome, str | None]:
        """The one RefSeq locus sharing the feature's stop codon, unless the feature
        spans two or more RefSeq pseudogene fragments (a split pseudogene).
        """
        tags = sorted(self.by_stop.get((feature.strand, feature.stop), []))
        if not tags:
            return StepOutcome(step=step, result="no_refseq_stop_match"), None
        if len(tags) > 1:
            return (
                StepOutcome(
                    step=step,
                    result="several_refseq_stop_matches",
                    candidates=tuple(tags),
                ),
                None,
            )
        fragments = self.pseudogenes_overlapping(feature)
        if len(fragments) > 1:
            return (
                StepOutcome(
                    step=step, result="split_pseudogene", candidates=tuple(fragments)
                ),
                None,
            )
        return StepOutcome(step=step, result="resolved"), tags[0]


#: MG1655 resolver statuses that name exactly one current gene.
_ONE_GENE: Final = (GeneNameStatus.CURRENT, GeneNameStatus.RENAMED)


def _resolved(
    symbol: str,
    step: ResolutionStep,
    tag: str,
    index: _RefseqIndex,
    attempts: list[StepOutcome],
    eck_jw_b: tuple[str, str, str] | None,
) -> GeneResolution:
    locus = index.loci[tag]
    return GeneResolution(
        symbol=symbol,
        status="resolved",
        step=step,
        locus_tag=tag,
        refseq_symbol=locus.symbol,
        refseq_pseudo=locus.pseudo,
        eck_jw_b=eck_jw_b,
        attempts=tuple(attempts),
    )


#: One step's verdict: its outcome, the locus it reached, the crosswalk it went through.
RouteResult = tuple[StepOutcome, str | None, tuple[str, str, str] | None]


def refseq_route(symbol: str, index: _RefseqIndex) -> RouteResult:
    """Step 1: the symbol is the ``/gene`` of exactly one RefSeq gene feature."""
    tags = sorted(index.by_symbol.get(symbol, []))
    if len(tags) == 1:
        return StepOutcome(step="refseq_symbol", result="resolved"), tags[0], None
    outcome = StepOutcome(
        step="refseq_symbol",
        result="no_feature" if not tags else "several_features",
        candidates=tuple(tags),
    )
    return outcome, None, None


def genbank_route(
    symbol: str, index: _RefseqIndex, genbank: GenbankMember
) -> RouteResult:
    """Step 2: one GenBank gene feature, its stop codon on one RefSeq locus."""
    features = [g for g in genbank.genes if g.symbol == symbol]
    if len(features) != 1:
        outcome = StepOutcome(
            step="genbank_symbol_eck_jw_b",
            result="no_feature" if not features else "several_features",
            candidates=tuple(f"{g.start}..{g.end}({g.strand})" for g in features),
        )
        return outcome, None, None
    gene = features[0]
    outcome, tag = index.by_stop_codon("genbank_symbol_eck_jw_b", gene)
    triple = next(
        (
            c.eck_jw_b
            for c in genbank.cds
            if (c.start, c.end, c.strand) == (gene.start, gene.end, gene.strand)
            and c.eck_jw_b is not None
        ),
        None,
    )
    return outcome, tag, triple if tag is not None else None


def mg1655_route(
    symbol: str,
    index: _RefseqIndex,
    genbank: GenbankMember,
    mg1655_resolve: Callable[[str], GeneNameResolution],
) -> RouteResult:
    """Step 3: one MG1655 gene (exact case), its b-number on one GenBank CDS
    ``ECK:JW:b`` note, that CDS's stop codon on one RefSeq locus.
    """
    named = mg1655_resolve(symbol)
    exact = named.note is None or "case-insensitive" not in named.note
    if named.status not in _ONE_GENE or named.systematic_name is None or not exact:
        outcome = StepOutcome(
            step="mg1655_symbol_eck_jw_b",
            result="mg1655_not_one_gene",
            candidates=tuple(named.candidates)
            or (f"{named.status.value}:{named.systematic_name}",),
        )
        return outcome, None, None
    b_number = named.systematic_name
    cds = [
        c for c in genbank.cds if c.eck_jw_b is not None and c.eck_jw_b[2] == b_number
    ]
    if len(cds) != 1:
        outcome = StepOutcome(
            step="mg1655_symbol_eck_jw_b",
            result="no_genbank_cds" if not cds else "several_genbank_cds",
            candidates=(b_number,),
        )
        return outcome, None, None
    outcome, tag = index.by_stop_codon("mg1655_symbol_eck_jw_b", cds[0])
    return outcome, tag, cds[0].eck_jw_b if tag is not None else None


def resolve_symbol(
    symbol: str,
    index: _RefseqIndex,
    genbank: GenbankMember,
    mg1655_resolve: Callable[[str], GeneNameResolution],
) -> GeneResolution:
    """Resolve one symbol by the three steps in order (module docstring)."""
    routes: tuple[tuple[ResolutionStep, Callable[[], RouteResult]], ...] = (
        ("refseq_symbol", lambda: refseq_route(symbol, index)),
        ("genbank_symbol_eck_jw_b", lambda: genbank_route(symbol, index, genbank)),
        (
            "mg1655_symbol_eck_jw_b",
            lambda: mg1655_route(symbol, index, genbank, mg1655_resolve),
        ),
    )
    attempts: list[StepOutcome] = []
    for step, route in routes:
        outcome, tag, triple = route()
        attempts.append(outcome)
        if tag is not None:
            return _resolved(symbol, step, tag, index, attempts, triple)
    return GeneResolution(
        symbol=symbol,
        status="unresolved",
        step=None,
        locus_tag=None,
        refseq_symbol=None,
        refseq_pseudo=None,
        eck_jw_b=None,
        attempts=tuple(attempts),
    )


def _release_collisions(
    resolutions: dict[str, GeneResolution],
) -> dict[str, GeneResolution]:
    """Settle two symbols resolving to one locus by step precedence.

    A record keys on the locus, so two symbols cannot both hold it. The symbol resolved
    by the EARLIEST step keeps the locus; the others become unresolved with a
    ``locus_held_by_another_symbol`` outcome naming the holder. When the earliest step
    resolved two of them, none keeps it. Measured on 2026-10-12: one collision, ``pat``
    (step 3, MG1655 b3073, whose ``pat`` is the putrescine aminotransferase) against
    ``ygjG`` (step 1) on Y75_RS16000, while Table S4 describes ``pat`` as "Protein
    lysine acetyltransferase", another gene.
    """
    by_tag: dict[str, list[str]] = {}
    for resolution in resolutions.values():
        if resolution.locus_tag is not None:
            by_tag.setdefault(resolution.locus_tag, []).append(resolution.symbol)
    out = dict(resolutions)
    for names in by_tag.values():
        if len(names) < 2:
            continue
        rank = {n: STEPS.index(_step(out[n])) for n in names}
        first = min(rank.values())
        holders = [n for n in names if rank[n] == first]
        holder = holders[0] if len(holders) == 1 else None
        for name in names:
            if name == holder:
                continue
            released = out[name]
            out[name] = GeneResolution(
                symbol=name,
                status="unresolved",
                step=None,
                locus_tag=None,
                refseq_symbol=None,
                refseq_pseudo=None,
                eck_jw_b=None,
                attempts=(
                    *released.attempts,
                    StepOutcome(
                        step=_step(released),
                        result="locus_held_by_another_symbol",
                        candidates=tuple(sorted(set(names) - {name})),
                    ),
                ),
            )
    return out


def _step(resolution: GeneResolution) -> ResolutionStep:
    if resolution.step is None:
        raise ValueError(f"{resolution.symbol} is unresolved")
    return resolution.step


def resolve_symbols(
    symbols: Iterable[str],
    refseq_loci: Mapping[str, GenBankLocus],
    genbank: GenbankMember,
    mg1655_resolve: Callable[[str], GeneNameResolution],
) -> dict[str, GeneResolution]:
    """``{symbol: GeneResolution}`` for every distinct symbol, in sorted order, with
    two symbols on one locus settled by step precedence (:func:`_release_collisions`).
    """
    index = _RefseqIndex(refseq_loci)
    return _release_collisions(
        {
            s: resolve_symbol(s, index, genbank, mg1655_resolve)
            for s in sorted(set(symbols))
        }
    )


class RouteConflict(BaseModel):
    """A ``refseq_symbol`` resolution the GenBank route would place elsewhere."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    refseq_symbol_locus: str
    genbank_route: StepResult
    genbank_route_loci: tuple[str, ...]


def genbank_route_conflicts(
    resolutions: Mapping[str, GeneResolution],
    refseq_loci: Mapping[str, GenBankLocus],
    genbank: GenbankMember,
) -> tuple[RouteConflict, ...]:
    """Every step-1 resolution whose symbol the GenBank route would NOT place there.

    The GenBank route is step 2's rule run on a symbol step 1 already resolved; a symbol
    with no single GenBank gene feature is not compared. Each conflict is a finding for
    the note (the Table S4 description decides it), not a refusal.
    """
    index = _RefseqIndex(refseq_loci)
    conflicts: list[RouteConflict] = []
    for resolution in resolutions.values():
        if resolution.step != "refseq_symbol" or resolution.locus_tag is None:
            continue
        outcome, tag, _ = genbank_route(resolution.symbol, index, genbank)
        if outcome.result in ("no_feature", "several_features"):
            continue
        if tag == resolution.locus_tag:
            continue
        conflicts.append(
            RouteConflict(
                symbol=resolution.symbol,
                refseq_symbol_locus=resolution.locus_tag,
                genbank_route=outcome.result,
                genbank_route_loci=(tag,) if tag is not None else outcome.candidates,
            )
        )
    return tuple(conflicts)


class StepAgreement(BaseModel):
    """A step 2 resolution against step 3's rule run on the same symbol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agree: tuple[str, ...]
    disagree: tuple[str, ...]
    step3_unresolved: tuple[str, ...]


def step_agreement(
    resolutions: Mapping[str, GeneResolution],
    refseq_loci: Mapping[str, GenBankLocus],
    genbank: GenbankMember,
    mg1655_resolve: Callable[[str], GeneNameResolution],
) -> StepAgreement:
    """Run step 3's rule on every step-2 resolution: where both reach a locus, they
    should reach the same one. A disagreement is a finding, not a refusal.
    """
    index = _RefseqIndex(refseq_loci)
    agree: list[str] = []
    disagree: list[str] = []
    unresolved: list[str] = []
    for resolution in resolutions.values():
        if resolution.step != "genbank_symbol_eck_jw_b":
            continue
        _, tag, _ = mg1655_route(resolution.symbol, index, genbank, mg1655_resolve)
        if tag is None:
            unresolved.append(resolution.symbol)
        elif tag == resolution.locus_tag:
            agree.append(resolution.symbol)
        else:
            disagree.append(resolution.symbol)
    return StepAgreement(
        agree=tuple(agree), disagree=tuple(disagree), step3_unresolved=tuple(unresolved)
    )


class ResolutionReport(BaseModel):
    """How Table S4's symbols landed, by step, with the unresolved kept by name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_symbols: int
    resolved_by_step: dict[str, int]
    resolved_to_pseudogenes: int
    unresolved: tuple[str, ...]
    genbank_route_conflicts: tuple[RouteConflict, ...]
    step2_against_step3: StepAgreement


def resolution_report(
    resolutions: Mapping[str, GeneResolution],
    refseq_loci: Mapping[str, GenBankLocus],
    genbank: GenbankMember,
    mg1655_resolve: Callable[[str], GeneNameResolution],
) -> ResolutionReport:
    """The counts the dendron note states, recomputed from the resolutions."""
    return ResolutionReport(
        n_symbols=len(resolutions),
        resolved_by_step={
            step: sum(1 for r in resolutions.values() if r.step == step)
            for step in STEPS
        },
        resolved_to_pseudogenes=sum(
            1 for r in resolutions.values() if r.refseq_pseudo is True
        ),
        unresolved=tuple(
            sorted(s for s, r in resolutions.items() if r.status == "unresolved")
        ),
        genbank_route_conflicts=genbank_route_conflicts(
            resolutions, refseq_loci, genbank
        ),
        step2_against_step3=step_agreement(
            resolutions, refseq_loci, genbank, mg1655_resolve
        ),
    )


# --------------------------------------------------------------------------- #
# The measurement, written to RESULT
# --------------------------------------------------------------------------- #
#: Measured on 2026-10-12 by :func:`measure` (the result JSON holds the full ledger).
RESOLVED_BY_STEP: Final = {
    "refseq_symbol": 3_607,
    "genbank_symbol_eck_jw_b": 65,
    "mg1655_symbol_eck_jw_b": 9,
}
UNRESOLVED: Final = (
    "dcuC",
    "dmpG",
    "gatA",
    "mddA",
    "nanS",
    "pat",
    "rzoD",
    "tal",
    "tuf",
    "wbbL",
)


def measure(data_root: str) -> dict[str, object]:
    """Every number the script's dendron note states, off the pinned files."""
    rows = te.read_table_s4(te.raw_path(te.TABLE_S4_RELPATH, data_root))
    symbols = sorted({r.symbol for r in rows})
    w3110 = bacterial_genome("ecoli", "W3110", data_root)
    mg1655 = bacterial_genome("ecoli", "MG1655", data_root)
    if not isinstance(w3110, EcoliK12W3110Genome) or not isinstance(
        mg1655, EcoliK12MG1655Genome
    ):
        raise TypeError("the W3110 and MG1655 genomes resolved to the wrong classes")
    genbank = read_genbank_member(
        resolve(ECOLI_K12_W3110, GENBANK_MEMBER, data_root=data_root)
    )
    resolutions = resolve_symbols(
        symbols, w3110.genbank.loci, genbank, mg1655.resolve_gene_name
    )
    report = resolution_report(
        resolutions, w3110.genbank.loci, genbank, mg1655.resolve_gene_name
    )
    return {
        "annotation_routes": [r.model_dump() for r in annotation_routes(data_root)],
        "symbol_routes": symbol_routes(tuple(symbols), data_root),
        "resolution": report.model_dump(mode="json"),
        # step 1 is one exact /gene match; the ledger keeps every other symbol
        "ledger_off_step_1": [
            r.model_dump(mode="json")
            for r in resolutions.values()
            if r.step != "refseq_symbol"
        ],
        "values_per_unresolved_symbol": dict(
            sorted(
                Counter(
                    r.symbol
                    for r in rows
                    if resolutions[r.symbol].status == "unresolved"
                ).items()
            )
        ),
    }


def main() -> None:
    """Measure and write :data:`RESULT`."""
    load_dotenv()
    result = measure(os.environ["DATA_ROOT"])
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(RESULT)


if __name__ == "__main__":
    main()

# torchcell/datasets/gene_alias_resolution
# [[torchcell.datasets.gene_alias_resolution]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/gene_alias_resolution
# Test file: tests/torchcell/datasets/test_gene_alias_resolution.py
"""Refusing gene-name resolution for S. cerevisiae loaders that read common names (#886).

A common gene name can belong to more than one R64 ORF: ``TFC7`` is the standard name of
``YOR110W`` and a secondary alias of ``YNL039W`` (BDP1). A loader that takes the first
candidate stores whichever ORF the alias table happens to list first; Xue 2025 stored
``TFC7`` as ``YNL039W`` that way. This module replaces first-match with three outcomes:

- a live systematic name, or a name with exactly ONE candidate ORF, resolves to it;
- a name with several candidates resolves ONLY through a :class:`PinnedAliasResolution`
  the loader declares for it, and the pin is re-checked against its stated rule at build
  time (:class:`AliasResolutionRule`): either the R64 GFF names the chosen ORF as the
  name's standard name, or the paper's own gene list pairs the name with that ORF;
- anything else raises :class:`GeneNameRefused`, carrying a typed
  :class:`GeneNameRefusal` (reason + candidates). Nothing is guessed.

The candidate set of a name is the union of the genome's ``alias_to_systematic`` entry and
the live genes whose GFF ``gene=`` attribute (the SGD standard name) is that name, so a
standard name that some other gene lists as an alias counts as ambiguous even when the
owner does not repeat it in its own ``Alias`` list.

:func:`check_ambiguous_aliases` is the build-time check: given the (source name, stored
systematic name) pairs a loader is about to write, it refuses the build when any pair's
source name is ambiguous and the stored ORF is not the one its pin records.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.verification.report import Provenance


class GenomeNameIndex(Protocol):
    """The three genome lookups resolution reads (``SCerevisiaeGenome`` satisfies it)."""

    @property
    def gene_set(self) -> Any:
        """Live systematic gene ids (supports ``in``)."""
        ...

    @property
    def alias_to_systematic(self) -> dict[str, list[str]]:
        """Upper-case alias -> systematic ids listing it in their GFF ``Alias``."""
        ...

    @property
    def feature_index(self) -> dict[str, Any]:
        """Locus index; ``standard_to_ids`` maps a GFF ``gene=`` name to locus ids."""
        ...


class AliasResolutionRule(StrEnum):
    """The sourced rule a pinned resolution of an ambiguous name follows."""

    #: The R64 GFF gives the chosen ORF this name as its standard name (``gene=``),
    #: and no other live gene has it as its standard name.
    SGD_STANDARD_NAME = "sgd_standard_name"
    #: The paper's own released gene list pairs this name with the chosen ORF.
    PAPER_GENE_LIST = "paper_gene_list"


class GeneNameRefusalReason(StrEnum):
    """Why a source gene name was refused instead of stored."""

    #: Not a live systematic id, not an alias and not a standard name in R64.
    NOT_IN_GENOME = "not_in_genome"
    #: Several candidate ORFs and the loader declares no pinned resolution.
    AMBIGUOUS_ALIAS_UNPINNED = "ambiguous_alias_unpinned"
    #: The pinned ORF is not among the name's candidates in the injected genome.
    PIN_NOT_A_CANDIDATE = "pin_not_a_candidate"
    #: The pin's stated rule does not hold for the injected genome or paper list.
    PIN_RULE_CONTRADICTED = "pin_rule_contradicted"
    #: A stored ORF differs from the ORF the source name's pin records.
    STORED_ORF_DIFFERS_FROM_PIN = "stored_orf_differs_from_pin"


class GeneNameRefusal(BaseModel):
    """A typed refusal: the source name, why it was refused, and its candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    reason: GeneNameRefusalReason
    candidates: list[str] = Field(default_factory=list)
    detail: str


class GeneNameRefused(RuntimeError):
    """Raised when a source gene name cannot be resolved without guessing."""

    def __init__(self, refusal: GeneNameRefusal) -> None:
        """Carry the typed refusal; the message names the reason and candidates."""
        self.refusal = refusal
        super().__init__(
            f"gene name {refusal.name!r} refused ({refusal.reason.value}): "
            f"{refusal.detail}; candidates {refusal.candidates}"
        )


class PinnedAliasResolution(BaseModel):
    """One ambiguous name, the ORF a loader stores for it, and the quoted source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: str = Field(description="The ambiguous source name, upper case.")
    systematic_name: str = Field(description="The R64 ORF stored for the name.")
    rule: AliasResolutionRule
    provenance: Provenance
    quote: str = Field(
        description="Verbatim text of the source row that justifies the ORF."
    )

    @model_validator(mode="after")
    def _require_auditable(self) -> "PinnedAliasResolution":
        if self.alias != self.alias.upper():
            raise ValueError(f"pinned alias {self.alias!r} must be upper case")
        if not self.provenance.sha256:
            raise ValueError(f"pin {self.alias!r}: provenance.sha256 is required")
        quote = self.quote.upper()
        if self.systematic_name not in quote or self.alias not in quote:
            raise ValueError(
                f"pin {self.alias!r}: quote must contain the alias and the ORF "
                f"{self.systematic_name} (case-insensitive)"
            )
        return self


def pin_table(
    pins: Iterable[PinnedAliasResolution],
) -> dict[str, PinnedAliasResolution]:
    """Index pins by alias, refusing a duplicate alias."""
    table: dict[str, PinnedAliasResolution] = {}
    for pin in pins:
        if pin.alias in table:
            raise ValueError(f"alias {pin.alias!r} is pinned twice")
        table[pin.alias] = pin
    return table


def candidate_orfs(genome: GenomeNameIndex, name: str) -> list[str]:
    """Every live ORF a name can denote: alias-table entries plus standard-name owners."""
    n = name.strip().upper()
    standard_owners = [
        orf
        for orf in genome.feature_index["standard_to_ids"].get(n, [])
        if orf in genome.gene_set
    ]
    return sorted(set(genome.alias_to_systematic.get(n, [])) | set(standard_owners))


def _check_pin(
    genome: GenomeNameIndex,
    pin: PinnedAliasResolution,
    candidates: list[str],
    paper_gene_list: Mapping[str, str] | None,
) -> None:
    """Raise unless the pin names a candidate and its stated rule holds."""
    if pin.systematic_name not in candidates:
        raise GeneNameRefused(
            GeneNameRefusal(
                name=pin.alias,
                reason=GeneNameRefusalReason.PIN_NOT_A_CANDIDATE,
                candidates=candidates,
                detail=f"pinned ORF {pin.systematic_name} is not a candidate",
            )
        )
    if pin.rule is AliasResolutionRule.SGD_STANDARD_NAME:
        owners = sorted(
            orf
            for orf in genome.feature_index["standard_to_ids"].get(pin.alias, [])
            if orf in genome.gene_set
        )
        if owners != [pin.systematic_name]:
            raise GeneNameRefused(
                GeneNameRefusal(
                    name=pin.alias,
                    reason=GeneNameRefusalReason.PIN_RULE_CONTRADICTED,
                    candidates=candidates,
                    detail=(
                        f"rule {pin.rule.value}: the genome's standard-name owners are "
                        f"{owners}, not [{pin.systematic_name}]"
                    ),
                )
            )
        return
    listed = None if paper_gene_list is None else paper_gene_list.get(pin.alias)
    if listed != pin.systematic_name:
        raise GeneNameRefused(
            GeneNameRefusal(
                name=pin.alias,
                reason=GeneNameRefusalReason.PIN_RULE_CONTRADICTED,
                candidates=candidates,
                detail=(
                    f"rule {pin.rule.value}: the paper's gene list pairs the name with "
                    f"{listed}, not {pin.systematic_name}"
                ),
            )
        )


def resolve_gene_name_strict(
    genome: GenomeNameIndex,
    name: str,
    pins: Mapping[str, PinnedAliasResolution],
    paper_gene_list: Mapping[str, str] | None = None,
) -> str:
    """Resolve a source gene name to one live R64 ORF, or raise :class:`GeneNameRefused`.

    ``paper_gene_list`` maps an upper-case gene symbol to the ORF the paper's own released
    list gives it; it is required only by pins whose rule is ``PAPER_GENE_LIST``.
    """
    n = name.strip().upper()
    if n in genome.gene_set:
        return n
    candidates = candidate_orfs(genome, n)
    if not candidates:
        raise GeneNameRefused(
            GeneNameRefusal(
                name=n,
                reason=GeneNameRefusalReason.NOT_IN_GENOME,
                detail="no live gene, alias or standard name matches",
            )
        )
    if len(candidates) == 1:
        return candidates[0]
    pin = pins.get(n)
    if pin is None:
        raise GeneNameRefused(
            GeneNameRefusal(
                name=n,
                reason=GeneNameRefusalReason.AMBIGUOUS_ALIAS_UNPINNED,
                candidates=candidates,
                detail="ambiguous name with no pinned resolution in the loader",
            )
        )
    _check_pin(genome, pin, candidates, paper_gene_list)
    return pin.systematic_name


class AmbiguousAliasAudit(BaseModel):
    """What the build-time check saw: pairs checked and the ambiguous names in them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int
    ambiguous: dict[str, str] = Field(
        description="Ambiguous source name -> the stored (pinned) ORF."
    )


def check_ambiguous_aliases(
    genome: GenomeNameIndex,
    pairs: Iterable[tuple[str, str]],
    pins: Mapping[str, PinnedAliasResolution],
    paper_gene_list: Mapping[str, str] | None = None,
) -> AmbiguousAliasAudit:
    """Refuse a build whose stored ORF came from an ambiguous name without its pin.

    ``pairs`` are (source gene name, stored systematic name). A pair whose source name is
    the stored systematic name, or a live systematic id, is not a resolution and is
    skipped. Every other source name with more than one candidate ORF must have a pin
    whose rule holds and whose ORF equals the stored one.
    """
    n_pairs = 0
    ambiguous: dict[str, str] = {}
    for source, stored in pairs:
        n_pairs += 1
        n = source.strip().upper()
        if n == stored.upper() or n in genome.gene_set:
            continue
        candidates = candidate_orfs(genome, n)
        if len(candidates) <= 1:
            continue
        pin = pins.get(n)
        if pin is None:
            raise GeneNameRefused(
                GeneNameRefusal(
                    name=n,
                    reason=GeneNameRefusalReason.AMBIGUOUS_ALIAS_UNPINNED,
                    candidates=candidates,
                    detail=f"stored as {stored} with no pinned resolution",
                )
            )
        _check_pin(genome, pin, candidates, paper_gene_list)
        if pin.systematic_name != stored:
            raise GeneNameRefused(
                GeneNameRefusal(
                    name=n,
                    reason=GeneNameRefusalReason.STORED_ORF_DIFFERS_FROM_PIN,
                    candidates=candidates,
                    detail=f"stored as {stored}, pinned to {pin.systematic_name}",
                )
            )
        ambiguous[n] = stored
    return AmbiguousAliasAudit(n_pairs=n_pairs, ambiguous=ambiguous)

# torchcell/datasets/systematic_name_currency
# [[torchcell.datasets.systematic_name_currency]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/systematic_name_currency
# Test file: tests/torchcell/datasets/test_systematic_name_currency.py
"""Released SGA systematic names that R64-4-1 no longer carries as loci (#893, #897).

The Costanzo 2016 and Kuzmin 2020 releases name every strain by the ORF its deletion
collection was built against, and 51 of those names are not live loci of the R64-4-1
annotation. Each one is decided here, by a stated rule, and the decision is re-checked
against the injected genome on every build:

- **renamed** (16): SGD's locus history of the current ORF records the released name as
  that ORF's former systematic name, changed for nomenclature. The name maps to the
  current ORF through a :class:`PinnedAliasResolution` whose rule is
  ``SGD_SYSTEMATIC_RENAME`` and whose quote is that history sentence.
- **merged** (28): R64 keeps the released ORF only as an ``Alias`` of a longer locus it
  was merged into. The released strain deleted part of that locus, not the locus, and
  the release carries a separate deletion strain of the locus itself, so the record is
  refused (``MERGED_INTO_LOCUS``) rather than stored under either name.
- **retired** (7): R64 has no locus id, standard name or alias for the name. The record
  is refused (``RETIRED_SYSTEMATIC_NAME``).

A released name that is not a live locus and is in neither table stops the build with
``NOT_CURRENT_UNPINNED``: no name is mapped, kept or dropped by default.

Sources. The R64-4-1 GFF (``R64_GFF``, the genomes-tier file, sha256 ``64f61e31...``) says
what is current and which locus lists a released name as an alias, and that is what the
build re-checks. The GFF does not say whether an alias records a rename or a merge, so
that distinction is read from SGD's per-locus history: the locus JSON cached by
``SCerevisiaeGenome`` under ``$DATA_ROOT/data/sgd/genome/genes/<ORF>.json`` (file mtime
2025-08-28, read from disk, no SGD call), each file pinned by its own sha256 here and its
quote re-read by ``test_every_pin_matches_its_source_bytes``. The four released names
merged into a non-gene locus (a blocked reading frame or a pseudogene) have no cached
locus JSON, so their quote is the locus's GFF row.

The module lives under ``torchcell/datasets/`` because the Costanzo 2016 and Kuzmin
2020 loaders are on the raw-pin debt list (``test_raw_pins.UNPINNED_LOADERS``), which
refuses any sha256-bearing identifier in their modules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.datamodels.strain_background import R64_GFF
from torchcell.datasets.gene_alias_resolution import (
    AliasResolutionRule,
    GeneNameRefusal,
    GeneNameRefusalReason,
    GeneNameRefused,
    GenomeNameIndex,
    PinnedAliasResolution,
    check_pin,
    pin_table,
)
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.verification.report import Provenance

__all__ = [
    "SGA_SYSTEMATIC_REFUSALS",
    "SGA_SYSTEMATIC_RENAMES",
    "SYSTEMATIC_NAME_LEDGER_FILENAME",
    "PinnedSystematicNameRefusal",
    "SystematicNameCurrency",
    "SystematicNameGenome",
    "SystematicNameLedger",
    "SystematicNameOutcome",
    "refusal_table",
]

#: Written by each loader beside ``data.csv`` in ``preprocess/``.
SYSTEMATIC_NAME_LEDGER_FILENAME = "systematic_name_ledger.json"


class SystematicNameGenome(GenomeNameIndex, Protocol):
    """The lookups the currency check reads (``SCerevisiaeGenome`` satisfies it)."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        """The layered R64 resolution of one name."""
        ...


_REFUSABLE = frozenset(
    {
        GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
        GeneNameRefusalReason.MERGED_INTO_LOCUS,
    }
)


class PinnedSystematicNameRefusal(BaseModel):
    """One released systematic name refused by a stated, re-checkable reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="The released systematic name, upper case.")
    reason: GeneNameRefusalReason
    locus: str | None = Field(
        description="The locus a merged ORF is now part of; None for a retired name."
    )
    provenance: Provenance
    quote: str | None = Field(
        description="Verbatim source text naming the released ORF and its locus (merged "
        "only; a retired name is an absence, re-checked against the genome)."
    )
    basis: str = Field(description="What the source and the release show, in one line.")

    @model_validator(mode="after")
    def _require_auditable(self) -> PinnedSystematicNameRefusal:
        if self.name != self.name.upper():
            raise ValueError(f"refused name {self.name!r} must be upper case")
        if self.reason not in _REFUSABLE:
            raise ValueError(
                f"{self.name}: a pinned refusal is retired or merged, not "
                f"{self.reason.value}"
            )
        if not self.provenance.sha256:
            raise ValueError(f"{self.name}: provenance.sha256 is required")
        if self.reason is GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME:
            if self.locus is not None or self.quote is not None:
                raise ValueError(f"{self.name}: a retired name has no locus or quote")
            return self
        if self.locus is None or self.quote is None:
            raise ValueError(f"{self.name}: a merged name needs its locus and a quote")
        quote = self.quote.upper()
        if self.name not in quote or self.locus not in quote:
            raise ValueError(
                f"{self.name}: quote must contain the released name and the locus "
                f"{self.locus} (case-insensitive)"
            )
        return self


def refusal_table(
    refusals: Iterable[PinnedSystematicNameRefusal],
) -> dict[str, PinnedSystematicNameRefusal]:
    """Index refusals by name, refusing a duplicate name."""
    table: dict[str, PinnedSystematicNameRefusal] = {}
    for refusal in refusals:
        if refusal.name in table:
            raise ValueError(f"name {refusal.name!r} is refused twice")
        table[refusal.name] = refusal
    return table


class SystematicNameOutcome(BaseModel):
    """What happened to one released name, and how many input rows carried it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    released_name: str
    stored_name: str | None = Field(description="The current ORF; None when refused.")
    reason: GeneNameRefusalReason | None = Field(description="None when renamed.")
    locus: str | None
    n_rows: int = Field(description="Input rows naming it in a checked column.")


class SystematicNameLedger(BaseModel):
    """Every non-current released name a build met, and the rows it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str
    annotation: str = Field(description="The genome release the names were judged by.")
    n_rows_in: int
    n_rows_refused: int
    n_rows_out: int
    renamed: list[SystematicNameOutcome]
    refused: list[SystematicNameOutcome]

    @model_validator(mode="after")
    def _rows_add_up(self) -> SystematicNameLedger:
        if self.n_rows_in - self.n_rows_refused != self.n_rows_out:
            raise ValueError(
                f"{self.dataset_name}: {self.n_rows_in} rows in - "
                f"{self.n_rows_refused} refused != {self.n_rows_out} out"
            )
        return self


def _live(resolution: GeneNameResolution, name: str) -> bool:
    """A live R64 gene, or a non-gene locus (pseudogene, blocked reading frame) by its id."""
    if resolution.status is GeneNameStatus.CURRENT:
        return True
    return (
        resolution.status is GeneNameStatus.NON_GENE_FEATURE
        and resolution.systematic_name == name
    )


def _refusal_holds(
    genome: SystematicNameGenome, refusal: PinnedSystematicNameRefusal
) -> None:
    """Raise unless the genome still shows what the refusal states."""
    resolution = genome.resolve_gene_name(refusal.name)
    if refusal.reason is GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME:
        holds = resolution.status is GeneNameStatus.RETIRED
    else:
        holds = (
            resolution.status
            in (GeneNameStatus.RENAMED, GeneNameStatus.NON_GENE_FEATURE)
            and resolution.systematic_name == refusal.locus
        )
    if not holds:
        raise GeneNameRefused(
            GeneNameRefusal(
                name=refusal.name,
                reason=GeneNameRefusalReason.PIN_RULE_CONTRADICTED,
                candidates=[]
                if resolution.systematic_name is None
                else [resolution.systematic_name],
                detail=(
                    f"pinned {refusal.reason.value} (locus {refusal.locus}), but the "
                    f"genome resolves it as {resolution.status.value} -> "
                    f"{resolution.systematic_name}"
                ),
            )
        )


class SystematicNameCurrency:
    """Map renamed and refuse retired or merged released names, counting both.

    Construction re-checks every pin and refusal against the injected genome, so a
    genome whose annotation no longer shows what a pin states stops the build before any
    row is read. :meth:`apply` may be called once per record kind; the counts add up
    across calls and :meth:`ledger` reports them.
    """

    def __init__(
        self,
        genome: SystematicNameGenome,
        renames: Mapping[str, PinnedAliasResolution],
        refusals: Mapping[str, PinnedSystematicNameRefusal],
    ) -> None:
        """Check every pin and refusal against ``genome``."""
        overlap = sorted(set(renames) & set(refusals))
        if overlap:
            raise ValueError(f"names both renamed and refused: {overlap}")
        for pin in renames.values():
            if pin.rule is not AliasResolutionRule.SGD_SYSTEMATIC_RENAME:
                raise ValueError(
                    f"{pin.alias}: a systematic-name pin must use "
                    f"{AliasResolutionRule.SGD_SYSTEMATIC_RENAME.value}"
                )
            check_pin(genome, pin)
        for refusal in refusals.values():
            _refusal_holds(genome, refusal)
        self._genome = genome
        self._renames = renames
        self._refusals = refusals
        self._decided: dict[str, str | None] = {}
        self._rows: Counter[str] = Counter()
        self._n_in = 0
        self._n_refused = 0

    def stored_name(self, name: str) -> str | None:
        """The name to store for a released name, or None when its rows are refused."""
        if name in self._decided:
            return self._decided[name]
        if name in self._renames:
            stored: str | None = self._renames[name].systematic_name
        elif name in self._refusals:
            stored = None
        elif _live(self._genome.resolve_gene_name(name), name):
            stored = name
        else:
            resolution = self._genome.resolve_gene_name(name)
            raise GeneNameRefused(
                GeneNameRefusal(
                    name=name,
                    reason=GeneNameRefusalReason.NOT_CURRENT_UNPINNED,
                    candidates=[]
                    if resolution.systematic_name is None
                    else [resolution.systematic_name],
                    detail=(
                        f"resolves as {resolution.status.value} with neither a rename "
                        "pin nor a pinned refusal"
                    ),
                )
            )
        self._decided[name] = stored
        return stored

    def apply(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """``frame`` without refused rows and with renamed names replaced in ``columns``.

        A row is refused when any of ``columns`` holds a refused name. Every cell of
        ``columns`` must be a name: a blank cell raises, since it would otherwise pass
        unchecked. The index is kept; the caller resets it.
        """
        blank = frame[list(columns)].isna().any(axis=1)
        if blank.any():
            raise ValueError(
                f"{int(blank.sum())} rows have a blank systematic name in {list(columns)}"
            )
        names = pd.unique(frame[list(columns)].to_numpy().ravel())
        decided = {str(name): self.stored_name(str(name)) for name in names}
        renamed = {
            name: new for name, new in decided.items() if new not in (None, name)
        }
        refused_names = [name for name, new in decided.items() if new is None]
        refused = pd.Series(False, index=frame.index)
        for column in columns:
            carried = frame[column].isin(list(renamed) + refused_names)
            for name, n_rows in frame.loc[carried, column].value_counts().items():
                self._rows[str(name)] += int(n_rows)
            refused |= frame[column].isin(refused_names)
        self._n_in += len(frame)
        self._n_refused += int(refused.sum())
        kept = frame.loc[~refused].copy()
        for column in columns:
            kept[column] = kept[column].replace(renamed)
        return kept

    def ledger(self, dataset_name: str) -> SystematicNameLedger:
        """The renamed and refused names this build met, with their row counts."""
        renamed: list[SystematicNameOutcome] = []
        refused: list[SystematicNameOutcome] = []
        for name in sorted(self._rows):
            if name in self._renames:
                renamed.append(
                    SystematicNameOutcome(
                        released_name=name,
                        stored_name=self._renames[name].systematic_name,
                        reason=None,
                        locus=self._renames[name].systematic_name,
                        n_rows=self._rows[name],
                    )
                )
            else:
                refusal = self._refusals[name]
                refused.append(
                    SystematicNameOutcome(
                        released_name=name,
                        stored_name=None,
                        reason=refusal.reason,
                        locus=refusal.locus,
                        n_rows=self._rows[name],
                    )
                )
        return SystematicNameLedger(
            dataset_name=dataset_name,
            annotation="S288C R64-4-1 (2023-08-30)",
            n_rows_in=self._n_in,
            n_rows_refused=self._n_refused,
            n_rows_out=self._n_in - self._n_refused,
            renamed=renamed,
            refused=refused,
        )


# =========================================================================== #
# The 51 released SGA names that are not live R64-4-1 loci (measured 2026-10-11 by
# experiments/036-dataset-fixes-before-kg-build/scripts/yeast_gene_id_currency.py over
# every systematic name of the Costanzo 2016 Data File S1 and the Kuzmin 2020 Tables
# S1/S3/S5). Each quote is a verbatim substring of the cited file; " [...] " joins two
# substrings of the same file.
# =========================================================================== #


def _sgd_locus(orf: str, sha256: str) -> Provenance:
    """Provenance of one cached SGD locus record (``locus.history[].note``)."""
    return Provenance(
        source_uri=f"data/sgd/genome/genes/{orf}.json",
        sha256=sha256,
        method="SGD locus record cached by SCerevisiaeGenome; the quote is a substring "
        "of one locus.history[].note (after its bold category label)",
        page="locus.history",
        retrieved="2025-08-28",
    )


def _rename(alias: str, orf: str, sha256: str, quote: str) -> PinnedAliasResolution:
    return PinnedAliasResolution(
        alias=alias,
        systematic_name=orf,
        rule=AliasResolutionRule.SGD_SYSTEMATIC_RENAME,
        provenance=_sgd_locus(orf, sha256),
        quote=quote,
    )


def _merged(
    name: str, locus: str, provenance: Provenance, quote: str, basis: str
) -> PinnedSystematicNameRefusal:
    return PinnedSystematicNameRefusal(
        name=name,
        reason=GeneNameRefusalReason.MERGED_INTO_LOCUS,
        locus=locus,
        provenance=provenance,
        quote=quote,
        basis=basis,
    )


def _retired(name: str) -> PinnedSystematicNameRefusal:
    return PinnedSystematicNameRefusal(
        name=name,
        reason=GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
        locus=None,
        provenance=R64_GFF,
        quote=None,
        basis="no locus ID, gene= or Alias= value in the R64-4-1 GFF",
    )


_NOMENCLATURE = (
    "{old} used to be the systematic name for this ORF. Since this name was incorrect, "
    "this ORF name has been changed to {new} to properly represent the systematic ORF "
    "nomenclature."
)
_SGD_1998 = (
    "{lead} name: {old}; new name: {new}; date: 11/1998; old coord: {coord}; "
    "SGDID: {sgdid}; Name changed due to nomenclature"
)

#: Released names SGD renamed for nomenclature; each maps to the current ORF. All 16 are
#: Kuzmin 2020 array strains; Costanzo 2016 names the same strain by the current ORF
#: (``YAL043C-A_dma42`` there is ``YAL042C-A_dma42``) for 13 of them.
SGA_SYSTEMATIC_RENAMES: dict[str, PinnedAliasResolution] = pin_table(
    [
        _rename(
            "YAL043C-A",
            "YAL042C-A",
            "d0dbe21193683dbc450df04f9515d959ce3e9da9ba2d24d9249c65a967624e3e",
            _NOMENCLATURE.format(old="YAL043C-A", new="YAL042C-A"),
        ),
        _rename(
            "YAL058C-A",
            "YAL056C-A",
            "347f65f57391e6123300566212dea753e5167eaac898375e1f139f7801e36c64",
            "Please be advised that YAL058C-A used to be the systematic name for this "
            "ORF. This name was incorrect. The systematic name of this ORF has corrected "
            "to YAL056C-A.",
        ),
        _rename(
            "YBR090C-A",
            "YBR089C-A",
            "10c488fca0751ddce20ef7a3d89503bb24986ef3e85ffb8ebe099d10b6ebf2ab",
            _SGD_1998.format(
                lead="old",
                old="YBR090C-A",
                new="YBR089C-A",
                coord="ChrII 426447 426148",
                sgdid="S0002157",
            ),
        ),
        _rename(
            "YFL035C-B",
            "YFL034C-A",
            "6d8014f3bf1a58d3d12b879e730339e24e2431388ab8c7d4dc5df4a7148f64ad",
            _SGD_1998.format(
                lead="Old",
                old="YFL035C-B",
                new="YFL034C-A",
                coord="ChrVI 64932 64243",
                sgdid="S0002967",
            ),
        ),
        _rename(
            "YHR079C-B",
            "YHR079C-A",
            "644ed6995cc1744a680cda66bab43064b2cb9e5554723e95d17ee160a6d286d4",
            _SGD_1998.format(
                lead="Old",
                old="YHR079C-B",
                new="YHR079C-A",
                coord="ChrVIII 262554 262402",
                sgdid="S0001957",
            ),
        ),
        _rename(
            "YIL015C-A",
            "YIL014C-A",
            "cbd286b04d19708dd08aea35f82071a08f925affc81297192b7b22eeb694fb6a",
            _NOMENCLATURE.format(old="YIL015C-A", new="YIL014C-A"),
        ),
        _rename(
            "YIR020W-B",
            "YIR020W-A",
            "12fdbf5ef12946c6b528923862dcd4c74dab587fd15803f680a1008b338f393d",
            _NOMENCLATURE.format(old="YIR020W-B", new="YIR020W-A"),
        ),
        _rename(
            "YJL206C-A",
            "YJL205C",
            "92328fbeea8552208c342d974d58bf18c87173561d2e17a60937203eb5606985",
            _SGD_1998.format(
                lead="Old",
                old="YJL206C-A",
                new="YJL205C-A",
                coord="ChrX 50443 50139",
                sgdid="S0003742",
            )
            + " [...] Please be advised that YJL205C-A used to be the systematic name "
            "for this ORF. Since this name was incorrectly named, this ORF name has been "
            "changed to YJL205C to properly represent the systematic ORF nomenclature.",
        ),
        _rename(
            "YML010C-B",
            "YML009C-A",
            "23bd8e2fe9f4acd815c9ae87cf21c5d72006fea67e054677bc102ddfd175d667",
            "Please be advised that between 7-17-99 and 6-14-02 the systematic name for "
            "this ORF was YML010C-B. This name was incorrect, and in order to properly "
            "represent the systematic nomenclature, the name has been changed to "
            "YML009C-A.",
        ),
        _rename(
            "YML035C-A",
            "YML034C-A",
            "9c3d6f40abf7aca1238690c11a1b205e8a354fe5f0880de345e1dcfd828a1fdd",
            _NOMENCLATURE.format(old="YML035C-A", new="YML034C-A"),
        ),
        _rename(
            "YML048W-A",
            "YML047W-A",
            "8a3feeacce2beb025d5cc46b82eb24f33f18691bf421669952e2220cf40aa266",
            _NOMENCLATURE.format(old="YML048W-A", new="YML047W-A"),
        ),
        _rename(
            "YML058C-A",
            "YML057C-A",
            "0b3d2130b74ef1da19b55f6b105aaecdd4b183e4fd4cc50f386701d1f8244d3b",
            _NOMENCLATURE.format(old="YML058C-A", new="YML057C-A"),
        ),
        _rename(
            "YML102C-A",
            "YML101C-A",
            "9141b08a437b6abd7550b8bc49d52194e89d0cb62401a6642f16d2e874d36560",
            _NOMENCLATURE.format(old="YML102C-A", new="YML101C-A"),
        ),
        _rename(
            "YML117W-A",
            "YML116W-A",
            "ca499fa7c2bfd1e4991f1eb1c8b408f44f246890538a9948906082fe49b1d72f",
            _NOMENCLATURE.format(old="YML117W-A", new="YML116W-A"),
        ),
        _rename(
            "YMR158W-A",
            "YMR158W-B",
            "2bcb6cb968a0b05bd9cbd9c2bdcf4cd5696c105f79e38f8e64a3cfdc6edac265",
            _NOMENCLATURE.format(old="YMR158W-A", new="YMR158W-B"),
        ),
        _rename(
            "YOL053C-A",
            "YOL052C-A",
            "1b7d705dc00d805e780717aef6aa061cd649eb9da53c88c296af53b38fc95ffe",
            _SGD_1998.format(
                lead="Old",
                old="YOL053C-A",
                new="YOL052C-A",
                coord="ChrXV 231753 231568",
                sgdid="S0005413",
            ),
        ),
    ]
)

_TWO_STRAINS = "each release naming it also carries a deletion strain of {locus}"


def _merged_gene(
    name: str, locus: str, sha256: str, quote: str
) -> PinnedSystematicNameRefusal:
    return _merged(
        name,
        locus,
        _sgd_locus(locus, sha256),
        quote,
        f"SGD merged {name} into the gene {locus}; " + _TWO_STRAINS.format(locus=locus),
    )


def _merged_locus(
    name: str, locus: str, feature: str, quote: str
) -> PinnedSystematicNameRefusal:
    return _merged(
        name,
        locus,
        R64_GFF,
        quote,
        f"R64 keeps {name} only as an Alias of the {feature} {locus}; "
        + _TWO_STRAINS.format(locus=locus),
    )


#: Released names R64 does not carry as loci, each refused with its reason: 24 ORFs
#: merged into a gene, 4 merged into a blocked reading frame or pseudogene, 7 retired.
SGA_SYSTEMATIC_REFUSALS: dict[str, PinnedSystematicNameRefusal] = refusal_table(
    [
        _merged_gene(
            "YAR044W",
            "YAR042W",
            "8680da04cea2db0188f597cfb57b5ba6444a59941898d215d0ae456290e5e458",
            "Insertion of a single G after the G at 193300 merged SWH1/YAR042W and "
            "OSH1/YAR044W.",
        ),
        _merged_gene(
            "YBR075W",
            "YBR074W",
            "9db6b9759eaf4f9174e9235c231a20c1fcd60451e4cc4d622323795161a665ec",
            "Due to deletion of a C after the C at position 387430 and a deletion of G "
            "after the G at position 387625, YBR074W and YBR075W were merged.",
        ),
        _merged_gene(
            "YBR100W",
            "YBR098W",
            "ed49c7da65ddad7784f40f85d4f07a93f6ca386e12795f5057b50856af2efa05",
            "Due to a change to the systematic sequence of chromosome II (C deletion at "
            "442868), YBR098W and YBR100W have been merged to one single ORF of 691 aa.",
        ),
        _merged_gene(
            "YCL012W",
            "YCL014W",
            "666a5f4d1ea2a20adeb2f45adb7cfb88c7b1bcafe3f7390f78f780f6b9bebba0",
            "Several sequence changes were made in the systematic sequence in the region "
            "encompassing feature YCL014W, resulting in the merge of ORF YCL012W into "
            "YCL014W.",
        ),
        _merged_gene(
            "YCL060C",
            "YCL061C",
            "8e6f9ea8f3d5dd001dacbd1dc68934f7720018f2400abd9e101abcf61e54980f",
            "Due to several sequence changes in the region encompassing ORFs YCL060C and "
            "YCL061C, YCL060C has been merged into adjacent ORF YCL061C.",
        ),
        _merged_gene(
            "YCL062W",
            "YCL063W",
            "d9e9b0eb11bfe364a06fe4139d464524f0135c06a5ea39ab7fdeac00a2d9347e",
            "Due to several sequence changes in the region encompassing ORFs YCL062W and "
            "YCL063W, YCL062W has been merged into adjacent ORF YCL063W.",
        ),
        _merged_gene(
            "YCR062W",
            "YCR061W",
            "396380ac53581d14ffd57d62967112da1c23e41aefa53f7c8d64f88294ac6408",
            "After resequencing and annotation of Chromosome III, ORF YCR062W has been "
            "merged into neighboring ORF YCR061W.",
        ),
        _merged_gene(
            "YDL038C",
            "YDL039C",
            "f26ba3b71213c769697f583f3b0063a4b4f006bd97ff5646d76d971b8e04408e",
            "As a result of this change, ORF YDL038C was merged into ORF PRM7/YDL039C.",
        ),
        _merged_gene(
            "YDR474C",
            "YDR475C",
            "46bb901c9b6dab312e76e397bc1904b194d5d9f3208766ec8c7a7d370da93a8d",
            "Due to insertion of G at position 1409511 and a C at position 1409657, "
            "YDR474C and JIP4/YDR475C have been merged.",
        ),
        _merged_gene(
            "YFL006W",
            "YFL007W",
            "1a344d074e169cefc55c64f0775778866d083d029d8bea3d2ded4c7a203ec967",
            "Due to the insertion of a G at 128870 on Chromosome VI, YFL006W has been "
            "merged with BLM10/YFL007W.",
        ),
        _merged_gene(
            "YFL043C",
            "YFL042C",
            "e8b2ae41dcbfacccddc6c6ca3152716cd24162a2da77f02f33b84d8f5fe489f3",
            "Due to the deletion of G at chromosomal coordinate 46151, which was within "
            "the ORF YFL043C, YFL043C is now part of YFL042C.",
        ),
        _merged_gene(
            "YFR024C",
            "YFR024C-A",
            "8779ca4e9061bd5284770deeaf9543cc7e6502a56a67fee7ab3c8e6c7b371957",
            "YFR024C is not a genuine ORF, but is the C-terminus of LSB3/YFR024C-A.",
        ),
        _merged_gene(
            "YGL046W",
            "YGL045W",
            "c324ee4d681152b3a78f1443d59c2622af8d1a61a30d3cb19e96eef8bc78596f",
            "Due to the following changes to the systematic sequence of Chromosome VII, "
            "YGL045W and YGL046W have been merged into one single larger ORF",
        ),
        _merged_gene(
            "YJL017W",
            "YJL016W",
            "273119f070c50ab14d1bd903393c8ce52467332430da160994a6d948f4965740",
            "Due to insertion of G after the G at position 406238, the ORF YJL017W "
            "(405284-406261 (1-978)) was merged into YJL016W (406453-406968 (1-516)).",
        ),
        _merged_gene(
            "YJL021C",
            "YJL020C",
            "f18b7ef688c225a1fdc73192a6ebb14c11463a861647ed3089761682452f502b",
            "Due to these 4 G insertions, the earlier ORFs YJL020C and YJL021C are now "
            "merged into one long ORF YJL020C of 1157 amino acids.",
        ),
        _merged_gene(
            "YKL158W",
            "YKL157W",
            "ef3e8f2330abf65409f8d0af1381ff20902ae3b4f8fb07ca42669b4126c252ae",
            "YKL157W and YKL158W were merged into a single ORF, designated YKL157W, based "
            "on evidence published by Davis et al. 2000 demonstrating the presence of an "
            "intron between these two ORFs.",
        ),
        _merged_gene(
            "YKL199C",
            "YKL198C",
            "d5c6714d288eec7acc78b4334869c32c2f4fbe0be58e832ce5aa238727d0e9c3",
            "Due to the deletion of the G at coordinate 69156 on Chromosome XI, YKL198C "
            "and YKL199C have been merged into one single ORF - PTK1/YKL198C - encoding "
            "a protein of 649 amino acids (1950 nucleotides).",
        ),
        _merged_gene(
            "YKL200C",
            "YKL201C",
            "04f98bbefde65705a14e81048f0c2173dcfa86720246052ac2c7e08b9b22f67f",
            "MIPS made sequence corrections in the area covering YKL200C that resulted in "
            "its subsumption into MNN4/YKL201C, which now encompasses both ORFs plus "
            "additional sequence (coordinates 67463-63927).",
        ),
        _merged_gene(
            "YLR312C-B",
            "YLR313C",
            "74ef84aac5f06261f1245e1ef91d8f8296b16255aec4eb59e6f94c5a82c9305f",
            "As a consequence of this sequence change, SPH1/YLR313C was extended at the "
            "3' and merged with an adjacent ORF, YLR312C-B, that was in the same frame.",
        ),
        _merged_gene(
            "YLR391W",
            "YLR390W-A",
            "4fde822d7302a5ca11dd80f380f38f1e4566fc1c2b4f8542cfc6f8d159696ce4",
            "YLR391W was an ORF that was named in the original sequence of Chromosome "
            "XII. Since it was completely contained in a larger ORF with a better codon "
            "bias, it was deleted from SGD and GenBank in favor of the larger ORF, "
            "YLR391W-A. [...] Due to a nomenclature error, the systematic name of ORF "
            "YLR391W-A was changed to YLR390W-A.",
        ),
        _merged_gene(
            "YML033W",
            "YML034W",
            "cc5a05d3ee9b0ec84389aad759a2d1adf0533b211ef14aa62bf4e5e52e543102",
            "Previously annotated as two ORFs: YML033W and YML034W.Dr. Perez-Ortin "
            "identified an intron that removes the stop codon for YML034W; YML033W and "
            "YML034W are now one ORF, designated YML034W.",
        ),
        _merged_gene(
            "YOR088W",
            "YOR087W",
            "617fa8202f9069bc1a4ef4118dea95be8bd251826931a9f7b928205f41d05494",
            "Due to the insertion of a T after the A at 488439 on chromosome 15, YOR087W "
            "and YOR088W have been merged to one single ORF - YVC1/YOR087W, with YOR088W "
            "as an alias.",
        ),
        _merged_gene(
            "YOR240W",
            "YOR239W",
            "f7866514cb5590544bae581ee8b922d0d27c342d96832a42eba2b6c90798f42e",
            "it has been reported in Asakura et al. 1998 that a +1 translational "
            "frameshift merges YOR239W and YOR240W into one ORF.",
        ),
        _merged_gene(
            "YPR090W",
            "YPR089W",
            "cc8c9b3a7f5dab2d5f28508cd5389a5eb648658b345cc990cf544539c46853e7",
            "Due to deletion of a T at position 713683, YPR089W and YPR090W were merged.",
        ),
        _merged_locus(
            "YER108C",
            "YER109C",
            "blocked reading frame",
            "blocked_reading_frame\t375215\t377614\t.\t-\t.\tID=YER109C;Name=YER109C;"
            "gene=FLO8;Alias=FLO8,PHD5,STA10,YER108C",
        ),
        _merged_locus(
            "YFL057C",
            "YFL056C",
            "pseudogene",
            "pseudogene\t14305\t15431\t.\t-\t.\tID=YFL056C;Name=YFL056C;gene=AAD6;"
            "Alias=AAD6,YFL057C",
        ),
        _merged_locus(
            "YIL168W",
            "YIL167W",
            "blocked reading frame",
            "blocked_reading_frame\t29032\t30048\t.\t+\t.\tID=YIL167W;Name=YIL167W;"
            "gene=SDL1;Alias=SDL1,L-serine%20ammonia-lyase,YIL168W",
        ),
        _merged_locus(
            "YIR044C",
            "YIR043C",
            "blocked reading frame",
            "blocked_reading_frame\t437043\t438179\t.\t-\t.\tID=YIR043C;Name=YIR043C;"
            "Alias=YIR044C",
        ),
        _retired("YAR037W"),
        _retired("YAR040C"),
        _retired("YAR043C"),
        _retired("YCL006C"),
        _retired("YCL013W"),
        _retired("YCL026C"),
        _retired("YIL080W"),
    ]
)

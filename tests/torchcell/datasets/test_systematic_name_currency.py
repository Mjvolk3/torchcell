# tests/torchcell/datasets/test_systematic_name_currency.py
# [[tests.torchcell.datasets.test_systematic_name_currency]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_systematic_name_currency.py
"""Released SGA systematic names R64 no longer carries (#893, #897) on a genome stub.

Stub genome: live genes YAL042C-A, YAR042W, YAL001C; a live blocked reading frame
YER109C (a non-gene locus resolving to itself); YAL043C-A is an alias of YAL042C-A
alone (the rename), YAR044W an alias of YAR042W (the merge), YER108C an alias of the
non-gene locus YER109C (a merge into a non-gene locus), YAR037W absent (retired), and
YZZ999W an alias of YAL001C that no table pins.

The last two tests read the real tables: their counts and classes (hermetic), and, when
``DATA_ROOT`` holds the files, every pin and refusal against its source bytes and the
real R64-4-1 genome.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from torchcell.datamodels.strain_background import R64_GFF
from torchcell.datasets import systematic_name_currency as snc
from torchcell.datasets.gene_alias_resolution import (
    AliasResolutionRule,
    GeneNameRefusalReason,
    GeneNameRefused,
    PinnedAliasResolution,
    pin_table,
)
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.verification.report import Provenance

_PROV = Provenance(source_uri="sgd", sha256="0" * 64)

_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str | None]] = {
    "YAL042C-A": (GeneNameStatus.CURRENT, "YAL042C-A"),
    "YAR042W": (GeneNameStatus.CURRENT, "YAR042W"),
    "YAL001C": (GeneNameStatus.CURRENT, "YAL001C"),
    "YER109C": (GeneNameStatus.NON_GENE_FEATURE, "YER109C"),
    "YAL043C-A": (GeneNameStatus.RENAMED, "YAL042C-A"),
    "YAR044W": (GeneNameStatus.RENAMED, "YAR042W"),
    "YER108C": (GeneNameStatus.NON_GENE_FEATURE, "YER109C"),
    "YAR037W": (GeneNameStatus.RETIRED, "YAR037W"),
    "YZZ999W": (GeneNameStatus.RENAMED, "YAL001C"),
}


class _Genome:
    """Three lookups of ``GenomeNameIndex`` plus a table-driven ``resolve_gene_name``."""

    def __init__(self, resolutions: dict[str, tuple[GeneNameStatus, str | None]]):
        self._resolutions = resolutions
        self.gene_set = {
            n for n, (st, _) in resolutions.items() if st is GeneNameStatus.CURRENT
        }
        self.alias_to_systematic: dict[str, list[str]] = {
            n: [orf]
            for n, (st, orf) in resolutions.items()
            if st is GeneNameStatus.RENAMED and orf is not None
        }
        self.feature_index: dict[str, Any] = {"standard_to_ids": {}}

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, orf = self._resolutions[name]
        return GeneNameResolution(input_name=name, status=status, systematic_name=orf)


def _rename(alias: str = "YAL043C-A", orf: str = "YAL042C-A") -> PinnedAliasResolution:
    return PinnedAliasResolution(
        alias=alias,
        systematic_name=orf,
        rule=AliasResolutionRule.SGD_SYSTEMATIC_RENAME,
        provenance=_PROV,
        quote=f"{alias} used to be the systematic name; changed to {orf}.",
    )


def _merged(name: str, locus: str) -> snc.PinnedSystematicNameRefusal:
    return snc.PinnedSystematicNameRefusal(
        name=name,
        reason=GeneNameRefusalReason.MERGED_INTO_LOCUS,
        locus=locus,
        provenance=_PROV,
        quote=f"{name} was merged into {locus}.",
        basis="merged",
    )


def _retired(name: str = "YAR037W") -> snc.PinnedSystematicNameRefusal:
    return snc.PinnedSystematicNameRefusal(
        name=name,
        reason=GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
        locus=None,
        provenance=_PROV,
        quote=None,
        basis="absent",
    )


_RENAMES = pin_table([_rename()])
_REFUSALS = snc.refusal_table(
    [_merged("YAR044W", "YAR042W"), _merged("YER108C", "YER109C"), _retired()]
)


def _currency(
    resolutions: dict[str, tuple[GeneNameStatus, str | None]] = _RESOLUTIONS,
) -> snc.SystematicNameCurrency:
    return snc.SystematicNameCurrency(_Genome(resolutions), _RENAMES, _REFUSALS)


def test_apply_maps_renames_refuses_merged_and_retired_rows_and_keeps_the_rest() -> (
    None
):
    """Six rows over two columns: rows 1, 3 and 4 carry a refused name (YAR044W,
    YER108C, YAR037W) and leave; row 2's YAL043C-A becomes YAL042C-A; the live
    non-gene locus YER109C stays itself; the index is kept.
    """
    frame = pd.DataFrame(
        {
            "q": ["YAL001C", "YAR044W", "YAL043C-A", "YER108C", "YAL001C", "YER109C"],
            "a": ["YAR042W", "YAL001C", "YAL001C", "YAL001C", "YAR037W", "YAL043C-A"],
            "value": [0, 1, 2, 3, 4, 5],
        }
    )
    currency = _currency()
    kept = currency.apply(frame, ["q", "a"])
    assert kept.to_dict(orient="list") == {
        "q": ["YAL001C", "YAL042C-A", "YER109C"],
        "a": ["YAR042W", "YAL001C", "YAL042C-A"],
        "value": [0, 2, 5],
    }
    assert kept.index.tolist() == [0, 2, 5]
    ledger = currency.ledger("toy")
    assert ledger.model_dump(mode="json") == {
        "dataset_name": "toy",
        "annotation": "S288C R64-4-1 (2023-08-30)",
        "n_rows_in": 6,
        "n_rows_refused": 3,
        "n_rows_out": 3,
        "renamed": [
            {
                "released_name": "YAL043C-A",
                "stored_name": "YAL042C-A",
                "reason": None,
                "locus": "YAL042C-A",
                "n_rows": 2,
            }
        ],
        "refused": [
            {
                "released_name": "YAR037W",
                "stored_name": None,
                "reason": "retired_systematic_name",
                "locus": None,
                "n_rows": 1,
            },
            {
                "released_name": "YAR044W",
                "stored_name": None,
                "reason": "merged_into_locus",
                "locus": "YAR042W",
                "n_rows": 1,
            },
            {
                "released_name": "YER108C",
                "stored_name": None,
                "reason": "merged_into_locus",
                "locus": "YER109C",
                "n_rows": 1,
            },
        ],
    }


def test_counts_add_up_across_calls_for_two_record_kinds() -> None:
    """The DMF Kuzmin 2020 shape: two frames, two column sets, one ledger."""
    currency = _currency()
    currency.apply(pd.DataFrame({"x": ["YAR044W", "YAL001C"]}), ["x"])
    currency.apply(pd.DataFrame({"y1": ["YAL043C-A"], "y2": ["YAR037W"]}), ["y1", "y2"])
    ledger = currency.ledger("two kinds")
    assert (ledger.n_rows_in, ledger.n_rows_refused, ledger.n_rows_out) == (3, 2, 1)
    assert [o.released_name for o in ledger.renamed] == ["YAL043C-A"]
    assert [(o.released_name, o.n_rows) for o in ledger.refused] == [
        ("YAR037W", 1),
        ("YAR044W", 1),
    ]


def test_a_non_current_name_in_neither_table_stops_the_build() -> None:
    with pytest.raises(GeneNameRefused) as info:
        _currency().apply(pd.DataFrame({"x": ["YAL001C", "YZZ999W"]}), ["x"])
    assert info.value.refusal.model_dump() == {
        "name": "YZZ999W",
        "reason": GeneNameRefusalReason.NOT_CURRENT_UNPINNED,
        "candidates": ["YAL001C"],
        "detail": "resolves as renamed with neither a rename pin nor a pinned refusal",
    }


def test_a_blank_name_cell_is_refused_not_skipped() -> None:
    frame = pd.DataFrame({"x": ["YAL001C", None]})
    with pytest.raises(ValueError, match=r"1 rows have a blank systematic name"):
        _currency().apply(frame, ["x"])


@pytest.mark.parametrize(
    ("name", "changed", "detail"),
    [
        (
            "YAR037W",
            (GeneNameStatus.CURRENT, "YAR037W"),
            "pinned retired_systematic_name (locus None), but the genome resolves it "
            "as current -> YAR037W",
        ),
        (
            "YAR044W",
            (GeneNameStatus.RENAMED, "YAL001C"),
            "pinned merged_into_locus (locus YAR042W), but the genome resolves it as "
            "renamed -> YAL001C",
        ),
    ],
    ids=["retired-now-live", "merged-into-another-locus"],
)
def test_a_refusal_the_genome_contradicts_stops_construction(
    name: str, changed: tuple[GeneNameStatus, str | None], detail: str
) -> None:
    with pytest.raises(GeneNameRefused) as info:
        _currency({**_RESOLUTIONS, name: changed})
    assert info.value.refusal.reason is GeneNameRefusalReason.PIN_RULE_CONTRADICTED
    assert info.value.refusal.detail == detail


def test_construction_refuses_a_contradicted_or_misruled_rename_and_an_overlap() -> (
    None
):
    genome = _Genome({**_RESOLUTIONS, "YAL043C-A": (GeneNameStatus.CURRENT, None)})
    with pytest.raises(GeneNameRefused) as info:
        snc.SystematicNameCurrency(genome, _RENAMES, {})
    assert info.value.refusal.reason is GeneNameRefusalReason.PIN_NOT_A_CANDIDATE
    standard = pin_table(
        [
            PinnedAliasResolution(
                alias="YAL043C-A",
                systematic_name="YAL042C-A",
                rule=AliasResolutionRule.SGD_STANDARD_NAME,
                provenance=_PROV,
                quote="YAL043C-A YAL042C-A",
            )
        ]
    )
    with pytest.raises(ValueError, match="must use sgd_systematic_rename"):
        snc.SystematicNameCurrency(_Genome(_RESOLUTIONS), standard, {})
    both = snc.refusal_table([_merged("YAL043C-A", "YAL042C-A")])
    with pytest.raises(ValueError, match=r"both renamed and refused: \['YAL043C-A'\]"):
        snc.SystematicNameCurrency(_Genome(_RESOLUTIONS), _RENAMES, both)


def test_refusal_validation_and_duplicates() -> None:
    with pytest.raises(ValidationError, match="must be upper case"):
        _retired("yar037w")
    with pytest.raises(ValidationError, match="is retired or merged, not ambiguous"):
        snc.PinnedSystematicNameRefusal(
            name="YAR037W",
            reason=GeneNameRefusalReason.AMBIGUOUS_ALIAS_UNPINNED,
            locus=None,
            provenance=_PROV,
            quote=None,
            basis="x",
        )
    with pytest.raises(ValidationError, match="provenance.sha256 is required"):
        snc.PinnedSystematicNameRefusal(
            name="YAR037W",
            reason=GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
            locus=None,
            provenance=Provenance(source_uri="gff"),
            quote=None,
            basis="x",
        )
    with pytest.raises(ValidationError, match="a retired name has no locus or quote"):
        snc.PinnedSystematicNameRefusal(
            name="YAR037W",
            reason=GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
            locus=None,
            provenance=_PROV,
            quote="YAR037W",
            basis="x",
        )
    with pytest.raises(ValidationError, match="needs its locus and a quote"):
        snc.PinnedSystematicNameRefusal(
            name="YAR044W",
            reason=GeneNameRefusalReason.MERGED_INTO_LOCUS,
            locus="YAR042W",
            provenance=_PROV,
            quote=None,
            basis="x",
        )
    with pytest.raises(ValidationError, match="must contain the released name"):
        snc.PinnedSystematicNameRefusal(
            name="YAR044W",
            reason=GeneNameRefusalReason.MERGED_INTO_LOCUS,
            locus="YAR042W",
            provenance=_PROV,
            quote="OSH1 was merged into SWH1/YAR042W",
            basis="x",
        )
    with pytest.raises(ValueError, match="name 'YAR037W' is refused twice"):
        snc.refusal_table([_retired(), _retired()])


def test_a_ledger_whose_rows_do_not_add_up_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"10 rows in - 3 refused != 6 out"):
        snc.SystematicNameLedger(
            dataset_name="bad",
            annotation="R64",
            n_rows_in=10,
            n_rows_refused=3,
            n_rows_out=6,
            renamed=[],
            refused=[],
        )


def test_the_sga_tables_hold_16_renames_28_merges_and_7_retirements() -> None:
    merged = [
        r
        for r in snc.SGA_SYSTEMATIC_REFUSALS.values()
        if r.reason is GeneNameRefusalReason.MERGED_INTO_LOCUS
    ]
    retired = sorted(
        r.name
        for r in snc.SGA_SYSTEMATIC_REFUSALS.values()
        if r.reason is GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME
    )
    assert len(snc.SGA_SYSTEMATIC_RENAMES) == 16
    assert len(merged) == 28
    assert retired == [
        "YAR037W",
        "YAR040C",
        "YAR043C",
        "YCL006C",
        "YCL013W",
        "YCL026C",
        "YIL080W",
    ]
    # the four merged into a non-gene locus quote the R64 GFF; the rest SGD's history
    assert sorted(r.name for r in merged if r.provenance == R64_GFF) == [
        "YER108C",
        "YFL057C",
        "YIL168W",
        "YIR044C",
    ]


_GENOMES_TIER_GFF = (
    "torchcell-genomes/sgd_S288C_R64-4-1_20230830/"
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff"
)


def _history_notes(raw: bytes) -> list[str]:
    """Each ``locus.history[].note`` with its bold category label removed."""
    history = json.loads(raw)["locus"]["history"]
    return [re.sub(r"^<b>[^<]*</b>\s*", "", entry["note"]) for entry in history]


def test_every_pin_matches_its_source_bytes() -> None:
    """Each pin and refusal hashes to its file and quotes it; skipped off this machine.

    SGD-history quotes are matched against the decoded ``note`` strings of the cached
    locus JSON (``" [...] "`` joins two of them); GFF quotes against the genomes-tier
    GFF, whose bytes are the ``R64_GFF`` sha256.
    """
    root = os.environ.get("DATA_ROOT")
    gff = None if root is None else Path(root) / _GENOMES_TIER_GFF
    if root is None or gff is None or not gff.exists():
        pytest.skip("DATA_ROOT with the R64-4-1 genomes tier is not on this machine")
    gff_bytes = gff.read_bytes()
    assert hashlib.sha256(gff_bytes).hexdigest() == R64_GFF.sha256
    gff_text = gff_bytes.decode("utf-8")
    pins: list[PinnedAliasResolution | snc.PinnedSystematicNameRefusal] = [
        *snc.SGA_SYSTEMATIC_RENAMES.values(),
        *snc.SGA_SYSTEMATIC_REFUSALS.values(),
    ]
    for pin in pins:
        if pin.provenance == R64_GFF:
            if pin.quote is not None:
                assert pin.quote in gff_text, pin.quote
            continue
        raw = (Path(root) / pin.provenance.source_uri).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin.provenance.sha256
        assert pin.quote is not None
        notes = _history_notes(raw)
        for segment in pin.quote.split(" [...] "):
            assert any(segment in note for note in notes), segment

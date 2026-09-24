"""Peter gene-FASTA header parsing and the no-silent-drop isolate accounting.

Tripwire for issue 73. Peter 2018 writes two header-token forms and BOTH name the
isolate verbatim: ``AEE_YAL001C_TFC3`` and ``SACE_YAU_YAL001C_TFC3``. ``SACE_YAU`` is
the isolate's own code, not a species prefix on ``YAU`` -- it indexes
``genesMatrix_PresenceAbsence.tab.gz`` (93 of 1011) and Caudal's ``Strain`` column (78
of the 943 built isolates) in exactly that form. Stripping the prefix would leave those
78 isolates unmatchable and drop 469,170 of 5,672,145 gene records without a word, so
these tests pin the verbatim behavior against a well-meant "fix".

The second test pins the loud half: an isolate that never appears in a header must raise,
never be assembled with an empty genotype.
"""

import pytest

from torchcell.datasets.scerevisiae.caudal2024 import (
    _assert_all_isolates_seen,
    _isolate_and_symbol,
)


def test_plain_header_token_yields_bare_isolate_code() -> None:
    """``AEE_YAL001C_TFC3`` -> isolate ``AEE``, symbol ``TFC3``."""
    assert _isolate_and_symbol("AEE_YAL001C_TFC3", "YAL001C") == ("AEE", "TFC3")


def test_sace_header_token_keeps_the_prefix_in_the_isolate_code() -> None:
    """``SACE_YAU_YAL001C_TFC3`` -> isolate ``SACE_YAU`` (prefix KEPT), symbol ``TFC3``."""
    assert _isolate_and_symbol("SACE_YAU_YAL001C_TFC3", "YAL001C") == (
        "SACE_YAU",
        "TFC3",
    )


def test_both_forms_match_the_isolate_codes_peter_and_caudal_use() -> None:
    """Both parsed codes are the codes the strain sets carry, so neither is dropped."""
    matched = {"AEE", "SACE_YAU"}
    seen = {
        _isolate_and_symbol(token, "YAL001C")[0]
        for token in ("AEE_YAL001C_TFC3", "SACE_YAU_YAL001C_TFC3")
    }
    assert seen == matched
    _assert_all_isolates_seen(matched, seen)


def test_token_without_the_gene_key_raises() -> None:
    """A token carrying no ``_<systematic>_`` key is unparseable, not skippable."""
    with pytest.raises(ValueError, match="carries no"):
        _isolate_and_symbol("AEE_YAL002W_VPS8", "YAL001C")


def test_token_with_an_empty_isolate_raises() -> None:
    """A token that starts with the gene key names no isolate."""
    with pytest.raises(ValueError, match="empty isolate"):
        _isolate_and_symbol("_YAL001C_TFC3", "YAL001C")


def test_unseen_isolate_raises_with_count_and_codes() -> None:
    """A built isolate absent from every header must fail loudly, with its code."""
    matched = {"AEE", "SACE_YAU", "BCS"}
    seen = {"AEE"}
    with pytest.raises(ValueError) as excinfo:
        _assert_all_isolates_seen(matched, seen)
    message = str(excinfo.value)
    assert "2 of 3 matched isolates" in message
    assert "SACE_YAU" in message
    assert "BCS" in message


def test_the_sace_strip_fix_would_trip_the_accounting() -> None:
    """The issue-73 prefix strip is what the accounting now catches."""
    matched = {"AEE", "SACE_YAU"}
    stripped = set()
    for token in ("AEE_YAL001C_TFC3", "SACE_YAU_YAL001C_TFC3"):
        token = token[5:] if token.startswith("SACE_") else token
        stripped.add(_isolate_and_symbol(token, "YAL001C")[0])
    assert stripped == {"AEE", "YAU"}
    with pytest.raises(ValueError, match="SACE_YAU"):
        _assert_all_isolates_seen(matched, stripped & matched)

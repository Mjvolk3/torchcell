# tests/torchcell/datasets/test_gene_alias_resolution.py
# [[tests.torchcell.datasets.test_gene_alias_resolution]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_gene_alias_resolution.py
"""Refusing gene-name resolution (#886) on a three-lookup genome stub.

Stub genome: live genes YOR110W, YNL039W, YAL001C, YBR001C, YCR001W. ``TFC7`` is the
standard name of YOR110W and an alias of YNL039W, listed first (the R64 shape behind
#886). ``ONLY1`` is a unique alias of YAL001C. ``AMB`` is an alias of YAL001C and
YBR001C with no standard-name owner. ``STDONLY`` is the standard name of YCR001W and an
alias of YBR001C, but YCR001W does not repeat it in its own Alias list, so the
candidate set must union both maps to see the ambiguity.

The last test re-reads the source of every pin the loaders still declare (the R64 GFF)
when it is on this machine, and checks the bytes against the pinned sha256 and the quote
against the bytes.

Since 2026-10-12 the module is a downstream utility (owner: "the data is the data!!!!"):
``test_no_loader_outside_the_sweep_list_imports_the_resolution_modules`` pins which
loaders still import it at load time, so a new loader cannot start gating on it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datasets.gene_alias_resolution import (
    AliasResolutionRule,
    GeneNameRefusalReason,
    GeneNameRefused,
    PinnedAliasResolution,
    candidate_orfs,
    check_ambiguous_aliases,
    pin_table,
    resolve_gene_name_strict,
)
from torchcell.verification.report import Provenance

_PROV = Provenance(source_uri="gff", sha256="0" * 64)


class _Genome:
    gene_set = {"YOR110W", "YNL039W", "YAL001C", "YBR001C", "YCR001W"}
    alias_to_systematic: dict[str, list[str]] = {
        "TFC7": ["YNL039W", "YOR110W"],
        "ONLY1": ["YAL001C"],
        "AMB": ["YAL001C", "YBR001C"],
        "STDONLY": ["YBR001C"],
    }
    feature_index: dict[str, Any] = {
        "standard_to_ids": {"TFC7": ["YOR110W"], "STDONLY": ["YCR001W"]}
    }


def _pin(
    alias: str = "TFC7",
    orf: str = "YOR110W",
    rule: AliasResolutionRule = AliasResolutionRule.SGD_STANDARD_NAME,
) -> PinnedAliasResolution:
    return PinnedAliasResolution(
        alias=alias,
        systematic_name=orf,
        rule=rule,
        provenance=_PROV,
        quote=f"ID={orf};gene={alias};",
    )


def test_candidates_union_the_alias_table_and_standard_name_owners() -> None:
    g = _Genome()
    assert candidate_orfs(g, " tfc7 ") == ["YNL039W", "YOR110W"]
    assert candidate_orfs(g, "STDONLY") == ["YBR001C", "YCR001W"]
    assert candidate_orfs(g, "ONLY1") == ["YAL001C"]
    assert candidate_orfs(g, "NOPE") == []


def test_unique_and_live_names_resolve_without_a_pin() -> None:
    g = _Genome()
    assert resolve_gene_name_strict(g, " yal001c ", {}) == "YAL001C"
    assert resolve_gene_name_strict(g, "only1", {}) == "YAL001C"


def test_an_ambiguous_name_resolves_only_through_its_pin() -> None:
    """TFC7's first alias-table candidate is YNL039W; the pin gives YOR110W."""
    g = _Genome()
    pins = pin_table([_pin()])
    assert resolve_gene_name_strict(g, "TFC7", pins) == "YOR110W"
    with pytest.raises(GeneNameRefused) as info:
        resolve_gene_name_strict(g, "TFC7", {})
    assert info.value.refusal.model_dump() == {
        "name": "TFC7",
        "reason": GeneNameRefusalReason.AMBIGUOUS_ALIAS_UNPINNED,
        "candidates": ["YNL039W", "YOR110W"],
        "detail": "ambiguous name with no pinned resolution in the loader",
    }
    assert str(info.value) == (
        "gene name 'TFC7' refused (ambiguous_alias_unpinned): ambiguous name with no "
        "pinned resolution in the loader; candidates ['YNL039W', 'YOR110W']"
    )


def test_unknown_name_is_refused_as_not_in_genome() -> None:
    with pytest.raises(GeneNameRefused) as info:
        resolve_gene_name_strict(_Genome(), "nope", {})
    assert info.value.refusal.reason is GeneNameRefusalReason.NOT_IN_GENOME
    assert info.value.refusal.name == "NOPE"
    assert info.value.refusal.candidates == []


def test_a_pin_off_the_candidates_or_against_its_rule_is_refused() -> None:
    g = _Genome()
    off = pin_table([_pin(orf="YCR001W")])
    with pytest.raises(GeneNameRefused) as info:
        resolve_gene_name_strict(g, "TFC7", off)
    assert info.value.refusal.reason is GeneNameRefusalReason.PIN_NOT_A_CANDIDATE
    # YNL039W is a candidate, but the GFF names YOR110W as TFC7's standard-name owner.
    wrong_owner = pin_table([_pin(orf="YNL039W")])
    with pytest.raises(GeneNameRefused) as info:
        resolve_gene_name_strict(g, "TFC7", wrong_owner)
    assert info.value.refusal.reason is GeneNameRefusalReason.PIN_RULE_CONTRADICTED
    assert info.value.refusal.detail == (
        "rule sgd_standard_name: the genome's standard-name owners are ['YOR110W'], "
        "not [YNL039W]"
    )
    # AMB has no standard-name owner, so only the paper's own list can decide it.
    std_amb = pin_table([_pin(alias="AMB", orf="YBR001C")])
    with pytest.raises(GeneNameRefused):
        resolve_gene_name_strict(g, "AMB", std_amb)


def test_a_paper_gene_list_pin_needs_the_list_to_agree() -> None:
    g = _Genome()
    pins = pin_table(
        [_pin(alias="AMB", orf="YBR001C", rule=AliasResolutionRule.PAPER_GENE_LIST)]
    )
    assert resolve_gene_name_strict(g, "AMB", pins, {"AMB": "YBR001C"}) == "YBR001C"
    for paper in (None, {}, {"AMB": "YAL001C"}):
        with pytest.raises(GeneNameRefused) as info:
            resolve_gene_name_strict(g, "AMB", pins, paper)
        assert info.value.refusal.reason is (
            GeneNameRefusalReason.PIN_RULE_CONTRADICTED
        )


def test_pin_validation_and_duplicate_aliases() -> None:
    with pytest.raises(ValidationError, match="must be upper case"):
        _pin(alias="tfc7")
    with pytest.raises(
        ValidationError, match="quote must contain the alias and the ORF"
    ):
        PinnedAliasResolution(
            alias="TFC7",
            systematic_name="YOR110W",
            rule=AliasResolutionRule.SGD_STANDARD_NAME,
            provenance=_PROV,
            quote="ID=YNL039W;gene=TFC7;",
        )
    with pytest.raises(ValidationError, match="provenance.sha256 is required"):
        PinnedAliasResolution(
            alias="TFC7",
            systematic_name="YOR110W",
            rule=AliasResolutionRule.SGD_STANDARD_NAME,
            provenance=Provenance(source_uri="gff"),
            quote="ID=YOR110W;gene=TFC7;",
        )
    with pytest.raises(ValueError, match="alias 'TFC7' is pinned twice"):
        pin_table([_pin(), _pin()])


def test_check_passes_pinned_pairs_and_skips_non_resolutions() -> None:
    """Live ids and unique names are not counted as ambiguous; TFC7 -> YOR110W is."""
    audit = check_ambiguous_aliases(
        _Genome(),
        [
            ("YAL001C", "YAL001C"),
            ("ONLY1", "YAL001C"),
            ("TFC7", "YOR110W"),
            ("ybr001c", "YBR001C"),
        ],
        pin_table([_pin()]),
    )
    assert audit.model_dump() == {"n_pairs": 4, "ambiguous": {"TFC7": "YOR110W"}}


def test_check_refuses_the_first_match_store_of_issue_886() -> None:
    """The pre-fix Xue store: TFC7 stored as YNL039W. Without a pin the check refuses
    it as unpinned; with the pin it refuses the stored ORF as differing from the pin.
    """
    g = _Genome()
    with pytest.raises(GeneNameRefused) as info:
        check_ambiguous_aliases(g, [("TFC7", "YNL039W")], {})
    assert info.value.refusal.reason is GeneNameRefusalReason.AMBIGUOUS_ALIAS_UNPINNED
    assert info.value.refusal.detail == "stored as YNL039W with no pinned resolution"
    with pytest.raises(GeneNameRefused) as info:
        check_ambiguous_aliases(g, [("TFC7", "YNL039W")], pin_table([_pin()]))
    assert info.value.refusal.model_dump() == {
        "name": "TFC7",
        "reason": GeneNameRefusalReason.STORED_ORF_DIFFERS_FROM_PIN,
        "candidates": ["YNL039W", "YOR110W"],
        "detail": "stored as YNL039W, pinned to YOR110W",
    }


def _data_root() -> Path | None:
    root = os.environ.get("DATA_ROOT")
    return Path(root) if root else None


def test_every_loader_pin_matches_its_source_bytes() -> None:
    """Each pin's quote is in the bytes it cites, and those bytes hash to the pin.

    The pins left in loaders (Xue 2025, Sameith 2015) are GFF pins and quote a substring
    of the R64-4-1 GFF. da Silveira 2014 declares none since 2026-10-12: it stores the
    names its own table gives and resolves nothing. Skipped where the GFF is not on this
    machine.
    """
    from torchcell.datasets.scerevisiae import sameith2015, xue2025

    root = _data_root()
    gff = None if root is None else root / R64_PATH
    if gff is None or not gff.exists():
        pytest.skip("R64 GFF not on this machine")
    gff_bytes = gff.read_bytes()
    gff_text = gff_bytes.decode("utf-8")
    for module in (xue2025, sameith2015):
        for pin in module.AMBIGUOUS_ALIAS_PINS.values():
            assert pin.rule is AliasResolutionRule.SGD_STANDARD_NAME, pin.alias
            assert pin.provenance.sha256 == hashlib.sha256(gff_bytes).hexdigest()
            assert pin.quote in gff_text, pin.alias


R64_PATH = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff"
)


#: Modules under ``torchcell/datasets`` that still import a resolution module at load
#: time on this branch (2026-10-12), each left to the branch rewriting it: Costanzo
#: 2016, Kuzmin 2020 and Yoshikawa 2009 to ``fix/audit-yeast-refusals-1045`` (which
#: deletes ``systematic_name_currency``), Xue 2025 to ``fix/keep-records-887-sd-zero``,
#: Sameith 2015 to the loader sweep. The set only shrinks.
LOAD_TIME_RESOLUTION_PENDING_SWEEP = frozenset(
    {
        "torchcell.datasets.scerevisiae.costanzo2016",
        "torchcell.datasets.scerevisiae.kuzmin2020",
        "torchcell.datasets.scerevisiae.sameith2015",
        "torchcell.datasets.scerevisiae.xue2025",
        "torchcell.datasets.scerevisiae.yoshikawa2009",
        "torchcell.datasets.systematic_name_currency",
    }
)
RESOLUTION_MODULES = frozenset(
    {
        "torchcell.datasets.gene_alias_resolution",
        "torchcell.datasets.systematic_name_currency",
    }
)


def test_no_loader_outside_the_sweep_list_imports_the_resolution_modules() -> None:
    """Owner, 2026-10-12: name resolution is downstream and gates no record at load time.

    Every module under ``torchcell/datasets`` that imports ``gene_alias_resolution`` or
    ``systematic_name_currency`` must be one the sweep has yet to rewrite; a loader
    added or changed to gate on them fails here. da Silveira 2014 left the list on
    2026-10-12 (it stores its table's own names).
    """
    import ast

    import torchcell.datasets as datasets_package

    root = Path(datasets_package.__file__).parent
    importers = set()
    for path in sorted(root.rglob("*.py")):
        module = "torchcell.datasets." + ".".join(
            path.relative_to(root).with_suffix("").parts
        )
        if module in RESOLUTION_MODULES - {
            "torchcell.datasets.systematic_name_currency"
        }:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module in RESOLUTION_MODULES:
                importers.add(module)
    assert importers - LOAD_TIME_RESOLUTION_PENDING_SWEEP == set()
    assert "torchcell.datasets.scerevisiae.dasilveira2014" not in importers

# tests/torchcell/datasets/ecoli/test_typas2008.py
# [[tests.torchcell.datasets.ecoli.test_typas2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_typas2008.py
"""The Typas 2008 provenance record: the decision, and the measurement behind it.

The hermetic tests re-run the one claim that is a claim about THIS repository rather
than about the release: that ``GeneInteractionPhenotype`` refuses a released term and
accepts a float, which is what makes "the release carries no score" a blocker. They also
assert that the module registers no dataset, that the table census accounts for every
released pair, and that the Babu 2014 overlap is a pair-level measurement.

The ``@pytest.mark.data`` tests read the real mirror: they audit every quote against the
pinned ``sha256`` and re-derive the deposit reconciliation's own file digest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import torchcell.datasets.ecoli.typas2008 as typas
from torchcell.data import file_sha256
from torchcell.datamodels.schema import GeneInteractionPhenotype
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import RetrievalMethod
from torchcell.verification.sourced import audit_sourced_value

DATA_ROOT = os.environ.get("DATA_ROOT")


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_the_module_registers_no_dataset() -> None:
    assert typas.SETTLED.loaded_records == 0
    assert typas.SETTLED.status == "blocked"
    assert typas.SETTLED.row == 37
    assert not [
        name
        for name, cls in dataset_registry.items()
        if cls.__module__ == typas.__name__
    ]


def test_the_accession_half_of_the_row_is_settled_by_a_complete_deposit() -> None:
    deposit = typas.DEPOSIT
    assert deposit.n_supplementary_objects == deposit.n_declared_supplementary == 1
    assert deposit.published_not_mirrored == ()
    assert deposit.mirrored_not_published == ()
    assert deposit.complete is True
    assert typas.SETTLED.accession_confirmed is True
    # the one deposited object is the one this record pins, by the scriptable route
    assert typas.SI_PDF.retrieval_method is RetrievalMethod.pmc_cloud
    assert typas.PMC_ID in typas.SI_PDF.source_url
    # the OCR is DERIVED, so it names no retrieval method and carries its recipe instead
    assert typas.SI_MARKDOWN.retrieval_method is None
    assert typas.SI_MARKDOWN.processing_command is not None
    assert "mineru" in typas.SI_MARKDOWN.processing_command


# --------------------------------------------------------------------------- #
# The release carries no score
# --------------------------------------------------------------------------- #
def test_the_table_census_accounts_for_every_released_pair() -> None:
    pairs = typas.RELEASED_PAIRS
    assert (
        pairs.table_2a_pal_interactions
        + pairs.table_2b_yrap_suppressors
        + pairs.table_1_cotransduction_pairs
        == pairs.total
        == 42
    )
    assert pairs.with_a_numeric_value == pairs.table_1_cotransduction_pairs == 4
    assert "Co-inheritance" in pairs.numeric_statistic
    # exactly one of the ten tables carries a number at all
    numeric = [t for t in typas.SI_TABLE_CENSUS if t.n_numeric_cells]
    assert [t.table_index for t in numeric] == [4]
    assert numeric[0].n_numeric_cells == 8
    assert len(typas.SI_TABLE_CENSUS) == 10


def test_the_twelve_by_twelve_cross_is_a_figure_whose_axis_genes_are_recoverable() -> (
    None
):
    genes = typas.CROSS_GENES
    assert len(genes) == len(set(genes)) == 12
    assert typas.CROSS_DISTINCT_DOUBLES == len(genes) * (len(genes) - 1) // 2 == 66
    assert typas.CROSS_CAPTION.note is not None
    assert "read off a colour" in typas.CROSS_CAPTION.note
    # the four panel tables of the census hold labels and no values, which is the claim
    panels = [t for t in typas.SI_TABLE_CENSUS if t.table_index in (0, 1, 2, 3)]
    assert len(panels) == 4
    assert {t.n_numeric_cells for t in panels} == {0}


# --------------------------------------------------------------------------- #
# The schema probe, re-run against the live leaf
# --------------------------------------------------------------------------- #
def _leaf(value: object) -> GeneInteractionPhenotype:
    return GeneInteractionPhenotype(gene_interaction=value)  # type: ignore[arg-type]


def test_the_interaction_leaf_refuses_a_released_term_and_accepts_a_float() -> None:
    """The recorded probe is re-run, so a schema change that adds a categorical mode
    makes this test fail rather than leaving a stale refusal on the record.
    """
    with pytest.raises(ValidationError) as term:
        _leaf("neg (lethal)")
    assert [e["type"] for e in term.value.errors()] == ["float_parsing"]
    with pytest.raises(ValidationError) as missing:
        GeneInteractionPhenotype()  # type: ignore[call-arg]
    assert [e["type"] for e in missing.value.errors()] == ["missing"]
    assert _leaf(-0.21).gene_interaction == -0.21

    probe = typas.SCHEMA_PROBE
    assert probe.float_accepted is True
    assert probe.has_categorical_mode is False
    assert [a.error_type for a in probe.attempts] == ["float_parsing", "missing"]
    assert [a.accepted for a in probe.attempts] == [False, False]


# --------------------------------------------------------------------------- #
# Not subsumed
# --------------------------------------------------------------------------- #
def test_the_babu_overlap_is_two_verification_pairs_and_no_screen_pair() -> None:
    subsumption = typas.BABU_SUBSUMPTION
    assert subsumption.subsumed is False
    assert subsumption.our_pairs_in_theirs == len(typas.OVERLAPPING_PAIRS) == 2
    assert set(subsumption.query_genes_among_their_donors.values()) == {False}
    assert sorted(subsumption.query_genes_among_their_donors) == ["pal", "yraP"]
    # the 38 screen pairs are the 42 minus the 4 co-transduction verification pairs
    pairs = typas.RELEASED_PAIRS
    screen_pairs = pairs.total - pairs.table_1_cotransduction_pairs
    assert screen_pairs == 38
    # and both overlapping pairs name a query gene, so neither is a screen-only pair
    assert {gene for pair in typas.OVERLAPPING_PAIRS for gene in pair} >= {"pal"}


def test_the_partners_babu_does_hold_are_recorded_so_the_claim_stays_pair_level() -> (
    None
):
    partners = typas.BABU_PARTNERS_OF_QUERY_GENES
    assert sorted(partners) == ["pal", "yraP"]
    assert len(partners["pal"]) == 10
    assert len(partners["yraP"]) == 2
    assert "ompA" in partners["pal"]
    # every recorded partner list is sorted and free of duplicates
    for genes in partners.values():
        assert list(genes) == sorted(set(genes))


def test_the_row_states_what_would_reopen_it() -> None:
    assert "Supplementary " in typas.SETTLED.reopens_on
    assert "author release" in typas.SETTLED.reopens_on


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")
def test_every_quote_audits_against_the_pinned_supplementary_ocr() -> None:
    library = Path(str(DATA_ROOT)) / "torchcell-library"
    for name, value in typas.SOURCED_VALUES.items():
        result = audit_sourced_value(value, library)
        assert result.passed, f"{name}: {result.message}"


@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")
def test_both_mirrored_artifacts_match_their_recorded_digest_and_size() -> None:
    base = Path(str(DATA_ROOT)) / typas.LIBRARY_DIR_REL
    for artifact in (typas.SI_PDF, typas.SI_MARKDOWN):
        path = base / artifact.relpath
        if not path.exists():
            pytest.skip(f"{path} is not in the literature mirror")
        assert file_sha256(path) == artifact.sha256
        assert path.stat().st_size == artifact.n_bytes

# tests/torchcell/candidates/test_ledger.py
# [[tests.torchcell.candidates.test_ledger]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_ledger.py
"""The ledger renders byte-exact and appends one dated section per day."""

from pathlib import Path

import pytest

from tests.torchcell.candidates._builders import (
    COMMIT,
    admissible_verdict,
    compendium_paper,
    cross_check,
    evidence,
    gate,
    source,
    unmeasured_tail,
    verdict,
)
from torchcell.candidates import ledger
from torchcell.candidates.verdict import AggregationRecord, CandidateVerdict


def test_heading() -> None:
    """ISO date -> dendron date heading; anything else raises."""
    assert ledger.heading("2026-10-10") == "## 2026.10.10 - Candidate gate ledger"
    with pytest.raises(ValueError, match="not an ISO date"):
        ledger.heading("2026.10.10")


def test_render_row_escapes_and_lists_issues() -> None:
    """Glyphs in gate order, bold outcome, issues, date and a 9-character commit."""
    v = admissible_verdict(gap_issue=854, row_name="A | B")
    assert ledger.render_row(v) == (
        "| keyPaper2020 | bacteria | A \\| B | ✓ | ✓ | ✓ | ✓ | ◐ | "
        f"**admissible_with_gaps** | #854 | 2026-10-10 | `{COMMIT[:9]}` |"
    )


def test_render_section_is_sorted_and_exact() -> None:
    """Heading, legend, header, rows by (table, row)."""
    yeast = verdict(citation_key="y", table="yeast", row_name="Alpha")
    bacteria = verdict(citation_key="b", row_name="Zeta")
    section = ledger.render_section([yeast, bacteria], "2026-10-10")
    assert section == (
        "## 2026.10.10 - Candidate gate ledger\n\n"
        f"{ledger.LEGEND}\n\n"
        f"{ledger.HEADER}\n"
        f"| b | bacteria | Zeta | ✗ | · | · | · | · | **refused** |  | 2026-10-10 | `{COMMIT[:9]}` |\n"
        f"| y | yeast | Alpha | ✗ | · | · | · | · | **refused** |  | 2026-10-10 | `{COMMIT[:9]}` |\n"
    )


@pytest.mark.parametrize(
    ("existing", "joined"),
    [
        ("intro\n\n", "intro\n\n## S\n"),
        ("intro\n", "intro\n\n## S\n"),
        ("intro", "intro\n\n## S\n"),
    ],
)
def test_append_section_separates_with_one_blank_line(
    tmp_path: Path, existing: str, joined: str
) -> None:
    """Exactly one blank line between the note and the new section."""
    note = tmp_path / "ledger.md"
    note.write_text(existing)
    ledger.append_section(note, "## S\n")
    assert note.read_text() == joined


def test_append_section_refuses_a_second_section_the_same_day(tmp_path: Path) -> None:
    """One ledger per day."""
    note = tmp_path / "ledger.md"
    note.write_text("x\n\n## S\n")
    with pytest.raises(ValueError, match="already has '## S'"):
        ledger.append_section(note, "## S\nbody\n")


def _decomposed_verdict() -> CandidateVerdict:
    """An aggregation pending at G2 with three decomposed studies."""
    sources = (
        source(cross_check=cross_check()),
        source(
            "doi:10.1/w", own_release_access="walled", citation_key=None, mirrored=False
        ),
        compendium_paper(n_samples_in_compendium=2, n_samples_served=0),
    )
    record = AggregationRecord(
        n_source_studies=3,
        attribution_field="Publication",
        value_origin="re_measured",
        n_sources_mirrored=2,
        net_new_vs_served=None,
        evidence=(evidence(),),
        sources=sources,
    )
    return verdict(
        citation_key="c",
        row_name="Compendium",
        gates=(gate("G1"), *unmeasured_tail(("G2", "G3", "G4", "G5"))),
        source_kind="aggregation",
        aggregation=record,
    )


def test_render_section_appends_the_decomposition_exactly() -> None:
    """A decomposed aggregation adds the summary and per-source tables after the ledger."""
    section = ledger.render_section([_decomposed_verdict()], "2026-10-10")
    head, decomposition = section.split("\n\n### Aggregation decomposition\n\n")
    assert head.endswith(f"| **pending** |  | 2026-10-10 | `{COMMIT[:9]}` |")
    assert decomposition == (
        f"{ledger.DECOMPOSITION_LEGEND}\n\n"
        f"{ledger.DECOMPOSITION_SUMMARY_HEADER}\n"
        "| c | 3 | 1 | 1 | 1 | 0 | 10 (6) |\n\n"
        f"{ledger.DECOMPOSITION_HEADER}\n"
        "| c | FixtureDataset | doi:10.1/a | original_study | 4 (3) | per_record_values | "
        "scriptable | yes | no | max_abs_diff 0.0004 (pass) |\n"
        "| c | FixtureDataset | doi:10.1/w | original_study | 4 (3) | per_record_values | "
        "walled | no | no | · |\n"
        "| c | FixtureDataset | doi:10.1/c | compendium_paper | 2 (0) | none | · | yes | "
        "no | · |\n"
    )


def test_render_source_row_names_a_loaded_class_and_an_unbounded_pin() -> None:
    """A loaded study shows its class; a check with no tolerance reads as a pin."""
    row = ledger.render_source_row(
        "c",
        source(
            loaded_as_own_dataset=True,
            own_dataset_class="OwnDataset",
            cross_check=cross_check(0.98, None),
        ),
    )
    assert row.endswith("| yes | OwnDataset | max_abs_diff 0.98 (pin) |")
    failing = ledger.render_source_row("c", source(cross_check=cross_check(0.01)))
    assert failing.endswith("| max_abs_diff 0.01 (FAIL) |")

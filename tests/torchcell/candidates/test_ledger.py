# tests/torchcell/candidates/test_ledger.py
# [[tests.torchcell.candidates.test_ledger]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_ledger.py
"""The ledger renders byte-exact and appends one dated section per day."""

from pathlib import Path

import pytest

from tests.torchcell.candidates._builders import COMMIT, admissible_verdict, verdict
from torchcell.candidates import ledger


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

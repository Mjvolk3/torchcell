"""
scripts/git_merge_weekly_note.py
[[scripts.git_merge_weekly_note]]
https://github.com/Mjvolk3/torchcell/tree/main/scripts/git_merge_weekly_note.py

Custom git merge driver for weekly task notes.

The weekly notes are deliberately additive (see ``.gitattributes``): task
bullets accumulate, so when two branches both append under the same
``## YYYY.MM.DD`` we want both kept, not a conflict. The built-in
``merge=union`` driver does that for the body, but it is line-based and
indiscriminate -- it also unions the YAML *frontmatter*. When two branches
carry divergent frontmatter (e.g. each worktree minted a fresh ``id`` /
``created`` by running ``dendron-cli note write`` for a brand-new ISO-week
note), union concatenates both blocks into one ``---`` fence. The stacked,
duplicate-key result is invalid YAML and silently breaks every Dendron parse.

This driver fires at the exact moment of damage (merge / rebase) and splits
the work:

* the **body** (everything after the frontmatter fence) is unioned via
  ``git merge-file --union`` -- identical additive behaviour to before;
* the **frontmatter** is reconciled to a single block by keeping OURS
  (first-occurrence wins). For weekly notes the frontmatter is stable
  bookkeeping, so ours-wins can never produce invalid YAML and is sufficient.

Self-contained: no project-internal imports. Git invokes it as
``python scripts/git_merge_weekly_note.py %O %A %B %P`` -- %O ancestor, %A
ours (git reads the result back from this file), %B theirs, %P pathname.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Return ``(inner_frontmatter, body)``.

    ``inner_frontmatter`` is the content between the leading ``---`` fences
    (without the fences), or ``None`` if the file has no frontmatter. ``body``
    is everything after the closing fence (or the whole text when there is no
    frontmatter).
    """
    if not text.startswith("---"):
        return None, text
    # Split on the fences: ["", inner, body...]
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None, text
    return parts[1], parts[2]


def _union_bodies(ours: str, base: str, theirs: str) -> str:
    """Union three body texts with ``git merge-file -p --union``."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for name, content in (("ours", ours), ("base", base), ("theirs", theirs)):
            p = Path(tmp) / name
            p.write_text(content, encoding="utf-8")
            paths[name] = str(p)
        result = subprocess.run(
            ["git", "merge-file", "-p", "--union",
             paths["ours"], paths["base"], paths["theirs"]],
            capture_output=True, text=True,
        )
    return result.stdout


def merge(o_path: str, a_path: str, b_path: str) -> int:
    """Merge weekly note %B (theirs) into %A (ours) using %O (ancestor)."""
    text_o = Path(o_path).read_text(encoding="utf-8")
    text_a = Path(a_path).read_text(encoding="utf-8")
    text_b = Path(b_path).read_text(encoding="utf-8")

    a_inner, a_body = split_frontmatter(text_a)
    _, b_body = split_frontmatter(text_b)
    _, o_body = split_frontmatter(text_o)

    if a_inner is None:
        # No frontmatter to protect -- behave exactly like merge=union.
        Path(a_path).write_text(_union_bodies(text_a, text_o, text_b), encoding="utf-8")
        return 0

    merged_body = _union_bodies(a_body, o_body, b_body)
    # Ours-frontmatter-wins: a single, valid block; never stacks.
    Path(a_path).write_text("---\n" + a_inner + "---\n" + merged_body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    # argv: %O (ancestor) %A (ours/result) %B (theirs) [%P (pathname)]
    sys.exit(merge(sys.argv[1], sys.argv[2], sys.argv[3]))

# notes-tex/common/figure_cards.py
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/figure_cards.py
r"""One single-page PDF per figure of a built notes-tex document, caption included.

The review copy of a figure is its page of the built document, so the caption reads
exactly as it is typeset, numbers and all, and a figure is never reviewed apart from
what it claims (owner, 2026-10-18: the default when comparing figures). The page is cut
out of ``<doc>.pdf`` unchanged, which is why a figure the document places on a page of
its own gives a clean card.

How a figure is found. LaTeX numbers figures in source order, so the n-th figure label
in the document's source (``\\label{fig:...}``, or the label argument of ``\\tcfig`` and
``\\tcfigfit``) is Figure n, and the page whose text begins a line with ``Figure n:`` is
where its caption is set. The card is named for the label (``fig:main`` ->
``review/fig-main.pdf``), so it keeps its name when figures are inserted before it.

Nothing is guessed: a label count that differs from the captions found in the PDF, or
a figure whose caption page cannot be found, stops the run with the mismatch named.

    python notes-tex/common/figure_cards.py <doc-dir> [<doc-name>]

``make cards`` runs it after every build (``make all`` depends on it).
"""

from __future__ import annotations

import os
import os.path as osp
import re
import subprocess
import sys

LABEL = re.compile(r"(?:\\label\{|\}\{)(fig:[A-Za-z0-9_:-]+)\}")
CAPTION = re.compile(r"^\s*Figure (\d+):", re.MULTILINE)


def source_text(doc_dir: str, doc: str) -> str:
    r"""The document body with every ``\input`` expanded in order, comments removed."""

    def expand(path: str) -> str:
        with open(path) as f:
            raw = f.read()
        lines = [re.sub(r"(?<!\\)%.*", "", line) for line in raw.splitlines()]
        text = "\n".join(lines)
        return re.sub(
            r"\\input\{([^}]+)\}",
            lambda m: expand(
                osp.join(doc_dir, m.group(1).removesuffix(".tex") + ".tex")
            ),
            text,
        )

    body = expand(osp.join(doc_dir, f"{doc}.tex"))
    return body.split("\\begin{document}", 1)[1]


def caption_pages(pdf: str) -> dict[int, int]:
    """Figure number -> the page its caption starts on."""
    pages = int(
        re.search(
            r"Pages:\s+(\d+)",
            subprocess.run(
                ["pdfinfo", pdf], capture_output=True, text=True, check=True
            ).stdout,
        ).group(1)
    )
    found: dict[int, int] = {}
    for page in range(1, pages + 1):
        text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for number in CAPTION.findall(text):
            found.setdefault(int(number), page)
    return found


def main() -> int:
    """Cut one card per figure of the document named on the command line."""
    doc_dir = osp.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    doc = sys.argv[2] if len(sys.argv) > 2 else osp.basename(doc_dir)
    pdf = osp.join(doc_dir, f"{doc}.pdf")
    labels = LABEL.findall(source_text(doc_dir, doc))
    pages = caption_pages(pdf)
    if len(labels) != len(pages) or sorted(pages) != list(range(1, len(labels) + 1)):
        print(
            f"figure_cards: {len(labels)} figure labels in the source but captions for "
            f"figures {sorted(pages)} in {pdf}; not writing cards",
            file=sys.stderr,
        )
        return 1
    out_dir = osp.join(doc_dir, "review")
    os.makedirs(out_dir, exist_ok=True)
    for number, label in enumerate(labels, start=1):
        card = osp.join(out_dir, label.replace(":", "-") + ".pdf")
        subprocess.run(
            ["qpdf", "--empty", "--pages", pdf, str(pages[number]), "--", card],
            check=True,
        )
        print(f"  Figure {number:>2} p{pages[number]:<3} {label:<20} -> {card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

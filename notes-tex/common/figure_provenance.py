#!/usr/bin/env python
# notes-tex/common/figure_provenance.py
# [[notes-tex.common.figure_provenance]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/figure_provenance.py
"""Provenance for numbers drawn inside a hand-built figure.

A matplotlib panel carries its own provenance: the script that drew it read the
data. A draw.io figure does not. Numbers are typed into a canvas by hand, and
once exported to PDF there is nothing left connecting "up to 96 per chip" to the
sentence in a paper that justifies it. That gap is exactly where a figure goes
quietly stale or quietly wrong, and it is not hypothetical -- the first review of
this document found three bad numbers in one figure band: a study total presented
as a per-experiment throughput, a method's row filled in from a different
method's paper, and a protocol capacity claim plotted as an observation.

So every number drawn in a draw.io figure gets a record here, in the same shape
the rest of the repo uses for sourced values: citation key, line in that paper's
mirrored ``paper.md``, and a verbatim quote. The records render as a table in the
document, so a reader can check a figure without access to our library, and
``check()`` fails the build when a figure has no records at all.

**This module is the mechanism; the records live per experiment.** Point
``figure_sources.py`` in an experiment folder at it and import ``FigureNumber``.
Any future notes-tex document with a hand-drawn figure uses the same model, which
is the point -- the drawing skill should populate one of these rather than
inventing a per-figure convention.

Deliberately NOT automatic: nothing parses the ``.drawio`` XML to discover
numbers. A parser would silently pass a figure whose labels were reworded, and
the value here is a human asserting "this number means that quote", which is not
recoverable from the canvas.
"""

from __future__ import annotations

import os.path as osp

from pydantic import BaseModel


class FigureNumber(BaseModel):
    """One number drawn in a figure, and where it came from."""

    figure: str  # drawio basename, e.g. "scrnaseq-method-taxonomy"
    panel: str  # panel letter or band, e.g. "a"
    element: str  # what the number labels, as a reader would name it
    value: str  # exactly as drawn, so a mismatch is visible
    quote: str  # verbatim source sentence
    citation_key: str | None = None  # None => not in the mirror; see `note`
    line: int | None = None
    # Set when the drawn number needed a judgement call, or when the source says
    # something subtly different from what the figure shows. A populated note is
    # a standing invitation to re-check, and it is rendered.
    note: str | None = None

    @property
    def sourced(self) -> bool:
        return self.citation_key is not None


def check(records: list[FigureNumber], figures: list[str]) -> list[str]:
    """Return a list of problems: figures with no records, records with no source."""
    problems = []
    have = {r.figure for r in records}
    for f in figures:
        stem = osp.splitext(osp.basename(f))[0]
        if stem not in have:
            problems.append(f"figure {stem} has no provenance records")
    for r in records:
        if not r.sourced and not r.note:
            problems.append(
                f"{r.figure}/{r.element}: no citation_key and no note explaining why"
            )
    return problems

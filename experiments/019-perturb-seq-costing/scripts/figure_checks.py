# experiments/019-perturb-seq-costing/scripts/figure_checks.py
# [[experiments.019-perturb-seq-costing.scripts.figure_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/figure_checks
"""Legibility checks that run at plot time, so a clash fails loudly.

Every figure in this review has been hand-inspected at least twice, and label
collisions still shipped: an iso-line label sitting on top of the y-axis, two
study labels interleaving where their markers happened to land a decade apart.
Eyeballing does not scale, because the thing that moves a label is usually a
DATA change several files away -- a new study, a corrected UMI count -- and
nobody re-opens the PDF after editing a constant.

So the check lives with the drawing. Each plot script calls ``check_overlaps``
before saving; a real collision raises, which stops the SVG being written at
all. That is deliberate: a figure that silently ships unreadable is worse than a
build that stops.

Two things this deliberately does NOT do. It does not try to fix a collision by
nudging -- automatic label placement produces figures nobody can predict, and
every position in these panels is a considered choice documented at its call
site. And it does not check aesthetics, only overlap of rendered ink boxes.

Everything works in DISPLAY coordinates via ``get_window_extent``, which is the
only frame where "do these two pieces of text touch" is a well-posed question;
data coordinates say nothing about how wide a 5 pt string renders.
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox


class FigureLegibilityError(AssertionError):
    """Raised when a figure would ship with overlapping or clipped labels."""


def _shrink(bb: Bbox, pad: float) -> Bbox:
    """Bounding box inset by ``pad`` points on every side.

    Text extents include a little side bearing, and two labels whose boxes share
    a single pixel of that padding are not actually touching. Shrinking before
    the intersection test is what keeps the check from crying wolf on a figure a
    reader would call fine.
    """
    return Bbox.from_extents(bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def _texts(ax: Axes, ticks: bool = False) -> list[Text]:
    """Annotations and free text on an axes; optionally the x tick labels too.

    The axis label and title are always excluded: they sit outside the data area
    by construction. Annotations are always included, because they are the text
    a human positioned by hand.

    X TICK LABELS ARE INCLUDED BY DEFAULT AT THE CALL SITE, and that was learned
    the hard way. The first version of this module skipped them on the reasoning
    that matplotlib lays them out and so they cannot collide. That is true of
    numeric ticks and false of categorical ones: five two-line category names on
    a 57 mm panel ran together into "not rRNAsurvives QC", which shipped, while
    the checker reported the figure clean. Y ticks stay out -- they stack
    vertically and have never collided.
    """
    out = [t for t in ax.texts if t.get_text().strip()]
    if ticks:
        out += [t for t in ax.get_xticklabels() if t.get_text().strip()]
    return out


def check_overlaps(
    fig: Figure,
    axes: list[Axes] | None = None,
    pad: float = 0.5,
    ignore: set[frozenset[str]] | None = None,
    ticks: bool = True,
) -> list[str]:
    """Return a list of label pairs whose rendered boxes overlap.

    ``ignore`` holds pairs of label strings that are ALLOWED to overlap, keyed
    as frozensets so order does not matter. Use it only for a deliberate overlay
    (a value printed on top of its own bar), never to silence a real collision:
    the entry is a written record that a human looked and accepted it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ignore = ignore or set()
    problems: list[str] = []
    for ax in axes or fig.axes:
        ts = _texts(ax, ticks=ticks)
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                key = frozenset({a.get_text(), b.get_text()})
                if key in ignore:
                    continue
                ba = _shrink(a.get_window_extent(renderer), pad)
                bb = _shrink(b.get_window_extent(renderer), pad)
                if ba.overlaps(bb):
                    problems.append(
                        f"text overlap: {a.get_text()!r} <-> {b.get_text()!r}"
                    )
    return problems


def check_inside_axes(
    fig: Figure,
    axes: list[Axes] | None = None,
    slack: float = 1.0,
    exempt: set[str] | None = None,
) -> list[str]:
    """Return labels that spill outside their axes box.

    A label that leaves the frame is the failure mode that produced the
    ``10^6 UMIs`` iso-line label sitting on the y-axis: it was positioned as a
    fraction along a line whose visible segment barely entered the panel, so the
    text ended up half outside. ``slack`` allows a point or two of overhang,
    which reads as touching the frame rather than escaping it.

    ``exempt`` names labels deliberately placed outside -- a legend caption
    under the axes, for instance.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    exempt = exempt or set()
    problems: list[str] = []
    for ax in axes or fig.axes:
        box = _shrink(ax.get_window_extent(renderer), -slack)
        for t in _texts(ax):
            if t.get_text() in exempt:
                continue
            bb = t.get_window_extent(renderer)
            if not box.contains(bb.x0, bb.y0) or not box.contains(bb.x1, bb.y1):
                problems.append(f"text outside axes: {t.get_text()!r}")
    return problems


def assert_legible(
    fig: Figure,
    axes: list[Axes] | None = None,
    pad: float = 0.5,
    slack: float = 1.0,
    ignore: set[frozenset[str]] | None = None,
    exempt: set[str] | None = None,
    ticks: bool = True,
) -> None:
    """Run every check and raise with all findings at once.

    All findings together rather than the first one, because fixing label
    placement is iterative and being told about one collision per run turns a
    five-minute edit into twenty.
    """
    problems = check_overlaps(fig, axes, pad, ignore, ticks) + check_inside_axes(
        fig, axes, slack, exempt
    )
    if problems:
        raise FigureLegibilityError(
            "figure would ship unreadable:\n  " + "\n  ".join(problems)
        )

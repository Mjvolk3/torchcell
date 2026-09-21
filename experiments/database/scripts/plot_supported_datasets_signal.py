# experiments/database/scripts/plot_supported_datasets_signal.py
# [[experiments.database.scripts.plot_supported_datasets_signal]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_supported_datasets_signal
"""VIEW off the raw-data JSON: scatter of Instances (dataset length) vs Signal
(gzip bits), one point per built dataset, labeled and colored by phenotype
category. Reads only the JSON produced by build_supported_datasets_table.py.

The JSON stores the gzip size in BYTES; this view -- like the table view -- reports
BITS, so the two artifacts of the report carry the same unit.

Repo figure standard (CLAUDE.md "Figure & Plotting Standards"): the ordered,
green-free ``PLOT_PALETTE`` taken in category order, a strict panel width from
``PANEL_WIDTHS_MM``, Arial 6 pt with ``svg.fonttype: none``, a fully boxed axis, and
a true-size SVG saved next to the PNG. Eight phenotype categories take the first
eight palette entries, so the two *dark* repeats (bronze in the orange slot, maroon
in the red slot) land next to their primaries; per the standard we disambiguate with
SHAPE rather than by inventing colors, so each category also gets its own marker.

Run from the repo root (defaults to the newest pre-build snapshot):
  python experiments/database/scripts/plot_supported_datasets_signal.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import random
from collections import Counter, deque
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402

from torchcell.paper.tables import DatasetSignalRecord  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
BITS_PER_BYTE = 8

INK = "#000000"
GRID = "#4A4A4A"
PANEL_W_MM = PANEL_WIDTHS_MM["full"]  # 179 mm -- 51 labeled points need the width
# Width is fixed by the standard, so height is the only axis left to buy label room on.
# 152 mm held 49 labels under adjustText; the 50th and 51st (Bloom 2019, Cooper 2010)
# needed 163 mm, still under MAX_HEIGHT_MM (170). The slot placer below (place_labels)
# replaced adjustText at that height; if it ever reports no collision-free layout, this
# is the constant to raise.
PANEL_H_MM = 163.0
# One marker per category, so the dark repeats (palette 7/8) never rest on color
# alone. Index order matches the JSON's ``sections`` order.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
# Point annotations only. Axes/legend type stays at the 6 pt Nature minimum; the 51
# dataset labels drop to 5 pt so they de-collide on this panel. Bump back to 6 if this
# figure is ever submitted rather than kept as a note/report panel.
LABEL_PT = 5
# Label slot search (see place_labels). Radii are the gap in points between the marker
# center and the near edge of the label box; directions are compass slots with a
# penalty (in points of equivalent radius) so a right-hand slot is preferred over a
# left-hand one at the same distance, and above/below are the last resort.
RADII_PT = (3.0, 4.5, 6.0, 8.0, 10.5, 13.5, 17.0, 21.0, 26.0, 32.0, 40.0, 50.0, 62.0)
DIRECTIONS = (  # (angle in degrees, penalty); 0 is to the right, 90 is up
    (0, 0.0),
    (180, 1.0),
    (45, 0.5),
    (-45, 0.5),
    (135, 1.5),
    (-135, 1.5),
    (90, 2.0),
    (-90, 2.0),
)
# Once a label is far enough out to need a leader, the eight compass slots leave gaps
# that the finer angles fill. Half-integer multiples of 45 degrees, all penalized alike.
FINE_DIRECTIONS = tuple((a + 22.5, 1.0) for a in range(-180, 180, 45))
FINE_FROM_PT = 13.5
DIRECTION_PENALTY_PT = 2.0
LEADER_MIN_PT = 6.0  # a label this far from its marker gets a leader line
MARKER_RADIUS_PT = 18**0.5 / 2  # scatter s=18 is an area in pt^2
MARKER_PAD_PT = 1.0
CROWD_PT = 30.0  # neighborhood half-width used to order the placement
PLACEMENT_ATTEMPTS = 80  # placement orders tried; the cheapest collision-free one wins
PLACEMENT_SEED = 0
MAX_EJECT_AT_ONCE = 3  # a slot blocked by more placed labels than this is not taken
MAX_EJECTIONS_PER_LABEL = 6  # the cycling guard for the repair
HALO_PAD_EM = 0.12
# A tight translucent white plate behind each label so the text reads over the grid.
# Kept translucent: at full strength a plate erases whatever it lands on.
HALO = {
    "boxstyle": f"square,pad={HALO_PAD_EM}",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.55,
}


class Box(NamedTuple):
    """Axis-aligned box in display pixels."""

    x0: float
    y0: float
    x1: float
    y1: float


def _overlaps(a: Box, b: Box) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _inside(a: Box, outer: Box) -> bool:
    return (
        a.x0 >= outer.x0 and a.y0 >= outer.y0 and a.x1 <= outer.x1 and a.y1 <= outer.y1
    )


class Segment(NamedTuple):
    """Line segment in display pixels."""

    x0: float
    y0: float
    x1: float
    y1: float


class Placement(NamedTuple):
    """A label's chosen slot: its cost, offset (radius in points, angle in degrees),
    halo box, and leader segment (None when it rests beside its marker).
    """

    score: float
    r_pt: float
    angle: float
    box: Box
    leader: Segment | None


def _clip(seg: Segment, box: Box) -> tuple[float, float] | None:
    """Liang-Barsky: the parameter interval of ``seg`` that lies inside ``box``."""
    dx, dy = seg.x1 - seg.x0, seg.y1 - seg.y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, seg.x0 - box.x0),
        (dx, box.x1 - seg.x0),
        (-dy, seg.y0 - box.y0),
        (dy, box.y1 - seg.y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return t0, t1


def _crosses(seg: Segment, box: Box) -> bool:
    return _clip(seg, box) is not None


def _trim(seg: Segment, d: float) -> Segment | None:
    """``seg`` with its first ``d`` pixels removed, or None if it is that short."""
    length = math.hypot(seg.x1 - seg.x0, seg.y1 - seg.y0)
    if length <= d:
        return None
    t = d / length
    return Segment(
        seg.x0 + (seg.x1 - seg.x0) * t, seg.y0 + (seg.y1 - seg.y0) * t, seg.x1, seg.y1
    )


def _intersect(a: Segment, b: Segment, skip: float) -> bool:
    """Whether two leaders cross, ignoring their first ``skip`` pixels: leaders from
    markers drawn on top of each other share a start and fan out, which is not a
    crossing.
    """
    ta, tb = _trim(a, skip), _trim(b, skip)
    if ta is None or tb is None:
        return False

    def orient(
        px: float, py: float, qx: float, qy: float, rx: float, ry: float
    ) -> float:
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ta.x0, ta.y0, ta.x1, ta.y1, tb.x0, tb.y0)
    o2 = orient(ta.x0, ta.y0, ta.x1, ta.y1, tb.x1, tb.y1)
    o3 = orient(tb.x0, tb.y0, tb.x1, tb.y1, ta.x0, ta.y0)
    o4 = orient(tb.x0, tb.y0, tb.x1, tb.y1, ta.x1, ta.y1)
    return o1 * o2 < 0 and o3 * o4 < 0


def _leader(cx: float, cy: float, box: Box) -> Segment:
    """The visible leader ``ax.annotate`` draws: from the marker toward the label box
    center, ending where it meets the box.
    """
    mx, my = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    full = Segment(cx, cy, mx, my)
    clipped = _clip(full, box)
    assert clipped is not None, "a box always contains its own center"
    t_in = clipped[0]
    return Segment(cx, cy, cx + (mx - cx) * t_in, cy + (my - cy) * t_in)


def newest_json() -> Path:
    """The most recent pre-build snapshot's JSON."""
    cands = sorted(RESULTS.glob("pre-build/*/supported_datasets.json"))
    if not cands:
        raise FileNotFoundError(
            "No pre-build snapshot; run build_supported_datasets_table.py first"
        )
    return cands[-1]


def _apply_rc() -> None:
    """Repo type + rule standards (Arial 6 pt, editable SVG text, 0.5 pt lines)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Liberation Sans is the metric-compatible Arial substitute on Linux.
            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 7,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            # The legend is a key, not data: dropping it to 5 pt shrinks its box and
            # hands the freed lower-right corner back to the point labels.
            "legend.fontsize": 5,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,  # torchcell.mplstyle sets 'tight'; would break true-size
        }
    )


def _box(ax: Axes) -> None:
    """Full black border on all four spines + a light grid behind the marks."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, which="major", linewidth=0.4, alpha=0.25, color=GRID)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.12, color=GRID)
    ax.set_axisbelow(True)


def _unit(angle_deg: float) -> tuple[float, float, int, int]:
    """Unit offset for a slot angle plus the sign of each component (0 when the
    component is negligible), which picks the text alignment.
    """
    ux, uy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    sx = 0 if abs(ux) < 1e-9 else (1 if ux > 0 else -1)
    sy = 0 if abs(uy) < 1e-9 else (1 if uy > 0 else -1)
    return ux, uy, sx, sy


def _slot_box(
    cx: float, cy: float, angle: float, r_px: float, w: float, h: float, pad: float
) -> Box:
    """Display-space halo box of a label anchored ``r_px`` from marker (cx, cy) at
    ``angle``. Matches what ``ax.annotate`` draws for the same offset with ha/va chosen
    by ``_align``: the text box grows AWAY from the marker.
    """
    ux, uy, sx, sy = _unit(angle)
    ax_ = cx + ux * r_px
    ay_ = cy + uy * r_px
    x0 = {1: ax_, 0: ax_ - w / 2, -1: ax_ - w}[sx] - pad
    y0 = {1: ay_, 0: ay_ - h / 2, -1: ay_ - h}[sy] - pad
    return Box(x0, y0, x0 + w + 2 * pad, y0 + h + 2 * pad)


def _align(angle: float) -> tuple[str, str]:
    _, _, sx, sy = _unit(angle)
    return (
        {1: "left", 0: "center", -1: "right"}[sx],
        {1: "bottom", 0: "center", -1: "top"}[sy],
    )


def place_labels(
    fig: Figure, ax: Axes, leg: Legend, records: list[DatasetSignalRecord]
) -> None:
    """Label every marker with its dataset name, each label beside its OWN point.

    Deterministic slot search rather than a force solver. adjustText was the previous
    approach and, at 51 points, it let labels drift up to 90 pt from their markers with
    leader lines crossing half the panel, so a reader could not tell which label belonged
    to which point. Here each label tries the compass slots around its marker at
    increasing radii (``RADII_PT``), nearest first, right-hand side first, and takes the
    first slot that is inside the axes and clear of every marker, the legend, and every
    label already placed. Crowded points (most neighbors within ``CROWD_PT``) are placed
    first, so the dense 10^5-10^6 band gets the close slots and the isolated points take
    what is left. A leader line is drawn only when the label had to move past
    ``LEADER_MIN_PT``; a label resting beside its marker needs none.

    Raises if some label has no collision-free slot: that is the signal to give the panel
    more height (width is fixed by the figure standard), not to accept an overlap.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
    px_per_pt = fig.dpi / 72.0
    fp = FontProperties(family=plt.rcParams["font.sans-serif"], size=LABEL_PT)
    halo_pad = HALO_PAD_EM * LABEL_PT * px_per_pt

    axes_box = Box(*ax.get_window_extent(renderer).extents)
    axes_box = Box(axes_box.x0 + 1, axes_box.y0 + 1, axes_box.x1 - 1, axes_box.y1 - 1)
    obstacles = [Box(*leg.get_window_extent(renderer).extents)]

    centers = ax.transData.transform(
        [(r.instances, r.signal_bytes * BITS_PER_BYTE) for r in records]
    )
    m = (MARKER_RADIUS_PT + MARKER_PAD_PT) * px_per_pt
    marker_boxes = [Box(cx - m, cy - m, cx + m, cy + m) for cx, cy in centers]
    extents = [
        renderer.get_text_width_height_descent(r.name, fp, ismath=False)
        for r in records
    ]

    crowd = CROWD_PT * px_per_pt

    def crowding(i: int) -> int:
        cx, cy = centers[i]
        return sum(
            1
            for j, (x, y) in enumerate(centers)
            if j != i and abs(x - cx) < crowd and abs(y - cy) < crowd
        )

    slots = sorted(
        (r + DIRECTION_PENALTY_PT * pen, r, angle)
        for r in RADII_PT
        for angle, pen in DIRECTIONS + (FINE_DIRECTIONS if r >= FINE_FROM_PT else ())
    )

    def blockers(
        i: int, box: Box, leader: Segment | None, placed: dict[int, Placement]
    ) -> set[int]:
        """Placed labels this slot conflicts with: box on box, box under a leader,
        leader over a box, leader over a leader.
        """
        out = set()
        for j, p in placed.items():
            if _overlaps(box, p.box) or (
                p.leader is not None and _crosses(p.leader, box)
            ):
                out.add(j)
            elif leader is not None and (
                _crosses(leader, p.box)
                or (p.leader is not None and _intersect(leader, p.leader, m))
            ):
                out.add(j)
        return out

    def find_slot(
        i: int, placed: dict[int, Placement], tabu: frozenset[int] = frozenset()
    ) -> tuple[Placement, set[int]] | Counter[str]:
        """The cheapest slot for label ``i`` clear of every placed label, else the slot
        with the fewest blocking labels (ties to the cheaper), with those blockers.
        A slot blocked by a ``tabu`` label is never offered. Returns the rejection
        tally when no slot is left.
        """
        cx, cy = centers[i]
        w, h, _ = extents[i]
        others = [b for j, b in enumerate(marker_boxes) if j != i]
        # A leader necessarily starts inside any marker that overlaps this one (Kuzmin
        # 2020 tmi and tmf are drawn on top of each other), so those are not obstacles
        # to the line, only to the box.
        clear_of = [b for b in others if not _overlaps(b, marker_boxes[i])]
        rejected: Counter[str] = Counter()
        best: tuple[int, float, Placement, set[int]] | None = None
        for score, r_pt, angle in slots:
            box = _slot_box(cx, cy, angle, r_pt * px_per_pt, w, h, halo_pad)
            if not _inside(box, axes_box):
                rejected["outside axes"] += 1
                continue
            if any(_overlaps(box, b) for b in obstacles):
                rejected["on legend"] += 1
                continue
            if any(_overlaps(box, b) for b in others):
                rejected["on marker"] += 1
                continue
            leader = _leader(cx, cy, box) if r_pt >= LEADER_MIN_PT else None
            if leader is not None and any(
                _crosses(leader, b) for b in obstacles + clear_of
            ):
                rejected["leader over marker"] += 1
                continue
            blocking = blockers(i, box, leader, placed)
            if blocking & tabu:
                rejected["blocked by ejector"] += 1
                continue
            placement = Placement(score, r_pt, angle, box, leader)
            if not blocking:
                return placement, blocking
            if best is None or (len(blocking), score) < (best[0], best[1]):
                best = (len(blocking), score, placement, blocking)
        if best is None:
            return rejected
        return best[2], best[3]

    def solve(order: list[int]) -> dict[int, Placement] | str:
        """Greedy placement in ``order`` with min-conflicts repair: a label with no
        clear slot takes the slot with the fewest blocking labels, ejects exactly
        those, and they go to the back of the queue. A label ejected more than
        ``MAX_EJECTIONS_PER_LABEL`` times ends the attempt, which is what stops two
        labels trading one slot forever. Returns the placement, or why it gave up.
        """
        queue = deque(order)
        placed: dict[int, Placement] = {}
        ejected: Counter[int] = Counter()
        # Who ejected whom: a label coming back may not eject its ejector, which is
        # what stops two labels trading one slot back and forth.
        ejector: dict[int, int] = {}
        while queue:
            i = queue.popleft()
            tabu = frozenset({ejector[i]}) if i in ejector else frozenset()
            found = find_slot(i, placed, tabu)
            if isinstance(found, Counter):
                return (
                    f"{records[i].name!r} has no slot left "
                    f"({len(placed)} of {len(records)} placed; rejected {dict(found)})"
                )
            slot, blocking = found
            if len(blocking) > MAX_EJECT_AT_ONCE:
                return (
                    f"{records[i].name!r} is blocked by {len(blocking)} labels in its best "
                    f"slot ({len(placed)} of {len(records)} placed)"
                )
            for j in blocking:
                ejected[j] += 1
                if ejected[j] > MAX_EJECTIONS_PER_LABEL:
                    return (
                        f"{records[j].name!r} was ejected {ejected[j]} times; cycling"
                    )
                del placed[j]
                ejector[j] = i
                queue.append(j)
            placed[i] = slot
        return placed

    # One greedy order can wall in the last label of a dense cluster with the labels and
    # leaders placed before it, so several orders are tried and the cheapest solution
    # kept: crowded-first (the natural order), then seeded shuffles of it.
    base = sorted(range(len(records)), key=lambda i: (-crowding(i), -extents[i][0]))
    rng = random.Random(PLACEMENT_SEED)
    best: dict[int, Placement] | None = None
    best_cost = math.inf
    failures: list[str] = []
    for attempt in range(PLACEMENT_ATTEMPTS):
        order = list(base)
        if attempt:
            rng.shuffle(order)
        result = solve(order)
        if isinstance(result, str):
            failures.append(result)
            continue
        cost = sum(p.score for p in result.values())
        if cost < best_cost:
            best, best_cost = result, cost
    if best is None:
        raise RuntimeError(
            f"No collision-free labeling in {PLACEMENT_ATTEMPTS} orders; raise PANEL_H_MM. "
            f"First failure: {failures[0]}"
        )
    print(
        f"labels placed: {len(failures)} of {PLACEMENT_ATTEMPTS} orders failed; "
        f"best cost {best_cost:.0f} pt"
    )

    for i, p in best.items():
        r_pt, angle = p.r_pt, p.angle
        ha, va = _align(angle)
        ux, uy, _, _ = _unit(angle)
        arrow = (
            {
                "arrowstyle": "-",
                "color": GRID,
                "linewidth": 0.3,
                "shrinkA": MARKER_RADIUS_PT + 0.5,
                "shrinkB": 0.5,
            }
            if r_pt >= LEADER_MIN_PT
            else None
        )
        ax.annotate(
            records[i].name,
            xy=(records[i].instances, records[i].signal_bytes * BITS_PER_BYTE),
            xytext=(ux * r_pt, uy * r_pt),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=LABEL_PT,
            color=INK,
            zorder=5,
            bbox=HALO,
            arrowprops=arrow,
        )


def main() -> None:
    """Draw the instances-vs-signal scatter and save PNG + true-size SVG."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="supported_datasets.json (default: newest)",
    )
    args = ap.parse_args()
    load_dotenv()

    data_path = args.data or newest_json()
    payload = json.loads(data_path.read_text())
    records = [
        DatasetSignalRecord(**d)
        for d in payload["datasets"]
        if d["built"] and d["signal_bytes"] > 0
    ]

    _apply_rc()
    sections = payload["sections"]
    color = {s: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, s in enumerate(sections)}
    marker = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(sections)}

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM)))
    for s in sections:
        pts = [r for r in records if r.section == s]
        if not pts:
            continue
        ax.scatter(
            [r.instances for r in pts],
            [r.signal_bytes * BITS_PER_BYTE for r in pts],
            s=18,
            color=color[s],
            marker=marker[s],
            edgecolor=INK,
            linewidth=0.3,
            label=s,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Instances (dataset length)")
    ax.set_ylabel("Signal (gzip, bits)")
    ax.set_title(
        f"Training signal vs scale — {payload['status']} ({payload['date']})",
        loc="left",
        color=INK,
    )
    # Pad the limits by a fraction of a decade so the 6 pt labels have somewhere to
    # go: adjustText moves text but cannot push it outside the axes, and a label that
    # runs off the panel edge is the failure mode on a fixed 179 mm width.
    xs = [r.instances for r in records]
    ys = [r.signal_bytes * BITS_PER_BYTE for r in records]
    # Extra room on the right: the longest labels sit there.
    ax.set_xlim(min(xs) / 4, max(xs) * 20)
    ax.set_ylim(min(ys) / 6, max(ys) * 4)
    _box(ax)
    # Legend INSIDE the axes (the points ride the diagonal, so the lower right is
    # empty), which keeps the panel at exactly 179 mm.
    leg = ax.legend(
        title="Phenotype category",
        loc="lower right",
        framealpha=0.95,
        edgecolor=INK,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    leg.get_frame().set_linewidth(0.5)
    leg.get_title().set_fontsize(5)

    place_labels(fig, ax, leg, records)

    out_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")
    os.makedirs(out_dir, exist_ok=True)
    stem = osp.join(out_dir, "supported-datasets-instances-vs-signal")
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    savefig_true_size_svg(fig, f"{stem}.svg", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.png + .svg  ({len(records)} datasets)")


if __name__ == "__main__":
    main()

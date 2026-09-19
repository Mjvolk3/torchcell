# experiments/024-perturb-seq-costing/scripts/pilot_options.py
# [[experiments.024-perturb-seq-costing.scripts.pilot_options]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/pilot_options
"""Price the candidate FIRST experiments, for both hosts, and draw the decision.

Sec. 5 of the review prices a genome-scale screen. Nothing there answers the
question a bench scientist actually has to act on, which is what the first run
costs and what it returns, so this module composes the same platform constants
over pilot-sized designs instead.

Everything is read from the committed model:

* ``cost_model``    -- the four platforms, their yields, their per-batch costs
* ``uiuc_core_data``-- the Carver Center rate card, per lane and per 10x channel
* ``method_data``   -- the published yields the projections stand on

Nothing is introduced here except the pilot DESIGNS themselves, which are the
runs under consideration and are declared in ``CANDIDATES`` with a note saying
what each one buys.

Two departures from Sec. 5 are deliberate, and both make the pilot figures
cheaper than the genome-scale ones rather than dearer:

1. **Sequencing is priced on the cheapest option that fits, not on the biggest
   flow cell.** A genome-scale screen fills NovaSeq X 25B lanes, where the price
   per read is lowest. A pilot does not fill one, and a lane is an indivisible
   purchase, so the cheapest *total* is usually a MiSeq run or a 10B split lane.
   ``cheapest_sequencing`` scans every paired-end configuration the core sells.
2. **The first 10x channel costs more than the next.** The rate card charges
   $2,650 for one RNA library and $2,240 each for two to seven, so a two-arm
   pilot is not twice a one-arm pilot.

Run:  python experiments/024-perturb-seq-costing/scripts/pilot_options.py
Writes: results/pilot_options.json
        notes-tex/024-perturb-seq-pilot/tables/t1-pilot-options.tex
        $ASSET_IMAGES_DIR/024-perturb-seq-costing/pilot_options.{svg,png}

The table path is written out rather than shared with ``render_tex_tables.py``,
whose ``OUT`` still points at ``notes-tex/microbe-perturb-seq/tables`` and has
done since that directory was renamed to ``024-perturb-seq-costing``.
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import cost_model as CM  # noqa: E402
import uiuc_core_data as UC  # noqa: E402
from figure_checks import assert_legible  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)


def place_panel_letters(fig, axes, letters) -> None:
    """Bold panel letters outside each panel's full extent.

    The same helper the other figures in this document use (plot_economics,
    plot_environments), rather than ``torchcell.utils.panel_label``: a letter
    drawn on the AXES is flagged by ``check_inside_axes``, which is the gate this
    document runs on every figure, so the letters are figure-level text placed
    against each panel's tight bounding box.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        # Clamped to the canvas. A panel whose y tick labels are long (the
        # horizontal-bar panels here) has a tight bounding box that already
        # starts at the figure edge, and the usual 0.010 offset then puts the
        # letter off the canvas, where check_inside_figure catches it.
        fig.text(
            max(bb.x0 - 0.010, 0.002),
            min(bb.y1 + 0.012, 0.988),
            letter,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=20,
        )


# --- published guide/genotype assignment rates -------------------------------
# q, the per-cell probability that the perturbation carried is actually read out.
# It is the one parameter that enters a multiplex design exponentially (q^k), and
# the two published microbial values are three-fold apart, which is why the first
# S. cerevisiae experiment exists to measure it.
#
# Brandner et al. in E. coli: "We sequenced 365,000 cells and retained 76,068 of
# them after assignment" (brandnerPooledSinglecellCRISPRa2025 paper.md:89).
BRANDNER_SEQUENCED = 365_000
BRANDNER_RETAINED = 76_068
Q_BRANDNER = BRANDNER_RETAINED / BRANDNER_SEQUENCED
# Nadal-Ribelles et al. in S. cerevisiae, genotype assigned for more than 71% of
# cells, and NOT from the transcriptome alone -- a separate one-step PCR amplicon
# library off the URA3 3'UTR does part of the work (Sec. 3.7).
Q_NADAL = 0.71

# A profiling run with no library has no guide to read, so q does not apply to it;
# what applies is the platform's own usable fraction (QC survival).


class PilotCost(BaseModel):
    """One candidate first experiment, priced end to end."""

    key: str
    host: str
    label: str
    route: str
    n_conditions: int
    usable_cells: int
    sequenced_cells: int
    n_batches: int
    reagents_usd: float
    sequencing_usd: float
    sequencing_option: str
    recurring_usd: float
    one_time_usd: float
    read_pairs: float
    returns: str
    note: str = ""


def cheapest_sequencing(n_read_pairs: float) -> tuple[str, float]:
    """Cheapest paired-end configuration the core sells that covers the reads.

    Whole lanes, because a lane is the indivisible purchase. Every paired-end
    option is scanned rather than assuming the largest flow cell: at pilot scale
    the 25B lane that is cheapest per read is the most expensive per experiment,
    since a single one holds 3.2 billion read pairs and a pilot wants 0.5.
    """
    best: tuple[str, float] | None = None
    for opt in UC.NOVASEQ_X + UC.MISEQ_I100:
        if "paired" not in opt.read_type:
            continue
        lanes = math.ceil(n_read_pairs / opt.read_pairs_per_lane)
        usd = lanes * opt.usd_per_lane
        if best is None or usd < best[1]:
            label = opt.label if lanes == 1 else f"{opt.label} x{lanes}"
            best = (label, usd)
    if best is None:
        raise ValueError("no paired-end sequencing option found in the rate card")
    return best


def tenx_channel_usd(n_channels: int, crispr_addon: bool) -> float:
    """What n 10x channels cost at the published tiered rate.

    The card prices the first RNA library at $2,650, the second through seventh
    at $2,240 each and the eighth onward at $1,880. CRISPR screening is a
    per-sample add-on on top, which a profiling run with no library does not pay.
    """
    usd = 0.0
    for i in range(n_channels):
        if i == 0:
            usd += UC.TENX_PRICES["rna_library_1"]
        elif i < 7:
            usd += UC.TENX_PRICES["rna_library_2_to_7_each"]
        else:
            usd += UC.TENX_PRICES["rna_library_8plus_each"]
    if crispr_addon:
        usd += n_channels * UC.TENX_PRICES["feature_barcoding_addon_per_sample"]
    return usd


def droplet_pilot(
    key: str,
    host: str,
    label: str,
    n_conditions: int,
    returns: str,
    crispr_addon: bool,
    guide_reads_per_cell: float = 0.0,
    channels_per_condition: int = 1,
    note: str = "",
) -> PilotCost:
    """A pilot on the 10x route, one or more channels.

    An unmodified droplet channel carries no condition label, so conditions
    cannot share one: ``n_conditions`` channels are bought whatever the cell
    count (Sec. 7.3). That is the property panel (b) exists to show.

    ``guide_reads_per_cell`` is 10x's stated minimum for a guide library, 5,000
    read pairs per cell against 20,000 for gene expression. It is pooled with the
    expression library rather than submitted separately, so it adds reads and not
    a second submission.
    """
    p = CM.TENX
    n_channels = n_conditions * channels_per_condition
    sequenced = n_channels * p.cells_per_batch
    usable = int(sequenced * p.usable_fraction)
    reads = sequenced * (p.reads_per_cell + guide_reads_per_cell)
    reads /= p.usable_read_fraction
    opt, seq_usd = cheapest_sequencing(reads)
    reagents = tenx_channel_usd(n_channels, crispr_addon)
    return PilotCost(
        key=key,
        host=host,
        label=label,
        route="10x droplet",
        n_conditions=n_conditions,
        usable_cells=usable,
        sequenced_cells=sequenced,
        n_batches=n_channels,
        reagents_usd=reagents,
        sequencing_usd=seq_usd,
        sequencing_option=opt,
        recurring_usd=reagents + seq_usd,
        one_time_usd=0.0,
        read_pairs=reads,
        returns=returns,
        note=note,
    )


def splitpool_pilot(
    key: str,
    host: str,
    label: str,
    n_conditions: int,
    usable_per_condition: int,
    returns: str,
    note: str = "",
) -> PilotCost:
    """A pilot on the plate route, where conditions share one protocol run.

    Round 1 is a 96-well plate and its index is read out of every cell whether or
    not it has been given a meaning, so up to 96 conditions ride in one run at no
    extra barcode cost (Sec. 7.3). The run is priced as published rather than
    depleted: rRNA depletion is itself one of the things a first run measures.
    """
    p = CM.SPLITSEQ_PUBLISHED
    usable = usable_per_condition * n_conditions
    sequenced = math.ceil(usable / p.usable_fraction)
    n_runs = max(1, math.ceil(sequenced / p.cells_per_batch))
    n_sub = math.ceil(sequenced / (p.cells_per_sublibrary or sequenced))
    reads = sequenced * p.reads_per_cell / p.usable_read_fraction
    opt, seq_usd = cheapest_sequencing(reads)
    reagents = n_runs * p.cost_per_batch_usd + n_sub * p.cost_per_sublibrary_usd
    return PilotCost(
        key=key,
        host=host,
        label=label,
        route="split-pool plate",
        n_conditions=n_conditions,
        usable_cells=usable,
        sequenced_cells=sequenced,
        n_batches=n_runs,
        reagents_usd=reagents,
        sequencing_usd=seq_usd,
        sequencing_option=opt,
        recurring_usd=reagents + seq_usd,
        one_time_usd=p.startup_usd,
        read_pairs=reads,
        returns=returns,
        note=note,
    )


# --- the candidate first experiments -----------------------------------------
# Each is a run somebody could start next month. The S. cerevisiae ones use the
# MAGIC single-guide CRISPRi library, which exists; the P. kudriavzevii ones
# carry no library, because pooled library-scale screening in that host waits on
# transformation efficiency (Sec. 6.2) and none of these do.
def candidates() -> list[PilotCost]:
    return [
        droplet_pilot(
            key="sc_guide_1ch",
            host="S. cerevisiae",
            label="Guide capture, 1 channel",
            n_conditions=1,
            crispr_addon=True,
            guide_reads_per_cell=5_000.0,
            returns="q, the per-cell guide detection rate",
            note=(
                "The gate. One channel of the existing library on the core's 5' "
                "kit, which captures the guide with a scaffold primer rather "
                "than an engineered capture sequence (Sec. 3.7)."
            ),
        ),
        droplet_pilot(
            key="sc_guide_2ch",
            host="S. cerevisiae",
            label="Guide capture, low vs high copy",
            n_conditions=2,
            crispr_addon=True,
            guide_reads_per_cell=5_000.0,
            returns="q on two vector backbones",
            note=(
                "The same measurement with a second arm on a higher-copy vector, "
                "which is the only reason to touch copy number this early: more "
                "guide transcript per cell may raise q. Untested."
            ),
        ),
        droplet_pilot(
            key="pk_profile_1ch",
            host="P. kudriavzevii",
            label="Unperturbed profile, 1 medium",
            n_conditions=1,
            crispr_addon=False,
            returns="wall digestion, UMIs per cell, rRNA fraction",
            note=(
                "No library and no guides, so it waits on none of the delivery "
                "questions still open in this host."
            ),
        ),
        droplet_pilot(
            key="pk_profile_4med",
            host="P. kudriavzevii",
            label="Unperturbed profile, 4 media",
            n_conditions=4,
            crispr_addon=False,
            returns="the same three, plus a 4-condition response atlas",
            note=(
                "Four media on the droplet route is four channels: nothing in a "
                "channel says which medium a cell came from."
            ),
        ),
        splitpool_pilot(
            key="pk_plate_8med",
            host="P. kudriavzevii",
            label="Plate route, 8 media in one run",
            n_conditions=8,
            usable_per_condition=2_500,
            returns="the same three, across 8 media, plus the plate chemistry",
            note=(
                "Eight media ride in one protocol run because round 1 is a plate "
                "of wells. The plate set is a $10,000 one-time purchase that "
                "serves about 215 runs."
            ),
        ),
    ]


# --- panel (b): conditions on each route -------------------------------------
def condition_scaling(
    n_conditions: list[int], usable_per_condition: int = 11_000
) -> dict[str, list[float]]:
    """Recurring cost against number of media, at a fixed cells-per-medium.

    The default is one 10x channel's usable yield, so the droplet curve starts at
    exactly one channel and the comparison is at matched cells rather than at
    matched effort. The three routes differ only in whether a condition label
    exists: droplet buys a channel per condition, the two plate routes pool.
    """
    out: dict[str, list[float]] = {
        "n": [],
        "droplet": [],
        "preindexed": [],
        "plate": [],
    }
    for e in n_conditions:
        out["n"].append(e)
        for name, p in (
            ("droplet", CM.TENX),
            ("preindexed", CM.TENX_SCIFI_PROJECTED),
            ("plate", CM.SPLITSEQ_PUBLISHED),
        ):
            per_cond_sequenced = math.ceil(usable_per_condition / p.usable_fraction)
            total_sequenced = per_cond_sequenced * e
            if name == "droplet":
                # No condition label: channels are bought per condition.
                batches = e * math.ceil(per_cond_sequenced / p.cells_per_batch)
            else:
                # A round-1 well carries the condition, so cells pool into shared
                # batches and only the rounding is paid.
                batches = max(1, math.ceil(total_sequenced / p.cells_per_batch))
            if name == "plate":
                n_sub = math.ceil(
                    total_sequenced / (p.cells_per_sublibrary or total_sequenced)
                )
                reagents = (
                    batches * p.cost_per_batch_usd + n_sub * p.cost_per_sublibrary_usd
                )
            else:
                reagents = tenx_channel_usd(batches, crispr_addon=False)
            reads = total_sequenced * p.reads_per_cell / p.usable_read_fraction
            _, seq = cheapest_sequencing(reads)
            out[name].append(reagents + seq)
    return out


TEX_HEADER = (
    "%% GENERATED FILE -- do not hand-edit.\n"
    "%% SOURCE: experiments/024-perturb-seq-costing/scripts/pilot_options.py\n"
)


def tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def emit_table(cands: list[PilotCost], out_dir: str) -> None:
    """The candidate runs as a booktabs table, for the pilot document.

    Host and cell count are what a bench scientist needs beside the price, so
    they are columns rather than caption prose. The one-time column is separate
    because it is a purchase that outlives the experiment: the plate set covers
    about 215 protocol runs.
    """
    rows = []
    for c in cands:
        one_time = f"\\${c.one_time_usd:,.0f}" if c.one_time_usd else "--"
        rows.append(
            " & ".join(
                [
                    tex_escape(c.label),
                    f"\\org{{{c.host}}}",
                    f"{c.usable_cells:,}",
                    f"\\${c.recurring_usd:,.0f}",
                    one_time,
                    tex_escape(c.returns),
                ]
            )
            + r" \\"
        )
    body = "\n".join(
        [
            r"\begin{table}[htbp]\centering",
            r"\footnotesize",
            r"\caption[]{\textbf{The candidate first runs, priced end to end.} "
            r"Usable cells are those expected to survive quality control, 55\% of "
            r"a droplet channel's 20{,}000 and 25\% of a protocol run's 480{,}000. "
            r"Recurring cost is reagents plus sequencing at the cheapest "
            r"configuration the Carver Center sells that holds the reads, which at "
            r"this scale is never the flow cell a genome-scale screen would use. "
            r"The one-time column is the split-pool barcode plate set, which "
            r"covers about 215 protocol runs. The route is the plate one for the "
            r"eight-media row and 10x droplet for every other.}\label{tab:pilots}",
            # Fixed-width p{} columns for the two free-text fields. As plain `l`
            # columns the table ran 28 mm past the text block, which is the
            # overfull-hbox failure the document gate exists to catch.
            r"\begin{tabular}{@{}"
            r">{\raggedright\arraybackslash}p{40mm}"
            r">{\raggedright\arraybackslash}p{23mm}rrr"
            r">{\raggedright\arraybackslash}p{44mm}@{}}",
            r"\toprule",
            r"run & host & usable cells & recurring & one-time & returns \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, "t1-pilot-options.tex")
    with open(path, "w") as f:
        f.write(TEX_HEADER + body + "\n")
    print(f"wrote {path}")


def main() -> None:
    load_dotenv()
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")
    os.makedirs(images, exist_ok=True)
    os.makedirs(CM.RESULTS_DIR, exist_ok=True)

    cands = candidates()
    n_cond = list(range(1, 13))
    scaling = condition_scaling(n_cond)

    # Scale reference: what a pilot buys against what a screen needs.
    screen_100 = CM.ScreenDesign(cells_per_gene=100)
    screen_250 = CM.ScreenDesign(cells_per_gene=250)
    scale_rows = [
        ("1 droplet channel", int(CM.TENX.cells_per_batch * CM.TENX.usable_fraction)),
        (
            "1 preindexed channel",
            int(
                CM.TENX_SCIFI_PROJECTED.cells_per_batch
                * CM.TENX_SCIFI_PROJECTED.usable_fraction
            ),
        ),
        (
            "1 plate run",
            int(
                CM.SPLITSEQ_PUBLISHED.cells_per_batch
                * CM.SPLITSEQ_PUBLISHED.usable_fraction
            ),
        ),
        ("screen, 100 cells/gene", screen_100.usable_cells_needed),
        ("screen, 250 cells/gene", screen_250.usable_cells_needed),
    ]

    payload = {
        "generated_by": "experiments/024-perturb-seq-costing/scripts/pilot_options.py",
        "rates_effective": UC.RATES_EFFECTIVE,
        "rates_retrieved": UC.RETRIEVED,
        "candidates": [c.model_dump() for c in cands],
        "condition_scaling": scaling,
        "scale_reference": {k: v for k, v in scale_rows},
        "q": {
            "brandner_ecoli": Q_BRANDNER,
            "brandner_sequenced": BRANDNER_SEQUENCED,
            "brandner_retained": BRANDNER_RETAINED,
            "nadal_scerevisiae": Q_NADAL,
        },
    }
    repo = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    emit_table(cands, osp.join(repo, "notes-tex", "024-perturb-seq-pilot", "tables"))

    out_json = osp.join(CM.RESULTS_DIR, "pilot_options.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}")
    for c in cands:
        print(
            f"  {c.host:16s} {c.label:34s} ${c.recurring_usd:8,.0f}"
            f"  +${c.one_time_usd:,.0f} one-time  [{c.sequencing_option}]"
        )

    # ---------------- figure ----------------
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )
    C_SC = PLOT_PALETTE[0]
    C_PK = PLOT_PALETTE[2]
    C_DROP = PLOT_PALETTE[1]
    C_PI = PLOT_PALETTE[3]
    C_PLATE = PLOT_PALETTE[4]
    C_GRAY = PLOT_PALETTE[5]

    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(104))
    )

    # (a) what each candidate costs
    ax = axes[0][0]
    y = np.arange(len(cands))[::-1]
    reag = [c.reagents_usd for c in cands]
    seq = [c.sequencing_usd for c in cands]
    face = [C_SC if c.host.startswith("S.") else C_PK for c in cands]
    ax.barh(y, reag, color=face, edgecolor="black", linewidth=0.5, height=0.6)
    ax.barh(
        y,
        seq,
        left=reag,
        color="white",
        edgecolor="black",
        linewidth=0.5,
        height=0.6,
        hatch="///",
    )
    for yi, c in zip(y, cands):
        total = c.recurring_usd
        # Escaped: matplotlib treats a PAIR of unescaped $ as a mathtext span, so
        # "$4,990 + $10,000 once" silently renders as "4, 990 + 10,000 once"
        # with both dollar signs eaten and the comma respaced.
        txt = rf"\${total:,.0f}"
        if c.one_time_usd:
            txt += rf" + \${c.one_time_usd:,.0f} once"
        ax.text(total + 400, yi, txt, va="center", ha="left", fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels([c.label for c in cands], fontsize=5.5)
    ax.set_xlabel("recurring cost, USD")
    ax.set_xlim(0, 24_000)
    ax.set_title("What each candidate first run costs", fontsize=6, loc="left")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_SC, ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=C_PK, ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", lw=0.5, hatch="///"),
    ]
    ax.legend(
        handles,
        ["S. cerevisiae reagents", "P. kudriavzevii reagents", "sequencing"],
        loc="upper right",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        borderpad=0.4,
    )

    # (b) cost against number of media
    ax = axes[0][1]
    ax.plot(
        scaling["n"],
        scaling["droplet"],
        color=C_DROP,
        lw=1.0,
        label="10x droplet, one channel per medium",
    )
    ax.plot(
        scaling["n"],
        scaling["preindexed"],
        color=C_PI,
        lw=1.0,
        ls=(0, (4, 1.5)),
        label="10x + preindexing (projected)",
    )
    ax.plot(
        scaling["n"],
        scaling["plate"],
        color=C_PLATE,
        lw=1.0,
        label="split-pool plate, one run",
    )
    ax.set_xlabel("media in the experiment")
    ax.set_ylabel("recurring cost, USD")
    ax.set_xlim(1, 12)
    ax.set_ylim(0, 40_000)
    ax.set_title(
        "A medium is free on a plate and a channel in a droplet", fontsize=6, loc="left"
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        borderpad=0.4,
    )

    # (c) pilot against screen, on cells
    ax = axes[1][0]
    labels = [r[0] for r in scale_rows][::-1]
    vals = [r[1] for r in scale_rows][::-1]
    cols = [C_GRAY, C_GRAY, C_GRAY, C_PLATE, C_PLATE][::-1]
    yy = np.arange(len(vals))
    ax.barh(yy, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.6)
    for yi, v in zip(yy, vals):
        ax.text(v * 1.25, yi, f"{v:,}", va="center", ha="left", fontsize=5.5)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xscale("log")
    ax.set_xlim(5e3, 1e7)
    ax.set_xlabel("usable cells")
    ax.set_title(
        "What one run buys, against what a screen needs", fontsize=6, loc="left"
    )

    # (d) what q does
    ax = axes[1][1]
    q = np.linspace(0.05, 1.0, 200)
    for k, style in ((1, "-"), (2, (0, (4, 1.5))), (3, (0, (1, 1.2)))):
        ax.plot(
            q,
            q**k,
            color=C_GRAY if k == 1 else C_DROP,
            ls=style,
            lw=1.0,
            label=f"{k} guide" + ("" if k == 1 else "s") + " per cell",
        )
    ax.axvline(Q_BRANDNER, color=C_PLATE, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
    ax.axvline(Q_NADAL, color=C_SC, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
    ax.text(Q_BRANDNER + 0.02, 0.93, "E. coli\n0.21", fontsize=5.5, va="top")
    ax.text(Q_NADAL - 0.02, 0.93, "yeast\n0.71", fontsize=5.5, va="top", ha="right")
    ax.set_xlabel("q, cells whose perturbation is read")
    ax.set_ylabel("fraction of cells usable")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "q is linear at one guide and exponential past it", fontsize=6, loc="left"
    )
    # Lower right: the upper left is where the two published q values are
    # annotated, and a legend there hid the E. coli one entirely.
    ax.legend(
        loc="lower right",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        borderpad=0.4,
    )

    for row in axes:
        for a in row:
            for s in a.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)

    fig.tight_layout(pad=0.6, w_pad=1.6, h_pad=2.4)
    # tight_layout packs the top row against the canvas edge, which leaves the
    # panel letters (drawn above each panel's tight bounding box) hanging off it.
    fig.subplots_adjust(top=0.92)
    flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    place_panel_letters(fig, flat, ["a", "b", "c", "d"])
    assert_legible(fig, axes=flat)
    stem = osp.join(images, "pilot_options")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()

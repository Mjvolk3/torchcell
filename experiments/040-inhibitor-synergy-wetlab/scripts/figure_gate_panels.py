# experiments/040-inhibitor-synergy-wetlab/scripts/figure_gate_panels.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.figure_gate_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/figure_gate_panels

r"""Mockups of the hydrolysate inhibitor figure (manuscript R5) and its supplement.

Five sheets, every plotted value read from a committed result file; nothing is typed
into a panel. They are held for a decision in ``notes-tex/wet-lab/env-case-study-gate/``
before any of them enters the paper. Nothing in them is approved: the marks say only
what is on file.

* ``gate_concept``: the task (a compound and its dose enter the cell graph transformer
  as a token the genes attend to; a public-screen strain is read at its deleted genes,
  the bAID host at the mean over genes), the two data sources, and the evaluation.
* ``gate_main``: the four-by-three plan of the experiment note, a to l.
* ``gate_si1_benchmarks``: the public-benchmark detail behind d, e and f.
* ``gate_si2_growth``: the single-agent and growth-call detail behind j, k and l.
* ``gate_si3_profiles``: the composition-rule and profile detail behind h and i.

ENCODING, two channels and no more. COLOR is the source of a predicted value: a
measured value; a model-free rule, one color per rule (Bliss independence, Loewe
additivity, highest single agent, the mean of the singles); a learned model (ridge and
its kernel and nearest-neighbor siblings on chemistry features, the cell graph
transformer); and gray for a gene-mean or permutation control and for chance. HATCH is
the software growth call, drawn only where a panel shows both calls; a plain bar is the
served raw-curve call, which is primary. The learner's kernel and encoder are written in
the label and never encoded. A light fill appears only where a panel compares two
readings of the same arm, and that panel's legend names it. Text in a panel is always
black.

STATUS. Each panel letter carries two symbols: the section status (the author's
approval mark, red cross everywhere today, since nothing has been reviewed) and the
figure status in a set of its own (filled circle, every drawn value is in a committed
result file; half circle, a run is unfinished or a planned arm is absent; open circle,
idea only). None of the three means approved. ``GATE_STATUS_MARKS=0`` leaves them out.

Writes the sheets to ``$ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/`` (PNG and
true-size SVG, timestamped), the manifest ``results/figure_gate_manifest.json``, the
caption numbers as ``results/figure_gate_numbers.json`` and as LaTeX macros in
``notes-tex/wet-lab/env-case-study-gate/tables/gate_numbers.tex`` and
``gate_figure_names.tex``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/040-inhibitor-synergy-wetlab/scripts/figure_gate_panels.py

The 033, 035 and 038 result files are read by ABSOLUTE path out of their own worktrees,
because those branches are not landed; every such path is recorded in the manifest under
``external_sources`` so the sheet can be rebuilt from the same bytes.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import IDENTITY_GAP_SMILES  # noqa: E402
from mixture_rules import monotone_alive  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
apply_paper_style()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
SLUG = "040-inhibitor-synergy-wetlab"
R040 = osp.join(EXPERIMENT_ROOT, SLUG, "results")
REPO = osp.dirname(EXPERIMENT_ROOT)
DOC_TABLES = osp.join(REPO, "notes-tex", "wet-lab", "env-case-study-gate", "tables")

#: Unlanded sibling worktrees. Absolute, recorded in the manifest, never copied. A
#: worktree sits at ``<worktrees>/<group>/<branch>``, so the worktrees root is two
#: levels above this repository's root.
WT = osp.dirname(osp.dirname(REPO))
R033 = osp.join(
    WT, "exp/033-env-chemgen-pooled/experiments/033-env-chemgen-pooled/results"
)
R035 = osp.join(
    WT,
    "exp/035-env-chemgen-vanacloig-cgt/experiments/035-env-chemgen-vanacloig-cgt/results",
)
R038 = osp.join(
    WT,
    "exp/038-env-chemgen-vanacloig-cgt-corrected/experiments/"
    "038-env-chemgen-vanacloig-cgt-corrected/results",
)

STATUS_MARKS = os.environ.get("GATE_STATUS_MARKS", "1") == "1"

# ------------------------------------------------------------------ the two channels

#: COLOR: where a predicted value comes from. The only color channel on every sheet.
SRC_COLOR = {
    "measured": PLOT_PALETTE[0],
    "bliss": PLOT_PALETTE[1],
    "loewe": PLOT_PALETTE[2],
    "hsa": PLOT_PALETTE[3],
    "mean": PLOT_PALETTE[6],
    "ridge": PLOT_PALETTE[4],
    "cgt": PLOT_PALETTE[10],
    "control": PLOT_PALETTE[5],
}
SRC_FILL = {
    "measured": PLOT_PALETTE_FILL[0],
    "bliss": PLOT_PALETTE_FILL[1],
    "loewe": PLOT_PALETTE_FILL[2],
    "hsa": PLOT_PALETTE_FILL[3],
    "mean": PLOT_PALETTE_FILL[6],
    "ridge": PLOT_PALETTE_FILL[4],
    "cgt": PLOT_PALETTE_FILL[10],
    "control": PLOT_PALETTE_FILL[5],
}
SRC_LABEL = {
    "measured": "a measured value (the public screens, the Bioscreen wells)",
    "bliss": "model-free rule: Bliss independence (the product of the singles)",
    "loewe": "model-free rule: Loewe additivity (the dose-equivalent sum)",
    "hsa": "model-free rule: the highest single agent",
    "mean": "model-free rule: the mean of the single-compound profiles",
    "ridge": "learned on chemistry features (ridge, kernel ridge, nearest neighbor)",
    "cgt": "learned by the cell graph transformer",
    "control": "gene-mean or permutation control, and chance",
}
SRC_ORDER = ["measured", "bliss", "loewe", "hsa", "mean", "ridge", "cgt", "control"]
#: the arm drawn as THE model on the wet-lab panels (single source, held-out protocol,
#: the CELL readout: the host is the empty genotype read through the shared head,
#: trained on the anchors and the ex21 titration) and its zero-shot twin (anchors
#: only, so its ex21 curve is a prediction of singles it never saw)
MODEL_ARM = "r9_van_e150_cell_bliss"
ZERO_SHOT_ARM = "r9_van_e150_cell_anchors_bliss"
#: HATCH: the software growth call, where a panel draws both calls. Plain is served.
SOFTWARE_HATCH = "////"

GRAY = PLOT_PALETTE[5]
SLATE = PLOT_PALETTE[16]
BLUE = PLOT_PALETTE[4]
RED = PLOT_PALETTE[1]
#: Heatmaps: white to palette gray for an unsigned quantity, palette blue through white
#: to palette red for a signed one. No amber or yellow colormap anywhere.
CMAP_PLAIN = LinearSegmentedColormap.from_list("plain", ["#FFFFFF", SLATE], N=256)
CMAP_SIGNED = LinearSegmentedColormap.from_list("signed", [BLUE, "#FFFFFF", RED], N=256)
#: observed fitness on the isobole grids: white at none, the measured color at full
CMAP_FITNESS = LinearSegmentedColormap.from_list(
    "fitness", ["#FFFFFF", SRC_COLOR["measured"]], N=256
)

#: Figure status symbols, a set of their own so they cannot be read as the section
#: status glyphs (check, square, cross), which are the author's approval marks.
STATUS_SYMBOL = {"ready": "●", "partial": "◐", "todo": "○"}
STATUS_WORD = {"ready": "on file", "partial": "partial", "todo": "idea only"}
SECTION_SYMBOL = {
    "todo": ("✗", "#E53935"),
    "tent": ("■", "#E0A020"),
    "final": ("✓", "#43A047"),
}
SECTION_WORD = {
    "todo": "not done, agents may edit freely",
    "tent": "author-reviewed",
    "final": "final",
}
#: Every panel's section status today: nothing has been reviewed.
SECTION_STATUS = "todo"

#: The six inhibitors, in the order the wet-lab runs name them.
INHIB = ["FF", "AA", "HMF", "FA", "LVA", "LA"]
INHIB_NAME = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-HMF",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
INHIB_LONG = {
    "FF": "furfural",
    "AA": "acetic acid",
    "HMF": "5-(hydroxymethyl)furfural",
    "FA": "formic acid",
    "LVA": "levulinic acid",
    "LA": "lactic acid",
}
LONG_TO_ABBR = {
    "furfural": "FF",
    "acetic acid": "AA",
    "5-(hydroxymethyl)furfural": "HMF",
    "formic acid": "FA",
    "levulinic acid": "LVA",
    "lactic acid": "LA",
}
RULE_SRC = {
    "bliss_ex21": "bliss",
    "bliss_ex23": "bliss",
    "loewe_ex21": "loewe",
    "hsa_ex21": "hsa",
    "hsa_ex23": "hsa",
}
RULE_LABEL = {
    "bliss_ex21": "Bliss, ex21 fits",
    "bliss_ex23": "Bliss, ex23 singles",
    "loewe_ex21": "Loewe, ex21 fits",
    "hsa_ex21": "highest single, ex21",
    "hsa_ex23": "highest single, ex23",
}

DIGIT_WORDS = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
]

# --------------------------------------------------------------------------- numbers

NUM: dict[str, Any] = {}


def num(key: str, value: Any) -> Any:
    """Record one caption number and return it unchanged."""
    NUM[key] = value
    return value


def macro_name(key: str) -> str:
    """A LaTeX macro name: letters only, digits spelled out."""
    return "".join(
        DIGIT_WORDS[int(ch)] if ch.isdigit() else ch for ch in key if ch.isalnum()
    )


def fmt(v: Any) -> str:
    """Format one recorded value the way a caption prints it."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if abs(v) >= 1 and v.is_integer():
            return f"{int(v):,}"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


# --------------------------------------------------------------------------- readers


class Reads:
    """Every result file the sheets draw from, opened once."""

    def __init__(self) -> None:
        """Open every result file and record the paths read."""
        self.paths: dict[str, str] = {}
        self.mix_scores = self.csv(osp.join(R040, "mixture_scores.csv"))
        self.mix_comb = self.csv(osp.join(R040, "mixture_combinations.csv"))
        self.pair_dev = self.csv(osp.join(R040, "pair_deviation.csv"))
        self.isobole_cells = self.csv(osp.join(R040, "isobole_cells.csv"))
        self.isobole_sum = self.csv(osp.join(R040, "isobole_summary.csv"))
        self.fits = self.csv(osp.join(R040, "single_agent_fits.csv"))
        self.boot = self.csv(osp.join(R040, "single_agent_bootstrap.csv"))
        self.ic30 = self.csv(osp.join(R040, "vanacloig_ic30.csv"))
        self.wells = self.csv(osp.join(R040, "wetlab_wells.csv"))
        self.call_check = self.csv(osp.join(R040, "wetlab_growth_call_check.csv"))
        self.het_rule = self.csv(osp.join(R040, "het_rule_fit.csv"))
        self.het_em = self.csv(osp.join(R040, "het_emergent_masked.csv"))
        self.het_rep = self.csv(osp.join(R040, "het_near_replicates.csv"))
        self.prof_pm = self.csv(osp.join(R040, "profile_predicted_vs_measured.csv"))
        self.prof_sim = self.csv(osp.join(R040, "profile_similarity.csv"))
        self.go = self.csv(osp.join(R040, "go_overlap_measured_vs_predicted.csv"))
        self.nom = self.csv(osp.join(R040, "nominations_summary.csv"))
        self.svd = self.csv(osp.join(R040, "similarity_vs_deviation.csv"))
        self.svd_pairs = self.csv(osp.join(R040, "similarity_vs_deviation_pairs.csv"))
        self.het_sum = self.json(osp.join(R040, "het_summary.json"))
        self.prov = self.json(osp.join(R040, "wetlab_provenance.json"))
        self.pool = self.csv(osp.join(R033, "pool_totals.csv"))
        self.axes = self.csv(osp.join(R033, "store_axes.csv"))
        self.curve = self.csv(osp.join(R035, "learning_curve", "ridge_curve.csv"))
        self.ladder = self.csv(osp.join(R038, "ladder", "ladder_r2_summary.csv"))
        self.ladder_scores = self.csv(osp.join(R038, "ladder", "ladder_r2_scores.csv"))
        self.bil = self.folds(osp.join(R038, "factorized", "r9_control"))
        self.envenc = self.folds(osp.join(R038, "factorized", "r10_envenc"))
        self.boundary = self.csv(osp.join(R040, "isobole_boundary.csv"))
        self.rc_pairs = self.csv(osp.join(R040, "rule_choice_pairs.csv"))
        self.rc_lopo = self.csv(osp.join(R040, "rule_choice_lopo.csv"))
        self.rc_orders = self.csv(osp.join(R040, "rule_choice_higher_orders.csv"))
        self.arms = {
            osp.basename(d): self.folds(d)
            for d in sorted(glob.glob(osp.join(R040, "mixture", "r1_*")))
        }
        self.rounds = self.round_scores()
        self.basis = self.csv(osp.join(R040, "source_basis.csv"))
        #: every arm's host-head prediction on every Bioscreen medium
        #: (r3_host_predictions.py); the single-source held-out-protocol arm is the
        #: model drawn on the wet-lab panels, the anchors-only arm its zero-shot twin
        self.host_pred = self.csv(osp.join(R040, "host_predictions.csv"))
        #: 038 round 17, the environment encoder's learning curve in fitted compounds
        self.curve_enc = self.folds(osp.join(R038, "factorized", "r17_curve"))
        self.cgt_compounds = self.compound_scores()
        self.convergence = self.csv(osp.join(R040, "r3_convergence_arms.csv"))
        #: what one four-source run of the mixture trainer read, per source: the cells,
        #: genes, compounds and conditions it trains on, and the host records by split
        #: (``train_mixture.py`` writes ``<name>_counts.csv`` beside every run's scores;
        #: the counts are the whole source and do not depend on the fold)
        self.counts = self.csv(
            sorted(
                glob.glob(
                    osp.join(
                        R040,
                        "mixture-delta",
                        "r3_vanhophetwil_e150_p0",
                        "*_f0_counts.csv",
                    )
                )
            )[0]
        )
        #: the curated-identity SMILES of every profiled compound; formic and lactic
        #: acid carry an identity gap and are the trainer's own SMILES
        self.smiles = {
            r.compound: r.smiles
            for r in self.csv(
                osp.join(R040, "profile_fingerprint_check.csv")
            ).itertuples()
            if isinstance(r.smiles, str)
        }
        self.smiles.update(IDENTITY_GAP_SMILES)

    def compound_scores(self) -> pd.DataFrame:
        """Every held-out compound's score from every round-3 and round-4 run."""
        frames = []
        for machine, sub in (("gilahyper", "mixture"), ("delta", "mixture-delta")):
            for path in sorted(
                glob.glob(osp.join(R040, sub, "r[34]*", "*_vanacloig_compounds.csv"))
            ):
                if "smoke" in path:
                    continue
                frames.append(self.csv(path).assign(machine=machine))
        out = pd.concat(frames, ignore_index=True)
        return out[out.member.str.startswith("seed")]

    def host_mean(self, sweep: str) -> pd.DataFrame:
        """One arm's host prediction per medium, mean over its folds."""
        d = self.host_pred[self.host_pred.sweep == sweep].copy()
        assert len(d), f"no host predictions for {sweep}"
        # the control medium has no compounds, which the CSV holds as an empty cell
        for column in ("compounds", "mM", "g_per_l", "wells"):
            d[column] = d[column].fillna("").astype(str)
        keep = [
            "split",
            "run",
            "compounds",
            "n_compounds",
            "mM",
            "g_per_l",
            "observed",
            "observed_grown",
            "grew",
            "trained_on",
        ]
        out = d.groupby("key").agg(
            {**{k: "first" for k in keep}, "predicted": "mean", "fold": "nunique"}
        )
        return out.rename(columns={"fold": "folds"})

    def round_scores(self) -> pd.DataFrame:
        """Every round-3 and round-4 score row, on this machine and pulled from Delta.

        ``results/mixture/`` holds the GilaHyper runs, ``results/mixture-delta/`` the
        Delta runs' tables; the same config at the same seed was run on both for
        twenty-two arms, and the ``machine`` column is what tells them apart. ``arm``
        is the config name without its fold and seed suffixes.
        """
        frames = []
        for machine, sub in (("gilahyper", "mixture"), ("delta", "mixture-delta")):
            for d in sorted(glob.glob(osp.join(R040, sub, "r[34]*"))):
                if "smoke" in d:
                    continue
                t = self.folds(d)
                if t.empty:
                    continue
                frames.append(t.assign(machine=machine, sweep=osp.basename(d)))
        out = pd.concat(frames, ignore_index=True)
        out["arm"] = (
            out["name"]
            .str.replace(r"_f\d+$", "", regex=True)
            .str.replace(r"_s\d+$", "", regex=True)
        )
        return out

    def csv(self, path: str) -> pd.DataFrame:
        """Read one CSV and record its path."""
        self.paths[osp.relpath(path, REPO) if path.startswith(REPO) else path] = path
        return pd.read_csv(path)

    def json(self, path: str) -> dict[str, Any]:
        """Read one JSON and record its path."""
        self.paths[osp.relpath(path, REPO) if path.startswith(REPO) else path] = path
        with open(path) as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    def folds(self, directory: str) -> pd.DataFrame:
        """Every ``*_scores.csv`` of one factorized round, empty when the round has none."""
        files = sorted(glob.glob(osp.join(directory, "*_scores.csv")))
        if not files:
            return pd.DataFrame(columns=["compound", "target", "spearman", "fold"])
        return pd.concat([self.csv(f) for f in files], ignore_index=True)

    def score(self, call: str, subset: str, rule: str) -> pd.Series:
        """One row of ``mixture_scores.csv``."""
        d = self.mix_scores
        hit = d[(d.call == call) & (d.subset.astype(str) == subset) & (d.rule == rule)]
        return hit.iloc[0]

    def fit(self, source: str, inhibitor: str, variant: str = "zero") -> pd.Series:
        """One row of ``single_agent_fits.csv``."""
        d = self.fits
        hit = d[
            (d.source == source) & (d.inhibitor == inhibitor) & (d.variant == variant)
        ]
        return hit.iloc[0]

    def ladder_row(self, model: str, kernel: str) -> pd.Series:
        """One centered-target row of the 038 ladder summary at full compound count."""
        d = self.ladder
        hit = d[
            (d.model == model)
            & (d.kernel == kernel)
            & (d.target == "centered")
            & (d.compounds == d.compounds.max())
        ]
        return hit.iloc[0]


# ------------------------------------------------------------------ drawing helpers


def new_sheet(height_mm: float) -> Any:
    """A full-width sheet at a given height, constrained layout."""
    plt.rcParams["hatch.linewidth"] = 0.35
    return plt.figure(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(height_mm)),
        layout="constrained",
    )


def rows(fig: Any, n_rows: int, n_cols: int) -> list[Any]:
    """One subfigure per row, so a long tick label cannot set another row's margin."""
    subs = fig.subfigures(n_rows, 1)
    subs = [subs] if n_rows == 1 else list(subs)
    out: list[Any] = []
    for sf in subs:
        axes = sf.subplots(1, n_cols)
        out.extend(list(np.atleast_1d(axes)))
    return out


def save(fig: Any, name: str) -> dict[str, str]:
    """Write the PNG and the true-size SVG of one sheet."""
    out_dir = osp.join(ASSET_IMAGES_DIR, SLUG)
    os.makedirs(out_dir, exist_ok=True)
    stamp = timestamp()
    png = osp.join(out_dir, f"{name}_{stamp}.png")
    svg = osp.join(out_dir, f"{name}_{stamp}.svg")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    return {"name": f"{name}_{stamp}", "png": png, "svg": svg}


def title(ax: Any, text: str, legend_rows: int = 0) -> None:
    """Panel title, padded to clear a legend of ``legend_rows`` rows above the axes."""
    ax.set_title(text, pad=4.0 + 7.4 * legend_rows)


def panel_legend(
    ax: Any, handles: list[Any], ncol: int = 1, align: str = "left"
) -> int:
    """A panel's own legend, outside the data area; returns the rows it takes."""
    ax.legend(
        handles=handles,
        loc="lower left" if align == "left" else "lower right",
        bbox_to_anchor=(0.0, 1.01) if align == "left" else (1.0, 1.01),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.25,
        fontsize=4.6,
        ncol=ncol,
    )
    return int(np.ceil(len(handles) / ncol))


def sheet_legend(fig: Any, sources: list[str], hatch: bool, ncol: int = 3) -> None:
    """The sheet's one legend: color is the source, hatch is the growth call."""
    handles = [
        Patch(
            facecolor=SRC_COLOR[s], edgecolor="black", linewidth=0.4, label=SRC_LABEL[s]
        )
        for s in SRC_ORDER
        if s in sources
    ]
    if hatch:
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=0.4,
                hatch=SOFTWARE_HATCH,
                label="hatched: the software growth call (plain: the served call, primary)",
            )
        )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        handleheight=0.9,
        columnspacing=1.2,
        labelspacing=0.3,
        fontsize=5.2,
    )


def letters_status(fig: Any, axes: list[Any], statuses: list[str]) -> None:
    """Bold lowercase panel letters, each followed by its two status symbols."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tops = {id(ax): ax.get_tightbbox(renderer).y1 for ax in axes}
    near = 0.04 * fig.bbox.height
    for ax, ch, status in zip(axes, "abcdefghijklmnop", statuses):
        box_ = ax.get_tightbbox(renderer)
        same = [
            tops[id(o)]
            for o in axes
            if o.get_figure() is ax.get_figure()
            and abs(tops[id(o)] - tops[id(ax)]) < near
        ]
        x0, y1 = inv.transform((box_.x0, max(same)))
        t = fig.text(
            x0, y1 + 0.003, ch, fontsize=8, fontweight="bold", ha="left", va="bottom"
        )
        t.set_in_layout(False)
        if not STATUS_MARKS:
            continue
        glyph, color = SECTION_SYMBOL[SECTION_STATUS]
        g1 = fig.text(
            x0 + 0.013,
            y1 + 0.004,
            glyph,
            fontsize=5,
            color=color,
            ha="left",
            va="bottom",
            family="DejaVu Sans",
        )
        g1.set_in_layout(False)
        g2 = fig.text(
            x0 + 0.022,
            y1 + 0.004,
            STATUS_SYMBOL[status],
            fontsize=5,
            color=GRAY,
            ha="left",
            va="bottom",
            family="DejaVu Sans",
        )
        g2.set_in_layout(False)


def status_key(fig: Any) -> None:
    """No key on the sheet: the document's status legend on its first page (the
    tcstatuslegend macro in tcdoc.sty) is the one key for the per-panel marks
    (owner, 2026-10-13); a sheet carries the marks only.
    """
    return None


def hgrid(ax: Any, step: float) -> None:
    """Gridlines every ``step`` on a horizontal metric axis, minor at a half step."""
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="x", which="both", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)


def vgrid(ax: Any, step: float) -> None:
    """The same rule on a vertical metric axis."""
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)


def bars(
    ax: Any,
    data: list[tuple[str, Any, Any, str, bool]],
    xlabel: str,
    xmax: float,
    step: float = 0.1,
    xmin: float | None = None,
) -> None:
    """Horizontal bars, first row on top.

    Each row is (label, value, error or None, source key, software call). A value that
    is NaN is a planned arm: its label is drawn and no bar, which is how a sheet says
    "this has not been run" without leaving a hole in the axis.
    """
    y = np.arange(len(data))[::-1]
    for yi, (_label, v, e, src, soft) in zip(y, data):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.barh(
            yi,
            v,
            xerr=e if e else None,
            height=0.72,
            color=SRC_COLOR[src],
            hatch=SOFTWARE_HATCH if soft else None,
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.5, "capsize": 1.2, "ecolor": "black"},
            zorder=2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in data])
    ax.set_xlabel(xlabel)
    vals = [r[1] for r in data if r[1] is not None and not np.isnan(r[1])]
    lo = min(0.0, min(vals) - 0.02) if xmin is None else xmin
    ax.set_xlim(lo, xmax)
    if lo < 0:
        ax.axvline(0.0, color="black", linewidth=0.4, zorder=3)
    ax.set_ylim(-0.6, len(data) - 0.4)
    hgrid(ax, step)


def box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str,
    fill: str,
    fs: float = 4.8,
    lw: float = 0.6,
    dashed: bool = False,
) -> None:
    """One rounded box of a concept sheet, its text centered and black."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=fill,
            edgecolor=color,
            linewidth=lw,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        zorder=3,
        linespacing=1.2,
    )


def arrow(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    """One arrow of a concept sheet."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=5,
            linewidth=0.6,
            color="black",
            zorder=4,
        )
    )


def blank(ax: Any, w: float, h: float) -> None:
    """A bare drawing surface with no spines and no ticks."""
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def text_table(
    ax: Any,
    columns: list[str],
    table: list[list[str]],
    widths: list[float],
    fs: float = 4.4,
) -> None:
    """A table drawn as a panel: header rule, one rule under the header, no grid."""
    n_rows = len(table)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    xs = np.concatenate([[0.0], np.cumsum(widths)])
    y_head = n_rows + 0.55
    for j, col in enumerate(columns):
        ax.text(
            xs[j],
            y_head,
            col,
            fontsize=fs,
            ha="left",
            va="center",
            style="italic",
            color="black",
        )
    ax.plot([0, 1], [n_rows + 1.15] * 2, color="black", linewidth=0.5)
    ax.plot([0, 1], [n_rows + 0.15] * 2, color="black", linewidth=0.5)
    for i, row in enumerate(table):
        yy = n_rows - 1 - i + 0.5
        for j, cell in enumerate(row):
            ax.text(xs[j], yy, cell, fontsize=fs, ha="left", va="center", color="black")
    ax.plot([0, 1], [-0.35] * 2, color="black", linewidth=0.5)


def hill(dose: np.ndarray, top: float, ic50: float, h: float) -> np.ndarray:
    """The Hill curve the single-agent fits store, as fitness against dose."""
    return top / (1.0 + (dose / ic50) ** h)


# --------------------------------------------------------------- derived quantities


def wet_counts(R: Reads) -> dict[str, Any]:
    """Counts of the private Bioscreen table, from ``wetlab_wells.csv``."""
    w = R.wells
    comb = R.mix_comb[R.mix_comb.call == "served"]
    out = {
        "n_wells": int(len(w)),
        "n_runs": int(w.run.nunique()),
        "n_ex21": int((w.run == "ex21").sum()),
        "n_ex23": int((w.run == "ex23").sum()),
        "n_isobole_wells": int(w.run.isin(["ex26", "ex27", "ex28"]).sum()),
        "n_combinations": int((comb.n_compounds >= 1).sum()),
        "n_grew_served": int(comb.grew.sum()),
    }
    soft = R.mix_comb[R.mix_comb.call == "software"]
    out["n_grew_software"] = int(soft.grew.sum())
    return out


def ladder_table(R: Reads, n: int) -> pd.DataFrame:
    """The top ``n`` centered-target ladder arms at the full compound count."""
    d = R.ladder
    d = d[(d.target == "centered") & (d.compounds == d.compounds.max())]
    return d.sort_values("spearman_median", ascending=False).head(n)


# ----------------------------------------------------------------- the concept sheet


def fig_concept(R: Reads) -> dict[str, str]:
    """One sheet: the task, the two data sources, and the evaluation."""
    fig = new_sheet(104)
    ax = fig.subplots(1, 1)
    blank(ax, 200, 108)
    wc = wet_counts(R)
    pool = R.pool.iloc[0]
    n_pool_cells = num("nPoolCells", int(pool.cells))
    n_pool_cmp = num("nPoolCompounds", int(pool.compounds))
    n_pool_genes = num("nPoolGenes", int(pool.genes))
    n_screens = num("nScreens", int(len(R.axes)))
    n_molar = num("nPoolCellsWithMolar", int(pool.cells_with_molar))
    num("nWells", wc["n_wells"])
    num("nExTwoOneWells", wc["n_ex21"])
    num("nExTwoThreeWells", wc["n_ex23"])
    num("nIsoboleWells", wc["n_isobole_wells"])
    num("nCombinations", wc["n_combinations"])
    num("nGrewServed", wc["n_grew_served"])
    num("nGrewSoftware", wc["n_grew_software"])
    num("nInhibitors", len(INHIB))
    num("nIsoboles", int(len(R.isobole_sum)))
    loewe = R.score("served", "all", "loewe_ex21")
    bliss = R.score("served", "all", "bliss_ex23")

    ax.text(2, 105, "the task", fontsize=5.6, style="italic", ha="left", va="center")
    marks = [(105, 18, "todo")]
    box(
        ax,
        2,
        86,
        40,
        13,
        "a compound\nand its molar dose",
        SRC_COLOR["measured"],
        SRC_FILL["measured"],
    )
    box(
        ax,
        2,
        68,
        40,
        13,
        "the strain: a deletion,\nor the bAID host with none",
        SRC_COLOR["measured"],
        SRC_FILL["measured"],
    )
    arrow(ax, 42, 92.5, 54, 86)
    arrow(ax, 42, 74.5, 54, 80)
    box(
        ax,
        54,
        72,
        44,
        22,
        "cell graph transformer\none compound token in front of\nthe gene tokens; the genes attend to it",
        SRC_COLOR["cgt"],
        SRC_FILL["cgt"],
    )
    arrow(ax, 98, 88, 110, 92)
    arrow(ax, 98, 78, 110, 74)
    box(
        ax,
        110,
        85,
        46,
        14,
        "read at the deleted genes:\nthe deletion profile\n(the public screens)",
        SRC_COLOR["cgt"],
        "white",
    )
    box(
        ax,
        110,
        66,
        46,
        14,
        "read at the mean over genes:\nhost growth and fitness\n(the bAID wet-lab readout)",
        SRC_COLOR["cgt"],
        "white",
    )
    box(
        ax,
        162,
        66,
        36,
        33,
        "a mixture is a SET of\ncompound tokens, each\nmodulated by its own\nlog-molar dose\n(proposed, not run)",
        GRAY,
        "white",
        dashed=True,
    )

    ax.text(2, 60, "the data", fontsize=5.6, style="italic", ha="left", va="center")
    marks.append((60, 20, "ready"))
    box(
        ax,
        2,
        36,
        94,
        20,
        f"the pooled chemogenomic table: {n_screens} screens, "
        f"{n_pool_cells:,} cells,\n{n_pool_cmp:,} compounds over {n_pool_genes:,} genes; "
        f"{n_molar:,} cells carry a molar dose\n" + ", ".join(R.axes.dataset),
        SRC_COLOR["measured"],
        SRC_FILL["measured"],
    )
    box(
        ax,
        102,
        36,
        96,
        20,
        f"the private Bioscreen runs of 2021: {wc['n_wells']} wells, "
        f"{len(INHIB)} inhibitors, one strain\n"
        f"ex21 titrations ({wc['n_ex21']} wells), ex23 combinations "
        f"({wc['n_ex23']} wells, {wc['n_combinations']} of them),\n"
        f"{len(R.isobole_sum)} isobole grids ({wc['n_isobole_wells']} wells)",
        SRC_COLOR["measured"],
        SRC_FILL["measured"],
    )

    ax.text(
        2, 29, "the evaluation", fontsize=5.6, style="italic", ha="left", va="center"
    )
    marks.append((29, 29, "todo"))
    box(
        ax,
        2,
        8,
        60,
        17,
        "public screens: compound-cold folds,\nthe per-compound centered Spearman\n"
        "against each compound's reliability ceiling",
        SRC_COLOR["ridge"],
        SRC_FILL["ridge"],
    )
    box(
        ax,
        68,
        8,
        64,
        17,
        f"wet lab: growth over {wc['n_combinations']} combinations (AUROC)\n"
        f"and fitness over the {wc['n_grew_served']} that grew (Spearman),\n"
        f"against Bliss ({bliss.auroc:.2f}) and Loewe ({loewe.auroc:.2f})",
        SRC_COLOR["loewe"],
        SRC_FILL["loewe"],
    )
    box(
        ax,
        138,
        8,
        60,
        17,
        "the industrial reading:\nrank the worst combination\nscenarios of a hydrolysate",
        GRAY,
        "white",
        dashed=True,
    )
    arrow(ax, 62, 16.5, 68, 16.5)
    arrow(ax, 132, 16.5, 138, 16.5)
    concept_marks(ax, marks)
    title(ax, "")
    ax.set_title("")
    status_key(fig)
    return save(fig, "gate_concept")


def concept_marks(ax: Any, marks: list[tuple[float, float, str]]) -> None:
    """Status symbols after each concept header, and the key along the bottom."""
    if not STATUS_MARKS:
        return
    for yy, xx, status in marks:
        glyph, color = SECTION_SYMBOL[SECTION_STATUS]
        ax.text(
            xx,
            yy,
            glyph,
            fontsize=5,
            color=color,
            ha="left",
            va="center",
            family="DejaVu Sans",
        )
        ax.text(
            xx + 3,
            yy,
            STATUS_SYMBOL[status],
            fontsize=5,
            color=GRAY,
            ha="left",
            va="center",
            family="DejaVu Sans",
        )


# -------------------------------------------------------------------- the main sheet


def structure(ax: Any, smiles: str, cx: float, cy: float, width: float) -> None:
    """One skeletal structure as vector lines, centered on (cx, cy), ``width`` wide.

    RDKit lays the molecule out (its 2D coordinates, Kekulized so a furan ring draws
    with explicit double bonds); the bonds are matplotlib lines, so the structure stays
    vector in the SVG like every other mark, and the heteroatoms are text. Carbons are
    the vertices, as in a skeletal formula.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    Chem.Kekulize(mol, clearAromaticFlags=True)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    xy = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
            for i in range(mol.GetNumAtoms())
        ]
    )
    xy -= xy.mean(axis=0)
    span = max(np.ptp(xy[:, 0]), 1.0)
    scale = width / span
    xy = xy * scale + np.array([cx, cy])
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p, q = xy[i], xy[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="black", linewidth=0.5, zorder=3)
        if bond.GetBondTypeAsDouble() == 2.0:
            d = q - p
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * 0.13 * scale
            ring = next((r for r in rings if i in r and j in r), None)
            if ring is not None:
                # the second line goes inside the ring, toward its center, and is
                # shortened so it reads as a ring double bond
                center = xy[list(ring)].mean(axis=0)
                if np.dot(center - (p + q) / 2, n) < 0:
                    n = -n
                p2, q2 = p + n + 0.15 * d, q + n - 0.15 * d
            else:
                p2, q2 = p + n, q + n
            ax.plot(
                [p2[0], q2[0]], [p2[1], q2[1]], color="black", linewidth=0.5, zorder=3
            )
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            continue
        label = atom.GetSymbol() + ("H" if atom.GetTotalNumHs() == 1 else "")
        x, y = xy[atom.GetIdx()]
        ax.text(
            x,
            y,
            label,
            fontsize=4.4,
            ha="center",
            va="center",
            zorder=4,
            bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "linewidth": 0},
        )


#: The six inhibitors as the archive and the loaders name them, in the two families a
#: hydrolysate carries them in: the furan aldehydes from sugar dehydration, and the
#: weak acids (``INHIB`` order within each family).
INHIB_FAMILY = {
    "furan aldehydes": ["FF", "HMF"],
    "weak acids": ["AA", "FA", "LVA", "LA"],
}


def draw_molecules(ax: Any, R: Reads) -> None:
    """Panel a: the six inhibitors drawn, with the host's half-maximal dose under each."""
    blank(ax, 100, 50)
    fits = {ab: R.fit("ex21", ab) for ab in INHIB}
    # the two furans sit on the upper row with the pretreatment note beside them; the
    # four acids share the lower row. (center x, center y, drawn width)
    placements = {
        "FF": (13, 38, 13),
        "HMF": (38, 38, 19),
        "AA": (11, 13, 10),
        "FA": (34, 13, 8),
        "LVA": (60, 13, 18),
        "LA": (86, 13, 13),
    }
    for family, y_label in (("furan aldehydes", 49), ("weak acids", 24.5)):
        ax.text(
            1, y_label, family, fontsize=4.6, style="italic", ha="left", va="center"
        )
    for ab, (cx, cy, w) in placements.items():
        structure(ax, R.smiles[INHIB_LONG[ab]], cx, cy + 2.0, w)
        fit = fits[ab]
        ax.text(
            cx,
            cy - 6.5,
            f"{INHIB_NAME[ab]}\nIC50 {float(fit.ic50_g_per_l):.2g} g/L",
            fontsize=4.2,
            ha="center",
            va="top",
            linespacing=1.15,
        )
    ax.text(
        54,
        38,
        "dilute-acid pretreatment of\nlignocellulose: pentoses dehydrate\n"
        "to furfural, hexoses to 5-HMF;\n5-HMF opens to levulinic and\n"
        "formic acid; acetyl groups release\nacetic acid. The host meets\n"
        "them together, at once.",
        fontsize=4.0,
        ha="left",
        va="center",
        linespacing=1.2,
    )
    title(ax, "the six inhibitors, the host's IC50 under each")


#: Culture condition of each gene-level source and of the host readout, from the
#: loaders' sourced values: vanacloig2022.py ``AEROBICITY``/``STATIC_CULTURE``,
#: hoepfner2014.py ``shaking_rpm`` (550) and ``aerobicity``, hillenmeyer2008.py (the SOM
#: states no vessel, volume or shaking; aerobicity is the schema default),
#: wildenhain2015.py ``STATIC_INCUBATION_RPM`` (0) and ``aerobicity``, and
#: private_torchcell/bioscreen.py (aerobic liquid YPD at 30 C; the archive states no
#: shaking, ``SHAKING_GAP``; the Bioscreen C is run shaken, per the operator).
CULTURE = {
    "vanacloig": "anaerobic, static",
    "hoepfner_hop": "aerobic, 550 rpm",
    "hillenmeyer_het": "aeration not stated",
    "wildenhain": "aerobic, static",
    "host": "aerobic, shaken, YPD",
}
SOURCE_TITLE = {
    "vanacloig": "Vanacloig 2022, the task",
    "hoepfner_hop": "Hoepfner 2014 HOP",
    "hillenmeyer_het": "Hillenmeyer 2008 HET",
    "wildenhain": "Wildenhain 2015",
}
HOST_TITLE = {
    "host_anchor": "public IC30 anchors",
    "host_ex21": "ex21 single-agent titration",
    "host_ex23": "ex23 combinations",
    "host_isobole": "ex26 to ex28 isobole grids",
}


def draw_sources(ax: Any, R: Reads) -> None:
    """Panel b: what the mixture trainer reads, per source, with its culture condition."""
    c = R.counts.set_index("source")
    rows_: list[tuple[str, str, int, str, bool]] = []
    for src in ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"]:
        r = c.loc[src]
        # one short line per bar: the cell count and the culture; genes and
        # conditions per source are in the note's table, not on the sheet
        rows_.append(
            (
                SOURCE_TITLE[src],
                f"{int(r.n_cells):,} cells; {CULTURE[src]}",
                int(r.n_compounds),
                src,
                True,
            )
        )
    for split in ["host_anchor", "host_ex21", "host_ex23", "host_isobole"]:
        r = c.loc[split]
        unit = "conditions" if split == "host_anchor" else "wells"
        n = int(r.n_conditions) if split == "host_anchor" else int(r.n_cells)
        detail = f"{n:,} {unit}"
        trained = split in ("host_anchor", "host_ex21")
        rows_.append((HOST_TITLE[split], detail, int(r.n_compounds), split, trained))
    # the four gene-level sources on top, a gap, then the host records by split
    y = np.array([9, 8, 7, 6, 3, 2, 1, 0], dtype=float)
    for yi, (_label, detail, n_cmp, _src, trained) in zip(y, rows_):
        ax.barh(
            yi,
            n_cmp,
            height=0.7,
            color=SRC_COLOR["measured"] if trained else SRC_FILL["measured"],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        ax.text(
            n_cmp * 1.3,
            yi,
            detail,
            fontsize=5.0,
            ha="left",
            va="center",
            color="black",
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows_], fontsize=5.0)
    ax.set_ylim(-0.6, 9.6)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 3e6)
    ax.set_xlabel("compounds in the source")
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.text(
        0.85,
        4.4,
        "the host readout, " + CULTURE["host"],
        fontsize=5.0,
        style="italic",
        ha="left",
        va="center",
    )
    v = c.loc["vanacloig"]
    num("nVanacloigCompounds", int(v.n_compounds))
    num("nVanacloigCells", int(v.n_cells))
    num("nVanacloigGenes", int(v.n_genes))
    num(
        "nFourSourceCells",
        int(
            c.loc[
                ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"]
            ].n_cells.sum()
        ),
    )
    num("nHoepfnerCells", int(c.loc["hoepfner_hop"].n_cells))
    num("nHillenmeyerCells", int(c.loc["hillenmeyer_het"].n_cells))
    num("nWildenhainCells", int(c.loc["wildenhain"].n_cells))
    num("nAnchorsFour", int(c.loc["host_anchor"].n_conditions))
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="trained on",
            ),
            Patch(
                facecolor=SRC_FILL["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="held out, the wet-lab test",
            ),
        ],
        ncol=2,
    )
    title(ax, "what the trainer reads, and under what culture", n)


def draw_evaluation(ax: Any, R: Reads) -> None:
    """Panel c: the model in one strip, and how it is scored on each side."""
    blank(ax, 100, 46)
    wc = wet_counts(R)
    loewe = R.score("served", "all", "loewe_ex21")
    bliss = R.score("served", "all", "bliss_ex23")
    n_fit = int(R.ladder_scores.n_fit_compounds.max())
    num("nFitCompounds", n_fit)
    box(
        ax,
        1,
        33,
        28,
        12,
        "a compound token\nwith its\nlog-molar dose",
        SRC_COLOR["measured"],
        SRC_FILL["measured"],
        fs=4.0,
    )
    arrow(ax, 29, 39, 33, 39)
    box(
        ax,
        33,
        33,
        32,
        12,
        "cell graph\ntransformer: the token\nin front of the genes",
        SRC_COLOR["cgt"],
        SRC_FILL["cgt"],
        fs=4.0,
    )
    arrow(ax, 65, 41, 69, 42.5)
    arrow(ax, 65, 37, 69, 35.5)
    box(
        ax,
        69,
        39.5,
        30,
        5.5,
        "deleted genes: a profile",
        SRC_COLOR["cgt"],
        "white",
        fs=4.0,
    )
    box(
        ax, 69, 33, 30, 5.5, "gene mean: host growth", SRC_COLOR["cgt"], "white", fs=4.0
    )
    box(
        ax,
        1,
        17,
        98,
        13,
        f"public screens: compound-cold folds, {n_fit} compounds fitted,\n"
        "the held-out compound's centered Spearman against its ceiling;\n"
        "the bar is nested ridge on fingerprint counts",
        SRC_COLOR["ridge"],
        SRC_FILL["ridge"],
        fs=4.2,
    )
    box(
        ax,
        1,
        1,
        98,
        13,
        f"wet lab: growth over {wc['n_combinations']} combinations (AUROC) and fitness\n"
        f"over the {wc['n_grew_served']} that grew (Spearman), against the two rules;\n"
        f"the bars are Loewe at AUROC {loewe.auroc:.3f} and Bliss at Spearman "
        f"{bliss.spearman:.3f}",
        SRC_COLOR["loewe"],
        SRC_FILL["loewe"],
        fs=4.2,
    )
    title(ax, "the model, and how it is scored")


def draw_ladder(ax: Any, R: Reads) -> None:
    """Panel d: the 038 round-1 ladder, nested ridge on fingerprint counts first."""
    top = ladder_table(R, 10)
    data = []
    for _, r in top.iterrows():
        data.append(
            (f"{r.model}, {r.kernel}", float(r.spearman_median), None, "ridge", False)
        )
    ridge = R.ladder_row("krr", "linear:fcfp4_count")
    num("ladderRidge", float(ridge.spearman_median))
    num("ladderRidgeMean", float(ridge.spearman_mean))
    num("ladderNArms", int(len(R.ladder[(R.ladder.target == "centered")])))
    num("ladderCompounds", int(R.ladder.compounds.max()))
    num("ladderSecond", float(top.iloc[1].spearman_median))
    num("ladderTenth", float(top.iloc[-1].spearman_median))
    bars(ax, data, "median centered Spearman, compound-cold", 0.40, step=0.1)
    ax.tick_params(axis="y", labelsize=4.4)
    title(ax, "the ladder on the corrected store: the ten best arms")


def draw_cgt_vs_ridge(ax: Any, R: Reads) -> None:
    """Panel e: the transformer against ridge, as far as round 2 has run."""
    ridge = R.ladder_row("krr", "linear:fcfp4_count")
    data = [
        (
            "nested ridge, fingerprint counts",
            float(ridge.spearman_median),
            None,
            "ridge",
            False,
        )
    ]
    for frame, arm, key in (
        (R.bil, "bilinear control", "bilinear"),
        (R.envenc, "environment encoder", "envenc"),
    ):
        if not len(frame):
            data.append((f"{arm} (running)", float("nan"), None, "cgt", False))
            num(f"{key}Folds", 0)
            continue
        cen = frame[frame.target == "centered"]
        # One bar per MEMBER, over every compound-evaluation of the round, so the panel
        # compares like with like against the ridge bar above it (which is also a median
        # over all of them). Drawing one bar per fold, as this did while the round was
        # part run, shows fold-to-fold spread and invites reading the best fold as the
        # arm's score.
        for member in ("seed0", "ensemble"):
            g = cen[cen.member == member]
            if not len(g):
                continue
            label = f"{arm}, " + (
                "single seed" if member == "seed0" else "3-seed ensemble"
            )
            data.append((label, float(g.spearman.median()), None, "cgt", False))
        ens = cen[cen.member == "ensemble"]
        num(f"{key}Configs", int(cen.name.nunique()))
        num(f"{key}Evaluations", int(len(ens)))
        num(f"{key}Median", float(ens.spearman.median()))
        num(f"{key}Compounds", int(cen.compound.nunique()))
    # The mixture trainer's single-source arm (experiment 040), on its own basis: 28
    # compounds with a molar dose, the held-out protocol, folds 0 to 2 of fold seed 0,
    # the three-seed ensemble per fold; not the 96-evaluation basis of the rows above.
    mix = R.rounds[
        (R.rounds.arm == "mix_van_e150")
        & (R.rounds.machine == "gilahyper")
        & (R.rounds.fold_seed == 0)
        & (R.rounds.task == "vanacloig")
        & (R.rounds.subset == "test_compounds")
        & (R.rounds.metric == "centered_spearman_median")
    ]
    ens = mix[(mix.member == "ensemble") & (mix.fold <= 2)]
    if len(ens):
        num("mixVanEnsemble", float(ens.value.median()))
        num("mixVanEnsembleFolds", int(ens.fold.nunique()))
        data.append(
            (
                "mixture trainer, one source, 3-seed ensemble (28 compounds, 3 folds)",
                float(ens.value.median()),
                None,
                "cgt",
                False,
            )
        )
    # The paired read, which is the panel's actual claim: the encoder against its own
    # control on the same compound and fold.
    key_cols = ["compound", "fold", "fold_seed"]
    if len(R.bil) and len(R.envenc):
        a = R.bil[(R.bil.target == "centered") & (R.bil.member == "ensemble")]
        b = R.envenc[(R.envenc.target == "centered") & (R.envenc.member == "ensemble")]
        paired = a.merge(b, on=key_cols, suffixes=("_bil", "_enc"))
        delta = paired.spearman_enc - paired.spearman_bil
        num("pairedEvaluations", int(len(paired)))
        num("pairedMedian", float(delta.median()))
        num("pairedUp", int((delta > 0).sum()))
    bars(ax, data, "median centered Spearman, compound-cold", 0.40, step=0.1)
    ax.tick_params(axis="y", labelsize=4.4)
    title(ax, "which model predicts an unseen compound's profile")


def draw_curve(ax: Any, R: Reads, both_targets: bool = False) -> None:
    """Panel f: the learning curve in fitted compounds."""
    d = R.curve
    for target, src, style in (("centered", "ridge", "-"), ("raw", "ridge", "--")):
        if target == "raw" and not both_targets:
            continue
        g = d[d.target == target].copy()
        g["x"] = g.n_fit_compounds_min.astype(float)
        g = g.sort_values("x")
        ax.plot(
            g.x,
            g.ridge_spearman_median,
            style,
            color=SRC_COLOR[src],
            marker="o",
            markersize=2.0,
            linewidth=0.8,
            zorder=3,
            label=f"ridge, {target}",
        )
    g = d[d.target == "centered"].sort_values("n_fit_compounds_min")
    num("curveEight", float(g[g["size"] == "8"].ridge_spearman_median.iloc[0]))
    num("curveSixteen", float(g[g["size"] == "16"].ridge_spearman_median.iloc[0]))
    num("curveTwentyFour", float(g[g["size"] == "24"].ridge_spearman_median.iloc[0]))
    num("curvePool", float(g[g["size"] == "pool"].ridge_spearman_median.iloc[0]))
    num(
        "curvePerDoubling",
        float(
            (
                g[g["size"] == "24"].ridge_spearman_median.iloc[0]
                - g[g["size"] == "8"].ridge_spearman_median.iloc[0]
            )
            / 2.0
        ),
    )
    num("curveEvaluations", int(g.compound_evaluations.iloc[0]))
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 24, 32])
    ax.set_xticklabels(["8", "16", "24", "32"])
    ax.set_xlim(7, 40)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("compounds fitted")
    ax.set_ylabel("median centered Spearman")
    vgrid(ax, 0.1)
    handles = [
        Line2D(
            [],
            [],
            color=SRC_COLOR["ridge"],
            linewidth=0.8,
            marker="o",
            markersize=2.0,
            label="ridge on fingerprint counts",
        )
    ]
    if both_targets:
        handles.append(
            Line2D(
                [],
                [],
                color=SRC_COLOR["ridge"],
                linewidth=0.8,
                linestyle="--",
                label="the same, uncentered target",
            )
        )
    else:
        e = R.curve_enc
        e = e[(e.target == "centered") & e.member.str.startswith("seed")].copy()
        e["size"] = e.name.str.extract(r"_n(\d+)_")[0].astype(int)
        enc = e.groupby("size").spearman.median()
        ax.plot(
            enc.index.to_numpy(),
            enc.to_numpy(),
            color=SRC_COLOR["cgt"],
            linewidth=0.8,
            linestyle=":",
            marker="o",
            markersize=2.0,
            zorder=3,
        )
        for size, value in enc.items():
            num(f"curveEnc{macro_name(str(size))}", float(value))
        num("curveEncEvaluations", int(e.groupby("size").size().max()))
        handles.append(
            Line2D(
                [],
                [],
                color=SRC_COLOR["cgt"],
                linewidth=0.8,
                linestyle=":",
                marker="o",
                markersize=2.0,
                label="the environment encoder (038 round 17)",
            )
        )
    n = panel_legend(ax, handles, ncol=1)
    title(ax, "the learning curve in compounds", n)


def draw_dose(ax: Any, R: Reads) -> None:
    """Panel g: the host's ex21 potency against Vanacloig's screen potency."""
    ic = R.ic30.set_index("inhibitor")
    labels, ex21, lo, hi, van = [], [], [], [], []
    for ab in INHIB:
        fit = R.fit("ex21", ab)
        labels.append(INHIB_NAME[ab])
        ex21.append(float(fit.ic30_mM))
        lo.append(float(fit.ic30_mM) - float(fit.ic30_mM_lo))
        hi.append(float(fit.ic30_mM_hi) - float(fit.ic30_mM))
        v = (
            ic.vanacloig_ic30_mM.get(ab, float("nan"))
            if ab in ic.index
            else float("nan")
        )
        van.append(float(v) if pd.notna(v) else float("nan"))
        num(f"icThirty{macro_name(ab.capitalize())}", float(fit.ic30_mM))
    x = np.arange(len(labels))
    ax.bar(
        x - 0.19,
        ex21,
        width=0.36,
        yerr=[lo, hi],
        color=SRC_COLOR["measured"],
        edgecolor="black",
        linewidth=0.4,
        error_kw={"elinewidth": 0.5, "capsize": 1.0, "ecolor": "black"},
        zorder=2,
    )
    drawn = [i for i, v in enumerate(van) if not np.isnan(v)]
    ax.bar(
        x[drawn] + 0.19,
        [van[i] for i in drawn],
        width=0.36,
        color=SRC_FILL["measured"],
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    for ab in ("FF", "HMF"):
        row = ic.loc[ab]
        num(f"vanacloig{macro_name(ab.capitalize())}", float(row.vanacloig_ic30_mM))
        num(
            f"ratio{macro_name(ab.capitalize())}",
            float(row.ratio_bioscreen_over_vanacloig),
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=4.6)
    ax.set_ylabel("IC30 (mM)")
    ax.set_ylim(1, 1500)
    ax.grid(axis="y", which="major", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="the bAID host, ex21 Hill fit (aerobic YPD)",
            ),
            Patch(
                facecolor=SRC_FILL["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="Vanacloig deletion pool (anaerobic SYNH3)",
            ),
        ],
        ncol=1,
    )
    title(ax, "dose: the same compound in two media", n)


def draw_het_rule(ax: Any, R: Reads) -> None:
    """Panel h: how a pair's deletion profile composes from its singles, on HET."""
    d = R.het_rule
    rules = [("r2_mean", "mean"), ("r2_sum", "bliss"), ("r2_max", "hsa")]
    names = {
        "r2_mean": "mean of the singles",
        "r2_sum": "sum (Bliss on log2)",
        "r2_max": "highest single",
    }
    rng = np.random.default_rng(0)
    for i, (col, src) in enumerate(rules):
        vals = d[col].astype(float).to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=3.0,
            facecolor=SRC_COLOR[src],
            edgecolor="black",
            linewidth=0.2,
            zorder=3,
        )
        med = float(np.median(vals))
        ax.plot([i - 0.3, i + 0.3], [med] * 2, color="black", linewidth=0.9, zorder=4)
        num(f"het{macro_name(col)}", med)
    best = R.het_sum["all"]["best_rule_r2_counts"]
    num("hetMeanWins", int(best.get("mean", 0)))
    num("hetPairs", int(R.het_sum["n_pairs"]))
    num("hetLinA", float(R.het_sum["all"]["median"]["lin_a"]))
    num("hetLinB", float(R.het_sum["all"]["median"]["lin_b"]))
    num("hetSpearmanMean", float(R.het_sum["all"]["median"]["spearman_sum"]))
    num("hetCtlNull", float(R.het_sum["all"]["median_ctlnull_spearman_sum"]))
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(
        [names[c] for c, _ in rules], rotation=20, ha="right", fontsize=4.6
    )
    ax.set_ylim(-1.2, 1.0)
    ax.axhline(0.0, color="black", linewidth=0.4, zorder=1)
    ax.set_ylabel("R-squared, pair profile from its singles")
    vgrid(ax, 0.5)
    title(ax, "the composition rule on public pairs")


def draw_profiles(ax: Any, R: Reads) -> None:
    """Panel i: the predicted profile against the measured one, for the two in store."""
    d = R.prof_pm
    loco = d[(d.comparison == "loco_32") & (d.target == "centered")]
    data = []
    for name, key in (("furfural", "Ff"), ("5-(hydroxymethyl)furfural", "Hmf")):
        row = loco[loco.compound == name].iloc[0]
        short = INHIB_NAME[LONG_TO_ABBR[name]]
        data.append(
            (
                f"{short}: screen repeats itself",
                float(row.measured_reliability),
                None,
                "measured",
                False,
            )
        )
        data.append(
            (
                f"{short}: ridge, leave-one-compound-out",
                float(row.spearman),
                None,
                "ridge",
                False,
            )
        )
        data.append(
            (
                f"{short}: ridge on the 038 folds",
                float(row.spearman_038_folds_mean),
                None,
                "ridge",
                False,
            )
        )
        num(f"prof{key}Ridge", float(row.spearman))
        num(f"prof{key}Reliability", float(row.measured_reliability))
        num(f"prof{key}Folds", float(row.spearman_038_folds_mean))
    for name, key in (("furfural", "Ff"), ("5-(hydroxymethyl)furfural", "Hmf")):
        c = R.cgt_compounds
        c = c[(c.compound == name) & (c.target == "centered")]
        short = INHIB_NAME[LONG_TO_ABBR[name]]
        num(f"prof{key}Cgt", float(c.spearman.mean()))
        num(f"prof{key}CgtRuns", int(len(c)))
        data.append(
            (
                f"{short}: cell graph transformer, held out ({len(c)} runs)",
                float(c.spearman.mean()),
                float(c.spearman.std(ddof=1)),
                "cgt",
                False,
            )
        )
    num("profMedianLoco", float(loco.spearman.median()))
    num("profLocoCompounds", int(loco.compound.nunique()))
    # The GO count is for the two compounds this panel draws, not the maximum over
    # every compound in the file: acetic acid shares eleven terms and would otherwise
    # be read as furfural's.
    go = R.go[
        (R.go.target == "raw")
        & (R.go.compound.isin(["furfural", "5-(hydroxymethyl)furfural"]))
    ]
    shared = int(go.n_sig_shared.fillna(0).max())
    num("goSharedTermsStore", shared)
    num("goPredictedTermsStore", int(go.n_sig_predicted.fillna(0).max()))
    bars(ax, data, "centered Spearman against the measured", 0.9, step=0.2, xmin=-0.2)
    ax.tick_params(axis="y", labelsize=4.4)
    ax.text(
        0.98,
        0.5,
        "GO terms shared by the measured\nand the predicted profile, "
        f"either compound: {shared}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=4.2,
        zorder=5,
    )
    title(ax, "the predicted profile against the measured")


def draw_single_agents(ax: Any, R: Reads) -> None:
    """Panel j: the ex21 titrations with their Hill fits."""
    wells = R.wells
    ex21 = wells[wells.run == "ex21"]
    for ab in INHIB:
        fit = R.fit("ex21", ab)
        col = f"dose_g_per_l_{ab}"
        sub = ex21[(ex21[col] > 0) & (ex21.n_compounds == 1)]
        if len(sub):
            g = sub.groupby(col).fitness.mean()
            norm = g.index.to_numpy() / float(fit.ic50_g_per_l)
            ax.scatter(
                norm,
                g.to_numpy(),
                s=2.4,
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.2,
                zorder=3,
            )
        xs = np.logspace(-1.3, 1.1, 200)
        ax.plot(
            xs,
            hill(xs, float(fit.top), 1.0, float(fit.h)),
            color=SRC_COLOR["measured"],
            linewidth=0.5,
            alpha=0.85,
            zorder=2,
        )
    for sweep, style in ((ZERO_SHOT_ARM, ":"), (MODEL_ARM, "--")):
        h = R.host_mean(sweep)
        h = h[(h.run == "ex21") & (h.n_compounds == 1)]
        for ab in INHIB:
            name = INHIB_LONG[ab]
            sub = h[h.compounds == name].copy()
            if sub.empty:
                continue
            fit = R.fit("ex21", ab)
            sub["x"] = sub.g_per_l.astype(float) / float(fit.ic50_g_per_l)
            sub = sub.sort_values("x")
            ax.plot(
                sub.x.to_numpy(),
                sub.predicted.to_numpy(),
                color=SRC_COLOR["cgt"],
                linewidth=0.5,
                linestyle=style,
                zorder=2,
            )
        singles = h[h.n_compounds == 1]
        rho = float(
            pd.Series(singles.predicted.to_numpy()).corr(
                pd.Series(singles.observed.to_numpy()), method="spearman"
            )
        )
        tag = "ZeroShot" if sweep == ZERO_SHOT_ARM else "Fitted"
        num(f"ex21{tag}Spearman", rho)
        num(f"ex21{tag}Conditions", int(len(singles)))
    num("nEx21Fits", int((R.fits.source == "ex21").sum()))
    num("nBootstrapDraws", int(R.boot.draw.max()) + 1)
    num("hillMax", float(R.fits[R.fits.source == "ex21"].h.max()))
    num("hillMin", float(R.fits[R.fits.source == "ex21"].h.min()))
    ax.set_xscale("log")
    ax.set_xlim(0.05, 12)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("dose, in units of the compound's own IC50")
    ax.set_ylabel("fitness")
    vgrid(ax, 0.5)
    n = panel_legend(
        ax,
        [
            Line2D(
                [],
                [],
                color=SRC_COLOR["measured"],
                linewidth=0.5,
                label="Hill fit, one curve per inhibitor",
            ),
            Line2D(
                [],
                [],
                color=SRC_COLOR["cgt"],
                linewidth=0.5,
                linestyle=":",
                label="model, anchors only (never saw a single)",
            ),
            Line2D(
                [],
                [],
                color=SRC_COLOR["cgt"],
                linewidth=0.5,
                linestyle="--",
                label="model trained on these singles (a fit)",
            ),
        ],
        ncol=1,
    )
    title(ax, "the six single-agent titrations", n)


def draw_combinations(ax: Any, R: Reads) -> None:
    """Panel k: the combinations observed against Bliss and Loewe, by size."""
    comb = R.mix_comb[(R.mix_comb.call == "served") & (R.mix_comb.grew)].copy()
    # the model's prediction per combination, joined on the set of compounds
    h = R.host_mean(MODEL_ARM)
    h = h[h.run == "ex23"]
    by_set = {
        frozenset(c.split("|")): float(v) for c, v in zip(h.compounds, h.predicted)
    }
    comb["model"] = [
        by_set.get(frozenset(c.split("|")), float("nan")) for c in comb.combination
    ]
    sizes = sorted(comb.n_compounds.unique())
    width = 0.2
    x = np.arange(len(sizes))
    series = [
        ("observed_grown_mean", "measured"),
        ("bliss_ex23", "bliss"),
        ("loewe_ex21", "loewe"),
        ("model", "cgt"),
    ]
    all_ex23 = h[h.n_compounds >= 1]
    num(
        "modelAuroc",
        float(roc_auc_score(all_ex23.grew.astype(bool), all_ex23.predicted)),
    )
    num(
        "modelGrownSpearman",
        float(
            pd.Series(comb.model.to_numpy()).corr(
                pd.Series(comb.observed_grown_mean.to_numpy()), method="spearman"
            )
        ),
    )
    num("modelFolds", int(h.folds.iloc[0]))
    for j, (col, src) in enumerate(series):
        means = [float(comb[comb.n_compounds == s][col].mean()) for s in sizes]
        ax.bar(
            x + (j - 1.5) * width,
            means,
            width=width,
            color=SRC_COLOR[src],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
    loewe = R.score("served", "all", "loewe_ex21")
    bliss = R.score("served", "all", "bliss_ex23")
    num("loeweAuroc", float(loewe.auroc))
    num("loeweAccuracy", float(loewe.accuracy))
    num("loeweSpearman", float(loewe.spearman))
    num("loeweSignedError", float(loewe.mean_signed_error))
    num("blissAuroc", float(bliss.auroc))
    num("blissAccuracy", float(bliss.accuracy))
    num("blissSpearman", float(bliss.spearman))
    num("blissSignedError", float(bliss.mean_signed_error))
    hsa = R.score("served", "all", "hsa_ex21")
    num("hsaAuroc", float(hsa.auroc))
    num("hsaSpearman", float(hsa.spearman))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in sizes])
    ax.set_xlabel("inhibitors in the combination")
    ax.set_ylabel("fitness over the combinations that grew")
    ax.set_ylim(0, 1.18)
    vgrid(ax, 0.2)
    ax.text(
        0.98,
        0.95,
        f"growth AUROC: Loewe {loewe.auroc:.3f}, Bliss {bliss.auroc:.3f}, "
        f"model {NUM['modelAuroc']:.3f}\n"
        f"fitness Spearman: Bliss {bliss.spearman:.3f}, Loewe {loewe.spearman:.3f}, "
        f"model {NUM['modelGrownSpearman']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.2,
        zorder=5,
    )
    title(ax, "grown combinations: the two rules and the model")


def draw_isoboles(ax: Any, R: Reads) -> None:
    """Panel l: the three isobole grids as observed fitness, with each rule's front."""
    cells = R.isobole_cells
    runs = list(R.isobole_sum.run)
    # Each run has its OWN dose ladder. Indexing every run into one pooled ladder left
    # each grid as a few sparse stripes, which is what a shared axis costs here.
    gap = 1.6
    norm = Normalize(vmin=0.0, vmax=1.2)
    x0 = 0.0
    ny = 0
    for run in runs:
        sub = cells[cells.run == run]
        doses_x = sorted(sub.inhibitor_g_per_l.unique())
        doses_y = sorted(sub.acetic_acid_g_per_l.unique())
        nx, ny = len(doses_x), len(doses_y)
        for _, r in sub.iterrows():
            i = doses_x.index(r.inhibitor_g_per_l)
            j = doses_y.index(r.acetic_acid_g_per_l)
            ax.add_patch(
                plt.Rectangle(
                    (x0 + i, j),
                    1,
                    1,
                    facecolor=CMAP_FITNESS(norm(float(r.observed_mean))),
                    edgecolor="#CCCCCC",
                    linewidth=0.2,
                    zorder=2,
                )
            )
            if bool(r.grew_any_plate):
                ax.plot(
                    [x0 + i + 0.5],
                    [j + 0.5],
                    marker="o",
                    markersize=0.8,
                    color="black",
                    zorder=4,
                )
        ax.add_patch(
            plt.Rectangle(
                (x0, 0),
                nx,
                ny,
                facecolor="none",
                edgecolor="black",
                linewidth=0.9,
                zorder=6,
            )
        )
        srow = R.isobole_sum[R.isobole_sum.run == run].iloc[0]
        tau = float(srow.tau)
        # A rule's front is where its predicted fitness crosses the growth threshold,
        # not where it is merely positive: the rule predicts a fitness for every cell,
        # so a positive-value test puts the front on the grid's edge.
        for column, src, style in (("bliss", "bliss", "-"), ("loewe", "loewe", "-")):
            front = sub[sub[column] >= tau]
            if len(front):
                edge = front.groupby("acetic_acid_g_per_l").inhibitor_g_per_l.max()
                xs = [x0 + doses_x.index(v) + 1.0 for v in edge.to_numpy()]
                ys = [doses_y.index(v) + 0.5 for v in edge.index.to_numpy()]
                ax.plot(
                    xs,
                    ys,
                    color=SRC_COLOR[src],
                    linewidth=0.8,
                    linestyle=style,
                    zorder=5,
                )
        # The model's front: its predicted fitness per cell against the same growth
        # threshold, then the monotone closure (growth never returns as either dose
        # rises), so the front is read on the rules' own axis; the cells the closure
        # drops are the predicted surface's growth islands.
        hp = R.host_mean(MODEL_ARM)
        hp = hp[hp.run == run]
        grid = np.full((len(doses_y), len(doses_x)), np.nan)
        inhibitor = [p for p in srow["pair"].split("|") if p != "acetic acid"][0]
        for _, r in hp.iterrows():
            names = [n for n in r.compounds.split("|") if n]
            gl = [float(v) for v in r.g_per_l.split("|") if v]
            dose = dict(zip(names, gl, strict=True))
            xi, yi = dose.get(inhibitor, 0.0), dose.get("acetic acid", 0.0)
            if xi in doses_x and yi in doses_y:
                grid[doses_y.index(yi), doses_x.index(xi)] = r.predicted
        alive = np.where(np.isnan(grid), False, grid >= tau)
        closed = monotone_alive(alive)
        num(
            f"isobole{macro_name(run.capitalize())}ModelIslands",
            int(alive.sum() - closed.sum()),
        )
        num(
            f"isobole{macro_name(run.capitalize())}ModelCells",
            int(np.isfinite(grid).sum()),
        )
        mx = [
            x0 + int(np.max(np.where(closed[j])[0])) + 1.0
            for j in range(len(doses_y))
            if closed[j].any()
        ]
        my = [j + 0.5 for j in range(len(doses_y)) if closed[j].any()]
        if mx:
            ax.plot(
                mx, my, color=SRC_COLOR["cgt"], linewidth=0.8, linestyle=":", zorder=5
            )
        parts = [LONG_TO_ABBR.get(p, p) for p in srow["pair"].split("|")]
        label = " x ".join(INHIB_NAME.get(p, p) for p in reversed(parts))
        flag = "\n(flagged)" if isinstance(srow["flag"], str) and srow["flag"] else ""
        ax.text(
            x0 + nx / 2,
            ny + 0.4,
            f"{label}{flag}",
            fontsize=4.2,
            ha="center",
            va="bottom",
            color="black",
            linespacing=1.1,
        )
        key = macro_name(run.capitalize())
        num(f"isobole{key}Excess", float(srow.mean_excess_over_bliss))
        num(f"isobole{key}Lo", float(srow.mean_excess_lo))
        num(f"isobole{key}Hi", float(srow.mean_excess_hi))
        num(f"isobole{key}Cells", int(srow.n_interior_cells))
        num(f"isobole{key}Grew", int(srow.n_interior_grew))
        num(f"isobole{key}BlissGrowNone", int(srow.n_interior_bliss_grow_observed_none))
        num(f"isobole{key}LoeweGrowNone", int(srow.n_interior_loewe_grow_observed_none))
        x0 += nx + gap
    ax.set_xlim(-0.4, x0 - gap + 0.4)
    ax.set_ylim(-0.4, ny + 2.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("inhibitor dose increasing to the right, acetic acid upward")
    sm = plt.cm.ScalarMappable(cmap=CMAP_FITNESS, norm=norm)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.028, pad=0.015)
    cb.set_label("observed fitness", fontsize=4.6)
    cb.ax.tick_params(labelsize=4.4, length=1.5)
    cb.outline.set_linewidth(0.4)
    n = panel_legend(
        ax,
        [
            Line2D(
                [], [], color=SRC_COLOR["bliss"], linewidth=0.8, label="Bliss front"
            ),
            Line2D(
                [], [], color=SRC_COLOR["loewe"], linewidth=0.8, label="Loewe front"
            ),
            Line2D(
                [],
                [],
                color=SRC_COLOR["cgt"],
                linewidth=0.8,
                linestyle=":",
                label="model front (monotone closure)",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markersize=1.6,
                linestyle="none",
                label="a cell that grew",
            ),
        ],
        ncol=2,
    )
    title(ax, "the three isobole grids: observed fitness, each rule's front", n)


def fig_main(R: Reads) -> dict[str, str]:
    """The four-by-three sheet of the plan, a to l."""
    fig = new_sheet(170)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.962))
    axes = rows(fig, 4, 3)
    draw_molecules(axes[0], R)
    draw_sources(axes[1], R)
    draw_evaluation(axes[2], R)
    draw_ladder(axes[3], R)
    draw_cgt_vs_ridge(axes[4], R)
    draw_curve(axes[5], R)
    draw_dose(axes[6], R)
    draw_het_rule(axes[7], R)
    draw_profiles(axes[8], R)
    draw_single_agents(axes[9], R)
    draw_combinations(axes[10], R)
    draw_isoboles(axes[11], R)
    sheet_legend(
        fig,
        ["measured", "bliss", "loewe", "hsa", "mean", "ridge", "cgt"],
        hatch=False,
        ncol=2,
    )
    letters_status(
        fig,
        axes,
        [
            # a concept, b data, c evaluation, d ladder, e the two heads (round 2 is
            # complete since 2026-10-10, so e is on file; the stack is not run and the
            # panel does not draw it), f curve (encoder arm not rerun), g dose, h the
            # composition rule, i profiles (the transformer's are not scored yet),
            # j to l the wet lab (the rules are on file, the model is running).
            "ready",
            "ready",
            "todo",
            "ready",
            "ready",
            "ready",
            "ready",
            "ready",
            "ready",
            "ready",
            "ready",
            "ready",
        ],
    )
    status_key(fig)
    return save(fig, "gate_main")


# ----------------------------------------------------------- supplement 1, benchmarks


def draw_ladder_detail(ax: Any, R: Reads) -> None:
    """Every ladder arm by encoder and kernel, as a heatmap of the median score."""
    d = R.ladder
    d = d[(d.target == "centered") & (d.compounds == d.compounds.max())].copy()
    d[["kind", "encoder"]] = d.kernel.str.split(":", n=1, expand=True)
    d["cell"] = d.model + ", " + d.kind
    piv = d.pivot_table(index="encoder", columns="cell", values="spearman_median")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    arr = piv.to_numpy(dtype=float)
    norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.add_patch(
                plt.Rectangle(
                    (j, arr.shape[0] - 1 - i),
                    1,
                    1,
                    facecolor=CMAP_PLAIN(norm(v)),
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=2,
                )
            )
            ax.text(
                j + 0.5,
                arr.shape[0] - 1 - i + 0.5,
                f"{v:.2f}",
                fontsize=3.6,
                ha="center",
                va="center",
                color="black",
                zorder=3,
            )
    ax.set_xlim(0, arr.shape[1])
    ax.set_ylim(0, arr.shape[0])
    ax.set_xticks([j + 0.5 for j in range(arr.shape[1])])
    ax.set_xticklabels(list(piv.columns), rotation=40, ha="right", fontsize=4.2)
    ax.set_yticks([i + 0.5 for i in range(arr.shape[0])])
    ax.set_yticklabels(list(piv.index)[::-1], fontsize=4.2)
    num("ladderEncoders", int(piv.shape[0]))
    num("ladderCells", int(np.isfinite(arr).sum()))
    title(ax, "the ladder in full: median centered Spearman per encoder and kernel")


def draw_per_compound(ax: Any, R: Reads) -> None:
    """Each compound's ridge score against its own reliability ceiling."""
    d = R.ladder_scores
    d = d[
        (d.model == "krr")
        & (d.kernel == "linear:fcfp4_count")
        & (d.target == "centered")
    ]
    g = (
        d.groupby("compound")
        .agg(spearman=("spearman", "mean"), ceiling=("ceiling", "first"))
        .reset_index()
    )
    ax.scatter(
        g.ceiling,
        g.spearman,
        s=3.0,
        facecolor=SRC_COLOR["ridge"],
        edgecolor="black",
        linewidth=0.2,
        zorder=3,
    )
    lims = [-0.2, 1.0]
    ax.plot(lims, lims, color="black", linewidth=0.4, linestyle="--", zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.4, zorder=1)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("the compound's reliability ceiling")
    ax.set_ylabel("ridge, centered Spearman")
    vgrid(ax, 0.2)
    hgrid(ax, 0.2)
    num("perCompoundN", int(len(g)))
    num("perCompoundMedian", float(g.spearman.median()))
    num("perCompoundAbove", int((g.spearman > g.ceiling).sum()))
    num("perCompoundNegative", int((g.spearman < 0).sum()))
    title(ax, "every compound, its score and its ceiling")


def draw_rule_table(ax: Any, R: Reads) -> None:
    """Loewe against Bliss by combination size, as the confusion the rules make."""
    d = R.mix_scores
    sizes = [
        s for s in ["1", "2", "3", "4", "5", "6"] if (d.subset.astype(str) == s).any()
    ]
    columns = [
        "size",
        "combinations",
        "grew",
        "Loewe: called, right, missed",
        "Bliss: called, right, false",
    ]
    table = []
    for s in sizes:
        lo = R.score("served", s, "loewe_ex21")
        bl = R.score("served", s, "bliss_ex23")
        table.append(
            [
                s,
                f"{int(lo.n_combinations)}",
                f"{int(lo.n_grew)}",
                f"{int(lo.tp + lo.fp)}, {int(lo.tp)}, {int(lo.fn)}",
                f"{int(bl.tp + bl.fp)}, {int(bl.tp)}, {int(bl.fp)}",
            ]
        )
        num(f"loeweMissed{macro_name(s)}", int(lo.fn))
        num(f"loeweGrew{macro_name(s)}", int(lo.n_grew))
        num(f"blissFalse{macro_name(s)}", int(bl.fp))
        num(f"blissDead{macro_name(s)}", int(bl.n_combinations - bl.n_grew))
    text_table(ax, columns, table, [0.07, 0.17, 0.11, 0.33, 0.32])
    title(ax, "what each rule gets wrong, by combination size")


def fig_si1(R: Reads) -> dict[str, str]:
    """Supplement 1: the public-benchmark detail behind d, e and f."""
    fig = new_sheet(150)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.94))
    axes = rows(fig, 2, 2)
    draw_ladder_detail(axes[0], R)
    draw_per_compound(axes[1], R)
    draw_curve(axes[2], R, both_targets=True)
    draw_rule_table(axes[3], R)
    sheet_legend(fig, ["ridge"], hatch=False, ncol=1)
    letters_status(fig, axes, ["ready", "ready", "ready", "ready"])
    status_key(fig)
    return save(fig, "gate_si1_benchmarks")


# --------------------------------------------------------------- supplement 2, growth


def draw_all_fits(ax: Any, R: Reads) -> None:
    """Every single-agent fit: each run, each inhibitor, both no-growth variants."""
    d = R.fits.copy()
    d["key"] = d.source + ", " + d.inhibitor.map(INHIB_NAME)
    order = sorted(d.key.unique())
    y = np.arange(len(order))[::-1]
    pos = dict(zip(order, y))
    for _, r in d.iterrows():
        yy = pos[r.key] + (0.18 if r.variant == "zero" else -0.18)
        ax.barh(
            yy,
            float(r.ic30_mM),
            height=0.32,
            xerr=[
                [float(r.ic30_mM) - float(r.ic30_mM_lo)],
                [float(r.ic30_mM_hi) - float(r.ic30_mM)],
            ],
            color=SRC_COLOR["measured"]
            if r.variant == "zero"
            else SRC_FILL["measured"],
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.4, "capsize": 0.8, "ecolor": "black"},
            zorder=2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=4.0)
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_xlabel("IC30 (mM)")
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    num("nFitsAll", int(len(d)))
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="no-growth wells kept at zero fitness",
            ),
            Patch(
                facecolor=SRC_FILL["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="no-growth wells excluded from the fit",
            ),
        ],
        ncol=1,
    )
    title(ax, "every single-agent fit, both no-growth conventions", n)


def draw_call_sensitivity(ax: Any, R: Reads) -> None:
    """Every claim-1 score under both growth calls."""
    rules = ["loewe_ex21", "bliss_ex23", "bliss_ex21", "hsa_ex23", "hsa_ex21"]
    data = []
    for rule in rules:
        for call in ("served", "software"):
            r = R.score(call, "all", rule)
            # Only the served row is labelled: the hatched row under it is the same
            # rule read under the other call, and repeating the name reads as two rules.
            data.append(
                (
                    RULE_LABEL[rule] if call == "served" else "",
                    float(r.auroc),
                    None,
                    RULE_SRC[rule],
                    call == "software",
                )
            )
    bars(ax, data, "growth AUROC over the combinations", 1.0, step=0.2)
    ax.tick_params(axis="y", labelsize=4.2)
    disputed = R.call_check
    num("nDisputedWells", int(len(disputed)))
    num("nDisputedCombinations", int(disputed.combination.nunique()))
    num("odRiseMin", float(disputed.od_rise.min()))
    num("odRiseMax", float(disputed.od_rise.max()))
    # A clear strip under the last bar, rather than over it: every bar here runs the
    # width of the panel, so there is no empty corner to annotate into.
    ax.set_ylim(-1.8, len(data) - 0.4)
    ax.text(
        0.0,
        -1.1,
        f"the two calls disagree on {len(disputed)} ex23 wells, "
        f"OD rise {disputed.od_rise.min():.3f} to {disputed.od_rise.max():.3f}",
        ha="left",
        va="center",
        fontsize=4.2,
        zorder=5,
    )
    title(ax, "the growth call as a sensitivity, every rule")


def draw_pair_deviation(ax: Any, R: Reads) -> None:
    """Each pair's departure from Bliss, with its bootstrap interval, under both calls."""
    d = R.pair_dev.copy()
    d = d.sort_values("obs_minus_bliss_ex23_served")
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        for off, call in ((0.18, "served"), (-0.18, "software")):
            v = float(r[f"obs_minus_bliss_ex23_{call}"])
            lo = v - float(r[f"obs_minus_bliss_ex23_lo_{call}"])
            hi = float(r[f"obs_minus_bliss_ex23_hi_{call}"]) - v
            ax.barh(
                yi + off,
                v,
                height=0.32,
                xerr=[[lo], [hi]],
                color=SRC_COLOR["bliss"],
                hatch=SOFTWARE_HATCH if call == "software" else None,
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.4, "capsize": 0.8, "ecolor": "black"},
                zorder=2,
            )
    labels = [
        p.replace("|", " + ").replace("5-(hydroxymethyl)furfural", "5-HMF")
        for p in d.pair
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=4.0)
    ax.axvline(0.0, color="black", linewidth=0.4, zorder=3)
    cols = [f"obs_minus_bliss_ex23_lo_{c}" for c in ("served", "software")]
    ax.set_xlim(float(d[cols].min().min()) - 0.03, 0.15)
    ax.set_xlabel("observed minus Bliss, fitness")
    hgrid(ax, 0.1)
    calls = d.call_bliss_ex23_served.value_counts()
    num("nPairs", int(len(d)))
    num("nPairsSynergy", int(calls.get("synergy", 0)))
    num("nPairsAdditive", int(calls.get("additive", 0)))
    num("nPairsAntagonism", int(calls.get("antagonism", 0)))
    title(ax, "each pair against Bliss, both calls")


def draw_extrapolation(ax: Any, R: Reads) -> None:
    """Where Loewe had to extrapolate past the measured dose range."""
    comb = R.mix_comb[R.mix_comb.call == "served"]
    sizes = sorted(comb.n_compounds.unique())
    columns = [
        "size",
        "combinations",
        "Loewe beyond the measured range",
        "components beyond it",
    ]
    table = []
    total = 0
    for s in sizes:
        sub = comb[comb.n_compounds == s]
        n_beyond = int(sub.loewe_beyond_measured_range.sum())
        total += n_beyond
        parts = sorted(
            {
                p.strip()
                for v in sub.loewe_components_beyond_range.dropna()
                for p in str(v).split("|")
                if p.strip()
            }
        )
        short = ", ".join(INHIB_NAME.get(LONG_TO_ABBR.get(p, p), p) for p in parts[:3])
        table.append([str(int(s)), str(len(sub)), str(n_beyond), short or "none"])
    num("loeweBeyondTotal", total)
    fronts = R.isobole_sum.loewe_front_beyond_grid.astype(str)
    num("isoboleFrontsBeyondGrid", int((fronts == "True").sum()))
    text_table(ax, columns, table, [0.08, 0.17, 0.33, 0.42])
    title(ax, "the Loewe extrapolation flags")


def fig_si2(R: Reads) -> dict[str, str]:
    """Supplement 2: the single-agent and growth-call detail behind j, k and l."""
    fig = new_sheet(154)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.94))
    axes = rows(fig, 2, 2)
    draw_all_fits(axes[0], R)
    draw_call_sensitivity(axes[1], R)
    draw_pair_deviation(axes[2], R)
    draw_extrapolation(axes[3], R)
    sheet_legend(fig, ["measured", "bliss", "loewe", "hsa"], hatch=True, ncol=2)
    letters_status(fig, axes, ["ready", "ready", "ready", "ready"])
    status_key(fig)
    return save(fig, "gate_si2_growth")


# ------------------------------------------------------------- supplement 3, profiles


def draw_emergent(ax: Any, R: Reads) -> None:
    """Emergent and masked gene rates against how alike the two singles are."""
    d = R.het_em
    # Both rates are measured quantities, so both take the measured color and the
    # lighter fill separates the second reading, as it does in every other panel.
    for col, face in (
        ("emergent_fraction", SRC_COLOR["measured"]),
        ("masked_fraction", SRC_FILL["measured"]),
    ):
        ax.scatter(
            d.single_similarity_spearman,
            d[col],
            s=3.0,
            facecolor=face,
            edgecolor="black",
            linewidth=0.2,
            zorder=3,
        )
    ax.scatter(
        d.single_similarity_spearman,
        d.ctlnull_emergent_mean,
        s=3.0,
        facecolor=SRC_COLOR["control"],
        edgecolor="black",
        linewidth=0.2,
        zorder=2,
        label="control-matched null",
    )
    rep = R.het_rep
    num("hetReplicateDisagreeMin", float(rep.b_hits_not_in_a_fraction.min()))
    num("hetReplicateDisagreeMax", float(rep.a_hits_not_in_b_fraction.max()))
    num("hetEmergentMedian", float(R.het_sum["all"]["median_emergent_fraction"]))
    num("hetMaskedMedian", float(R.het_sum["all"]["masked_fraction_pooled"]))
    num("hetNullEmergent", float(R.het_sum["all"]["median_null_emergent_fraction"]))
    ax.set_xlim(-0.1, 0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Spearman between the two single-compound profiles")
    ax.set_ylabel("fraction of hits")
    vgrid(ax, 0.2)
    hgrid(ax, 0.2)
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="emergent in the pair",
            ),
            Patch(
                facecolor=SRC_FILL["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="masked in the pair",
            ),
            Patch(
                facecolor=SRC_COLOR["control"],
                edgecolor="black",
                linewidth=0.4,
                label="the control-matched null",
            ),
        ],
        ncol=1,
    )
    title(ax, "emergent and masked genes, against the pair's own similarity", n)


def draw_mtx_grid(ax: Any, R: Reads) -> None:
    """The one dose grid HET holds: methotrexate against 5-fluorouracil."""
    d = R.het_rule[R.het_rule.family == "5FU x MTX"].copy()
    d["mtx"] = d.pair_label.str.extract(r"MTX(\d+)").astype(float)
    d["fu"] = d.pair_label.str.extract(r"5FU([\d.]+)").astype(float)
    mtx = sorted(d.mtx.unique())
    fu = sorted(d.fu.unique())
    norm = Normalize(vmin=0.0, vmax=1.0)
    for _, r in d.iterrows():
        i, j = fu.index(r.fu), mtx.index(r.mtx)
        ax.add_patch(
            plt.Rectangle(
                (i, j),
                1,
                1,
                facecolor=CMAP_PLAIN(norm(float(r.lin_a))),
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
        )
        ax.text(
            i + 0.5,
            j + 0.5,
            f"{float(r.lin_a):.2f}",
            fontsize=4.2,
            ha="center",
            va="center",
            color="black",
            zorder=3,
        )
    ax.set_xlim(0, len(fu))
    ax.set_ylim(0, len(mtx))
    ax.set_xticks([i + 0.5 for i in range(len(fu))])
    ax.set_xticklabels([f"{v:g}" for v in fu], fontsize=4.4)
    ax.set_yticks([j + 0.5 for j in range(len(mtx))])
    ax.set_yticklabels([f"{v:g}" for v in mtx], fontsize=4.4)
    ax.set_xlabel("5-fluorouracil (uM)")
    ax.set_ylabel("methotrexate (uM)")
    num("mtxCells", int(len(d)))
    num("mtxLinAMin", float(d.lin_a.min()))
    num("mtxLinAMax", float(d.lin_a.max()))
    num("mtxMeanWins", int((d.best_rule_r2 == "mean").sum()))
    title(ax, "the weight on the first single, across one dose grid")


def similarity_matrix(
    ax: Any, R: Reads, version: str, label: str, colorbar: bool = False
) -> None:
    """The six inhibitors' profile similarity, one version of the profile matrix."""
    d = R.prof_sim[R.prof_sim.version == version]
    n = len(INHIB)
    mat = np.full((n, n), np.nan)
    for _, r in d.iterrows():
        i = INHIB.index(r.abbreviation_a)
        j = INHIB.index(r.abbreviation_b)
        mat[i, j] = mat[j, i] = float(r.spearman_centered)
    norm = Normalize(vmin=-1.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(
                    plt.Rectangle(
                        (j, n - 1 - i),
                        1,
                        1,
                        facecolor="#EEEEEE",
                        edgecolor="white",
                        linewidth=0.3,
                        zorder=2,
                    )
                )
                continue
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.add_patch(
                plt.Rectangle(
                    (j, n - 1 - i),
                    1,
                    1,
                    facecolor=CMAP_SIGNED(norm(v)),
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=2,
                )
            )
            ax.text(
                j + 0.5,
                n - 1 - i + 0.5,
                f"{v:.2f}",
                fontsize=3.8,
                ha="center",
                va="center",
                color="black",
                zorder=3,
            )
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels(
        [INHIB_NAME[a] for a in INHIB], rotation=40, ha="right", fontsize=4.2
    )
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels([INHIB_NAME[a] for a in INHIB][::-1], fontsize=4.2)
    key = macro_name(version.replace("_", " ").title())
    finite = mat[np.isfinite(mat)]
    num(f"sim{key}Max", float(finite.max()))
    num(f"sim{key}Min", float(finite.min()))
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=CMAP_SIGNED, norm=norm)
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("centered Spearman, one scale for both matrices", fontsize=4.4)
        cb.ax.tick_params(labelsize=4.2, length=1.5)
        cb.outline.set_linewidth(0.4)
    title(ax, label)


def draw_similarity_vs_deviation(ax: Any, R: Reads) -> None:
    """Does profile similarity anticipate the pair's departure from Bliss."""
    d = R.svd_pairs
    for version, src, marker in (
        ("best_available", "measured", "o"),
        ("all_predicted", "ridge", "o"),
    ):
        g = d[d.version == version]
        ax.scatter(
            g.spearman_centered,
            g.obs_minus_bliss_ex23_served,
            s=4.0,
            marker=marker,
            facecolor=SRC_COLOR[src],
            edgecolor="black",
            linewidth=0.2,
            zorder=3,
        )
    s = R.svd
    for version, key in (("best_available", "Best"), ("all_predicted", "Pred")):
        row = s[
            (s.version == version)
            & (s.similarity == "spearman_centered")
            & (s.growth_call == "served")
        ].iloc[0]
        num(f"svd{key}Spearman", float(row.spearman))
        num(f"svd{key}P", float(row.permutation_p))
        num(f"svd{key}Pairs", int(row.n_pairs))
    raw = s[
        (s.version == "all_predicted")
        & (s.similarity == "spearman_raw")
        & (s.growth_call == "software")
    ].iloc[0]
    num("svdPredRawSpearman", float(raw.spearman))
    num("svdPredRawP", float(raw.permutation_p))
    ax.axhline(0.0, color="black", linewidth=0.4, zorder=1)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(-0.35, 0.15)
    ax.set_xlabel("centered profile similarity of the pair")
    ax.set_ylabel("observed minus Bliss, fitness")
    vgrid(ax, 0.1)
    hgrid(ax, 0.2)
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="measured profile where one exists",
            ),
            Patch(
                facecolor=SRC_COLOR["ridge"],
                edgecolor="black",
                linewidth=0.4,
                label="every profile predicted by ridge",
            ),
        ],
        ncol=1,
    )
    title(ax, "similarity against deviation, both profile versions", n)


def draw_go(ax: Any, R: Reads) -> None:
    """The GO read: measured against predicted against the gene-mean control."""
    d = R.go[R.go.target == "raw"].copy()
    d = d[d.compound.isin([INHIB_NAME[a] for a in INHIB] + list(LONG_TO_ABBR))]
    d["short"] = d.compound.map(lambda c: INHIB_NAME.get(LONG_TO_ABBR.get(c, ""), c))
    x = np.arange(len(d))
    width = 0.26
    series = [
        ("n_sig_measured", "measured"),
        ("n_sig_predicted", "ridge"),
        ("n_sig_control", "control"),
    ]
    for k, (col, src) in enumerate(series):
        ax.bar(
            x + (k - 1) * width,
            d[col].fillna(0).astype(float),
            width=width,
            color=SRC_COLOR[src],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(d.short), rotation=35, ha="right", fontsize=4.4)
    ax.set_ylabel("significant GO terms in the top hundred")
    ax.set_ylim(0, 70)
    vgrid(ax, 20)
    num("goMeasuredMax", int(d.n_sig_measured.fillna(0).max()))
    num("goControlTerms", int(d.n_sig_control.fillna(0).max()))
    num("goSharedMax", int(d.n_sig_shared.fillna(0).max()))
    num("goSharedControlMax", int(d.n_sig_shared_predicted_control.fillna(0).max()))
    nom = R.nom
    core = nom[(nom.category == "core") & (nom.rule == "mean")]
    num("nomCoreBest", int(core[core.version == "best_available"].n_genes.max()))
    num("nomCorePredicted", int(core[core.version == "all_predicted"].n_genes.max()))
    n = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR["measured"],
                edgecolor="black",
                linewidth=0.4,
                label="the measured profile",
            ),
            Patch(
                facecolor=SRC_COLOR["ridge"],
                edgecolor="black",
                linewidth=0.4,
                label="the ridge-predicted profile",
            ),
            Patch(
                facecolor=SRC_COLOR["control"],
                edgecolor="black",
                linewidth=0.4,
                label="the gene mean over the fitted compounds",
            ),
        ],
        ncol=1,
    )
    title(ax, "the GO read, and the control that explains it", n)


def fig_si3(R: Reads) -> dict[str, str]:
    """Supplement 3: the composition-rule and profile detail behind h and i."""
    fig = new_sheet(176)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.95))
    axes = rows(fig, 2, 3)
    draw_emergent(axes[0], R)
    draw_mtx_grid(axes[1], R)
    similarity_matrix(
        axes[2], R, "best_available", "profile similarity, measured where one exists"
    )
    similarity_matrix(
        axes[3],
        R,
        "all_predicted",
        "profile similarity, every profile predicted",
        colorbar=True,
    )
    draw_similarity_vs_deviation(axes[4], R)
    draw_go(axes[5], R)
    sheet_legend(fig, ["measured", "ridge", "control"], hatch=False, ncol=2)
    letters_status(fig, axes, ["ready", "ready", "ready", "ready", "ready", "ready"])
    status_key(fig)
    return save(fig, "gate_si3_profiles")


# ------------------------------------------------------------------------------ main


# ----------------------------------------------- supplement 4, the mixture rounds

SET_ARMS = [
    ("mix_van_e150", "one source: Vanacloig"),
    ("mix_vanhop_e150", "two: with Hoepfner HOP"),
    ("mix_vanhophet_e150", "three: with Hillenmeyer HET"),
]


def round_runs(R: Reads, task: str, subset: str, metric: str, call: str | None) -> Any:
    """Per-run rows of one readout, seed members only (never the seed ensemble)."""
    d = R.rounds
    d = d[
        (d.task == task)
        & (d.subset == subset)
        & (d.metric == metric)
        & d.member.str.startswith("seed")
    ]
    if call is not None:
        d = d[d.call == call]
    return d


def draw_source_sets(ax: Any, R: Reads, readout: str) -> None:
    """The one-, two- and three-source arms over every run on fold seed 0.

    A run is one seed on one fold on one machine; rounds 3 and 4 give up to six seeds
    on five folds per arm, and the twenty-two arms run on both machines contribute both
    copies. The bar is the mean over runs, the whisker its standard deviation, and the
    count is written on the bar.
    """
    if readout == "gene":
        d = round_runs(
            R, "vanacloig", "test_compounds", "centered_spearman_median", None
        )
        bar_label = "nested ridge on FCFP4 counts"
        bar_value = float(NUM["ladderRidge"])
        bar_src = "ridge"
        xlabel = "compound-cold centered Spearman (fold seed 0)"
        xmax = 0.6
    else:
        d = round_runs(R, "ex23", "combinations", "growth_auroc", "served")
        bar_label = "Loewe from the titration fits"
        bar_value = float(NUM["loeweAuroc"])
        bar_src = "loewe"
        xlabel = "growth AUROC over the 63 combinations (served call)"
        xmax = 1.0
    d = d[d.fold_seed == 0]
    data: list[tuple[str, Any, Any, str, bool]] = []
    for arm, label in SET_ARMS:
        v = d[d.arm == arm].value
        tag = {
            "mix_van_e150": "one",
            "mix_vanhop_e150": "two",
            "mix_vanhophet_e150": "three",
        }[arm]
        num(f"r3_{tag}_{readout}_mean", float(v.mean()))
        num(f"r3_{tag}_{readout}_sd", float(v.std(ddof=1)))
        num(f"r3_{tag}_{readout}_n", int(len(v)))
        num(f"r3_{tag}_{readout}_folds", int(d[d.arm == arm].fold.nunique()))
        data.append(
            (
                f"{label} (n = {len(v)})",
                float(v.mean()),
                float(v.std(ddof=1)),
                "cgt",
                False,
            )
        )
    data.append((bar_label, bar_value, None, bar_src, False))
    bars(ax, data, xlabel, xmax)
    title(
        ax,
        "adding public screens: "
        + (
            "gene level, mean over runs"
            if readout == "gene"
            else "wet lab, mean over runs"
        ),
    )


def draw_dose_pairs(ax: Any, R: Reads) -> None:
    """The dose channel on the wet lab: FiLM against off, paired by arm and machine.

    Each pair is the same arm with the log-molar dose modulation on and off, scored on
    the same folds of the same machine at seed 0; the plotted value is the median over
    the folds both copies have. Rounds 1 and 3 both carry such pairs.
    """
    d = round_runs(R, "ex23", "combinations", "growth_auroc", "served")
    g = round_runs(
        R, "isobole_ex26", "interior_cells", "mean_observed_minus_predicted", "served"
    )
    pairs = [
        ("mix_van_e150", "mix_van_e150_film", "one source"),
        ("mix_vanhop_e150", "mix_vanhop_e150_film", "two sources"),
        ("mix_vanhophet_e150", "mix_vanhophet_e150_film", "three sources"),
    ]
    rows_: list[tuple[str, float, float, float, float]] = []
    for off, film, label in pairs:
        for machine in ("gilahyper", "delta"):
            a = d[(d.arm == off) & (d.machine == machine) & (d.fold_seed == 0)]
            b = d[(d.arm == film) & (d.machine == machine) & (d.fold_seed == 0)]
            folds = sorted(set(a.fold) & set(b.fold))
            if not folds:
                continue
            a0 = (
                a[a.fold.isin(folds) & (a.member == "seed0")]
                .groupby("fold")
                .value.mean()
            )
            b0 = (
                b[b.fold.isin(folds) & (b.member == "seed0")]
                .groupby("fold")
                .value.mean()
            )
            ga = g[
                (g.arm == off)
                & (g.machine == machine)
                & g.fold.isin(folds)
                & (g.member == "seed0")
            ].value.median()
            gb = g[
                (g.arm == film)
                & (g.machine == machine)
                & g.fold.isin(folds)
                & (g.member == "seed0")
            ].value.median()
            rows_.append(
                (
                    f"{label}, {machine}, {len(folds)} fold{'s' if len(folds) > 1 else ''}",
                    float(a0.median()),
                    float(b0.median()),
                    float(ga),
                    float(gb),
                )
            )
    y = np.arange(len(rows_))[::-1]
    lower = 0
    for yi, (label, off_v, film_v, _, _) in zip(y, rows_):
        ax.plot([off_v, film_v], [yi, yi], color="black", linewidth=0.5, zorder=2)
        ax.plot(
            off_v,
            yi,
            "o",
            color=SRC_COLOR["cgt"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=3.6,
            zorder=3,
        )
        ax.plot(
            film_v,
            yi,
            "o",
            color=SRC_FILL["cgt"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=3.6,
            zorder=3,
        )
        lower += int(film_v < off_v)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows_])
    ax.set_ylim(-0.6, len(rows_) - 0.4)
    ax.set_xlim(0.3, 0.7)
    ax.set_xlabel("growth AUROC over the 63 combinations (served call)")
    hgrid(ax, 0.1)
    ax.axvline(
        float(NUM["loeweAuroc"]), color=SRC_COLOR["loewe"], linewidth=0.8, zorder=1
    )
    num("r3_dose_pairs", int(len(rows_)))
    num("r3_dose_pairs_filmLower", lower)
    num("r3_dose_pairs_gridMoreNegative", int(sum(1 for r in rows_ if r[4] < r[3])))
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=SRC_COLOR["cgt"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="",
            markersize=3.6,
            label="dose modulation off",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=SRC_FILL["cgt"],
            markeredgecolor="black",
            markeredgewidth=0.4,
            linestyle="",
            markersize=3.6,
            label="dose FiLM on (light fill)",
        ),
        Line2D([], [], color=SRC_COLOR["loewe"], linewidth=0.8, label="Loewe"),
    ]
    n = panel_legend(ax, handles, ncol=3)
    title(ax, "the dose channel costs the growth call: paired, same folds", n)


def draw_per_source(ax: Any, R: Reads) -> None:
    """What the transformer predicts on the public screens it learns from.

    For each auxiliary source, the held-out COMPOUND score of the plain arms that carry
    it (centered Spearman, mean over runs), beside the chemistry-feature learner's
    compound-cold score on the same source from ``source_basis.csv``; for Hoepfner also
    the held-out DOSE score of the FiLM dose arms.
    """
    d = R.rounds[
        R.rounds.member.str.startswith("seed")
        & (R.rounds.metric == "centered_spearman_median")
    ]
    basis = R.basis.set_index("source")
    plain = {
        "hoepfner_hop": ("mix_vanhop_e150", "mix_vanhophet_e150"),
        "hillenmeyer_het": ("mix_vanhophet_e150",),
        "wildenhain": ("mix_vanwil_e150", "mix_vanhophetwil_e150"),
    }
    names = {
        "hoepfner_hop": "Hoepfner HOP",
        "hillenmeyer_het": "Hillenmeyer HET",
        "wildenhain": "Wildenhain",
    }
    #: the row of source_basis.csv that holds each source's chemistry-feature score
    basis_row = {
        "hoepfner_hop": "Hoepfner",
        "hillenmeyer_het": "Hillenmeyer HET",
        "wildenhain": "Wildenhain",
    }
    data: list[tuple[str, Any, Any, str, bool]] = []
    for source, arms in plain.items():
        v = d[
            (d.task == source) & (d.subset == "heldout_compound") & d.arm.isin(arms)
        ].value
        tag = source.split("_")[0]
        num(f"r3_src_{tag}_cgt", float(v.mean()))
        num(f"r3_src_{tag}_cgt_n", int(len(v)))
        data.append(
            (
                f"{names[source]}, transformer (n = {len(v)})",
                float(v.mean()),
                float(v.std(ddof=1)),
                "cgt",
                False,
            )
        )
        if source == "hoepfner_hop":
            w = d[(d.task == source) & (d.subset == "heldout_dose")].value
            num("r3_src_hoepfner_dose", float(w.mean()))
            num("r3_src_hoepfner_dose_n", int(len(w)))
            data.append(
                (
                    f"Hoepfner HOP, held-out DOSE, FiLM arms (n = {len(w)})",
                    float(w.mean()),
                    float(w.std(ddof=1)),
                    "cgt",
                    False,
                )
            )
        ridge = float(basis.loc[basis_row[source], "spearman_median"])
        num(f"r3_src_{tag}_ridge", ridge)
        data.append(
            (
                f"{names[source]}, {basis.loc[basis_row[source], 'best_model']} on {basis.loc[basis_row[source], 'best_features']}",
                ridge,
                None,
                "ridge",
                False,
            )
        )
    bars(ax, data, "held-out centered Spearman on the source", 0.5)
    title(ax, "the public screens are barely learned")


def draw_machines(ax: Any, R: Reads) -> None:
    """The same config at the same seed on the two machines."""
    d = round_runs(R, "vanacloig", "test_compounds", "centered_spearman_median", None)
    d = d[d.member == "seed0"]
    piv = d.pivot_table(index=["name", "fold"], columns="machine", values="value")
    piv = piv.dropna()
    x, y = piv["gilahyper"].to_numpy(), piv["delta"].to_numpy()
    ax.plot([-0.2, 0.6], [-0.2, 0.6], color="black", linewidth=0.4, zorder=1)
    ax.scatter(
        x, y, s=7, color=SRC_COLOR["cgt"], edgecolor="black", linewidth=0.3, zorder=3
    )
    diff = np.abs(x - y)
    num("r3_machines_n", int(len(diff)))
    num("r3_machines_medianAbsDiff", float(np.median(diff)))
    num("r3_machines_maxAbsDiff", float(diff.max()))
    num(
        "r3_machines_spearman",
        float(pd.Series(x).corr(pd.Series(y), method="spearman")),
    )
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(-0.2, 0.6)
    ax.set_xlabel("GilaHyper (RTX 6000 Ada), seed 0")
    ax.set_ylabel("Delta (A40), seed 0")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.text(
        0.03,
        0.97,
        f"{len(diff)} config-fold pairs\nmedian |diff| {np.median(diff):.3f}, max {diff.max():.2f}",
        transform=ax.transAxes,
        fontsize=4.6,
        ha="left",
        va="top",
    )
    title(ax, "the same config and seed on two machines: gene-level score")


def draw_fold_seeds(ax: Any, R: Reads) -> None:
    """The partition effect: the three source-set arms on three fold seeds.

    The fold-seed arms are named ``<arm>_fs1`` and ``<arm>_fs2``; each is one seed on
    folds 0 to 2 of that partition, and the plotted value is the median over its
    folds. The Delta copy is used where both machines ran an arm.
    """
    d = round_runs(R, "vanacloig", "test_compounds", "centered_spearman_median", None)
    d = d[d.member == "seed0"]
    data: list[tuple[str, Any, Any, str, bool]] = []
    tags = {
        "mix_van_e150": "one",
        "mix_vanhop_e150": "two",
        "mix_vanhophet_e150": "three",
    }
    for arm, label in SET_ARMS:
        for fs in (0, 1, 2):
            name = arm if fs == 0 else f"{arm}_fs{fs}"
            sub = d[(d.arm == name) & (d.fold_seed == fs)]
            machine = "delta" if (sub.machine == "delta").any() else "gilahyper"
            v = sub[sub.machine == machine].groupby("fold").value.mean()
            if len(v) == 0:
                data.append(
                    (
                        f"{label}, fold seed {fs} (not run)",
                        float("nan"),
                        None,
                        "cgt",
                        False,
                    )
                )
                continue
            num(f"r3_fs_{tags[arm]}_{fs}", float(v.median()))
            num(f"r3_fs_{tags[arm]}_{fs}_folds", int(len(v)))
            data.append(
                (
                    f"{label}, fold seed {fs} ({len(v)} fold{'s' if len(v) > 1 else ''})",
                    float(v.median()),
                    None,
                    "cgt",
                    False,
                )
            )
    bars(ax, data, "compound-cold centered Spearman, median over folds, one seed", 0.6)
    title(ax, "the partition is the largest effect: three fold seeds")


def fig_si4(R: Reads) -> dict[str, str]:
    """Supplement 4: what the mixture rounds say, read over both machines."""
    fig = new_sheet(165)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.94))
    axes = rows(fig, 3, 2)
    draw_source_sets(axes[0], R, "gene")
    draw_source_sets(axes[1], R, "growth")
    draw_dose_pairs(axes[2], R)
    draw_per_source(axes[3], R)
    draw_machines(axes[4], R)
    draw_fold_seeds(axes[5], R)
    sheet_legend(fig, ["loewe", "ridge", "cgt"], hatch=False, ncol=3)
    letters_status(
        fig, axes, ["partial", "partial", "ready", "ready", "ready", "partial"]
    )
    status_key(fig)
    return save(fig, "gate_si4_rounds")


#: the chemistry encoders of the 038 ladder, by family: what each embeds a compound as
ENCODER_FAMILY = {
    "fcfp4_count": ("FCFP4 counts", "fingerprint"),
    "ecfp4_count": ("ECFP4 counts", "fingerprint"),
    "ecfp4_bit": ("ECFP4 bits", "fingerprint"),
    "maccs": ("MACCS keys", "fingerprint"),
    "rdkit_2d": ("RDKit 2D descriptors", "descriptor"),
    "mol2vec": ("mol2vec", "pretrained"),
    "mole_static": ("MolE", "pretrained"),
    "unimol_v1": ("Uni-Mol v1 (3D)", "pretrained"),
    "chemberta2_mtr": ("ChemBERTa-2 MTR", "pretrained"),
    "chemberta2_mlm": ("ChemBERTa-2 MLM", "pretrained"),
    "molformer_xl": ("MoLFormer-XL", "pretrained"),
    "roberta_zinc_480m": ("RoBERTa ZINC 480M", "pretrained"),
    "fingerprints": ("all fingerprints, concatenated", "combo"),
    "pretrained": ("all pretrained, concatenated", "combo"),
    "all": ("every embedding, concatenated", "combo"),
}
FAMILY_SRC = {
    "fingerprint": "ridge",
    "descriptor": "mean",
    "pretrained": "control",
    "combo": "hsa",
}


def draw_encoders(ax: Any, R: Reads) -> None:
    """Supplement seven, a: every chemistry encoder of the ladder, its best learner as
    the bar and every learner it was run with as a point, on the one basis the ladder
    scores on (centered Spearman across strains, median over the 96 compound-cold
    evaluations at the full compound count).
    """
    d = R.ladder[(R.ladder.target == "centered") & (R.ladder.model != "selected")]
    d = d[d.compounds == d.compounds.max()].copy()
    d["embedding"] = d.kernel.str.split(":").str[-1]
    d["learner"] = d.model + ", " + d.kernel.str.split(":").str[0]
    best = (
        d.sort_values("spearman_median", ascending=False)
        .groupby("embedding", as_index=False)
        .first()
        .sort_values("spearman_median", ascending=False)
    )
    data = []
    for _, r in best.iterrows():
        label, family = ENCODER_FAMILY.get(r.embedding, (r.embedding, "pretrained"))
        data.append(
            (
                f"{label} ({r.learner})",
                float(r.spearman_median),
                None,
                FAMILY_SRC[family],
                False,
            )
        )
    bars(ax, data, "median centered Spearman, compound-cold (96 evaluations)", 0.40)
    y = np.arange(len(data))[::-1]
    for yi, (_, r) in zip(y, best.iterrows()):
        others = d[d.embedding == r.embedding]
        ax.scatter(
            others.spearman_median,
            np.full(len(others), yi),
            s=4,
            color="black",
            zorder=4,
            linewidths=0,
        )
    ax.tick_params(axis="y", labelsize=4.6)
    n_rows = panel_legend(
        ax,
        [
            Patch(
                facecolor=SRC_COLOR[FAMILY_SRC[f]],
                edgecolor="black",
                linewidth=0.4,
                label=lab,
            )
            for f, lab in (
                ("fingerprint", "fingerprint"),
                ("descriptor", "RDKit descriptors"),
                ("pretrained", "pretrained molecular model"),
                ("combo", "concatenation"),
            )
        ],
        ncol=4,
    )
    num("encoderN", int(best.embedding.nunique()))
    num("encoderLearnerRuns", int(len(d)))
    num("encoderBest", float(best.spearman_median.iloc[0]))
    num("encoderBestName", ENCODER_FAMILY[best.embedding.iloc[0]][0])
    worst = best[~best.embedding.isin(["all", "fingerprints", "pretrained"])]
    num("encoderWorst", float(worst.spearman_median.iloc[-1]))
    num("encoderWorstName", ENCODER_FAMILY[worst.embedding.iloc[-1]][0])
    pre = best[
        best.embedding.map(lambda e: ENCODER_FAMILY.get(e, ("", "pretrained"))[1])
        == "pretrained"
    ]
    num("encoderBestPretrained", float(pre.spearman_median.iloc[0]))
    num("encoderBestPretrainedName", ENCODER_FAMILY[pre.embedding.iloc[0]][0])
    title(
        ax,
        "the chemistry encoders on one basis: best learner (bar), every learner (points)",
        n_rows,
    )


def draw_paired_encoder(ax: Any, R: Reads) -> None:
    """Supplement seven, b: the learned environment encoder against its bilinear
    control, paired on compound and fold: the per-evaluation difference, and where the
    ridge bar and the mixture trainer sit on the same axis.
    """
    if not (len(R.bil) and len(R.envenc)):
        blank(ax, 100, 46)
        return
    key_cols = ["compound", "fold", "fold_seed"]
    a = R.bil[(R.bil.target == "centered") & (R.bil.member == "ensemble")]
    b = R.envenc[(R.envenc.target == "centered") & (R.envenc.member == "ensemble")]
    paired = a.merge(b, on=key_cols, suffixes=("_bil", "_enc")).sort_values(
        "spearman_bil"
    )
    x = np.arange(len(paired))
    ax.vlines(
        x,
        paired.spearman_bil,
        paired.spearman_enc,
        color=SRC_COLOR["cgt"],
        linewidth=0.5,
        alpha=0.6,
        zorder=2,
    )
    ax.scatter(
        x, paired.spearman_bil, s=4, color=SRC_COLOR["control"], zorder=3, linewidths=0
    )
    ax.scatter(
        x, paired.spearman_enc, s=4, color=SRC_COLOR["cgt"], zorder=4, linewidths=0
    )
    ridge = R.ladder_row("krr", "linear:fcfp4_count")
    ax.axhline(
        float(ridge.spearman_median), color=SRC_COLOR["ridge"], linewidth=0.8, zorder=1
    )
    ax.axhline(0.0, color="black", linewidth=0.4, zorder=1)
    ax.set_xlim(-1, len(paired))
    ax.set_xticks([])
    ax.set_xlabel("the 96 compound-evaluations, sorted by the control's score")
    ax.set_ylabel("centered Spearman across strains")
    ax.set_ylim(-0.6, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)
    delta = paired.spearman_enc - paired.spearman_bil
    n = panel_legend(
        ax,
        [
            Line2D(
                [],
                [],
                marker="o",
                color=SRC_COLOR["control"],
                linestyle="",
                markersize=2.5,
                label="bilinear control, 3-seed ensemble",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color=SRC_COLOR["cgt"],
                linestyle="",
                markersize=2.5,
                label="environment encoder, 3-seed ensemble",
            ),
            Line2D(
                [],
                [],
                color=SRC_COLOR["ridge"],
                linewidth=0.8,
                label=f"nested ridge, fingerprint counts ({float(ridge.spearman_median):.3f})",
            ),
        ],
        ncol=1,
    )
    ax.text(
        0.02,
        0.04,
        f"encoder minus control: median {delta.median():+.3f}, up on {int((delta > 0).sum())} of {len(delta)}",
        transform=ax.transAxes,
        fontsize=5,
        ha="left",
        va="bottom",
    )
    title(ax, "the learned encoder against its control, paired", n)


def fig_si7(R: Reads) -> dict[str, str]:
    """Supplement seven: the encoders compared, and how."""
    fig = new_sheet(80)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.94))
    axes = rows(fig, 1, 2)
    draw_encoders(axes[0], R)
    draw_paired_encoder(axes[1], R)
    letters_status(fig, axes, ["ready", "ready"])
    return save(fig, "gate_si7_encoders")


def newest_svg(stem: str) -> dict[str, str]:
    """The newest ``<stem>_<timestamp>.svg`` of the experiment's image directory.

    The boundary and rule-choice figures are drawn by mixture_rules.py and
    rule_choice.py, which are committed scripts of this experiment; the gate document
    places them unchanged, so the sheet is the newest file each script wrote.
    """
    files = sorted(glob.glob(osp.join(ASSET_IMAGES_DIR, SLUG, f"{stem}_*.svg")))
    assert files, f"no {stem}_*.svg under {osp.join(ASSET_IMAGES_DIR, SLUG)}"
    svg = files[-1]
    name = osp.splitext(osp.basename(svg))[0]
    return {"name": name, "png": svg[:-4] + ".png", "svg": svg}


ARM_SHORT = {
    "r1_mix_van_ex21_film_L1_lh1": "jointFilm",
    "r1_mix_van_anchors_film_L1_lh1": "anchorsFilm",
    "r1_mix_van_ex21_off_L1_lh1": "jointOff",
    "r1_mix_van_anchors_off_L1_lh1": "anchorsOff",
    "r1_mix_van_ft_film_L1_lh1": "ftHead",
    "r1_mix_van_ftenv_film_L1_lh1": "ftEnv",
    "r1_mix_van_ex21_film_L2_lh1": "jointTwoLayer",
    "r1_mix_van_ex21_film_L1_lh0p1": "jointLambdaTenth",
    "r1_mix_vanhop_ex21_film_L1_lh1": "jointHoepfner",
    "r1_mix_vanhophet_ex21_film_L1_lh1": "jointHoepfnerHet",
}


def boundary_numbers(R: Reads) -> None:
    """Caption numbers for the boundary readout, the rule choice and the model arms."""
    b = R.boundary
    for _, r in b.iterrows():
        key = f"bnd_{r['run']}_{r['observed']}_{r['rule']}"
        num(f"{key}_jaccard", float(r["jaccard_alive"]))
        num(f"{key}_aliveDead", int(r["n_predicted_alive_observed_dead"]))
        num(f"{key}_deadAlive", int(r["n_predicted_dead_observed_alive"]))
        num(f"{key}_edgeMae", float(r["edge_by_column_mae_steps"]))
    for run, g in b.groupby("run"):
        num(f"bnd_{run}_islands", int(g["n_islands_dropped"].iloc[0]))
        num(
            f"bnd_{run}_aliveRaw",
            int(g[g.observed == "raw"]["n_alive_observed"].iloc[0]),
        )
        num(
            f"bnd_{run}_aliveMonotone",
            int(g[g.observed == "monotone"]["n_alive_observed"].iloc[0]),
        )
    p = R.rc_pairs
    for cls, g in p.groupby("pair_class"):
        tag = cls.replace("|", "_")
        num(f"rc_{tag}_n", int(len(g)))
        num(f"rc_{tag}_loewe", int((g.closer_rule == "loewe").sum()))
        num(f"rc_{tag}_bliss", int((g.closer_rule == "bliss").sum()))
    num("rc_nPairs", int(len(p)))
    num("rc_nGrown", int(p.grew.sum()))
    lopo = R.rc_lopo.set_index("predictor")
    for name, tag in (
        ("bliss", "bliss"),
        ("loewe", "loewe"),
        ("global_w", "globalW"),
        ("class_rule", "classRule"),
        ("class_w", "classW"),
        ("similarity_best_available_spearman_centered", "simBest"),
        ("similarity_all_predicted_spearman_centered", "simPred"),
    ):
        num(f"lopo_{tag}_mae", float(lopo.loc[name, "mae"]))
        num(f"lopo_{tag}_acc", float(lopo.loc[name, "growth_accuracy"]))
        num(f"lopo_{tag}_spGrown", float(lopo.loc[name, "spearman_grown"]))
    o = R.rc_orders
    for rule, tag in (("loewe_ex21", "loewe"), ("class_hierarchy", "hier")):
        row = o[(o.subset == "all") & (o.rule == rule)].iloc[0]
        num(f"orders_{tag}_auroc", float(row["auroc"]))
        num(f"orders_{tag}_rmse", float(row["rmse_grown"]))
        num(f"orders_{tag}_worst", float(row["worst_scenario_spearman"]))
    for arm, scores in R.arms.items():
        tag = ARM_SHORT[arm]
        if scores.empty or "member" not in scores:
            continue
        e = scores[scores.member == "ensemble"]

        def med(task: str, subset: str, metric: str, call: str | None) -> float:
            s = e[(e.task == task) & (e.subset == subset) & (e.metric == metric)]
            if call is not None:
                s = s[s.call == call]
            return float(s.value.median())

        num(f"arm_{tag}_folds", int(e.fold.nunique()))
        num(f"arm_{tag}_auroc", med("ex23", "combinations", "growth_auroc", "served"))
        num(
            f"arm_{tag}_worst",
            med("ex23", "combinations", "worst_scenario_spearman", "served"),
        )
        num(
            f"arm_{tag}_fitGrown",
            med("ex23", "combinations_that_grew", "fitness_spearman", "served"),
        )
        num(
            f"arm_{tag}_isoTwoSix",
            med(
                "isobole_ex26",
                "interior_cells",
                "mean_observed_minus_predicted",
                "served",
            ),
        )
        num(
            f"arm_{tag}_van",
            med("vanacloig", "test_compounds", "centered_spearman_median", None),
        )
    num("arms_scored", int(sum(1 for a in R.arms.values() if not a.empty)))


def main() -> None:
    """Draw every sheet, then write the manifest, the numbers and the macros."""
    R = Reads()
    figures = {
        "concept": fig_concept(R),
        "main": fig_main(R),
        "si1": fig_si1(R),
        "si2": fig_si2(R),
        "si3": fig_si3(R),
    }
    boundary_numbers(R)
    figures["si4"] = fig_si4(R)
    figures["si5"] = newest_svg("r3_convergence")
    figures["si6"] = newest_svg("r3_arms_overview")
    figures["si7"] = fig_si7(R)
    conv = R.convergence.set_index(["arm", "source"])
    for arm, source, tag in (
        ("mix_van_e150", "vanacloig", "Van"),
        ("mix_vanhop_e150", "hoepfner_hop", "Hop"),
        ("mix_vanhophet_e150", "hillenmeyer_het", "Het"),
        ("mix_vanwil_e150", "wildenhain", "Wil"),
    ):
        row = conv.loc[(arm, source)]
        num(f"conv{tag}TrainEarly", float(row.train_early))
        num(f"conv{tag}TrainEnd", float(row.train_end))
        num(f"conv{tag}Removed", float(row.heldout_removed_from_first))
        num(f"conv{tag}MinEpoch", float(row.heldout_min_epoch))
        num(f"conv{tag}Rise", float(row.heldout_rise_after_min))
    figures["boundary"] = newest_svg("mixture_rules_isoboles")
    figures["rulechoice"] = newest_svg("rule_choice")
    internal = {k: v for k, v in R.paths.items() if not k.startswith("/")}
    external = {k: v for k, v in R.paths.items() if k.startswith("/")}
    manifest = {
        "generated_by": (
            "experiments/040-inhibitor-synergy-wetlab/scripts/figure_gate_panels.py"
        ),
        "generated_at": timestamp(),
        "figures": figures,
        "sources": sorted(internal),
        "external_sources": sorted(external),
        "n_numbers": len(NUM),
    }
    with open(osp.join(R040, "figure_gate_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(osp.join(R040, "figure_gate_numbers.json"), "w") as fh:
        json.dump(NUM, fh, indent=2, sort_keys=True, default=str)
    os.makedirs(DOC_TABLES, exist_ok=True)
    header = (
        "%% SOURCE: written by experiments/040-inhibitor-synergy-wetlab/scripts/"
        "figure_gate_panels.py from the result files it reads\n"
    )
    names = [macro_name(k) for k in sorted(NUM)]
    assert len(set(names)) == len(names), "two numbers share a macro name"
    with open(osp.join(DOC_TABLES, "gate_numbers.tex"), "w") as fh:
        fh.write(header)
        for k, name in zip(sorted(NUM), names):
            fh.write(f"\\newcommand{{\\hyd{name}}}{{{fmt(NUM[k])}}}\n")
    with open(osp.join(DOC_TABLES, "gate_figure_names.tex"), "w") as fh:
        fh.write(header)
        for key, f in figures.items():
            fh.write(
                f"\\newcommand{{\\fig{macro_name(key.capitalize())}}}"
                f"{{figures/{f['name']}.pdf}}\n"
            )
    for key, f in figures.items():
        print(key, f["png"])
    print(f"{len(NUM)} numbers -> {DOC_TABLES}/gate_numbers.tex")


if __name__ == "__main__":
    main()

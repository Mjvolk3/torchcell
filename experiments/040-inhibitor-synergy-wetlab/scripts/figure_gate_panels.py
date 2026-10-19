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
import re
import subprocess
import sys
from typing import Any

import matplotlib

# The PDF backend, not Agg: the model panel sets its equations in real LaTeX (usetex),
# which Agg cannot rasterize on GilaHyper (no dvipng) while the PDF and SVG backends
# read the DVI themselves. The PNG of every sheet is rasterized from its PDF.
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import IC30_FITNESS, IDENTITY_GAP_SMILES, fcfp4_count  # noqa: E402
from mixture_rules import monotone_alive  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    acs_structure,
    acs_structure_svg,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
    text_clashes,  # noqa: E402
)

load_dotenv()
apply_paper_style()
# Real LaTeX for the equations of the model panel only (``usetex=True`` per text
# artist): Helvetica through helvet and sansmath, so a formula set by LaTeX sits in the
# same sans face as the Arial labels around it. matplotlib wraps usetex text in
# ``\rmfamily`` when ``font.family`` is a face it does not know (Arial), so the ROMAN
# default is what has to be Helvetica. Everything else stays Arial mathtext.
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{helvet}\usepackage{sansmath}\sansmath"
    r"\renewcommand{\rmdefault}{phv}"
)
#: the cell graph transformer's gene table, one token per gene (train_mixture.py)
GENE_TOKENS = 6607
#: the compound fingerprint the trainer featurizes a molecule as (mixture_data.py)
FINGERPRINT_BITS = 2048

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
    "ridge": PLOT_PALETTE[5],
    "cgt": PLOT_PALETTE[4],
    "control": PLOT_PALETTE[10],
}
SRC_FILL = {
    "measured": PLOT_PALETTE_FILL[0],
    "bliss": PLOT_PALETTE_FILL[1],
    "loewe": PLOT_PALETTE_FILL[2],
    "hsa": PLOT_PALETTE_FILL[3],
    "mean": PLOT_PALETTE_FILL[6],
    "ridge": PLOT_PALETTE_FILL[5],
    "cgt": PLOT_PALETTE_FILL[4],
    "control": PLOT_PALETTE_FILL[10],
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
#: ARM names, not sweep names: every sweep of the arm pools into the host readout
#: (round 9 seed 0, round 10 seeds 1 and 2, the fold-seed twins and seeds 3 to 5),
#: on both machines; ``host_arm_of`` strips the round prefix and the seed, fold-seed
#: and pack suffixes a sweep name carries
MODEL_ARM = "van_e150_cell_bliss"
ZERO_SHOT_ARM = "van_e150_cell_anchors_bliss"
#: the fitted model with the dose offset (round 10): the compound tokens stay
#: dose-free and the dose scales the medium offset, so it is the one arm of the
#: sheet whose single-agent curve moves with dose
DOSE_ARM = "van_e150_cell_bliss_doff"
#: (arm, line style, legend label, number tag) of the three models on the sheet
SHEET_MODELS: list[tuple[str, str, str, str]] = [
    (ZERO_SHOT_ARM, ":", "model, zero-shot", "ZeroShot"),
    (MODEL_ARM, "--", "model fit on these wells", "Fitted"),
    (DOSE_ARM, "-.", "fit, with the dose offset", "Dose"),
]


def host_arm_of(sweep: str) -> str:
    """``r10gh_van_e150_cell_bliss_s345_p1`` -> ``van_e150_cell_bliss``."""
    name = re.sub(r"^r\d+[a-z]*_", "", sweep)
    return re.sub(r"(_s\d+|_fs\d+|_p\d+)+$", "", name)


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
        #: what a label is in each source, and which inhibitors of this study any
        #: screen holds (``source_labels.py``), and what pairs of screens share
        #: (``source_axes.py``, from 033's cross-source cells)
        self.label_hist = self.csv(osp.join(R040, "source_label_histograms.csv"))
        self.label_sum = self.csv(osp.join(R040, "source_label_summary.csv"))
        self.overlap = self.csv(osp.join(R040, "source_inhibitor_overlap.csv"))
        self.agreement = self.csv(osp.join(R040, "source_agreement.csv"))

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

    def host_mean(self, arm: str) -> pd.DataFrame:
        """One arm's host prediction per medium, mean over its runs (every sweep of
        the arm, every fold, every machine; ``folds`` counts the distinct folds,
        ``runs`` the prediction files averaged).
        """
        d = self.host_pred[self.host_pred.sweep.map(host_arm_of) == arm].copy()
        assert len(d), f"no host predictions for {arm}"
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
            {
                **{k: "first" for k in keep},
                "predicted": "mean",
                "fold": "nunique",
                "machine": "size",
            }
        )
        return out.rename(columns={"fold": "folds", "machine": "runs"})

    def round_scores(self) -> pd.DataFrame:
        """Every score row of rounds three to ten, on this machine and pulled from
        Delta.

        ``results/mixture/`` holds the GilaHyper runs, ``results/mixture-delta/`` the
        Delta runs' tables; the same config at the same seed was run on both for
        most arms, and the ``machine`` column is what tells them apart. ``arm`` is
        the config name without its fold and seed suffixes, so the seed twins of one
        arm (``_s12``, ``_s345``) fold into it and the fold-seed twins (``_fs1``) do
        not.
        """
        frames = []
        for machine, sub in (("gilahyper", "mixture"), ("delta", "mixture-delta")):
            for d in sorted(glob.glob(osp.join(R040, sub, "r[0-9]*"))):
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
    # the review PDF keeps its text as text (TrueType, not Type 3 outlines)
    plt.rcParams["pdf.fonttype"] = 42
    return plt.figure(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(height_mm)),
        layout="constrained",
    )


def rows(
    fig: Any,
    n_rows: int,
    n_cols: int | list[int],
    widths: list[list[float] | None] | None = None,
    heights: list[float] | None = None,
    placed: dict[int, list[list[float]]] | None = None,
) -> list[Any]:
    """One subfigure per row, so a long tick label cannot set another row's margin.

    ``n_cols`` is one count for every row or one per row; ``widths`` gives a row's
    column width ratios (``None`` for equal columns); ``heights`` the row ratios;
    ``placed`` gives a row's axes as explicit ``[x0, y0, w, h]`` boxes in the row's
    fraction instead of a gridspec. The layout engine leaves an axes made that way
    alone, so one axes of the row may be shorter than its neighbors, which is how
    a panel with an x axis ends its x label level with the bottoms of two without.
    """
    subs = fig.subfigures(n_rows, 1, height_ratios=heights)
    subs = [subs] if n_rows == 1 else list(subs)
    cols = [n_cols] * n_rows if isinstance(n_cols, int) else n_cols
    ratios = widths or [None] * n_rows
    out: list[Any] = []
    for i, (sf, n, ratio) in enumerate(zip(subs, cols, ratios, strict=True)):
        if placed and i in placed:
            out.extend(sf.add_axes(box) for box in placed[i])
            continue
        axes = sf.subplots(1, n, gridspec_kw={"width_ratios": ratio} if ratio else None)
        out.extend(list(np.atleast_1d(axes)))
    return out


def save(fig: Any, name: str) -> dict[str, str]:
    """Write the PNG and the true-size SVG of one sheet, and its review PDF.

    The timestamped PNG and SVG are what the document converts into its figures; the
    PDF is written under the sheet's bare name, overwritten on every run, so the sheet
    can be opened on its own, at true size, apart from the document that embeds it
    (owner, 2026-10-15; the interactions and metabolism cases do the same).
    """
    # every sheet is checked for text that runs into other text before it is written
    # (owner, 2026-10-18); a clash is printed with the two strings so it can be fixed
    for a, b in text_clashes(fig):
        print(f"  CLASH {name}: {a!r} x {b!r}", flush=True)
    out_dir = osp.join(ASSET_IMAGES_DIR, SLUG)
    os.makedirs(out_dir, exist_ok=True)
    stamp = timestamp()
    png = osp.join(out_dir, f"{name}_{stamp}.png")
    svg = osp.join(out_dir, f"{name}_{stamp}.svg")
    pdf = osp.join(out_dir, f"{name}.pdf")
    savefig_true_size_svg(fig, svg)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    # the PNG is the PDF rasterized at 300 dpi: Agg cannot draw the usetex equations
    subprocess.run(
        ["pdftoppm", "-r", "300", "-png", "-singlefile", pdf, png[: -len(".png")]],
        check=True,
    )
    return {"name": f"{name}_{stamp}", "png": png, "svg": svg, "pdf": pdf}


#: what a panel letter and its two status marks take on the title line, in points:
#: ``letters_status`` sets them at the panel's left edge, and the title starts after
#: them
LETTER_BLOCK_PT = 17.0


def title(ax: Any, text: str, legend_rows: int = 0) -> None:
    """Panel title, padded to clear a legend of ``legend_rows`` rows above the axes.

    The title is set flush left after the letter block, as Nature's panels read
    ("e", the marks, the title), so the whole panel width less the block is the
    title's; a title wider than that is set on two balanced lines (owner, 2026-10-18:
    a title that states a panel's takeaway is longer than one that names its subject,
    and centered it ran into the panel letters and the neighbors' labels). The
    panel's width is read after a layout pass, so the wrap is decided on the real
    box; a title that does not fit on two lines stops the run.
    """
    pad = 4.0 + 7.4 * legend_rows
    fig = ax.figure.get_figure(root=True)
    t = ax.set_title(text, pad=pad, loc="left")
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    scale = 72.0 / fig.dpi
    width = ax.get_window_extent().width * scale
    avail = width - LETTER_BLOCK_PT
    if t.get_window_extent(renderer).width * scale > avail:
        words = text.split(" ")
        best = None
        for cut in range(1, len(words)):
            lines = [" ".join(words[:cut]), " ".join(words[cut:])]
            widths = [
                ax.set_title(line, pad=pad, loc="left")
                .get_window_extent(renderer)
                .width
                * scale
                for line in lines
            ]
            if best is None or max(widths) < best[0]:
                best = (max(widths), lines)
        assert best is not None and best[0] <= avail, (
            f"the title {text!r} does not fit {width:.0f} pt on two lines"
        )
        t = ax.set_title("\n".join(best[1]), pad=pad, loc="left", linespacing=1.1)
    t.set_x(LETTER_BLOCK_PT / width)


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
    fig.draw_without_rendering()
    renderer = fig._get_renderer()
    inv = fig.transFigure.inverted()
    for ax, ch, status in zip(axes, "abcdefghijklmnop", statuses):
        # ON THE TITLE LINE, at the axes' left edge, the title starting after the
        # letter and its marks (``title``). A letter set above a row's content sat in
        # the gap the row above fills with its x labels (owner, 2026-10-18: j on f's x
        # axis, the same at g); a band reserved for the letters cost every row 4 mm
        # and squeezed the ladders to 9 mm. The title is already in the layout, so a
        # letter on its line takes no height and meets nothing.
        heading = ax._left_title  # ``title`` sets every panel title flush left
        assert heading.get_text(), f"panel {ch} has no title to letter against"
        title_top = heading.get_window_extent(renderer).y1
        x0, y1 = inv.transform((ax.get_window_extent().x0, title_top))
        t = fig.text(x0, y1, ch, fontsize=8, fontweight="bold", ha="left", va="top")
        t.set_in_layout(False)
        if not STATUS_MARKS:
            continue
        glyph, color = SECTION_SYMBOL[SECTION_STATUS]
        wide = 0.004 if ch in "mw" else 0.0  # a wide letter pushes its marks right
        g1 = fig.text(
            x0 + 0.013 + wide,
            y1 - 0.002,
            glyph,
            fontsize=5,
            color=color,
            ha="left",
            va="top",
            family="DejaVu Sans",
        )
        g1.set_in_layout(False)
        g2 = fig.text(
            x0 + 0.022 + wide,
            y1 - 0.002,
            STATUS_SYMBOL[status],
            fontsize=5,
            color=GRAY,
            ha="left",
            va="top",
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
    usetex: bool = False,
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
        usetex=usetex,
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
        "a mixture is a SET of\ncompound tokens: each is\nread alone through the head\nand the readings compose\nas Bliss (rounds 9 and 10)",
        SRC_COLOR["cgt"],
        "white",
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


#: The six inhibitors as the archive and the loaders name them, in the two families a
#: hydrolysate carries them in: the furan aldehydes from sugar dehydration, and the
#: weak acids (``INHIB`` order within each family).
INHIB_FAMILY = {
    "furan aldehydes": ["FF", "HMF"],
    "weak acids": ["AA", "FA", "LVA", "LA"],
}


#: The three Vanacloig 2022 compounds nearest the six inhibitors by FCFP4 count
#: Tanimoto over the pool's curated SMILES (``profile_fingerprint_check.csv``), one per
#: row of panel a beside the inhibitor it is nearest to, as the pool names them.
VANACLOIG_NEIGHBORS = {
    "FF": "4-hydroxybenzaldehyde",
    "HMF": "vanillin",
    "LVA": "methylglyoxal",
}
NEIGHBOR_NAME = {
    "4-hydroxybenzaldehyde": "4-hydroxybenzaldehyde",
    "vanillin": "vanillin",
    "methylglyoxal": "methylglyoxal",
}


def tanimoto_counts(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Tanimoto similarity of two count fingerprints (min over max, summed)."""
    return float(np.minimum(a, b).sum() / np.maximum(a, b).sum())


def draw_molecules(ax: Any, R: Reads) -> None:
    """Panel a: the six inhibitors of this study at the ACS standard, in two columns.

    The weak acids on the left, the furan aldehydes and the keto acid on the right, one
    row per pair, the host's IC50 after each name. Every structure is drawn by
    ``torchcell.utils.acs_structure`` at the fixed ACS bond length, so the grid is laid
    out from the structures' own widths and heights in points and nothing scales with
    the axes; the structures are kept out of the layout engine so an overhang cannot
    shrink the axes. The Vanacloig compound nearest each inhibitor by FCFP4 count
    Tanimoto is still measured for the caption, and no longer drawn: which inhibitors a
    screen holds is stated in panel c (owner, 2026-10-18: the neighbors confused the
    question of why the sources join at all).
    """
    blank(ax, 100, 50)
    fits = {ab: R.fit("ex21", ab) for ab in INHIB}
    grid = [["AA", "FF"], ["FA", "HMF"], ["LA", "LVA"]]
    smiles = {key: R.smiles[INHIB_LONG[key]] for row in grid for key in row}
    size = {key: acs_structure_svg(smi)[1:] for key, smi in smiles.items()}
    for ab, neighbor in VANACLOIG_NEIGHBORS.items():
        sim = tanimoto_counts(
            fcfp4_count(R.smiles[INHIB_LONG[ab]]), fcfp4_count(R.smiles[neighbor])
        )
        num(f"neighbor{macro_name(ab.capitalize())}Similarity", sim)
    # the axes as the sheet's layout leaves it, measured after a layout pass with the
    # title set, so the grid is centered on the real box (this panel is drawn last for
    # that reason); the structures are then kept out of the layout
    title(ax, "the six inhibitors (host IC50, g/L)")
    fig = ax.figure.get_figure(root=True)
    fig.draw_without_rendering()
    extent = ax.get_window_extent()
    axes_w_pt = extent.width / fig.dpi * 72.0
    axes_h_pt = extent.height / fig.dpi * 72.0
    num("panelAWidthMm", round(axes_w_pt / 72.0 * 25.4, 1))
    num("panelAHeightMm", round(axes_h_pt / 72.0 * 25.4, 1))
    name_pt, gap_x, gap_y, header_pt = 5.5, 4.0, 2.5, 6.5
    col_w = [max(size[row[j]][0] for row in grid) for j in range(2)]
    row_h = [max(size[key][1] for key in row) + name_pt for row in grid]
    total_w = sum(col_w) + gap_x
    total_h = sum(row_h) + 2 * gap_y + header_pt
    left0 = (axes_w_pt - total_w) / 2.0
    top = (axes_h_pt + total_h) / 2.0
    for j, family in enumerate(("weak acids", "furans, a keto acid")):
        x_head = left0 + sum(col_w[:j]) + j * gap_x + col_w[j] / 2.0
        head = ax.text(
            x_head / axes_w_pt,
            (top - 1.0) / axes_h_pt,
            family,
            transform=ax.transAxes,
            fontsize=4.4,
            ha="center",
            va="top",
            style="italic",
        )
        head.set_in_layout(False)
    y = top - header_pt
    for row, h in zip(grid, row_h, strict=True):
        x = left0
        # the structures of a row are centered on the row's structure band, and
        # every name sits on one line under the band
        band = h - name_pt
        for key, w in zip(row, col_w, strict=True):
            cx = (x + w / 2.0) / axes_w_pt
            box_, _w, _h = acs_structure(
                ax,
                smiles[key],
                (cx, (y - band / 2.0) / axes_h_pt),
                xycoords="axes fraction",
                box_alignment=(0.5, 0.5),
            )
            box_.set_in_layout(False)
            t = ax.annotate(
                f"{INHIB_NAME[key]}, {float(fits[key].ic50_g_per_l):.2g}",
                (cx, y / axes_h_pt),
                xycoords="axes fraction",
                xytext=(0, -(band + 1.5)),
                textcoords="offset points",
                fontsize=4.4,
                ha="center",
                va="top",
            )
            t.set_in_layout(False)
            x += w + gap_x
        y -= h + gap_y


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
    "vanacloig": "Vanacloig 2022 (task)",
    "hoepfner_hop": "Hoepfner 2014 HOP",
    "hillenmeyer_het": "Hillenmeyer 2008 HET",
    "wildenhain": "Wildenhain 2015",
}
#: the host records as the sheet names them: "this study" for the Bioscreen runs, the
#: run ids (ex21, ex23, ex26 to ex28) staying in the source table, the script and the
#: experiment note as the internal record (owner, 2026-10-18)
HOST_TITLE = {
    "host_anchor": "public IC30 doses (anchors)",
    "host_ex21": "this study: titration",
    "host_ex23": "this study: combinations",
    "host_isobole": "this study: isobole grids",
}
#: the internal record behind each "this study" row
HOST_RUNS = {"host_ex21": "ex21", "host_ex23": "ex23", "host_isobole": "ex26 to ex28"}
#: row position of each source on the two data panels, which share it: the four
#: gene-level screens on top, a gap, then the host records by split
SOURCE_Y = {
    "vanacloig": 8.0,
    "hoepfner_hop": 7.0,
    "hillenmeyer_het": 6.0,
    "wildenhain": 5.0,
    "host_anchor": 3.0,
    "host_ex21": 2.0,
    "host_ex23": 1.0,
    "host_isobole": 0.0,
}
#: a source's label, as the panel names it (``source_labels.py`` states the formula)
LABEL_SHORT = {
    "vanacloig": "log2 ratio to the control",
    "hoepfner_hop": "sensitivity score (MAD units)",
    "hillenmeyer_het": "fitness-defect log2 ratio, negated",
    "wildenhain": "growth z-score (IQR units)",
    "host_ex21": "no growth in {p:.0%} of wells",
    "host_ex23": "no growth in {p:.0%} of wells",
    "host_isobole": "no growth in {p:.0%} of wells",
}
#: the host rows of ``source_label_histograms.csv``
LABEL_ROW = {
    "host_ex21": "this study: titration",
    "host_ex23": "this study: combinations",
    "host_isobole": "this study: isobole grids",
}
TRAINED_ON = (
    "vanacloig",
    "hoepfner_hop",
    "hillenmeyer_het",
    "wildenhain",
    "host_anchor",
    "host_ex21",
)


def draw_sources(ax: Any, R: Reads) -> None:
    """Panel b, left: how much each source holds, and in what culture.

    One bar per source, the records the trainer reads on a log scale (a screen's cells,
    this study's wells, the anchors' conditions), dark where the source is trained on and
    light where it is held out; the compound count and the culture beside each bar.
    """
    c = R.counts.set_index("source")
    rows_: list[tuple[str, str, int, str]] = []
    for src in ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"]:
        r = c.loc[src]
        rows_.append(
            (
                src,
                f"{int(r.n_compounds):,} compounds\n{CULTURE[src]}",
                int(r.n_cells),
                SOURCE_TITLE[src],
            )
        )
    for split in ["host_anchor", "host_ex21", "host_ex23", "host_isobole"]:
        r = c.loc[split]
        if split == "host_anchor":
            detail = f"{int(r.n_compounds)} compounds\nfitness asserted, not measured"
            n = int(r.n_conditions)
        else:
            detail = f"{int(r.n_compounds)} inhibitors\n{CULTURE['host']}"
            n = int(r.n_cells)
        rows_.append((split, detail, n, HOST_TITLE[split]))
    for src, detail, n, _label in rows_:
        yi = SOURCE_Y[src]
        trained = src in TRAINED_ON
        ax.barh(
            yi,
            n,
            height=0.7,
            color=SRC_COLOR["measured"] if trained else SRC_FILL["measured"],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        ax.text(
            n * 1.5,
            yi,
            detail,
            fontsize=4.2,
            ha="left",
            va="center",
            color="black",
            zorder=3,
            linespacing=1.05,
        )
    ax.set_yticks([SOURCE_Y[r[0]] for r in rows_])
    ax.set_yticklabels([r[3] for r in rows_], fontsize=4.8)
    ax.set_ylim(-0.6, 8.6)
    ax.set_xscale("log")
    ax.set_xlim(30, 1e10)
    ax.set_xticks([1e2, 1e4, 1e6, 1e8])
    ax.set_xlabel("records read: cells, wells or anchor doses")
    ax.grid(axis="x", which="major", color="#DDDDDD", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
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
    num("nAnchorCompounds", int(c.loc["host_anchor"].n_compounds))
    num("nTitrationConditions", int(c.loc["host_ex21"].n_conditions))
    num("nTitrationWells", int(c.loc["host_ex21"].n_cells))
    num("nIsoboleWells", int(c.loc["host_isobole"].n_cells))
    handles = [
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
            label="held out: the test of this study",
        ),
    ]
    # above the axes, not inside it, where it covered the isobole bar (owner,
    # 2026-10-18)
    n_rows = panel_legend(ax, handles, ncol=2)
    title(ax, "the sources: size and culture", n_rows)


def draw_label_shapes(ax: Any, R: Reads) -> None:
    """Panel b, right: what a label is in each source, on the same rows as the bars.

    One histogram per source of the value the trainer reads before it standardizes
    the source (``source_labels.py``): a screen's value is one deletion strain's growth
    under one compound at one dose, relative to that screen's control, oriented
    sick-negative and on the screen's own scale, so each band spans its own 0.5th to
    99.5th percentile with the range at its ends and zero marked; a well of this study
    is the wild-type host's relative growth rate on one shared 0 to 1.6 scale with 1
    marked. The anchors are not measured: they are asserted at 0.70 at a screen's IC30
    dose and 1.0 compound-free, and drawn as two spikes.
    """
    hist = R.label_hist
    host_max = 1.6
    band_h, pad = 0.50, 0.05
    for src, yi in SOURCE_Y.items():
        if src == "host_anchor":
            for value, height in ((IC30_FITNESS, 1.0), (1.0, 0.25)):
                x = pad + (1 - 2 * pad) * value / host_max
                ax.plot(
                    [x, x],
                    [yi - 0.36, yi - 0.36 + band_h * height],
                    color=SRC_FILL["measured"],
                    linewidth=1.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot(
                    [x, x],
                    [yi - 0.36, yi - 0.36 + band_h * height],
                    color="black",
                    linewidth=0.4,
                    zorder=3,
                )
                ax.text(
                    x,
                    yi + 0.31,
                    f"{value:g}",
                    fontsize=3.9,
                    ha="center",
                    va="center",
                    color=GRAY,
                )
            ax.text(
                0.02,
                yi + 0.31,
                "asserted",
                fontsize=4.2,
                ha="left",
                va="center",
                color=GRAY,
            )
            continue
        key = LABEL_ROW.get(src, src)
        h = hist[hist.source == key].sort_values("bin_lo")
        lo, hi = float(h.bin_lo.min()), float(h.bin_hi.max())
        if src.startswith("host"):
            lo, hi = 0.0, host_max
        edges = np.concatenate([h.bin_lo.to_numpy(), [h.bin_hi.iloc[-1]]])
        xs = pad + (1 - 2 * pad) * (edges - lo) / (hi - lo)
        counts = h["count"].to_numpy(dtype=float)
        ys = yi - 0.36 + band_h * counts / counts.max()
        trained = src in TRAINED_ON
        ax.fill_between(
            xs,
            yi - 0.36,
            np.append(ys, ys[-1]),
            step="post",
            facecolor=SRC_COLOR["measured"] if trained else SRC_FILL["measured"],
            edgecolor="black",
            linewidth=0.3,
            zorder=2,
        )
        mark = 1.0 if src.startswith("host") else 0.0
        xm = pad + (1 - 2 * pad) * (mark - lo) / (hi - lo)
        ax.plot(
            [xm, xm], [yi - 0.42, yi + 0.24], color="black", linewidth=0.5, zorder=4
        )
        label = LABEL_SHORT[src]
        if src.startswith("host"):
            row = R.label_sum.set_index("source").loc[LABEL_ROW[src]]
            label = label.format(p=float(row.share_no_growth))
        ax.text(
            0.5, yi + 0.36, label, fontsize=4.2, ha="center", va="center", color=GRAY
        )
        if not src.startswith("host"):
            ax.text(
                pad,
                yi - 0.34,
                f"{lo:.1f}",
                fontsize=3.9,
                ha="left",
                va="bottom",
                zorder=5,
            )
            ax.text(
                1 - pad,
                yi - 0.34,
                f"{hi:.1f}",
                fontsize=3.9,
                ha="right",
                va="bottom",
                zorder=5,
            )
    for src in ("host_ex21", "host_ex23", "host_isobole"):
        row = R.label_sum.set_index("source").loc[LABEL_ROW[src]]
        num(f"labelNoGrowth{macro_name(src.capitalize())}", float(row.share_no_growth))
    summ = R.label_sum.set_index("source")
    for src in ("vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"):
        num(
            f"labelBelowZero{macro_name(src.capitalize())}",
            float(summ.loc[src].share_below_zero),
        )
    ax.text(pad, -0.34, "0", fontsize=3.9, ha="left", va="bottom", zorder=5)
    ax.text(
        1 - pad,
        -0.34,
        f"{host_max:.1f}",
        fontsize=3.9,
        ha="right",
        va="bottom",
        zorder=5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 8.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(
        "a screen: its own scale, 0 marked\nthis study: host growth rate, 0 to 1.6, 1 marked",
        fontsize=4.4,
        linespacing=1.1,
    )
    title(ax, "what a label is", 1)


#: the screens as panel c names them
SCREEN_SHORT = {
    "vanacloig": "Vanacloig",
    "hoepfner_hop": "Hoepfner HOP",
    "hillenmeyer_het": "Hillenmeyer HET",
    "wildenhain": "Wildenhain",
}


def draw_join(ax: Any, R: Reads) -> None:
    """Panel c: what the sources share, and what is trained on and tested on.

    Two blocks of text, top to bottom in a tall narrow panel. The first names the
    three axes every source shares (the gene table, the compound table, the growth
    reading, with the anchor that ties a screen's IC30 dose to the host's fitness). The
    second says what is trained on and what is tested on, on the public side and on
    this study's, with a bar at the left in the fills of panel b (dark trained on,
    light held out). Every count is read from the trainer's own counts file, 033's
    cross-source table and ``source_labels.py``.
    """
    blank(ax, 100, 150)
    c = R.counts.set_index("source")
    screens = ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"]
    genes = c.loc[screens].n_genes
    shared = R.agreement.shared_compounds
    over = R.overlap
    held_names = [INHIB_NAME[LONG_TO_ABBR[n]] for n in over.inhibitor.unique()]
    held_screens = [SCREEN_SHORT[s] for s in over.source.unique()]
    n_anchor = int(c.loc["host_anchor"].n_conditions)
    n_cmp = int(c.loc["vanacloig"].n_compounds)
    wc = wet_counts(R)
    num("joinGenesMin", int(genes.min()))
    num("joinGenesMax", int(genes.max()))
    num("joinSharedMin", int(shared.min()))
    num("joinSharedMax", int(shared.max()))
    num("joinInhibitorsInScreens", len(held_names))
    num("icThirtyFitness", IC30_FITNESS)
    fs = 4.2
    dark, light = SRC_COLOR["measured"], SRC_FILL["measured"]
    kw = dict(fontsize=fs, ha="left", va="top", linespacing=1.2)
    head_kw = dict(fontsize=4.6, ha="left", va="top", style="italic")
    # ---- what every source shares; the head word of each paragraph in bold
    ax.text(0, 150, "what every source shares", **head_kw)
    ax.text(
        0,
        141,
        r"$\bf{genes}$: one deletion collection,"
        f"\n{GENE_TOKENS:,} gene tokens; a screen covers"
        f"\n{int(genes.min()):,} to {int(genes.max()):,} of them",
        **kw,
    )
    ax.text(
        0,
        122,
        r"$\bf{compounds}$: one structure table"
        f"\n(FCFP4 counts); screens share {int(shared.min())} to {int(shared.max())};"
        f"\n{' and '.join(held_names)} are in {', '.join(held_screens)}"
        f"\nand this study, the other {len(INHIB) - len(held_names)} in no screen",
        **kw,
    )
    ax.text(
        0,
        97,
        r"$\bf{growth}$: a screen reads a strain against"
        "\nits control, this study the wild type;"
        f"\nan IC30 dose asserts the wild type at {IC30_FITNESS:.2f}:"
        f"\n{n_anchor} anchors",
        **kw,
    )
    # ---- trained on, tested on
    ax.text(0, 70, "what is trained on, what is tested on", **head_kw)
    for y_top, y_bot, fill, text in (
        (
            61,
            41,
            dark,
            r"$\bf{trained\ on}$: every screen, Vanacloig in"
            f"\ncompound-cold folds (all but one of {n_cmp});"
            f"\n{n_anchor} anchors and the titration of this study"
            f"\n({int(c.loc['host_ex21'].n_conditions)} conditions, "
            f"{int(c.loc['host_ex21'].n_cells)} wells)",
        ),
        (
            35,
            9,
            light,
            r"$\bf{tested\ on}$: the held-out compounds'"
            "\nprofiles, centered Spearman over genes;"
            f"\nthis study's {wc['n_combinations']} combinations (growth"
            "\nAUROC, fitness Spearman) and its"
            f"\n{int(c.loc['host_isobole'].n_cells)} isobole wells (the front)",
        ),
    ):
        ax.add_patch(
            plt.Rectangle(
                (0, y_bot),
                2.5,
                y_top - y_bot,
                facecolor=fill,
                edgecolor="black",
                linewidth=0.3,
                zorder=2,
            )
        )
        ax.text(5, y_top, text, **kw)
    title(ax, "what is shared, trained on, tested on")


def strip(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    n: int,
    fill: str,
    color: str,
    dark: tuple[int, ...] = (),
    lw: float = 0.3,
) -> None:
    """A row of ``n`` cells, a token strip; the ``dark`` cells are filled with ``color``."""
    cw = w / n
    for i in range(n):
        ax.add_patch(
            plt.Rectangle(
                (x + i * cw, y),
                cw,
                h,
                facecolor=color if i in dark else fill,
                edgecolor=color,
                linewidth=lw,
                zorder=2,
            )
        )


def token(ax: Any, x: float, y: float, s: float, color: str) -> None:
    """One compound token: a filled square ``s`` wide at (x, y), its lower-left."""
    ax.add_patch(
        plt.Rectangle(
            (x, y), s, s, facecolor=color, edgecolor="black", linewidth=0.3, zorder=3
        )
    )


def draw_evaluation(ax: Any, R: Reads) -> None:
    """Panel d: the model in three stages, top to bottom, the equations set in LaTeX.

    Stage one featurizes one inhibitor (furfural, drawn at the ACS standard) as its
    FCFP4 count fingerprint and passes it through the compound MLP to one token. Stage
    two puts that token in front of the gene tokens of the cell graph transformer; one
    further layer reads them together and ONE head reads either the rows at a strain's
    deleted genes (its profile) or, for the host, which carries no deletion, the mean
    over genes (its growth). Stage three composes a mixture: every token is read alone
    through the same head and the readings combine as Bliss independence, a product on
    the host's fitness scale and a sum on a screen's log response. The counts are the
    trainer's; the equations are typeset by LaTeX (``usetex``) in the sans face of the
    labels around them. The panel is tall and narrow (about 32 by 47 mm), so the flow
    runs down the panel: one x unit is about 0.32 mm, one y unit about 0.31 mm, and
    the structure is 13.4 mm wide (42 units).
    """
    blank(ax, 100, 150)
    wc = wet_counts(R)
    n_fit = int(R.ladder_scores.n_fit_compounds.max())
    num("nFitCompounds", n_fit)
    num("nGeneTokens", GENE_TOKENS)
    num("nFingerprintBits", FINGERPRINT_BITS)
    num("nCombinationsScored", wc["n_combinations"])
    cgt = SRC_COLOR["cgt"]
    meas, meas_fill = SRC_COLOR["measured"], SRC_FILL["measured"]
    fs = 4.2
    stage_kw = dict(fontsize=4.4, ha="left", va="top", fontweight="bold")
    # ---- stage one: a molecule becomes a token
    ax.text(0, 150, "1  featurize", **stage_kw)
    y1 = 130.0
    box_, _w, _h = acs_structure(
        ax, R.smiles["furfural"], (19, y1), box_alignment=(0.5, 0.5)
    )
    box_.set_in_layout(False)
    ax.text(19, y1 - 13.0, "furfural", fontsize=fs, ha="center", va="top")
    arrow(ax, 41, y1, 46, y1)
    strip(ax, 47, y1 - 2.4, 15, 4.8, 10, meas_fill, meas, dark=(1, 4, 7))
    ax.text(
        54.5,
        y1 - 4.0,
        f"{FINGERPRINT_BITS:,}\nFCFP4 counts",
        fontsize=fs,
        ha="center",
        va="top",
        linespacing=1.1,
    )
    arrow(ax, 63, y1, 68, y1)
    box(ax, 69, y1 - 4.2, 15, 8.4, "MLP", cgt, "white", fs=fs)
    arrow(ax, 85, y1, 90, y1)
    token(ax, 91, y1 - 2.6, 5.2, cgt)
    ax.text(
        0,
        y1 - 18.0,
        r"$t_i = \mathrm{MLP}(\mathrm{FCFP4}(m_i))$ per inhibitor;",
        fontsize=fs,
        ha="left",
        va="top",
        usetex=True,
    )
    ax.text(
        0,
        y1 - 23.5,
        "the dose scales the medium offset",
        fontsize=fs,
        ha="left",
        va="top",
    )
    # ---- stage two: the token in front of the genome, one head
    ax.text(0, 97, "2  read the cell", **stage_kw)
    y2 = 84.0
    token(ax, 0.5, y2 - 2.6, 5.2, cgt)
    strip(ax, 7, y2 - 2.6, 52, 5.2, 13, SRC_FILL["ridge"], GRAY, dark=(4, 9))
    ax.text(
        33,
        y2 - 4.0,
        f"{GENE_TOKENS:,} gene tokens;\ndeleted genes dark, the host none",
        fontsize=fs,
        ha="center",
        va="top",
        linespacing=1.1,
    )
    arrow(ax, 61, y2, 66, y2)
    box(ax, 67, y2 - 5.0, 33, 10, "one CGT\nlayer + head", cgt, "white", fs=fs)
    arrow(ax, 30, y2 - 13.0, 30, y2 - 19.0)
    arrow(ax, 83, y2 - 10.5, 83, y2 - 19.0)
    box(
        ax,
        0.5,
        y2 - 34.0,
        47,
        15,
        r"a strain:" "\n" r"rows at its" "\n" r"deleted genes $\to \hat r_g$",
        cgt,
        "white",
        fs=fs,
        usetex=True,
    )
    box(
        ax,
        52,
        y2 - 34.0,
        48,
        15,
        r"the host:" "\n" r"the mean" "\n" r"over genes $\to \hat f$",
        cgt,
        "white",
        fs=fs,
        usetex=True,
    )
    ax.text(
        0,
        y2 - 36.5,
        r"$\hat r_g$: a profile, $\hat f$: growth",
        fontsize=fs,
        ha="left",
        va="top",
        usetex=True,
    )
    # ---- stage three: a mixture composes as Bliss
    ax.text(0, 39, "3  compose a mixture", **stage_kw)
    y3 = 27.0
    for name, x in (("furfural", 0.5), ("acetic acid", 35.0), ("5-HMF", 73.0)):
        token(ax, x, y3 - 2.6, 5.2, cgt)
        ax.text(x + 6.5, y3, name, fontsize=fs, ha="left", va="center")
    ax.text(
        0,
        y3 - 7.0,
        "each token read alone through the head,\nthen composed as Bliss independence:",
        fontsize=fs,
        ha="left",
        va="top",
        linespacing=1.1,
    )
    ax.text(
        0,
        y3 - 19.0,
        r"$\hat f_{\mathrm{mix}} = \prod_i \hat f_i$ (fitness scale)",
        fontsize=fs,
        ha="left",
        va="top",
        usetex=True,
    )
    ax.text(
        0,
        y3 - 26.0,
        r"$\hat r_{g,\mathrm{mix}} = \sum_i \hat r_{g,i}$ (log scale)",
        fontsize=fs,
        ha="left",
        va="top",
        usetex=True,
    )
    title(ax, "the model: one token per inhibitor, one head")


#: the ladder's encoders as a tick label names them, short enough for a four-column row
SHORT_ENCODER = {
    "fcfp4_count": "FCFP4",
    "ecfp4_count": "ECFP4",
    "ecfp4_bit": "ECFP4 bits",
    "maccs": "MACCS",
    "rdkit_2d": "RDKit 2D",
    "mol2vec": "mol2vec",
    "mole_static": "MolE",
    "unimol_v1": "Uni-Mol",
    "chemberta2_mtr": "ChemBERTa",
    "chemberta2_mlm": "ChemBERTa MLM",
    "molformer_xl": "MoLFormer",
    "roberta_zinc_480m": "RoBERTa ZINC",
    "fingerprints": "fingerprints",
    "pretrained": "pretrained",
    "all": "all",
}


def short_arm(model: str, kernel: str) -> str:
    """A ladder arm's tick label: ``krr, linear:fcfp4_count`` to ``KRR lin, FCFP4``."""
    kind, _, encoder = kernel.partition(":")
    kind = {"linear": "lin", "rbf": "RBF", "combo": "combo"}.get(kind, kind)
    return f"{model.upper()} {kind}, {SHORT_ENCODER.get(encoder, encoder)}"


def draw_ladder(ax: Any, R: Reads) -> None:
    """Panel d: the 038 round-1 ladder, nested ridge on fingerprint counts first."""
    top = ladder_table(R, 10)
    data = []
    for _, r in top.iterrows():
        data.append(
            (
                short_arm(r.model, r.kernel),
                float(r.spearman_median),
                None,
                "ridge",
                False,
            )
        )
    ridge = R.ladder_row("krr", "linear:fcfp4_count")
    num("ladderRidge", float(ridge.spearman_median))
    num("ladderRidgeMean", float(ridge.spearman_mean))
    num("ladderNArms", int(len(R.ladder[(R.ladder.target == "centered")])))
    num("ladderCompounds", int(R.ladder.compounds.max()))
    num("ladderSecond", float(top.iloc[1].spearman_median))
    num("ladderTenth", float(top.iloc[-1].spearman_median))
    bars(ax, data, "centered Spearman, compound-cold", 0.40, step=0.1)
    ax.tick_params(axis="y", labelsize=4.4)
    title(ax, "Vanacloig: ridge leads the ladder")


def draw_cgt_vs_ridge(ax: Any, R: Reads) -> None:
    """Panel e: the transformer against ridge, as far as round 2 has run."""
    ridge = R.ladder_row("krr", "linear:fcfp4_count")
    data = [("nested ridge, FCFP4", float(ridge.spearman_median), None, "ridge", False)]
    for frame, arm, key in (
        (R.bil, "bilinear", "bilinear"),
        (R.envenc, "env. encoder", "envenc"),
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
            label = f"{arm}, " + ("one seed" if member == "seed0" else "ensemble")
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
            ("mixture trainer, ensemble", float(ens.value.median()), None, "cgt", False)
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
    bars(ax, data, "centered Spearman, compound-cold", 0.40, step=0.1)
    ax.tick_params(axis="y", labelsize=4.4)
    title(ax, "Vanacloig: no arm reaches ridge")


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
    ax.set_ylabel("centered Spearman")
    vgrid(ax, 0.1)
    handles = [
        Line2D(
            [],
            [],
            color=SRC_COLOR["ridge"],
            linewidth=0.8,
            marker="o",
            markersize=2.0,
            label="ridge, FCFP4 counts",
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
                label="environment encoder (038)",
            )
        )
    n = panel_legend(ax, handles, ncol=1)
    title(ax, "ridge gains, encoder trails", n)


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
    title(
        ax,
        f"host IC30 is {NUM['ratioFf']:.1f}x, {NUM['ratioHmf']:.1f}x the screen's",
        n,
    )


def draw_het_rule(ax: Any, R: Reads) -> None:
    """Panel h: how a pair's deletion profile composes from its singles, on HET."""
    d = R.het_rule
    rules = [("r2_mean", "mean"), ("r2_sum", "bliss"), ("r2_max", "hsa")]
    names = {
        "r2_mean": "mean of\nthe singles",
        "r2_sum": "sum\n(Bliss, log2)",
        "r2_max": "highest\nsingle",
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
    ax.set_xticklabels([names[c] for c, _ in rules], fontsize=4.6, linespacing=1.1)
    ax.set_ylim(-1.2, 1.0)
    ax.axhline(0.0, color="black", linewidth=0.4, zorder=1)
    ax.set_ylabel("R-squared")
    vgrid(ax, 0.5)
    title(ax, "Hillenmeyer HET pairs compose as the mean")


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
                f"{short}: screen repeats",
                float(row.measured_reliability),
                None,
                "measured",
                False,
            )
        )
        data.append(
            (f"{short}: ridge, LOCO", float(row.spearman), None, "ridge", False)
        )
        data.append(
            (
                f"{short}: ridge, 038 folds",
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
                f"{short}: transformer ({len(c)} runs)",
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
    bars(ax, data, "Spearman vs measured", 0.9, step=0.2, xmin=-0.2)
    ax.tick_params(axis="y", labelsize=4.4)
    # the shared-term count is stated in the caption (goSharedTermsStore); drawn in
    # the panel it sat on the bars at every corner
    title(ax, "furfural predicted, 5-HMF noise")


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
    # every sheet model is scored on these wells (the numbers go to the caption),
    # but only the dose-offset model is drawn, as one solid curve per inhibitor: the
    # other two read one flat value per compound, and eighteen blue lines in three
    # dash patterns could not be told apart (the owner, 2026-10-15)
    for sweep, _style, _label, tag in SHEET_MODELS:
        h = R.host_mean(sweep)
        h = h[(h.run == "ex21") & (h.n_compounds == 1)]
        for ab in INHIB:
            name = INHIB_LONG[ab]
            sub = h[h.compounds == name].copy()
            if sub.empty or sweep != DOSE_ARM:
                continue
            fit = R.fit("ex21", ab)
            sub["x"] = sub.g_per_l.astype(float) / float(fit.ic50_g_per_l)
            sub = sub.sort_values("x")
            ax.plot(
                sub.x.to_numpy(),
                sub.predicted.to_numpy(),
                color=SRC_COLOR["cgt"],
                linewidth=0.6,
                zorder=2,
            )
        singles = h[h.n_compounds == 1]
        rho = float(
            pd.Series(singles.predicted.to_numpy()).corr(
                pd.Series(singles.observed.to_numpy()), method="spearman"
            )
        )
        num(f"ex21{tag}Spearman", rho)
        num(f"ex21{tag}Conditions", int(len(singles)))
    num("nEx21Fits", int((R.fits.source == "ex21").sum()))
    num("nBootstrapDraws", int(R.boot.draw.max()) + 1)
    num("hillMax", float(R.fits[R.fits.source == "ex21"].h.max()))
    num("hillMin", float(R.fits[R.fits.source == "ex21"].h.min()))
    ax.set_xscale("log")
    ax.set_xlim(0.05, 12)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("dose, in units of the IC50")
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
                label="Hill fit per inhibitor",
            ),
            Line2D(
                [],
                [],
                color=SRC_COLOR["cgt"],
                linewidth=0.6,
                label="model with the dose offset, per inhibitor",
            ),
        ],
        ncol=1,
    )
    title(ax, "titrations of this study", n)


def draw_combinations(ax: Any, R: Reads) -> None:
    """Panel k: the combinations observed against Bliss and Loewe, by size."""
    comb = R.mix_comb[(R.mix_comb.call == "served") & (R.mix_comb.grew)].copy()
    # each model's prediction per combination, joined on the set of compounds: the
    # fitted Bliss-composed model (dark) and the same with the dose offset (light)
    for arm, column, tag in (
        (MODEL_ARM, "model", ""),
        (DOSE_ARM, "model_dose", "Dose"),
    ):
        h = R.host_mean(arm)
        h = h[h.run == "ex23"]
        by_set = {
            frozenset(c.split("|")): float(v) for c, v in zip(h.compounds, h.predicted)
        }
        comb[column] = [
            by_set.get(frozenset(c.split("|")), float("nan")) for c in comb.combination
        ]
        all_ex23 = h[h.n_compounds >= 1]
        num(
            f"model{tag}Auroc",
            float(roc_auc_score(all_ex23.grew.astype(bool), all_ex23.predicted)),
        )
        num(
            f"model{tag}GrownSpearman",
            float(
                pd.Series(comb[column].to_numpy()).corr(
                    pd.Series(comb.observed_grown_mean.to_numpy()), method="spearman"
                )
            ),
        )
        num(f"model{tag}Folds", int(h.folds.iloc[0]))
        num(f"model{tag}Runs", int(h.runs.iloc[0]))
    sizes = sorted(comb.n_compounds.unique())
    width = 0.16
    x = np.arange(len(sizes))
    series = [
        ("observed_grown_mean", SRC_COLOR["measured"]),
        ("bliss_ex23", SRC_COLOR["bliss"]),
        ("loewe_ex21", SRC_COLOR["loewe"]),
        ("model", SRC_COLOR["cgt"]),
        ("model_dose", SRC_FILL["cgt"]),
    ]
    for j, (col, color) in enumerate(series):
        means = [float(comb[comb.n_compounds == s][col].mean()) for s in sizes]
        ax.bar(
            x + (j - 2) * width,
            means,
            width=width,
            color=color,
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
    ax.set_ylabel("fitness")
    ax.set_ylim(0, 1.7)
    vgrid(ax, 0.4)
    # the growth AUROCs are the next panel's ladder; the fitness ranking is stated here
    ax.text(
        0.98,
        0.97,
        f"fitness Spearman\nBliss {bliss.spearman:.2f}, Loewe {loewe.spearman:.2f}\n"
        f"model {NUM['modelGrownSpearman']:.2f}, + dose "
        f"{NUM['modelDoseGrownSpearman']:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.2,
        zorder=5,
    )
    leader = "Loewe" if loewe.spearman > bliss.spearman else "Bliss"
    assert (loewe.auroc > bliss.auroc) == (leader == "Loewe"), "the rules split"
    title(ax, f"this study: {leader} leads")


def front_steps(alive: NDArray[np.bool_], x0: float) -> tuple[list[float], list[float]]:
    """The growth front of a monotone alive mask [ny, nx] as a staircase of cell edges.

    It starts on the grid's left edge at the top of the highest row that grows (the
    acetic acid dose the front crosses on its own) and ends on the bottom edge at the
    right of the growing cells of row zero (the inhibitor dose it crosses on its own),
    so a rule's front always meets both edges, as the dose-response it is built from
    does. Empty when nothing grows.
    """
    width = [int(np.max(np.where(row)[0])) + 1 if row.any() else 0 for row in alive]
    rows = [j for j, w in enumerate(width) if w > 0]
    if not rows:
        return [], []
    j_top = max(rows)
    xs, ys = [x0], [float(j_top + 1)]
    for j in range(j_top, -1, -1):
        xs.extend([x0 + width[j], x0 + width[j]])
        ys.extend([float(j + 1), float(j)])
    return xs, ys


#: a front's white halo: a solid colored line reads on the yellow fitness fill and on
#: an empty cell alike, where a dashed line of the same color did not
FRONT_HALO = [
    patheffects.Stroke(linewidth=1.8, foreground="white"),
    patheffects.Normal(),
]


def draw_isoboles(ax: Any, R: Reads) -> None:
    """Panel l: the three isobole grids as observed fitness, with each rule's front."""
    cells = R.isobole_cells
    runs = list(R.isobole_sum.run)
    # Each run has its OWN dose ladder. Indexing every run into one pooled ladder left
    # each grid as a few sparse stripes, which is what a shared axis costs here.
    gap = 3.0
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
                    markersize=0.5,
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
                linewidth=0.6,
                zorder=6,
            )
        )
        srow = R.isobole_sum[R.isobole_sum.run == run].iloc[0]
        tau = float(srow.tau)
        # A rule's front is where its predicted fitness crosses the growth threshold,
        # not where it is merely positive: the rule predicts a fitness for every cell,
        # so a positive-value test puts the front on the grid's edge. Every front is a
        # solid line over a white halo, so it reads on the yellow fill and on white
        # alike (owner, 2026-10-15: no dashed lines here).
        for column, src in (("bliss", "bliss"), ("loewe", "loewe")):
            rule = np.zeros((len(doses_y), len(doses_x)), dtype=bool)
            for _, r in sub.iterrows():
                rule[
                    doses_y.index(r.acetic_acid_g_per_l),
                    doses_x.index(r.inhibitor_g_per_l),
                ] = bool(r[column] >= tau)
            # the rule's surface falls with either dose, so its closure is itself;
            # the staircase runs from the left edge to the bottom edge
            xs, ys = front_steps(monotone_alive(rule), x0)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=SRC_COLOR[src],
                    linewidth=0.8,
                    path_effects=FRONT_HALO,
                    zorder=5,
                )
        # Each model's front: its predicted fitness per cell against the same growth
        # threshold, then the monotone closure (growth never returns as either dose
        # rises), so the front is read on the rules' own axis; the cells the closure
        # drops are the predicted surface's growth islands. The fitted model is
        # drawn; the dose-offset arm is scored (its numbers go to the caption and to
        # supplement eight) and not drawn, so the grid carries three solid fronts.
        inhibitor = [p for p in srow["pair"].split("|") if p != "acetic acid"][0]
        for arm, drawn, tag in ((MODEL_ARM, True, ""), (DOSE_ARM, False, "Dose")):
            hp = R.host_mean(arm)
            hp = hp[hp.run == run]
            grid = np.full((len(doses_y), len(doses_x)), np.nan)
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
                f"isobole{macro_name(run.capitalize())}Model{tag}Islands",
                int(alive.sum() - closed.sum()),
            )
            num(
                f"isobole{macro_name(run.capitalize())}Model{tag}Cells",
                int(np.isfinite(grid).sum()),
            )
            mx, my = front_steps(closed, x0)
            if mx and drawn:
                ax.plot(
                    mx,
                    my,
                    color=SRC_COLOR["cgt"],
                    linewidth=0.8,
                    path_effects=FRONT_HALO,
                    zorder=5,
                )
        # the grid's inhibitor is named on its lower edge (the x axis of that grid);
        # acetic acid is every grid's y axis and is named once, on the panel's
        flag = "\n(flagged)" if isinstance(srow["flag"], str) and srow["flag"] else ""
        ax.text(
            x0 + nx / 2,
            -0.6,
            f"{INHIB_NAME[LONG_TO_ABBR[inhibitor]]}{flag}",
            fontsize=4.4,
            ha="center",
            va="top",
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
    ax.set_ylim(-0.4, ny + 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("inhibitor dose, rising to the right", labelpad=10.0)
    ax.set_ylabel("acetic acid, rising upward")
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP_FITNESS, norm=norm)
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, shrink=0.7)
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
                label="model front (monotone closure)",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markersize=1.2,
                linestyle="none",
                label="a cell that grew",
            ),
        ],
        ncol=2,
    )
    title(ax, "this study's isobole grids: fitness and each front", n)


def fig_main(R: Reads) -> dict[str, str]:
    """The main sheet, a to o: four rows.

    Row one is the data and the model (the inhibitors; each source's size, culture and
    label; what the sources share and what is trained on and tested on; the model).
    Row two is the public benchmark at the gene level and the dose (the ladder, the
    unseen compound, the learning curve, dose in two media). Row three is composition
    (how pairs compose, the predicted profile, the pre-registered similarity test, the
    titrations). Row four is the wet lab (the combinations, the isobole grids, and the
    source ladder read on the public score and on the wet lab side by side).
    """
    fig = new_sheet(170)
    fig.get_layout_engine().set(
        rect=(0, 0.012, 1, 0.962), h_pad=3.0 / 72.0, hspace=0.012
    )
    axes = rows(
        fig,
        4,
        [5, 4, 4, 4],
        widths=[
            None,
            [1.0, 1.0, 1.0, 0.95],
            [1.1, 1.1, 0.95, 0.85],
            [0.8, 1.4, 0.95, 0.75],
        ],
        heights=[1.6, 1.0, 1.0, 1.0],
        # the top row by hand: a, c and d run to the row's bottom; the two halves of
        # b stop short by their tick labels and x labels, and the gap after a holds
        # b's y tick labels (owner, 2026-10-15: the bottom edges line up)
        placed={
            0: [
                [0.000, 0.02, 0.205, 0.86],
                [0.308, 0.20, 0.148, 0.66],
                [0.463, 0.20, 0.150, 0.66],
                [0.628, 0.02, 0.185, 0.86],
                [0.830, 0.02, 0.170, 0.86],
            ]
        },
    )
    a, b1, b2, c, d = axes[0:5]
    e, f, g, h = axes[5:9]
    i, j, k, l_ = axes[9:13]
    m, n, o1, o2 = axes[13:17]
    draw_evaluation(d, R)
    draw_join(c, R)
    draw_sources(b1, R)
    draw_label_shapes(b2, R)
    draw_ladder(e, R)
    draw_cgt_vs_ridge(f, R)
    draw_curve(g, R)
    draw_dose(h, R)
    draw_het_rule(i, R)
    draw_profiles(j, R)
    draw_similarity_vs_deviation(
        k, R, title_text="profile similarity does not predict synergy"
    )
    draw_single_agents(l_, R)
    draw_combinations(m, R)
    draw_isoboles(n, R)
    # the two ladders read NUM["ladderRidge"] and NUM["loeweAuroc"], which e and m set;
    # they share their rows, the wet-lab half without its own labels
    draw_bliss_ladder(
        o1, R, "gene", title_text="more screens: public flat", compact=True, paired=True
    )
    draw_bliss_ladder(
        o2,
        R,
        "growth",
        title_text="wet lab: zero-shot rises",
        compact=True,
        paired=True,
    )
    sheet_legend(
        fig,
        ["measured", "bliss", "loewe", "hsa", "mean", "ridge", "cgt"],
        hatch=False,
        ncol=3,
    )
    # a is drawn last: it measures its axes after every other panel has set its
    # decorations, because its structures have a fixed size and do not scale
    draw_molecules(a, R)
    for name, ax in (("B", b1), ("C", c), ("D", d)):
        extent = ax.get_window_extent()
        num(f"panel{name}WidthMm", round(extent.width / fig.dpi * 25.4, 1))
    lettered = [a, b1, c, d, e, f, g, h, i, j, k, l_, m, n, o1]
    letters_status(
        fig,
        lettered,
        # c and d are layouts (the join and the model), not measurements; every other
        # panel draws values that are on file
        ["ready", "ready", "todo", "todo"] + ["ready"] * 11,
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


def draw_similarity_vs_deviation(
    ax: Any,
    R: Reads,
    title_text: str | None = None,
    inside: bool = False,
    legend_cols: int = 1,
) -> None:
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
    ax.set_xlabel("profile similarity of the pair")
    ax.set_ylabel("observed minus Bliss")
    vgrid(ax, 0.1)
    hgrid(ax, 0.5)
    handles = [
        Patch(
            facecolor=SRC_COLOR["measured"],
            edgecolor="black",
            linewidth=0.4,
            label="measured where one exists",
        ),
        Patch(
            facecolor=SRC_COLOR["ridge"],
            edgecolor="black",
            linewidth=0.4,
            label="every profile by ridge",
        ),
    ]
    if inside:
        ax.legend(handles=handles, loc="upper left", fontsize=4.4, handlelength=1.2)
        n = 0
    else:
        n = panel_legend(ax, handles, ncol=legend_cols)
    title(ax, title_text or "similarity against deviation, both profile versions", n)


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


#: the arms of the gene readout (rounds three to five) that carry each public screen,
#: and the arms of the Bliss cell readout (round ten) that do
GENE_SOURCE_ARMS = {
    "hoepfner_hop": ("mix_vanhop_e150", "mix_vanhophet_e150"),
    "hillenmeyer_het": ("mix_vanhophet_e150",),
    "wildenhain": ("mix_vanwil_e150", "mix_vanhophetwil_e150"),
}
BLISS_SOURCE_ARMS = {
    "hoepfner_hop": (
        "mix_vanhop_e150_cell_bliss",
        "mix_vanhophet_e150_cell_bliss",
        "mix_vanhophetwil_e150_cell_bliss",
        "mix_vanhop_e150_cell_anchors_bliss",
        "mix_vanhophet_e150_cell_anchors_bliss",
        "mix_vanhop_e150_cell_bliss_doff",
    ),
    "hillenmeyer_het": (
        "mix_vanhet_e150_cell_bliss",
        "mix_vanhophet_e150_cell_bliss",
        "mix_vanhophetwil_e150_cell_bliss",
        "mix_vanhophet_e150_cell_anchors_bliss",
    ),
    "wildenhain": ("mix_vanwil_e150_cell_bliss", "mix_vanhophetwil_e150_cell_bliss"),
}


def draw_per_source(
    ax: Any,
    R: Reads,
    plain: dict[str, tuple[str, ...]] = GENE_SOURCE_ARMS,
    prefix: str = "r3",
    dose_arms: tuple[str, ...] | None = None,
) -> None:
    """What the transformer predicts on the public screens it learns from.

    For each auxiliary source, the held-out COMPOUND score of the ``plain`` arms that
    carry it (centered Spearman, mean over runs), beside the chemistry-feature
    learner's compound-cold score on the same source from ``source_basis.csv``; for
    Hoepfner also the held-out DOSE score of the FiLM dose arms (every arm with the
    readout, or ``dose_arms``). Numbers are written under ``prefix``.
    """
    d = R.rounds[
        R.rounds.member.str.startswith("seed")
        & (R.rounds.metric == "centered_spearman_median")
    ]
    basis = R.basis.set_index("source")
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
        num(f"{prefix}_src_{tag}_cgt", float(v.mean()))
        num(f"{prefix}_src_{tag}_cgt_n", int(len(v)))
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
            w = d[(d.task == source) & (d.subset == "heldout_dose")]
            if dose_arms is not None:
                w = w[w.arm.isin(dose_arms)]
            else:
                w = w[~w.arm.str.contains("bliss")]
            w = w.value
            num(f"{prefix}_src_hoepfner_dose", float(w.mean()))
            num(f"{prefix}_src_hoepfner_dose_n", int(len(w)))
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
        num(f"{prefix}_src_{tag}_ridge", ridge)
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
    letters_status(fig, axes, ["ready", "ready", "ready", "ready", "ready", "ready"])
    status_key(fig)
    return save(fig, "gate_si4_rounds")


#: round ten, the Bliss cell readout: the source ladder, the zero-shot ladder and
#: the dose arms, (arm, label, number tag)
BLISS_LADDER = [
    ("mix_van_e150_cell_bliss", "one source: Vanacloig", "one"),
    ("mix_vanhop_e150_cell_bliss", "two: with Hoepfner HOP", "hop"),
    ("mix_vanhet_e150_cell_bliss", "two: with Hillenmeyer HET", "het"),
    ("mix_vanwil_e150_cell_bliss", "two: with Wildenhain", "wil"),
    ("mix_vanhophet_e150_cell_bliss", "three: Hoepfner and Hillenmeyer", "three"),
    ("mix_vanhophetwil_e150_cell_bliss", "four: every public screen", "four"),
]
ZERO_SHOT_LADDER = [
    ("mix_van_e150_cell_anchors_bliss", "zero-shot, one source", "zsOne"),
    ("mix_vanhop_e150_cell_anchors_bliss", "zero-shot, two: Hoepfner", "zsTwo"),
    ("mix_vanhophet_e150_cell_anchors_bliss", "zero-shot, three", "zsThree"),
]
DOSE_ARMS = [
    ("mix_van_e150_cell_bliss", "no dose channel (the model)", "off"),
    ("mix_van_e150_cell_bliss_film", "dose FiLM on the tokens", "film"),
    ("mix_van_e150_cell_bliss_doff", "dose offset", "doff"),
    ("mix_van_e150_cell_ft_bliss_film", "FiLM, titration fine-tune", "ft"),
    ("mix_van_e150_cell_anchors_bliss", "zero-shot, no dose channel", "zsOff"),
    ("mix_van_e150_cell_anchors_bliss_film", "zero-shot, FiLM", "zsFilm"),
    ("mix_van_e150_cell_anchors_bliss_doff", "zero-shot, dose offset", "zsDoff"),
]


#: the ladder arms as a four-column row names them (by number tag)
SHORT_LADDER = {
    "one": "Vanacloig",
    "hop": "+ Hoepfner HOP",
    "het": "+ Hillenmeyer HET",
    "wil": "+ Wildenhain",
    "three": "+ HOP + HET",
    "four": "all four screens",
    "zsOne": "zero-shot, one",
    "zsTwo": "zero-shot, + HOP",
    "zsThree": "zero-shot, three",
}


def draw_bliss_ladder(
    ax: Any,
    R: Reads,
    readout: str,
    title_text: str | None = None,
    compact: bool = False,
    paired: bool = False,
) -> None:
    """Supplement eight, a and b: the source ladder under the Bliss cell readout.

    ``paired`` draws the two readouts on the same rows (every fitted and zero-shot
    arm on both), for the main sheet's side-by-side pair; the growth half then hides
    its tick labels and reads the gene half's.

    Every run on fold seed 0 (three seeds on three folds per arm, on both machines
    where an arm ran on both): the bar is the mean over runs, the whisker its
    standard deviation, the count on the label. ``gene`` reads the compound-cold
    centered Spearman against nested ridge; ``growth`` the growth AUROC over the 63
    combinations, with the zero-shot ladder below the fitted one and Loewe as the bar.
    """
    if readout == "gene":
        d = round_runs(
            R, "vanacloig", "test_compounds", "centered_spearman_median", None
        )
        ladder = BLISS_LADDER + ZERO_SHOT_LADDER if paired else BLISS_LADDER
        bar_label, bar_value, bar_src = (
            "nested ridge" if compact else "nested ridge on FCFP4 counts",
            float(NUM["ladderRidge"]),
            "ridge",
        )
        xlabel, xmax = (
            (
                "centered Spearman"
                if compact
                else "compound-cold centered Spearman (fold seed 0)"
            ),
            0.6,
        )
    else:
        d = round_runs(R, "ex23", "combinations", "growth_auroc", "served")
        ladder = BLISS_LADDER + ZERO_SHOT_LADDER
        bar_label, bar_value, bar_src = (
            "Loewe" if compact else "Loewe from the titration fits",
            float(NUM["loeweAuroc"]),
            "loewe",
        )
        xlabel, xmax = (
            ("growth AUROC" if compact else "growth AUROC over the 63 combinations"),
            1.0,
        )
    d = d[d.fold_seed == 0]
    data: list[tuple[str, Any, Any, str, bool]] = []
    for arm, label, tag in ladder:
        v = d[d.arm == arm].value
        num(f"r10_{tag}_{readout}_mean", float(v.mean()))
        num(f"r10_{tag}_{readout}_sd", float(v.std(ddof=1)))
        num(f"r10_{tag}_{readout}_n", int(len(v)))
        name = SHORT_LADDER[tag] if compact else label
        data.append(
            (
                f"{name} ({len(v)})" if compact else f"{label} (n = {len(v)})",
                float(v.mean()),
                float(v.std(ddof=1)),
                "cgt",
                False,
            )
        )
    data.append((bar_label, bar_value, None, bar_src, False))
    bars(ax, data, xlabel, xmax, step=0.2 if readout == "growth" else 0.1)
    if compact:
        ax.tick_params(axis="y", labelsize=4.2)
    if paired and readout == "growth":
        ax.tick_params(axis="y", labelleft=False)
    title(
        ax,
        title_text
        or "under Bliss: "
        + (
            "the source ladder at the gene level"
            if readout == "gene"
            else "the same arms on the wet lab"
        ),
    )


def draw_dose_arms(ax: Any, R: Reads) -> None:
    """Supplement eight, c: the dose channel under Bliss, every arm at three seeds.

    Two bars per arm: the growth AUROC over the 63 combinations (dark) and the
    Spearman of the host's prediction against the 180 titration wells (light), which
    is a fit for the arms trained on the titration and a prediction for the zero-shot
    arms; both mean over runs on fold seed 0 with the standard deviation.
    """
    auroc = round_runs(R, "ex23", "combinations", "growth_auroc", "served")
    titr = R.rounds[
        (R.rounds.task == "ex21")
        & (R.rounds.subset == "all_wells")
        & R.rounds.metric.str.startswith("fitness_spearman")
        & R.rounds.member.str.startswith("seed")
    ]
    y = np.arange(len(DOSE_ARMS))[::-1]
    h = 0.36
    for yi, (arm, _label, tag) in zip(y, DOSE_ARMS):
        a = auroc[(auroc.arm == arm) & (auroc.fold_seed == 0)].value
        t = titr[(titr.arm == arm) & (titr.fold_seed == 0)].value
        num(f"r10_dose_{tag}_auroc", float(a.mean()))
        num(f"r10_dose_{tag}_titration", float(t.mean()))
        num(f"r10_dose_{tag}_n", int(len(a)))
        for offset, v, color in (
            (h / 2, a, SRC_COLOR["cgt"]),
            (-h / 2, t, SRC_FILL["cgt"]),
        ):
            ax.barh(
                yi + offset,
                float(v.mean()),
                xerr=float(v.std(ddof=1)),
                height=h,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.5, "capsize": 1.2, "ecolor": "black"},
                zorder=2,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{label} (n = {NUM[f'r10_dose_{tag}_n']})" for _a, label, tag in DOSE_ARMS]
    )
    ax.set_ylim(-0.6, len(DOSE_ARMS) - 0.4)
    ax.set_xlim(-0.4, 1.0)
    ax.axvline(0.0, color="black", linewidth=0.4, zorder=3)
    ax.axvline(
        float(NUM["loeweAuroc"]), color=SRC_COLOR["loewe"], linewidth=0.8, zorder=1
    )
    ax.set_xlabel("growth AUROC (dark) and titration Spearman (light), mean over runs")
    hgrid(ax, 0.2)
    handles = [
        Patch(
            facecolor=SRC_COLOR["cgt"],
            edgecolor="black",
            linewidth=0.4,
            label="growth AUROC over the 63 combinations",
        ),
        Patch(
            facecolor=SRC_FILL["cgt"],
            edgecolor="black",
            linewidth=0.4,
            label="Spearman against the 180 titration wells",
        ),
        Line2D([], [], color=SRC_COLOR["loewe"], linewidth=0.8, label="Loewe"),
    ]
    n = panel_legend(ax, handles, ncol=1)
    title(ax, "the dose channel under Bliss: what it buys and what it costs", n)


def fig_si8(R: Reads) -> dict[str, str]:
    """Supplement eight: round ten, the Bliss cell readout at three seeds."""
    fig = new_sheet(118)
    fig.get_layout_engine().set(rect=(0, 0.012, 1, 0.94))
    axes = rows(fig, 2, 2)
    draw_bliss_ladder(axes[0], R, "gene")
    draw_bliss_ladder(axes[1], R, "growth")
    draw_dose_arms(axes[2], R)
    draw_per_source(
        axes[3],
        R,
        plain=BLISS_SOURCE_ARMS,
        prefix="r10",
        dose_arms=("mix_vanhop_e150_cell_bliss_film_dose",),
    )
    sheet_legend(fig, ["loewe", "ridge", "cgt"], hatch=False, ncol=3)
    letters_status(fig, axes, ["ready", "ready", "ready", "ready"])
    return save(fig, "gate_si8_round10")


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
    figures["si8"] = fig_si8(R)
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

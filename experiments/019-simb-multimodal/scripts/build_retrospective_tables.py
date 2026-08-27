# experiments/019-simb-multimodal/scripts/build_retrospective_tables.py
# [[experiments.019-simb-multimodal.scripts.build_retrospective_tables]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/build_retrospective_tables
r"""Emit the generated tables for notes-tex/019-simb-multimodal/.

Two tables, both written into `notes-tex/019-simb-multimodal/tables/` and both `\\input`
by that document. Neither is ever hand-edited; the repo rule is that a table in a note
comes from a committed script reading the real result files.

TABLE 1 -- the strand summary. One row per phenotype strand: its reproducibility ceiling,
its best score, the fraction of the ceiling realized, and the epoch budget the best run
actually received. The epoch column is load-bearing rather than decorative, because the
score is a `roll_max` (max of a centered rolling mean), an upward-biased order statistic
whose bias grows with epochs run. A strand scored at 65 epochs and a strand scored at 9,997
are not comparable without it.

Ceilings come from three different scripts because the three measurement types need
different estimators, and mixing them silently would be the worst version of this table:

  expression     sqrt of the mean test-retest correlation over 82 deletions measured in
                 both Kemmeren and Sameith  (expression_ceiling_replicate.py)
  morphology     sqrt of broad-sense reliability from 122 wild-type replicates against the
                 across-mutant variance, averaged over 278 features
                 (morphology_noise_ceiling.py)
  betaxanthin    sqrt of reliability from the per-record standard error over a median of 15
                 colonies  (pigment_noise_ceiling.py)
  beta-carotene  RANK agreement between replicate max and min, because the target is a
                 subjective ordinal and a Pearson ceiling is the wrong object
                 (pigment_noise_ceiling.py)
  amino acid     NONE EXISTS. One replicate per strain and no released SE, so the cell is
                 left empty rather than filled with a plausible number.

TABLE 2 -- the betaxanthin/metabolome paired arms. The 023 Delta grid ran a control head
(`bx_ctrl`, betaxanthin only) against a joint head (`bx_m19`, betaxanthin plus the 19 amino
acids) in matched grid cells. Pairing is by CELL, recovered from the run tags, and the
difference is joint minus control on `val/betaxanthin/pearson_per_feature`.

THE CONFOUND THIS TABLE MUST NOT HIDE. `roll_max` grows with epochs run, so a pair whose
two arms ran different numbers of epochs has a difference that is part effect and part
unequal exposure. The table therefore carries each arm's mean epoch count and flags any
cell whose arms differ by more than `EPOCH_TOLERANCE`. A flagged row is not evidence.

Run from repo root (needs round_leaderboards.csv, from pull_round_leaderboards.py):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/build_retrospective_tables.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

EXPERIMENT = "019-simb-multimodal"
DOC_DIR = osp.join("notes-tex", "019-simb-multimodal", "tables")

# Two arms whose mean epoch counts differ by more than this are not a fair pair, because
# the scoring rule is a running maximum.
EPOCH_TOLERANCE = 50.0

# A grid cell whose CONTROL arm scored below this never learned the primary task at all;
# in the 023 grid every such cell is an lr = 1e-3 cell that collapsed toward a constant
# predictor. A difference between two failed runs is not a measurement of the auxiliary
# head, so those cells are marked and excluded from the headline mean. The threshold is set
# at 0.10 against a live-cell range of 0.36 to 0.42 and a collapsed range of 0.04 to 0.07,
# a gap wide enough that no cell sits near the boundary.
MIN_CONTROL_SCORE = 0.10

# Row order and display names for the strand table. The order is the order the document
# discusses them, not the score order, so the table and the sections agree.
STRAND_ROWS = [
    ("expression", "Expression (Kemmeren, Sameith)"),
    ("expression_masked", "Expression, masked-label objective"),
    ("morphology", "Morphology (CalMorph, Ohya)"),
    ("expression_morphology_joint", "Expression and morphology, joint"),
    ("betaxanthin", "Betaxanthin (Cachera)"),
    ("betaxanthin_amino_acid_joint", "Betaxanthin with metabolome head"),
    ("amino_acid", "Amino-acid pools (Mulleder)"),
    ("beta_carotene", "Beta-carotene (Ozaydin)"),
]


def _ceilings(results_dir: str) -> dict[str, tuple[float | None, str]]:
    """(ceiling, basis) per strand, each read from the script that measured it."""
    with open(osp.join(results_dir, "expression_ceiling_replicate.json")) as fh:
        expression = float(fh_json(fh)["primary_ceiling_mean_sqrt_r"]["ceiling"])
    with open(osp.join(results_dir, "morphology_noise_ceiling.json")) as fh:
        morphology = float(json.load(fh)["ceiling_mean_model_features"])
    with open(osp.join(results_dir, "pigment_noise_ceiling.json")) as fh:
        pigment = json.load(fh)
    betaxanthin = float(pigment["betaxanthin"]["ceiling_pearson"])
    carotene = float(pigment["beta_carotene"]["ceiling_spearman"])
    return {
        "expression": (expression, "test-retest, 82 paired deletions"),
        "expression_masked": (expression, "same target"),
        "morphology": (morphology, "122 WT replicates, 278 features"),
        "expression_morphology_joint": (None, "two ceilings, not one"),
        "betaxanthin": (betaxanthin, "per-record SE, median 15 colonies"),
        "betaxanthin_amino_acid_joint": (betaxanthin, "same target"),
        "amino_acid": (None, "one replicate per strain, none estimable"),
        "beta_carotene": (carotene, "replicate rank agreement, ordinal target"),
    }


def fh_json(fh):
    return json.load(fh)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def strand_table(board: pd.DataFrame, ceilings: dict) -> str:
    n_projects = int(board["project"].nunique())
    n_runs = int(len(board))
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/build_retrospective_tables.py",
        "%% SOURCE: results/round_leaderboards.csv + the three ceiling JSONs. Do not hand-edit.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"strand & ceiling & best & of ceiling & epochs (peak/last) & runs \\",
        r"\midrule",
    ]
    for strand, label in STRAND_ROWS:
        group = board[board["strand"] == strand]
        live = group[~group["is_collapsed"].fillna(False)]
        best_row = live.loc[live["primary_roll_max"].idxmax()]
        best = float(best_row["primary_roll_max"])
        ceiling, _basis = ceilings[strand]
        frac = "--" if ceiling is None else f"{100 * best / ceiling:.0f}\\%"
        epochs = (
            f"{best_row['primary_epoch_at_roll_max']:.0f}/{best_row['epochs']:.0f}"
            if np.isfinite(best_row["epochs"])
            else "--"
        )
        collapsed = int(group["is_collapsed"].fillna(False).sum())
        runs = f"{len(group)}" + (f" ({collapsed} flat)" if collapsed else "")
        lines.append(
            f"{label} & {_fmt(ceiling)} & \\textbf{{{best:.3f}}} & {frac} & {epochs} & {runs} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption[]{Every strand, scored the same way. \emph{best} is \texttt{roll\_max}, "
        r"the maximum of a centered five-point rolling mean of the validation curve, an "
        r"upward-biased order statistic whose bias grows with epochs run, which is why the "
        r"epoch column travels with it. \emph{epochs (peak/last)} gives where the best run "
        r"peaked and where it stopped: a peak in the last tenth means the run was cut off "
        r"while still improving. Ceilings are measured by three different estimators "
        r"because the three measurement types require them; the amino-acid cell is empty "
        r"because Mulleder has one replicate per strain, and no ceiling is estimable. "
        r"\emph{flat} counts runs whose validation metric never left zero. "
        + f"\\textbf{{Coverage: {n_projects} of 28 projects, {n_runs} runs.}} "
        r"The rest are queued (\cref{tab:coverage}); a strand maximum can only move "
        r"upward when they land, so every \emph{of ceiling} figure is a lower bound. "
        r"The betaxanthin row is known to be one such case: \cref{tab:bx-two-numbers} "
        r"carries $0.4301$ from a study in a queued project.}",
        r"\label{tab:strand-summary}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def paired_table(board: pd.DataFrame) -> tuple[str, dict]:
    joint = board[board["strand"] == "betaxanthin_amino_acid_joint"].copy()

    def cell_of(tags: str) -> str | None:
        match = re.search(r"s\d\d_[A-Za-z0-9_.]+", str(tags))
        return match.group(0) if match else None

    def arm_of(tags: str) -> str | None:
        if "bx_m19" in str(tags):
            return "m19"
        if "bx_ctrl" in str(tags):
            return "ctrl"
        return None

    joint["cell"] = joint["tags"].map(cell_of)
    joint["arm"] = joint["tags"].map(arm_of)
    joint = joint[joint["cell"].notna() & joint["arm"].notna()]

    grouped = joint.groupby(["cell", "arm"]).agg(
        n=("run_id", "size"),
        score=("primary_roll_max", "mean"),
        score_sd=("primary_roll_max", "std"),
        epochs=("epochs", "mean"),
        aux=("mulleder19_pearson_per_feature_roll_max", "mean"),
    )
    rows = []
    for cell in sorted(joint["cell"].unique()):
        if ("ctrl") not in grouped.loc[cell].index or ("m19") not in grouped.loc[cell].index:
            continue
        ctrl, m19 = grouped.loc[(cell, "ctrl")], grouped.loc[(cell, "m19")]
        unequal = abs(float(ctrl["epochs"]) - float(m19["epochs"])) > EPOCH_TOLERANCE
        rows.append(
            {
                "cell": cell,
                "n_ctrl": int(ctrl["n"]),
                "n_m19": int(m19["n"]),
                "ctrl": float(ctrl["score"]),
                "m19": float(m19["score"]),
                "delta": float(m19["score"] - ctrl["score"]),
                "epochs_ctrl": float(ctrl["epochs"]),
                "epochs_m19": float(m19["epochs"]),
                "aux": float(m19["aux"]) if np.isfinite(m19["aux"]) else float("nan"),
                "unequal_exposure": bool(unequal),
                "control_collapsed": bool(float(ctrl["score"]) < MIN_CONTROL_SCORE),
            }
        )
    table = pd.DataFrame(rows)

    # The comparison is readable only on cells whose two arms got the same exposure AND
    # whose control arm actually learned the task.
    fair = table[~table["unequal_exposure"]]
    live = fair[~fair["control_collapsed"]]

    def _stats(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
        if not len(frame):
            return {f"{prefix}_n": 0}
        sd = float(frame["delta"].std(ddof=1)) if len(frame) > 1 else None
        return {
            f"{prefix}_n": int(len(frame)),
            f"{prefix}_mean_delta": float(frame["delta"].mean()),
            f"{prefix}_sd_delta": sd,
            f"{prefix}_se_delta": (
                None if sd is None else float(sd / np.sqrt(len(frame)))
            ),
            f"{prefix}_n_positive": int((frame["delta"] > 0).sum()),
        }

    summary = {
        "n_cells": int(len(table)),
        "epoch_tolerance": EPOCH_TOLERANCE,
        "min_control_score": MIN_CONTROL_SCORE,
        **_stats(fair, "fair_exposure"),
        **_stats(live, "live"),
        "per_cell": table.to_dict("records"),
    }

    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/build_retrospective_tables.py",
        "%% SOURCE: results/round_leaderboards.csv, strand betaxanthin_amino_acid_joint.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"grid cell & control & joint & $\Delta$ & aux $r$ & epochs (ctrl/joint) \\",
        r"\midrule",
    ]
    # Live cells first, then the collapsed ones, so the readable comparison is at the top
    # rather than interleaved with cells in which nothing was learned.
    for row in sorted(rows, key=lambda r: (r["control_collapsed"], r["cell"])):
        flags = (r"$^{\dagger}$" if row["unequal_exposure"] else "") + (
            r"$^{\ast}$" if row["control_collapsed"] else ""
        )
        delta = f"{row['delta']:+.4f}"
        bold = r"\textbf{" + delta + "}" if row["delta"] > 0 else delta
        aux = "--" if not np.isfinite(row["aux"]) else f"{row['aux']:.3f}"
        lines.append(
            f"\\texttt{{{row['cell'].replace('_', chr(92) + '_')}}}{flags} & "
            f"{row['ctrl']:.4f} & {row['m19']:.4f} & {bold} & {aux} & "
            f"{row['epochs_ctrl']:.0f}/{row['epochs_m19']:.0f} \\\\"
        )

    def _mean_row(prefix: str, label: str) -> str:
        mean = summary.get(f"{prefix}_mean_delta")
        se = summary.get(f"{prefix}_se_delta")
        if mean is None:
            return f"{label} & & & -- & & \\\\"
        se_txt = "" if se is None else f" $\\pm$ {se:.4f}"
        n_pos = summary.get(f"{prefix}_n_positive")
        n = summary.get(f"{prefix}_n")
        return (
            f"{label} ({n} cells) & & & {mean:+.4f}{se_txt} & & "
            f"{n_pos} of {n} positive \\\\"
        )

    lines += [
        r"\midrule",
        _mean_row("live", "mean, cells where the control learned"),
        _mean_row("fair_exposure", "mean, all matched-exposure cells"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption[]{Betaxanthin score with and without a 19-amino-acid auxiliary head, "
        r"paired within grid cell, on \texttt{val/betaxanthin/pearson\_per\_feature}. "
        r"$\Delta$ is joint minus control. \emph{aux} is the auxiliary head's own score, "
        r"carried because an auxiliary head at $r \approx 0$ that still moved the primary "
        r"metric would mean the gain came from regularization rather than from shared "
        r"metabolic signal, and those are different findings. "
        r"$^{\dagger}$ marks a cell whose two arms ran epoch counts differing by more than "
        + f"{EPOCH_TOLERANCE:.0f}"
        + r"; since the score is a running maximum, such a difference is part effect and "
        r"part unequal exposure, and those rows are excluded from both means. "
        r"$^{\ast}$ marks a cell whose control arm scored below "
        + f"{MIN_CONTROL_SCORE:.2f}"
        + r", meaning neither arm learned the task; every one of them is an "
        r"$\mathrm{lr} = 10^{-3}$ cell, and a difference between two failed runs measures "
        r"nothing about the auxiliary head.}",
        r"\label{tab:bx-aa-paired}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n", summary


def calibration_table(board: pd.DataFrame) -> str:
    r"""Per strand: is the selected model better than predicting each feature's mean.

    `nmse` is normalized so 1.0 is exactly "predict each feature's training mean", and the
    column that matters is its value AT the correlation peak, because that is the model
    anyone would select. A minimum reached at epoch 200 says nothing about the model that
    gets shipped.

    Only rounds from the v8 era onward log `mse`/`nmse`; the metric was added mid-project.
    Earlier rounds are absent rather than zero, and the row count says how many runs the
    strand contributes.
    """
    have = board[board.get("nmse_at_primary_peak").notna()] if "nmse_at_primary_peak" in board else board.iloc[0:0]
    have = have[~have["is_collapsed"].fillna(False).astype(bool)]
    if not len(have):
        return "%% no runs log nmse yet\n"
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/build_retrospective_tables.py",
        "%% SOURCE: results/_leaderboard_cache/*.json, nmse_at_primary_peak.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"strand & runs & \file{nmse} $>1$ & best $r$ & \file{nmse} at peak & $1-r^2$ \\",
        r"\midrule",
    ]
    for strand, group in sorted(have.groupby("strand"), key=lambda kv: kv[0]):
        best = group.loc[group["primary_roll_max"].idxmax()]
        r = float(best["primary_roll_max"])
        nmse = float(best["nmse_at_primary_peak"])
        over = int((group["nmse_at_primary_peak"] > 1.0).sum())
        # The rescale is free: multiplying predictions by r/s changes no correlation and
        # takes nmse to its optimum 1 - r^2.
        lines.append(
            f"{strand.replace('_', ' ')} & {len(group)} & {over} of {len(group)} & "
            f"{r:.4f} & \\textbf{{{nmse:.4f}}} & {1.0 - r * r:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption[]{Whether the selected model beats ``predict each feature's mean'' in "
        r"squared error. \file{nmse} at peak is its value at the epoch the correlation "
        r"peaked, which is the model anyone would ship. Above 1 means the model loses to "
        r"the mean while still ordering features correctly, which is under-shrinkage rather "
        r"than incapacity: the last column is where \file{nmse} lands after multiplying "
        r"predictions by $r/s$, a post-hoc rescale that changes no correlation at all. "
        r"$1-r^2$ is where \file{nmse} would land after multiplying predictions by $r/s$, a post-hoc rescale that changes no correlation at all, and is the optimum for a predictor that is a purely scaled version of a correlated signal. A strand already BELOW it is not a contradiction: it means its predictions carry structure beyond a single global scaling, so the rescale would not help there. Only rounds from the v8 era log these metrics; earlier ones are absent, not zero.}",
        r"\label{tab:calibration}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def epoch_budget_table(board: pd.DataFrame) -> str:
    r"""Expression best per project against how long that project's best run was allowed.

    This exists because the ordering is the finding. Reading nine expression projects the
    same way, the best score is very nearly a function of the epoch budget alone and not of
    what any round was varying. That is the strongest available evidence that the binding
    constraint has been compute rather than mechanism, and it is only visible once every
    round sits in one table.
    """
    expr = board[board["strand"].isin(["expression", "expression_masked"])]
    rows = []
    for project, group in expr.groupby("project"):
        live = group[~group["is_collapsed"].fillna(False).astype(bool)]
        live = live[live["primary_roll_max"].notna()]
        if not len(live):
            continue
        best = live.loc[live["primary_roll_max"].idxmax()]
        rows.append(
            {
                "project": str(project).replace("torchcell_019_", ""),
                "best": float(best["primary_roll_max"]),
                "epochs": float(best["epochs"]) if pd.notna(best["epochs"]) else float("nan"),
                "n_train": (
                    int(best["n_train_supervised"])
                    if pd.notna(best["n_train_supervised"])
                    else None
                ),
                "n_runs": int(len(group)),
            }
        )
    rows.sort(key=lambda r: -r["best"])
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/build_retrospective_tables.py",
        "%% SOURCE: results/round_leaderboards.csv + _leaderboard_cache/, expression strands.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"project & runs & best & epochs of that run & $n_{\mathrm{train}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        epochs = "--" if not np.isfinite(row["epochs"]) else f"{row['epochs']:,.0f}"
        n_train = "--" if row["n_train"] is None else f"{row['n_train']:,}"
        lines.append(
            f"\\file{{{row['project']}}} & {row['n_runs']} & "
            f"\\textbf{{{row['best']:.4f}}} & {epochs} & {n_train} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption[]{Every expression round, read the same way and sorted by score. The "
        r"order is almost exactly the order of the epoch budgets, across rounds that were "
        r"varying capacity, targets, graphs, decoders and objectives. Whatever those rounds "
        r"were testing, the thing that moved the number was how long the run was allowed to "
        r"go. Training-set size moved by only ten percent across the same span and does not "
        r"order the column.}",
        r"\label{tab:expr-epoch-budget}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def coverage_table(board: pd.DataFrame, census_path: str) -> str:
    r"""Which projects have had their per-run history read, and which have not.

    GENERATED rather than hand-listed because the pull is resumable and the split moves
    every time it makes progress. A hand-written list goes stale silently, which is the
    worst failure mode for a table whose whole job is to state a limitation honestly.
    """
    read = sorted(board["project"].unique())
    with open(census_path) as fh:
        census = json.load(fh)
    every = sorted(r["project"] for r in census["per_project"])
    pending = [p for p in every if p not in set(read)]

    def short(names: list[str]) -> str:
        trimmed = [
            n.replace("torchcell_019_", "").replace("torchcell_0", "0") for n in names
        ]
        return ", ".join("\\file{" + n + "}" for n in trimmed)

    status = (
        r"\textbf{All projects are now read.}"
        if not pending
        else r"\textbf{" + str(len(pending)) + r" still pending.}"
    )
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/build_retrospective_tables.py",
        "%% SOURCE: results/round_leaderboards.csv + _leaderboard_cache/ vs project_census.json.",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\begin{tabular}{lrp{76mm}}",
        r"\toprule",
        r"status & projects & which \\",
        r"\midrule",
        f"history read & {len(read)} & {short(read)} \\\\",
        r"\addlinespace",
        f"history pending & {len(pending)} & {short(pending) if pending else '--'} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption[]{Finding a run's best score needs its full validation history, one "
        r"request per run. Against a rate-limited API that does not complete over 2,187 "
        r"runs in one sitting, so the pull caches per project and resumes, and the "
        r"remaining projects are queued rather than dropped. A strand maximum can only "
        r"move UPWARD as they land, so every \emph{of ceiling} figure is a lower bound. "
        + status
        + r"}",
        r"\label{tab:coverage}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def load_board(results_dir: str) -> pd.DataFrame:
    """CSV plus per-project cache, deduped by run id.

    `pull_round_leaderboards.py` writes the CSV only when it finishes every project, and
    against a rate-limited API it often does not. It caches each project as it completes,
    so the cache holds projects the CSV does not. Reading both is what lets a partial pull
    still widen the table instead of being wasted; `run_id` dedupes the overlap, keeping
    the CSV copy because it was written by a full-history pass.
    """
    frames = []
    csv_path = osp.join(results_dir, "round_leaderboards.csv")
    if osp.exists(csv_path):
        frames.append(pd.read_csv(csv_path))
    cache_dir = osp.join(results_dir, "_leaderboard_cache")
    if osp.isdir(cache_dir):
        for name in sorted(os.listdir(cache_dir)):
            if name.endswith(".json"):
                with open(osp.join(cache_dir, name)) as fh:
                    blob = json.load(fh)
                # Cache files gained a {schema, rows} envelope; older ones are a bare list.
                rows = blob["rows"] if isinstance(blob, dict) else blob
                if rows:
                    frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError(f"no leaderboard CSV or cache under {results_dir}")
    board = pd.concat(frames, ignore_index=True)
    # MERGE duplicate run ids per COLUMN rather than dropping whole rows. The CSV was
    # written by an earlier pass at a finer history resolution but before the mse/nmse
    # columns existed, so a plain `drop_duplicates(keep="first")` keeps the CSV row and
    # silently discards the error metrics for every run that appears in both. `first()`
    # after a groupby takes the first NON-NULL value in each column, which keeps the CSV's
    # finer roll_max and picks up the cache's newer columns.
    board = board.groupby("run_id", as_index=False, sort=False).first()
    # A summary-only row has no history and therefore no roll_max; it counts toward run
    # totals but must never be a candidate for a maximum.
    if "primary_roll_max" not in board:
        board["primary_roll_max"] = float("nan")
    if "is_collapsed" not in board:
        board["is_collapsed"] = False
    print(
        f"leaderboard: {board['project'].nunique()} projects, {len(board)} runs, "
        f"{int(board['primary_roll_max'].notna().sum())} with history"
    )
    return board


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    root = osp.dirname(experiment_root.rstrip("/"))
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    board = load_board(results_dir)

    tables_dir = osp.join(root, DOC_DIR)
    os.makedirs(tables_dir, exist_ok=True)

    with open(osp.join(tables_dir, "t-strand-summary.tex"), "w") as fh:
        fh.write(strand_table(board, _ceilings(results_dir)))
    with open(osp.join(tables_dir, "t-calibration.tex"), "w") as fh:
        fh.write(calibration_table(board))
    with open(osp.join(tables_dir, "t-expr-epoch-budget.tex"), "w") as fh:
        fh.write(epoch_budget_table(board))
    census_path = osp.join(results_dir, "project_census.json")
    if osp.exists(census_path):
        with open(osp.join(tables_dir, "t-coverage.tex"), "w") as fh:
            fh.write(coverage_table(board, census_path))
    paired_tex, paired_summary = paired_table(board)
    with open(osp.join(tables_dir, "t-bx-aa-paired.tex"), "w") as fh:
        fh.write(paired_tex)
    with open(osp.join(results_dir, "bx_aa_paired_summary.json"), "w") as fh:
        json.dump(paired_summary, fh, indent=2)

    print(json.dumps({k: v for k, v in paired_summary.items() if k != "per_cell"}, indent=2))
    print(f"-> {tables_dir}")


if __name__ == "__main__":
    main()

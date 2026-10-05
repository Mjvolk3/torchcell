# experiments/019-simb-multimodal/scripts/joint_checkpoint_readout.py
# [[experiments.019-simb-multimodal.scripts.joint_checkpoint_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/joint_checkpoint_readout
"""The checkpoint readout of the joint (v19) and conditioned (v20) rounds, with the
single-deletion cross-modal triangle beside them.

v19 is scored exactly as its config registers (conf/cgt_expr_v19_joint_clean.yaml): the
proteome head by the mean of val/proteome/pearson_per_feature over epochs 200 to 400, the
expression head by the mean of val/expression/pearson_per_feature over epochs 1,000 to
1,200, paired over split seeds, one-sided alpha 0.05 each (expression superiority,
proteome non-inferiority at margin 0.015). Only partitions whose three arms all reached
epoch 1,199 enter. Beside the registered score: the rolling maximum (an upward-biased
order statistic, labeled as such), the epoch of the validation-loss minimum, and the
score at that epoch.

v20 is in flight when this runs, so it is reported as PARTIAL: per run, the epoch reached
and the mean of the predicted head's score over its latest 20 epochs, against the v19
control of the same partition over the same epochs.

The triangle is read from the three covariation result files, not recomputed.

Writes results/joint_checkpoint_readout.json and four tables under
notes-tex/019-simb-multimodal-expression/tables/, and results/joint_checkpoint_curves.csv:
the per-epoch validation curves of every v19 run (run id, arm, split, epoch, val/loss, its
centered 5-epoch rolling mean, the one whose minimum is loss_min_epoch, and the
per-feature Pearson of each head the arm trains), so a figure script can draw training
curves without querying W&B. Each v19 run also records total_param_count from its W&B
summary.

    python experiments/019-simb-multimodal/scripts/joint_checkpoint_readout.py
"""

from __future__ import annotations

import itertools
import json
import os.path as osp
import re
from typing import Any

import numpy as np
import pandas as pd
import wandb
from scipy import stats

from torchcell.utils.paths import experiment_results_dir

ENTITY = "zhao-group"
V19 = "torchcell_019_prot_v19"
V20 = "torchcell_019_prot_v20"
WINDOWS = {"proteome": (200, 400), "expression": (1000, 1200)}
MARGIN = 0.015
FINAL_EPOCH = 1199
ROLL = 5
# The four runs of canary 2423179 (fast_dev_run, one batch, no logged history). The plain
# sync pass of 2026-10-04 uploaded them into the round's project; they are not round runs.
V20_CANARY_RUNS = {"qlfimncm", "or3sbju0", "7gmc5o2h", "lrnt88cv"}
RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
TABLES = osp.join(REPO, "notes-tex", "019-simb-multimodal-expression", "tables")
SCRIPT = "experiments/019-simb-multimodal/scripts/joint_checkpoint_readout.py"


def _curve(run: Any, key: str) -> pd.Series:
    rows = list(run.scan_history(keys=["epoch", key]))
    return pd.DataFrame(rows).groupby("epoch")[key].last()


def _key(head: str) -> str:
    return f"val/{head}/pearson_per_feature"


def _v19_runs(api: wandb.Api) -> tuple[pd.DataFrame, pd.DataFrame]:
    heads = {
        "K_prot": ["proteome"],
        "K_expr": ["expression"],
        "K_joint": ["proteome", "expression"],
    }
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for run in api.runs(f"{ENTITY}/{V19}"):
        tag = [t for t in run.tags if re.fullmatch(r"K_(prot|expr|joint)_s\d+", t)]
        if len(tag) != 1:
            raise ValueError(f"{run.id}: expected one K_* arm tag, got {tag}")
        arm, split = re.fullmatch(r"(K_\w+)_s(\d+)", tag[0]).groups()  # type: ignore[union-attr]
        raw_loss = _curve(run, "val/loss")
        loss = raw_loss.rolling(ROLL, center=True).mean()
        rec: dict[str, Any] = {
            "id": run.id,
            "arm": arm,
            "split": int(split),
            "total_param_count": int(run.summary["total_param_count"]),
            "loss_min_epoch": int(loss.idxmin()),
        }
        curve = pd.DataFrame({"val/loss": raw_loss, f"val/loss_roll{ROLL}": loss})
        for head in heads[arm]:
            c = _curve(run, _key(head))
            curve[_key(head)] = c
            lo, hi = WINDOWS[head]
            smooth = c.rolling(ROLL, center=True).mean()
            rec["last_epoch"] = int(c.index.max())
            rec[f"{head}_window"] = float(c.loc[lo:hi].mean())
            rec[f"{head}_roll_max"] = float(smooth.max())
            rec[f"{head}_roll_max_epoch"] = int(smooth.idxmax())
            rec[f"{head}_at_loss_min"] = float(smooth.loc[rec["loss_min_epoch"]])
        rows.append(rec)
        curve = curve.rename_axis("epoch").reset_index()
        curve.insert(0, "split", int(split))
        curve.insert(0, "arm", arm)
        curve.insert(0, "id", run.id)
        curves.append(curve)
    runs = pd.DataFrame(rows).sort_values(["split", "arm"]).reset_index(drop=True)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves = all_curves.sort_values(["split", "arm", "epoch"]).reset_index(drop=True)
    return runs, all_curves


def _paired(
    d: pd.DataFrame, alt: str, ref: str, col: str, margin: float
) -> dict[str, Any]:
    a = d[d.arm == alt].set_index("split")[col]
    b = d[d.arm == ref].set_index("split")[col]
    diff = (a - b).dropna()
    x = diff.to_numpy() + margin
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    flipped = (signs * x).mean(axis=1)
    return {
        "contrast": f"{alt} - {ref}",
        "score": col,
        "margin": margin,
        "n_partitions": n,
        "ref_mean": float(b.loc[diff.index].mean()),
        "alt_mean": float(a.loc[diff.index].mean()),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "se_diff": float(diff.std(ddof=1) / np.sqrt(n)),
        "n_positive": int((diff > 0).sum()),
        "p_one_sided_t": float(1.0 - stats.t.cdf(t, n - 1)),
        "p_sign_flip": float((flipped >= x.mean() - 1e-12).mean()),
        "per_partition": {int(k): float(v) for k, v in diff.items()},
    }


def _v20_runs(api: wandb.Api, v19: pd.DataFrame) -> pd.DataFrame:
    controls = {(r.arm, r.split): r.id for r in v19.itertuples()}
    rows: list[dict[str, Any]] = []
    for run in api.runs(f"{ENTITY}/{V20}"):
        if run.id in V20_CANARY_RUNS:
            continue
        tag = [t for t in run.tags if t.startswith("C_")]
        if len(tag) != 1:
            raise ValueError(f"{run.id}: expected one C_* arm tag, got {tag}")
        kind, split = re.fullmatch(r"C_(\w+?)_s(\d+)", tag[0]).groups()  # type: ignore[union-attr]
        head = "expression" if kind.startswith("expr") else "proteome"
        ref_arm = "K_expr" if head == "expression" else "K_prot"
        c = _curve(run, _key(head))
        last = int(c.index.max())
        lo = max(0, last - 19)
        ctrl = _curve(
            api.run(f"{ENTITY}/{V19}/{controls[(ref_arm, int(split))]}"), _key(head)
        )
        wlo, whi = WINDOWS[head]
        rows.append(
            {
                "id": run.id,
                "arm": f"C_{kind}",
                "split": int(split),
                "head": head,
                "last_epoch": last,
                "conditioned_latest20": float(c.loc[lo:last].mean()),
                "control_same_epochs": float(ctrl.loc[lo:last].mean()),
                "control_registered_window": float(ctrl.loc[wlo:whi].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["arm", "split"]).reset_index(drop=True)
    out["diff_same_epochs"] = out.conditioned_latest20 - out.control_same_epochs
    return out


def _triangle() -> list[dict[str, Any]]:
    with open(osp.join(RESULTS, "proteome_morphology_covariation.json")) as fh:
        pm = json.load(fh)
    with open(osp.join(RESULTS, "expression_morphology_covariation.json")) as fh:
        em = json.load(fh)
    ctx = pm["context"]
    return [
        {
            "given": "proteome",
            "predicted": "expression",
            "all": ctx["proteome_to_expression_ridge_per_feature"],
            "moving": None,
            "n_strains": 1349,
            "source": ctx["source"],
        },
        {
            "given": "expression",
            "predicted": "proteome",
            "all": ctx["expression_to_proteome_ridge_per_feature"],
            "moving": None,
            "n_strains": 1349,
            "source": ctx["source"],
        },
        {
            "given": "proteome",
            "predicted": "morphology",
            "all": pm["proteome_to_morphology"]["all_features"]["median"],
            "moving": pm["proteome_to_morphology"]["moving_features"]["median"],
            "n_strains": pm["n_shared_strains"],
            "source": "results/proteome_morphology_covariation.json",
        },
        {
            "given": "morphology",
            "predicted": "proteome",
            "all": pm["morphology_to_proteome"]["all_proteins"]["median"],
            "moving": None,
            "n_strains": pm["n_shared_strains"],
            "source": "results/proteome_morphology_covariation.json",
        },
        {
            "given": "expression",
            "predicted": "morphology",
            "all": em["expression_to_morphology"]["all_features"]["median"],
            "moving": em["expression_to_morphology"]["moving_features"]["median"],
            "n_strains": em["n_shared_strains"],
            "source": "results/expression_morphology_covariation.json",
        },
        {
            "given": "morphology",
            "predicted": "expression",
            "all": em["morphology_to_expression"]["all_genes"]["median"],
            "moving": None,
            "n_strains": em["n_shared_strains"],
            "source": "results/expression_morphology_covariation.json",
        },
    ]


def _write(
    name: str,
    source: str,
    caption: str,
    label: str,
    spec: str,
    head: str,
    body: list[str],
) -> None:
    lines = [
        f"%% GENERATED by {SCRIPT}. Do not edit by hand.",
        f"%% SOURCE: {source}",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        f"  \\caption{caption[:-1]} \\src{{{SCRIPT}}}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{spec}}}",
        "    \\toprule",
        f"    {head} \\\\",
        "    \\midrule",
        *[f"    {row} \\\\" for row in body],
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    with open(osp.join(TABLES, name), "w") as fh:
        fh.write("\n".join(lines))


def _f(x: float | None, nd: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"${x:+.{nd}f}$" if sign else f"{x:.{nd}f}"


def main() -> None:
    api = wandb.Api(timeout=180)
    v19, curves = _v19_runs(api)
    done = v19.groupby("split").last_epoch.min() >= FINAL_EPOCH
    n_arms = v19.groupby("split").arm.nunique()
    complete = sorted(int(s) for s in done.index if done[s] and n_arms[s] == 3)
    d = v19[v19.split.isin(complete)]
    tests = {
        "H1a_expression_superiority": _paired(
            d, "K_joint", "K_expr", "expression_window", 0.0
        ),
        "H1b_proteome_noninferiority": _paired(
            d, "K_joint", "K_prot", "proteome_window", MARGIN
        ),
        "descriptive_proteome_vs_zero": _paired(
            d, "K_joint", "K_prot", "proteome_window", 0.0
        ),
        "descriptive_proteome_roll_max": _paired(
            d, "K_joint", "K_prot", "proteome_roll_max", 0.0
        ),
        "descriptive_expression_roll_max": _paired(
            d, "K_joint", "K_expr", "expression_roll_max", 0.0
        ),
        "descriptive_expression_at_loss_min": _paired(
            d, "K_joint", "K_expr", "expression_at_loss_min", 0.0
        ),
        "descriptive_proteome_at_loss_min": _paired(
            d, "K_joint", "K_prot", "proteome_at_loss_min", 0.0
        ),
    }
    v20 = _v20_runs(api, v19)
    triangle = _triangle()
    out = {
        "generated_by": SCRIPT,
        "v19": {
            "project": f"{ENTITY}/{V19}",
            "windows": WINDOWS,
            "margin": MARGIN,
            "complete_partitions": complete,
            "incomplete_partitions": sorted(set(v19.split) - set(complete)),
            "tests": tests,
            "runs": v19.to_dict(orient="records"),
        },
        "v20_partial": {
            "project": f"{ENTITY}/{V20}",
            "runs": v20.to_dict(orient="records"),
        },
        "triangle": triangle,
    }
    with open(osp.join(RESULTS, "joint_checkpoint_readout.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    curves.to_csv(osp.join(RESULTS, "joint_checkpoint_curves.csv"), index=False)

    n = len(complete)
    rows = []
    for label, key in [
        (
            "H1a expression superiority, epochs 1,000 to 1,200",
            "H1a_expression_superiority",
        ),
        (
            "H1b proteome non-inferiority at 0.015, epochs 200 to 400",
            "H1b_proteome_noninferiority",
        ),
        ("proteome, rolling maximum (descriptive)", "descriptive_proteome_roll_max"),
        (
            "expression, rolling maximum (descriptive)",
            "descriptive_expression_roll_max",
        ),
        (
            "proteome at the validation-loss minimum (descriptive)",
            "descriptive_proteome_at_loss_min",
        ),
        (
            "expression at the validation-loss minimum (descriptive)",
            "descriptive_expression_at_loss_min",
        ),
    ]:
        t = tests[key]
        rows.append(
            f"{label} & {_f(t['ref_mean'])} & {_f(t['alt_mean'])} & {_f(t['mean_diff'], sign=True)}"
            f" & {_f(t['se_diff'])} & {t['n_positive']}/{t['n_partitions']} & {_f(t['p_one_sided_t'], 2)}"
            f" & {_f(t['p_sign_flip'], 2)}"
        )
    _write(
        "v19_registered_tests.tex",
        f"W&B {ENTITY}/{V19} run histories",
        (
            f"[The v19 registered tests]{{The deconfounded joint round on the {n} partitions whose three "
            "arms all finished 1,200 epochs, one initialization seed each, paired over split seeds. "
            "\\emph{single} is the single-head arm (\\texttt{K\\_expr} or \\texttt{K\\_prot}) and "
            "\\emph{joint} is \\texttt{K\\_joint}; scores are validation \\file{pearson_per_feature}. "
            "The first two rows are the pre-registered intersection-union test, one-sided at "
            "0.05 each: neither rejects. The $p$ of H1b tests the difference plus the margin. The "
            "remaining rows are descriptive: the rolling maximum is the maximum of a centered "
            "five-epoch mean, an upward-biased order statistic, and the last two rows read each run "
            "at the epoch where its own smoothed validation loss is lowest.}"
        ),
        "tab:v19-registered",
        "@{}lrrrrrrr@{}",
        "score & single & joint & joint $-$ single & SE & positive & $p$ ($t$) & $p$ (sign flip)",
        rows,
    )

    piv = d.pivot(index="split", columns="arm")
    rows = []
    for s in complete:
        e_s, e_j = (
            piv["expression_window"]["K_expr"][s],
            piv["expression_window"]["K_joint"][s],
        )
        p_s, p_j = (
            piv["proteome_window"]["K_prot"][s],
            piv["proteome_window"]["K_joint"][s],
        )
        rows.append(
            f"{s} & {_f(e_s)} & {_f(e_j)} & {_f(e_j - e_s, sign=True)} & {_f(p_s)} & {_f(p_j)}"
            f" & {_f(p_j - p_s, sign=True)} & {int(piv['proteome_roll_max_epoch']['K_prot'][s])}"
            f" & {int(piv['proteome_roll_max_epoch']['K_joint'][s])}"
            f" & {int(piv['loss_min_epoch']['K_joint'][s])}"
        )
    _write(
        "v19_partitions.tex",
        f"W&B {ENTITY}/{V19} run histories",
        (
            "[v19 by partition]{The registered window scores of the joint round by split seed. "
            "Expression is the mean over epochs 1,000 to 1,200 and proteome over 200 to 400. The "
            "last three columns give the epoch of the proteome head's rolling maximum in the "
            "single-head and the joint arm, and the epoch of the joint arm's smoothed "
            "validation-loss minimum: the joint proteome head peaks late, and the loss turns "
            "inside the first 110 epochs of a 1,200-epoch run.}"
        ),
        "tab:v19-partitions",
        "@{}rrrrrrrrrr@{}",
        (
            "split & \\texttt{K\\_expr} & joint & diff & \\texttt{K\\_prot} & joint & diff"
            " & peak, single & peak, joint & loss min"
        ),
        rows,
    )

    rows = [
        f"\\texttt{{{r.arm.replace('_', chr(92) + '_')}}} & {r.split} & {r.head} & {r.last_epoch}"
        f" & {_f(r.conditioned_latest20)} & {_f(r.control_same_epochs)}"
        f" & {_f(r.diff_same_epochs, sign=True)} & {_f(r.control_registered_window)}"
        for r in v20.itertuples()
    ]
    _write(
        "v20_partial.tex",
        f"W&B {ENTITY}/{V20} and {ENTITY}/{V19} run histories",
        (
            "[v20 while in flight]{PARTIAL: the conditioned round at the epoch each run had reached "
            "when this table was generated, of a 1,200-epoch budget. \\emph{conditioned} is the "
            "mean of the predicted head's validation \\file{pearson_per_feature} over the run's "
            "latest 20 epochs; \\emph{control} is the v19 single-head arm of the same partition "
            "over the same epochs; the last column is that control at its registered window. "
            "\\texttt{C\\_expr} predicts expression with the strain's proteome revealed and "
            "\\texttt{C\\_prot} the reverse; the \\texttt{perm} arms reveal another strain's "
            "measurement. Nothing here is a result until the runs finish.}"
        ),
        "tab:v20-partial",
        "@{}lrlrrrrr@{}",
        "arm & split & predicted & epoch & conditioned & control & diff & control, window",
        rows,
    )

    rows = [
        f"{t['given']} & {t['predicted']} & {_f(t['all'])} & {_f(t['moving'])} & {t['n_strains']:,}".replace(
            ",", "{,}"
        )
        for t in triangle
    ]
    _write(
        "triangle.tex",
        "results/proteome_morphology_covariation.json, results/expression_morphology_covariation.json, "
        "review/2026-09-27-joint-review/01_data_ceilings.md",
        (
            "[The single-deletion triangle]{Out-of-fold ridge from one measured modality to another "
            "on the deletions both panels measured: median held-out Pearson per feature of the "
            "predicted modality, five folds over strains. \\emph{moving} restricts morphology to the "
            "116 CalMorph features whose replicate reliability is at least 0.5. The proteome and "
            "expression rows are the mean per feature from the review of 2026-09-27. The three sides "
            "sit on different strain sets and none has been recomputed on a common one. Every "
            "strain-permuted null is at most 0.01.}"
        ),
        "tab:triangle",
        "@{}llrrr@{}",
        "given & predicted & all features & moving & shared strains",
        rows,
    )
    print(
        json.dumps(
            {
                "complete": complete,
                "tests": {
                    k: {kk: vv for kk, vv in v.items() if kk != "per_partition"}
                    for k, v in tests.items()
                },
            },
            indent=1,
        )
    )
    print(v20.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

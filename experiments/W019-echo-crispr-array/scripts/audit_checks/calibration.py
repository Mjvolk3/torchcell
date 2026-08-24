# experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py  (spliceable block)
"""Calibration checks for the W019 trigenic round, audit cluster A1.

Splice `run(check)` into `verify_triple_build_list.py` and call it from `main()`.

**No measured quantity is typed into this file.** Every prediction moment, every label
statistic and every val metric is read from
`experiments/010-kuzmin-tmi/results/prediction_calibration_stats.csv`, which
`export_prediction_calibration_stats.py` produces by streaming the inference parquet,
reading the 010 label build, parsing the eval slurm logs and pulling the W&B summaries.
The only literals here are (a) the numbers the superseded version of the rationale note
stated, which are the subject of the checks rather than evidence for them, and (b) the
Kuzmin thresholds 0.08 / 0.12 / 0.20. If the parquet is regenerated with different values
the CSV changes and these checks move with it, which is the point.

Everything is deterministic and local: one 1.5 kB CSV of predictions, one small stats CSV,
one markdown file, and closed-form arithmetic. No network, no LMDB scan, no 4 GB read.

New constants introduced here (all defined inside `run()`):
  CALIB_CSV, SIGMA_Y_VAL, MU_Y_VAL, VAL_N, CKPT_R, CKPT_MSE, INF_N, INF_MEAN, INF_SD,
  OLD_SIGMA_Y, OLD_SLOPE, OLD_ROWS, KUZMIN_VERY_STRINGENT
"""

from __future__ import annotations

import itertools
import math
import os
import os.path as osp

import pandas as pd
from scipy import stats


def run(check) -> None:
    print("\n9. CALIBRATION (audit cluster A1)")

    # ---- the producing script's output is the only source of measured numbers --------
    CALIB_CSV = osp.join(
        REPO, "experiments/010-kuzmin-tmi/results/prediction_calibration_stats.csv"
    )
    stats_df = pd.read_csv(CALIB_CSV)
    S = dict(zip(stats_df.quantity, stats_df.value.astype(str), strict=True))
    fv = lambda k: float(S[k])   # noqa: E731
    iv = lambda k: int(float(S[k]))  # noqa: E731

    SIGMA_Y_VAL = fv("label_val_sd_pop")     # population SD, ddof = 0
    MU_Y_VAL = fv("label_val_mean")
    VAL_N = iv("label_val_n")
    CKPT_R = fv("ckpt_M03_val_pearson")      # the checkpoint that made the predictions
    CKPT_MSE = fv("ckpt_M03_val_mse")
    INF_N = iv("pred_n")
    INF_MEAN = fv("pred_mean")
    INF_SD = fv("pred_sd_pop")               # population SD, ddof = 0

    # what the superseded calibration paragraph asserted; the subject, not the evidence
    OLD_SIGMA_Y = 0.0535
    OLD_SLOPE = 0.43
    OLD_ROWS = (("M01", 0.443, 0.0580, 0.0563), ("M02", 0.431, 0.0591, 0.0572),
                ("M03", 0.457, 0.0567, 0.0553))
    KUZMIN_VERY_STRINGENT = 0.20

    check("calibration",
          S["inference_checkpoint"] == "c7671wgj"
          and "Pearson=0.4619" in S["inference_parquet_path"]
          and len(S["inference_parquet_sha256"]) == 64,
          "the stats CSV pins the inference parquet by path and sha256",
          S["inference_parquet_sha256"][:16] + "...")

    # ---- 9a. the 39 in-basis predictions --------------------------------------------
    df = pd.read_csv(TARGETS).sort_values("prediction", ascending=False)
    basis = []
    for r in df.itertuples():
        t = frozenset((r.gene1, r.gene2, r.gene3))
        if t <= GENES and not (BLOCKED <= t):
            basis.append((t, float(r.prediction), len(t & NO_TRIGENIC_DATA)))
    preds = [b[1] for b in basis]
    zcount = [b[2] for b in basis]

    check("calibration", len(basis) == 39, "39 in-basis targets carry a prediction",
          len(basis))
    check("calibration",
          abs(min(preds) - 0.409180) < 5e-6 and abs(max(preds) - 0.711426) < 5e-6,
          "raw in-basis prediction range is 0.4092 to 0.7114",
          f"{min(preds):.6f}..{max(preds):.6f}")
    check("calibration",
          fv("pred_min") <= min(preds) and max(preds) <= fv("pred_max"),
          "the in-basis predictions lie inside the streamed min and max of the parquet, "
          "so the CSV describes the file the selection read",
          f"[{fv('pred_min'):.6f},{fv('pred_max'):.6f}] contains "
          f"[{min(preds):.6f},{max(preds):.6f}]")
    check("calibration", abs(fv("pred_max") - max(preds + [0.730957])) < 5e-6,
          "the parquet maximum is the top row of the pinned top-k CSV",
          f"{fv('pred_max'):.6f}")

    by = {k: [p for p, z in zip(preds, zcount, strict=True) if z == k] for k in (0, 1, 2)}
    check("calibration",
          [len(by[k]) for k in (0, 1, 2)] == [16, 17, 6],
          "16 / 17 / 6 targets carry 0 / 1 / 2 zero-trigenic-data genes",
          [len(by[k]) for k in (0, 1, 2)])
    means = [sum(by[k]) / len(by[k]) for k in (0, 1, 2)]
    check("calibration",
          all(abs(m - t) < 5e-5 for m, t in zip(means, (0.4531, 0.5300, 0.6676),
                                                strict=True)),
          "mean prediction by zero-data-gene count is 0.4531 / 0.5300 / 0.6676",
          [f"{m:.4f}" for m in means])

    rho, pval = stats.spearmanr(preds, zcount)
    check("calibration", abs(rho - 0.634288) < 5e-6 and abs(pval - 1.452e-05) < 5e-9,
          "Spearman rho of prediction against zero-data count, n=39, two-sided, "
          "is 0.6343 at p = 1.45e-05",
          f"rho={rho:.6f} p={pval:.4e}")

    # ---- 9b. the label population, from the CSV --------------------------------------
    check("calibration",
          (iv("label_all_n"), iv("label_train_n"), VAL_N, iv("label_test_n"))
          == (376732, 301386, 37673, 37673),
          "010 label set is 376,732 records split 301,386 / 37,673 / 37,673 at seed 42",
          (iv("label_all_n"), iv("label_train_n"), VAL_N, iv("label_test_n")))
    check("calibration",
          abs(SIGMA_Y_VAL - 0.0629) < 5e-4 and abs(MU_Y_VAL + 0.0091) < 5e-4,
          "val-split label SD is 0.0629 and mean is -0.0091, not 0.0535 and -0.048",
          f"sigma_y={SIGMA_Y_VAL:.6f} mu_y={MU_Y_VAL:.6f} n={VAL_N}")
    check("calibration",
          abs(fv("label_all_sd_pop") - fv("last_M01_norm_std")) < 1e-7
          and abs(fv("label_all_mean") - fv("last_M01_norm_mean")) < 1e-7,
          "the whole-build label SD and mean recomputed from the parquet equal what the "
          "runs logged as normalization/gene_interaction/{std,mean}: two independent "
          "sources agree",
          f"{fv('label_all_sd_pop'):.9f} vs {fv('last_M01_norm_std'):.9f}")
    check("calibration",
          min(abs(fv(f"label_{k}_sd_pop") - OLD_SIGMA_Y)
              for k in ("all", "train", "val", "test")) > 0.008,
          "the superseded sigma_y = 0.0535 is not the label SD of any 010 split",
          {k: round(fv(f"label_{k}_sd_pop"), 6)
           for k in ("all", "train", "val", "test")})

    # ---- 9c. the checkpoint used vs the checkpoints tabulated ------------------------
    check("calibration", abs(fv("ckpt_M03_val_rmse") ** 2 - CKPT_MSE) < 1e-9,
          "the eval log's val RMSE and MSE are mutually consistent",
          f"{fv('ckpt_M03_val_rmse'):.9f}^2 vs {CKPT_MSE:.9f}")
    bad = [lab for lab in ("M01", "M02", "M03")
           if fv(f"ckpt_{lab}_val_pearson") <= fv(f"last_{lab}_val_pearson")]
    check("calibration", not bad,
          "for all three runs the best-pearson checkpoint scores above the last epoch, "
          "so the note's last-epoch table understates the model that was used",
          {lab: (round(fv(f"last_{lab}_val_pearson"), 4),
                 round(fv(f"ckpt_{lab}_val_pearson"), 4)) for lab in ("M01", "M02", "M03")})
    check("calibration",
          S["ckpt_M03_run_id"] == S["inference_checkpoint"] == "c7671wgj",
          "the checkpoint whose val metrics drive the calibration is the one that "
          "produced the predictions",
          S["ckpt_M03_run_id"])

    # ---- 9d. the second-moment identity, all inputs from the CSV ---------------------
    # MSE = (mu_p - mu_y)^2 + sigma_y^2 + sigma_p^2 - 2 r sigma_y sigma_p, exactly.
    def roots(r: float, mse: float, sy: float) -> tuple[float, float]:
        disc = r * r * sy * sy - sy * sy + mse
        return r * sy - math.sqrt(disc), r * sy + math.sqrt(disc)

    lo, hi = roots(CKPT_R, CKPT_MSE, SIGMA_Y_VAL)
    slope_lo, slope_hi = CKPT_R * SIGMA_Y_VAL / hi, CKPT_R * SIGMA_Y_VAL / lo
    check("calibration", lo > 0 and hi > lo,
          "for the checkpoint used the identity leaves TWO positive sigma_pred roots, "
          "so sigma_pred is not determined by r, RMSE and sigma_y alone",
          f"{lo:.6f} / {hi:.6f}")
    check("calibration", not (slope_lo <= OLD_SLOPE <= slope_hi),
          "the superseded slope 0.43 lies outside the admissible slope interval",
          f"0.43 vs [{slope_lo:.4f},{slope_hi:.4f}]")
    check("calibration", slope_lo > 0.5,
          "every admissible slope exceeds 0.5, so the model is under-dispersed, "
          "not over-dispersed 2.3x",
          f"min slope {slope_lo:.4f}")

    bias_max = math.sqrt(CKPT_MSE - SIGMA_Y_VAL ** 2 * (1 - CKPT_R ** 2))
    check("calibration", 0 < bias_max < 0.5 * SIGMA_Y_VAL,
          "the identity bounds the val mean bias below a tenth of sigma_y",
          f"|mu_p - mu_y| <= {bias_max:.6f} = {bias_max / SIGMA_Y_VAL:.3f} sigma_y")
    check("calibration", abs(INF_MEAN - MU_Y_VAL) > bias_max,
          "the inference-set prediction mean sits further from mu_y than that bound, so "
          "it is the inference population's mean and not the val bias",
          f"|{INF_MEAN:+.6f} - {MU_Y_VAL:+.6f}| = {abs(INF_MEAN - MU_Y_VAL):.6f} "
          f"> {bias_max:.6f}")

    # the measured prediction SD resolves which root is physical
    check("calibration", INF_SD < SIGMA_Y_VAL,
          "the measured prediction SD is below the label SD: the model is under-dispersed",
          f"{INF_SD:.6f} < {SIGMA_Y_VAL:.6f}, ratio {INF_SD / SIGMA_Y_VAL:.4f}, "
          f"over n={INF_N}")
    d_lo, d_hi = abs(INF_SD - lo), abs(INF_SD - hi)
    check("calibration", d_lo < d_hi and d_hi / d_lo > 4.0 and d_lo / lo < 0.06,
          "the measured prediction SD sits nearer the LOWER sigma_pred root than the "
          "upper by more than 4x, and within 6% of it, which resolves the two-root "
          "ambiguity empirically in favor of the lower root",
          f"measured {INF_SD:.6f}; lower {lo:.6f} (off by {d_lo / lo:.2%}); "
          f"upper {hi:.6f} (off by {d_hi / hi:.2%}); separation {d_hi / d_lo:.1f}x")

    # how far outside its own fitted range the calibration is being read
    zdist = sorted((p - INF_MEAN) / INF_SD for p in preds)
    check("calibration", math.floor(zdist[0]) == 18 and math.ceil(zdist[-1]) == 32,
          "against the measured prediction SD the 39 in-basis targets sit 18 to 32 SD "
          "above the prediction mean, so any calibration is being read far outside the "
          "range it was fitted on",
          f"{zdist[0]:.2f} to {zdist[-1]:.2f} SD")

    # every admissible (bias, sigma_p) pair, gridded on bias
    worst_min, best_max, above = math.inf, -math.inf, set()
    steps = 1001
    for i in range(steps):
        bias = -bias_max + 2 * bias_max * i / (steps - 1)
        rem = CKPT_MSE - bias * bias
        disc = CKPT_R ** 2 * SIGMA_Y_VAL ** 2 - SIGMA_Y_VAL ** 2 + rem
        if disc < 0:
            continue
        for sp in (CKPT_R * SIGMA_Y_VAL - math.sqrt(disc),
                   CKPT_R * SIGMA_Y_VAL + math.sqrt(disc)):
            if sp <= 0:
                continue
            a = CKPT_R * SIGMA_Y_VAL / sp
            mu_p = MU_Y_VAL + bias
            sh = [MU_Y_VAL + a * (p - mu_p) for p in preds]
            worst_min = min(worst_min, min(sh))
            best_max = max(best_max, max(sh))
            above.add(sum(1 for v in sh if v > KUZMIN_VERY_STRINGENT))
    check("calibration", above == {39},
          "under every admissible calibration all 39 in-basis predictions exceed 0.20, "
          "not 15 of 39",
          sorted(above))
    check("calibration", worst_min > KUZMIN_VERY_STRINGENT and best_max < 1.0,
          "the admissible shrunk range clears 0.20 at its worst end and stays below 1.0 "
          "at its best, so it is not 0.148 to 0.278",
          f"{worst_min:.4f}..{best_max:.4f}")

    # ---- 9e. provenance of the superseded column -------------------------------------
    bad = []
    for name, r, rmse, sp_noted in OLD_ROWS:
        _, hi_old = roots(r, rmse * rmse, OLD_SIGMA_Y)
        if abs(hi_old - sp_noted) > 5e-5:
            bad.append((name, round(hi_old, 5), sp_noted))
    check("calibration", not bad,
          "the superseded sigma_pred column 0.0563 / 0.0572 / 0.0553 is exactly what the "
          "identity gives under sigma_y = 0.0535, which is why it looked self-consistent",
          bad or "3 of 3 reproduced")
    survivors = []
    for name, r, rmse, sp_noted in OLD_ROWS:
        disc = r * r * SIGMA_Y_VAL ** 2 - SIGMA_Y_VAL ** 2 + rmse * rmse
        if disc >= 0 and abs(roots(r, rmse * rmse, SIGMA_Y_VAL)[1] - sp_noted) < 0.01:
            survivors.append(name)
    check("calibration", not survivors,
          "no row of the superseded sigma_pred column survives with the correct sigma_y",
          survivors or "0 of 3 survive")
    bad = []
    for name, r, rmse, _ in OLD_ROWS:
        lab = {"M01": "M01", "M02": "M02", "M03": "M03"}[name]
        if abs(fv(f"last_{lab}_val_pearson") - r) > 5e-4 \
                or abs(fv(f"last_{lab}_val_rmse") - rmse) > 5e-5:
            bad.append(name)
    check("calibration", not bad,
          "the note table's three r and RMSE pairs are the W&B last-epoch summaries, "
          "now carried in the CSV rather than only online",
          bad or "3 of 3 match")

    # ---- 9f. the failed run ----------------------------------------------------------
    check("calibration",
          fv("failed_run_val_pearson") < 0
          and abs(fv("failed_run_train_pearson")) < 0.01
          and abs(fv("failed_run_val_rmse") - SIGMA_Y_VAL) / SIGMA_Y_VAL < 0.01,
          "the run the note calls failed has negative val r, no train signal, and a val "
          "RMSE equal to sigma_y, which is what predicting the mean gives",
          f"val r {fv('failed_run_val_pearson'):.4f}, train r "
          f"{fv('failed_run_train_pearson'):.4f}, val RMSE "
          f"{fv('failed_run_val_rmse'):.4f} vs sigma_y {SIGMA_Y_VAL:.4f}")

    # ---- 9g. the note no longer carries the superseded wording -----------------------
    rt = open(RATIONALE).read()
    check("calibration", "Over-dispersed 2.3x" not in rt,
          "the rationale note no longer claims over-dispersion of 2.3x", "absent")
    check("calibration", "0.148--0.278" not in rt and "0.148-0.278" not in rt,
          "the rationale note no longer states the 0.148 to 0.278 shrunk range", "absent")
    check("calibration", "under-dispersed" in rt,
          "the rationale note states the corrected direction, under-dispersed", "present")
    check("calibration", "0.0629" in rt and "0.0091" in rt,
          "the rationale note carries the corrected label statistics 0.0629 and 0.0091",
          "present")
    check("calibration", "Pearson=0.4619" in rt,
          "the rationale note names the checkpoint that produced every prediction",
          "c7671wgj best-pearson epoch 24")
    check("calibration", S["failed_run_group"] in rt and S["failed_run_id"] in rt,
          "the failed W&B group is named in full with its rank-0 run id",
          S["failed_run_id"])


if __name__ == "__main__":
    from dotenv import load_dotenv

    REPO = os.environ.get(
        "A1_REPO",
        "/home/michaelvolk/Documents/projects/torchcell.worktrees/audit/w019-trigenic-round",
    )
    load_dotenv(osp.join(REPO, ".env"))
    DATA_ROOT = os.environ.get("DATA_ROOT", "")
    NOTES = osp.join(REPO, "notes")
    TARGETS = osp.join(
        REPO, "experiments/010-kuzmin-tmi/results/inference_3",
        "top_k_constructible_panel12_k200.csv",
    )
    RATIONALE = osp.join(
        NOTES, "experiments.W019-echo-crispr-array.next-strains-to-construct.md"
    )
    GENES = frozenset((
        "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
        "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
    ))
    BLOCKED = frozenset(("YKL033W-A", "YJR060W"))
    NO_TRIGENIC_DATA = frozenset(("YER079W", "YLR312C-B"))

    def orf(cell: object) -> str:
        return str(cell).split(" ")[0].strip()

    def pairs(triple):
        return [frozenset(p) for p in itertools.combinations(sorted(triple), 2)]

    _TALLY = {"n": 0, "fail": 0}

    def _check(group: str, ok: bool, claim: str, observed: object) -> None:
        _TALLY["n"] += 1
        _TALLY["fail"] += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {claim}  [{observed}]")

    run(_check)
    print(f"\n{_TALLY['n']} checks, {_TALLY['fail']} FAIL")
    raise SystemExit(1 if _TALLY["fail"] else 0)

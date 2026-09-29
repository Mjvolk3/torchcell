# experiments/032-amino-acid-module/scripts/aa_label_medium_readout.py
# [[experiments.032-amino-acid-module.scripts.aa_label_medium_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/aa_label_medium_readout

r"""Rounds 208 to 212 on the amino acids: the medium controls, the label stores and the reverse co-head, paired.

The four rounds that landed on 2026-09-28 carried two essentiality recipes to the
M\"ulleder profile: the growth ratio's medium (w228 four media as a control, w229 the
screen's own SM under the incumbent's default affine initialization) and the graphs'
label propagation of the target (w230 the label store alone, w231 the label store with
the SM parsimonious vector appended, both TRANSDUCTIVE: their columns are other strains'
measured profiles, so a strain can be scored only with its neighbors' labels in hand).
Each arm is paired by seed against its control on the same split and statistic as
``target_transfer_readout.py`` (the mean per-amino-acid pinned-test Spearman over the 19
columns at the validation-selected epoch), and its six-seed ensemble is scored from the
dumped predictions as ``target_ensemble_readout.py`` scores them (the mean prediction
over seeds, the same mean per-column Spearman). The slurm job and its elapsed time come
from the pack's log name and ``sacct``. Writes ``results/aa_label_medium_readout.json``
and the document's table ``tables/aa_label_medium_rounds.tex``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/aa_label_medium_readout.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import re
import subprocess
import sys

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from scipy.stats import spearmanr

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
from notes_tex_tables import fit_columns  # noqa: E402

STUDY = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module")
RESULTS = osp.join(STUDY, "results")
LOGS = osp.join(STUDY, "slurm", "output")
TABLES = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex", "032-amino-acid-module", "tables")
SCRIPT = "experiments/032-amino-acid-module/scripts/aa_label_medium_readout.py"
SIX = tuple(range(2700, 2706))

#: stem -> (document round, short label). The controls carry no round of their own here.
ARMS: dict[str, tuple[int | None, str]] = {
    "w141_aadel_h12l1p1_e200": (None, "the control (deletion-only graphs, no store)"),
    "w143_aadelstore_h12l1p1_e200": (None, "the SM parsimonious store affine"),
    "w228_aadelstore4m_h12l1p1_e200": (208, "the SM store with four media's growth ratios (control)"),
    "w229_aadelstoredef_h12l1p1_e200": (209, "the SM store under default affine initialization, own medium"),
    "w230_aadelgraphlabel_h12l1p1_e200": (210, "the graphs' label propagation of the profile (transductive)"),
    "w231_aadelgraphlabelpfba_h12l1p1_e200": (211, "the label propagation with the SM store appended (transductive)"),
    "w225_aadelbxscoremlp256_h12l1p1_e200": (None, "the betaxanthin score as an input through 256"),
    "w232_aadelauxbx_h12l1p1_e200": (212, "the betaxanthin co-head at weight 1 on the control (inductive)"),
}
PAIRS: list[tuple[str, str, str]] = [
    ("four media's growth ratios against the SM ratio", "w228_aadelstore4m_h12l1p1_e200", "w143_aadelstore_h12l1p1_e200"),
    ("four media's growth ratios against the control", "w228_aadelstore4m_h12l1p1_e200", "w141_aadel_h12l1p1_e200"),
    ("default affine initialization against zero initialization", "w229_aadelstoredef_h12l1p1_e200", "w143_aadelstore_h12l1p1_e200"),
    ("default affine initialization against the control", "w229_aadelstoredef_h12l1p1_e200", "w141_aadel_h12l1p1_e200"),
    ("default affine initialization against four media", "w229_aadelstoredef_h12l1p1_e200", "w228_aadelstore4m_h12l1p1_e200"),
    ("label propagation alone against the SM store", "w230_aadelgraphlabel_h12l1p1_e200", "w143_aadelstore_h12l1p1_e200"),
    ("label propagation alone against the control", "w230_aadelgraphlabel_h12l1p1_e200", "w141_aadel_h12l1p1_e200"),
    ("label propagation with the SM store against the SM store", "w231_aadelgraphlabelpfba_h12l1p1_e200", "w143_aadelstore_h12l1p1_e200"),
    ("label propagation with the SM store against label propagation alone", "w231_aadelgraphlabelpfba_h12l1p1_e200", "w230_aadelgraphlabel_h12l1p1_e200"),
    ("label propagation with the SM store against the control", "w231_aadelgraphlabelpfba_h12l1p1_e200", "w141_aadel_h12l1p1_e200"),
    ("the betaxanthin co-head against the control", "w232_aadelauxbx_h12l1p1_e200", "w141_aadel_h12l1p1_e200"),
    ("the betaxanthin co-head against the score as an input", "w232_aadelauxbx_h12l1p1_e200", "w225_aadelbxscoremlp256_h12l1p1_e200"),
]


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    select_epoch: int
    select_val_pearson: float
    spearman: float
    n_records_with_any_feature: int


class Arm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stem: str
    round: int | None
    label: str
    cells: list[Cell]
    mean_spearman: float
    sd_spearman: float
    min_spearman: float
    max_spearman: float
    ensemble_six_spearman: float
    slurm_job: int | None
    elapsed: str | None


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arm: str
    control: str
    n: int
    arm_mean: float
    arm_sd: float
    control_mean: float
    control_sd: float
    mean_difference: float
    sd_difference: float
    n_positive: int


def load_cells(stem: str) -> tuple[list[Cell], dict[int, dict]]:
    cells, dumps = [], {}
    for s in SIX:
        path = osp.join(RESULTS, f"{stem}_s{s}.json")
        with open(path) as fh:
            r = json.load(fh)
        t = r["test"]["pinned"]
        cells.append(
            Cell(
                seed=s,
                select_epoch=int(r["select_epoch"]),
                select_val_pearson=float(r["select_val_pearson"]),
                spearman=float(t["mean_spearman"]),
                n_records_with_any_feature=int(t["n_records_with_any_feature"]),
            )
        )
        with open(r["test"]["dump_path"]) as fh:
            dump = json.load(fh)
        assert dump["cell"] == r["cell"], (stem, s)
        dumps[s] = dump
    return cells, dumps


def arrays(dump: dict) -> tuple[list[int], np.ndarray, np.ndarray]:
    recs = [r for r in dump["predictions"] if r["pinned"]]
    keys = [int(r["record_index"]) for r in recs]
    pred = np.array([r["pred"] for r in recs], dtype=float)
    true = np.array(
        [[np.nan if x is None else x for x in r["target"]] for r in recs], dtype=float
    )
    return keys, pred, true


def ensemble_six(dumps: dict[int, dict]) -> float:
    ref, _, true = arrays(dumps[SIX[0]])
    preds = []
    for s in SIX:
        keys, pred, _ = arrays(dumps[s])
        assert keys == ref, "pinned records differ between seeds"
        preds.append(pred)
    mean = np.mean(np.stack(preds), axis=0)
    rhos = []
    for j in range(true.shape[1]):
        finite = np.isfinite(mean[:, j]) & np.isfinite(true[:, j])
        if finite.sum() < 3:
            continue
        rhos.append(float(spearmanr(mean[finite, j], true[finite, j]).correlation))
    return float(np.mean(rhos))


def slurm_of(stem: str) -> tuple[int | None, str | None]:
    logs = sorted(glob.glob(osp.join(LOGS, f"032-pack_*_{stem}_s{SIX[0]}.out")))
    if not logs:
        return None, None
    job = int(re.search(r"032-pack_(\d+)_", osp.basename(logs[-1])).group(1))
    out = subprocess.run(
        ["sacct", "-j", str(job), "-X", "-n", "-o", "Elapsed"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    h, m, s = out.split(":")
    return job, f"{int(h)} h {int(m)} min"


def table(arms: dict[str, Arm], pairs: list[Pair]) -> str:
    rows = [
        f"    {p.name} & {p.n} & {p.arm_mean:.4f} ({p.arm_sd:.4f}) & "
        f"{p.control_mean:.4f} ({p.control_sd:.4f}) & ${p.mean_difference:+.4f}$ "
        f"({p.sd_difference:.4f}) & {p.n_positive}/{p.n} \\\\"
        for p in pairs
    ]
    level_rows = []
    for a in arms.values():
        if a.round is None:
            continue
        level_rows.append(
            f"    {a.round} & {a.label} & {a.min_spearman:.3f} to {a.max_spearman:.3f} & "
            f"{a.mean_spearman:.3f} ({a.sd_spearman:.3f}) & {a.ensemble_six_spearman:.3f} & "
            f"{min(c.select_epoch for c in a.cells)} to {max(c.select_epoch for c in a.cells)} & "
            f"{min(c.select_val_pearson for c in a.cells):.3f} to "
            f"{max(c.select_val_pearson for c in a.cells):.3f} \\\\"
        )
    return (
        "\\begin{table}[H]\n  \\centering\n  \\footnotesize\n"
        "  \\caption[Rounds 208 to 212: the medium controls, the label stores and the reverse co-head, paired]{"
        "The four rounds of 2026-09-28 and the reverse co-head of 2026-09-29 paired by seed against their controls on the same "
        "split (Merzbacher's 639 genes pinned to test, the 027 protocol) and statistic as "
        "Table~\\ref{tab:target-transfer}, the mean per-amino-acid pinned-test Spearman "
        "over the 19 columns at the epoch selected on validation Pearson. The two label "
        "stores are transductive: their columns are other strains' measured profiles; the "
        "co-head (round 212, \\texttt{--aux-betaxanthin} at weight 1) is inductive. "
        "Written by \\texttt{aa\\_label\\_medium\\_readout.py} from the per-seed result "
        "files.}\n"
        "  \\label{tab:aa-label-medium}\n"
        "  \\begin{tabular}{@{}p{58mm}rrrrr@{}}\n    \\hline\n"
        "    arm against control & seeds & arm (sd) & control (sd) & difference (sd) & "
        "higher \\\\\n    \\hline\n" + "\n".join(rows) + "\n    \\hline\n  \\end{tabular}\n"
        "\\end{table}\n\n"
        "\\begin{table}[H]\n  \\centering\n  \\footnotesize\n"
        "  \\caption[Rounds 208 to 212: seed levels and ensembles]{The same five arms by "
        "seed: the lowest and highest single-seed read, the mean with its standard "
        "deviation, the six-seed ensemble (the mean prediction over seeds scored as one "
        "cell, as \\texttt{target\\_ensemble\\_readout.py} scores it), the selected epochs "
        "and the validation Pearson at selection. Written by "
        "\\texttt{aa\\_label\\_medium\\_readout.py}.}\n"
        "  \\label{tab:aa-label-medium-levels}\n"
        "  \\begin{tabular}{@{}rp{52mm}rrrrr@{}}\n    \\hline\n"
        "    round & arm & seeds & mean (sd) & ensemble & epochs & validation \\\\\n"
        "    \\hline\n" + "\n".join(level_rows) + "\n    \\hline\n  \\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    arms: dict[str, Arm] = {}
    for stem, (rnd, label) in ARMS.items():
        cells, dumps = load_cells(stem)
        v = np.array([c.spearman for c in cells])
        job, elapsed = slurm_of(stem) if rnd is not None else (None, None)
        arms[stem] = Arm(
            stem=stem,
            round=rnd,
            label=label,
            cells=cells,
            mean_spearman=float(v.mean()),
            sd_spearman=float(v.std(ddof=1)),
            min_spearman=float(v.min()),
            max_spearman=float(v.max()),
            ensemble_six_spearman=ensemble_six(dumps),
            slurm_job=job,
            elapsed=elapsed,
        )
        print(
            f"{stem:42s} mean {v.mean():.4f} sd {v.std(ddof=1):.4f} "
            f"ensemble {arms[stem].ensemble_six_spearman:.4f} job {job} {elapsed}"
        )
    pairs: list[Pair] = []
    for name, a, c in PAIRS:
        va = np.array([x.spearman for x in arms[a].cells])
        vc = np.array([x.spearman for x in arms[c].cells])
        d = va - vc
        pairs.append(
            Pair(
                name=name,
                arm=a,
                control=c,
                n=len(d),
                arm_mean=float(va.mean()),
                arm_sd=float(va.std(ddof=1)),
                control_mean=float(vc.mean()),
                control_sd=float(vc.std(ddof=1)),
                mean_difference=float(d.mean()),
                sd_difference=float(d.std(ddof=1)),
                n_positive=int((d > 0).sum()),
            )
        )
        print(f"  {name:70s} {d.mean():+.4f} (sd {d.std(ddof=1):.4f}) {int((d > 0).sum())}/{len(d)}")
    out = {
        "generated_by": SCRIPT,
        "statistic": (
            "mean per-amino-acid pinned-test Spearman over the 19 columns at the epoch "
            "selected on validation Pearson (centered five-epoch moving average); pairs "
            "over seeds 2700 to 2705; ensembles score the mean prediction over the six "
            "seeds on the pinned records"
        ),
        "arms": {k: a.model_dump() for k, a in arms.items()},
        "pairs": [p.model_dump() for p in pairs],
    }
    with open(osp.join(RESULTS, "aa_label_medium_readout.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    os.makedirs(TABLES, exist_ok=True)
    path = osp.join(TABLES, "aa_label_medium_rounds.tex")
    with open(path, "w") as fh:
        fh.write(
            "%% notes-tex/032-amino-acid-module/tables/aa_label_medium_rounds.tex\n"
            f"%% GENERATED by {SCRIPT}\n"
            "%% SOURCE: results/aa_label_medium_readout.json\n" + fit_columns(table(arms, pairs))
        )
    print(path)


if __name__ == "__main__":
    main()

# experiments/019-simb-multimodal/scripts/wandb_v13_report.py
# [[experiments.019-simb-multimodal.scripts.wandb_v13_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_v13_report
"""Name a split-design round's runs, tag them for grouping, and build its W&B views.

Written for the v13 split round and parameterized by ``--round`` for the rounds that reuse
its design (v14, the same arms on the Messner proteome). The offline runs arrive on W&B
named after their sync directory (``run_compute-0-1-2397311_<hash>``), which says nothing
about the arm, and the arm is only recoverable from the tags. This script (1) renames
every run in the project to ``<arm>_seed<k>`` and writes four top-level config keys the UI
can group and filter on (``arm``, ``split``, ``readout``, ``partition``), (2) writes a W&B
report with one panel grid per question (headline by arm, per split with both readouts,
every validation metric by arm, the train side, split against partition), and (3)
overwrites the round's saved Charts view with six sections in rank order of importance,
grouped by arm with epoch on the x axis. Rerun after every sync; it is idempotent.

A round whose saved view does not exist yet (``view_id`` None) gets one created on the
first run; the script prints its id, which then goes into ``ROUNDS`` so later runs
overwrite it in place rather than creating another.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/wandb_v13_report.py --round v13
    python experiments/019-simb-multimodal/scripts/wandb_v13_report.py --round v14
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
X = "epoch"


@dataclass(frozen=True)
class Round:
    """What differs between the rounds that share the split design."""

    project: str
    report_title: str
    view_name: str
    # The saved Charts view this script owns. The workspace API refuses the personal
    # default view ("does not currently support user views"), so a saved view is the
    # nearest thing: open the project and pick the view in the dropdown.
    view_id: str | None
    arm_re: str
    phenotype: str
    splits: list[str]
    split_label: dict[str, str]
    intro: str
    max_runs: int
    label_key: str = "expression_log2_ratio"
    partitions: dict[str, str] = field(default_factory=dict)


ROUNDS: dict[str, Round] = {
    "v13": Round(
        project="torchcell_019_expr_v13",
        report_title="v13 split round: H_ref and H_concat on four partitions and a 90/10 fold",
        view_name="v13 split round by arm",
        view_id="ywphnc96tfh",
        arm_re=r"V_(ref|concat)_(s\d+(?:_90)?)",
        phenotype="expression",
        splits=["s0", "s0_90", "s1", "s2", "s3"],
        split_label={
            "s0": "split 0, 80/10/10",
            "s0_90": "split 0, test folded into train (90/10)",
            "s1": "split 1",
            "s2": "split 2",
            "s3": "split 3",
        },
        intro=(
            "24 runs, four per A40 card on the IGB gpu partition, 6,000 epochs "
            "(job 2397311, config cgt_expr_v13_split). H_ref is the v12 reference "
            "(E_full input, pinball, shared MLP readout); H_concat hands the readout "
            "[h_pert ; h_i ; c] instead of h_pert. Each card holds one split's two "
            "readouts for two seeds. Grouped lines are the mean over seeds with the "
            "min-max band. Nothing here is a result until the runs finish."
        ),
        max_runs=24,
    ),
    "v14": Round(
        project="torchcell_019_prot_v14",
        report_title="v14 proteome round: P_ref and P_concat on four partitions of the Messner knockout proteome",
        view_name="v14 proteome round by arm",
        view_id="yxor02gt7y9",
        arm_re=r"P_(ref|concat)_(s\d+)",
        phenotype="proteome",
        splits=["s0", "s1", "s2", "s3"],
        split_label={
            "s0": "split 0",
            "s1": "split 1",
            "s2": "split 2",
            "s3": "split 3",
        },
        intro=(
            "16 runs, four per RTX 6000 Ada card on cabbi, 2,000 epochs (config "
            "cgt_expr_v14_proteome): the v13 split design on the Messner 2023 knockout "
            "proteome, log2(strain / HIS3 reference) over the 1,850-protein union with "
            "NaN where a strain did not quantify a protein (scored on finite entries). "
            "P_ref is the v12 reference readout, P_concat hands the readout "
            "[h_pert ; h_i ; c]. Each card holds one split's two readouts for two seeds. "
            "Grouped lines are the mean over seeds with the min-max band. Nothing here "
            "is a result until the runs finish."
        ),
        max_runs=16,
        label_key="protein_abundance",
    ),
    "v15": Round(
        project="torchcell_019_expr_v15",
        report_title="v15 weight-decay round: 1e-2 and 1e-1 against 1e-8 on split seeds 1 and 2",
        view_name="v15 weight-decay round by arm",
        view_id="pe89umm24eb",
        arm_re=r"W_(ref|wd1e2|wd1e1)_(s\d+)",
        phenotype="expression",
        splits=["s1", "s2"],
        split_label={"s1": "split 1", "s2": "split 2"},
        intro=(
            "12 runs, four per RTX 6000 Ada card on cabbi, 6,000 epochs (config "
            "cgt_expr_v15_wd): the v13 reference under AdamW weight decay 1e-8 (W_ref), "
            "1e-2 (W_wd1e2) and 1e-1 (W_wd1e1) on split seeds 1 and 2, two init seeds. "
            "Grouped lines are the mean over seeds with the min-max band. Nothing here "
            "is a result until the runs finish."
        ),
        max_runs=12,
    ),
    "v16": Round(
        project="torchcell_019_prot_v16",
        report_title="v16 joint round: the proteome head, the expression head and both on one trunk",
        view_name="v16 joint round by arm",
        view_id=None,
        arm_re=r"J_(ref|expr|joint|joint05)_(s\d+)",
        phenotype="proteome",
        splits=["s0", "s1", "s2"],
        split_label={"s0": "split 0", "s1": "split 1", "s2": "split 2"},
        intro=(
            "18 runs, three per RTX 6000 Ada card on cabbi, 500 epochs (job 2409261, "
            "config cgt_expr_v16_joint): J_ref is the proteome head alone on "
            "proteome-carrying genotypes (the v14 reference at this budget), J_expr the "
            "expression head alone on the SAME fig3_proteome partition, and J_joint both "
            "heads on one trunk. Each card holds the three arms of one split and seed, so "
            "every contrast is paired within card. The proteome side is read against "
            "J_ref and the expression side against J_expr; v13's partition is not "
            "reproducible on this store. Grouped lines are the mean over seeds with the "
            "min-max band. Nothing here is a result until the runs finish."
        ),
        max_runs=18,
        label_key="protein_abundance",
    ),
    "v17": Round(
        project="torchcell_019_expr_v17",
        report_title="v17 perturbation-locality round: the self-indicator and 2-hop graph propagation",
        view_name="v17 locality round by arm",
        view_id=None,
        arm_re=r"L_(ref|self|prop2)_(s\d+)",
        phenotype="expression",
        splits=["s0", "s1", "s2", "s3"],
        split_label={
            "s0": "split 0",
            "s1": "split 1",
            "s2": "split 2",
            "s3": "split 3",
        },
        intro=(
            "36 runs, three per A40 card on the IGB gpu partition, 1,200 epochs (job "
            "2409262, config cgt_expr_v17_locality): L_ref is the v13 reference, where "
            "the deletion is one vector added to every gene token; L_self adds the hop-0 "
            "self-indicator (gate forced on) so a token knows it is the deleted gene; "
            "L_prop2 adds 1-hop and 2-hop reachability of the deletion along each of the "
            "nine gene-gene graphs. Each card holds the three arms of one split and seed. "
            "Scored as the mean over epochs 1,000 to 1,200, not a max over epochs. "
            "Grouped lines are the mean over seeds with the min-max band. Nothing here is "
            "a result until the runs finish."
        ),
        max_runs=36,
    ),
}


def val_metrics(pheno: str) -> list[str]:
    return [
        f"val/{pheno}/pearson_per_feature",
        f"val/{pheno}/spearman_per_feature",
        f"val/{pheno}/pearson_per_instance",
        f"val/{pheno}/pred_sd_ratio",
        "val/loss",
        f"val/{pheno}/nmse",
        f"val/{pheno}/mse",
        f"val/{pheno}/pearson_per_feature@k1",
        f"val/{pheno}/pearson_per_feature@k2",
        f"val/{pheno}/pearson_per_feature@k3",
        f"val/{pheno}/calib/coverage_50",
        f"val/{pheno}/calib/coverage_80",
        f"val/{pheno}/calib/pit_ks",
        "val/mean/pearson_per_feature",
    ]


def train_metrics(pheno: str) -> list[str]:
    return [
        f"traineval/{pheno}/pearson_per_feature",
        f"traineval/{pheno}/pred_sd_ratio",
        f"traineval/{pheno}/nmse",
        "train/loss",
        "train/grad_norm",
        "train/grad_norm_clip_frac",
    ]


def chart_sections(pheno: str) -> list[tuple[str, list[str | list[str]]]]:
    """The Charts tab, in rank order of importance.

    Section 1 is what decides the round; every later section is what to read when
    section 1 moves or fails to. A list entry is one panel with several metrics
    overlaid (train against validation).
    """
    return [
        (
            "1 headline: validation",
            [
                f"val/{pheno}/pearson_per_feature",
                f"val/{pheno}/spearman_per_feature",
                f"val/{pheno}/pearson_per_instance",
                f"val/{pheno}/pred_sd_ratio",
                "val/loss",
                "val/mean/pearson_per_feature",
            ],
        ),
        (
            "2 train side, the generalization gap",
            [
                f"traineval/{pheno}/pearson_per_feature",
                f"traineval/{pheno}/pred_sd_ratio",
                f"traineval/{pheno}/spearman_per_feature",
                f"traineval/{pheno}/pearson_per_instance",
                "traineval/loss",
                f"traineval/{pheno}/nmse",
            ],
        ),
        (
            # Train and validation on ONE panel each, for the interpolation watch: an
            # epoch-wise double descent would show as train MSE reaching zero while the
            # validation curve worsens and later recovers. Read 2026-09-16 at epoch
            # ~2,300: train MSE 0.015 and falling, val MSE flat at 0.040, val Pearson
            # still rising; nowhere near the threshold.
            "2b interpolation watch, train against validation",
            [
                [f"traineval/{pheno}/mse", f"val/{pheno}/mse"],
                [
                    f"traineval/{pheno}/pearson_per_feature",
                    f"val/{pheno}/pearson_per_feature",
                ],
                ["train/loss", "traineval/loss", "val/loss"],
                [f"traineval/{pheno}/nmse", f"val/{pheno}/nmse"],
                [f"traineval/{pheno}/pred_sd_ratio", f"val/{pheno}/pred_sd_ratio"],
                [
                    f"traineval/{pheno}/spearman_per_feature",
                    f"val/{pheno}/spearman_per_feature",
                ],
            ],
        ),
        (
            "3 masked conditioning, revealed 0 / 10 / 100 / 1000 genes",
            [
                f"val/{pheno}/pearson_per_feature@k0",
                f"val/{pheno}/pearson_per_feature@k1",
                f"val/{pheno}/pearson_per_feature@k2",
                f"val/{pheno}/pearson_per_feature@k3",
                "val/mask/loss@k0",
                "val/mask/loss@k1",
                "val/mask/loss@k2",
                "val/mask/loss@k3",
            ],
        ),
        (
            "4 error and calibration",
            [
                f"val/{pheno}/nmse",
                f"val/{pheno}/mse",
                f"val/{pheno}/calib/coverage_50",
                f"val/{pheno}/calib/coverage_80",
                f"val/{pheno}/calib/pit_ks",
                f"traineval/{pheno}/mse",
            ],
        ),
        (
            "5 optimization",
            [
                "train/loss",
                "train/grad_norm",
                "train/grad_norm_clip_frac",
                "train/mask/loss@k0",
                "train/mask/loss@k3",
                "perf/epoch_seconds",
            ],
        ),
        (
            "6 bookkeeping",
            [
                f"val/{pheno}/n_scored_genes@k0",
                "val/mask/n_revealed@k1",
                "val/mask/n_revealed@k3",
                "trainer/global_step",
            ],
        ),
    ]


def label_runs(api: wandb.Api, rnd: Round) -> int:
    n = 0
    prefix = rnd.arm_re.split("_")[0]
    for run in api.runs(f"{ENTITY}/{rnd.project}"):
        arms = [t for t in run.tags if t.startswith(f"{prefix}_")]
        if len(arms) != 1:
            raise ValueError(f"{run.id}: expected one {prefix}_* tag, got {arms}")
        arm = arms[0]
        m = re.fullmatch(rnd.arm_re, arm)
        if m is None:
            raise ValueError(f"{run.id}: arm tag {arm} does not parse")
        readout, split = m.group(1), m.group(2)
        seed = int(run.config["seed"])
        name = f"{arm}_seed{seed}"
        changed = run.name != name or run.config.get("arm") != arm
        run.name = name
        run.config["arm"] = arm
        run.config["split"] = split
        run.config["readout"] = readout
        run.config["partition"] = "90/10" if split.endswith("_90") else "80/10/10"
        run.update()
        n += int(changed)
    return n


def line(title: str, y: str, groupby: str | None = "arm") -> wr.LinePlot:
    kw = {}
    if groupby is not None:
        kw = {
            "groupby": wr.Config(groupby),
            "groupby_aggfunc": "mean",
            "groupby_rangefunc": "minmax",
        }
    return wr.LinePlot(
        title=title,
        x=X,
        y=[y],
        title_x="epoch",
        smoothing_type="none",
        legend_position="east",
        layout=wr.Layout(w=12, h=8),
        **kw,
    )


def runset(rnd: Round, name: str, filters: str | None = None) -> wr.Runset:
    return wr.Runset(
        entity=ENTITY, project=rnd.project, name=name, filters=filters or ""
    )


def build_report(rnd: Round) -> wr.Report:
    pf = f"val/{rnd.phenotype}/pearson_per_feature"
    blocks: list = [
        wr.H1(rnd.report_title),
        wr.P(rnd.intro),
        wr.TableOfContents(),
        wr.H2("Headline: validation Pearson by arm"),
        wr.PanelGrid(
            runsets=[runset(rnd, "all runs")],
            panels=[
                line("val pearson_per_feature, mean over seeds by arm", pf, "arm"),
                line("val pearson_per_feature, every run", pf, None),
                line(
                    "val pearson_per_feature by split (both readouts, all seeds)",
                    pf,
                    "split",
                ),
                line("val pearson_per_feature by readout (all splits)", pf, "readout"),
            ],
        ),
        wr.H2(
            "Per split: the reference against the concat readout on the same partition"
        ),
    ]
    for split in rnd.splits:
        blocks.append(wr.H3(rnd.split_label[split]))
        blocks.append(
            wr.PanelGrid(
                runsets=[
                    runset(rnd, rnd.split_label[split], f"Config('split') == '{split}'")
                ],
                panels=[
                    line(f"{split}: val pearson by readout", pf, "readout"),
                    line(f"{split}: val pearson, every run", pf, None),
                    line(
                        f"{split}: pred_sd_ratio",
                        f"val/{rnd.phenotype}/pred_sd_ratio",
                        "readout",
                    ),
                    line(f"{split}: val loss", "val/loss", "readout"),
                ],
            )
        )
    blocks.append(wr.H2("Every validation metric, by arm"))
    blocks.append(
        wr.PanelGrid(
            runsets=[runset(rnd, "all runs")],
            panels=[line(m, m, "arm") for m in val_metrics(rnd.phenotype)],
        )
    )
    blocks.append(wr.H2("Train side, by arm"))
    blocks.append(
        wr.PanelGrid(
            runsets=[runset(rnd, "all runs")],
            panels=[line(m, m, "arm") for m in train_metrics(rnd.phenotype)],
        )
    )
    if any(s.endswith("_90") for s in rnd.splits):
        blocks.append(wr.H2("Split 0: 90/10 against 80/10/10"))
        blocks.append(
            wr.PanelGrid(
                runsets=[
                    runset(rnd, "split 0 only", "Config('split') in ['s0', 's0_90']")
                ],
                panels=[
                    line("split 0: val pearson by partition", pf, "partition"),
                    line("split 0: val pearson by arm", pf, "arm"),
                    line(
                        "split 0: traineval pearson by partition",
                        f"traineval/{rnd.phenotype}/pearson_per_feature",
                        "partition",
                    ),
                    line("split 0: val loss by partition", "val/loss", "partition"),
                ],
            )
        )
    return wr.Report(
        entity=ENTITY,
        project=rnd.project,
        title=rnd.report_title,
        description="Generated by experiments/019-simb-multimodal/scripts/wandb_v13_report.py",
        blocks=blocks,
        width="fluid",
    )


def populate_view(rnd: Round) -> str:
    """Overwrite (or create) the round's saved Charts view with the ranked sections."""
    sections = [
        ws.Section(
            name=name,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
            panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
            panels=[
                wr.LinePlot(
                    x=X,
                    y=[m] if isinstance(m, str) else m,
                    title=m if isinstance(m, str) else " | ".join(m),
                    title_x="epoch",
                    layout=wr.Layout(w=8, h=6),
                )
                for m in metrics
            ],
        )
        for name, metrics in chart_sections(rnd.phenotype)
    ]
    settings = ws.WorkspaceSettings(
        x_axis=X,
        smoothing_type="none",
        max_runs=rnd.max_runs,
        sort_panels_alphabetically=False,
    )
    runset_settings = ws.RunsetSettings(
        groupby=[ws.Config("arm")],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    if rnd.view_id is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=rnd.project,
            name=rnd.view_name,
            sections=sections,
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        print(
            f"NEW saved view created: {view.url}\n"
            "  pin its `nw=` id into ROUNDS[...].view_id so later runs overwrite it"
        )
        return view.url
    view = ws.Workspace.from_url(
        f"https://wandb.ai/{ENTITY}/{rnd.project}?nw={rnd.view_id}"
    )
    view.name = rnd.view_name
    view.sections = sections
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=sorted(ROUNDS), default="v13")
    args = parser.parse_args()
    rnd = ROUNDS[args.round]
    api = wandb.Api()
    n = label_runs(api, rnd)
    print(f"labeled runs: {n} changed")
    report = build_report(rnd)
    existing = [
        r
        for r in api.reports(f"{ENTITY}/{rnd.project}")
        if r.display_name == rnd.report_title
    ]
    if existing:
        live = wr.Report.from_url(
            f"https://wandb.ai/{ENTITY}/{rnd.project}/reports/x--{existing[0].id}"
        )
        live.blocks = report.blocks
        live.width = report.width
        live.save()
        print(f"report updated: {live.url}")
    else:
        report.save()
        print(f"report created: {report.url}")
    print(f"charts view: {populate_view(rnd)}")


if __name__ == "__main__":
    main()

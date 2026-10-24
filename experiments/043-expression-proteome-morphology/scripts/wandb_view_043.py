# experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py
# [[experiments.043-expression-proteome-morphology.scripts.wandb_view_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/wandb_view_043
"""Saved W&B views for the 043 training rounds, loss curves first.

The 019 pattern (``experiments/019-simb-multimodal/scripts/wandb_v13_report.py``) for the
043 trainer: one ``Round`` per round in ``ROUNDS``. A run of this script (1) writes the
config keys ``round`` and ``arm_label`` on the round's runs (``arm_label`` is the
trainer's W&B group, ``<arm>[_balanced]_<loss>_<weighting>``, which is also the
checkpoint directory stem, so nothing is renamed), (2) tags the zombie runs of failed
smokes ``aborted`` (never deleted), and (3) overwrites the round's saved view and one saved
view per arm with the fixed section order: 1 loss, 2 headline validation, 3 per-dataset
validation, 4 train side, 5 bookkeeping. Views filter on ``Config('round')``, so runs
outside the round (the zombies included) never appear. Each view keeps only keys its own
runs log, so a panel appears the first time a run logs its key (``val/loss`` from commit
e6ad22d0c on); rerun after every sync and after every new run. Idempotent.

View ids are created on the first run (``save_as_new_view``) and recorded in
``results/wandb_view_ids.json`` under ``<round>`` and ``<round>/<arm>``, which is
committed; later runs overwrite those views in place. The group page
``/groups/<arm>/workspace`` is a personal view the API cannot write, so the arm view is
where an arm's panels live.

    python experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py --round overnight_2026_10_22
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import re
from dataclasses import dataclass

import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_043-expression-proteome-morphology"
VIEW_REGISTRY = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "wandb_view_ids.json"
)
#: Runs of failed smokes (no epoch logged): tagged, filtered out, never deleted.
ABORTED_RUNS = ("7y481s36", "epf3gh0d")
ABORTED_TAG = "aborted"
#: Every view shows every run (W&B's default is 10 per panel).
MAX_RUNS_SHOWN = 1000
#: Smoothing on by default: a five-epoch running average (the joint runs have 50 epochs,
#: so the ten-epoch average of the 019 views would flatten a fifth of a run).
SMOOTHING_TYPE = "average"
SMOOTHING_WEIGHT = 5
EPOCH = "epoch"
STEP = (
    "Step"  # the optimizer step (the trainer logs at step=step): the matched-steps axis
)


@dataclass(frozen=True)
class Round:
    """One review unit: its runs, its saved view, and its hardware line."""

    view_name: str
    run_ids: tuple[str, ...]
    hardware: str
    intro: str


ROUNDS: dict[str, Round] = {
    "overnight_2026_10_22": Round(
        view_name="043 overnight 2026-10-22 by arm",
        run_ids=("g5wda4mp", "cqdm487x", "5f3ijxa1", "kmlepsr2", "2391fl7t"),
        hardware=(
            "GilaHyper main, RTX 6000 Ada 48 GB, one run per card, jobs 6136, 6137, "
            "6150, 6152, 6151"
        ),
        intro=(
            "Seed 0 only, prep store of job 6134, registered window 0.8 to 1.0 of each "
            "run's epochs. joint_9 at 50 epochs (6,346 steps each); alone_kemmeren "
            "step-matched to it (317,300 steps, 2,102 epochs); joint_9_hlgauss_records "
            "deadline-capped at 39 epochs. No run logs val/loss (it predates e6ad22d0c)."
        ),
    ),
    "two_stage_2026_10_23": Round(
        view_name="043 two-stage smokes 2026-10-23 by arm",
        run_ids=("f3zk2sb3", "pc0zsb60", "l2fwphu9", "9rqoslxw"),
        hardware=(
            "GilaHyper main, RTX 6000 Ada 48 GB, one run per card, jobs 6172, 6173, "
            "6174, 6175"
        ),
        intro=(
            "Seed 0 smokes on prep_v2 (job 6171), mark cap 1024. Stage 1 arm all at 4 "
            "epochs (28,344 steps), plain residual and the ReZero run (removed since); stage 2 "
            "fine-tune on Kemmeren from the plain stage-1 last.pt (302 steps, lr 3e-4, "
            "warmup 30); alone_kemmeren from scratch at 28,646 steps (stage 1 + fine-tune)."
        ),
    ),
    "overnight_2026_10_23": Round(
        view_name="043 overnight 2026-10-23 (store v2) by arm",
        # stage 1 A/B (s0, s1), alone s0 (card D), stage 1 pinball (card C, job 6321);
        # the remaining runs are added as they start
        run_ids=(
            "gdqlsjpb",
            "oc881bqm",
            "18s6j2p2",
            "05d5anh5",
            "bc1awz4y",
            "jxlvzmmb",
            "zbsnrjqn",
            "fth871fz",
            "86cc6tgp",
        ),
        hardware=(
            "GilaHyper main, RTX 6000 Ada 48 GB, one run per card, jobs 6308-6313 and "
            "6316-6318 (DEADLINE 07:30 CDT 2026-10-24)"
        ),
        intro=(
            "Store v2 (prep_v3, job 6304; global_split_v2), every mark, plain residual. "
            "Stage 1 arm all at 80 epochs (7,101 steps each; window epochs 71-80), seeds 0 "
            "and 1; balanced sampler DEADLINE-capped; stage 2 fine-tune on Kemmeren, 20 "
            "epochs (3,020 steps, lr 3e-4, warmup 30, window 16-20); alone_kemmeren "
            "step-matched at 571,100 steps (3,783 epochs, validation every 20), seeds 0 and "
            "1; joint_9 DEADLINE-capped."
        ),
    ),
    "ontology_2026_10_23": Round(
        view_name="043 ontology conditioning 2026-10-23 (Polaris) by arm",
        # run id = o1023_<cell of alcf/ontology_grid_043.tsv> (resume_id), _ft = its fine-tune
        run_ids=(
            "o1023_all_onto_hlgauss_equal_s0",
            "o1023_all_onto_hlgauss_equal_s0_ft",
            "o1023_all_onto_hlgauss_equal_s1",
            "o1023_all_onto_hlgauss_equal_s1_ft",
            "o1023_all_onto_pinball_equal_s0",
            "o1023_all_onto_pinball_equal_s0_ft",
            "o1023_all_onto_pinball_equal_s1",
            "o1023_all_onto_pinball_equal_s1_ft",
            "o1023_all_onto_mse_equal_s0",
            "o1023_all_onto_mse_equal_s0_ft",
            "o1023_all_onto_mse_equal_s1",
            "o1023_all_onto_mse_equal_s1_ft",
            "o1023_all_name_hlgauss_equal_s0",
            "o1023_all_name_hlgauss_equal_s0_ft",
            "o1023_all_name_hlgauss_equal_s1",
            "o1023_all_name_hlgauss_equal_s1_ft",
            "o1023_all_name_pinball_equal_s0",
            "o1023_all_name_pinball_equal_s0_ft",
            "o1023_all_name_pinball_equal_s1",
            "o1023_all_name_pinball_equal_s1_ft",
            "o1023_all_name_mse_equal_s0",
            "o1023_all_name_mse_equal_s0_ft",
            "o1023_all_name_mse_equal_s1",
            "o1023_all_name_mse_equal_s1_ft",
            "o1023_lodo_hu_onto_hlgauss_equal_s0",
            "o1023_lodo_hu_onto_hlgauss_equal_s0_ft",
            "o1023_lodo_zelezniak_onto_hlgauss_equal_s0",
            "o1023_lodo_zelezniak_onto_hlgauss_equal_s0_ft",
            "o1023_alone_kemmeren_onto_hlgauss_equal_s0",
            "o1023_alone_kemmeren_onto_hlgauss_equal_s1",
            "o1023_all_onto_hlgauss_equal_bs16_s0",
            "o1023_all_onto_hlgauss_equal_bs16_s0_ft",
            "o1023_all_onto_hlgauss_equal_bs32_s0",
            "o1023_all_onto_hlgauss_equal_bs32_s0_ft",
            "o1023_all_onto_hlgauss_equal_s2",
            "o1023_all_onto_hlgauss_equal_s2_ft",
            "o1023_all_name_hlgauss_equal_s2",
            "o1023_all_name_hlgauss_equal_s2_ft",
        ),
        hardware=(
            "Polaris (ALCF), A100 40 GB, one PBS job per node "
            "(alcf/polaris_train_043_ontology_grid.pbs), queue and job ids in the note"
        ),
        intro=(
            "prep_v4 (prep_v3 plus the experiment ontology annotation, job 6324), store v2, "
            "every mark, equal weighting, batch 8 unless the group says bs16 / bs32, 80 "
            "epochs (window 71-80), each stage 1 chained to its Kemmeren fine-tune (20 "
            "epochs, lr 3e-4, warmup 30). Conditioning ontology (_onto groups: no dataset id "
            "enters forward) or name; lodo groups never train on that dataset; "
            "alone_kemmeren step-matched at 571,100 steps."
        ),
    ),
    "aurora_2026_10_24": Round(
        view_name="043 Aurora capacity grid 2026-10-24 by arm",
        # run id = a1024_<cell of alcf/aurora_grid_043.tsv> (resume_id), _ft = its fine-tune;
        # alone_kemmeren has no fine-tune
        run_ids=tuple(
            f"a1024_{c}{ft}"
            for c in (
                "all_onto_hlgauss_equal_s0",
                "all_onto_pinball_equal_s0",
                "all_name_hlgauss_equal_s0",
                "all_onto_mse_equal_s0",
                "all_onto_hlgauss_equal_bs16_s0",
                "all_name_hlgauss_equal_s1",
                "all_onto_hlgauss_equal_bs32_s0",
                "all_onto_hlgauss_equal_ffn2_s1",
                "all_onto_hlgauss_equal_h288_s0",
                "lodo_hu_onto_hlgauss_equal_s0",
                "all_onto_hlgauss_equal_ffn2_s0",
                "lodo_zelezniak_onto_hlgauss_equal_s0",
                "all_onto_hlgauss_equal_s1",
                "all_onto_hlgauss_equal_bs16_s2",
                "all_onto_hlgauss_equal_bs16_s1",
                "all_onto_hlgauss_equal_bs32_s2",
                "all_onto_hlgauss_equal_bs32_s1",
                "all_onto_hlgauss_equal_h288_s2",
                "all_onto_hlgauss_equal_h288_s1",
                "all_onto_hlgauss_equal_s2",
                "all_onto_hlgauss_equal_s3",
                "all_name_hlgauss_equal_s2",
                "all_name_hlgauss_equal_s3",
            )
            for ft in ("", "_ft")
        )
        + ("a1024_alone_kemmeren_onto_hlgauss_equal_s0",)
        # second submission (alcf/aurora_grid_043_b.tsv, micro-batch 1, PBS 8959835):
        # fresh ids for 12 of the cells that faulted in PBS 8959387/8959388
        + tuple(
            f"a1024b_{c}{ft}"
            for c in (
                "all_onto_hlgauss_equal_s0",
                "all_name_hlgauss_equal_s0",
                "all_onto_hlgauss_equal_bs16_s0",
                "all_onto_hlgauss_equal_bs32_s0",
                "all_onto_hlgauss_equal_h288_s0",
                "all_onto_hlgauss_equal_ffn2_s0",
                "all_onto_hlgauss_equal_s1",
                "all_onto_hlgauss_equal_bs16_s1",
                "all_onto_hlgauss_equal_bs32_s1",
                "all_onto_hlgauss_equal_h288_s1",
                "all_onto_pinball_equal_s0",
                "all_onto_mse_equal_s0",
                # grid c (alcf/aurora_grid_043_c.tsv, PBS 8959861)
                "all_onto_hlgauss_equal_s2",
                "all_name_hlgauss_equal_s1",
                "all_name_hlgauss_equal_s2",
                "all_onto_hlgauss_equal_bs16_s2",
                "all_onto_hlgauss_equal_bs32_s2",
                "all_onto_hlgauss_equal_h288_s2",
                "all_onto_hlgauss_equal_ffn2_s1",
                "all_name_hlgauss_equal_s3",
            )
            for ft in ("", "_ft")
        ),
        hardware=(
            "Aurora (ALCF), Intel Max 1550, one cell per tile; a1024_: 2 capacity nodes, "
            "PBS 8959387 and 8959388, micro-batch 2 (most faulted in epoch 1, ops/status in "
            "each run); a1024b_: PBS 8959835 and 8959861, micro-batch 1 "
            "(alcf/aurora_train_043_grid.pbs); alone_kemmeren resumed in 8959861"
        ),
        intro=(
            "prep_v4, store v2, every mark, equal weighting, HL-Gauss unless the group says "
            "pinball / mse, 80 epochs (window 71-80), each stage 1 chained to its Kemmeren "
            "fine-tune (20 epochs, lr 3e-4, warmup 30). Micro-batch 2 with gradient "
            "accumulation to the effective batch (8 unless the group says bs16 / bs32, lr "
            "3e-4 x sqrt(batch / 8)): masked SDPA on XPU forms the score matrix, so batch 8 "
            "does not fit a 64 GB tile. Capacity axis (bs16, bs32, h288, ffn2) under "
            "ontology conditioning, seeds 0-2; anchors all_onto / all_name shared with "
            "Polaris, seeds 0-3; loss family and lodo cells at seed 0; alone_kemmeren "
            "step-matched at 571,100 optimizer steps."
        ),
    ),
    "ghyper_sweep_2026_10_24": Round(
        view_name="043 GilaHyper sweep 2026-10-24 by arm",
        # run id = gs1024_<wave>_<arm> (resume_id), _ft = its Kemmeren fine-tune;
        # gh_train_043_sweep_v2.slurm, four slots chained wave 1 -> 2 -> 3. Resubmitted with
        # STOP_HOURS 14 (projected over 10 h; same run ids): smmss 6336 -> 6369, msmsms
        # 6354 -> 6370, h288 6356 -> 6371. Baseline smss 6362/6363 heads slot A.
        run_ids=tuple(
            f"gs1024_{w}_{a}{ft}"
            for w, arms in (
                ("w1", ("smss", "smmss", "mmmm", "ssss", "msmsms")),
                ("w2", ("mle", "binq", "qn", "h288")),
                ("w3", ("ffn2", "hh256", "lrhalf", "lrx2")),
            )
            for a in arms
            for ft in ("", "_ft")
        ),
        hardware=(
            "GilaHyper main, RTX 6000 Ada 48 GB, one run per card, four chained slots "
            "after the overnight round (dependency afterany:6311:6312:6317:6318:6322)"
        ),
        intro=(
            "Store v2, every mark, equal weighting, batch 8, HL-Gauss unless the group says "
            "otherwise, 80 fixed epochs (window 71-80), seed 0, each stage 1 chained to its "
            "Kemmeren fine-tune (20 epochs, lr 3e-4, warmup 30). Wave 1 trunk layer "
            "patterns (_L<pattern>: S dense, M multigraph-masked; the default trunk SMSS "
            "is the baseline); wave 2 loss family and preprocessing (mle, binned_quantile, "
            "_tqn) and hidden 288 (_h288); wave 3 trunk FFN 2x (_ffn2), head_hidden 256 "
            "(_hh256), lr 1.5e-4 and 6e-4. Composite gene token on in every run."
        ),
    ),
}

DATASETS = (
    "kemmeren",
    "messner",
    "zelezniak",
    "turnover",
    "caudal",
    "muenzner",
    "boocock",
    "nguessan",
    "hu",
    "nadal",
)

#: A panel: (title, x axis, metrics overlaid). Sections in the fixed order.
Panel = tuple[str, str, list[str]]


def sections() -> list[tuple[str, list[Panel]]]:
    """The fixed section order of every 043 view."""
    return [
        (
            "1 loss",
            [
                (
                    "train and validation loss by epoch",
                    EPOCH,
                    ["train/loss", "val/loss"],
                ),
                (
                    "train and validation loss by optimizer step (the matched-steps axis)",
                    STEP,
                    ["train/loss", "val/loss"],
                ),
                (
                    "Kemmeren loss by optimizer step (joint against alone)",
                    STEP,
                    ["train/kemmeren/loss", "val/kemmeren/loss"],
                ),
                # one train-and-validation pair per dataset (the owner reads these first)
                *[
                    (
                        f"{d} train and validation loss by epoch",
                        EPOCH,
                        [f"train/{d}/loss", f"val/{d}/loss"],
                    )
                    for d in (*DATASETS, "vaishnav")
                ],
                (
                    "per-type train loss by epoch",
                    EPOCH,
                    [
                        "train/expression/loss",
                        "train/protein/loss",
                        "train/promoter/loss",
                    ],
                ),
            ],
        ),
        (
            "2 headline validation",
            [
                (
                    "val expression per-gene median Pearson",
                    EPOCH,
                    ["val/expression/pearson_per_gene_median"],
                ),
                (
                    "val doubles, per-double centered Pearson, with the additive measured line",
                    EPOCH,
                    [
                        "val/doubles/pearson_per_strain_centered_median",
                        "val/doubles/additive_measured_pearson_per_strain_centered_median",
                    ],
                ),
                (
                    "val protein per-gene median Pearson",
                    EPOCH,
                    ["val/protein/pearson_per_gene_median"],
                ),
                ("val promoter Pearson", EPOCH, ["val/promoter/pearson"]),
            ],
        ),
        (
            "3 per-dataset validation",
            [
                (
                    f"val {d} per-gene median Pearson",
                    EPOCH,
                    [f"val/{d}/pearson_per_gene_median"],
                )
                for d in DATASETS
            ],
        ),
        (
            "4 train side",
            [
                (
                    "Kemmeren per-gene median Pearson, traineval against val",
                    EPOCH,
                    [
                        "traineval/kemmeren/pearson_per_gene_median",
                        "val/kemmeren/pearson_per_gene_median",
                    ],
                ),
                (
                    "traineval expression per-gene median Pearson",
                    EPOCH,
                    ["traineval/expression/pearson_per_gene_median"],
                ),
                (
                    "traineval protein per-gene median Pearson",
                    EPOCH,
                    ["traineval/protein/pearson_per_gene_median"],
                ),
                ("traineval promoter Pearson", EPOCH, ["traineval/promoter/pearson"]),
                ("gradient norm", EPOCH, ["train/grad_norm"]),
                ("gradient clip fraction", EPOCH, ["train/grad_norm_clip_frac"]),
                ("learning rate", STEP, ["train/lr"]),
                (
                    "operator branch norm relative to the context (is the operator live)",
                    EPOCH,
                    [
                        "operator/relative_norm",
                        "operator/ffn_relative_norm",
                        "operator/update_relative_norm",
                    ],
                ),
            ],
        ),
        (
            "5 bookkeeping",
            [
                ("epoch seconds", EPOCH, ["time/epoch_seconds"]),
                ("epochs fixed by the budget", EPOCH, ["setup/n_epochs"]),
                (
                    "registered window",
                    EPOCH,
                    ["setup/window_first_epoch", "setup/window_last_epoch"],
                ),
                ("steps per epoch", EPOCH, ["setup/steps_per_epoch"]),
                ("doubles scored", EPOCH, ["val/doubles/n_doubles"]),
            ],
        ),
    ]


def label_runs(api: wandb.Api, key: str, rnd: Round) -> dict[str, set[str]]:
    """Write ``round`` and ``arm_label``; return the logged keys per arm."""
    keys: dict[str, set[str]] = {}
    absent = []
    for rid in rnd.run_ids:
        # A round lists its chained stages (fine-tunes) before they exist; an in-flight
        # round therefore has ids with no run yet. They are reported, not an error.
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        except wandb.errors.CommError:
            absent.append(rid)
            continue
        if not run.group:
            raise ValueError(f"{rid}: no W&B group, cannot name its arm")
        if run.config.get("round") != key or run.config.get("arm_label") != run.group:
            run.config["round"] = key
            run.config["arm_label"] = run.group
            run.update()
        keys.setdefault(run.group, set()).update(run.summary.keys())
    if absent:
        print(
            f"{len(absent)} run ids of round {key} have no run yet: {' '.join(absent)}"
        )
    return keys


def tag_aborted(api: wandb.Api) -> list[str]:
    """Tag the zombie runs ``aborted``; return the ids changed."""
    changed = []
    for rid in ABORTED_RUNS:
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        if run.summary.get("epoch") is not None:
            raise ValueError(f"{rid} logged an epoch; it is not a zombie")
        if ABORTED_TAG not in run.tags:
            run.tags = [*run.tags, ABORTED_TAG]
            run.update()
            changed.append(rid)
    return changed


def _registry() -> dict[str, dict[str, str]]:
    if not osp.exists(VIEW_REGISTRY):
        return {}
    with open(VIEW_REGISTRY) as f:
        reg: dict[str, dict[str, str]] = json.load(f)
    return reg


def populate_view(key: str, rnd: Round, present: set[str], arm: str | None) -> str:
    """Overwrite (or create) one saved view; ``arm`` None is the round view."""
    built = []
    for name, panels in sections():
        kept = [(t, x, [m for m in ys if m in present]) for t, x, ys in panels]
        kept = [(t, x, ys) for t, x, ys in kept if ys]
        if not kept:
            continue
        built.append(
            ws.Section(
                name=name,
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
                panel_settings=ws.SectionPanelSettings(
                    smoothing_type=SMOOTHING_TYPE, smoothing_weight=SMOOTHING_WEIGHT
                ),
                panels=[
                    wr.LinePlot(
                        title=t,
                        x=x,
                        y=ys,
                        title_x="optimizer step" if x == STEP else "epoch",
                        smoothing_type=SMOOTHING_TYPE,
                        smoothing_factor=SMOOTHING_WEIGHT,
                        max_runs_to_show=MAX_RUNS_SHOWN,
                        layout=wr.Layout(w=8, h=6),
                    )
                    for t, x, ys in kept
                ],
            )
        )
    settings = ws.WorkspaceSettings(
        smoothing_type=SMOOTHING_TYPE,
        smoothing_weight=SMOOTHING_WEIGHT,
        max_runs=MAX_RUNS_SHOWN,
        sort_panels_alphabetically=False,
    )
    filters = [ws.Config("round") == key]
    if arm is not None:
        filters.append(ws.Config("arm_label") == arm)
    runset_settings = ws.RunsetSettings(
        filters=filters,
        groupby=[ws.Config("arm_label")],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    name = rnd.view_name if arm is None else f"043 {key}: {arm}"
    reg = _registry()
    slot = key if arm is None else f"{key}/{arm}"
    view_id = reg.get(slot, {}).get("id")
    if view_id is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=PROJECT,
            name=name,
            sections=built,
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        found = re.search(r"[?&]nw=([\w-]+)", view.url)
        if found is None:
            raise ValueError(f"no view id in {view.url}")
        reg[slot] = {"id": found.group(1), "url": view.url, "name": name}
        with open(VIEW_REGISTRY, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
            f.write("\n")
        return view.url
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={view_id}")
    view.name = name
    view.sections = built
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


#: Rounds that also get a total-loss-only view (owner, 2026-10-23: "curate charts so you
#: have train/loss and val/loss that's what i want to see"), registry slot <round>_loss.
LOSS_VIEW_ROUNDS = (
    "overnight_2026_10_23",
    "ontology_2026_10_23",
    "aurora_2026_10_24",
    "ghyper_sweep_2026_10_24",
)
LOSS_KEYS = ["train/loss", "val/loss"]


def populate_loss_view(key: str, rnd: Round) -> str:
    """Overwrite (or create) the round's total-loss view: train/loss and val/loss on one
    axis by epoch, then by optimizer step; every run of the round as its own line, no
    grouping, no smoothing, nothing else. A saved view has one runset, so a panel cannot
    hold a single run; each run's own curves are on its run page.
    """
    panels = [
        wr.LinePlot(
            title=title,
            x=x,
            y=LOSS_KEYS,
            title_x=title_x,
            smoothing_type="none",
            max_runs_to_show=MAX_RUNS_SHOWN,
            legend_template="${run:displayName} ${metricName}",
            layout=wr.Layout(w=12, h=9),
        )
        for title, x, title_x in (
            ("train/loss and val/loss by epoch", EPOCH, "epoch"),
            ("train/loss and val/loss by optimizer step", STEP, "optimizer step"),
        )
    ]
    section = ws.Section(
        name="total loss",
        is_open=True,
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
        panel_settings=ws.SectionPanelSettings(smoothing_type="none"),
        panels=panels,
    )
    settings = ws.WorkspaceSettings(
        smoothing_type="none", max_runs=MAX_RUNS_SHOWN, sort_panels_alphabetically=False
    )
    runset_settings = ws.RunsetSettings(
        filters=[ws.Config("round") == key],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    name = f"{rnd.view_name}: total loss only"
    slot = f"{key}_loss"
    reg = _registry()
    view_id = reg.get(slot, {}).get("id")
    if view_id is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=PROJECT,
            name=name,
            sections=[section],
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        found = re.search(r"[?&]nw=([\w-]+)", view.url)
        if found is None:
            raise ValueError(f"no view id in {view.url}")
        reg[slot] = {"id": found.group(1), "url": view.url, "name": name}
        with open(VIEW_REGISTRY, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
            f.write("\n")
        return view.url
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={view_id}")
    view.name = name
    view.sections = [section]
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


#: Keys of the registered-window readout, and the alone run each joint run is read against.
WINDOW_KEYS = (
    "val/kemmeren/pearson_per_gene_median",
    "val/doubles/pearson_per_strain_centered_median",
    "val/doubles/additive_measured_pearson_per_strain_centered_median",
    "val/muenzner/pearson_per_gene_median",
    "val/caudal/pearson_per_gene_median",
    "val/messner/pearson_per_gene_median",
    "val/vaishnav/pearson",
)
ALONE_ARM = "alone_kemmeren_hlgauss_equal"
TRANSFER_KEY = "val/kemmeren/pearson_per_gene_median"


def window_means(api: wandb.Api, key: str, rnd: Round) -> dict[str, object]:
    """Per run, the mean of each readout key over the evaluated epochs of its registered
    window (``setup/window_first_epoch`` .. ``setup/window_last_epoch``, read from the
    run), from the UNSAMPLED history; and the step-matched transfer on Kemmeren (joint
    minus alone). A run that has not reached its window is reported partial.
    """
    runs: dict[str, dict[str, object]] = {}
    for rid in rnd.run_ids:
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        rows = [
            r for r in run.scan_history(page_size=5000) if r.get("epoch") is not None
        ]
        first = int(run.summary["setup/window_first_epoch"])
        last = int(run.summary["setup/window_last_epoch"])
        n_epochs = int(run.summary["setup/n_epochs"])
        reached = int(max(r["epoch"] for r in rows))
        win = [r for r in rows if first <= r["epoch"] <= last]
        means: dict[str, dict[str, float | int]] = {}
        for k in WINDOW_KEYS:
            vals = [float(r[k]) for r in win if r.get(k) is not None]
            if vals:
                means[k] = {"mean": float(np.mean(vals)), "n_evaluations": len(vals)}
        runs[rid] = {
            "arm_label": run.group,
            "seed": int(run.config["seed"]),
            "state": run.state,
            "epochs_reached": reached,
            "n_epochs": n_epochs,
            "window": [first, last],
            "complete": reached >= n_epochs,
            "means": means,
        }
    alone = [
        r
        for r in runs.values()
        if r["arm_label"] == ALONE_ARM and r["complete"] and TRANSFER_KEY in r["means"]  # type: ignore[operator]
    ]
    transfer: dict[str, float] = {}
    if len(alone) == 1:
        # one alone run: every complete run is read against it (the 2026-10-22 round)
        a = alone[0]["means"][TRANSFER_KEY]["mean"]  # type: ignore[index]
        for rid, r in runs.items():
            m = r["means"]
            if r["arm_label"] != ALONE_ARM and r["complete"] and TRANSFER_KEY in m:  # type: ignore[operator]
                transfer[str(r["arm_label"])] = m[TRANSFER_KEY]["mean"] - a  # type: ignore[index]
    elif len(alone) > 1:
        # several alone seeds: each run is paired with the alone run of its own seed,
        # keyed <arm label>_s<seed>; a run whose seed has no alone run is not paired
        by_seed = {r["seed"]: r["means"][TRANSFER_KEY]["mean"] for r in alone}  # type: ignore[index]
        for rid, r in runs.items():
            m = r["means"]
            if (
                r["arm_label"] != ALONE_ARM
                and r["complete"]
                and TRANSFER_KEY in m  # type: ignore[operator]
                and r["seed"] in by_seed
            ):
                transfer[f"{r['arm_label']}_s{r['seed']}"] = (
                    m[TRANSFER_KEY]["mean"] - by_seed[r["seed"]]  # type: ignore[index]
                )
    seeds = sorted({int(r["seed"]) for r in runs.values()})  # type: ignore[call-overload]
    seed_text = "seed 0" if seeds == [0] else f"seeds {', '.join(map(str, seeds))}"
    return {
        "generated_by": "experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py",
        "round": key,
        "statistic": "mean over the evaluated epochs of each run's registered window "
        f"(setup/window_first_epoch to setup/window_last_epoch, listed per run as window), validation fold, {seed_text}, unsampled W&B history",
        "transfer_definition": "joint minus alone_kemmeren at matched optimizer steps, "
        f"on {TRANSFER_KEY}"
        + ("" if len(alone) <= 1 else "; paired by seed (same-seed alone run)"),
        "runs": runs,
        "transfer_kemmeren": transfer,
    }


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", choices=sorted(ROUNDS), required=True)
    parser.add_argument(
        "--window-means",
        action="store_true",
        help="also write results/<round>_window_means.json",
    )
    args = parser.parse_args()
    rnd = ROUNDS[args.round]
    api = wandb.Api(timeout=120)
    if args.window_means:
        out = osp.join(osp.dirname(VIEW_REGISTRY), f"{args.round}_window_means.json")
        with open(out, "w") as f:
            json.dump(window_means(api, args.round, rnd), f, indent=1)
            f.write("\n")
        print(f"window means -> {out}")
    print(f"tagged aborted: {tag_aborted(api)}")
    keys = label_runs(api, args.round, rnd)
    print(f"hardware: {rnd.hardware}")
    print(
        f"round view\n{populate_view(args.round, rnd, set().union(*keys.values()), None)}"
    )
    if args.round in LOSS_VIEW_ROUNDS:
        print(f"total-loss view\n{populate_loss_view(args.round, rnd)}")
    for arm in sorted(keys):
        print(f"{arm}: arm view, then the group page")
        print(populate_view(args.round, rnd, keys[arm], arm))
        print(f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{arm}/workspace")


if __name__ == "__main__":
    main()

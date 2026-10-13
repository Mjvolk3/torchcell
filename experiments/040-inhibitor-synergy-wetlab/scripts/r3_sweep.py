# experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep
r"""Write round 3's sweep files and the Delta submission script, one GPU per job.

Rounds 1 and 2 varied one field at a time around one recipe at 50 epochs. The W&B
history of the multi-source arms shows every auxiliary loss still falling at epoch 50
(Hoepfner 0.55 to 0.85, Hillenmeyer 0.9) while the single-source Vanacloig loss is flat
at 0.25, so a multi-source arm has been compared against a converged single-source one.
Round 3 trains longer, holds out part of every auxiliary source so each source carries
its own held-out score and held-out loss curve, and spends the budget on the edges of
the design space rather than on seeds.

Each arm becomes one sweep file of one or more configs; each config is one compound-cold
(or gene-cold) fold with three seeds. Jobs are packed so one job is one GPU and one
process, holding the folds of one arm, and no job is estimated past ``LIMIT_HOURS``.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r3_sweep.py --out-dir \\
        experiments/040-inhibitor-synergy-wetlab/conf/mixture

Writes ``r3_<arm>.yaml`` per arm, ``r3_submit_delta.sh`` (the sbatch lines, one per job),
and ``results/r3_plan.csv`` (one row per job with its arm, folds and cost estimate).

``--wave 2`` writes the SECOND WAVE (``r3w2_*``), submitted while the first is still
queued: the cells of the first wave's design that are worth running whatever the first
wave says. The seed ensemble and the last two folds of the three source-set arms (the
paper's data-combining comparison needs five folds and three seeds whichever way it
falls), the three-source arm on the other two partitions, the host-head arms under the
held-out protocol (round 1 scored them under pool-and-last), the dose FiLM twins the
first wave left out, and a compound-count learning curve with and without the public
sources. Nothing in it waits on a first-wave reading; what does wait (the ensemble on
the source encodings, weights, capacities and gene-cold arms) is round 4.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import cast

import pandas as pd
import yaml

EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")

#: Measured single-GPU cost of one config (one fold, three seeds) at 50 epochs, in hours,
#: from the round-1 and round-2 W&B runtimes: 0.78 h for one Vanacloig source on a
#: GilaHyper RTX 6000 Ada, 2.07 h for Vanacloig plus Hoepfner on a Delta A40, 2.84 h for
#: three sources on an A40. The A40 is taken as 1.4 times slower than the Ada, so these
#: are A40 hours and a GilaHyper job finishes sooner.
HOURS_PER_CONFIG_50: dict[int, float] = {1: 1.1, 2: 2.1, 3: 2.9, 4: 3.6}
#: Those are three-seed figures; this round is a SCREEN and runs one seed per fold, so an
#: arm's cost is a third of them per fold. The seed ensemble is what round 4 spends on the
#: directions this round says move.
SEEDS_PER_FOLD = 1
#: The folds a screening arm runs. Three of the five compound-cold folds cover 17 of the
#: 28 compounds, enough to order arms; a surviving arm runs all five in round 4.
SCREEN_FOLDS: list[int] = [0, 1, 2]
#: Every job asks for the partition limit rather than its estimate. The estimates below
#: are of a regime that has not converged, so they are the WORST kind of estimate to size
#: a job by, and a job that ends early costs nothing while one that is cut off loses its
#: unwritten folds.
LIMIT_HOURS = 12.0
#: What one job should aim to take; folds beyond this go to another job of the same arm.
TARGET_HOURS = 10.5

SOURCES: dict[str, list[str]] = {
    "van": ["vanacloig"],
    "vanhop": ["vanacloig", "hoepfner_hop"],
    "vanhophet": ["vanacloig", "hoepfner_hop", "hillenmeyer_het"],
    "vanhophetwil": ["vanacloig", "hoepfner_hop", "hillenmeyer_het", "wildenhain"],
    "vanwil": ["vanacloig", "wildenhain"],
    "vanhet": ["vanacloig", "hillenmeyer_het"],
}

#: The round's shared recipe: 038 round 10's converged settings, the host head trained on
#: the public anchors and the ex21 titration, the combination plate never trained on.
BASE: dict[str, object] = {
    "cgt_layers": 1,
    "cgt_lambda": 0.0,
    "eval_epochs": 5,
    # THE PROTOCOL CHANGED FOR THIS ROUND, and the first 150-epoch run is why. Under
    # ``fit_on: pool`` the fold's validation compounds are inside the training set, so the
    # logged validation score is in-sample: it climbs to 0.88 while the held-out score
    # peaks at 0.304 near epoch 85 and falls to 0.274 by 115, and the Vanacloig held-out
    # loss bottoms at 0.629 near epoch 35 and rises to 0.717 (run erueq4ns). Keeping
    # ``select: last`` at 150 epochs would therefore report an overfitted model on every
    # arm. ``fit_on: train`` holds the four validation compounds out so the selection
    # signal is honest, and ``select: val`` stops at that signal's best step. The cost is
    # four fitted compounds; the alternative is a round whose every arm is read past its
    # own optimum. The bridge arm keeps round 1's pool-and-last protocol so the two rounds
    # still join.
    "select": "val",
    "fit_on": "train",
    "gene_batch": 128,
    "env_chunk": 16,
    "host_batch": 16,
    "cond_batch": 24,
    "seeds": list(range(SEEDS_PER_FOLD)),
    "host_train": "anchors+ex21",
    "dose": "off",
    "env_layers": 1,
    "lambda_host": 1.0,
    "checkpoint": True,
}


def arm(
    name: str,
    sources: str,
    epochs: int,
    *,
    fold_seed: int = 0,
    split: str = "compound",
    aux_holdout: str | None = None,
    folds: list[int] | None = None,
    name_suffix: str = "",
    **overrides: object,
) -> dict[str, object]:
    """One arm: its source set, its length, and whatever it changes from ``BASE``.

    An arm with an auxiliary source holds out part of it by default (by compound), so the
    source has a held-out score; an arm whose question is the dose holds out doses.
    ``folds`` defaults to the screening folds; ``name_suffix`` goes between the arm and
    the fold in the config name (``mix_<arm><suffix>_f<fold>``), which is how the second
    wave's seed configs stay distinct from the first wave's while ``wandb_view.arm_of``
    still folds them into the same arm.
    """
    names = SOURCES[sources]
    if aux_holdout is None:
        aux_holdout = "compound" if len(names) > 1 else "none"
    assert len(names) > 1 or aux_holdout == "none", f"{name}: no auxiliary source"
    return {
        "arm": name,
        "sources": names,
        "epochs": epochs,
        "fold_seed": fold_seed,
        "split": split,
        "aux_holdout": aux_holdout,
        "folds": list(SCREEN_FOLDS) if folds is None else list(folds),
        "name_suffix": name_suffix,
        "overrides": overrides,
    }


def arms() -> list[dict[str, object]]:
    """The round's arms, grouped by the question each one answers."""
    out: list[dict[str, object]] = []

    # 1. Does combining data help, at a length where every source has converged? The
    #    150-epoch row is the paired comparison; 300 says whether 150 is itself enough.
    for sources in ("van", "vanhop", "vanhophet", "vanhophetwil", "vanwil"):
        out.append(arm(f"{sources}_e150", sources, 150))
    # 300 epochs on the heaviest source set only: the held-out loss is logged every five
    # epochs inside every 150-epoch arm, so saturation is read from those curves and this
    # arm is the check that reading is right rather than the way it is measured
    out.append(arm("vanhophet_e300", "vanhophet", 300))
    # the round-1 arm's own length AND its protocol, so the round-1 reading can be placed
    # on this round's axis; the only arm here that keeps pool-and-last
    out.append(arm("vanhop_e50", "vanhop", 50, fit_on="pool", select="last"))

    # 2. How the source enters: one learned token per source (the round-1 encoding), the
    #    typed attribute tokens, a per-source linear head, or both.
    out.append(
        arm("vanhophet_e150_attr", "vanhophet", 150, source_encoding="attributes")
    )
    out.append(arm("vanhophet_e150_head", "vanhophet", 150, source_head=True))
    out.append(
        arm(
            "vanhophet_e150_attr_head",
            "vanhophet",
            150,
            source_encoding="attributes",
            source_head=True,
        )
    )

    # 3. How much the auxiliary sources should count, and how many of their conditions a
    #    step sees.
    out.append(arm("vanhophet_e150_aw0p3", "vanhophet", 150, aux_weight=0.3))
    out.append(arm("vanhophet_e150_cb96", "vanhophet", 150, cond_batch=96))

    # 4. Dose. The public screens carry 83 Hoepfner and 66 Hillenmeyer compounds at two or
    #    more molar doses, so a held-out DOSE of a seen compound is a direct test of the
    #    dose channel; the compound hold-out twin says whether FiLM costs anything on the
    #    compound axis.
    out.append(
        arm("vanhop_e150_film_dose", "vanhop", 150, aux_holdout="dose", dose="film")
    )
    out.append(
        arm(
            "vanhophet_e150_film_dose",
            "vanhophet",
            150,
            aux_holdout="dose",
            dose="film",
        )
    )
    out.append(arm("vanhop_e150_film", "vanhop", 150, dose="film"))

    # 5. Gene-cold: a held-out deletion strain, every compound fitted. The in-house
    #    deletion assay is the only validation for a prediction on a gene outside the
    #    Vanacloig panel, so this is where that claim starts.
    for sources in ("van", "vanhop", "vanhophet"):
        out.append(arm(f"{sources}_e150_gene", sources, 150, split="gene"))

    # 6. Capacity, on the three-source set at the round's length.
    out.append(arm("vanhophet_e150_L2", "vanhophet", 150, env_layers=2))
    out.append(arm("vanhophet_e150_cgt2", "vanhophet", 150, cgt_layers=2))
    out.append(arm("vanhophet_e150_wide", "vanhophet", 150, dim=128, hidden=512))

    # 7. Fold seeds: the gene-level score moved 0.276 to 0.363 across partitions on one
    #    arm while nested ridge moved 0.017, so the two headline arms are repeated on the
    #    other two partitions.
    for sources in ("van", "vanhop"):
        for fold_seed in (1, 2):
            out.append(
                arm(f"{sources}_e150_fs{fold_seed}", sources, 150, fold_seed=fold_seed)
            )
    return out


def wave2_arms() -> list[dict[str, object]]:
    """The second wave: the first wave's design completed where no reading is needed."""
    out: list[dict[str, object]] = []
    headline = ("van", "vanhop", "vanhophet")

    # 1. The ensemble on the source-set arms. The first wave runs folds 0 to 2 with one
    #    seed; the paper's comparison needs five folds and three seeds, so this adds
    #    seeds 1 and 2 on the screened folds and all three seeds on the two others. The
    #    config name carries ``_s12`` for the seed-only configs so neither the results
    #    directory nor the W&B run collides with the first wave's.
    for sources in headline:
        out.append(
            arm(
                f"{sources}_e150",
                sources,
                150,
                folds=[0, 1, 2],
                name_suffix="_s12",
                seeds=[1, 2],
            )
        )
        out.append(arm(f"{sources}_e150", sources, 150, folds=[3, 4], seeds=[0, 1, 2]))

    # 2. The three-source arm on the other two partitions (the first wave repeats only
    #    the one- and two-source arms there).
    for fold_seed in (1, 2):
        out.append(
            arm(f"vanhophet_e150_fs{fold_seed}", "vanhophet", 150, fold_seed=fold_seed)
        )

    # 3. The host head under the held-out protocol. Round 1 scored anchors-only, the
    #    two-stage fine-tunes and the host weight under pool-and-last at 50 epochs; the
    #    wet-lab readouts of those arms are the panel, and they have not been measured on
    #    a model selected on a held-out signal.
    out.append(arm("van_e150_anchors", "van", 150, host_train="anchors"))
    out.append(arm("van_e150_ft", "van", 150, host_train="finetune_ex21"))
    out.append(
        arm(
            "van_e150_ftenv",
            "van",
            150,
            host_train="finetune_ex21",
            finetune_scope="host_head+env_encoder",
        )
    )
    out.append(arm("van_e150_lh0p1", "van", 150, lambda_host=0.1))
    out.append(
        arm(
            "vanhophet_e150_ftenv",
            "vanhophet",
            150,
            host_train="finetune_ex21",
            finetune_scope="host_head+env_encoder",
        )
    )

    # 4. The dose FiLM twins the first wave left out: the single-source arm with the
    #    dose channel on (round 1's dose ablation, now under the held-out protocol), and
    #    the three-source compound hold-out twin of ``vanhophet_e150_film_dose``.
    out.append(arm("van_e150_film", "van", 150, dose="film"))
    out.append(arm("vanhophet_e150_film", "vanhophet", 150, dose="film"))

    # 5. The compound-count learning curve, with and without the public sources. Under
    #    ``fit_on: train`` a fold fits its 18 or 19 train compounds; 5 and 10 are the
    #    points below that and the ``e150`` arm is the full-train point. 038 round 17 measured the curve
    #    single-source under pool-and-last (8, 16, 24 of 24); this is the same axis under
    #    the held-out protocol, and whether an auxiliary source shifts it is the
    #    data-combining question in its data-efficiency form.
    for sources in headline:
        for n_fit in (5, 10):
            out.append(
                arm(f"{sources}_e150_n{n_fit}", sources, 150, n_fit_compounds=n_fit)
            )
    return out


def round4_arms() -> list[dict[str, object]]:
    """Round 4: three more seeds on every fold of the three source-set arms.

    Written once both waves of round 3 had run. The same config at the same seed differs
    between GilaHyper and Delta by up to 0.23 on the gene-level score (median 0.05), the
    fold seed swings the single-source arm from 0.18 to 0.43, and the three-seed
    ensemble on folds 0 to 2 reads 0.290, 0.271 and 0.267 for one, two and three
    sources. The data-combining comparison is the paper's second theme and it is a null
    inside that noise, so the budget goes to power on it rather than to more directions:
    seeds 3 to 5 on folds 0 to 4, which with round 3's seeds 0 to 2 makes six seeds on
    five folds per arm. Config names carry ``_s345``.
    """
    return [
        arm(
            f"{sources}_e150",
            sources,
            150,
            folds=[0, 1, 2, 3, 4],
            name_suffix="_s345",
            seeds=[3, 4, 5],
        )
        for sources in ("van", "vanhop", "vanhophet")
    ]


def round5_arms() -> list[dict[str, object]]:
    """Round 5, for Delta: what rounds 3 and 4 said to run next.

    Three readings decide it. The same config at the same seed differs between
    machines by up to 0.23 and the fold seed swings the single-source arm from 0.18 to
    0.43, so any number the paper reports has to be an ensemble over seeds AND over
    partitions; the public screens are barely learned (held-out compound 0.03 to 0.10
    against the chemistry learners' 0.13 to 0.19) under every encoding, weight and
    capacity tried at one seed; and the Vanacloig held-out loss bottoms near epoch 35
    while the training loss keeps falling, which no arm has yet answered with
    regularization, since the width, dropout, learning rate and weight decay were taken
    from 038 round 10 and never moved here. Four blocks:

    1. the partition ensemble: seeds 1 and 2 of the three source-set arms on fold seeds
       1 and 2 (seed 0 ran in round 3), so the data-combining comparison is six seeds
       on fold seed 0 and three on each of the other two;
    2. a regularization grid on the single-source arm at three seeds: dropout 0.4,
       weight decay 1e-3, both, learning rate halved and doubled, and a narrower model;
    3. the selection protocol: eight validation compounds instead of four, and the
       round-1 length with last-step selection under the held-out fit, which separates
       the selector from the length;
    4. the auxiliary sources weighted up (3.0) with and without the larger condition
       batch, at three seeds on the two-source arm, which is the one direction the
       per-source held-out scores have not been tested in.
    """
    out: list[dict[str, object]] = []
    for sources in ("van", "vanhop", "vanhophet"):
        for fold_seed in (1, 2):
            out.append(
                arm(
                    f"{sources}_e150_fs{fold_seed}",
                    sources,
                    150,
                    fold_seed=fold_seed,
                    name_suffix="_s12",
                    seeds=[1, 2],
                )
            )
    three = [0, 1, 2]
    for name, overrides in (
        ("van_e150_do0p4", {"dropout": 0.4}),
        ("van_e150_wd1e3", {"weight_decay": 1e-3}),
        ("van_e150_do0p4_wd1e3", {"dropout": 0.4, "weight_decay": 1e-3}),
        ("van_e150_lr5e4", {"lr": 5e-4}),
        ("van_e150_lr2e3", {"lr": 2e-3}),
        ("van_e150_dim32", {"dim": 32, "hidden": 128}),
        ("van_e150_nval8", {"n_val": 8}),
        ("van_e50_last", {"select": "last"}),
    ):
        out.append(
            arm(name, "van", 150 if "e150" in name else 50, seeds=three, **overrides)
        )
    out.append(arm("vanhop_e150_aw3", "vanhop", 150, seeds=three, aux_weight=3.0))
    out.append(
        arm(
            "vanhop_e150_aw3_cb96",
            "vanhop",
            150,
            seeds=three,
            aux_weight=3.0,
            cond_batch=96,
        )
    )
    return out


def round5b_arms() -> list[dict[str, object]]:
    """Round 5b, for Delta: what the convergence read of rounds 3 and 4 adds.

    ``r3_convergence.py`` over 117 runs on both machines: the Vanacloig training loss
    falls six-fold while its held-out loss removes 11 percent of its first value,
    bottoms near epoch 34 and rises by more than it removed; Hoepfner's training loss
    falls 0.85 to 0.59 and its held-out loss removes 13 percent; Hillenmeyer's training
    loss RISES over training (0.85 to 0.92, 1.12 under auxiliary weight 0.3) and its
    held-out loss removes 5 percent, so that source is not learned at all; Wildenhain's
    held-out loss removes 41 percent, bottoms at epoch 80 to 100 and is still falling in
    every run, so the one source the model does learn is cut off by the schedule. Round
    5 already carries the regularization grid for the first reading and the auxiliary
    weight for Hoepfner; this adds the two the table left out:

    * Hillenmeyer weighted up with the larger condition batch on the three-source arm,
      at 3 and at 10 (two seeds: three would run eleven hours on an A40 against the
      twelve-hour limit, and a cut-off loses the whole config), and the two-source arm
      with Hillenmeyer alone, plain and weighted, so a source that competes with
      Hoepfner for the gradient is seen without it;
    * Wildenhain at 300 epochs and at the larger condition batch, one seed, since its
      held-out loss was still falling at 150.
    """
    out: list[dict[str, object]] = []
    three = [0, 1, 2]
    out.append(
        arm(
            "vanhophet_e150_aw3_cb96",
            "vanhophet",
            150,
            seeds=[0, 1],
            aux_weight=3.0,
            cond_batch=96,
        )
    )
    out.append(
        arm(
            "vanhophet_e150_aw10_cb96",
            "vanhophet",
            150,
            seeds=[0, 1],
            aux_weight=10.0,
            cond_batch=96,
        )
    )
    out.append(arm("vanhet_e150", "vanhet", 150, seeds=three))
    out.append(
        arm(
            "vanhet_e150_aw3_cb96",
            "vanhet",
            150,
            seeds=three,
            aux_weight=3.0,
            cond_batch=96,
        )
    )
    out.append(arm("vanwil_e300", "vanwil", 300))
    out.append(arm("vanwil_e150_cb96", "vanwil", 150, cond_batch=96))
    return out


def round6_arms() -> list[dict[str, object]]:
    """Round 6, GilaHyper: the graph regularization the model config carries but no
    round has switched on. ``cgt_lambda`` scales the KL between the nine regularized
    heads' attention on layer 1 and the nine graph priors (``cgt_model.yaml``); it has
    been 0.0 in every arm of this experiment, so the priors have shaped only the layer-0
    attention bias. The single-source arm at three seeds, with the penalty at 0.1 and
    at 1.0 (the 010 value).
    """
    return [
        arm("van_e150_cgt0p1", "van", 150, seeds=[0, 1, 2], cgt_lambda=0.1),
        arm("van_e150_cgt1", "van", 150, seeds=[0, 1, 2], cgt_lambda=1.0),
    ]


def round7_arms() -> list[dict[str, object]]:
    """Round 7: every record is a cell. ``readout: cell`` reads the public screens and
    the host through ONE head, the host as the record with the empty genotype (its own
    source token, scale and shift, no gene bias), so the two readouts of the main sheet
    are two measurements of one cell state rather than a gene head and a host head. The
    single-source arm is the equivalence check on the public side (the gene-level
    computation is unchanged in form); the dose arm asks whether a host that shares the
    gene-level head can now read dose; the two- and three-source arms ask whether more
    cells of the same record type move the shared head; ``lambda_host`` 3 weights the
    host records, which are 209 against the gene cells, inside the shared head.
    """
    specs = [
        ("van_e150_cell", "van", {}),
        ("van_e150_cell_film", "van", {"dose": "film"}),
        ("van_e150_cell_lh3", "van", {"lambda_host": 3.0}),
        ("vanhop_e150_cell", "vanhop", {}),
        ("vanhophet_e150_cell", "vanhophet", {}),
    ]
    out: list[dict[str, object]] = []
    for name, sources, overrides in specs:
        out.append(arm(name, sources, 150, readout="cell", **overrides))
    # the seed twins (seeds 1 and 2, as wave 2's ``_s12`` configs), so each arm reads
    # as a three-seed ensemble once both files are in; GilaHyper runs the seed-0 files,
    # Delta runs both
    for name, sources, overrides in specs:
        out.append(
            arm(
                name,
                sources,
                150,
                readout="cell",
                name_suffix="_s12",
                seeds=[1, 2],
                **overrides,
            )
        )
    return out


def round8_arms() -> list[dict[str, object]]:
    """Round 8: the round-5 questions re-asked under the cell readout, since every old
    readout job was cancelled (2026-10-12). Host side: the anchors-only arm is the
    zero-shot test of the shared head (it never sees a titration well), the fine-tune
    arm moves the shared head and tokens on the titration alone after the anchors. The
    partition arms (fold seeds 1 and 2) give the noise floor of the cell readout. Source
    side: the 96-condition auxiliary batch and the auxiliary weight 3 on the two-source
    cell arm, the two levers the gene readout never resolved.
    """
    return [
        arm("van_e150_cell_anchors", "van", 150, readout="cell", host_train="anchors"),
        arm(
            "van_e150_cell_ft",
            "van",
            150,
            readout="cell",
            host_train="finetune_ex21",
            finetune_scope="host_head+env_encoder",
        ),
        arm("van_e150_cell_fs1", "van", 150, readout="cell", fold_seed=1),
        arm("van_e150_cell_fs2", "van", 150, readout="cell", fold_seed=2),
        arm("vanhop_e150_cell_cb96", "vanhop", 150, readout="cell", cond_batch=96),
        arm("vanhop_e150_cell_aw3", "vanhop", 150, readout="cell", aux_weight=3.0),
    ]


def round9_arms() -> list[dict[str, object]]:
    """Round 9: can the shared head COMPOSE inhibitors. Rounds 7 and 8 read the same
    host fitness at one, two and three inhibitors on every cell arm (0.63 / 0.60 / 0.59
    on the single-source arm, 0.69 / 0.67 / 0.67 zero-shot), and the model reduces a
    medium's compound tokens by their MEAN in both places the host reads them (the
    medium offset and the token summary), so a three-inhibitor medium is read as the
    average of its inhibitors by construction.

    ``compound_pool: sum`` was the first lever (one fold ran before the Vanacloig store
    was rebuilt under it, 2026-10-12 19:18): it moved the composition the WRONG way,
    0.74 / 0.78 / 0.86 at one, two and three inhibitors, growth AUROC 0.30, because a
    summed token is outside anything the head is trained on. The single-source sum arm
    stays for the three-fold record; the other sum arms are replaced by ``bliss``: a
    k-compound medium is read as k single-compound media through the head, the host's
    fitness is the product of its single-agent fitness, a source's standardized
    response the sum, so every reading is of the kind the head is trained on and the
    composition carries no free parameter. The anchors arm reads it zero-shot, the
    FiLM arm with the dose channel, the three-source arm with Hillenmeyer's 26
    heterozygous pairs as the multi-compound training records the additive gene-level
    composition is fitted on. ``lambda_host`` 0.1 is the best public gene-readout arm
    of wave 2 re-asked under the cell readout, and ``_s345`` on folds 0 to 4 is the
    round-4 seed ensemble for the cell readout, the noise floor every other reading is
    held against.
    """
    return [
        arm("van_e150_cell_sum", "van", 150, readout="cell", compound_pool="sum"),
        arm("van_e150_cell_bliss", "van", 150, readout="cell", compound_pool="bliss"),
        arm(
            "van_e150_cell_anchors_bliss",
            "van",
            150,
            readout="cell",
            host_train="anchors",
            compound_pool="bliss",
        ),
        arm(
            "van_e150_cell_bliss_film",
            "van",
            150,
            readout="cell",
            compound_pool="bliss",
            dose="film",
        ),
        arm(
            "vanhophet_e150_cell_bliss",
            "vanhophet",
            150,
            readout="cell",
            compound_pool="bliss",
        ),
        arm("van_e150_cell_lh0p1", "van", 150, readout="cell", lambda_host=0.1),
        arm(
            "van_e150_cell",
            "van",
            150,
            readout="cell",
            folds=[0, 1, 2, 3, 4],
            name_suffix="_s345",
            seeds=[3, 4, 5],
        ),
    ]


def configs(spec: dict[str, object], folds: list[int]) -> list[dict[str, object]]:
    """One config per fold of ``spec``."""
    out = []
    for fold in folds:
        config = dict(BASE)
        config |= {
            "name": f"mix_{spec['arm']}{spec['name_suffix']}_f{fold}",
            "sources": spec["sources"],
            "epochs": spec["epochs"],
            "fold_seed": spec["fold_seed"],
            "split": spec["split"],
            "aux_holdout": spec["aux_holdout"],
            "folds": [fold],
        }
        config |= spec["overrides"]  # type: ignore[arg-type]
        out.append(config)
    return out


def spec_seeds(spec: dict[str, object]) -> list[int]:
    """The seeds a spec's configs run: its override, else the round's one seed."""
    overrides = cast(dict[str, object], spec["overrides"])
    return cast(list[int], overrides.get("seeds", BASE["seeds"]))


def hours(spec: dict[str, object], n_folds: int) -> float:
    """Estimated single-A40 hours for ``n_folds`` folds of ``spec``."""
    per = (
        HOURS_PER_CONFIG_50[len(spec["sources"])]
        * int(spec["epochs"])
        / 50
        * len(spec_seeds(spec))
        / 3
    )
    if spec["overrides"].get("cond_batch") == 96:  # type: ignore[union-attr]
        per *= 1.6
    if (
        spec["overrides"].get("env_layers") == 2
        or spec["overrides"].get("cgt_layers") == 2
    ):  # type: ignore[union-attr]
        per *= 1.4
    if spec["overrides"].get("dim") == 128:  # type: ignore[union-attr]
        per *= 1.3
    return per * n_folds


def pack(spec: dict[str, object]) -> list[list[int]]:
    """The arm's folds split into jobs that each stay near ``TARGET_HOURS``."""
    per_fold = hours(spec, 1)
    per_job = max(1, int(TARGET_HOURS // per_fold))
    folds = cast(list[int], spec["folds"])
    return [folds[i : i + per_job] for i in range(0, len(folds), per_job)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=osp.join(EXPERIMENT, "conf", "mixture"))
    parser.add_argument("--account", default="bfjt-delta-gpu")
    parser.add_argument(
        "--wave",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        help=(
            "1 writes the r3_* screen, 2 the r3w2_* second wave, 3 the r4_* seed"
            " ensemble, 4 the r5_* Delta round, 5 the r5b_* convergence follow-up,"
            " 6 the r6_* graph regularization, 7 the r7_* cell readout, 8 the r8_*"
            " round-5 questions under the cell readout, 9 the r9_* composition round"
        ),
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    prefix = {
        1: "r3",
        2: "r3w2",
        3: "r4",
        4: "r5",
        5: "r5b",
        6: "r6",
        7: "r7",
        8: "r8",
        9: "r9",
    }[args.wave]
    waves = {
        1: arms,
        2: wave2_arms,
        3: round4_arms,
        4: round5_arms,
        5: round5b_arms,
        6: round6_arms,
        7: round7_arms,
        8: round8_arms,
        9: round9_arms,
    }
    specs = waves[args.wave]()
    label = {
        1: "ROUND 3 WAVE 1",
        2: "ROUND 3 WAVE 2",
        3: "ROUND 4",
        4: "ROUND 5",
        5: "ROUND 5B",
        6: "ROUND 6",
        7: "ROUND 7",
        8: "ROUND 8",
        9: "ROUND 9",
    }[args.wave]

    rows, lines = [], []
    for spec in specs:
        groups = pack(spec)
        for k, folds in enumerate(groups):
            suffix = "" if len(groups) == 1 else f"_p{k}"
            sweep = f"{prefix}_{spec['arm']}{spec['name_suffix']}{suffix}"
            path = osp.join(args.out_dir, f"{sweep}.yaml")
            estimate = hours(spec, len(folds))
            limit = LIMIT_HOURS
            seeds = spec_seeds(spec)
            header = (
                f"# experiments/040-inhibitor-synergy-wetlab/conf/mixture/{sweep}.yaml\n"
                f"#\n"
                f"# {label}, ARM {spec['arm']}, folds {folds} of fold seed"
                f" {spec['fold_seed']}, seeds {seeds}, written by scripts/r3_sweep.py"
                f"{' --wave 2' if args.wave == 2 else ''}.\n"
                f"# Sources: {', '.join(spec['sources'])}."
                f" Split: {spec['split']}. Auxiliary hold-out: {spec['aux_holdout']}."
                f" Epochs: {spec['epochs']}.\n"
                f"# ESTIMATE, not a measurement: {estimate:.1f} A40 hours for this file"
                f" (round-1 runtimes scaled by epochs and sources); the job asks for"
                f" {limit:.1f} h.\n"
                f"#\n"
                f"#   sbatch --job-name=040-{sweep} --time={int(limit)}:00:00"
                f" --export=ALL,SWEEP=conf/mixture/{sweep}.yaml,PARALLEL=1 \\\n"
                f"#     experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm\n"
            )
            with open(path, "w") as f:
                f.write(header)
                yaml.safe_dump(
                    configs(spec, folds), f, sort_keys=False, default_flow_style=False
                )
            lines.append(
                f"sbatch --job-name=040-{sweep} --account={args.account}"
                f" --qos={args.account} --time={int(limit)}:00:00"
                f" --export=ALL,SWEEP=conf/mixture/{sweep}.yaml,PARALLEL=1 \\\n"
                f"  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm"
            )
            rows.append(
                {
                    "job": len(rows) + 1,
                    "sweep": sweep,
                    "arm": spec["arm"],
                    "sources": " ".join(spec["sources"]),
                    "n_sources": len(spec["sources"]),
                    "epochs": spec["epochs"],
                    "split": spec["split"],
                    "aux_holdout": spec["aux_holdout"],
                    "fold_seed": spec["fold_seed"],
                    "folds": " ".join(str(f) for f in folds),
                    "configs": len(folds),
                    "estimated_hours": round(estimate, 2),
                    "time_limit_hours": int(limit),
                    "overrides": ", ".join(
                        f"{k}={v}"
                        for k, v in spec["overrides"].items()  # type: ignore[union-attr]
                    ),
                }
            )

    plan = pd.DataFrame(rows)
    plan_path = osp.join(RESULTS, f"{prefix}_plan.csv")
    plan.to_csv(plan_path, index=False)
    script = osp.join(args.out_dir, f"{prefix}_submit_delta.sh")
    wave_note = (
        ""
        if args.wave == 1
        else (
            "# THE SECOND WAVE. The learning-curve arms (r3w2_*_n5, _n10) run a code path\n"
            "# added after the first wave was rsynced (n_fit_compounds under fit_on: train),\n"
            "# so re-rsync the tree before this script: anchored excludes, NO delete flags.\n"
            "#\n"
        )
    )
    with open(script, "w") as f:
        f.write(
            "#!/bin/bash\n"
            f"# experiments/040-inhibitor-synergy-wetlab/conf/mixture/{prefix}_submit_delta.sh\n"
            "#\n"
            f"# {label} SUBMISSION, written by scripts/r3_sweep.py"
            f"{' --wave 2' if args.wave == 2 else ''}. One GPU and one\n"
            "# process per job, no dependencies and no -preempt partition, so every job\n"
            "# is eligible the moment it is submitted.\n"
            "#\n"
            f"{wave_note}"
            "#   ssh delta\n"
            "#   cd /scratch/bbub/mjvolk3/torchcell-040\n"
            "#   mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output\n"
            f"#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/{prefix}_submit_delta.sh\n"
            "#\n"
            "# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets\n"
            "# the matching QOS as well (an account change alone leaves the old QOS and\n"
            "# the job is rejected as InvalidQOS).\n"
            "set -eu\n"
            f'ACCOUNT="${{ACCOUNT:-{args.account}}}"\n'
            "cd /scratch/bbub/mjvolk3/torchcell-040\n"
            "mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output\n\n"
        )
        for line in lines:
            f.write(
                line.replace(
                    f"--account={args.account}", '--account="$ACCOUNT"'
                ).replace(f"--qos={args.account}", '--qos="$ACCOUNT"')
                + "\n\n"
            )
    os.chmod(script, 0o755)

    print(plan.to_string(index=False))
    print(
        f"\n{len(plan)} jobs, {plan['configs'].sum()} configs, "
        f"{plan['estimated_hours'].sum():.0f} estimated A40 hours, "
        f"{plan['time_limit_hours'].sum()} hours requested"
    )
    print(f"wrote {script}")
    print(f"wrote {plan_path}")


if __name__ == "__main__":
    main()

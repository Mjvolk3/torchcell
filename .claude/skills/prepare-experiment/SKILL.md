---
name: prepare-experiment
description: Pre-launch gate for any training round (a new trainer, a new experiment, or a new arm on GilaHyper, IGB or Delta). Run it before the first sbatch of a chain, and again after any trainer change. It pins the logging contract (train AND validation loss on one axis, per target dataset), the smoke-then-chain order, the step-matched controls, the registered window, and the W&B view, and it fails the launch when a smoke run's history lacks a required curve. Use for "launch the round", "submit the chain", "set up the overnight tests", "new trainer".
---

# Prepare an experiment round

A round is submitted in two steps, never one: a SMOKE run (one card, a few epochs,
real data, real logger), then the chain. Between them the smoke's W&B history is
checked by a script, not by eye. This skill exists because the same omission has now
happened twice: 040 round 3 (2026-10-12) and the 043 overnight round (2026-10-22) each
ran a full night of GPU time with no validation loss logged, and a loss curve without
its validation twin cannot be read for overfitting. The owner, 2026-10-23: "this should
never happen and you have done it before." Memory `always-show-train-and-validation-loss-together`
had the rule; a memory is advisory, this gate is mechanical.

## 1. Logging contract (every trainer, every round)

Logged at every evaluation step, as curves (one row per evaluation, never a single
summary row at the end):

| key | meaning |
|---|---|
| `epoch` | the epoch counter |
| `train/loss` | the training objective, averaged over the epoch |
| `val/loss` | the SAME objective over the validation pass, same weighting |
| `train/<dataset>/loss`, `val/<dataset>/loss` | the per-target pair, so a target's train and validation loss share one axis |
| `train/lr`, `train/grad_norm` | optimizer state |
| per-target validation scores | the metric the decision rule reads (per-gene Pearson median, centered per-strain Pearson, ...) |
| test (fold 0) scores at the current step | so selection bias stays visible |

Composite objectives do not excuse a missing target pair: the composite total is extra,
never a substitute (memory `always-show-train-and-validation-loss-together`). Anything
written to a history csv is also logged to W&B at that step (memory
`log-agreed-metrics-to-wandb`). Names are plain: `train/`, `val/`, `test/` prefixes,
never trainer-internal names.

## 2. The gate

After the smoke run has logged at least two evaluations:

```bash
~/miniconda3/envs/torchcell/bin/python scripts/check_wandb_run_keys.py \
    zhao-group/<project>/<run_id> --require val/<target>/loss --require val/<target>/<score>
```

Exit 1 means the trainer is fixed and the smoke rerun before any chain is submitted.
The gate is run again after every trainer change, because a change between rounds is
how 043 lost the curve: the smokes were launched, the chain was built on them, and no
one looked at the key list. The gate's output (key, row count, verdict) goes into the
round's dated section in the experiment note, verbatim.

## 3. Chain design

- **Smoke first, chain second.** No `--dependency` chain is submitted off a smoke that
  has not passed the gate.
- **Honest `--time`** from the smoke's measured epoch wall time (training plus
  evaluation), about 1.5x, inside the owner's card-hours; a `DEADLINE` cap when the
  grant is a window ("about 10 hours overnight").
- **`afterany`** between chain members so a crash never stalls the chain; card cap as
  granted (GilaHyper: at most the cards the owner named, and never card 3 by name).
- **Controls matched on optimizer steps**, never on epochs: an alone run against a
  joint run needs `max_steps` equal to the joint run's total steps, and the trainer
  must refuse to train a shorter unmatched run.
- **Registered window** fixed before launch (the last 20 % of epochs; validation never
  selects the epoch), and the window checkpoints the round needs saved at every window
  epoch when the analysis will read them.
- **Seeds:** a single seed is a smoke, not a result; the decision rule (gain >= 0.02,
  5 of 6 paired init seeds, strain-resampled CI) names the seed count the chain must
  carry before any arm is called better.

## 4. W&B view before the first result is read

A committed `wandb_view_<exp>.py` (pattern: `experiments/043-expression-proteome-morphology/scripts/wandb_view_043.py`)
writes the round's saved view with the sections in this order: loss (train and
validation on one panel, grouped by arm, smoothing on; a step-axis twin when arms differ
in epochs per step), headline validation, per-dataset validation, train side,
bookkeeping; one arm view per group; view ids in `results/wandb_view_ids.json`; aborted
runs tagged, not deleted. Follow `wandb-view` for links: every URL on its own line,
`/groups/<arm>/workspace` beside each arm view.

## 5. Before saying "submitted"

- the gate output for the smoke, pasted in the note;
- the queue table (job id, run, dependency, `--time`, trainer budget, expected end);
- the sbatch lines added to the launcher;
- which arms were dropped for budget, by name.

Partial runs are partial; a window mean is reported with its epochs and seed count.

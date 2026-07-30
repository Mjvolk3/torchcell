# experiments/019-simb-multimodal/scripts/tag_runs.py
# [[experiments.019-simb-multimodal.scripts.tag_runs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/tag_runs
"""The W&B tagging convention for 019 experiments, and a backfill for runs that predate it.

WHY TAGS AND NOT CONFIG. W&B config is queryable but not FILTERABLE at a glance -- you cannot
scan a run list and see which arms share a hypothesis, or which ones depend on curated yeast
graphs. Tags are the axis the UI filters on, so anything used to GROUP or EXCLUDE experiments
belongs here. This exists because `A2_prop_sparse` was scheduled into a wave despite failing
the organism-transferability requirement: nothing in the run list showed that it routes over
curated yeast interaction graphs, so the constraint was not visible at review time.

THE AXES (each run gets one tag per axis, so any axis can be filtered independently):

  config    cgt_expr_008          which config file
  arm       A4_nullsink           the exact arm; the SCORER derives the arm from this tag
  seed      seed0                 the seed
  wave      wave3                 source-version cohort -- runs are NEVER pooled across waves
  stage     stage-arch            ladder (optimizer) vs arch (architecture)
  pack      pack2                 runs co-resident per GPU; part of the comparison conditions
  mech      mech-nullsink         the HYPOTHESIS CLASS, so sibling mechanisms group together
  xfer      xfer-yes / xfer-no    does the mechanism transfer to an organism WITHOUT curated
                                  yeast graphs? `xfer-no` means it cannot be part of the
                                  generalizable story regardless of how well it scores.

`xfer` is the one that would have caught A2. Graph-propagation arms are `xfer-no`: they need
`regulatory_interaction` / `tflink` / `physical_interaction` adjacencies that do not exist for
an arbitrary organism. The null sink, extra FFN depth, and the reference are `xfer-yes` --
they need only the perturbation set and a per-gene representation.

Usage:
    python experiments/019-simb-multimodal/scripts/tag_runs.py            # dry run
    python experiments/019-simb-multimodal/scripts/tag_runs.py --apply    # write to W&B
"""

import argparse

import wandb
from dotenv import load_dotenv

load_dotenv()

# arm -> (mechanism class, transfers to a graph-less organism?)
ARM_META: dict[str, tuple[str, bool]] = {
    # Reference / null. `A1_ref` is the paired denominator; `A5_sham` is the empirical null
    # -- numerically the identity, but on the same code path as the sink.
    "A1_ref": ("baseline", True),
    "A5_sham": ("nullsink-sham", True),
    # Supplies the missing pair-(p, i) term INSIDE the attention, from learned embeddings.
    "A4_nullsink": ("nullsink", True),
    # Supplies the same pair term from OUTSIDE, using curated yeast adjacencies. Not
    # transferable -- this is the axis that disqualifies it from the generalizable story.
    "A2_prop_sparse": ("graph-routing", False),
    "A3_prop_h1": ("graph-routing", False),
    "A4_prop_h2": ("graph-routing", False),
    "A6_prop_sparse": ("graph-routing", False),
    # No pair term -- capacity only.
    "A3_deep2": ("capacity", True),
    "F1_deep2": ("capacity", True),
    "F1_deep4": ("capacity", True),
    # Optimizer ladder: architecture untouched.
    "L0_lr3e4": ("optimizer", True),
    "L1_lr1e3": ("optimizer", True),
    "L2_lr3e3": ("optimizer", True),
    "L3_lr1e3_cosine": ("optimizer", True),
}

# ONE SENTENCE PER ARM: why this run is supposed to differ from the reference.
#
# ORTHOGONAL TO TAGS, and needed for a different reason. Tags are for FILTERING -- they
# answer "show me every transferable-mechanism arm in wave 3". They cannot carry intent: a
# run named `A4_nullsink` tells you what was switched on, never what was expected to happen
# or why. `run.notes` is free text shown in the W&B run table, so the hypothesis travels with
# the run instead of living in a config comment nobody opens six weeks later.
#
# Keep each to ONE sentence, stating the expected DIFFERENCE, not the mechanism's plumbing.
ARM_NOTE: dict[str, str] = {
    "A1_ref": (
        "Paired reference: unchanged A0 architecture, co-launched so every other arm is "
        "scored as a within-seed difference against it rather than against a past run."
    ),
    "A4_nullsink": (
        "Adds one learned scalar on a zero-valued attention sink so the |S|=1 softmax stops "
        "being degenerate; expected to make the perturbation context depend on the MEASURED "
        "gene (a pair-(p,i) term) instead of being one vector shared by all 6,607 genes."
    ),
    "A5_sham": (
        "Empirical null for A4: identical code path and parameter, but the sink is frozen "
        "shut (bias -20, p_null ~ 2e-9), so its paired delta must be ~0 -- if it is not, the "
        "paired estimator does not transfer to arms that change the forward pass at step 0."
    ),
    "A3_deep2": (
        "Second perturbation-transform layer: tests whether the gain available here is plain "
        "FFN CAPACITY on (h_i, context) rather than any missing pair-(p,i) structure."
    ),
    "A2_prop_sparse": (
        "Routes the deletion 2 hops over the three sharp yeast graphs, supplying the pair-"
        "(p,i) term from network topology instead of learning it; NOT organism-transferable, "
        "so it is a diagnostic only and cannot be part of the generalizable story."
    ),
    "A6_prop_sparse": (
        "Same as A2_prop_sparse under its wave-2 name; graph-routed pair term, yeast-only."
    ),
    "L0_lr3e4": (
        "Optimizer ladder control at the incumbent lr 3e-4, held since _006; architecture "
        "untouched so the ladder measures optimization, not capacity."
    ),
    "L1_lr1e3": "Ladder cell: 3.3x the incumbent lr, testing whether the train fit is step-size limited.",
    "L2_lr3e3": "Ladder cell: 10x the incumbent lr, the aggressive end of the step-size probe.",
    "L3_lr1e3_cosine": (
        "Ladder cell: 3.3x lr WITH cosine annealing, separating 'higher lr helps' from "
        "'higher lr helps but needs decay to settle'."
    ),
}

# Source-version cohorts. A run is scored only against runs in its own wave.
WAVE_BOUNDARIES = [("2026-07-28T22:26:23", "wave2"), ("2026-07-29T00:00:00", "wave3")]


def wave_of(created_at: str) -> str:
    wave = "wave1"
    for boundary, name in WAVE_BOUNDARIES:
        if str(created_at) >= boundary:
            wave = name
    return wave


def derive_tags(run: "wandb.apis.public.Run") -> list[str]:
    """Compute the full tag set for a run, preserving whatever is already there."""
    tags = set(run.tags)
    arm = next((t for t in run.tags if t in ARM_META), None)
    if arm is None:
        return sorted(tags)  # not an arm run (grid sweep, probe, ...) -- leave alone

    mech, xfer = ARM_META[arm]
    tags.add(wave_of(run.created_at))
    tags.add(f"mech-{mech}")
    tags.add("xfer-yes" if xfer else "xfer-no")
    tags.add("stage-ladder" if arm.startswith("L") else "stage-arch")
    # Packing is part of the comparison conditions, so it is a filterable axis: a run
    # compared against one packed differently is not a clean comparison.
    if wave_of(run.created_at) == "wave3":
        tags.add("pack2")
    return sorted(tags)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="torchcell_019_expr_v8")
    p.add_argument("--entity", default="zhao-group")
    p.add_argument("--apply", action="store_true", help="write tags (default: dry run)")
    args = p.parse_args()

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{args.entity}/{args.project}", per_page=300))
    changed = 0
    for run in runs:
        new = derive_tags(run)
        arm = next((t for t in run.tags if t in ARM_META), None)
        note = ARM_NOTE.get(arm or "", "")
        tags_differ = new != sorted(run.tags)
        # Only ever ADD a note; never overwrite one a human wrote by hand.
        note_differ = bool(note) and not (run.notes or "").strip()
        if tags_differ or note_differ:
            bits = []
            if tags_differ:
                bits.append(f"+{sorted(set(new) - set(run.tags))}")
            if note_differ:
                bits.append("+note")
            print(f"{run.id:<10} {run.created_at}  {' '.join(bits)}")
            changed += 1
            if args.apply:
                run.tags = new
                if note_differ:
                    run.notes = note
                run.update()
    print(f"\n{changed} of {len(runs)} runs {'updated' if args.apply else 'would change'}")
    if not args.apply and changed:
        print("re-run with --apply to write")


if __name__ == "__main__":
    main()

---
name: wandb-curate
description: Curate a W&B Charts view for a training round and hand back its link. Use whenever the user asks to see W&B, wants charts ranked by importance, asks for a grouped or across-arm comparison, says "show me the run", or after any sync of offline runs. Never send a bare project link or the personal workspace; always a saved view built by a committed script.
---

# W&B curation

The user reads training through W&B and wants the same three things every time:
the headline metric compared across arms, train and validation each on their own
panel AND together on one panel, and the panels ranked so the first row answers the
question of the round. The workspace API cannot write the personal default view
(`?nw=nwuser...`), so every curated view is a SAVED view whose `nw=` id is pinned in
a committed script and overwritten on each rerun.

## Where the scripts live

One script per experiment and split, beside the readout script it pairs with:

- `experiments/025-solid-growth/scripts/disjoint_embedding_wandb_view.py`, Q split, view `paezdq4q5ex`
- `experiments/025-solid-growth/scripts/random_split_wandb_view.py`, R split, view `vo1fa9efqdf`

A new experiment or split gets a new script by copying one of these: change `ARMS`,
`SECTIONS`, `VIEW_NAME`, set `VIEW_ID = None`, run once, pin the printed id, commit.
Both scripts label runs first (`label_runs`) and then build the view (`populate_view`).

## Procedure, every time the user asks to see W&B

1. Sync first if the runs are offline (IGB: `wandb sync --include-offline <dir>` on the
   login node, one dir per rank; only rank 0 carries `val/` metrics). `wandb sync` resets
   API-written tags and names, so labeling runs AFTER the sync.
2. Run the view script for that split. It renames runs `<arm>_seed<k>[_rank<r>]`, sets
   `run.group = arm` so `/groups/<arm>` exists, writes config keys `arm`, `seed_`,
   `split`, `rank0`, and overwrites the saved view.
3. Reply with the saved view link on its own line, the group link of the arm in
   question on its own line, and one example rank-0 run link on its own line. Never a
   URL inside prose or a table cell.
4. If the reply quotes numbers, state the scoring rule (window mean over fixed epochs;
   a max over epochs is an upward-biased order statistic and is labeled as such) and
   mark a running cell as partial with its epoch count.

## Section order and panel rules

Sections are numbered by importance and the first row must answer the round's question.

1. **Headline metric across arms** (for interaction rounds `val/gene_interaction/Pearson`):
   validation alone, train and validation on ONE plot, train alone, then the matching MSE.
   The runset is grouped by `arm`, so each panel shows one mean line per arm with its band.
2. **The auxiliary target** (fitness when it is trained): the same four panels.
3. **Per perturbation order** when `per_order_metrics` is on: fitness by order on one
   plot, interaction by order with the validation triples on the same plot, record counts.
4. **Losses**: total, then each term as train and validation on one plot.
5. **Operator and gradient probe**: perturbed-CLS across-strain sd, gradient norms by term.
6. **Bookkeeping**: GPU peak, learning rate, pool sizes.

Panel conventions: x axis `epoch`, no smoothing, `columns=4`, titles in words (not the
metric key), a train-and-val panel lists `[train_key, val_key]` as one `LinePlot`.
Filters: the runset carries `split == <Q|R>` so arms from the other split never appear;
failed and crashed runs and known abandoned partials are left unlabeled and fall out.

## What not to do

- Do not hand-edit the view in the browser; the script is the record and overwrites it.
- Do not report the personal workspace link or the project link as "the view".
- Do not compare arms across splits in one view; Q and R are different questions.
- Do not add a metric to the view without adding it to the readout script's scoring.

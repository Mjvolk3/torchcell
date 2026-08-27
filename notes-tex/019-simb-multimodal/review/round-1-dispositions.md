# Round 1 review dispositions

Comments pulled with `python notes-tex/common/zotero_comments.py 019-simb-multimodal`
off `019-simb-multimodal_2026-08-26-20-48-47_a4b21822.pdf`. Twelve annotations, all
carrying written text. Keys are the stable Zotero handles.

Two defects were found in the tooling while doing this and are fixed in the same branch:
`zotero_comments.py` imported `DOCS_COLLECTION`, a name `zotero_publish.py` stopped
exporting when it was generalized, so the script had been failing at import; and it
assumed a two-level collection path instead of walking the repo-relative directory.

## Raised outside the annotations, in the message

| # | concern | disposition |
|---|---|---|
| M1 | "not sure it covers all of the data we have on wandb" | **Confirmed. Fixed for enumeration, partial for scoring, and the split is now stated in the document.** The account is **28 projects / 2,187 runs**; revision 1 read 8 projects / 396 runs. New `project_census.py` reads summaries only and therefore covers **all 28**, giving run counts, metric names and training-set size per project. Per-run *history*, which is what a best-score needs, now covers **13 of 28**: one request per run against a rate-limited API does not finish over 2,187 runs, so `pull_round_leaderboards.py` was made cached-per-project and resumable, and the rest are queued. `\cref{tab:coverage}` names exactly which projects are read and which are pending, and notes that a strand maximum can only move upward when they land. Two historical metric names were also missing (`val/per_gene/pearson_per_gene` for expression, `val/global/pearson_per_gene` for morphology), so the eleven oldest projects would have returned nothing even if listed. |
| M2 | "best pearson for betaxanthin is < 0.40 but pretty sure we used one for inference with > 0.40" | **Correct, fixed.** `betaxanthin_002` reached **0.4301** (top five 0.4301/0.4234/0.4216/0.4211/0.4095). It lives in `torchcell_020_betaxanthin` / `_v3`, which train on **4,235** strains; `_v4` trains on 3,698 because the pinned Merzbacher test split removes genes. The 0.372 I reported was `_v4` only. Both numbers now appear with their training-set size, and the selection-inflated floor (0.366) is stated. |
| M3 | "010 uses tmi from both kuzmin papers, nearly 400,000 data points" | **Correct, fixed.** `001_small_build.cql` selects `TmiKuzmin2018Dataset` **and** `TmiKuzmin2020Dataset`: 91,111 + 301,798 = **392,909**. The label-accounting table said 91,111. |
| M4 | "was best expression a 4 day run?" | **Yes.** `hx8pxdic` ran **91.4 h = 3.81 days** for 9,999 epochs at 32.9 s/epoch. `b50f93ju`: 47.9 h = 2.00 days, 4,106 epochs, 42.0 s/epoch. Now a table row, and it is the cost basis for the campaign plan. |

## Annotations

| # | key | p | concern | disposition |
|---|---|---|---|---|
| 1 | `7WIDCY7B` | 3 | the two betaxanthin rows: "difference is unclear" | **Fixed.** New `\subsection{Terms used throughout}` defines head / metabolome head / auxiliary head before first use. The summary now states outright that the two rows are not comparable (different training populations) and points at the within-grid paired table as the valid contrast. |
| 2 | `FS634P75` | 3 | "pretty sure all of the mses minimize then go back up... but nmse might not on best run... i think it drops back down again" | **Correct, and my claim was wrong.** Measured on both long runs: val MSE rises to ~epoch 700-1,400 then falls to a **late** minimum (9,175 on v9; 3,922 on v8) essentially at the Pearson peak. `nmse` dips under 1 at epoch ~213, peaks at 1.073, then falls back for the rest of the run **without returning below 1**. Section rewritten; new figure. |
| 3 | `NIZ3DA6G` | 3 | "Fixing checkpoint is important but it seems we have more model issues first" | **Agreed, reframed.** Best-by-`mse` would have picked within 500 epochs (v9) or 1 epoch (v8) of the right checkpoint; only the *quantile* loss misleads. Checkpointing is now stated as the narrower problem it is, and explicitly not what holds the strand at 0.24. |
| 4 | `KSTPZI3C` | 3 | pair term "gives hope, but how to actualize... scGPT unmasking doesn't accomplish this" | **Agreed, and now proved rather than asserted.** At $k=0$ the revealed set is empty, every encoded feature is exactly zero, and the forward pass is identical to the unconditioned model, so masked unmasking contributes nothing where the score is taken. New passage separates the two axes and names what *can* carry a pair term at $k=0$. |
| 5 | `SIRC4VMW` | 3 | "are you sure review the morph wandb... i thought we did" | **Re-checked against all eight morphology projects, and the claim holds.** `project_census.json`: morphology alone (morph_v2/v3/v5) is 3 projects / 183 runs, morphology with expression (expr_morph .. _v5) is 5 projects / 214 runs, and $n_{\text{train}} = 1{,}161$ is the **only value observed in any of the 397**. The same census shows expression's training set growing 1,125 to 1,253 and betaxanthin's spanning 3,694 to 4,235, so the constant is a finding rather than an artifact of how the census reads. Now a table instead of a sentence. |
| 6 | `GXWTV4IR` | 3 | "unclear to me how this pred is actually made" | **Fixed.** Four-step description added: join on ORF, log + z-score the 19 concentrations, ridge with inner-CV penalty, score out of fold over three shuffles. Stated plainly that no network is involved. |
| 7 | `Z5HHL8SD` | 3 | "very confusing now you are mixing in costanzo fitness?" | **Fixed.** New paragraph says why fitness is there (both screens respond to growth, so a shared growth term would manufacture the correlation), that it is a control rather than a third phenotype, and that it shrinks the gene set to 3,708 which is why both fits are reported on that same set. |
| 8 | `UQSHMACI` | 3 | "the most logical way to resolve this is the PINN-like GEM metabolism... motivates things well in fig 6" | **Adopted.** Promoted from a Tier-2 caution to the named mechanism for Fig 6, with the coverage numbers that bound it kept as the sizing constraint rather than as an objection. |
| 9 | `36STRGL3` | 4 | "Is this just dataset splits?" | **Answered with a measurement.** Repeated identical configurations in `betaxanthin_002` give a replicate $\sigma = 0.0302$; the observed spread is several times that. Three components separated (collapsed lr cells, genuine config effects, run-to-run noise) plus selection inflation of 0.064. New table. |
| 10 | `CYJ52HQ9` | 4 | "last part of fig 6 is getting the structure right" | **Adopted** in the Fig-6 recommendation. |
| 11 | `86D3C5BL` | 4 | run `hx8pxdic`: "nmse drops near mse minimum as val pearson rises, but never hits mse min" | **Confirmed and quantified.** See #2. Added the calibration identity: `nmse` $= 1 + s^2 - 2rs$, minimized at $s^\star = r$. The run sits at $s = 0.460$ against $r = 0.236$, so it is $1.95\times$ over-dispersed, and a post-hoc rescale by $r/s$ moves `nmse` from 1.010 to 0.944 with no change to any correlation. |
| 12 | `IAIWFRXU` | 4 | "best expression pearson is at 10,000 training epochs but it is like it has to fight mse" | **Confirmed, and given the mechanism.** The fight is with the *quantile* loss, not MSE. The `nmse` $\ge 1$ finding is the precise form of "fighting": the model is no better than the mean in squared error while ordering genes at $r = 0.236$. |

## Also added

- **Fig-3 campaign plan** with the measured per-epoch cost, since the next campaign is
  expression + morphology and the arm count is set by GPU-days rather than by curiosity.
- Per-project leaderboard rows, so a strand whose projects trained on different instance
  sets can no longer be quoted as one number.

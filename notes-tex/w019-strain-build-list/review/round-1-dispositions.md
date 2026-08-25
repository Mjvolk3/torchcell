# Review round 1 -- dispositions

Comments pulled from `w019-strain-build-list-clean_2026-08-24-21-18-44_c482d47d.pdf`
(published version 1) with `notes-tex/common/zotero_comments.py`. Keys are Zotero
annotation ids and are stable, so a follow-up round can refer to them directly.

**8 annotations, all carrying a written comment** -- 8 addressed, 0 needed no action.

Three of the eight were factual challenges rather than wording, and all three were
checked against the pinned CSVs before anything was rewritten:

- **The usable panel is eleven, not twelve.** The design panel holds `YIL174W`, which
  was never built, and does not hold SPH1 (`YLR313C`), which was
  (`experiments/010-kuzmin-tmi/results/inference_3/singles_table_panel12_k200_queried.csv`
  against `results/run4_measured_summary_singles.csv`). Setting `YLR312C-B` aside would
  leave ten, which is the count the comment reached for. [3]
- **SPH1 is in zero of the twenty triples.** Counted from
  `results/triple_design_rank_sampling_selection.csv` at `strategy == "capped"`: the
  twenty triples use eleven distinct genes and `YLR313C` is not among them.
  `YLR312C-B` is in six of them, which is what makes the swap a decision with a cost. [5]
- **The published single-mutant column is a standard error, and the reviewer is right
  that it is bootstrap-derived.** Costanzo 2016 labels it a standard deviation, but the
  SOM states that bootstrapped means across replicate screens were used, so it is the
  spread of an estimate: our own provenance note settled this on 2026.07.04
  (`notes/torchcell.datasets.scerevisiae.costanzo2016.noise-computation.md`) and the
  loader consumes it as `bootstrap_se`. The double-mutant column is a different
  statistic, a sample SD over four to eight colonies of one screen. [4]

| # | Key | p | Status | Disposition |
|---|---|---|---|---|
| 1 | `M8CNE2H2` | 1 | **done** | The suggested wording is right and the original sentence was vaguer than it needed to be. "No cross waits on another" now reads "no triple waits on one of the 25 new doubles being finished", with the reason the new doubles exist at all stated in the same breath: tau is computed from all three pairs and all three singles, so they are needed for the measurement, not for the construction. |
| 2 | `ZIGX3Y69` | 1 | **done** | Removed, and removed as a class rather than by hand. This document uses no `\external` or `\secondhand` flag, so the key was explaining two symbols a reader never met. `\tcdoc` now records a flag's use in the `.aux` and prints the provenance row only in a document that carries one, so it disappears here and stays in microbe-perturb-seq, which uses thirteen. Verified in both. |
| 3 | `S98SE634` | 2 | **done** | Confirmed, with the number. Twelve singles are built, eleven are panel genes carrying a prediction, and the twelfth is SPH1. A new paragraph says which gene is which way round and adds the second half of the arithmetic: `YLR312C-B` is an SGD "ORF, Merged" feature with no discrete protein and zero trigenic training records, SPH1 was the recommended replacement and the swap was never made, and it sits in six of the twenty triples. Setting it aside gives the ten the comment estimated. |
| 4 | `F6WPDH27` | 2 | **done** | Yes, and the distinction is now a Terms entry rather than a column header doing the work alone. Their single-mutant column is a bootstrap SE and is used as one; ours resamples three replicate plates against their seventeen screens, so theirs is tighter for the same underlying spread and a strain-by-strain comparison of the two is not like for like. Their double-mutant column is a sample SD over four to eight colonies of one screen, which the authors warn underestimates, and is comparable in kind to our plate SD instead. Both table captions now name the statistic. |
| 5 | `YPW8YZJK` | 2 | **done** | Confirmed against the selection file, and the sentence now says the narrower thing: SPH1 is in none of the twenty triples in this build list, not that it never appeared in any triple. It was never a node in the prediction panel, which is why no triple reaches it. |
| 6 | `S7FCKMM5` | 2 | **done** | Cut. The clause said nothing the sentence did not already say, and "at no cost in construction work" is the wrong register for a sheet handed to someone doing the bench work. Now: the rest are re-measured alongside them, which gives a second reading on each of those pairs. |
| 7 | `R2NIE93S` | 2 | **done** | Rewritten in bench terms: "One designed double failed to construct: no colonies after transformation." The body says the transformation gave zero colonies and that no attempt date and no retry are on record, which keeps the account of what happened separate from what was written down at the time. |
| 8 | `QPDPWDUQ` | 4 | **done** | Added, and kept to four sentences as asked. It gives the shape of the assay rather than the layout that will be run; every strain fits on one plate and several plates can be run; how many biological replicates to spread across them is the open question, because each plate is scored against its own wild-type wells and a plate whose wild type sits low shifts every fitness on it, as run-3 P2 did. The old "these counts are approximate" paragraph is folded into it rather than left beside it. |

## What changed outside the document

- `notes-tex/common/tcdoc.sty` -- the provenance key is now detected rather than
  declared [2]. The switch is written to the `.aux` with `\global`, without which
  `\newif`'s local assignment is undone before page 1 is typeset and the key silently
  never returns. That failure was hit and fixed here, not reasoned about: the key
  vanished from microbe-perturb-seq's build until the prefix was added.
- `experiments/W019-echo-crispr-array/scripts/build_list_tables.py` -- both table
  captions name their statistic [4]. Tables are regenerated from the script, never
  hand-edited.

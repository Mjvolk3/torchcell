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
| 4 | `F6WPDH27` | 2 | **done** | Yes on both counts, and the fix belongs in the table, which is where the comment was pointing. The first pass put the answer in the caption and Terms and left the headers reading `boot SE` against `SE`, so the table went on asserting a difference in kind that does not exist. Both columns are now headed **`boot SE`** with a unit row underneath, `over 3 plates` against `over 17 screens`, so the table says what is the same and what differs without the caption. The doubles column is renamed **`colony SD`**: it is a sample SD over the four to eight colonies of one screen, a spread rather than an error on a mean, so it compares to our plate SD and not to either boot SE. `boot SE`, `colony SD` and `plate SD` are all Terms entries. Chain for the seventeen: the panel's Costanzo records are the `_sn` strain ids, `sn` is the NatMX query strain (`costanzo2016.py:213`), and the SI puts query-strain SMF at 17 replicate control screens with colonies averaged within a screen before resampling (`costanzo2016.py:64-74`, verbatim quote). |
| 5 | `YPW8YZJK` | 2 | **done** | Confirmed against the selection file, and the sentence now says the narrower thing: SPH1 is in none of the twenty triples in this build list, not that it never appeared in any triple. It was never a node in the prediction panel, which is why no triple reaches it. |
| 6 | `S7FCKMM5` | 2 | **done** | Cut, in both places it occurred. The clause said nothing the sentence did not already say, and "at no cost in construction work" is the wrong register for a sheet handed to someone doing the bench work. The first pass fixed only the sentence the highlight sat on and left the same phrase in the Table 5 caption; both now read "which gives a second reading on each of those pairs". |
| 7 | `R2NIE93S` | 2 | **done** | Rewritten in bench terms: "One designed double failed to construct: no colonies after transformation." The body says the transformation gave zero colonies and that no attempt date and no retry are on record, which keeps the account of what happened separate from what was written down at the time. |
| 8 | `QPDPWDUQ` | 4 | **done** | Added, and kept to four sentences as asked. It gives the shape of the assay rather than the layout that will be run; every strain fits on one plate and several plates can be run; how many biological replicates to spread across them is the open question, because each plate is scored against its own wild-type wells and a plate whose wild type sits low shifts every fitness on it, as run-3 P2 did. The old "these counts are approximate" paragraph is folded into it rather than left beside it. |

## Verification, and why it is listed

Version 2 was published with [4] and [6] only half done: [4] was answered in the caption
while the table kept the header the comment was pointing at, and [6] was cut from the
sentence but left standing in the Table 5 caption. Both are fixed in version 3, and the
close-out is now a grep of the rendered text per comment rather than a reading: nine
strings that must appear and four that must not, run against `main-clean.pdf` before the
publish rather than after it.

## What changed outside the document

- `notes-tex/common/tcdoc.sty` -- the provenance key is now detected rather than
  declared [2]. The switch is written to the `.aux` with `\global`, without which
  `\newif`'s local assignment is undone before page 1 is typeset and the key silently
  never returns. That failure was hit and fixed here, not reasoned about: the key
  vanished from microbe-perturb-seq's build until the prefix was added.
- `experiments/W019-echo-crispr-array/scripts/build_list_tables.py` -- both table
  captions name their statistic [4]. Tables are regenerated from the script, never
  hand-edited.

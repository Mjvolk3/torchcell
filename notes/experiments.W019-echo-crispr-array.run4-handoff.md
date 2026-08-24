---
id: 0o4s4047lv7msxnhkx6g2gf
title: run4-handoff
desc: ''
updated: 1786592282957
created: 1786592282957
---

## 2026.08.12 - Handoff: state, open questions, and how to check the work

Written to hand this to a fresh session. Everything below is either **MEASURED** (with the
script that produced it) or explicitly labelled a hypothesis.

### Where the work lives

- Branch `chore/w019-rename`, worktree `~/Documents/projects/torchcell.worktrees/chore/w019-rename`
- **15 commits, NEVER PUSHED, no PR.** Base is `origin/main` @ `5786a212`, clean fast-forward.
- Document: `paper/notes/experiments.010-kuzmin-tmi.scripts.construction_validation_doubles.pdf`
  (15 pp). Build: `make -C paper/notes`. Sources in `paper/notes/sections/construction-validation-doubles/`.
- Scripts: `experiments/W019-echo-crispr-array/scripts/`
  - `run4_doubles_48h.py` -- segmentation -> fitness -> eps -> figures -> the LaTeX table
  - `run4_wt_reference_diagnostic.py` -- plate-median re-score (DIAGNOSTIC, not a correction)
  - `incubation_times.py` -- per-plate incubation from Echo report + EXIF

### What was done (all committed)

1. Landed run 3 (PR #219) -> `origin/main` `5786a212`.
2. Renamed `019-echo-crispr-array` -> `W019-echo-crispr-array` (351 renames, 281 refs);
   verified by re-running run 3 bit-identical.
3. Run 4 data committed with `PROVENANCE.md` + sha256 manifest.
4. Built `paper/notes/` -- LaTeX lab notes inheriting the manuscript's editing style by
   symlink/`\input` from `../nature-biotech/`.
5. Scored run 4; wrote the run-4 sections.

### MEASURED results

| quantity | value | source |
|---|---|---|
| run 4 QC | P1/P2/P3 pass the WT-CV gate; occupancy 320/345/369; 6/6 blanks empty; WT CV 0.153/0.158/0.122. All three FAIL the later frac-above-WT gate (0.40/0.32/0.24 of deletions above WT, limit 0.20) | `run4_plate_qc.csv`, `run4_strain_scores_by_plate.csv` |
| singles vs Costanzo SMF | r=0.706, p=0.015, n=11 | `run4_singles_vs_reference.csv` |
| run 4 vs run 3 singles | r=0.685, p=0.020, n=11; run 4 is 1.26x higher (median ratio) | `run4_doubles_48h.py` |
| doubles vs Costanzo DMF | r=0.255, p=0.42, n=12 | `run4_doubles_vs_reference.csv` |
| eps vs Costanzo eps | r=-0.245, p=0.44, n=12 | same |
| corr(eps, f_a*f_b) | **-0.926**, p=5e-6 | same |
| f_a*f_b > 1 | 10 of 13 (max 1.712) | same |
| eps negative | 13 of 13 | same |
| double/single ratio | 0.758 (multiplicative model wants ~1.07) | same |
| plate-median re-score | every diagnostic unchanged or WORSE | `run4_wt_diagnostic_*.csv` |

### THE OPEN PROBLEM

eps is **not reportable**. Two separable effects:

1. **Per-round reference scale.** Singles score above the on-plate WT (8/12), run 4 sits
   1.26x above run 3. Plausible mechanism: the WT is picked once per round, so the whole
   round's denominator rests on one colony. Consistent, NOT proven.
2. **Doubles-vs-singles gap.** double/single = 0.758 where a multiplicative model wants
   ~1.07. **No reference choice can produce this** -- a common factor cancels from a
   strain-to-strain ratio (demonstrated in the note with a k-sweep table). This is what
   corrupts eps, and it has NO mechanism yet.

REJECTED hypotheses (do not re-propose):

- WT wells specifically -- tested by plate-median swap, made it worse.
- Two selection cassettes on doubles -- WRONG, these are full ORF deletions.

### Replicate structure -- the thing that most limits us

**Every strain is plated from ONE picked colony**, grown and diluted into all 14 wells and
all 3 plates. So our replicates are TECHNICAL. The across-plate SE therefore does NOT
contain colony-picking variance and is an under-estimate w.r.t. the biological unit; every
p-value in the note inherits that.

Contrast, sourced:

- **Costanzo SMF** = average of ~350 replicate control screens (array) / ~17 (query).
  The "4 colonies" applies to the double-mutant screens, NOT to SMF.
- **Kuzmin 2018 replicate reproducibility** -- digenic eps and raw trigenic scores
  **r = 0.90--0.91**; adjusted trigenic tau **r = 0.74--0.81**. Verbatim, from the SI
  section "Evaluation of reproducibility of genetic interactions": ``The screen noise was
  similar for double mutants (Fig. S5B left) compared to raw triple mutant scores (Fig. S5B
  middle) with the correlation between independent replicates of 0.9-0.91. However, the
  adjusted trigenic interaction scores showed more variability with the correlation
  coefficient between replicates decreasing to 0.74-0.81 (Fig. S5B right).'' Fig. S5A is
  triple-mutant FITNESS replicate correlation for one representative screen, with an inset
  distribution over n = 172 screens; the SI text gives no single r for it.
  Source: `$DATA_ROOT/torchcell-library/kuzminSystematicAnalysisComplex2018/si/si1.md`,
  sha256 `2ec80d05d823976e12add17699ad759bcd983768d2f53e7eb6c0185b963b8291`.
  **Corrected 2026.08.23.** This entry previously read "digenic eps r = 0.88, raw trigenic
  0.88, adjusted trigenic tau 0.59", read off Fig. S5B rather than from the SI text. The
  adjusted-tau figure was the material error: 0.59 understates their reproducibility by
  0.15--0.22, which makes our own tau precision look closer to theirs than it is.
- **Their replicates are NOT clonal.** SGA regenerates the genotype each replicate: query
  lawn -> mate to array -> sporulate -> haploid selection. Final colonies of two replicates
  are independent meiotic segregants. So r = 0.90--0.91 is across independent
  CONSTRUCTIONS -- a harder bar than our technical replication, and they still hit it.

Discriminating experiment: **independent pickings** -- 2+ colonies per strain carried as
separate lineages through the round.

### Normalization -- how it is actually applied

- **WT is taken WITHIN plate, never pooled.** `torchcell/sga/score.py:55` selects
  `strain == cfg.wt_name` from that plate's own table; `relative_fitness = median(norm|strain)
  / median(norm|WT)` on the same plate. Bootstrapping then averages the per-plate relative
  fitnesses across plates.
- **There is NO cross-batch correction anywhere.** The across-plate bootstrap MEASURES batch
  spread; nothing removes it. Cross-round comparisons are raw.

### Incubation (recovered, `incubation_times.csv`)

| round | elapsed (Echo dispense -> photo) | official |
|---|---|---|
| run2 P1/P2 | 43:52 / 43:48 | -- |
| run3 P1/P2/P3 | 48:09 / 48:05 / 48:01 | -- |
| run4 P1/P2/P3 | 48:40 / 48:36 / 48:32 | 48:12 |

Two things follow: the rounds are NOT at matched incubation (run 4 ~30 min longer than
run 3), and the official 48:12 implies t0 = incubator entry, not Echo dispense.
**Runs 2 and 3 official times are still needed from the bench** -- left blank deliberately.

### TO CHECK IN A NEW SESSION

1. `make -C paper/notes` -- should be 15 pp, zero errors.
2. Re-run `run4_doubles_48h.py` (GPU) and confirm the MEASURED table above reproduces.
3. Verify WT-within-plate claim at `torchcell/sga/score.py:55`.
4. Confirm no `strong` column and that bold in Table 4 = Costanzo intermediate tier.
5. Confirm Fig 7/8 y-error is `dmf_se`, NOT `eps_se` (this was a real bug, fixed).
6. Push + PR + `/enqueue-merge` -- 15 commits are local-only.

### NOT DONE

- The 12-panel assay note (`notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md`)
  is NOT ported to LaTeX. This was the larger deliverable.
- No cross-batch correction / mixed model pooling runs 2-4 as batches.
- Costanzo CBF1 strain ambiguity (`sn154` 0.5900 vs `dma2646` 0.9230, picked by dict
  collision) is documented but NOT resolved.

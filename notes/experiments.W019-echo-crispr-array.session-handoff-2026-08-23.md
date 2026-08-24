---
id: 7vr15hvltdg7z66q55hxtmq
title: session-handoff-2026-08-23
desc: ''
updated: 1787526533597
created: 1787526533597
---

## 2026.08.23 - Session handoff: capped triple design, for independent review

Everything below is either MEASURED (with the producing script) or explicitly labelled a
hypothesis. **All paths absolute.** Worktree: `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename`, branch `chore/w019-rename`.

### What this session did

1. Added a wild-type crosshair at (1,1) to six fitness-vs-fitness scatter panels.
2. Corrected the target-triple basis from **10 genes / 31 targets** to **11 genes / 39
   targets** -- `YLR104W` (LCL2) is both built and a prediction node; the frozen `TEN` set
   predated run 4's decision to keep LCL2 rather than swap to LCL1.
3. Established that the prediction model was trained on **trigenic data only**, and that
   two panel genes have **zero** trigenic training records.
4. Derived the model's calibration from logged val metrics alone (no re-run).
5. Built and compared six selection strategies; settled on **`capped`** (cap 6, 20 triples).
6. Produced a bench-facing build list of **45 strains: 25 doubles + 20 triples**.

### Strain inventory -- what exists today

**12 singles built** (`s1`-`s12`), all present:

| id | ORF | common | id | ORF | common |
|---|---|---|---|---|---|
| s1 | YJR060W | CBF1 | s7 | YLR104W | LCL2 |
| s2 | YPL081W | RPS9A | s8 | YLL012W | YEH1 |
| s3 | YDR057W | YOS9 | s9 | YER079W | -- |
| s4 | YPL046C | ELC1 | s10 | YKL033W-A | -- |
| s5 | YBR203W | COS111 | s11 | YLR313C | SPH1 |
| s6 | YGL087C | MMS2 | s12 | YLR312C-B | -- |

**11 of the 12 are prediction nodes.** `YLR313C` (SPH1) was built but was never a node in
the inference panel, so it appears in no predicted triple and contributes nothing.

**13 doubles built** (`d1`-`d13`):

| id | pair | id | pair |
|---|---|---|---|
| d1 | YPL081W + YDR057W | d8 | YPL046C + YJR060W |
| d2 | YPL081W + YPL046C | d9 | YPL046C + YBR203W |
| d3 | YPL081W + YER079W | d10 | YLL012W + YDR057W |
| d4 | YPL081W + YLR312C-B | d11 | YLL012W + YPL046C |
| d5 | YDR057W + YGL087C | d12 | YER079W + YJR060W |
| d6 | YDR057W + YER079W | d13 | YER079W + YLR312C-B |
| d7 | YDR057W + YLR312C-B | | |

**0 triples built.** Run 4 carried singles and doubles only.

### Could not be built

**`YKL033W-A x YJR060W` (CBF1)** -- the 14th designed double. Reported failed at the
2026-08-11 handover; **the attempt date was never recorded**. It blocks **0** of the 39
targets (no top-k triple contains that pair), so it costs no trigenic score. Excluded from
every candidate set. *Do not re-propose until it is known whether the failure repeats.*

**Not a failure, but a gap:** `YLR104W` (LCL2) has **zero doubles** in the current panel,
because it sat outside the `TEN` set the original set-cover ran on. That is why three
targets (ranks 26, 32, 37) have no existing parent double. **The set-cover guarantee itself
is intact: 31 of 31 within-TEN targets have a built parent.**

### The finding that drove the design

The training query pulls **only** `TmiKuzmin2018Dataset` + `TmiKuzmin2020Dataset` -- no
SMF, DMF, DMI or essentiality. Kuzmin record counts:

| tier | genes | trigenic records |
|---|---|---|
| well supported | YLL012W, YBR203W, YPL046C, YJR060W, YGL087C | 319--1129 |
| thin | YPL081W, YDR057W, YLR104W, YKL033W-A | 8--10 |
| **none** | **YER079W, YLR312C-B** | **0** |

Both zero-data genes have Costanzo data (YER079W 4 SMF; YLR312C-B 2 SMF, ~3,544 digenic
partners) -- it was simply never in this model's training set. They then occupy **ranks
1--10** of the 39 targets, and prediction tracks *absence* of data (Spearman 0.634,
p = 1.5e-05).

### The design chosen: `capped`

| | value |
|---|---|
| triples | 20 |
| new doubles | 25 |
| **to construct** | **45** |
| on plate (incl. WT) | 65, 5 wells/strain |
| triples touching a zero-data gene | **6 of 20** (the cap) |
| doubles touching one | 7 of 25 |
| serial dependencies | **none** |
| genes covered | 11 of 11, range 2--8 |
| mean predicted interaction | 0.5300 |

If YER079W and YLR312C-B are later dropped, **14 triples and 18 doubles survive**.

### Open, unresolved -- please check these

1. **`YLR312C-B` in or out.** SGD "ORF, Merged", no discrete protein, zero trigenic
   training data, 0/10 significant in-panel digenics. Swap to SPH1 recommended in
   `[[experiments.010-kuzmin-tmi.inference-dataset-3]]` and never actioned.
2. **tau is not callable at this precision.** SE(tau) ~ sqrt(7)*s = 2.65s; at 2 picks /
   3 plates s ~ 0.121 so the smallest callable |tau| is **0.63**. Kuzmin's thresholds are
   0.08 / 0.12 / 0.20 and their median SE(tau) is 0.031. **0 of 39** targets clear our
   floor. Needs ~9x more replication.
3. **Calibration assumes an unbiased mean.** The 0.43 slope assumes mu_pred ~ mu_y. One
   read of the existing inference parquet would settle it.
4. **`run4-handoff.md` contains WRONG Kuzmin reproducibility numbers** -- it says digenic
   eps r=0.88 / adjusted trigenic tau 0.59, read off a figure. The SI text says
   **0.90--0.91** and **0.74--0.81**. Not yet corrected.
5. **Pick-replication round vs triple round compete for the same plate.** The replication
   round is 26 strains / 57 tubes; this triple round is 65 strains / 65 tubes at one pick.
   Recommended order: replication first, construct these 45 during it.
6. **Stale artefacts on the superseded 31-target basis** still on disk:
   `next_doubles_selection.py`, `triple_coverage_from_built_doubles.py` and their outputs.

### Files -- all absolute

**Notes**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/experiments.W019-echo-crispr-array.build-list.md` -- bench-facing, D01--D25 + T01--T20
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/experiments.W019-echo-crispr-array.next-strains-to-construct.md` -- rationale
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/experiments.W019-echo-crispr-array.run4-handoff.md` -- prior round (**has the wrong Kuzmin numbers**)
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/experiments.010-kuzmin-tmi.inference-dataset-3.md` -- YLR312C-B swap recommendation
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md`

**Scripts (all UNTRACKED)**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py` -- **the live one**
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/next_doubles_selection.py` -- superseded (31-target basis)
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/triple_coverage_from_built_doubles.py` -- superseded

**Scripts (MODIFIED, crosshair only)**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/run4_doubles_48h.py`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/run3_48h_3rand.py`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/run_plate5_volume.py`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/scripts/compare_modalities.py`

**Results**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/results/triple_design_rank_sampling_summary.csv`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/results/triple_design_rank_sampling_selection.csv`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/results/triple_design_rank_sampling_gene_frequency.csv`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/results/triple_build_construction_check.csv`

**Figures**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/assets/images/W019-echo-crispr-array/triple_design_rank_sampling.png`
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/notes/assets/images/W019-echo-crispr-array/triple_design_gene_frequency.png`

**Inputs (read-only, pinned)**

- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/010-kuzmin-tmi/results/inference_3/top_k_constructible_panel12_k200.csv` -- **the target set**
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/010-kuzmin-tmi/queries/001_small_build.cql` -- training query
- `/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/W019-echo-crispr-array/data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv`
- `/scratch/projects/torchcell-scratch/data/torchcell/tmi_kuzmin{2018,2020}/processed/lmdb`

**Do NOT use** (earlier inference, different gene panel):
`/home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename/experiments/010-kuzmin-tmi/results/constructible_triples_panel12_*.parquet`, and
`.../inference_3/triples_table_panel12_k200.csv` (122 rows -- all constructible, not the
top-k the set-cover ran on).

### Reproduce

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/chore/w019-rename
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py
```

Deterministic -- candidate iteration is sorted, so repeated runs are byte-identical.

### Related

- [[experiments.W019-echo-crispr-array.build-list]]
- [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [[experiments.W019-echo-crispr-array.run4-handoff]]

## 2026.08.23 - Review pass: audited, two items closed, one error found

A second session audited everything above with
`experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py`
([[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]), which
re-derives the inventory, basis and selection from the pinned inputs without importing the
design script. **52 checks, 0 fail.** Report:
`experiments/W019-echo-crispr-array/results/verify_triple_build_list_checks.csv`.

### Confirmed

- **Inventory.** 12 singles, 13 doubles, 0 triples; 11 of the 12 singles are prediction
  nodes and YLR313C is the exception; YLR104W has zero built doubles; the blocked pair is
  neither built nor proposed.
- **Tau-computability, the point of the round.** After the 45 strains are built all 20
  triples have all three doubles and all three singles on the plate, and the 65 plate
  strains are exactly the closure of that requirement -- nothing extra, nothing missing.
- **Basis and cover.** 39 in-basis targets; ranks 1-10 all touch a zero-data gene; the
  no-parent targets are ranks 26, 32, 37 and all contain YLR104W; the set-cover guarantee
  of 31 of 31 within-TEN targets holds.
- **Both note tables, row by row** -- every D-row, T-row, build-from parent, third single,
  route label, star, rank, prediction, new-double count and serves-triples list.
- **Kuzmin counts read from the LMDBs**, not quoted: the trigenic column matches exactly,
  and YER079W and YLR312C-B are absent from both `tmi_kuzmin{2018,2020}` and
  `dmi_kuzmin{2018,2020}`.
- **Calibration arithmetic.** sigma_pred re-solved from val r + val RMSE + sigma_y
  reproduces 0.0563 / 0.0572 / 0.0553; the shrunk range 0.148-0.278 and "15 of 39 above
  0.20" reproduce at slope 0.43 with the intercept. The val metrics themselves were not
  re-derived from W&B.
- **The design script is deterministic** -- a re-run leaves all four CSVs byte-identical.

### One error found and fixed

**The drop-the-two contingency was wrong in both notes.** They said 18 new doubles survive.
18 of the 25 contain neither flagged gene, but `YLR104W + YPL081W` serves only rank 3
(`YLR104W + YLR312C-B + YPL081W`). So **17** feed a surviving triple and the core is
**31** strains, not the 32 stated or the 34 the rationale note also claimed in the same
paragraph. Both notes corrected; the verifier now asserts 14 / 17 / 18 and names the
orphan.

### Open items, updated

- **Item 4 (wrong Kuzmin numbers) is CLOSED.** `run4-handoff.md` now carries the SI text
  verbatim: digenic eps and raw trigenic **r = 0.90-0.91**, adjusted trigenic tau
  **r = 0.74-0.81**, sourced to
  `kuzminSystematicAnalysisComplex2018/si/si1.md` sha256 `2ec80d05...b963b8291`, with the
  superseded 0.88 / 0.59 recorded as corrected.
- **Item 6 (stale artefacts) is CLOSED.** The two superseded scripts, their nine result
  CSVs, their six images and their two note stubs -- 19 files on the 10-gene / 31-target
  basis -- were moved out of the repo into the graveyard
  `/scratch/projects/torchcell-deprecated/2026-08-24_104110__*/`, each with its own
  `DEPRECATION.txt` recording the original absolute path, git HEAD and the reason. Moved,
  not deleted; purging the graveyard is a manual chore. The BUILDABLE-vs-SCORABLE
  distinction those scripts introduced survives in
  [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]] and in the
  "Why all 65 and not just the 20 triples" section of
  [[experiments.W019-echo-crispr-array.build-list]], so nothing of substance was lost.
  The paths cited under "Files" and item 6 in the section above are therefore historical.
- **Items 1, 2, 3, 5 stand as written** -- YLR312C-B in or out, tau not callable at this
  precision, the unbiased-mean assumption behind the 0.43 slope, and the plate contention
  with the replication round.

### Two things to read with care

- **The digenic column is not training data.** The rationale note's per-gene table sits
  directly under "the model was trained on trigenic only", and the digenic column comes
  from `dmi_kuzmin{2018,2020}`, which the query excludes. The table now says so. It is
  there to show that a gene's absence is absence from Kuzmin altogether.
- **The 6-plate floor of 0.55 is not reproducible from the note.** Scaling s by
  1/sqrt(2) from 3 to 6 plates gives 0.44, not 0.55. A nested model in which only the
  plate term shrinks would land near 0.55, which is the more careful assumption, but the
  note does not show which variance decomposition was used. Worth one line when item 2 is
  next touched. Separately, the "~9x more replication" figure reaches per-strain s ~0.04,
  a callable floor near 0.21 -- it does not reach Kuzmin's SE(tau) of 0.031, which the
  neighbouring sentence could be read as implying.

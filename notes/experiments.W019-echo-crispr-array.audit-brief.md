---
id: fb8tx8hr1giz5z1pp54j7zp
title: audit-brief
desc: ''
updated: 1787598709303
created: 1787598709303
---

## 2026.08.24 - Audit brief: first trigenic round design and the run-4 record

Scope for an independent audit. Sections 1 to 3 state what exists and why. Section 4
separates what is machine-verified from what is not. Section 5 is the audit charter.

Repository state: `main` at `dab3874c`. Working tree clean.

### 1. Objective

Select and document the first set of triple-knockout strains for the W019 Echo/CRISPR
array, such that a trigenic interaction score can be computed for each. The trigenic
interaction is

$$\tau_{abc} = f_{abc} - f_{ab}f_c - f_{ac}f_b - f_{bc}f_a + 2 f_a f_b f_c$$

which consumes seven measured fitness values. A selected triple therefore yields no
score unless all three constituent doubles and all three constituent singles are
measured on the same plate. Closure over that requirement, not the count of triples, is
the design constraint.

### 2. What was done

| # | Action | Rationale |
|:--|:-------|:----------|
| 1 | Corrected the candidate basis from 10 genes / 31 targets to 11 genes / 39 targets | `YLR104W` (LCL2) is both a constructed single and a panel-12 prediction node. The frozen `TEN` set predated run 4's decision to retain LCL2 rather than substitute LCL1 (`YPL056C`), so 8 targets were excluded in error, one at prediction 0.7012 |
| 2 | Established that the prediction model was trained on trigenic records only | `experiments/010-kuzmin-tmi/queries/001_small_build.cql` selects `TmiKuzmin2018Dataset` and `TmiKuzmin2020Dataset`; all other dataset blocks are commented out. Two panel genes (`YER079W`, `YLR312C-B`) have zero trigenic records, and predicted interaction strength correlates with absence of training data (Spearman 0.634, p = 1.5e-05) |
| 3 | Compared six selection strategies and adopted `capped` | `capped` is the only strategy that constrains exposure to the two zero-trigenic-data genes to a set value (6 of 20) while covering all eleven panel genes. `rank` places 17 of 20 on those genes, `balanced` 15 |
| 4 | Produced a bench build list of 45 strains (25 doubles, 20 triples), 65 on plate | Closure of the tau requirement over the 20 selected triples: 11 singles, 33 doubles (8 existing, 25 new), 20 triples, WT |
| 5 | Wrote an independent verifier | The design script selects; the verifier checks. It re-derives inventory, basis and selection consequences from the pinned inputs without importing the design script, and asserts every number stated in the two bench notes |
| 6 | Added the run-4 measured record and a gap table | The bench sheet previously stated which strains exist but not what they scored, and did not distinguish "measured" from "has a published value to compare against" |
| 7 | Corrected three documented errors | See section 3 |
| 8 | Retired the superseded 10-gene artifacts | 19 files moved to `/scratch/projects/torchcell-deprecated/2026-08-24_104110__*/` with per-file provenance manifests. Not deleted |

### 3. Errors found and corrected

| Error | Correction | Evidence |
|:------|:-----------|:---------|
| Both bench notes stated 18 new doubles survive if `YER079W` and `YLR312C-B` are dropped, and the rationale note also stated a 34-strain core in the same paragraph | 17 new doubles are still needed; the core is 31 strains. 18 of the 25 contain neither flagged gene, but `YLR104W + YPL081W` serves only rank 3 and is orphaned by the drop | Verifier group `contingency`, 5 checks |
| `run4-handoff.md` reported Kuzmin replicate reproducibility as digenic eps r = 0.88 and adjusted trigenic tau r = 0.59, read from Fig. S5B | Digenic eps and raw trigenic r = 0.90-0.91; adjusted trigenic tau r = 0.74-0.81, quoted verbatim from the SI text | `kuzminSystematicAnalysisComplex2018/si/si1.md`, sha256 `2ec80d05d823976e12add17699ad759bcd983768d2f53e7eb6c0185b963b8291`, section "Evaluation of reproducibility of genetic interactions" |
| The per-gene digenic column sat under a statement that the model was trained on trigenic data only, implying it was training data | Labeled as not training data. It derives from `dmi_kuzmin{2018,2020}`, which the training query excludes; it is shown to establish that a gene's absence is absence from Kuzmin altogether | Verifier group `kuzmin`, counts read from the LMDBs |

The adjusted-tau figure was the material error of the three: 0.59 understates the
published reproducibility by 0.15 to 0.22, which makes our own tau precision appear
closer to theirs than it is.

### 4. Verification status

**Machine-verified.** `experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py`,
61 checks, 0 fail, exit code 1 on any failure. Report:
`experiments/W019-echo-crispr-array/results/verify_triple_build_list_checks.csv`.

| Group | n | Covers |
|:------|--:|:-------|
| `inventory` | 8 | 12 singles, 13 doubles, 0 triples built; `YLR313C` is the one non-node single; `YLR104W` has no built doubles; the blocked pair is absent |
| `basis` | 7 | 39 in-basis targets; ranks 1-11 all touch a zero-data gene (corrected from 1-10 by the audit); no-parent targets are ranks 26, 32, 37; set-cover intact at 31 of 31 within-TEN |
| `selection` | 5 | 20 distinct in-basis triples; cap of 6 held; every triple has a built parent; 15 single-route |
| `doubles` | 5 | 25 new, 8 existing reused, 33 distinct on plate, all parents built |
| `tau` | 5 | Closure: all 20 triples fully scorable after the build, and the 65 plate strains are exactly the closure, nothing extra or missing |
| `contingency` | 5 | 14 triples and 17 needed doubles survive dropping the two genes; names the orphaned double |
| `notes` | 15 | Every row of both bench tables against the computed selection: IDs, gene sets, build-from parents, third singles, routes counts, flags, ranks, predictions, new-double counts, serves-triples lists, gene participation |
| `measured` | 8 | Every run-4 fitness, bootstrap SE, across-plate SD, published reference and tier in the note against the generated CSVs; which single and which double lack a published value |
| `kuzmin` | 3 | Per-gene trigenic and digenic record counts read directly from `tmi_kuzmin{2018,2020}` (392,909 records) and `dmi_kuzmin{2018,2020}` (1,043,196 records) |

**Reproducibility.** Two consecutive full regenerations of both scripts leave every
output CSV byte-identical. The two SVG figures differ between runs in the embedded
`<dc:date>` and in randomized clip-path element IDs only; vector content is identical.
This is a matplotlib default (`svg.hashsalt` unset) and is a genuine, fixable
reproducibility gap.

**Not verified.** The following are stated in the notes and are outside the verifier.
This list is the audit's primary work queue.

| Claim | Location | Status |
|:------|:---------|:-------|
| Validation metrics per checkpoint (r = 0.443 / 0.431 / 0.457; RMSE = 0.0580 / 0.0591 / 0.0567) and W&B run ids `lzs9pcj3`, `yv4r30bi`, `c7671wgj` | rationale note, calibration table | Not re-derived from W&B |
| Label statistics sigma_y = 0.0535, mean = -0.048 | rationale note | Not re-derived |
| sigma_pred column and the 0.43 calibration slope | rationale note | Re-derived by hand this session from the stated r, RMSE and sigma_y; not in the verifier |
| Shrunk prediction range 0.148-0.278 and "above very stringent for 15 of 39" | rationale note | Re-derived by hand this session; not in the verifier |
| Mean prediction by zero-data-gene count (0.4531 / 0.5300 / 0.6676) and Spearman 0.634, p = 1.5e-05 | rationale note | Re-derived by hand this session; not in the verifier |
| Per-strain s = 0.121 at 2 picks / 3 plates | rationale note | Not traced to a producing artifact |
| Six-plate floor of 0.55 | rationale note | Reproducible, but from an undisclosed and unsupported model: with a P-independent floor of 0.090 and a plate term equal to run 3's failed-plate-inflated SD of 0.140, 6 plates give 0.553. Under the components run 4 supports the floor is 0.446, and 25 of 39 raw predictions clear it. Corrected (A2) |
| "roughly 9x more replication" | rationale note | Reaches per-strain s ~0.04 and a callable floor near 0.21. The adjacent sentence cites Kuzmin's median SE(tau) of 0.031, which 9x does not reach. Ambiguous as written |
| Kuzmin median SE(tau) = 0.031, back-solved from tau and p | rationale note | Not re-derived this session |
| Costanzo coverage for the two zero-data genes (`YER079W` 4 SMF; `YLR312C-B` 2 SMF, ~3,544 digenic partners) | rationale note | Not verified |
| W&B group `compute-3-3-2059267_f83395...` is a failed run at val r = -0.03 | rationale note | Not verified |
| `top_k_constructible_panel12_k200.csv` (the prediction source) | pinned input | Consumed as given. Generation and provenance not audited |
| `run4_strain_bootstrap.csv` (all measured fitness) | pinned input | Consumed as given. Not re-derived from segmentation |
| `reference_smf_12panel.csv` and `construction_validation_doubles.csv` | pinned inputs | Read, generation not re-run |
| Run-4 epsilon | excluded by design | Documented as not reportable (double/single ratio 0.758 against a multiplicative expectation of ~1.07). Deliberately absent from all tables |

### 5. Open items carried forward

1. **`YLR312C-B` in or out.** SGD "ORF, Merged" (web-sourced, yeastgenome.org, not in any
   mirrored artifact); no discrete protein; zero trigenic training data; **0 of its 10
   in-panel digenics is significant, and all 10 are measured**, so this is measured and
   null rather than unmeasured. The "9" recorded below in section 3 was a regression: it
   re-derived the denominator from the superseded TEN gene set, which is the very basis
   action 1 corrects. Over the eleven-gene basis the gene has 10 partners. Substitution with SPH1
   (`YLR313C`) was recommended in [[experiments.010-kuzmin-tmi.inference-dataset-3]] and
   never actioned. Gates 11 of the 45 strains.
2. **Callability turns on plate count, not on the model.** Every clause of the previous
   version was wrong. The delta-method multiplier is **2.98--3.94, mean 3.44**, measured
   on the four run-4 triples with all six parents scored, not sqrt(7) = 2.65; sqrt(7)
   holds only where all seven fitnesses equal 1, and a well-behaved panel at f = 0.85
   would give 2.18. The per-strain SE of 0.121 traces to no artifact; at ONE pick, the
   structure actually run, it is 0.092 at 3 plates and 0.066 at 6, both lower bounds
   because colony picking has never been replicated. Callable means |tau| > 1.96 SE(tau),
   a convention the note had left unstated, which puts the floor at **0.617 at 3 plates
   and 0.446 at 6**. Of the 39 in-basis targets, **10 clear the 3-plate floor and 25 the
   6-plate floor on the raw predictions**; calibrated at the derived slope of 1.254 those
   become 18 and 39, but that pair is calibration-dependent and the targets sit 18 to 31
   prediction-SDs outside the range the calibration was fitted on. Kuzmin's magnitude
   thresholds are 0.08 and 0.12; 0.20 is a negative-only cutoff, tau < -0.2, and was
   never a magnitude threshold. Their median SE(tau) is **0.0785**, not 0.031, because
   the released p column is one-sided; we sit at 4.0x that at 3 plates and 2.9x at 6.
   **The round therefore turns on how many plates it gets**: at 3 most targets are out of
   reach, at 6 most or all are inside, and at 8 every target clears even without
   calibration. Corrected (A2, with A1's calibration and A3's Kuzmin re-derivation).
3. **Calibration assumes an unbiased mean.** The 0.43 slope assumes mu_pred ~ mu_y. One
   read of the inference parquet would settle it.
4. **Plate contention.** The pick-replication round (26 strains, 57 tubes) and this
   round (65 strains, 65 tubes at one pick) compete for the same plate.
5. **`YKL033W-A x YJR060W`.** Failed to construct; neither attempt date nor cause
   recorded. Blocks 0 of the 39 targets. Do not re-propose until an attempt is run and
   its outcome recorded.

### 6. Files

Primary checkout: `/home/michaelvolk/Documents/projects/torchcell`.

**Notes**

- `notes/experiments.W019-echo-crispr-array.build-list.md` -- bench sheet
- `notes/experiments.W019-echo-crispr-array.next-strains-to-construct.md` -- rationale
- `notes/experiments.W019-echo-crispr-array.run4-handoff.md` -- prior round
- `notes/experiments.W019-echo-crispr-array.session-handoff-2026-08-23.md` -- session record
- `notes/assets/pdf-output/experiments.W019-echo-crispr-array.build-list.pdf` -- rendered sheet

**Scripts** (`experiments/W019-echo-crispr-array/scripts/`)

- `triple_design_rank_sampling.py` -- selects the round
- `verify_triple_build_list.py` -- audits the round
- `run4_measured_summary.py` -- run-4 measured record and gap table

**Pinned inputs**

- `experiments/010-kuzmin-tmi/results/inference_3/top_k_constructible_panel12_k200.csv`
- `experiments/010-kuzmin-tmi/queries/001_small_build.cql`
- `experiments/W019-echo-crispr-array/data/run4_doubles_2026-08-06/Single-and-Double-KO-Strains-List-Order.csv`
- `experiments/010-kuzmin-tmi/results/construction_validation_doubles.csv`
- `$DATA_ROOT/data/torchcell/{tmi,dmi}_kuzmin{2018,2020}/processed/lmdb`

**Do not use** (earlier inference, different gene panel):
`experiments/010-kuzmin-tmi/results/constructible_triples_panel12_*.parquet` and
`.../inference_3/triples_table_panel12_k200.csv` (122 rows, all constructible, not the
top-k the set-cover ran on).

### 7. Reproduce

```bash
cd /home/michaelvolk/Documents/projects/torchcell
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/run4_measured_summary.py
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py
```

The verifier exits 1 on any failed check. Render the bench sheet with:

```bash
bash notes/assets/publish/scripts/bib_tex_pdf_landscape.sh \
  "$PWD/notes/experiments.W019-echo-crispr-array.build-list.md" "$PWD/notes" \
  experiments.W019-echo-crispr-array.build-list
```

### 8. Audit charter

The work below is for an independent session. It is an audit with authority to fix.

**Objective.** Establish whether every quantitative claim in the W019 build list,
rationale note and run-4 handoff is correct and traceable to a producing artifact, and
correct those that are not.

**Priority order.**

1. The unverified claims in section 4, in the order listed. Each either gets traced to
   an artifact and added to the verifier, or is corrected, or is relabeled as a
   hypothesis.
2. The five open items in section 5, specifically item 3, which one read of the
   inference parquet resolves.
3. The upstream inputs consumed as given: the prediction file, the measured-fitness
   bootstrap, and the two reference tables. Confirm each is generated by a committed
   script from a pinned source.
4. The SVG reproducibility gap: set `svg.hashsalt` and suppress the embedded date so
   figure output is byte-stable.

**Method.**

- Do not trust this brief. Re-derive independently and report disagreement.
- Every numeric claim must trace to a committed script reading a real result file.
  Values that cannot be traced are relabeled, not silently retained.
- Distinguish "not measured", "measured and null", and "measured and significant".
  Label every hypothesis as such in the same sentence.
- Extend `verify_triple_build_list.py` rather than writing parallel checkers, so a
  single command gates the notes. It must remain at 0 failures.
- Quote sources verbatim with sha256 where a published value is involved.

**Constraints.**

- All work in a git worktree via `/setup-worktree`; land via `/enqueue-merge`. The
  primary checkout stays on `main`.
- Do not edit `experiments/W019-echo-crispr-array/data/` or any pinned input.
- Retire superseded artifacts with `/deprecate`; do not delete.
- Prose follows [[writing-style-guide]]: no em-dashes, American spelling, verbatim
  quotes unaltered.

**Deliverable.** A dated section appended to this note recording, per claim: the
verdict, the evidence, and the action taken. Plus the verifier at a higher check count
with 0 failures, and a re-rendered bench sheet if any table changed.

### 9. Prompt for the audit session

```text
Audit the W019 first-trigenic-round design and the run-4 record, and fix what is wrong.

Read first, in this order:
  notes/experiments.W019-echo-crispr-array.audit-brief.md   <- charter, sections 4, 5, 8
  notes/experiments.W019-echo-crispr-array.build-list.md
  notes/experiments.W019-echo-crispr-array.next-strains-to-construct.md
  notes/experiments.W019-echo-crispr-array.run4-handoff.md

Do not trust the brief. Re-derive independently and report every disagreement.

Baseline to preserve: experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py
currently passes 61 checks with 0 failures and exits 1 on failure. It must still pass, at a
higher check count, when you are done.

Work the queue in section 4 "Not verified", in the order listed. For each claim, do one of:
  - trace it to a committed script reading a real result file, and add a check for it
    to verify_triple_build_list.py; or
  - correct it in the note, with the evidence; or
  - relabel it as an explicit hypothesis if it cannot be traced.
Never silently retain an untraceable number.

Then, in order:
  - section 5 item 3: read the inference parquet and settle whether the 0.43 calibration
    slope's unbiased-mean assumption holds.
  - confirm each pinned input in section 6 is generated by a committed script from a
    pinned source: the prediction file, run4_strain_bootstrap.csv, reference_smf_12panel.csv,
    construction_validation_doubles.csv.
  - fix SVG byte-reproducibility: matplotlib emits a dc:date and randomized clip-path ids,
    so figure output differs between identical runs. Set svg.hashsalt and suppress the date.

Rules:
  - Every numeric claim traces to a committed script in experiments/W019-echo-crispr-array/.
  - Label hypotheses in the same sentence. Distinguish "not measured" from "measured and
    null" from "measured and significant".
  - Quote published values verbatim with sha256.
  - Extend the existing verifier; do not write a parallel checker.
  - Work in a git worktree (/setup-worktree), land via /enqueue-merge, primary checkout
    stays on main.
  - Do not edit experiments/W019-echo-crispr-array/data/ or any pinned input.
  - Retire superseded artifacts with /deprecate, never delete.
  - Prose: no em-dashes, American spelling, verbatim quotes unaltered.

Deliverable: a dated section appended to the audit brief recording, per claim, the verdict,
the evidence and the action taken; the verifier green at a higher check count; and a
re-rendered bench-sheet PDF if any table changed.
```

### Related

- [[experiments.W019-echo-crispr-array.build-list]]
- [[experiments.W019-echo-crispr-array.next-strains-to-construct]]
- [[experiments.W019-echo-crispr-array.run4-handoff]]
- [[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]]
- [[experiments.W019-echo-crispr-array.scripts.run4_measured_summary]]
- [[experiments.W019-echo-crispr-array.scripts.triple_design_rank_sampling]]

## 2026.08.24 - Audit result

Independent audit against the charter in section 8, run as five parallel auditors over
disjoint claim clusters, integrated and cross-checked by the driver. Sections 1 to 7 above
describe the **pre-audit** state and are left intact as the record of what was believed;
where a row there is now known to be wrong it has been marked in place.

### The headline

**The stated reason this round is blocked does not survive.** Section 5 item 2 read "tau is
not callable at current replication ... 0 of 39 targets clear it." Two independent errors
pushed that conclusion the same way:

1. `sigma_y = 0.0535` is the label SD of `tmi_kuzmin2018` alone (n = 91,111), not of the
   010 label population (0.0629 on the val split, n = 37,673). The wrong value inverted the
   dispersion, turning a slope near 1.25 into 0.43 and shrinking predicted tau about 3x.
2. `SE(tau) = sqrt(7) s` holds only where all seven fitnesses equal 1. The measured
   delta-method multiplier over the four run-4 triples with all six parents scored is
   2.98 to 3.94, mean **3.44**, which raised the floor.

Corrected, **10 of 39** targets clear the 3-plate floor and **25 of 39** the 6-plate floor
on the raw predictions; calibrated they are 18 and 39. The round turns on plate count, not
on the model. Neither error is visible from inside its own cluster, which is why the two
had stood.

### Verifier

`experiments/W019-echo-crispr-array/scripts/verify_triple_build_list.py`: **235 checks,
0 fail**, exit 1 on any failure, up from the 61-check baseline. The five audit groups live
in `scripts/audit_checks/` and are imported by the verifier, which stays the single entry
point, shares one `check()`, one report CSV and one exit code, and injects its own constants
into each module so the gene panel and pinned paths have exactly one definition.

| group | n | | group | n |
|:------|--:|---|:------|--:|
| `rederive` | 47 | | `notes` | 15 |
| `calibration` | 35 | | `measured` | 10 |
| `published` | 33 | | `basis` | 9 |
| `precision` | 28 | | `inventory` | 8 |
| `provenance` | 24 | | `kuzmin` | 6 |
| `contingency` / `doubles` / `selection` / `tau` | 5 each | | | |

### The section 4 queue, per claim

| Claim | Verdict | Evidence and action |
|:------|:--------|:--------------------|
| Val metrics per checkpoint and W&B run ids | **Confirmed, misattributed** | The stated pairs are last-epoch summaries (epoch 63/63/48). Every prediction comes from `c7671wgj` best-pearson **epoch 24**, re-evaluated at r = 0.4619, RMSE = 0.0562. Table now carries both columns; values read from `prediction_calibration_stats.csv` |
| `sigma_y = 0.0535`, mean `-0.048` | **Wrong** | Val split is 0.062914 / -0.009139 (n = 37,673); whole build 0.063264 / -0.008024, matching all three runs' logged `normalization/gene_interaction/*`. The stated pair is `tmi_kuzmin2018` alone, 23% of training records. Corrected |
| `sigma_pred` column, 0.43 slope, "over-dispersed 2.3x" | **Wrong, sign inverted** | Algebra is right; inputs were not. With the correct `sigma_y` the identity leaves **two** positive roots, so `sigma_pred` is not identified: slope lies in [0.805, 1.321] and the model is **under-dispersed**. Measured `sigma_pred` = 0.023179 over the full 465,735,532-row inference parquet matches the lower root, giving slope 1.254. Corrected |
| Shrunk range 0.148-0.278, "very stringent for 15 of 39" | **Wrong, twice** | Corrected range is 0.33 to 0.94. Separately, Kuzmin's 0.20 is a **negative-only** cutoff (`tau < -0.2`, Fig. S15); no magnitude form exists, so it was applied to positive predictions. Both corrected |
| Mean prediction by zero-data count, Spearman 0.634 | **Confirmed** | 0.45310 / 0.52996 / 0.66764 at n = 16 / 17 / 6; rho = 0.634288, p = 1.4521e-05, n = 39, two-sided. Now checked |
| Per-strain `s = 0.121` at 2 picks / 3 plates | **Untraceable** | No script or CSV produces it. Measured at ONE pick, the structure actually run: 0.092 at 3 plates, 0.066 at 6, both lower bounds since colony picking has never been replicated. The "2 picks" half is not estimable at all. Replaced, with the unmeasured component labeled |
| Six-plate floor 0.55 | **Wrong, and the brief's reason was also wrong** | It IS reproducible, from an undisclosed model using run 3's failed-plate-inflated plate SD. Correct floor is 0.446. Corrected |
| "roughly 9x more replication" | **Wrong** | Reaching s = 0.04 takes ~20 plates (6.6x). Matching Kuzmin needs s = 0.023, about 175 plates (58x), which is reachable in a single day because it sits above the `sigma_day` floor. Corrected |
| Kuzmin median SE(tau) = 0.031 | **Wrong by 2.5x** | Their released `P-value` column caps at exactly 0.500, so it is **one-sided**: `SE = abs(tau)/Phi^-1(1-p)` gives median **0.0785** (n = 392,817). Sidedness is measured, not assumed, by testing both readings against Kuzmin's own digenic SD column (Spearman 0.980 vs 0.411, n = 410,295). Corrected |
| Costanzo coverage for the two zero-data genes | **Numbers confirmed, phrasing wrong** | "~3,544 digenic partners" belongs to `YLR312C-B` alone and is exactly 3,544; `YER079W` has 5,232. "4 SMF" is not distinctive, every panel gene except `YLR312C-B` has 4. Corrected |
| Failed W&B group at val r = -0.03 | **Confirmed** | Rank-0 run `vjfp4d83`, val r = -0.0314, train r = 0.0007, val RMSE 0.0629 equal to `sigma_y`. Now named in full |
| `top_k_constructible_panel12_k200.csv` | **Traced, one gap** | Generated from the 465M-row inference parquet for the epoch-24 checkpoint. "Constructible" means panel-membership, **not** lab feasibility, and the 12-gene panel was itself chosen to maximize top-200 coverage, so panel and targets are co-derived. The parquet had no sha256; it is now pinned |
| `run4_strain_bootstrap.csv` | **Traced** | `run4_doubles_48h.py`, from three sha256-pinned plate JPEGs, resampling unit the plate (n = 3), N_BOOT = 4000, seed 1234. `boot_se` confirmed a bootstrap SE and `across_plate_sd` a ddof=1 sample SD numerically, ratio 0.463 to 0.477 against the analytic 0.4714 |
| `reference_smf_12panel.csv`, `construction_validation_doubles.csv` | **Traced; one defect found** | Both re-ran byte-identical. But the SMF panel still listed LCL1 (`YPL056C`) from the pre-run-4 design, which is why the bench sheet showed no published SMF for `YLR104W`. Costanzo publishes it at 1.0322 ± 0.0453. Panel rebuilt, gap closed |
| Run-4 epsilon excluded by design | **Confirmed** | Remains deliberately absent |

### Defects found outside the queue

| Defect | Status |
|:-------|:-------|
| The `zero-data tri` column read `0` for `no_ylr` and `--` for `count` and `uniform`. Real values are **9**, 15 and 11, so the table made the cheapest design look like the safest | Corrected. The column is now emitted by the design script rather than hand-entered, along with a `one_wave` column: `no_ylr` is the one strategy selecting a triple with no built parent, so its 40 strains do not build in one wave |
| The flagged block runs ranks **1-11**, not 1-10; the first clean target is rank 12. Wrong in the note, the design script comment and the verifier check alike | Corrected in all three |
| Both the design script and the verifier sorted predictions with pandas' unstable default quicksort, and the basis contains exactly one tie (0.42041015625, ranks 34/35). The verifier copied the same sort, so it structurally could not detect the problem | Both moved to `mergesort`. The selection is invariant to the tie; the printed rank numbers were not |
| `YKL033W-A` also has zero built doubles, not only `YLR104W`. The build-list bullet additionally named `D13`, which contains no `YLR104W` | Corrected, both genes named with their correct six doubles each |
| Wells per strain came from `(378 - 28) // measure`, where `28` has no source anywhere in the repo and conflicts with `next_round_layout.py`'s 20-well WT reserve | Replaced with named sourced constants. 58 wells are left unallocated to WT, which the note now states |
| Two picks was reported as 135 tubes from `measure * 2 + 7`; it is 130 | Corrected |
| The kuzmin note-table check was vacuously passable: its `if g in noted_tbl` guard meant deleting a gene row left it green, and its reported "23 rows compared" counted 12 rows from unrelated tables | Rewritten to parse the table properly, assert 11-gene coverage, and check the **digenic** column too, which was never gated |
| The `used` and `serves (rank)` columns of the existing-doubles table were never gated | Recomputed: 13 of 13 correct. Check added |
| A check asserted that the SMF reference had **no** row for `YLR104W`, encoding the defect as expected state | Inverted to assert the fix |
| `YER079W x YLR104W` at eps = -0.5672, P = 1.83e-04 is the strongest significant in-panel digenic and sits inside selected triple rank 5, unmentioned anywhere | Documented. It **weakens** rank 5: the interaction is not novel, and the reciprocal Costanzo screen disagrees by 26x, so the sign is reliable and the magnitude is not |
| The prior session's correction of "0 of 10" to "0 of 9" was itself a regression: it re-derived the denominator from the superseded TEN set, dropping the very gene whose reinstatement was that session's headline correction | Restored to **0 of 10**, all measured, none significant, over the eleven-gene basis |
| SVG output was not byte-reproducible (`svg.hashsalt` unset, `dc:date` stamped) | Fixed in `torchcell.utils.savefig_true_size_svg`, which covers all 17 callers repo-wide rather than the two W019 scripts. Verified byte-identical across two clean runs |

### Open items after the audit

1. **How many plates does this round get.** At 3, most targets are unreachable; at 6, most
   or all are inside; at 8, every target clears without relying on calibration. This is now
   the decision the round rests on.
2. **All three run-4 plates fail the frac-above-WT gate** (0.40 / 0.32 / 0.24 against a
   limit of 0.20). `run4_plate_qc.csv` passes them on WT CV alone. The fitness LEVELS, and
   therefore the measured multiplier, are less trustworthy than the SPREADS.
3. **`YLR312C-B` in or out** is unchanged and still gates 11 of the 45 strains.
4. **The WT well count** is unstated, and run 4's epsilon was unreportable precisely
   because its WT denominator rested on one colony.
5. **The inference parquet is now sha256-pinned but has no provenance record.** It is
   owned by uid 7474 and is the single upstream artifact the whole design rests on.
6. **A retracted claim is still live in committed code**: the docstring of
   `experiments/010-kuzmin-tmi/scripts/investigate_YLR313C_smf_and_interactions.py` and its
   note still say the `YLR312C-B` KO deletes SPH1. R64-4-1 refutes it.

### Not done

- The per-sample val and test prediction parquets are gone from this machine. One eval
  slurm job would collapse the [0.805, 1.321] slope interval to a single number. Not run;
  the interval is reported instead of a point estimate.
- W&B is the only source for the three last-epoch metric pairs and the failed run.
  `experiments/*/slurm/` is gitignored repo-wide, so the eval logs cannot be committed;
  `prediction_calibration_stats.csv` is the committed artifact and pins each log by sha256.
- The exact SGA significance test is stated in neither Kuzmin, Costanzo nor the mirrored
  Baryshnikova 2010 markdown. It would be settled by opening that paper's software zip.
- Cellpose segmentation bit-determinism is not measured, so `run4_strain_bootstrap.csv` is
  traced but not proven reproducible.

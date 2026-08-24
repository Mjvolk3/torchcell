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
| `basis` | 7 | 39 in-basis targets; ranks 1-10 all touch a zero-data gene; no-parent targets are ranks 26, 32, 37; set-cover intact at 31 of 31 within-TEN |
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
| Six-plate floor of 0.55 | rationale note | Not reproducible from the note. Scaling s by 1/sqrt(2) gives 0.44. A nested model in which only the plate term shrinks would approach 0.55, but the decomposition used is not shown |
| "roughly 9x more replication" | rationale note | Reaches per-strain s ~0.04 and a callable floor near 0.21. The adjacent sentence cites Kuzmin's median SE(tau) of 0.031, which 9x does not reach. Ambiguous as written |
| Kuzmin median SE(tau) = 0.031, back-solved from tau and p | rationale note | Not re-derived this session |
| Costanzo coverage for the two zero-data genes (`YER079W` 4 SMF; `YLR312C-B` 2 SMF, ~3,544 digenic partners) | rationale note | Not verified |
| W&B group `compute-3-3-2059267_f83395...` is a failed run at val r = -0.03 | rationale note | Not verified |
| `top_k_constructible_panel12_k200.csv` (the prediction source) | pinned input | Consumed as given. Generation and provenance not audited |
| `run4_strain_bootstrap.csv` (all measured fitness) | pinned input | Consumed as given. Not re-derived from segmentation |
| `reference_smf_12panel.csv` and `construction_validation_doubles.csv` | pinned inputs | Read, generation not re-run |
| Run-4 epsilon | excluded by design | Documented as not reportable (double/single ratio 0.758 against a multiplicative expectation of ~1.07). Deliberately absent from all tables |

### 5. Open items carried forward

1. **`YLR312C-B` in or out.** SGD "ORF, Merged"; no discrete protein; zero trigenic
   training data; 0 of its 9 in-panel digenics is significant. Substitution with SPH1
   (`YLR313C`) was recommended in [[experiments.010-kuzmin-tmi.inference-dataset-3]] and
   never actioned. Gates 11 of the 45 strains.
2. **tau is not callable at current replication.** SE(tau) ~ sqrt(7)*s = 2.65s; at
   s ~ 0.121 the smallest callable |tau| is 0.63, against Kuzmin's thresholds of 0.08,
   0.12 and 0.20. 0 of 39 targets clear it.
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

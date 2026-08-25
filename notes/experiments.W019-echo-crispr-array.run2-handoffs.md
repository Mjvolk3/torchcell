---
id: reg535oujxmzbkfr7b36yx1
title: run2-handoffs
desc: ''
updated: 1787625464373
created: 1787625464373
---

**Why these are kept, and what has since replaced them.**

Two wet-lab handoff notes from the run-2 colony-fitness assay, written 2026-07-19
and 2026-07-21. They were `scratch.*`, were lost from disk on 2026.08.24, and
were recovered out of dangling git objects; graduating them here is what stops
that happening again.

They are kept as **history, not as instructions**. Three things in them are now
superseded, and following them would be a mistake:

- The **PDF build recipe** (2026-07-21 SS8) described a one-off that swapped
  `.svg` to `.png` and sanitized glyphs into a `.build_assay.md` intermediate.
  That is folded into `notes/assets/publish/scripts/bib_tex_pdf.sh`, which no
  longer swaps to PNG at all: it renders SVG as vector through `rsvg-convert`,
  so figures stay zoomable. Use the script.
- The **experiment path** was `experiments/019-echo-crispr-array/`. It is now
  `experiments/W019-echo-crispr-array/`.
- The **segmentation approach** was the `quant/approach_1..3` exploration.
  That was replaced by `torchcell/sga/cellpose_seg.py`; see
  [[experiments.W019-echo-crispr-array.cellpose-segmentation-plan]].

What remains useful is the record of what was decided and why: the SOPs, the
open-issues ordering, and the state of the assay at run 2. Later rounds are in
[[experiments.W019-echo-crispr-array.run4-handoff]].

## 2026.07.19 - Handoff, as written

## SCRATCH — Wet-lab fitness assay: handoff to next session (2026-07-19, rev 2)

**This is a throwaway handoff note. Do NOT `git add` it — `scratch.*` notes are never committed in this repo (CLAUDE.md, "Scratch Notes Are Not Committed"). Delete it once its content has been folded into the real note `notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md`.**

**This file IS the handoff — there is no second scratch note.** It was written in place over the earlier 2026-07-19 draft (which predated any inspection of the new data), so any cross-reference below to "the older/previous scratch note" means *the prior contents of this same path*, now gone. Everything worth keeping from it — including the §8 PDF build recipe — was carried forward into this text. **Before deleting this file, harvest §8; it is now the sole record of that recipe.**

## STATUS 2026-07-19 (later session) — the analysis is DONE; much of §5 is now history

Run 2 has been quantified, scored, compared to published SMF, and written up. **The results live in
the real note** (`notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md`, section
`## 2026.07.19 - Run 2`), produced by the committed script
`experiments/W019-echo-crispr-array/scripts/run2_volume_timepoints.py`.

Closed since this note was written: §5.2/§5.3 (geometry + the `image.py` n_cols bug — fixed, and a new
`grid_mode="lattice"` was needed because the backlit captures broke `_plate_roi` outright), §5.4 (a
run-2 runner now exists), §5.6 (both plates quantified), §5.7 (both timepoints done), §5.9 partly
(orientation resolved empirically). **Orientation finding overturns §3: the ops are `identity` — A1
reads top-LEFT in these backlit photos, matching Plate 5's B2-top-left convention — not `flip_h`.**

Still open and still worth doing: **§5.1 (commit the `019` tree — still untracked)**, §5.5
(TransferReport parser), §5.8 (batch centring as such — between-batch variance is now measured, but
per-batch median centring was not separately implemented), §5.10 (`tc-lit` sourcing), §5.11 (cleanups),
§6 (the `sga` -> `wetlab` rename), §8 (PDF build helper).

## 0. WHERE THE WORK LIVES — read this first

**Everything in this project is untracked and lives in the primary checkout. Do NOT create a worktree.**

- Path: `/Users/michaelvolk/Documents/projects/torchcell` (the primary checkout), currently on branch **`paper/figures-fig1`** — an unrelated Fig-1 branch, not `main`.
- **VERIFIED untracked** (`git status --porcelain` → `?? torchcell/sga/`, `?? experiments/W019-echo-crispr-array/`): `torchcell/sga/`, `experiments/W019-echo-crispr-array/` (scripts, data, results), `notes/assets/images/019-echo-crispr-array/`, the real note, and this scratch note. `git ls-files torchcell/sga experiments/019-echo-crispr-array` → **empty**.
- **Why this matters:** CLAUDE.md mandates that work go through `/setup-worktree`, but **untracked files do not follow into a new worktree** — a session that dutifully creates one will find the entire assay missing. Work in the primary checkout for now; the untracked tree is also why §6's rename is cheap and why `git mv` will fail.
- **Corollary (VERIFIED):** `experiments/010-kuzmin-tmi/results/optimized_doubles_setcover_panel12.csv` exists on `origin/main` but is **absent from this working tree** (`ls` → No such file). Fetch it from `origin/main` if the doubles design is needed (see §7.1).

## 1. What this is / how to resume

We are running a wet-lab colony-fitness assay: ECHO acoustic dispensing of a 12-gene CRISPR single-KO panel (+ BY4741 WT + blank media) onto 384-format agar plates, imaged after ~2 days, quantified in-repo by a SGAtools-derived pipeline at `torchcell/sga/` (to be renamed `torchcell/wetlab/`, §6). Plate 5 is fully analysed and written up. **Run 2 (two plates, P1=2.5 nL / P2=5 nL, plated 2026-07-17) is staged but NOT yet quantified — that is the main job.** It now has **two imaging timepoints, ~43.7 h and ~50.3 h**, with a segmentation-grade image for each plate at each — four images, a full 2×2 (§3). Three facts settled by the user on 2026-07-19 shape the work and are easy to miss: **A1 sits at the TOP RIGHT** of every imaged plate (§3, changes how registration should be validated); the **LCL1/SPH1 panel carry-over is deliberate** because this is assay development, not production (§7.1); and **P1 and P2 share one layout**, so they are a clean volume comparison but *not* independent randomizations (§3). To resume: read the real note `notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md` (**494 lines**, untracked; H2 = dated session, newest at bottom), then work §5's TODO list top-down. Everything marked VERIFIED was read out of the actual files; everything marked UNRESOLVED was not.

## 2. State of the world

### Done — Plate 5 (the 2026.07.16 + 2026.07.17 sections of the real note)

- One plate, OD1, **both volumes on one plate** (cols 2–12 = 2.5 nL, cols 13–23 = 5 nL), border-free inner array **B2–O23** (14 rows × 22 cols = 308 wells).
- **Replicates: 22 wells per sample = 11 per scored condition.** Volume is a within-plate block, so each strain×volume cell has **n ≤ 11**, not 22 — the note says so at L384-385 (*"`n` = colonies used of 11 placed"*), as does `run_plate5_volume.py:123`. Do not quote "22 replicates" as the scored n; it matters for the §5.10 header fix and for any power comparison against run 2's 29/plate.
- Image → results end-to-end; corrected full-colony segmentation replaced an early per-cell segmentation that under-measured colonies: **WT CV fell 0.26 → 0.085 (2.5 nL) and 0.18 → 0.083 (5 nL)** (note L376), median 5 nL fitness rose ~0.6 → 0.78 (L378, L410), and the apparent "systematic ours-below-published offset" turned out to be **largely a segmentation artifact**, not biology.
- **Modality control: dark-field vs transillumination gives r = 0.9520** — imaging modality does not change scores (`results/plate5_modality_comparison.csv`, `scripts/compare_modalities.py`).
- **Reference comparison is 12/12 against the PLATED set** (`results/reference_smf_12panel.csv`, built by `scripts/build_reference_smf.py`), after backfilling LCL1 from a Costanzo validation-panel override (`COSTANZO_OVERRIDE = {"YPL056C": (0.9802, 0.0815)}`, the `YPL056C_sn3389` comment at `build_reference_smf.py:73-74`) and SPH1 from an extra queried-singles table. **Reconcile this with the panel error below before writing:** 12/12 covers the strains actually on the plate, which include LCL1/SPH1. **Against the canonical panel-12 it is 10/12.** Do not let "12/12" read as canonical-panel validation.
- **Design confound (documented):** volume is blocked by plate side on Plate 5.
- Math documented **by SGA effect** in the note (L42–122): plate-specific ✓, row/col ✓, spatial ✓, **competition partial (spatial step absorbs some; no explicit neighbour-size model)**, **batch ✗ (not implemented)**, linkage explicitly declined (no query → no query–array linkage).
- **Known mistake:** Plate 5 carries **LCL1 (YPL056C)** and **SPH1 (YLR313C)** in place of the canonical panel-12 genes **YIL174W** and **YLR104W**. Per the author this was a plating mistake; LCL1/SPH1 appear zero times in any panel-selection row. Canonical panel-12 at k=200 (note L474): COS111, YOS9, YER079W, MMS2, **YIL174W**, CBF1, YKL033W-A, YEH1, **YLR104W**, YLR312C-B, ELC1, RPS9A.
- **Plate 5's own incubation time and imaging time were never recorded** — the "~48 h at 26 °C" in the note is the *literature target* sourced from Kuzmin 2018, not a measurement of our plate. If the new section compares run-2 timing to Plate 5, say this explicitly.

### The pipeline (all VERIFIED against source)

Package `torchcell/sga/` — 9 modules, **untracked in git, zero tests**. Public surface re-exported by `__init__.py` (`__all__`, lines 46–63):

| module | key public functions (verbatim, with file:line) |
|---|---|
| `io.py` | `well_to_rowcol` (:26), `read_gitter_dat` (:42), `read_echo_picklist` (:60), `merge_layout` (:80) |
| `image.py` | `quantify_plate_image(path, n_rows=14, n_cols=22, overlay_path=None, circularity_flag=0.80, polarity="auto")` (:172) |
| `register.py` | `resolve_orientation(grid, layout, n_rows=14, n_cols=22, inner_row0=2, inner_col0=2, blank_name="Blank_media", empty_thresh=1.0)` (:37) |
| `normalize.py` | `normalize_plate(df, cfg)` (:117) |
| `score.py` | `score_plate(df, cfg, plate_id)` (:39), `score_table(report)` (:130) |
| `assay.py` | `shape_by_volume` (:32), `volume_position_confound` (:61), `zfactor` (:91), `volume_assay_metrics` (:103), `recommend_volume` (:146) |
| `viz.py` | `plate_heatmap` (:49), `layout_heatmap` (:90), `value_histogram` (:119), `colony_shape_by_volume` (:133), `strain_fitness_plot` (:169) — **not** in `__all__`, import from submodule |
| `models.py` | pydantic `NormalizationConfig` (:22), `StrainScore` (:94), `ScoreReport` (:129) |

Runner: `experiments/W019-echo-crispr-array/scripts/run_plate5_volume.py`. Call order: [1] `quantify_plate_image` (:91) → [2] `read_echo_picklist` (:98) + `resolve_orientation` (:99) → [3] `normalize_plate` (:103) → [4] `groupby("volume_nl")` → `score_plate`/`score_table` (:108–115) → [5] `volume_position_confound` (:144) → [6] `volume_assay_metrics` + `recommend_volume` (:148–151) → [7] figures → [8] optional reference compare (:174–185). `merge_layout` is never called — `resolve_orientation` does the layout join internally (`register.py:61`).

**No time/growth axis exists anywhere.** `grep -rn "hour\|incubat\|timepoint\|time_h\|growth_time" torchcell/sga/` → **zero matches** (VERIFIED). Colony size is treated as static. Tolerable because every downstream quantity is a *within-image ratio* (`norm = size_spatial / ref_med`, `normalize.py:134`; `relative_fitness = median_norm / wt_median`, `score.py:79-83`), so absolute growth between two timepoints largely cancels and each timepoint can be processed as an independent run.

## 3. NEW DATA — `experiments/W019-echo-crispr-array/data/run2_2026-07-17/`

Run 2, plated 2026-07-17 by **AS = Aurosish Sharma** (identity carried forward from the previous session's note, L24; the plate-marker glyphs read `AS.` and corroborate it).

| file | what |
|---|---|
| `P1_2p5nL_cherrypick_13strain.csv` | ECHO cherrypick picklist, plate 1, 2.5 nL, 384 rows |
| `P2_5nL_cherrypick_13strain.csv` | same layout, plate 2, 5 nL, 384 rows |
| `P1_2p5nL_transfer_report.csv` | ECHO fluid-op log, plate 1 (403 lines, 3-block) |
| `P2_5nL_transfer_report.csv` | same, plate 2 |
| `P1_2p5nL_plate_AA9E816C.jpeg` | labelled photo of P1 — **not segmentation-grade** |
| `P2_5nL_plate_B51C8A75.jpeg` | labelled photo of P2 — **not segmentation-grade** |
| `plate_view_3558C0B4.jpeg` | clean transillumination top-down, unlabelled — **= P1 (2.5 nL), CONFIRMED by EXIF** |
| `plate_view_D4CA749C.jpeg` | clean transillumination top-down, unlabelled — **= P2 (5 nL), CONFIRMED by EXIF** |
| `both_plates_incubator_agar-topside_IMG_4576.jpeg` | both plates on the incubator shelf, **inverted (agar topside, lid underneath)**; documents incubation orientation only, no colonies resolvable |

### Decoded facts (VERIFIED)

**Sample names — 14 unique, not 13** (verbatim, in file order of first appearance):

```
YLR312C-B, YEH1, MMS2, YJR060W, BY4741, YOS9, COS111, YPL081W,
SPH1, ELC1, LCL1, YER079W, YKL033W-A, Blank_media
```

"13strain" in the filename = **12 KO strains + BY4741**. The 13th strain is **BY4741**, the WT parental — it is *not* a HIS3/URA3 marker reference; no such entry exists in the files. This resolves the previous session's open guess ("find the extra one, likely a control / fixed WT ref"). `Blank_media` is a separate media-only control, making 14 distinct `Sample Name` values.

**The LCL1/SPH1 mistake was NOT fixed.** LCL1 and SPH1 are still present; **YIL174W and YLR104W are absent from both P1 and P2**. The sample-name set is **identical to Plate 5** — set difference empty in both directions.

**RESOLVED by the user (2026-07-19) — this is accepted, not an oversight to correct now.** In their words: *"we know that these 2 strains are in but maybe shouldn't be... we are just still using them to develop the assay."* So run 2 is an **assay-development run**, where the job is to make the measurement work; the exact panel membership is not what is being tested. Do not spend the next session re-plating to fix it, and do not write it up as an error in the run-2 section — write it as a known, deliberate carry-over.

**But carry the consequence forward:** these runs remain **off-panel for the canonical panel-12**, so no fitness number from Plate 5 or run 2 can be fed into the doubles design built on `optimized_doubles_setcover_panel12.csv` for the LCL1/SPH1 slots. The canonical genes **YIL174W and YLR104W have still never been measured by us.** When the assay graduates from development to production, the panel must be corrected first — flag that as a gate, and state the limitation plainly in the real note so a later reader does not mistake 12/12 plated coverage for canonical-panel coverage (§2).

Mapping read from the repo's own table (`scripts/build_reference_smf.py` `PANEL` dict at :41 + `results/reference_smf_12panel.csv`), which states the swap explicitly at **lines 13–16**:

| Sample Name | Systematic ORF | Common |
|---|---|---|
| YEH1 | YLL012W | YEH1 |
| YER079W | YER079W | — |
| YOS9 | YDR057W | YOS9 |
| MMS2 | YGL087C | MMS2 |
| YPL081W | YPL081W | RPS9A |
| ELC1 | YPL046C | ELC1 |
| YKL033W-A | YKL033W-A | — |
| YLR312C-B | YLR312C-B | — |
| **LCL1** | **YPL056C** | LCL1 |
| **SPH1** | **YLR313C** | SPH1 |
| YJR060W | YJR060W | CBF1 |
| COS111 | YBR203W | COS111 |

**Geometry — A1 origin, full 384, borders used.** Destination wells span rows **A–P** (all 16) × cols **1–24** (all 24) = 384, no duplicates, no empty border ring. This is the biggest difference from Plate 5's border-free B2–O23 inner array, and it breaks three pipeline defaults (§5.2).

**PHYSICAL ORIENTATION — A1 sits in the TOP RIGHT of the imaged plate.** Stated by the user (2026-07-19) and, in their words, *"that is how it has been"* — i.e. this holds for **every plate imaged in this project, Plate 5 included**, not just run 2. In all clean views the plate is landscape: **24 columns run horizontally, 16 rows vertically, with well A1 at the top-right corner**, so column index increases **right → left** and row index increases top → bottom. This is a horizontal mirror of the naive "A1 at top-left" reading.

Why this matters more than it looks: `resolve_orientation` (`register.py:37`) does **not** know this — it *infers* the orientation by trying all four dihedral operations and scoring blank-vs-grown agreement (`register.py:65`). Run 2 supplies only **6 blanks** (vs 22 on Plate 5), so that score is weak and a near-tie is plausible, and **the code neither detects nor warns about a near-tie** (§5.6). Now that the true orientation is known a priori, the next session should **assert it rather than infer it**: resolve as usual, then *check* the winning operation against A1-top-right and fail loudly on disagreement. That converts the weakest link in registration from a silent-wrongness risk into an assertion. **UNRESOLVED (mechanical):** which of the four ops corresponds to A1-top-right in this code's convention has not been worked out — derive it from the source, do not guess.

**Replicates:** 12 KO strains × **29** wells, BY4741 × **30**, `Blank_media` × **6** (wells E6, E12, E18, L6, L12, L18). 348 + 30 + 6 = 384 ✓. Layout is **randomized/scrambled**, not blocked (row A reads YLR312C-B, YEH1, MMS2, YJR060W, BY4741, YOS9, COS111, YPL081W, SPH1, COS111, BY4741, …).

**Source wells** (identical in both plates): BY4741=A1, YJR060W=A3, YPL081W=A5, YOS9=A7, ELC1=A9, COS111=A11, MMS2=A13, LCL1=A15, YEH1=A17, YER079W=A19, YKL033W-A=A21, SPH1=A23, YLR312C-B=C1, Blank_media=C3+C5.

**P1 and P2 are the SAME layout at two volumes.** Identical destination-well set, identical `Destination Well → Sample Name` map (0 mismatches), identical `Destination Well → Source Well` map. They differ **only** in `Transfer Volume` (P1 = `2.5` on all 384; P2 = `5` on all 384). Volume is a *between-plate* factor in run 2, versus a *within-plate* left/right split on Plate 5.

**DESIGN LIMITATION the user identified (2026-07-19) — the two plates are not independent randomizations.** In their words: *"we cannot use them for randomization because of the same layout."* This is correct and worth stating precisely, because it is easy to lose. The single layout **is** randomized/scrambled within a plate (row A reads YLR312C-B, YEH1, MMS2, YJR060W, BY4741, YOS9, …), which does break up *within-plate* spatial structure. What it does **not** do is vary **between** plates: every strain occupies the **same 29 wells on P1 as on P2**. So each strain's exposure to plate position — edge vs centre, any residual row/column or spatial gradient the normalization does not fully remove — is **identical across both plates rather than resampled.**

Three consequences to carry into the analysis:

1. **Positional bias does not average out across the two plates; it is reproduced.** Two plates give ~58 wells per strain, but not 58 *independent* positional draws — the residual position effect is common-mode, so averaging P1 and P2 reduces measurement noise while leaving positional bias essentially untouched. Do not treat n=58 as if it bought positional robustness.
2. **Position is now confounded with strain in the same way on both plates**, so a strain that happens to sit disproportionately near an edge carries that bias into every run-2 number. This is exactly what a *second, differently scrambled* layout would fix.
3. **It compounds the volume confound already noted.** P1 vs P2 differ in volume *and* nothing else — same layout, same source wells, same protocol — which is what makes them clean for a volume comparison, and simultaneously what makes the §5.8 "treat them as two batches" test unable to separate batch from volume.

**Recommendation for the next design: generate a second, independently scrambled layout** so replicate plates resample position. That is a picklist-generation change, not a pipeline change, and it is the single highest-value design improvement available. **Raise it with the user (§7.7)** — for an assay-development run the shared layout is arguably a feature (it isolates volume perfectly), so this is a decision about when to switch, not an error to fix.

**Plate-identity hazard (VERIFIED, unfixed):** both cherrypick CSVs carry `Destination Plate Name = Plate_13strain_384_2p5nL` — the 5 nL file still says `2p5nL`. Both transfer reports print that same name plus the same `Protocol Name = P1_Protocol.ecp`. Neither has a plate barcode. **Plate identity rests on filenames and on `Sample Pick List File Name` / `Run ID`, not on any plate-name field.** Trustworthy discriminators: `Run ID` `1784323560` (P1) vs `1784323869` (P2); and `Sample Pick List File Name`, which is a **Windows path, not a bare filename** — verbatim line 6 of each report:

```
Sample Pick List File Name,E:\20260717_TC_12sKO_Plating\TC_cherrypick_P1_13strain_384_2p5nL.csv
Sample Pick List File Name,E:\20260717_TC_12sKO_Plating\TC_cherrypick_P2_13strain_384_5nL.csv
```

**Match on the basename or a substring, never string-equality against a bare filename.** Bonus provenance: the operator's run directory is `20260717_TC_12sKO_Plating`.

#### TransferReport schema (VERIFIED — parse with a block splitter, NOT `pd.read_csv` alone)

Three blocks: header key/value **lines 1–10** → blank line 11 → `[DETAILS]` marker line 12 → column header line 13 → data lines 14–397 → blank line 398 → footer key/value lines 399–402 → trailing blank 403 (1-indexed). **The footer has no `[...]` marker**, so a "skip to `[DETAILS]`, read to EOF" parser silently ingests 4 ragged 2-field rows as transfers. **Terminate the data block on the first blank line.** Instrument identity is in the *footer*: `Instrument Name,E6XX-25034` · `Instrument Model,Echo 650` · `Instrument Serial Number,E6XX-25034` · `Instrument Software Version,3.2.3`.

20 columns, verbatim, in order:

`Date Time Point`, `Source Plate Name`, `Source Plate Barcode`, `Source Plate Type`, `Source Well`, `Destination Plate Name`, `Destination Plate Barcode`, `Destination Plate Type`, `Destination Well`, `Destination Well X Offset`, `Destination Well Y Offset`, `Sample ID`, `Sample Name`, `Transfer Volume`, `Actual Volume`, `Current Fluid Volume`, `Fluid Composition`, `Fluid Units`, `Fluid Type`, `Transfer Status`

- **384 data records per file**, 0 ragged, 0 duplicate destination wells.
- **`Transfer Status` is an EMPTY FIELD on all 768 rows — zero wells flagged.** No skips, no short dispenses, no survey failures. **After `pd.read_csv` it parses as `NaN`, not `""`** (VERIFIED: `t["Transfer Status"].unique()` → `[nan]`), same for `Sample ID` and `Fluid Units`. A check written as `== ""` matches zero rows — use `.isna()`. Treat empty as the success sentinel; the files give no example of a populated value, so **do not hardcode an enum**.
- **`Actual Volume` == `Transfer Volume` bit-identically in 768/768 rows.** *Caveat to carry into the note:* this makes `Actual Volume` a nominal echo of the request, **not** an independent per-drop metrology readout. Interpret it as "no exception raised," not as verified nL delivery.
- The genuinely measured field is **`Current Fluid Volume`** — per-transfer source-well remaining volume, strictly monotonic-decreasing within each source well (0 exceptions; the acoustic survey signal). Depletion confirms units are **µL**: P1 source A3 drops 0.070 µL over 29 × 2.5 nL (0.0725 expected); P2 A3 drops 0.140 over 29 × 5 nL (0.145 expected). Source volumes were 57.513–64.377 µL, far above 384PP dead volume.
- **`Source Plate Type` DIFFERS between file types** (VERIFIED): transfer reports say `384PP_AQ_SP2`; **cherrypick CSVs say `384PP_AQ_SP` (no trailing `2`)** on all 384 rows of both plates. Any cross-file join or column validation must tolerate this.
- Other static fields: `Source Plate Name=Source[1]`, `Destination Plate Type=384PP_Dest`, both offsets `0`, `Fluid Composition=100`, `Fluid Type=AQ`. **Both barcode columns empty on every row** — barcodes exist only in the JPEG filenames.
- **Picklist reconciliation is clean 1:1 on both plates:** 0 missing, 0 extra, 0 mismatched (source, destination) pairs, 0 sample-name or volume mismatches. **But row order differs (only 2/384 positional matches)** — the Echo reorders to group by source well. **Join on `Destination Well`, never on row index.**

#### Images (EXIF read this session — the P1/P2 assignment is now CONFIRMED, not inferred)

All four plate photos are 768 × 1024, iPhone 13 Pro. `DateTimeOriginal`, VERIFIED via PIL:

| file | timestamp | device |
|---|---|---|
| `plate_view_3558C0B4.jpeg` | 2026:07:19 **11:12:47** | iPhone 13 Pro |
| `P1_2p5nL_plate_AA9E816C.jpeg` | 2026:07:19 **11:13:19** | iPhone 13 Pro |
| `plate_view_D4CA749C.jpeg` | 2026:07:19 **11:13:53** | iPhone 13 Pro |
| `P2_5nL_plate_B51C8A75.jpeg` | 2026:07:19 **11:14:13** | iPhone 13 Pro |
| `both_plates_incubator_agar-topside_IMG_4576.jpeg` | 2026:07:17 **15:31:52** | iPhone 16 Pro |

**Capture order is clean-view → its labelled shot, twice.** Therefore **`plate_view_3558C0B4` = P1 (2.5 nL)** and **`plate_view_D4CA749C` = P2 (5 nL)** — settled by acquisition timestamp, not by an occupancy heuristic. (The earlier revision of this note assigned these at ~85 % confidence from a dropout-rate argument; that argument is now moot and has been cut. It reached the same conclusion, but the estimator that produced it also mis-read the lattice as ≈19 × 30 when the picklists prove 16 × 24 — do not reuse its numbers.) The stated imaging time "11:13" is confirmed (window 11:12:47–11:14:13).

Per-image assessment:

- `P1_2p5nL_plate_AA9E816C.jpeg` — label reads verbatim `7/17  AS.  OD1  2.5nL  1.`. **Unusable for quantification:** plate hand-held in front of a lightbox, so the top ~45 % is transillumination (dark colonies on bright agar) and the bottom ~55 % falls off the panel into dark-field (bright on dark) — **two polarities in one frame** — plus gloved fingers occluding the lower-right/lower-centre and noticeable keystone.
- `P2_5nL_plate_B51C8A75.jpeg` — label `7/17  AS.  OD1  5nL  2.`. Same hybrid split, slightly cleaner, still not segmentation-grade.
  > **Supersedes** the previous note's L33 reading of this label as `"FHT AS, OD1, 5nL, 2"`. `FHT` was a misread of the date `7/17`; the corrected read matches P1's prefix.
- `plate_view_3558C0B4.jpeg` / `plate_view_D4CA749C.jpeg` — **no label in frame**, clean transillumination, flat, near-orthographic. Segmentation-suitable. These are the two images to quantify.
- `both_plates_incubator_agar-topside_IMG_4576.jpeg` — 2261 × 1142, nothing measurable: the dark ovals inside the agar are shelf perforations seen through the plastic, **not colonies**. Confirms two rectangular OmniTray-style plates side by side, **inverted (base → agar slab → air gap → lid skirt)**, colony surfaces facing down. Red writing on both plate edges is illegible at that grazing angle.

**Retired assumption:** the previous session's plan relied on `quantify_plate_image`'s `polarity="auto"` (`image.py:172`) to handle dark-field vs transillumination, and expected the labelled JPEGs to work on that basis. **That premise fails** — auto-polarity is per-image, and the labelled JPEGs contain two polarities within a single frame. Do not retry them.

**All other image-derived numbers from the previous probe (colony counts, pitch in px, colony diameter, dropout percentages, lattice estimate) are NOT reproducible under CLAUDE.md's STRICT RULE** — they came from a throwaway scratchpad script, not a committed one in `experiments/W019-echo-crispr-array/`. **They must not enter the real note.** Regenerate anything needed from the committed run-2 runner (§5.5). The one qualitative observation worth carrying, to be re-derived and cited: at ~44 h the colonies were well separated with **no touching or merged colonies** in either clean view — a good measurement regime with margin before merging.

#### SECOND TIMEPOINT — staged 2026-07-19 in `run2_2026-07-17/t50/` (the "48 h" re-image, actually ~50.3 h)

The re-image planned in §4 **has been taken and staged.** Three files, EXIF-verified (iPhone 13 Pro, 768×1024, same as t44):

| file | EXIF timestamp | elapsed | what |
|---|---|---|---|
| `P1_2p5nL_view_t50_D9516A9F.jpeg` | 2026:07:19 17:47:50 | **50.30 h** | **clean transillumination view of P1 (2.5 nL)** — flat on the light panel, near-orthographic, segmentation-grade |
| `P1_2p5nL_labeled_t50_386087EB.jpeg` | 2026:07:19 17:48:20 | 50.30 h | labelled shot of P1, label verbatim `7/17  AS.  OD1  2.5nL  1.` — **again unusable**, hand-held at an angle, two polarities in one frame, bottom half off the panel |
| `P2_5nL_view_t50_E30F9F19.jpeg` | 2026:07:19 17:48:49 | 50.31 h | **clean transillumination view of P2 (5 nL)** — flat, near-orthographic, segmentation-grade |

**Naming correction, recorded because the user's message said otherwise:** the user described `E30F9F19` as the labelled 5 nL shot (`"7/17 AS. ODI 5nL 2."`). **Visual inspection shows no label anywhere in that frame** — it is a clean view. The plate identity (P2, 5 nL) is taken from the user; only the "labelled" part was wrong, and the filename reflects the corrected reading. There is **no labelled photograph of P2 at t50.**

**Assignment evidence — two independent lines agree.** (a) Capture order repeats the t44 pattern of clean-then-labelled: `D9516A9F` (17:47:50) immediately precedes the labelled *P1* shot (17:48:20), making it P1; `E30F9F19` follows as the remaining plate. (b) **Independent corroboration from the images themselves:** `E30F9F19` is visibly denser — more occupied positions and fewer dropouts — than `D9516A9F`, which is exactly what 5 nL vs 2.5 nL should produce, since double the dispensed volume delivers roughly double the founding cells. The user's stated identity, the capture order, and the biology all point the same way. **Deliberately kept qualitative:** per CLAUDE.md's STRICT RULE, no occupancy/dropout number may enter the real note unless a committed script in `experiments/W019-echo-crispr-array/scripts/` produced it. **Quantifying that density difference is a real result to generate (§5.6), not a claim to assert here.**

**Both clean views are usable, so the paired-timepoint comparison is fully enabled** — two plates × two timepoints = four segmentation-grade images:

| | ~43.7 h | ~50.3 h |
|---|---|---|
| **P1 (2.5 nL)** | `plate_view_3558C0B4.jpeg` | `t50/P1_2p5nL_view_t50_D9516A9F.jpeg` |
| **P2 (5 nL)** | `plate_view_D4CA749C.jpeg` | `t50/P2_5nL_view_t50_E30F9F19.jpeg` |

The pattern from t44 held exactly: **the flat-on-the-panel shots are good and the hand-held labelled shots are not.** That is now a two-for-two record and should be stated to the user as settled technique, not preference (§7.6).

## 4. TIMELINE and the paired-timepoint design

- **Plates into incubator:** 2026-07-17, **15:30** (user-supplied). **Corroborated by EXIF:** `IMG_4576` — both plates on the incubator shelf — was shot at **2026-07-17 15:31:52** on a second, independent device (iPhone 16 Pro).
- **Staged images taken:** 2026-07-19, **11:12:47–11:14:13** (EXIF).
- **Elapsed at imaging: 43 h 43 min ≈ 43.7 h** — about **4.3 h short** of the ~48 h the SGA protocol assumes.
- **48 h mark was ≈ 2026-07-19 15:30. The re-image HAS BEEN TAKEN — but at 17:47:50–17:48:49, i.e. ~2.3 h PAST the 48 h mark, giving 50.30 h, not 48 h.** Staged in `run2_2026-07-17/t50/` (§3). Call this timepoint **t50, not t48** — the directory and the proposed filename scheme are named accordingly, and writing "48 h" anywhere would be a fabricated number.
- **The two timepoints are 43.71 h and 50.30 h, separated by 6.58 h** (both computed from EXIF against the 15:30 incubator time; arithmetic done this session, reproduce it in the committed runner). That is a **+15 % increase in growth time**, a more substantial separation than the ~4.3 h originally anticipated — which makes the comparison *more* informative, not less.
- That yields a **paired two-timepoint comparison (43.7 h vs 50.3 h) on physically identical plates** — a genuine within-plate growth-time control, exactly analogous to the r=0.95 modality control. Quantify both timepoints with the *same* pipeline and compare.
- **Both elapsed figures inherit the unresolved Echo-clock question below.** If the instrument clock is right and 15:30 is wrong, every elapsed number shifts down by ~1 h (42.7 h / 49.3 h). The *difference* between timepoints, 6.58 h, is unaffected — it comes from two EXIF stamps on the same device — so **the growth-time comparison is robust even if the absolute hours move.** Publish the difference with confidence; footnote the absolutes.

**The ECHO clock appears to be ~1 h fast — use 43.7 h.** The transfer reports' `Date Time Point` puts physical dispensing at **P1 16:27:05.915 → 16:29:29.863** and **P2 16:31:55.665 → 16:34:17.030** on 2026-07-17, ~1 h *after* the stated 15:30 incubator time. The earlier revision of this note proposed resolving this by reading 15:30 as media/plate prep; **`IMG_4576`'s 15:31:52 EXIF makes that reading much less likely** — it shows both plates already on the shelf 55 min before the Echo's own first-dispense timestamp. Two independently network-synced phones agree with the user's 15:30; only the Echo 650 disagrees, by almost exactly an hour, which smells like a DST/timezone misconfiguration. **Leading hypothesis, not proof** — the photo could conceivably show blank plates pre-warming. **UNRESOLVED pending user confirmation (§7.2); until confirmed, publish 43.7 h and footnote the Echo discrepancy rather than switching to 42.7 h.** (Header `Run Date/Time` 15:20:59 / 15:26:08 timestamps run *creation*, not execution, so it does not arbitrate.)

Recorded dissent, now **largely OVERTAKEN BY EVENTS** — kept only so the argument is not silently re-litigated. A probe judged the re-image not worth it on growth grounds, reasoning that +4.3 h is marginal growth and that no amount of extra time recovers a dropout (a position that received zero viable cells will never grow one). **The premise was wrong on magnitude:** the actual separation is **6.58 h (+15 %)**, not 4.3 h. The dropout point still stands and is worth carrying — extra incubation changes colony *size*, not colony *presence*, so a dropout at t44 must still be a dropout at t50, and **that is itself a testable consistency check on the segmentation**: any position "empty" at 43.7 h but occupied at 50.3 h is a segmentation/threshold artifact, not biology. The probe's counter-recommendation — reshoot for image quality, flat on the panel, full-plate, label edge in frame — was independently borne out, since the labelled hand-held shots failed again at t50. Report the growth-time result and the imaging-quality observation separately.

## 5. TODO for next session

1. **Commit the `019` tree.** `experiments/W019-echo-crispr-array/` is entirely untracked, so **every Plate 5 number already in the real note has the same provenance gap** that §5.4 flags for the new parser. CLAUDE.md's STRICT RULE ("any artifact used in a note must be produced by a committed script") is currently violated repo-wide for this experiment. Commit `scripts/`, `results/`, and the run-2 `data/` before adding a new section to the note. Note the branch situation in §0 first.

2. **Make the pipeline handle A1-origin full-384.** The parameterization exists — `resolve_orientation`'s offset at `register.py:58-60`, `g.assign(row=g["row"] + (inner_row0 - 1), col=g["col"] + (inner_col0 - 1))` — but three call sites ride on Plate-5 defaults:
   - `register.py:40-43` defaults (or the call at `run_plate5_volume.py:99`): `n_rows=14, n_cols=22, inner_row0=2, inner_col0=2` → **`n_rows=16, n_cols=24, inner_row0=1, inner_col0=1`** (offset becomes +0).
   - `image.py:174-175` (or the call at `run_plate5_volume.py:91`): `n_rows=14, n_cols=22` → **16, 24**.
   - Pass `n_rows=16, n_cols=24` explicitly to the lattice fit rather than letting `_fit_lines` infer geometry.
   - **Failure mode if skipped is silent, not loud:** the inner join at `register.py:61` would mismatch for *all four* dihedral ops, so `resolve_orientation` returns a confidently-wrong orientation rather than erroring.
   - **Assert the known orientation instead of trusting the inference.** A1 is at the **top right** on every plate in this project (§3, user-stated). Run 2 has only 6 blanks to score orientation with (Plate 5 had 22), and `register.py:65` neither reports its margin nor warns on a near-tie — so add an explicit check that the winning dihedral op corresponds to A1-top-right and **fail loudly on disagreement**. Also log the score margin between the best and second-best op; a thin margin is the signal that registration is guessing. Derive the op↔orientation mapping from the source rather than guessing it.

3. **Two `image.py` issues that A1-origin exposes:**
   - **Genuine bug — `image.py:86`:** `lo, hi = 25, (0.6 * (c1 - c0) / 22) ** 2 * np.pi * 4` hardcodes the literal `22` independent of `n_cols`. On a 24-column image this over-estimates pitch by ~9 % and inflates the max-area cutoff, letting merged colonies through the lattice fit. Should be `/ n_cols` — but `n_cols` is **not** in `_detect_blobs`'s signature (`image.py:64-67`, params are `g, enh, roi, invert`), so it must be threaded through.
   - **Needs checking, not assumed — `image.py:106`:** `m = 0.03 * (c1 - c0)` drops blobs within that margin of the ROI edge, commented (`:104-105`) *"the plated array is inset ~1 well from the wall."* That comment's premise is false for A1-origin, but the ROI comes from `_plate_roi(g, pad_frac=0.06)` — the **plate** bounding box shrunk 6 %, not the array bounding box. **Whether row-A / column-1 colonies actually fall inside the margin depends on the agar-to-array inset in the real image and is NOT established.** Measure it against `plate_view_3558C0B4.jpeg` before relaxing or conditioning the filter.

4. **Decide: fork the runner, or parameterize it.** `experiments/W019-echo-crispr-array/scripts/` contains only `build_reference_smf.py`, `compare_modalities.py`, `run_plate5_volume.py`, `run_sgatools_clone.py`, `spread_comparison.py` — **there is no run-2 entry point.** `run_plate5_volume.py` is hardcoded to 14×22 / B2-origin / within-plate volume split and cannot serve a 16×24 / A1-origin / between-plate run as-is. **Recommendation: parameterize the shared steps into a helper and add a thin `run2_volume_plates.py`**, rather than forking a near-duplicate — but make the call explicitly and say so in the note. §5.2 and §5.10 both assume whichever choice is made.

5. **Write a committed TransferReport parser** under `experiments/W019-echo-crispr-array/scripts/`. Requirements from §3: block-split on blank lines (do not read to EOF), join on `Destination Well` (row order differs), tolerate `Source Plate Type` differing between cherrypick and report, match `Sample Pick List File Name` on basename (it is a Windows path), treat empty `Transfer Status` as `NaN`, surface `Run ID` + `Date Time Point` + `Current Fluid Volume` depletion as provenance, and report `Actual Volume` with the "nominal echo, not metrology" caveat attached.

6. **Quantify both plates and add a `## 2026.07.17 - Run 2` H2 section to the real note.** Use **`plate_view_3558C0B4` = P1 (2.5 nL)** and **`plate_view_D4CA749C` = P2 (5 nL)** — EXIF-confirmed (§3), and the labelled JPEGs are unusable. Register against `P1_2p5nL_cherrypick_13strain.csv` / `P2_5nL_cherrypick_13strain.csv` via `read_echo_picklist` (it hard-requires columns `{"Destination Well", "Sample Name", "Transfer Volume"}`, `io.py:63` — the run-2 CSVs have all three). Note the run-2 blank count is only **6** (vs 22 on Plate 5), which weakens `resolve_orientation`'s agreement score (`register.py:65`, blanks-empty XOR plated-grew) — **watch for a near-tie between orientations; the code does not detect or warn about one.** Follow the note's conventions: H2 dated section at the bottom, images as `![alt](assets/images/019-echo-crispr-array/<name>.<ext>)` with an italic caption line, pipe tables in the canonical header form `| strain | ORF | 2.5 nL ± SD (n) | 5.0 nL ± SD (n) | Costanzo SMF ± SD |` ordered by 5 nL fitness ascending with `**BY4741**` last, and a closing `### Provenance` block. **Never delete superseded numbers** — banner them with a blockquote, as at L136-138.
   - **Output naming — pick a scheme and state it in the note.** Every existing artifact is `plate5_*` (13 CSVs in `results/`, ~30 files in the images dir). Two plates × two timepoints will collide without one. **Proposed: `run2_p1_2p5nL_t44_*` / `run2_p2_5nL_t44_*`, and `…_t50_*` for the second timepoint** — confirm with the user (§7.5) before generating dozens of files. **Use `t50`, not `t48`:** the re-image landed at 50.3 h (§3, §4), and a `t48` label would put a number in the filenames that never happened.

7. **Second timepoint: DONE and staged — quantify both timepoints with the same pipeline and compare.** ⚠️ **This item changed status on 2026-07-19: the images now exist** in `run2_2026-07-17/t50/`, at **50.3 h** (not 48 h), with segmentation-grade clean views for **both** plates (§3). Nothing is blocked on the user here. The four images to process are in the 2×2 table in §3. This is a within-plate growth-time control on physically identical plates (**43.7 h vs 50.3 h**), so it isolates colony age from every other factor — the strongest such control available, and the missing half of the note's "Growth / incubation time" section, which currently documents only a *planned, never-executed* 24/48/72 h timecourse. **Mechanics:** there is no time axis in the package (§2), so process each timepoint as an independent run and compare per-strain fitness, as was done for modality (r=0.95). Colony identity across timepoints is recoverable **only** via `(row, col)` after each image is independently registered — and if the two images resolve to *different* orientations the join silently pairs the wrong wells, so **assert that both timepoints resolved to the same `best_op` before joining.** If a timepoint column is wanted, note that `volume_assay_metrics` (`assay.py:109`) and `volume_position_confound` (`assay.py:70`) hardcode the column name `"volume_nl"` and cannot be reused for a timepoint axis without parameterization. Growth *rate* (differencing two timepoints, doubling time) does not exist anywhere and is new work.
   - **Where the images land: RESOLVED — `data/run2_2026-07-17/t50/`**, already populated and named `P{1,2}_{2p5,5}nL_view_t50_<hash>.jpeg`. Follow that convention for any future timepoint (`t<hours>/`).
   - **Still worth asking the user for future runs:** shoot each plate flat on the light panel, full-plate, **with the label edge in the same frame**. Both timepoints so far have needed a *separate* hand-held shot to capture the label, and both hand-held shots were unusable — so plate identity has twice depended on EXIF capture order rather than on anything visible in the usable image (§7.6).

8. **Batch correction, treating P1 and P2 as two batches** (the user's request). Batch is *absent*, not merely unimplemented: `grep -rn "batch" torchcell/sga/` returns **zero matches** across all nine modules — no config field, no column, no groupby, no `ScoreReport` dimension (`plate_id: str`, `models.py:132`, is free-text passed by the caller at `score.py:40`). Multi-plate aggregation does not exist; `normalize_plate` and `score_plate` are strictly single-table. Implement per-batch median centring as documented in the note (L95-98): `norm_batch_i = norm_i / med_{R∩batch b}(norm)`. **Caveat to state prominently: batch is confounded with volume — P1 = 2.5 nL, P2 = 5 nL** — so this shows whether batch centring *moves* anything, not a clean batch effect. The user explicitly said: don't necessarily pull in the older Plate 5; just P1 vs P2. Related structural point: each plate's `norm` is already divided by its **own** plate reference median (`normalize.py:133`), so cross-plate `relative_fitness` is only valid because each plate carries its own BY4741 — **nothing in the code enforces or checks that.**

9. **Upside-down incubation / dome-shape / circumference question (the user's hypothesis).** `IMG_4576` confirms the plates were incubated **agar-side up (inverted)**. The user's reasoning, restored verbatim in meaning from the previous note L69-70: inverting flattens the colony dome, and they want colony mass going to **circumference**, because **a larger 2-D circumference/area is easier to distinguish than height** — i.e. this is a claim about which *measurand* is more discriminable, not about big colonies vs small ones. It is directly grounded in the pipeline: our measured `size` is projected 2-D area, so **height is not measured at all**; a flatter colony reads **larger** (good for the assay) while a tall dome reads smaller. Add `IMG_4576` to the note documenting the orientation. Investigate via colony area vs circularity across the run, and whether the inverted run reads flatter/larger than Plate 5. **Also confirm it does not distort the fitness proxy:** if flattening scales all colonies by ~a constant factor, the WT ratio cancels it — that cancellation is the thing to check, not assume. *(The earlier revision added "possibly by some predetermined factor that could be characterised." That phrase has no antecedent in the previous note — treat it as an unattributed inference, not a user statement, unless the user confirms it.)*

10. **`tc-lit` sourcing of `kuzminSyntheticGeneticArray2016`** (the step-by-step SGA protocol, not previously mirrored locally). Use `TC_LIT_URL`/`TC_LIT_API_KEY` from repo-root `.env` via the `/lit-pull` skill and **verify `X-Artifact-SHA256`** on every download. Two targets: (a) the exact incubation/imaging time — we currently have only "~48 h at 26 °C" inferred from Kuzmin 2018's per-step durations, and Costanzo 2016's mirrored SI gives temperature (30 °C for nonessential-deletion × DMA; 26 °C for TS alleles) but **no explicit final-growth duration**, deferring to Baryshnikova 2010 / Tong 2004 which are not mirrored; (b) the exact **competition-correction** formula, so the explicit neighbour-size term flagged as a gap in the math section (note L79-85, sketched as `s_comp_i = s_sp_i / (1 + β·B̂_i)` with `B̂_i` the centred neighbour biomass) can be implemented rather than left as "absorbed by the spatial step."

11. **Cosmetic/robustness cleanups exposed by a 13-strain, two-plate run** — none blocking, all silent-wrongness rather than crashes:
    - `run_plate5_volume.py:122-123` hardcodes the printed header **"(of 11 placed each)"** (run 2 has 29–30).
    - `:176-177` prints `f"{n_ref}/12 genes"` and indexes `ref[["kuzmin_smf","costanzo_smf"]]` assuming both columns exist.
    - Docstring `:8` says `12-gene panel`; **title `:296` says `"12-panel: our CRISPR fitness vs published SMF"` — the string is `12-panel`, NOT `12-gene panel`.** A grep-and-replace on `"12-gene panel"` silently misses `:296`.
    - `assay.py:73-74` `continue`s unless `len(vols) == 2`, so any grouping variable that isn't exactly two levels gets a **silent no-op** confound check.
    - `viz.py:99` wraps the palette with `PLOT_PALETTE[i % len(PLOT_PALETTE)]` — 13 strains + WT + blank = 15 categories is under the 18-colour ceiling so this is fine today, but it is the wrap-silently path, not an error.
    - `viz.py:72,106` build row labels as `chr(ord("A") + i)` from `df["row"].max()`, i.e. block-relative — **correct for A1-origin**, and it was mislabelling the B2-origin Plate 5 case.

12. **PDF build for the note — see §8.** Budget for the svg→png step; there is a working converter invocation in the repo to model the helper on.

## 6. Refactor: `torchcell.sga` → `torchcell.wetlab`

**Rationale (the user's, preserved).** We are taking ownership of the SGAtools-derived code as an **embedded submodule of torchcell** — NOT a second package with its own `pyproject.toml`. One pyproject stays. On the name: **not "experimental"** — most experiments in this repo are computational, so that name misleads; **not "analytical"** — confusing; **"wetlab"** is right, because what distinguishes this code is that its inputs come off a bench.

**Feasibility: LOW RISK, and do it now, before any of it is committed** — but note §5.1 wants the `019` tree committed for provenance. Sequence the rename *first*, then commit, so the committed history uses the final name. `torchcell/sga/` is entirely untracked (VERIFIED), so there is zero history to preserve and **`git mv` will fail** (`git ls-files torchcell/sga` → empty) — use a plain `mv`.

**Packaging mechanism (verbatim from `pyproject.toml`):**

```toml
[build-system]
requires = ["setuptools>=69.0.2", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["torchcell*"]
```

Setuptools automatic discovery, flat layout, one glob. `include = ["torchcell*"]` globs *dotted package names*, so **`torchcell.wetlab` is picked up automatically — no config change needed**, provided `__init__.py` carries over (it will). `[tool.setuptools.package-data]` needs no edit since `wetlab/` is pure `.py` — but if a YAML/asset is ever added there it must be listed explicitly.

**To exclude it from the published wheel** (the user's optional ask), `packages.find` takes a sibling `exclude` applied after `include`:

```toml
[tool.setuptools.packages.find]
include = ["torchcell*"]
exclude = ["torchcell.wetlab*"]
```

**The trailing `*` is load-bearing** — without it only the top level is dropped and children still ship, giving a broken half-package. **Real consequence:** the directory keeps working for a source checkout or `pip install -e .`, but `pip install torchcell` then yields `ModuleNotFoundError: No module named 'torchcell.wetlab'`, and because the `019` scripts import it at module top level **those scripts become unrunnable for wheel users**. Exclude if wetlab is meant to be private/heavy-dependency; ship if it is part of the product surface. **UNRESOLVED — user decision (§7.3).**

**Call sites: 59 references to `torchcell.sga` across 14 files, of which 18 are executable imports (12 internal + 6 external).**

- **Internal, 41 hits in `torchcell/sga/`.** All 9 modules carry the repo 3-line frontmatter block at lines 1–3 (path comment, `[[dendron]]` link, GitHub `tree/main/torchcell/sga/...` URL); **all three lines change in all 9 files.**
  - **The 12 executable imports:** `__init__.py` :24, :30, :35, :41, :42, :43, :44 (7); `assay.py:20`; `normalize.py:32`; `score.py:26`; `viz.py:21`; `viz.py:137` (a deferred import inside a function — easy to miss).
  - **`models.py` has ZERO imports.** `models.py:13` is a **docstring line** — verbatim: ``to the on-plate wild-type. See `[[torchcell.sga]]` and CLAUDE.md provenance``. It still needs its `[[torchcell.sga]]` link updated, but **rewriting it as an import statement would corrupt the docstring.** (A prior revision of this note mislabelled it as an import.)
  - Also a docstring usage at `__init__.py:14`.
- **External Python, 7 hits in `experiments/W019-echo-crispr-array/scripts/`** — `compare_modalities.py:23` (block ends :31); `run_plate5_volume.py:29` (block ends :38), `:41` `from torchcell.sga.assay import shape_by_volume`, `:42` `from torchcell.sga.viz import colony_shape_by_volume, plate_heatmap, value_histogram`; `run_sgatools_clone.py:4` (docstring), `:34` (block ends :42), `:43` `from torchcell.sga.viz import (`.
- **Notes, 11 hits** — `notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md:21,29,47,60,356,362,373,493` and the scratch notes.
- **Zero hits** in config YAML, notebooks, `.github/`, `Makefile`, `torchcell.egg-info/SOURCES.txt`.

**Do NOT blanket-`sed` the string `sga`.** A repo-wide case-insensitive search hits ~100 unrelated files: `sgd`, `nsgaii` (Optuna sampler comments), the perturbation-type enum value `sga_kanmx_deletion`, SGAtools/SGA-the-assay in prose, and the filename `run_sgatools_clone.py` (an external tool's name — **keep that filename**). Scope every replacement to `torchcell.sga` / `torchcell/sga`. Two grep-matching test files are **false positives and must not be touched**: `tests/torchcell/verification/test_verification.py:74` (`"perturbation_type": "sga_kanmx_deletion"`) and `tests/torchcell/literature/test_provenance.py:52,107,110` (`path="software/sga.zip"`).

**Recommended sequence:**

1. `rm -rf torchcell/sga/__pycache__` — 9 gitignored `.pyc` files, already stale (`cpython-311` while `requires-python = ">=3.13"`); moving them carries wrong embedded module paths. Check `experiments/W019-echo-crispr-array/**/__pycache__` likewise.
2. `mv torchcell/sga torchcell/wetlab` (plain `mv`, not `git mv`).
3. Rewrite the 3-line frontmatter in all 9 modules.
4. Rewrite the 12 internal + 6 external real imports, **plus the two docstring references** (`models.py:13`, `__init__.py:14`).
5. Update the 11 notes references.
6. Decide ship-vs-exclude (§7.3), then commit (§5.1).

**Residual risk: there are no tests.** No test file imports `torchcell.sga` and no test filename contains `sga`, so the rename's only verification is import-time. Run a smoke check — `python -c "import torchcell.wetlab"` plus executing each of the three `019` scripts' import blocks. `torchcell.egg-info/SOURCES.txt` has no `sga` entries and `top_level.txt` is just `torchcell`; both regenerate on build, but re-run `pip install -e .` if an editable install is active.

**Loose end:** `notes/torchcell.sga.md` **does not exist**, so every `[[torchcell.sga]]` frontmatter link is already dangling. The rename is the natural moment to create `notes/torchcell.wetlab.md` with `dendron-cli note write --fname "torchcell.wetlab"`.

## 7. Open decisions — need the USER

1. ~~**Was run 2 supposed to fix the LCL1/SPH1 mistake?**~~ **ANSWERED 2026-07-19 — closed, no action.** The user: *"we know that these 2 strains are in but maybe shouldn't be... we are just still using them to develop the assay."* Run 2 is an assay-development run; the panel carry-over is accepted. Write it up as deliberate, not as an error. **Residual (not a question, a gate):** the canonical panel-12 genes **YIL174W and YLR104W remain unmeasured by us**, so nothing from Plate 5 or run 2 can feed the doubles design built on `experiments/010-kuzmin-tmi/results/optimized_doubles_setcover_panel12.csv` (the 11-double set-cover from `experiments/010-kuzmin-tmi/scripts/optimized_doubles_setcover.py`, notes at `notes/experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover.md` — **all three on `origin/main`, none in this working tree**). Correct the panel before the assay goes from development to production.
2. **Is the Echo clock ~1 h fast?** `Date Time Point` puts dispensing at 16:27–16:34 on 2026-07-17, but `IMG_4576` (independent device) shows both plates on the incubator shelf at 15:31:52 (§4). Confirm the instrument clock offset — if confirmed, 43.7 h stands; if the photo is actually of pre-warming blank plates, elapsed is ~42.7 h. Needed before any elapsed-hours number is published.
3. **Wheel packaging: ship `torchcell.wetlab` or exclude it?** Excluding silently breaks the `019` scripts for anyone who installed from a wheel (§6). Default recommendation is to ship.
4. **Is the volume-confounded batch test acceptable?** P1 = 2.5 nL and P2 = 5 nL, so "batch" and "volume" cannot be separated. Fine as a "does batch centring move anything" smoke test, or wait for same-volume replicate plates?
5. **Run-2 output naming.** The image drop location is now settled (`data/run2_2026-07-17/t50/`, already populated). Still confirm the *results* naming scheme `run2_p1_2p5nL_t44_*` / `run2_p2_5nL_t50_*` (§5.6) before dozens of files are generated. **Note it is `t50`, not `t48`.**
6. **Plate labelling protocol going forward.** Now a two-for-two pattern: at both timepoints the flat-on-the-panel shots were segmentation-grade and both hand-held labelled shots were unusable (two polarities in one frame). **Plate identity has therefore rested on EXIF capture order twice running, not on anything visible in the image being measured** — and at t50 there is no labelled photograph of P2 at all. That has been recoverable so far, but it is a single-point-of-failure: one out-of-order shot and two plates become unattributable. Confirm the extra handling to get the label edge into the flat frame.
7. **Second, independently scrambled layout for future replicate plates?** The user spotted that P1 and P2 share one layout, so replicate plates do not resample position and positional bias is reproduced rather than averaged (§3). For an assay-development run the shared layout is arguably a *feature* — it isolates volume perfectly. The question is **when to switch**: keep the shared layout while developing the measurement, or start varying it now so positional effects can be estimated? This is a picklist-generation change, not a pipeline change.

*(The second timepoint is no longer an open question — it has been taken and staged. A probe's earlier dissent about its value is recorded as a caveat in §4, not a veto, and is now moot: the actual separation came out at 6.58 h / +15 %, larger than the ~4.3 h that dissent was arguing about.)*

## 8. PDF build recipe for the note — RESTORED, do not lose again

The real note is rendered to `notes/assets/pdf-output/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.pdf`. **This section is now the only surviving record of the recipe** — it was rescued from the previous draft of this file before that draft was overwritten. Verbatim ingredients as recorded there:

> built via **pandoc + xelatex with a wrap-header**; **svg→png** conversion; **image-separation**; **`seqsplit`** for wrapping.

Confirmed context:

- The pipeline is `notes/assets/publish/scripts/bib_tex_pdf.sh`, invoked by VSCode task `.vscode/tasks.json:53`.
- Its header declares `\DeclareGraphicsExtensions{.png,.pdf,.jpg}` (`header-includes.tex:13`) — **`.svg` is not in the list**, which is why the swap is needed. Every `.svg` in `notes/assets/images/019-echo-crispr-array/` already has a `.png` sibling.
- **A converter invocation already exists in the repo** and is the obvious model for the helper: `notes/assets/publish/scripts/mermaid_pdf.sh:48` (`pdftocairo -svg …`) and **`:51` (`rsvg-convert -z 4 "${outdir}/${base}.svg" -o "${outdir}/${base}.png"`)**. (A prior revision of this note wrongly claimed no such converter existed anywhere — it does.)
- `seqsplit` genuinely appears nowhere in the repo outside the scratch note, so that piece of the wrap-header is undocumented in code. **Write the helper script and commit it**, rather than leaving the recipe in a throwaway note again.

## 2026.07.21 - Handoff, as written

## SCRATCH - Wet-lab colony-fitness assay: handoff to next session (2026-07-21)

**Throwaway handoff (do NOT `git add`; `scratch.*` is uncommitted). Delete once folded
into the real notes.** Hands the next session the full state of the wet-lab colony-fitness
assay work plus the double-mutant selection.

## 1. What this project is

A wet-lab CRISPR single-KO colony-fitness assay by ECHO acoustic dispensing onto 384 agar
plates, imaged on a backlight and quantified in-repo, to validate against published
single-mutant fitness (Costanzo 2016) and eventually score double/triple mutants. Two live
threads:

- **Assay (exp 019)** - image -> colony-size -> fitness pipeline. Code:
  `torchcell/sga/` (`image.py`, `register.py`, `normalize.py`, `score.py`, `viz.py`,
  `models.py`). Runner: `experiments/W019-echo-crispr-array/scripts/run2_volume_timepoints.py`.
  Write-up: `notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md` (the real
  note; 657 lines). PDF: `notes/assets/pdf-output/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.pdf`.
- **Doubles selection (exp 010, on origin/main)** - the 13 double mutants to build next.
  Note: `[[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]]`. A reordered
  PDF was built at `notes/assets/pdf-output/construction_validation_doubles.pdf`.

## 2. State of the assay (Run 2)

Run 2 = two plates, **P1 = 2.5 nL, P2 = 5 nL** (SAME randomized layout: 12 single-KO strains + BY4741 WT + 6 Blank_media controls), each imaged at **43.7 h, 50.3 h, 72.2 h** = six
(plate x timepoint) conditions of the same physical colonies. It is a SETTINGS SWEEP (find
best volume + growth time), not replicate batches; volume is confounded with plate, and the
shared layout means position is not resampled between plates.

**All six conditions are reliable**: every plate registers at identity (plate A1 = image
top-left), 6/6 empty no-cell controls, strain-structure Kruskal-Wallis H = 53-102.

Corrected conclusions (from the current pipeline):

- **Measure at 2.5 nL** - larger fitness range and better Costanzo agreement (best 2.5 nL /
  50 h, Pearson r = 0.81; 5 nL plates r = 0.29-0.50).
- **Image ~44-50 h, not 72 h** - colony footprint saturates (2.5 nL: 11.4% -> 18.7% -> 22.3%
  of the well cell; growth rate 1.10%/h then 0.16%/h) and the sick-strain signal erodes
  (CBF1/YJR060W 0.623 -> 0.715 by 72 h; published 0.590).
- **2.5 nL dropout ~24%** is the one cost to fix (a plating problem, not measurement).
- **Panel cannot validate** - 11/12 genes are published near-neutral (0.955-1.085); the
  double-mutant validation set is the answer.

## 3. The image pipeline (how it works now)

- **Preprocess** (in the runner, `_preprocess`): crop to the bright plate (drops the metal
  incubator shelf) + downscale to **PROCESS_WIDTH = 1400 px**. The full 12 MP fragments the
  lattice fit (colony texture + satellite blobs); 1400 px averages texture into solid blobs
  so detection/registration are robust. This was THE fix that made the "72 h overrun" reading
  go away - it was a detection failure at full resolution, not biology.
- **quantify_plate_image(grid_mode="lattice")** in `torchcell/sga/image.py`: colony lattice
  defines the region. Per-cell segmentation is **border/bright-referenced**: threshold each
  cell against the bright agar level (high percentile), not the cell median, then
  morphological CLOSING (bridge the domed-colony backlight glare band) + fill_holes -> full
  circumference. Median circularity is now 1.000. The old cell-median threshold produced
  half-moons (only the darker bottom of each colony).
- **Multi-colony 'M' flag**: a second colony-sized blob > 0.4*pitch away -> reject the cell
  (competing colonies both shrink). A glare fragment is bridged by closing, not counted.
- **Overlays**: green colony outline + cross, red for C/S flags, MAGENTA box for M cells
  (kept green for consistency with Plate 5; the earlier blue was reverted).
- **Registration**: `resolve_and_check` tries the 4 orientations and picks the max
  strain-structure H (all resolve to identity here); blanks are QC, not the gate.

Images: full-res 12.2 MP originals in `data/run2_2026-07-17/` -
`P{1,2}_{2p5,5}nL_view_t44.jpg`, `t50/*_view_t50.jpg`, `t72/*_view_t72_up.jpg`. The old
0.8 MP Photos-`derivatives/` previews were discarded ([[scratch.2026.07.20.200116-discard]]).

## 4. OPEN ISSUES (priority order)

1. **Green boundary looks jagged in the PDF (page 15 etc.) - display artifact, not a
   detection error.** Verified: at the overlay's NATIVE 1400 px the green outline hugs each
   colony's full circumference and the crosses are centred. The problem is that the 1400 px
   overlay embedded and shown small in the PDF renders the 1-px green outline thin/broken.
   **Fix candidates:** (a) draw a THICKER outline (2 px) in `_draw_overlay`
   (`torchcell/sga/image.py`); and/or (b) decouple resolution - process/measure at 1400 px
   but re-draw the overlay on a HIGHER-res copy (map grid nodes up), so the figure is crisp
   while the numbers stay from the robust 1400 px pass. (b) is the real fix; (a) is a quick
   mitigation.
2. **P2_t72 (5 nL, 72 h) top-row, frame-adjacent colonies** are still measured slightly small
   (a couple of them). The bright-percentile agar reference fixed most; the residual is at the
   plate frame. The user's idea: detect the plate's actual **6-sided chamfered rectangle** and
   bound the array to it, so frame shadow never intrudes on edge cells. Not yet implemented.
3. **Note reconstruction needs a read.** A greedy DOTALL regex accidentally deleted a chunk of
   the note (Plate 5 07.17 tail + run-2 preamble); it was reconstructed from CSVs and prior
   text. The **Plate 5 07.17 modality/spread prose (real note ~lines 403-419) is a
   reconstruction** - the figures and the r = 0.96 (n=13) modality result are exact, the
   wording is not verbatim. Review it. Damaged copy saved at `/tmp/_assay_damaged_backup.md`.
   LESSON: never batch-edit an untracked long note with span-matching regex; anchor to unique
   text or edit line-bounded blocks, and back up first.

## 5. Cosmetic/labelling state (all DONE this session)

- Condition axes show **volume + time** ("2.5 nL, 44 h"), not P1/P2 codes.
- Heatmaps + correlation map use **magma** (`viz.SEQUENTIAL_CMAP` = magma; was a warm ramp /
  cividis).
- Gene names render **systematic (common)** via `display_gene` (built from the reference CSV).
- WT-quality panel retitled "WT replicate spread (reproducibility)" with a caption noting the
  WT median is 1.0 by construction (the box shows single-colony precision).
- reference_bars error bar clarified as **SD across the six conditions** (not a standard error).
- Modality figure title is a statement ("Imaging modality barely changes scores"), font shrunk
  so it no longer bleeds into the y-label (`compare_modalities.py`).

## 6. SOPs decided

- **Imaging:** export the full-res HEIC as JPEG from Photos (File -> Export -> Export
  Unmodified Original), NEVER drag from the Photos window (that gives the 0.8 MP derivative).
- **Incubation orientation:** hold ONE orientation for the whole run (removes the agar-up/down
  variable).
- **Panel:** run 2 still carries the LCL1/SPH1 (YPL056C/YLR313C) mis-plating; canonical
  YIL174W/YLR104W were never built. Accepted for assay development; correct before production.

## 7. The double mutants to build next (exp 010, origin/main)

13 doubles = 8 coverage (reconstruct all 31 within-10 top-k triples) + 5 validation (add DMF
dynamic range + the only 3 significant Costanzo interactions in the panel). Buildable from the
10 on-plate genes; span all 10. Hubs: YDR057W/YOS9 (in 5 doubles), YPL046C/ELC1 (3). DMF SD is
a 4-colony sample SD, apples-to-apples with our colony SD. The reordered-by-gene-reuse PDF is
`notes/assets/pdf-output/construction_validation_doubles.pdf`. Note + CSV
(`construction_validation_doubles.{md,csv}`) are on origin/main; read from there with
`git show origin/main:<path>` (this checkout is on branch `paper/figures-fig1`, ~94 behind
main, with all the untracked assay work - do NOT flip the branch or create a worktree).

## 8. PDF build recipe (both notes)

pandoc -> xelatex. The build header lacks `.svg` graphics support and some Unicode glyphs, so
build a copy with image links swapped `.svg`->`.png` (every figure has a png sibling) and
problem glyphs sanitized (rho, ~, eps, etc.). Exact command used this session is in the assay
note's git-less history; the pattern:

```bash
cd notes
python -c "...swap .svg)->.png), sanitize unicode into .build_assay.md..."
pandoc --metadata link-citations=true -s .build_assay.md \
  -o assets/pdf-output/<name>.pdf --pdf-engine=xelatex --citeproc \
  --bibliography assets/publish/bib/bib.bib --metadata csl=assets/publish/bib/nature.csl \
  -V geometry:'top=2cm, bottom=1.5cm, left=2cm, right=2cm' \
  --include-in-header=assets/publish/tex-templates/header-includes.tex \
  --strip-comments --dpi=600
```

Run the assay pipeline from repo root:
`/Users/michaelvolk/miniconda3/bin/python experiments/W019-echo-crispr-array/scripts/run2_volume_timepoints.py`
(writes results/run2_*.csv, figures + overlays under notes/assets/images/019-echo-crispr-array/).

---
id: q037nkjnj529ipuja0phpa2
title: Cellpose Finetune Handoff
desc: ''
updated: 1784767194332
created: 1784767194332
---

## 2026.07.22 - Continuation instructions (fine-tune Cellpose + open TODOs)

Handoff for the 019 CRISPR fitness assay after a long session. Main note:
`[[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]]`. Branch/worktree:
`paper/figures-fig1`. Run everything from the worktree root with
`PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python ...`.

### Where we are (committed)

- **Segmenter** `torchcell/sga/cellpose_seg.py`: Cellpose-SAM + Otsu size-tightening
  (`tighten_size`, `tighten_grow_px=3`, removes the ~22% halo), colony-validity model
  (green/red-`M`/orange-`N`/purple-`C`), `multi_min_frac=0.5` gate, `edge_margin_frac`,
  grid **relaxation** (`relax_grid`, snaps rows/cols to Cellpose centroids), and a
  `precomputed_masks=` arg that re-runs grid/gel/assignment on cached masks WITHOUT a
  GPU forward pass (model can be None).
- **Benchmark** `experiments/019-echo-crispr-array/scripts/costanzo_kuzmin_comparison.py`:
  Fig 33 correlation now uses the **bootstrap-across-plates mean + SE** (Costanzo r=0.60
  p=0.037, Spearman 0.52; Kuzmin r=0.80; mean SE 0.031). Fig 34 SD = single 5nL/50h.
- **Bootstrap demo** `bootstrap_across_plates.py`: pooled colony SD/sqrt(n) SE (0.014) is
  ~2.2x too small vs bootstrap-across-plates SE (0.031).
- SD-method + bootstrap + Kuzmin-means (verbatim) documented in the main note and in
  `[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]`.

### The core unsolved problem: MISSED FAINT COLONIES (detection floor)

Generalist `cpsam` misses faint backlit colonies; contrast (CLAHE/flatfield) and cellprob
sweeps are exhausted (occupied flat ~281 on 2.5nL). P2_t72 is a separate PATHOLOGICAL
image (perspective distortion: row pitch 117-124 vs col ~113 vs NN 101; peak-detect finds
18 rows/26 cols with frame artifacts) -- do NOT keep tuning grid heuristics for it; the
48h re-image with square-on geometry is the real fix. **Decision: fine-tune Cellpose.**

### TASK 1 (main): fine-tune Cellpose-SAM on our plates

Rationale: the human-in-the-loop result (`cellpose_init`) reaches near within-human AP with
1-5 labelled images -- data-efficient because we start from pretrained `cpsam`.

**API** (from cellpose docs, verified):

```python
from cellpose import io, models, train
model = models.CellposeModel(gpu=True)
model_path, tr, te = train.train_seg(
    model.net, train_data=imgs, train_labels=lbls,
    test_data=timgs, test_labels=tlbls,
    learning_rate=1e-5, weight_decay=0.1, n_epochs=100, batch_size=1)
```

Labels = integer instance-mask arrays (or `image_masks.tif` / GUI `_seg.npy`);
`min_train_masks=5`.

**Grid-guided auto-labeling (the efficient path -- do this):** we KNOW the array grid
(`_fit_lattice`+`_relax_lattice` in `cellpose_seg`) and which wells are plated (the ECHO
picklist, read by `run2_volume_timepoints.read_echo_picklist`). Build training labels by,
for each DESIGNED-occupied well, placing an Otsu/intensity-thresholded colony mask at that
node (reuse `_tighten_instance`-style logic) -- this INCLUDES the faint colonies cpsam
misses. Steps:

  1. New script `experiments/019-echo-crispr-array/scripts/cellpose_finetune_prep.py`:
     for each of the 6 crops (`quant/cellpose_proc/run2_cellpose_best_crop_*.png`), start
     from the cached Cellpose masks, then for every occupied well with NO instance, probe a
     disk at the node; if intensity depression > threshold, add a thresholded colony to the
     label. Tile each plate into 512x512 windows -> `train/<name>.tif` + `<name>_masks.tif`.
     Hold out 1-2 windows as test.
  2. New script `cellpose_finetune.py` calling `train.train_seg` (params above), save to
     `experiments/019-echo-crispr-array/models/cpsam_echo_ft`. GPU via a `gh_*.slurm`
     wrapper (copy `gh_cellpose_recipe.slurm`; ONE sbatch per call, sandbox disabled).
  3. Wire the fine-tuned model: add `model_path` to `CellposeSegConfig` /
     `load_cellpose_model(pretrained_model=...)`; re-run the finalize + montage; compare
     occupied counts vs the generalist (target: recover the ~20 faint misses/plate).
  4. Sanity: also fixes alignment indirectly (more complete centroids -> better relax).

Fallback if auto-labels are noisy: hand-correct 3-5 windows in the Cellpose GUI
(`python -m cellpose`), save `_seg.npy`, train from those.

### TASK 2 (quick): isolate-detection test (was inconclusive -- Cellpose CPU too slow)

Re-run on GPU: crop the bottom-left of P2_t50 (two missed faint colonies, Fig 31) and run
cpsam on just that window (raw vs CLAHE). If isolation detects them, the miss is a
tiling/context effect and better tiling (`--tile_norm`/smaller tiles) may suffice before
fine-tuning. Scratch: `scratchpad/isolate_test.py`.

### TASK 3 (small, do next): note prose for Fig 33

The correlation paragraph in the main note (section "2026.07.22 - Benchmark vs published
SMF") still describes the SINGLE 5nL/50h version (Pearson 0.27). Update it to the
bootstrap-across-plates version (r=0.60, p=0.037; Spearman 0.52; y-error = across-plate SE
0.031; point = bootstrap mean over 6 plates). Then rebuild the PDF
(`bash notes/assets/publish/scripts/bib_tex_pdf.sh $PWD/notes/experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay.md $PWD/notes experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay`).

### TASK 4 (pending compute): refresh scores with relaxation

Job **1037** (`gh_cellpose_segmentation.slurm`) is queued behind the user's `019-expr-grid`
(~2h). When it runs it refreshes `run2_cellpose_fitness_by_condition.csv` with the relaxed
grid; then re-run `cellpose_error_analysis.py`, `costanzo_kuzmin_comparison.py`,
`bootstrap_across_plates.py` (numbers move only slightly; P2_t72 occ 323->319). If it never
runs, re-derive on CPU via `precomputed_masks` (see `scratchpad/rederive_color.py` pattern,
but also normalize+score).

### Gotchas

- GPU: `sbatch`/`squeue` need `dangerouslyDisableSandbox`; ONE sbatch per call; poll with a
  background `until squeue empty` loop. See memory `[[gilahyper-sbatch-sandbox]]`.
- Cellpose on CPU is very slow (model load alone minutes) -- use GPU/SLURM for real runs;
  `precomputed_masks` avoids Cellpose entirely for grid/score re-derivation.
- Cached masks in `quant/cellpose_proc/run2_cellpose_best_*` are ALREADY tightened -> pass
  `tighten_size=False` when re-deriving from them.
- P2_t72 is pathological; don't over-invest in its grid -- 48h re-image + fine-tune.
- Imaging next round: try DARK background, fixed square-on geometry (removes distortion at
  source).

---
id: 6a35txadolkpsn2czhyrzny
title: Dcell_training_compose_figure
desc: ''
updated: 1788478281564
created: 1788478281565
---

Composes `notes/assets/drawio/FigS-dcell-training.drawio` from the four true-size panel SVGs of [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]] plus a lettered placeholder box. Script: `experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure.py`.

## 2026.09.03 - Layout and export

- Two rows of two 88 x 52 mm panels (a, b; c, d) at 6-unit gaps, then (e) a dashed placeholder box (fontSize 8.3, prints at 6 pt) for the per-operation profiler breakdown of a DCell training step, which needs a `torch.profiler` run of `dcell.py` on a cluster GPU node. Whole figure 709 x 491 draw.io units = 180 x 125 mm.
- Panel letters fontSize 11.1 bold lowercase. `drawio_font_band.py --check` passes (one 8.3 label, five 11.1 letters).
- Export: `"/Applications/draw.io.app/Contents/MacOS/draw.io" -x -f pdf --crop -o paper/nature-biotech/figures/FigS-dcell-training.pdf notes/assets/drawio/FigS-dcell-training.drawio` gives a 181.0 x 126.3 mm page; `check-figures.sh` passes (within the 2 mm grace).
- The draw.io file is overwritten on every run; never hand-edit it.

## 2026.09.03 - White-cross layout

Author review: panel letters must never sit over a y-axis label or a neighbor's title. The script now uses the layout constants shared by all four composed SI figures: `COL_GAP = 12` (3 mm), `ROW_GAP = 22` (5.5 mm), `TOP_STRIP = 16`, letters at `(panel_x, row_top)` in the strip above each row, second column at `half_w + COL_GAP` (so the figure is 705 units wide, not flush to 709), and the placeholder box (e) spans the full width from x = 0 with its letter in the strip. Figure 705 x 539 units = 179.0 x 137.0 mm; exported PDF 179.6 x 137.6 mm; `check-figures.sh` and `drawio_font_band.py --check` pass.

## 2026.09.04 - Three rows: the placeholder becomes the CPU profile, plus a data-effect panel

Second author review. The layout is now a loop over three rows of two half-width panels: (a, b) as before at 52 mm; (c, d) at 50 mm, with d the redesigned stage table of [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]]; (e) `dcell_training_cpu_profile.svg` from [[experiments.006-kuzmin-tmi.scripts.dcell_training_cpu_profile]] and (f) `dcell_training_data_effect.svg`, both 48 mm. The placeholder cell and its style are gone. Same constants (`COL_GAP = 12`, `ROW_GAP = 22`, `TOP_STRIP = 16`, letters at `(panel_x, row_top)`). Figure 705 x 651 units = 179.0 x 165.2 mm; exported PDF 179.2 x 165.4 mm, under the 170 mm cap; `check-figures.sh` and `drawio_font_band.py --check` pass (six letters at 11.1, no other text cells).

## 2026.09.05 - Recomposed after the third author review

Panels c, d (hours per epoch; the later-day reruns as a range, [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]]) and e (the red stand-in line, [[experiments.006-kuzmin-tmi.scripts.dcell_training_cpu_profile]]) were regenerated at their previous sizes, so the layout and the script are unchanged: 705 x 651 units = 179.0 x 165.2 mm; exported PDF 179.2 x 165.4 mm; `check-figures.sh` and `drawio_font_band.py --check` pass.

## 2026.09.05 - Caption detail moved out of the paper

The full caption ran off the page, so the paper keeps a short one. The measured detail it carried is kept here verbatim (LaTeX source):

```latex
\textbf{DCell training on the trigenic interaction
task: one run, its checkpoint statistic, and its cost.} \textbf{a},~Validation Pearson~$r$ per epoch
for the single full DCell run (wandb \texttt{eni948by}, job 1922684, four GPUs; top axis,
wall-clock) and the stopped run without auxiliary losses (job 1921740; 10 epochs, 23~h,
$r=0.009$; partial, no result). Validation $r$ rose from zero over the first 100 epochs and
then fluctuated (mean 0.088, SD 0.028 over epochs 100--332, consecutive epochs differing by
up to 0.10), peaking at epoch 150. Open circles mark the three evaluations averaged in
Fig.~\ref{fig:ggi}d; the dotted line is their mean. \textbf{b},~Training loss (every 10 optimizer
steps) and validation loss (per epoch) of the same run on a log scale; dotted verticals mark
the three evaluations. The learning rate was constant. \textbf{c},~GPU-hours from the start of
training to the best validation epoch (left) and wall-clock hours per epoch (right; the
median over the run's epochs, log axis) for DCell (four GPUs), DANGO (two runs on the same
build, two GPUs each) and CGT (three replicate runs, four GPUs each); bars are the mean over
runs, open circles the individual runs, and whiskers the SEM where a model has more than one
run (\supptab{tab:dcell-training-cost}). \textbf{d},~The DCell speed-up work on a shared four-GPU
workstation as a table of stages: what each stage changed relative to the one before it,
its batch per GPU (8 loader workers; fp32 through stage 3, mixed precision from stage 4),
the resulting training samples per second, and, as bars, seconds per optimizer step from
the progress-bar logs (elapsed time divided by steps completed, an upper bound on the
steady-state step time). Stage numbers are the rows of \texttt{speedup\_stages.csv}. Arrows
mark the cumulative chain 1 to 5, each stage keeping every earlier change, measured in one
sitting; stages 8 and 9 are stage 5 with \texttt{torch.compile} (recompilation limit 64) at
batch 500 and 600. Rows 6 and 7 of the file, the stage-5 configuration rerun on later days
(row 7 with 12 loader workers), took 119 and 99~s per step (9 and 10 samples per second)
and are drawn not as bars but as the dashed range on the stage-5 row: the workstation was
shared with other jobs, so absolute step times drift from day to day by more than several
of the optimizations, and only stages measured in the same sitting are comparable. The
compiled stages (49 and 59~s per step, 40 samples per second) fall between the stage-5
chain value and its later-day reruns, so the measurements establish no gain from
compilation. A rewrite that spreads a stratum's per-term calls over four CUDA streams
(\texttt{dcell\_opt}), still one module call per term, ran at 13\% GPU utilization and
80--100~W of 300~W at 1~h~40~min per epoch, slower than the loop, and was not used. The
cluster run's 72~s per step at global batch 2,400 sits in the range of the workstation
measurements at batch 2,000--2,400 (40 samples per second).
\textbf{e},~Where one training step spends its time, measured on a CPU (Apple M1 Max, torch
2.14.0, fp32, batch 8; \texttt{dcell\_training\_cpu\_profile.py}), not on the cluster GPUs.
The panel is a stand-in, as its red line states: a \texttt{torch.profiler} run of the
training step on the cluster GPUs (gilahyper) is pending and will replace it. Shown is
the trained architecture rebuilt from the frozen DAG (20,613,037 parameters, equal to the
logged count), a synthetic batch in the training script's layout, one forward, loss,
backward, clipping, and AdamW step under \texttt{torch.profiler}, split into exclusive
phases; purple, the forward pass. The absolute times are CPU times and do not transfer; the
structure does: the step issues 644,603 leaf ops (67 distinct), of which 205,057 in the
forward, growing by about 13,600 per added strain because the gene-state gather indexes each
term separately for each strain in the batch, and on the CPU that gather (17\%) and the
2,655 BatchNorm calls (15\% of the step, most of the subsystem time) cost more than the
matrix multiplies (1.5\%). Each leaf op is one kernel launch on a GPU.
\textbf{f},~Best validation Pearson~$r$ (maximum over epochs) of the same DCell implementation on
the Kuzmin 2018-only build of \suppnoteref{note:dango-repro} (91,050 records; wandb
\texttt{biucpv7p}, job 1811673, 258 epochs, best at epoch 227) and on the experiment-006
build (332,313 records; \texttt{eni948by}, best at epoch 150); bar, mean over runs; open
circle, the run; no whisker because each build has one run. The two builds are split
separately, so the validation sets differ in membership as well as in size, and the 005 run
trained without auxiliary losses in fp32 at batch 256 per GPU where the 006 run used
auxiliary losses in bf16 at batch 600; the comparison is the only measured data-size
comparison for DCell, not a controlled ablation.
```

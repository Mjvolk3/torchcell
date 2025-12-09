---
id: bghm7kq9sn6sa0trswiycei
title: Summary
desc: ''
updated: 1762490882466
created: 1762490557716
---
Short version: the full-step profiler you and Claude built told you something very important: the *model step itself* is only ~1.6× slower in Lazy Hetero, so the 40× wall-clock slowdown over full DDP training must be coming from the *training loop plumbing* (dataloading, collation, GPU transfers, mask generation, DDP sync, etc.), not from the core message passing.

I’ll first condense where things stand, then propose a concrete, staged plan (including the “EmptyRegressionTask” idea) that you can actually run.

---

## 1. Where things stand (compressed summary)

**Empirical facts:**

* Real training (4×GPU DDP, string12 configs):

  * **DANGO**: ~400 epochs / 24 h.
  * **Lazy Hetero**: ~10 epochs / 24 h.
  * → ~40× slower per epoch in real life.

* Full-step profiler (single GPU, pre-saved batch, 50 iterations, forward+loss+backward+opt):

  * **DANGO** ≈ 25–26 ms / step

    * Forward ~18.5 ms (~72%)
    * Backward ~4.5 ms (~18%)
    * Optimizer ~2.4 ms (~9%)
  * **Lazy Hetero** ≈ 40–41 ms / step

    * Forward ~29 ms (~72%)
    * Backward ~9 ms (~22%)
    * Optimizer ~2.5 ms (~6%)
  * → Only ~**1.6× slower / step** using that baked batch.

So the giant gap (40×) is not explained by per-step model compute in that setup.

**Key architectural differences (from code + config context):**

* **Lazy Hetero model family** (`hetero_cell_bipartite_dango_gi_lazy.py` / `int_hetero_cell.py`):

  * Does a *dual forward* to compute wildtype vs perturbed graphs and then an **embedding difference**.
  * Uses **GPU mask generation** for multiple edge types per batch (STRING + regulatory + physical...), via `GPUEdgeMaskGenerator` in the trainer.
  * Batch is heavier: more fields per sample (perturbation indices, ptrs, masks, per-edge-type masks, etc.).

* **DANGO model family** (`dango.py` / `int_dango.py`):

  * Pretrain model runs on a single full cell graph; perturbations are handled mostly via *indexing* into that embedding.
  * Fewer edge types and (apparently) no per-batch per-edge-type mask generation of the same complexity.
  * Bigger batch size (64 vs 28 in your real configs); Lazy Hetero OOMs at 64.

**Important mismatch between profiler and real training:**

The full-step profiling scripts use:

* A **single pre-saved batch** (no DataLoader / collation work),
* **Single GPU**, no DDP,
* Same batch reused 50 times (no per-batch variation in perturbations, masks, etc.).

Actual training uses:

* **DataLoader + collate** on the fly,
* **Multi-GPU DDP**, frequent gradient sync,
* Different perturbation patterns per batch, hence potentially different masking / graph surgery each time.

So all the scary overhead is almost certainly in:

> *Data pipeline + per-batch graph/mask construction + DDP comms + memory effects*,
> not the raw GNN forward/backward alone.

---

## 2. Working hypotheses

Let me state them explicitly:

1. **Data pipeline hypothesis**
   Lazy Hetero’s batches are larger (more fields, more masks) and more complex to build. Collation + CPU→GPU copies + per-batch mask generation are much heavier than DANGO’s, and this dominates at scale.

2. **Mask generation hypothesis**
   `GPUEdgeMaskGenerator` (or similar logic in `int_hetero_cell`) is doing expensive `torch.isin` / indexing over large edge_index tensors *per batch* and for more edge types than DANGO.

3. **DDP scaling hypothesis**
   Lazy Hetero uses smaller effective batch size per GPU (28 vs 64), so:

   * more steps per epoch,
   * more frequent all-reduce of gradients,
   * more relative overhead from sync vs compute.

4. **Dual forward / embedding difference design**
   Running a full forward on `cell_graph` and then on `batch` every step (instead of packaging and reusing embeddings à la DANGO) adds *some* overhead but, based on the single-step profile, is not enough to explain 40× by itself. It might interact badly with DDP + masks, though.

We need to *measure* how much each contributes in the real Lightning loop, not in a hand-crafted profiler.

---

## 3. Plan: structured experiments with minimal code surgery

I’d structure the work into three phases:

### Phase A — Instrument pure Lightning plumbing (EmptyRegressionTask)

**Goal:** Measure the cost of “everything except the model” for DANGO vs Lazy Hetero, under the *real* training scripts, configs, DDP, and dataloaders.

**A1. Implement `EmptyRegressionTask` for each trainer**

You want this for both:

* `torchcell/trainers/int_hetero_cell.py`
* `torchcell/trainers/int_dango.py`

Pattern:

* New class `EmptyRegressionTask` in each file that:

  * Uses the **same dataloaders / dataset / transforms / collate** as the real RegressionTask.
  * In `training_step(batch, batch_idx)`:

    * Do **everything the real task does before the actual model call**:

      * If you normally move `batch` to device or call `.to(self.device)`, do it.
      * If you normally call `gpu_mask_generator` in the trainer, *do it*, because that’s likely expensive.
      * Any other pre-step manipulations (feature assembly, indexing) → keep them.

    * Then **do NOT call `self.model`**.

    * Instead, construct a trivial scalar loss on GPU and backprop it, so Lightning’s loop is realistic:

      ```python
      # pseudo-code
      dummy = torch.zeros((), device=self.device, requires_grad=True)
      loss = dummy * 0.0
      self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
      return loss
      ```

      Or if you want to eliminate backward too, you can also test a variant with no backward/opt, but I’d start with a real backward on a tiny graph to keep the Lightning machinery identical.

* The module should still define `configure_optimizers` etc., but optimizer will be trivial (no real params).

**A2. Add config toggles for “empty task” mode**

For each relevant Hydra config:

* `experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi_gh_085.yaml`
* `experiments/006-kuzmin-tmi/conf/dango_kuzmin2018_tmi_string12_0.yaml`

Add a switch like:

```yaml
trainer:
  task_type: "regression"  # or "empty"
```

and in `int_hetero_cell.py` / `int_dango.py` have a small factory that chooses between `RegressionTask` and `EmptyRegressionTask` based on that flag.

Alternatively, define explicit “debug” configs:

* `hetero_cell_bipartite_dango_gi_gh_085_empty.yaml`
* `dango_kuzmin2018_tmi_string12_0_empty.yaml`

that just swap out the task class via `_target_`.

**A3. Run DDP with EmptyRegressionTask for both models**

Use a Slurm script analogous to:

* `experiments/006-kuzmin-tmi/scripts/gh_hetero_cell_bipartite_dango_gi_lazy-ddp_085.slurm`
* whatever DANGO DDP script you already have for string12.

For each model:

1. Run **4-GPU DDP**, 1–2 epochs, `max_steps` = maybe 100–200, with EmptyRegressionTask.
2. Log:

   * Wall-clock time per epoch (or per 100 steps),
   * Steps/sec from Lightning logs.

Compare:

* `Empty DANGO` vs `Empty Lazy Hetero`.

Interpretation:

* If `Empty Lazy Hetero` is already ≫ `Empty DANGO` (e.g. 10× slower), then the bottleneck is clearly in data & pre-step logic (collation, mask generation, DDP comms on batch metadata).
* If `Empty` tasks are similar (~1–2×), then the extra 40× must come from the actual model + gradient graph interplay with DDP and multi-edge structure, and we need Phase B.

---

### Phase B — Time decomposition inside the trainer

Assuming Phase A points to the data/mask/DDP pipeline, we then refine.

**B1. Add fine-grained timers in `training_step` for Lazy Hetero**

In `IntHeteroCell.training_step`:

Rough structure:

```python
t0 = time.perf_counter()

# 1. batch preparation / move to device
...

t1 = time.perf_counter()

# 2. GPU mask generation & any pre-message-passing graph surgery
...

t2 = time.perf_counter()

# 3. model forward + loss + backward
...

t3 = time.perf_counter()

# 4. (optional) do nothing, Lightning will step optimizer outside
...

self.log_dict({
    "t_batch_prep": t1 - t0,
    "t_masks": t2 - t1,
    "t_model": t3 - t2,
}, ...)
```

Do the same for DANGO, but especially watch Lazy Hetero.

Run a **short 4-GPU DDP training** (real RegressionTask, not empty), 1 epoch, and inspect averages of these timers.

This will tell you exactly where the 40× is coming from:

* If `t_masks` dominates, the culprit is `GPUEdgeMaskGenerator` and total number of edges/edge types.
* If `t_batch_prep` dominates, collation + CPU→GPU copies + heavy batch structure is the issue.
* If `t_model` really is just ~1.6×, the message passing difference is confirmed as minor.

**B2. Compare steps per epoch**

Check:

* Number of training steps per epoch for DANGO vs Lazy Hetero (Lightning prints this).
* If Lazy Hetero has significantly more steps per epoch due to smaller batch size, multiply:

  [
  \text{epoch time ratio} \approx
  \frac{\text{steps}*\text{lazy}}{\text{steps}*\text{dango}} \times
  \frac{\text{time/step}*\text{lazy}}{\text{time/step}*\text{dango}}
  ]

and see how close this gets to the observed ~40×. This will help separate:

* **“per-step cost”** vs
* **“more steps per epoch”** vs
* **“extra unaccounted slowdown”** (e.g. DDP pathological behavior).

---

### Phase C — Architectural changes (only after measurement)

Once Phases A/B tell you *what* hurts, then we touch the architecture.

The obvious candidates:

1. **Package `cell_graph` into the batch** (your idea).

   * Change collation so each batch carries both wildtype and perturbed graphs or at least a handle to the static `cell_graph` embeddings.
   * In the model, compute wildtype embedding once (or even once per epoch) and reuse / index.
   * Goal: kill the dual forward and move Lazy Hetero closer to DANGO’s “single message passing, then indexing” pattern.

2. **Reduce per-batch mask work.**
   Depending on B1:

   * If GPU mask generation is dominant:

     * Precompute as much as possible offline,
     * Use edge-index filtering instead of boolean masks where possible,
     * Or collapse some edge types if biologically acceptable.
   * If CPU→GPU copy of masks is dominant:

     * Keep static masks on GPU and only tweak small “perturbed” indices per batch,
     * Or encode perturbations more compactly (indices + on-the-fly masking in kernel).

3. **DDP micro-batching and accumulation.**

   If Lazy Hetero is forced to use small batch_size (OOM at 64):

   * Consider gradient accumulation steps to restore an effective per-GPU batch size similar to DANGO, reducing the number of all-reduce calls per epoch.
   * That changes effective steps/epoch and can reduce the relative impact of communication overhead.

---

## 4. Concrete next actions (short checklist)

If you want an immediate to-do list:

1. **Add `EmptyRegressionTask` to `int_hetero_cell.py` and `int_dango.py`.**
2. **Add “empty task” configs** for both models (or a `task_type` toggle).
3. **Run 4-GPU DDP, 1 epoch with empty tasks**, log steps/sec and epoch time; compare DANGO vs Lazy Hetero.
4. **Add timers in `training_step`** (batch prep, mask generation, model) and run short real trainings to get averages.
5. **Compute predicted epoch time ratios** from step cost × steps/epoch and see how close to 40× you get.
6. **Based on which timer dominates**, choose one of:

   * (a) Focus on mask generation (optimize/remove), or
   * (b) Focus on packaging `cell_graph` into batch and avoiding dual forwards, or
   * (c) Focus on DDP / batch-size scaling tricks.

If you’d like, next step I can sketch the actual `EmptyRegressionTask` class skeletons for both trainers, plus the minimal Hydra changes, so you can drop them straight into TorchCell.

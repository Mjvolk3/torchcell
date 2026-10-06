---
id: zm4d3300ik1ovahwwx6otnc
title: Test_gpu_edge_mask_generator
desc: ''
updated: 1790979692729
created: 1790979692729
---

## 2026.10.01 - Masks against hand sets and a loop oracle on CPU

Test file: `tests/torchcell/models/test_gpu_edge_mask_generator.py`, target [[torchcell.models.gpu_edge_mask_generator]].

### Fixture

Five genes; physical edge_index `[[0,1,3,0,1,2,3,4],[1,2,4,0,1,2,3,4]]`, regulatory `[[2,4,0,1,2,3,4],[0,2,0,1,2,3,4]]`, plus a gpr hyperedge type the generator ignores. Device `cpu`.

### Expected values

- Incidence (self loop listed once): physical 0:[0,3] 1:[0,1,4] 2:[1,5] 3:[2,6] 4:[2,7]; regulatory 0:[0,2] 1:[3] 2:[0,1,4] 3:[5] 4:[1,6]; padded to width 3 with -1.
- Masks: {1} physical `[F,F,T,T,F,T,T,T]`, regulatory `[T,T,T,F,T,T,T]`; {0,3} `[F,T,F,F,T,T,F,T]`, `[F,T,F,T,T,F,T]`; {4} `[T,T,F,T,T,T,T,F]`, `[T,F,T,T,T,T,F]`.
- Both batch paths equal the concatenated per-sample masks and the plain-loop oracle on 60 seeded random batches.
- `get_memory_usage`: 20 int64 elements = 160 B, 15 bool = 15 B, in MiB.

### Findings

- `generate_single_mask` has no bounds check (line 420): -1 silently masks gene 4's edges; 5 raises a bare list IndexError.
- `generate_batch_masks` ignores `batch_size`; the vectorized path uses it (all-empty batch returns `batch_size` blocks, line 295; a mismatch fails inside `repeat_interleave`).
- An empty batch returns `{}` from the vectorized path but raises from the loop path's `torch.cat([])` (line 253).
- `.to(device)` moves only the base-mask buffers; `self.device`, `incidence_cache`, `incidence_tensors` stay on the construction device (shown with the meta device).

The device-mismatch branches (lines 227, 311, 334-348, 407) need two devices and are left uncovered on a CPU-only run.

## 2026.10.05 - Audit 2 corrections

- Reach: all four generator findings are latent. Only the 006 `test_*.py` diagnostic scripts use the generator; trainer use was removed in 53c257c22.
- The `repeat_interleave` message is anchored and marked torch-owned; `test_edge_type_with_no_edges` no longer requests the unused fixture.
- No test seeds the global RNG (the random-batch test uses its own `torch.Generator`), so no `fork_rng` fixture is needed.

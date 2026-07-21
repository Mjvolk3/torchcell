---
id: glekv7k9il5zoirgm7e3b2p
title: Image
desc: ''
updated: 1784674985413
created: 1784674985413
---

## 2026.07.21 - Grid-constrained segmentation: gel-boundary gate + one acceptance predicate

`quantify_plate_image` produces a gitter-style colony-size grid from a standardized plate photo.
This pass made the backlit (`grid_mode='lattice'`) path trustworthy by fixing why the overlay drew
boundaries that the numbers did not agree with, and by bounding detection to the physical gel.
Motivation and full analysis: [[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]]; the
segmenter is being superseded by Cellpose next ([[experiments.019-echo-crispr-array.cellpose-segmentation-plan]]),
so this is the classical baseline the deep-learning path must beat.

- The overlay `det` mask and the DataFrame had diverged (mask written for every blob, the size cull
  applied only to the numbers afterward), so noise/frame/empty-agar drew phantom boundaries. Now a
  single acceptance predicate (size, aspect, extent, near-node, inside-gel) gates BOTH `det` and the
  numbers, and `det` is clipped to `gel_mask & ~gash`.
- Added `_gel_polygon` (6-sided chamfered-rectangle gel boundary fit from the lattice; short
  chamfers on the two bottom corners per the load-chamfers-down SOP) + `_signed_dist`; detections
  outside the gel are rejected, straddlers flagged `E`. New params `gel_detect`, `edge_policy`,
  `node_tol`.
- keep-largest-central blob (not strictly nearest) so a noise speck no longer steals a cell; and a
  backlit-threshold repair (p90 agar, 2.5-sigma cut + 6-level floor) that captures faint and
  cell-filling colonies. Added `seg_method` ('threshold' | 'watershed') and `return_masks`.

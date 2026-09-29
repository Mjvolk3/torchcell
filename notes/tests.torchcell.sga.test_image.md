---
id: 9k4yxtdcilf5cvr19n39l4g
title: Test_image
desc: ''
updated: 1790649895488
created: 1790649895488
---

## 2026.09.28 - Colony-array image quantification on synthetic 3x4 plates (Phase 9)

40 tests on the plates built in `tests/torchcell/sga/conftest.py` (dark-field, backlit and a disk variant, pitch 60 px, with an off-lattice secondary blob, a U-shaped colony, a bar and a gash), every area, circularity, ROI and centroid derived in the docstring (45, 77, 189 and 37, 69, 193 px areas; the 577-px chamfered gel mask; `plate_roi` (14, 85, 23, 76); the 6.3 percent S fraction). Two values are pinned numerically and labeled as such: the 77-px watershed basin, because the sobel ridge of the 1-px-blurred step is a two-pixel tie broken inside skimage, and the 16 of 35 wells found on a plate rotated 6 degrees, alongside the behavioral assertion that fewer than 35 are found.

Findings, all six confirmed by the audit: `_rotate(v, t)` maps a direction at angle a to a - t, and `_estimate_angle` returns a itself, so `_rotate(cents, -theta)` at lines 491 to 492 (and `cellpose_seg.py` 227 to 228) doubles the tilt instead of removing it (row y-spread 20.99 to 41.93 px on a 0.05 rad lattice; the correct call is `+theta`); `_segment_watershed` (lines 394 to 399) yields an all-False mask on a cell whose upper 45 percent of intensities are equal, because `np.median` of the empty `cell > p55` is NaN and `float(nan) or 1.0` keeps the NaN; the inverted-branch tear threshold `g > min(255, 1.3 * agar_median)` (line 506) is `g > 255` once the agar median exceeds 196.2, so the S flag never fires on typical backlit plates; with `margin_frac=0.6` and `chamfer_pitch=1.3` the two bottom-corner grid nodes lie `(1.3 - 1.2) / sqrt(2) = 0.0707` pitch outside the gel polygon (signed distance -4.2426 px), carry E and are dropped under `edge_policy="drop"`; `det &= gel_mask` (lines 679 to 682) then strips those colonies from the overlay while the table keeps size 37 (0 of 37 pixels survive at the left corner and 7 at the right, where the coarse even-spacing fit puts the node column 3.66 px right of the true column), against the "one predicate" comment at 591 to 594; `_detect_blobs` never reads its `enh` argument (lines 84 to 142). Coverage 0% to 95%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

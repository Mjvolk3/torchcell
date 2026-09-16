---
id: nyc1302qh1sengjntppfywy
title: Drawio_doc
desc: ''
updated: 1789598729895
created: 1789598729895
---

## 2026.09.16 - Why the mxGraph writer is shared, and why the export check is not the exit code

Two figures of the 008 perspective emit draw.io XML directly, the network overlay
([[experiments.008-xue-ffa.scripts.ffa_network_overlay_drawio]]) and the epistasis-model
explainer ([[experiments.008-xue-ffa.scripts.epistasis_model_intuition_drawio]]). A second
copy of the writer would drift on exactly the parts a reader has to trust are identical:
the unit constant, the page size and the geometry element. Moving it out was verified by
re-generating the network figure and diffing: byte-identical.

### The export check cannot use the exit code, and cannot use the file's existence

Both are wrong, and they are wrong in opposite directions:

- `xvfb-run` returns 1 after a SUCCESSFUL export, because its own cleanup kill finds no
  process. So a non-zero exit says nothing.
- drawio-desktop returns 0 in cases where it wrote nothing.
- The output file is still sitting there from the previous export, so a failed run reports
  the size of the LAST good figure and everything downstream keeps using it.

That third one is what actually bit: the network figure kept reporting `176.4 x 107.9 mm`
for three runs after a change that made its export fail outright. The condition is that the
file be written DURING the call, which `export()` checks with an mtime stamp taken before
the run.

### Two draw.io shape traps

- The polygon shape is `mxgraph.basic.polygon`. A bare `shape=polygon` is not a shape
  draw.io knows: it silently falls back to the cell's bounding rectangle, so a ring of
  triangles exports as a stack of squares with no error anywhere.
- `polyCoords` must be formatted explicitly. A numpy float rendered through an f-string of
  a python list serializes as `np.float64(0.5)`, which draw.io cannot parse and drops.

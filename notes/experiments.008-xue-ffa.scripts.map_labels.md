---
id: ic5lsi7d57538vlafvk01ov
title: Map_labels
desc: ''
updated: 1789696990101
created: 1789696990101
---

## 2026.09.17 - Label placement shared by the reference-map panels

Extracted from [[experiments.008-xue-ffa.scripts.ffa_ipath_map]] so that
[[experiments.008-xue-ffa.scripts.ffa_kegg_map]] uses the same rules; the iPath panel
regenerated pixel-identical after the move (rsvg render, `ImageChops.difference` bbox None).

- `Ink`: every drawn element sampled onto a 0.5 mm grid, the route weighted 1 and the
  background 0.08. Queries: ink under a rectangle, ink along a segment.
- `place_label`: 16 directions at 4 leader lengths; a candidate is rejected if its plate
  leaves the map, overlaps a placed plate, covers a labeled node, sits on a leader or link,
  or its leader passes through a plate or crosses another leader or link. The least-inked
  survivor wins, with a small right-hand bias and a small cost per millimetre of leader.
- `Placer.group`: a group of labels placed in the best of all orders, committed together.
- `attachment_window` + `column_layout`: the attached-species column tried at every 1 mm
  window on both sides, scored by the ink under it plus the ink each link would cross,
  rejecting a window whose link passes through its own labels or that sits within 3 mm of a
  labeled node.

Each rule was added after a render showed the failure it prevents; the list is in the
2026.09.17 section of the iPath note.

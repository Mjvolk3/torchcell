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

## 2026.09.18 - Plates around the ink, and a per-label search narrowing

`plate_rect(px, yc, tw, th, sub_h)` is the one place a label's box is computed, used by
`place_label` and by `column_layout`, so the collision rectangle and the drawn rectangle
cannot drift apart. It pads `INK_H * th + sub_h` evenly by `PAD`, where `INK_H` is the
fraction of the em that a mixed-case line with digits actually covers and `INK_DY` shifts
the box up for the caps that reach above the x-height an SVG `dominant-baseline="middle"`
centers on. Padding the em box instead put a band of white under every label and none
beside it.

`place_label` takes `sub_w` and sizes the plate on `max(label width, sub width)`. A second
line set in the smaller size can be the wider one; sized to the first it prints outside its
own box, which only became visible once the plates were stroked.

`Placer.group(..., pref={name: (directions, leads)})` narrows the search for one named
label. A review that asks for one label to move is answered by giving that label a smaller
candidate set, not by placing it: the ink search still picks among what is left, and the
narrowing lives in the calling script.

Plates are drawn with `PLATE_RADIUS` 0.55 mm and a `PLATE_STROKE` `#666666` edge at
`PLATE_WIDTH` 0.12 mm. On a drawing this dense an unstroked white patch reads as a gap in
the network rather than as something placed over it.

## 2026.09.18 - Review round 7: text placed by baseline, plates around the real ink

`dominant-baseline="middle"` centers the x-height, so a label with capitals sat high in its plate and a two-line plate left a band under its gene line. Now `ink(text, em)` gives each line's glyph box from Arial's cap height (0.716 em) and descender (0.212 em, only when the text has one of `gjpqy,;`), `line_stack` stacks the lines with `LINE_GAP` 0.6 mm, `plate_rect(px, yc, tw, label, sub)` pads that stack, and `baselines(yc, label, sub)` returns where each `<text>` goes. No `dominant-baseline` on plate text anymore. `place_label` takes `sub=` (the text) instead of `sub_h=`; `column_layout` sizes for the tallest possible plate ("Xy" over "Xy").

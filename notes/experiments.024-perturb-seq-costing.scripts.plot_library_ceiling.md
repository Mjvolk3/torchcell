---
id: tt8i0dghgs5wrrzrjgt1s2g
title: Plot_library_ceiling
desc: ''
updated: 1787625568983
created: 1787625568984
---

## 2026.08.24 - What one yeast transformation can actually cover

`experiments/024-perturb-seq-costing/scripts/plot_library_ceiling.py`

Sec. 4.7 stated the genome-scale conclusion as one arithmetic line. A reviewer
asked to see the whole surface, and the request was right: the interesting
content is WHERE the ceiling is crossed, not that it is crossed at the genome.

The two routes it keeps apart, because conflating them is what makes this
confusing:

- **Cloned combinations** need one clone per COMBINATION, so the requirement
  grows as `C(T, k)` and explodes. Pairs cap the panel at 258 genes at 1e6
  clones and 816 at 1e7; triples at 59 and 126.
- **Co-transformed singles** need one clone per GUIDE, so the requirement is
  `30T` and does not grow with `k` at all: 180,000 clones cover all 6,000 genes,
  inside one transformation, at any plex.

That contrast is the argument for route B of Sec. 7.1 and was previously only
implicit. Renders Fig. 12.

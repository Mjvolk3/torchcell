---
id: mnxfm2wrrsmhizf84w5zwa3
title: Remap_010_checkpoints
desc: ''
updated: 1788318036474
created: 1788318036474
---

## 2026.09.01 - Making the 010 Checkpoints Loadable by Current Code

`EquivariantPerturbationTransform` became a ModuleList of blocks after the 010
runs, so the 010 checkpoints store `perturbation_transform.cross_attn.*`,
`ffn.N.*`, `norm1.*` and `norm2.*` while the current class also expects
`cross_attn_layers.0.*` and friends. The class keeps the old names as aliases
onto element 0, so its own state dict carries BOTH spellings and a strict load
needs both. This adds the ModuleList spelling and keeps the alias.

Only needed for the stock eval path.
[[experiments.010-kuzmin-tmi.scripts.score_010_checkpoints_directly]] loads the
originals with `strict=False` and does not need a remapped copy. The stock eval
path is blocked anyway by the `perturbation_type` schema change, so this script
documents the rename rather than unblocking anything on its own.

Findings: [[experiments.010-kuzmin-tmi.additive-baseline-analysis]]

---
id: t2mb36tq00kh9k1or69ilir
title: Create_ffa_bipartite_network
desc: ''
updated: 1786496498269
created: 1786496498269
---

## 2026.08.11 - Make the Network Figure Runnable Off the Original Mac

This script builds the TF/gene/reaction/FFA bipartite network figure for 008. It was
authored on the M1 Mac and its results path was an absolute `/Users/michaelvolk/...`
literal, so it could not run anywhere else -- on GilaHyper it failed before drawing
anything. The path now resolves from `EXPERIMENT_ROOT`, which is the repo-wide convention
and what every other experiment already uses.

The motivation is that the FFA analysis moved to GilaHyper: the trajectory work
([[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]]) reuses this experiment's
loaders and needs the whole 008 script set to be machine-portable, not just the new files.

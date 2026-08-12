---
id: 9j50eh8k9f3v6digq5cwwc6
title: Identify_ffa_reactions
desc: ''
updated: 1786496519981
created: 1786496519981
---

## 2026.08.11 - Make Reaction Identification Runnable Off the Original Mac

Identifies the Yeast9 reactions that produce or consume the measured FFA species, feeding
the network figures. Its results directory was a hardcoded `/Users/michaelvolk/...` path,
so it only ran on the Mac it was written on; it now resolves from `EXPERIMENT_ROOT`.

Part of making the whole 008 script set machine-portable so the FFA analysis can continue
on GilaHyper -- see [[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]].

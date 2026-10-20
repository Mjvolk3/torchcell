---
id: yqirvhtz0tuirpngfiobmg5
title: R3_host_predictions
desc: ''
updated: 1791806727335
created: 1791806727336
---

## 2026.10.12 - Every arm's host prediction on every Bioscreen medium

The trainer writes, per config and fold, the seed-mean host prediction over its host
records (the public IC30 anchors, the ex21 titration, the ex23 combination plate and the
three isobole grids) to `$DATA_ROOT/experiments/040-inhibitor-synergy-wetlab/
predictions/<sweep>/<config>_fold<k>_seed<fold_seed>_host.npy`, in the trainer's record
order and without the records beside it. `r3_host_predictions.py` assembles the same
records (once per distinct source set, because every IC30-dosed source adds its anchors:
29 with Vanacloig alone, 103 with Hoepfner) and joins each prediction to its medium: run,
compounds, doses in mM and in g/L (read back from `wetlab_wells.csv` through the medium
key the trainer builds), the observed fitness and growth call, the split, and whether the
arm trained on that record (`trained_on`: anchors always; ex21 unless `host_train` is
`anchors`; never ex23 or the isoboles).

Output `results/host_predictions.csv`, 54,780 rows over 116 configs on 2026.10.12. The
figure gate's panels j, k and l read it: the single-source held-out-protocol arm
(`r3_van_e150`) is the model on the wet-lab panels, the anchors-only arm
(`r3w2_van_e150_anchors`) its zero-shot twin. What they show is that the host head is
nearly flat in dose: the zero-shot head predicts the anchors' assumed fitness of 0.70
everywhere, the fitted head about 0.5 everywhere, so the model calls every isobole cell
alive and predicts the same fitness at one, two and three inhibitors.

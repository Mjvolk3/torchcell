---
id: am7kv1m0mzuhc4h4f3x92je
title: Calmorph_moving_features_conf
desc: ''
updated: 1791623046054
created: 1791623046054
---

## 2026.10.10 - The moving CalMorph features as the morphology target

Writes `conf/calmorph_moving_features.yaml`, a Hydra fragment (`# @package _global_`) that drops the 162 quiet CalMorph features (replicate reliability below 0.5 in `results/morphology_feature_ceiling.csv`, from `morphology_noise_ceiling.py`) plus the 3 degenerate ones from the global head, and sets the head's `output_dim` to the 116 kept features. A feature moves when its knockout variance is at least twice the wild-type replicate variance; the ridge from the measured proteome reaches 0.280 on these against 0.126 on the quiet ones, so the quiet features are mostly replicate noise. Used by `conf/cgt_morph_v24.yaml`. Also writes `results/calmorph_moving_features.json` with both lists.

---
id: neg0voe80zupgdc9jv6un8q
title: Generate_triple_combinations_inference_3
desc: ''
updated: 1781028999819
created: 1781028999819
---

## 2026.06.09 - Build the Inference-3 Candidate Triple Set Sized for a Statistically Powered Improvement Test

This script exists to produce a larger, more permissive pool of constructible three-gene combinations than inference_2, deliberately relaxing the fitness filters so the resulting set is big enough to support a formal statistical test of the monotonic-improvement claim rather than a hand-picked few. The scientific framing shifts from "every gene must strictly improve" to "at least one triple shows significant monotonic fitness improvement," validated downstream with a Jonckheere-Terpstra test, and the thresholds here are chosen to give that test adequate power.

### Why these choices matter

- Relaxed thresholds: all singles > 0.90 with max > 1.04, and all doubles > 0.90 with a fixed max > 1.08 cutoff (the 0.04 gap targets roughly 96% JT power at n=8 and near-100% at n=16).
- Replaces inference_2's gap-based doubles rule (max(doubles) > max(singles) + 0.03) with a fixed absolute threshold, which is simpler and admits far more candidates.
- Retains the same Costanzo2016 SMF source, essential-gene exclusion, TMI removal, Parquet streaming, and `--plot-only` replot path as inference_2 so the two datasets differ only in their filtering criteria.

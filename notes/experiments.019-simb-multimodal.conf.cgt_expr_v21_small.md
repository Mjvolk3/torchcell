---
id: 8ucv3tw99nczpl11hu975nr
title: Cgt_expr_v21_small
desc: ''
updated: 1791241001624
created: 1791241001624
---

## 2026.10.05 - The small trunk: ProtT5 alone, width 90, six layers

The config header carries the design. In one paragraph: the 2026-10-04 audit measured that
ProtT5 alone is at or above every composite embedding, that 85 percent of the v19 model's
6.68 million parameters sit in the linear layer over the 3,328-d stack, and that wall time
per epoch does not depend on parameter count. The GilaHyper benchmark of 2026-10-05 added
that card memory does not either (8 to 10 GB per run at every shape) and that depth and
width do not change the epoch time at zero workers. So the only scaling-down that is free
is the input: ProtT5 alone, 1.45 million parameters, width and depth unchanged from v19 so
the arms stay comparable to that round. The default arm is the single-head expression model
with the masked objective off; the S_* arms in `gh_expr_008_arm.sh` each change one thing
(masked objective restored, sigmoid-gated operator, two-hop propagation, operator dropout
off, the four-part stack, the proteome head). Hypothesis (untested until the round reads
out): the ProtT5-only trunk reaches the v19 single-head level on these rows.

---
id: oe8qr09nxwfjdurm4uep2fx
title: '43'
desc: ''
updated: 1729531861734
created: 1729524650965
---

## 2024.10.21

- [ ] Figure out why diffpool models are outputting nans. → Really unsure where this is coming from. Just general training instabilities. Trying lots of different regularization techniques.
- [x] Consider gradient accumulation for stability → [lightning grad accumulation](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) → Add optional gradient accumulation. Had to add to manual since we are using manual optimization with lightning.
- [x] Add self loops. → instead of using function `add_self_loops` or `add_remaining_self_loops`, we just add ones to the diagonal since we are already in dense representation.
- [ ] Add optional learning rate scheduler.
- [ ]

***

- [ ] in `IndexSplit` changes `indices` to `index`...

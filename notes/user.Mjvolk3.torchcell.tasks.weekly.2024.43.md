---
id: oe8qr09nxwfjdurm4uep2fx
title: '43'
desc: ''
updated: 1730424151144
created: 1729524650965
---

## 2024.10.21

- [x] Figure out why diffpool models are outputting nans. → Really unsure where this is coming from. Just general training instabilities. Trying lots of different regularization techniques. → regularization helps
- [x] Consider gradient accumulation for stability → [lightning grad accumulation](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) → Add optional gradient accumulation. Had to add to manual since we are using manual optimization with lightning.
- [x] Add self loops. → instead of using function `add_self_loops` or `add_remaining_self_loops`, we just add ones to the diagonal since we are already in dense representation.
- [x] Add optional learning rate scheduler. → added but it never even comes into effect since nans effect us earlier.

## 2024.10.23

- [x] Investigate the proposed dense loader found on github branch. → need additional transform
- [x] Implement single dense sparse dense Diffpool
- [x] dense loading sizes with temp code show matching sizes `adj` compared to `edge_index` [[221008|dendron://torchcell/scratch.2024.10.23.221008]] → for now we can try to move ahead with this to see if training works.

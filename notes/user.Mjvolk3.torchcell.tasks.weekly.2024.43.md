---
id: oe8qr09nxwfjdurm4uep2fx
title: '43'
desc: ''
updated: 1729712793104
created: 1729524650965
---

## 2024.10.21

- [ ] Figure out why diffpool models are outputting nans. → Really unsure where this is coming from. Just general training instabilities. Trying lots of different regularization techniques.
- [x] Consider gradient accumulation for stability → [lightning grad accumulation](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html) → Add optional gradient accumulation. Had to add to manual since we are using manual optimization with lightning.
- [x] Add self loops. → instead of using function `add_self_loops` or `add_remaining_self_loops`, we just add ones to the diagonal since we are already in dense representation.
- [x] Add optional learning rate scheduler. → added but it never even comes into effect since nans effect us earlier.
- [ ] Consider `global_add_pool`, or `global_mean_pool`

- [ ] Check losses can be back prop.


## 2024.10.22

## 2024.10.23

- [ ] Investigate the proposed dense loader found on github branch.
- [ ] Implement single dense sparse dense Diffpool
- [ ] Implement single Dense DiffPool


***

- [ ] in `IndexSplit` changes `indices` to `index`...

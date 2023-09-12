---
id: mrcmzr371c14cekrqc5038c
title: Cell
desc: ''
updated: 1694206770971
created: 1692304235385
---

## CellDataset with DMFCostanzo2016LargeDataset is Slow

Progress bar steps

1. File check on cell raw
2. File check on costanzo2016 raw
3. File check on costanzo2016 processed
4. ⏳ Process loop for merging in `CellDataset` (**Time Sink**)
5. ⏳ Save `data_{#}.pt` in processed dir (**Time Sink**)

It might be possible to speed up (4.) but (5.) is already using multithreading on saving.

```bash
(torchcell) michaelvolk@M1-MV torchcell % python src/torchcell/datasets/cell.py                                                                                                       22:12
Checking files: 100%|██████████████| 1/1 [00:00<00:00, 11748.75it/s]
Processing...
Done!
Checking files: 100%|██████████████| 4/4 [00:00<00:00, 5813.31it/s]
Checking files: 100%|██████████████| 13295364/13295364 [07:42<00:00, 28750.03it/s]
Processing...
100%|█████████████| 13295364/13295364 [3:36:51<00:00, 1021.82it/s]
13294910it [2:32:18, 1454.81it/s]
Done!
```

## Merging Dataset Issues

- I think the cleanest way to avoid merging all together is first to just select the largest non-overlapping datasets. This way each data instances in the dataset can be a single graph and a single label. We could add the other labels associated with duplicate graphs to the same instance but then we have to loop over all graphs to find such instances. This seems like a problem that will inevitably show, but we can start like this and consider functions like `reduce` to join up duplicate graphs.
- In the case of `dmf` data we also have `smf` available on all of the instances for each of contributing genes, but this would add two more graphs to each of the data instances, or I guess it could return three graphs. This starts to get confusing. I think that each data instance should only return one graph with possibly multiple labels.

---
id: bbhxpkj59g5grb4wboz7we2
title: '044046'
desc: ''
updated: 1738406466971
created: 1738406448213
---

Getting metabolism genes.

```python
  # Get indices for each split
  all_pert_indices = (
      perturbation_subset_data_module.index.train
      + perturbation_subset_data_module.index.val
      + perturbation_subset_data_module.index.test
  )
  train_pert_indices = perturbation_subset_data_module.index.train
  val_pert_indices = perturbation_subset_data_module.index.val
  test_pert_indices = perturbation_subset_data_module.index.test

  # Filter for genes in GEM
  in_gem = {
      k: v
      for k, v in dataset.is_any_perturbed_gene_index.items()
      if k in gem.gene_set
  }

  # Calculate total dataset instances with GEM genes
  total_dataset_indices = list(range(len(dataset)))
  total_in_gem = [
      idx for idx in total_dataset_indices if any(idx in v for v in in_gem.values())
  ]
  percent_total_in_gem = len(total_in_gem) / len(total_dataset_indices) * 100

  # Get instances that contain GEM genes for each split
  in_gem_in_pert_train = [
      idx for idx in train_pert_indices if any(idx in v for v in in_gem.values())
  ]
  in_gem_in_pert_val = [
      idx for idx in val_pert_indices if any(idx in v for v in in_gem.values())
  ]
  in_gem_in_pert_test = [
      idx for idx in test_pert_indices if any(idx in v for v in in_gem.values())
  ]
  in_gem_in_pert_all = [
      idx for idx in all_pert_indices if any(idx in v for v in in_gem.values())
  ]

  # Calculate percentages
  percent_gem_over_pert_train = (
      len(in_gem_in_pert_train) / len(train_pert_indices) * 100
  )
  percent_gem_over_pert_val = len(in_gem_in_pert_val) / len(val_pert_indices) * 100
  percent_gem_over_pert_test = len(in_gem_in_pert_test) / len(test_pert_indices) * 100
  percent_gem_over_pert_all = len(in_gem_in_pert_all) / len(all_pert_indices) * 100

  print(f"Total dataset stats:")
  print(
      f"Total instances with GEM genes: {len(total_in_gem)} / {len(total_dataset_indices)} ({percent_total_in_gem:.2f}%)"
  )
  print("\nSplit stats:")
  print(
      f"Train instances with GEM genes: {len(in_gem_in_pert_train)} / {len(train_pert_indices)} ({percent_gem_over_pert_train:.2f}%)"
  )
  print(
      f"Val instances with GEM genes: {len(in_gem_in_pert_val)} / {len(val_pert_indices)} ({percent_gem_over_pert_val:.2f}%)"
  )
  print(
      f"Test instances with GEM genes: {len(in_gem_in_pert_test)} / {len(test_pert_indices)} ({percent_gem_over_pert_test:.2f}%)"
  )
  print(
      f"All split instances with GEM genes: {len(in_gem_in_pert_all)} / {len(all_pert_indices)} ({percent_gem_over_pert_all:.2f}%)"
  )

  ##
```

---
id: bmnetajrmhj9a5z8a4os6a8
title: Experiment_dataset
desc: ''
updated: 1711303072790
created: 1711155503944
---

## 2024.03.24 - Optional Multiprocessing on Batch

On M1 this slowed things down a little, but we might be able to see more performance improvement on `Delta` due to IO/processing tradeoff.

```python
# Outside of class
def extract_systematic_gene_names(genotype):
    gene_names = []
    for perturbation in genotype.get("perturbations"):
        gene_name = perturbation.get("systematic_gene_name")
        gene_names.append(gene_name)
    return gene_names


def process_batch(batch):
    # Assuming `extract_systematic_gene_names` is now adapted to work with
    # data directly (e.g., a standalone function) and returns gene names.
    return [
        extract_systematic_gene_names(data["experiment"]["genotype"]) for data in batch
    ]


class ExperimentDataset(Dataset, ABC):
...
  def compute_gene_set_parallel(self, batch_size=int(1e5), num_workers=10):
      gene_set = GeneSet()
      log.info("Computing gene set in parallel...")
      data_loader = CpuExperimentLoaderMultiprocessing(
          self, batch_size=batch_size, num_workers=num_workers
      )

      # Process each batch in parallel as it is generated
      with Pool(processes=num_workers) as pool:
          for batch in tqdm(data_loader, total=len(data_loader)):
              async_result = pool.apply_async(process_batch, (batch,))
              gene_names_list = (
                  async_result.get()
              )  # This should be a list of gene names.

              # Option 1: Flatten the list before updating
              flat_list = [
                  gene_name for sublist in gene_names_list for gene_name in sublist
              ]
              gene_set.update(flat_list)

      return gene_set
```

Takes 20 min without multirpocess on the batch

```bash
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/torchcell/data
sets/scerevisiae/costanzo2016.py
INFO:torchcell.data.experiment_dataset:Computing gene set in parallel...
INFO:torchcell.data.experiment_dataset:Computing gene set in parallel...
 19%|█████████████████████▍                                                                                            | 39/208 [07:18<34:51, 12.37s/it]
```

---
id: bnqyg8l1541odnjkoondwsp
title: Sgd
desc: ''
updated: 1710646938828
created: 1695170292621
---

## Yeastmine Data Comparison to SGD Backend API

Does the yeastmined data have the pvalues and the sga interaction scores? Looks like for trigenic they are contained in a "note" field... you've got to be kidding me... populated in a "note" field... and for they don't look populated for digenic.... they are populated for Costanzo 2016 in an "alleles" field, but they are not populated for 2010... This data for networks is probably better pulled from the original data, but then there is potential confliction `MultiDiGraph` and experiments.

## 2024.03.16 - Rebuild SCerevisiaeGraph Failed

```bash
█| 50/50 [01:22<00:00,  1.65s/it]
100%|████████████████████████████████████████████████████████████████| 50/50 [00:41<00:00,  1.22it/s]
  0%|                                                                         | 0/50 [00:00<?, ?it/s]ERROR:root:Saved unexpected response to: /var/folders/t3/hcfdx0qs0rsd9bm4230xv_zc0000gn/T/tmpb4mt21rf.html
ERROR:root:Unexpected content type: text/html. URL: <https://www.yeastgenome.org/backend/locus/YDR322C-A/sequence_details>
  0%|                                                                         | 0/50 [00:04<?, ?it/s]
Traceback (most recent call last):
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 269, in <module>
    main_get_all_genes()
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 265, in main_get_all_genes
    asyncio.run(download_gene_chunk(chunk, create_gene, False))
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/asyncio/base_events.py", line 653, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 252, in download_gene_chunk
    await download_genes(chunk, create_gene_fn, validate_flag)
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 236, in download_genes
    await asyncio.gather(*tasks)
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 213, in process_gene
    await gene.fetch_data()
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 69, in fetch_data
    await self._data_task
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 79, in download_data
    self._data["sequence_details"] = await self.sequence_details()
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 163, in sequence_details
    data = await self._get_data(url)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/graph/sgd.py", line 137, in _get_data
    raise ContentTypeError(error_message)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ClientResponseError.__init__() missing 1 required positional argument: 'history'
```

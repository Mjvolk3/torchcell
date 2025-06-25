---
id: zfnut7p12bhr1rt6imnlkg2
title: '25'
desc: ''
updated: 1750736862851
created: 1750220864493
---

- [ ] Add Gene Expression datasets
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.
- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work
- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.
- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...

***

## 2025.06.16

[[2025.06.16 - With Use Transform No Other Change|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250616---with-use-transform-no-other-change]]

```bash
(torchcell) mjvolk3@biologin-2 ~/projects/torchcell $ squeue -p cabbi
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1826088     cabbi     HCPD  mjvolk3  R 1-07:24:54      1 compute-3-3
           1826116     cabbi     HCPD  mjvolk3  R   15:44:35      1 compute-3-3
```

## 2025.06.17

- [x] [[2025.06.17 - I Cannot Find Pattern Of Jumping Correlation|dendron://torchcell/experiments.005-kuzmin2018-tmi.results#20250617---i-cannot-find-pattern-of-jumping-correlation]] → Our model looks state of the art now according to our own benchmarking. Still have disagreement with the original DCell paper.
- [x] Transfer data via globus
- [x] Run model on `006` locally
- [x] Run Dango on `006` query

## 2025.06.18

- [x] Run larger model

## 2025.06.19

- [x] No issues with transforms. [[200734 Transform Comparison|dendron://torchcell/scratch.2025.06.19.200734-transform-comparison]]

## 2025.06.20

- [ ] Just selecting metabolic process does not give us enough genes. We are targeting 220 genes.

```python
genome.go_dag['GO:0008152']
GOTerm('GO:0008152'):
  id:GO:0008152
  item_id:GO:0008152
  name:metabolic process
  namespace:biological_process
  _parents: 1 items
    GO:0009987
  parents: 1 items
    GO:0009987 level-01 depth-01 cellular process [biological_process]
  children: 84 items
  level:2
  depth:2
  is_obsolete:False
  alt_ids: 2 items
    GO:0044710
    GO:0044236
```

```python
graph.go_to_genes['GO:0008152']
GeneSet(size=102, items=['YAL015C', 'YAL038W', 'YBR001C']...)
```

- [ ] We have 207 kinase activity genes. This would give nice cross over for the kinase dataset.

```python
graph.go_to_genes['GO:0016301']
GeneSet(size=207, items=['YAL017W', 'YAL038W', 'YAR018C']...)
```

```python
genome.go_dag['GO:0016301']
GOTerm('GO:0016301'):
  id:GO:0016301
  item_id:GO:0016301
  name:kinase activity
  namespace:molecular_function
  _parents: 1 items
    GO:0016772
  parents: 1 items
    GO:0016772 level-03 depth-03 transferase activity, transferring phosphorus-containing groups [molecular_function]
  children: 115 items
  level:4
  depth:4
  is_obsolete:False
  alt_ids: 0 items
graph.go_to_genes['GO:0016301']
GeneSet(size=207, items=['YAL017W', 'YAL038W', 'YAR018C']...)
```

## 2025.06.23

- [ ] Ran DCell 29 days.
- [ ] 
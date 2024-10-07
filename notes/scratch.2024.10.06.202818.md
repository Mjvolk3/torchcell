---
id: lta22nobrqh6ljikokmgcz9
title: '202818'
desc: ''
updated: 1728319293996
created: 1728264501110
---
```python
# Train
data_module.split_index.train.phenotype_label_index
{'fitness': IndexSplit(indices=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 20, 21, 22, 23, 24... 1340836, 1340837, 1340839], count=988904), 'gene_interaction': IndexSplit(indices=[3, 4, 5, 6, 8, 10, 15, 20, 21, 23, 27, 28, 32, 34, 39, 43, 44, 46... 1340830, 1340831, 1340836], count=785559)}
data_module.split_index.train.perturbation_count_index
{2: IndexSplit(indices=[0, 1, 3, 4, 5, 6, 8, 10, 15, 16, 20, 21, 23, 27, 28, 32, 34, 36, ... 1340830, 1340831, 1340836], count=793767), 3: IndexSplit(indices=[1048579, 7, 9, 1048586, 11, 14, 22, 24, 26, 29, 31, 38, 40, 47, 1... 1048525, 1048553, 1048570], count=191514), 1: IndexSplit(indices=[319489, 647169, 368643, 376836, 376838, 360457, 532489, 352265, 4...2243, 294898, 376828, 376829], count=3623)}

# Val
data_module.split_index.val.phenotype_label_index
{'fitness': IndexSplit(indices=[917505, 131073, 655365, 393221, 655366, 1179661, 524304, 655389, ...57, 655339, 524273, 1048571], count=22514), 'gene_interaction': IndexSplit(indices=[917505, 622593, 655365, 393221, 655366, 950279, 589829, 1015814, ...4725, 950266, 196603, 32764], count=19413)}
data_module.split_index.val.perturbation_count_index
{2: IndexSplit(indices=[917505, 622593, 655365, 393221, 655366, 950279, 589829, 1015814, ...4725, 950266, 196603, 32764], count=19530), 3: IndexSplit(indices=[860160, 131073, 204800, 213006, 286736, 213018, 1294363, 40988, 2...974, 253942, 1204215, 155643], count=2930), 1: IndexSplit(indices=[339968, 512386, 507907, 365828, 121992, 369679, 339223, 452376, 3... 20337, 377716, 440567, 376575], count=54)}

# Test
data_module.split_index.test.phenotype_label_index
{'fitness': IndexSplit(indices=[262146, 393238, 1048603, 30, 655391, 1310753, 786466, 131111, 655...34, 655346, 1048567, 524282], count=22647), 'gene_interaction': IndexSplit(indices=[950273, 1146883, 720911, 98323, 819220, 393238, 1146906, 1048603,...408, 1212409, 524282, 98302], count=19451)}
data_module.split_index.test.perturbation_count_index
{2: IndexSplit(indices=[950273, 1146883, 720911, 98323, 819220, 393238, 1146906, 1048603,...408, 1212409, 524282, 98302], count=19568), 3: IndexSplit(indices=[262146, 229386, 196619, 57357, 32783, 81935, 286739, 1277976, 245...253939, 229364, 40954, 49151], count=3028), 1: IndexSplit(indices=[378625, 450817, 316283, 376708, 341896, 519563, 498958, 806158, 3...324086, 350586, 378235, 365950], count=51)}
```

According to this output this means that in total there are only `3623 + 54 + 51 = 3,728`, 1 perturbations.

```python
{k:len(v) for k,v in dataset.perturbation_count_index.items()}
{2: 1036030, 3: 299146, 1: 5665}
```

When in fact there are `5665` total 1 perturbations...

Also we noticed another discrepancy... it appears that some of the indices is being left out due to selection... this should not happen.

```python
data_module.index
{'train': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 20, 21, 22, 23, 24, ...], 'val': [917505, 131073, 655365, 393221, 655366, 1179661, 524304, 655389, 1048608, 393250, 1179692, 786480, 655419, 393276, 131140, 1179718, 1310795, 393297, 1179735, ...], 'test': [262146, 393238, 1048603, 30, 655391, 1310753, 786466, 131111, 655409, 1179705, 917567, 393281, 524358, 1048657, 262226, 1048662, 393303, 655448, 524389, ...]}
{k:len(v) for k,v in data_module.index.items()}
{'train': 988904, 'val': 22514, 'test': 22647}
sum([len(v) for k,v in data_module.index.items()])
1034065
len(dataset)
1340841
```

With updates...

```python
# Train
data_module.split_index.train.phenotype_label_index
{'fitness': IndexSplit(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,...1340836, 1340837, 1340839], count=1072673), 'gene_interaction': IndexSplit(indices=[2, 3, 4, 5, 6, 8, 10, 13, 15, 17, 18, 19, 20, 21, 23, 25, 27, 28,... 1340830, 1340831, 1340836], count=836868)}
data_module.split_index.train.perturbation_count_index
{2: IndexSplit(indices=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 2... 1340830, 1340831, 1340836], count=846562), 3: IndexSplit(indices=[1048579, 7, 9, 1048586, 11, 14, 1048593, 22, 24, 26, 29, 31, 33, ... 1048535, 1048553, 1048570], count=222365), 1: IndexSplit(indices=[319489, 647169, 368643, 376836, 376838, 376840, 532489, 352265, 4...5866, 352243, 376828, 376829], count=3746)}

# Val
data_module.split_index.val.phenotype_label_index
{'fitness': IndexSplit(indices=[262144, 786432, 262146, 262147, 786434, 1310721, 262150, 786438, ...1, 786422, 1048571, 786428], count=134085), 'gene_interaction': IndexSplit(indices=[786432, 1179648, 786434, 917505, 393221, 786438, 655366, 655365, ..., 1179644, 1179645, 1179647], count=78191)}
data_module.split_index.val.perturbation_count_index
{2: IndexSplit(indices=[786432, 786434, 1310724, 786438, 786442, 1310733, 524304, 1310740...11, 786424, 1048571, 786428], count=79883), 3: IndexSplit(indices=[262144, 1310721, 262146, 262147, 1310723, 131073, 262150, 1310726...705, 131065, 262140, 131071], count=54027), 1: IndexSplit(indices=[812033, 649221, 340486, 312824, 270865, 770066, 324119, 1321498, ...1238, 1199609, 377339, 269820], count=175)}

# Test
data_module.split_index.test.phenotype_label_index
{'fitness': IndexSplit(indices=[1048603, 30, 1310753, 786466, 524358, 1048657, 262226, 1048662, 5..., 1048565, 524282, 1048566], count=134083), 'gene_interaction': IndexSplit(indices=[1048603, 30, 1310753, 786466, 524358, 1048657, 262226, 1048662, 5..., 1048565, 524282, 1048566], count=108137)}
data_module.split_index.test.perturbation_count_index
{2: IndexSplit(indices=[1048603, 30, 1310753, 786466, 524358, 1048657, 262226, 1048662, 5..., 1048565, 524282, 1048566], count=109585), 3: IndexSplit(indices=[917516, 917538, 917653, 917664, 1310919, 917720, 1310947, 917734,...44, 262129, 1048562, 917503], count=22754), 1: IndexSplit(indices=[376832, 507907, 376839, 360457, 376843, 368658, 335891, 344083, 3...4036, 286691, 360423, 376821], count=1744)}
```

1 perturbation counts. There are `175` in val and `1744` in test. We would like these to be closer.

|    | split | index_type               | key              | count   |
|----|-------|--------------------------|------------------|---------|
| 0  | train | phenotype_label_index    | fitness          | 1072673 |
| 1  | train | phenotype_label_index    | gene_interaction | 854275  |
| 2  | train | perturbation_count_index | 2                | 863417  |
| 3  | train | perturbation_count_index | 3                | 204284  |
| 4  | train | perturbation_count_index | 1                | 4972    |
| 5  | val   | phenotype_label_index    | fitness          | 134084  |
| 6  | val   | phenotype_label_index    | gene_interaction | 84578   |
| 7  | val   | perturbation_count_index | 2                | 86407   |
| 8  | val   | perturbation_count_index | 3                | 47332   |
| 9  | val   | perturbation_count_index | 1                | 345     |
| 10 | test  | phenotype_label_index    | fitness          | 134084  |
| 11 | test  | phenotype_label_index    | gene_interaction | 84343   |
| 12 | test  | perturbation_count_index | 2                | 86206   |
| 13 | test  | perturbation_count_index | 3                | 47530   |
| 14 | test  | perturbation_count_index | 1                | 348     |

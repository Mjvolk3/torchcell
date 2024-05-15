---
id: 93w5fpz75h3jmlm8jf6hs1i
title: yeast-dataset
desc: ''
updated: 1715732304646
created: 1715732228869
---
## Deliverable for Yeast Dataset

1. `python -m pip install torchcell==0.0.?`. I recommend you used a virtual environment for this.
2. Unzip the data I sent you and put it in any `data/*` folder your have.
3. Import the dataset
4. Load dataset
5. Iterate in dataset, and access data

```python
from torchcell.dataset import Yeast9FitnessDataset
yeast9_fitness_dataset = Yeast9FitnessDataset(root:str="data/*")
for data in yeast9_fitness_dataset:
  print(data['experiment'].genotype)
  print(data['experiment'].phenotype)
  print(data['reference'])
```

### Desireable Dataset Features I Want

1. `Yeast9FitnessDataset` calls query in the lmdb.
2. Basic indexing features. For instance it should hold an index for each dataset that a given data point comes from. Pydantic class for index using pydantic to pass standardized data. Do `isinstance` check on on say genotype, environment, or phenotype, then could do another layer of `isinstance.`
3. `indices` property that returns a list of index objects. These index objects need to be general enough to apply to a lot different types of data. They can be based off the current pydantic data model.
4. This will be the zipped up process, preprocess, and raw dirs. So it will contain the lmdb. preprocess can capture the index or indices property.

```python
class BaseCellDataset(Dataset):
  pass

class Yeast9FitnessDataset(BaseCellDataset):
  pass
```

### Desireable Features For the User

I want to be able to give @Junyu-Chen the following protocol which should be easy. → [[Deliverable for Yeast Dataset|dendron://torchcell/user.Junyu-Chen.yeast-dataset#deliverable-for-yeast-dataset]]

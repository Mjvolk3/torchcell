---
id: 9p8raxm87m5eoy9c5u2az10
title: Neo_costanzo2016
desc: ''
updated: 1704437112668
created: 1704437089340
---

```python
def save(self):
    if not osp.exists(self.processed_dir):
        os.makedirs(self.processed_dir)
    save_path = osp.join(self.processed_dir, "dataset.json")
    # Create a dictionary to store the serialized data
    serialized_data = {
        "data": [experiment.model_dump() for experiment in self.data],
        "reference": [ref.model_dump() for ref in self.reference],
        "reference_index": [
            ref_idx.model_dump() for ref_idx in self.reference_index.data
        ],
    }
    with open(save_path, "w") as file:
        json.dump(serialized_data, file, indent=4)
def load(self):
    logging.info("Loading Dataset")
    load_path = osp.join(self.processed_dir, "dataset.json")
    if not osp.exists(load_path):
        raise FileNotFoundError("Saved dataset not found.")
    with open(load_path, "r") as file:
        serialized_data = json.load(file)
    # Deserialize 'data' back into the appropriate classes
    logging.info("Deserializing data")
    self.data = []
    for exp in tqdm(serialized_data["data"], desc="Loading Data"):
        self.data.append(FitnessExperiment.model_validate(exp))
    # Deserialize 'reference' back into the appropriate classes
    logging.info("Deserializing reference")
    self.reference = []
    for ref in tqdm(serialized_data["reference"], desc="Loading Reference"):
        self.reference.append(FitnessExperimentReference.model_validate(ref))
    # Deserialize 'reference_index' back into the appropriate classes
    logging.info("Deserializing reference index")
    reference_index_data = []
    for ref_idx in tqdm(serialized_data["reference_index"], desc="Loading Reference Index"):
        reference_index_data.append(ExperimentReferenceIndex.model_validate(ref_idx))
    self.reference_index = ReferenceIndex(data=reference_index_data)
```

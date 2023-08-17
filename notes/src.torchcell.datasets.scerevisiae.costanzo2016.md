---
id: hg6o7t1z649hm6v57ywecwp
title: Costanzo2016
desc: ''
updated: 1692275303206
created: 1692239635047
---

## DmfCostanzoDataset Out Memory Dataset

- As an alternative I tried to use the out of memory `Dataset` since the `InMemoryDataset` takes a while to load. Using this method takes a ton of time to write the data but could possible pay off during the training, especially since there will be other large objects in memory.

```python
from torch_geometric.data import Dataset

class DMFCostanzo2016Dataset(Dataset):
    url = (
        "https://thecellmap.org/costanzo2016/data_files/"
        "Raw%20genetic%20interaction%20datasets:%20Pair-wise%20interaction%20format.zip"
    )

    def __init__(
        self,
        root: str = "data/scerevisiae/costanzo2016",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.data_list = self.process()

    @property
    def raw_file_names(self) -> list[str]:
        return ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_dmf_{i}.pt" for i in range(self.len())]

    def download(self):
        path = download_url(self.url, self.raw_dir)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        os.remove(path)

        # Move the contents of the subdirectory to the parent raw directory
        sub_dir = os.path.join(
            self.raw_dir,
            "Data File S1. Raw genetic interaction datasets: Pair-wise interaction format",
        )
        for filename in os.listdir(sub_dir):
            shutil.move(os.path.join(sub_dir, filename), self.raw_dir)
        os.rmdir(sub_dir)

    def process(self):
        data_list = []

        # Process the DMF Files
        print("Processing DMF Files...")
        for file_name in tqdm(self.raw_file_names):
            file_path = os.path.join(self.raw_dir, file_name)
            df = pd.read_csv(file_path, sep="\t", header=0)

            # Extract genotype information
            query_id = df["Query Strain ID"].str.split("_").str[0].tolist()
            array_id = df["Array Strain ID"].str.split("_").str[0].tolist()

            query_genotype = [
                {
                    "id": id_val,
                    "intervention": "deletion",
                    "id_full": full_id,
                }
                for id_val, full_id in zip(query_id, df["Query Strain ID"])
            ]
            array_genotype = [
                {
                    "id": id_val,
                    "intervention": "deletion",
                    "id_full": full_id,
                }
                for id_val, full_id in zip(array_id, df["Array Strain ID"])
            ]

            # Combine the genotypes
            combined_genotypes = list(zip(query_genotype, array_genotype))

            # Extract observation information
            observations = [
                {
                    "smf_fitness": [
                        row["Query single mutant fitness (SMF)"],
                        row["Array SMF"],
                    ],
                    "dmf_fitness": row["Double mutant fitness"],
                    "dmf_std": row["Double mutant fitness standard deviation"],
                    "genetic_interaction_score": row["Genetic interaction score (ε)"],
                    "genetic_interaction_p-value": row["P-value"],
                }
                for _, row in df.iterrows()  # This part is still a loop due to the complexity of the data structure
            ]

            # Create environment dict
            environment = {"media": "YPD", "temperature": 30}

            # Combine everything
            combined_data = [
                {
                    "genotype": genotype,
                    "phenotype": {
                        "observation": observation,
                        "environment": environment,
                    },
                }
                for genotype, observation in zip(combined_genotypes, observations)
            ]

            # Convert to Data objects and save each instance to a separate file
            for idx, item in enumerate(combined_data):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]
                data_list.append(data)

                # Save each Data object to its own file
                file_name = f"data_dmf_{idx}.pt"
                torch.save(data, os.path.join(self.processed_dir, file_name))

        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        sample = self.data_list[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    @property
    def gene_set(self):
        gene_ids = set()
        for data in self.data_list:
            for genotype in data.genotype:
                gene_ids.add(genotype["id"])
        return gene_ids

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"
```

# torchcell/datasets/scerevisiae/baryshnikovna2010.py
# [[torchcell.datasets.scerevisiae.baryshnikovna2010]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/baryshnikovna2010.py
# Test file: torchcell/datasets/scerevisiae/test_baryshnikovna2010.py
import os
from typing import Callable, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


os.makedirs("data/scerevisiae/baryshnikovna2010", exist_ok=True)


class Baryshnikovna2010Dataset(InMemoryDataset):
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/MediaObjects/41592_2010_BFnmeth1534_MOESM168_ESM.xls"

    def __init__(
        self,
        root: str = "data/scerevisiae/baryshnikovna2010",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["data.xls"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        path = download_url(self.url, self.raw_dir)
        os.rename(path, os.path.join(self.raw_dir, "data.xls"))

    def process(self):
        xls_path = os.path.join(self.raw_dir, "data.xls")
        df = pd.read_excel(xls_path, names=["id", "fitness", "std"])

        data_list = []
        for index, row in df.iterrows():
            id = row["id"].split("_")[0]
            id_full = row["id"]
            genotype = {
                "id": id,
                "interventions": "deletion",
                "id_full": id_full,
            }
            observation = {
                "smf_fitness": torch.tensor([row["fitness"]], dtype=torch.float),
                "smf_std": torch.tensor([row["std"]], dtype=torch.float),
            }
            environment = {
                "media": "YPD",
                "temperature": "30C",
            }
            phenotype = {"observation": observation, "environment": environment}

            data = Data()
            data.genotype = genotype
            data.phenotype = phenotype
            data_list.append(data)

        if self.pre_transform:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the processed data
        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    dataset = Baryshnikovna2010Dataset(root="data/scerevisiae/baryshnikovna2010")
    print(40 * "=")
    print("Lets take a quick look at the dataset...")
    print(f">>> dataset\n{dataset}")
    print(f">>> dataset[0]\n{dataset[0]}")
    print(f">>> dataset[0].genotype\n{dataset[0].genotype}")
    print(f">>> dataset[0].phenotype\n{dataset[0].phenotype}")
    print(f">>> dataset[0].environment\n{dataset[0].environment}")
    print(40 * "=")

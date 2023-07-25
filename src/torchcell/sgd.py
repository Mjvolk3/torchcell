import requests
from typing import Any
from dataclasses import dataclass, field
import os.path as osp
import os
import json


@dataclass
class Gene:
    locusID: str = "YAL001C"
    sgd_url: str = "https://www.yeastgenome.org/backend/locus"
    headers: dict[str, str] = field(
        default_factory=lambda: {"accept": "application/json"}
    )
    _data: dict[str, dict[Any, Any] | list[Any]] = field(default_factory=dict)
    base_data_dir: str = "data/sgd"

    def __post_init__(self) -> None:
        if not osp.exists(self.base_data_dir):
            os.makedirs(self.base_data_dir)
        self.save_path: str = osp.join(self.base_data_dir, f"{self.locusID}.json")

    @property
    def data(self) -> dict[str, dict[Any, Any] | list[Any]]:
        if self._data != {}:
            return self._data
        elif osp.exists(self.save_path):
            self._data = self.read()
        else:
            self._data["locus"] = self.locus()
            self._data["sequence_details"] = self.sequence_details()
            self._data["neighbor_sequence_details"] = self.neighbor_sequence_details()
            self._data["posttranslational_details"] = self.posttranslational_details()
            self._data["protein_experiment_details"] = self.protein_experiment_details()
            self._data["protein_domain_details"] = self.protein_domain_details()
            self._data["go_details"] = self.go_details()
            self._data["phenotype_details"] = self.phenotype_details()
            self._data["interaction_details"] = self.interaction_details()
            self._data["regulation_details"] = self.regulation_details()
            self._data["literature_details"] = self.literature_details()
            self.write()
        return self._data

    def write(self) -> None:
        with open(self.save_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def read(self) -> dict[str, dict[Any, Any] | list[Any]]:
        if not osp.exists(self.save_path):
            raise ValueError(f"File {self.save_path} does not exist")
        with open(self.save_path, "r") as f:
            data_in = json.load(f)
        for key, value in data_in.items():
            data_in[key] = value
        if isinstance(data_in, dict):
            return data_in
        else:
            raise ValueError("Read unexpected data type")

    def _get_data(self, url: str) -> dict[Any, Any] | list[Any]:
        response = requests.get(url, headers=self.headers)
        data = response.json()
        if isinstance(data, (dict, list)):
            return data
        else:
            raise ValueError("Unexpected response from server")

    def locus(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID)
        data = self._get_data(url)
        return data

    def sequence_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "sequence_details")
        data = self._get_data(url)
        return data

    def neighbor_sequence_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "neighbor_sequence_details")
        data = self._get_data(url)
        return data

    def posttranslational_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "posttranslational_details")
        data = self._get_data(url)
        return data

    def protein_experiment_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "protein_experiment_details")
        data = self._get_data(url)
        return data

    def protein_domain_details(self) -> dict[Any, Any] | list[Any]:
        # The only one that returns a list
        url = osp.join(self.sgd_url, self.locusID, "protein_domain_details")
        data = self._get_data(url)
        return data

    def go_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "go_details")
        data = self._get_data(url)
        return data

    def phenotype_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "phenotype_details")
        data = self._get_data(url)
        return data

    def interaction_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "interaction_details")
        data = self._get_data(url)
        return data

    def regulation_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "regulation_details")
        data = self._get_data(url)
        return data

    def literature_details(self) -> dict[Any, Any] | list[Any]:
        url = osp.join(self.sgd_url, self.locusID, "literature_details")
        data = self._get_data(url)
        return data


if __name__ == "__main__":
    gene = Gene()
    gene.data
    print()

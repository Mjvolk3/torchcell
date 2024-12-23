from pathlib import Path
import zipfile
import requests
import cobra
import networkx as nx
from typing import Optional
from attrs import define, field


@define
class YeastGEM:
    root: Path = field(converter=Path, default=Path("data/torchcell/yeast-GEM"))
    version: str = field(default="9.0.2")
    _model: Optional[cobra.Model] = field(default=None, init=False)
    _metabolic_graph: Optional[nx.Graph] = field(default=None, init=False)
    model_dir: Path = field(init=False)

    def __attrs_post_init__(self):
        """Initialize derived attributes after instance creation."""
        self.model_dir = self.root / f"yeast-GEM-{self.version}"

    def download(self) -> None:
        if self.model_dir.exists():
            print(f"Model already exists at {self.model_dir}")
            return

        # Create directories if they don't exist
        self.root.mkdir(parents=True, exist_ok=True)

        # Download zip file
        url = f"https://github.com/SysBioChalmers/yeast-GEM/archive/refs/tags/v{self.version}.zip"
        response = requests.get(url)
        response.raise_for_status()

        # Save and extract zip file
        zip_path = self.root / f"yeast-GEM-{self.version}.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        # Remove zip file after extraction
        zip_path.unlink()
        print(f"Successfully downloaded and extracted model to {self.model_dir}")

    @property
    def model(self) -> cobra.Model:
        if self._model is None:
            if not self.model_dir.exists():
                self.download()
            model_path = self.model_dir / "model" / "yeast-GEM.xml"
            self._model = cobra.io.read_sbml_model(str(model_path))
        return self._model

    @property
    def metabolic_graph(self) -> nx.Graph:
        if self._metabolic_graph is not None:
            return self._metabolic_graph

        # Initialize new undirected graph
        G = nx.Graph()

        # Get model if not already loaded
        model = self.model

        # Add all metabolites as nodes
        for metabolite in model.metabolites:
            G.add_node(metabolite.id, name=metabolite.name)

        # Process reactions to create edges
        for reaction in model.reactions:
            # Get reaction metabolites
            metabolites = list(reaction.metabolites.keys())

            # Get genes associated with the reaction
            genes = [gene.id for gene in reaction.genes]
            genes_str = "|".join(genes) if genes else "no_gene"

            # Create edges between all pairs of metabolites in the reaction
            for i in range(len(metabolites)):
                for j in range(i + 1, len(metabolites)):
                    G.add_edge(
                        metabolites[i].id,
                        metabolites[j].id,
                        genes=genes_str,
                        reaction=reaction.id,
                    )

        # Cache the graph
        self._metabolic_graph = G
        return G


def main():
    # Initialize the class
    gem = YeastGEM()

    # Download the model (if not already present)
    gem.download()

    # Access the COBRA model
    model = gem.model

    # Access the metabolic graph
    graph = gem.metabolic_graph


if __name__ == "__main__":
    main()

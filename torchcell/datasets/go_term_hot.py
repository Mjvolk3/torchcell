# torchcell/datasets/go_term_hot
# [[torchcell.datasets.go_term_hot]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/go_term_hot
# Test file: tests/torchcell/datasets/test_go_term_hot.py

"""Gene Ontology multi-hot gene embeddings from SGD's per-gene GO annotations.

One column per GO term, one row per gene; a gene's row marks the terms it is annotated
to. The annotations are the ``Ontology_term`` attributes of the SGD GFF the genome is
built from, which carry the term ids and no evidence codes. Two things make the vector:

- **propagation**: with ``propagate`` the row also marks every ancestor of an annotated
  term through ``is_a`` and ``part_of`` (the true-path rule, the convention DCell and
  every GO enrichment use), so a gene annotated to a leaf lights its whole lineage; the
  direct form marks the annotated terms only;
- **the term floor**: a term is a column only if at least ``min_genes`` genes carry it
  after propagation, which drops the singleton leaves that name one gene.

The column order is the sorted term id, written beside the tensor as
``<model_name>.terms.json`` (``term_id``, ``name``, ``namespace``, ``n_genes``) so a
learned weight can be read back to a term. Caveat for any endpoint derived from deletion
phenotypes: SGD annotates genes from mutant phenotypes (evidence code IMP), and the GFF
does not carry evidence codes, so a fraction of these columns encode knowledge that came
from the same deletion collections the labels did; filtering by evidence code needs the
gene association file, which this dataset does not read.
"""

import json
import os.path as osp
from collections.abc import Callable
from typing import Any, cast

import torch
from goatools.obo_parser import GODag, GOTerm
from torch_geometric.data import Data
from tqdm import tqdm

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.sequence.genome.scerevisiae.s288c import (
    SCerevisiaeGene,
    SCerevisiaeGenome,
)

NAMESPACES = ("biological_process", "molecular_function", "cellular_component")


class GOTermEmbeddingDataset(BaseEmbeddingDataset):
    """Multi-hot gene embedding over GO terms, direct or propagated, per namespace."""

    #: model name -> (propagate to ancestors, namespaces kept, minimum genes per term).
    MODEL_TO_WINDOW: dict[str, tuple[bool, tuple[str, ...], int]] = {
        "go": (True, NAMESPACES, 10),
        "go_direct": (False, NAMESPACES, 2),
        "go_bp": (True, ("biological_process",), 10),
        "go_mf": (True, ("molecular_function",), 10),
        "go_cc": (True, ("cellular_component",), 10),
    }

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Store the genome (its genes carry the GO ids; its GO root the OBO) and process."""
        assert model_name in self.MODEL_TO_WINDOW, model_name
        self.genome = genome
        super().__init__(root, model_name, transform, pre_transform)
        self.process()

    def initialize_model(self) -> None:
        """Return nothing; this dataset uses no learned model."""
        pass

    @property
    def processed_file_names(  # type: ignore[override]  # widens the base return
        self,
    ) -> list[str]:
        """Return the processed file name for the selected model."""
        return [f"{self.model_name}.pt"]

    @property
    def terms_path(self) -> str:
        """The JSON sidecar naming the columns."""
        return osp.join(self.processed_dir, f"{self.model_name}.terms.json")

    @staticmethod
    def ancestors(dag: GODag, term_id: str, memo: dict[str, set[str]]) -> set[str]:
        """Every ancestor of ``term_id`` through ``is_a`` and ``part_of``, memoized."""
        if term_id in memo:
            return memo[term_id]
        term = cast(GOTerm, dag[term_id])
        parents: set[str] = {p.item_id for p in term.parents}
        part_of = getattr(term, "relationship", {}).get("part_of", set())
        parents |= {p.item_id for p in part_of}
        out: set[str] = set(parents)
        for p in parents:
            out |= GOTermEmbeddingDataset.ancestors(dag, p, memo)
        memo[term_id] = out
        return out

    def process(self) -> None:
        """Build the multi-hot rows, write the tensor and the term sidecar."""
        if osp.isfile(self.processed_paths[0]) and osp.isfile(self.terms_path):
            return
        propagate, namespaces, min_genes = self.MODEL_TO_WINDOW[
            cast(str, self.model_name)
        ]
        genome = self.genome
        dag = GODag(
            osp.join(genome.go_root, "go.obo"),
            optional_attrs={"relationship"},
            prt=None,
        )
        memo: dict[str, set[str]] = {}
        gene_terms: dict[str, set[str]] = {}
        dropped: set[str] = set()
        for gene_id in tqdm(genome.gene_set, desc=f"GO terms ({self.model_name})"):
            gene = cast(SCerevisiaeGene, genome[gene_id])
            direct = set(gene.go) if gene.go is not None else set()
            # A term the OBO no longer carries (obsolete, or an alt id the GFF still
            # uses) is resolved through the DAG's alt-id map when it has one and
            # dropped otherwise; the dropped ids are written into the sidecar.
            resolved: set[str] = set()
            for t in direct:
                if t in dag:
                    resolved.add(cast(GOTerm, dag[t]).item_id)
                else:
                    dropped.add(t)
            terms = set(resolved)
            if propagate:
                for t in resolved:
                    terms |= self.ancestors(dag, t, memo)
            terms = {t for t in terms if cast(GOTerm, dag[t]).namespace in namespaces}
            # The three namespace roots light every annotated gene and say nothing.
            terms = {t for t in terms if cast(GOTerm, dag[t]).level > 0}
            gene_terms[gene_id] = terms
        counts: dict[str, int] = {}
        for terms in gene_terms.values():
            for t in terms:
                counts[t] = counts.get(t, 0) + 1
        columns = sorted(t for t, n in counts.items() if n >= min_genes)
        index = {t: i for i, t in enumerate(columns)}
        data_list = []
        for gene_id in sorted(gene_terms):
            row = torch.zeros(len(columns), dtype=torch.float)
            for t in gene_terms[gene_id]:
                if t in index:
                    row[index[t]] = 1.0
            data = Data(id=gene_id)
            data.embeddings = {self.model_name: row.unsqueeze(0)}
            data_list.append(data)
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
        with open(self.terms_path, "w") as fh:
            json.dump(
                {
                    "model_name": self.model_name,
                    "propagate": propagate,
                    "namespaces": list(namespaces),
                    "min_genes": min_genes,
                    "n_genes": len(gene_terms),
                    "n_genes_annotated": sum(1 for t in gene_terms.values() if t),
                    "n_columns": len(columns),
                    "dropped_term_ids": sorted(dropped),
                    "terms": [
                        {
                            "term_id": t,
                            "name": cast(GOTerm, dag[t]).name,
                            "namespace": cast(GOTerm, dag[t]).namespace,
                            "n_genes": counts[t],
                        }
                        for t in columns
                    ],
                },
                fh,
                indent=1,
            )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    for name in GOTermEmbeddingDataset.MODEL_TO_WINDOW:
        ds = GOTermEmbeddingDataset(
            root=osp.join(data_root, "data/scerevisiae/go_term_hot"),
            genome=genome,
            model_name=name,
        )
        with open(ds.terms_path) as fh:
            meta = json.load(fh)
        print(
            name,
            meta["n_columns"],
            "columns,",
            meta["n_genes_annotated"],
            "annotated genes",
        )

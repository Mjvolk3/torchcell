# torchcell/datasets/solve_vector
# [[torchcell.datasets.solve_vector]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/solve_vector

"""The deletion's solved flux vector as a gene embedding.

One row per gene, one column per reaction of the GEM (4,131 in yeast-GEM 9.0.2): the
row is ``log(1 + |v|)`` of the flux the genome-scale program returns with that gene's
reactions closed, the encoding the 028 flux head reads (rounds 21 to 41). The vectors
come from a solve store written by ``experiments/028-gene-essentiality/scripts/
ess_reference_flux_table.py`` (the parsimonious two-stage solve, or the growth program
by another solver under ``--solver``) or by ``solve_tolerance_read.py`` (the PDLP
store): ``$DATA_ROOT/data/torchcell/reference_flux/ess_solve_encodings_<medium>[_<solver>].npz``
with ``genes`` and ``v_deletion``. A gene the store does not cover reads the zero row,
which under this encoding is the flux of a gene with no reaction.

This is the solve entering the cell's REPRESENTATION, as a node input beside the
sequence embedding and the annotation, rather than its logit through the affine
(028 round 33); it is what the layer-free form looks like when every phenotype head
can read it.
"""

import hashlib
import json
import os
import os.path as osp
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch_geometric.data import Data

from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class SolveVectorEmbeddingDataset(BaseEmbeddingDataset):
    """``log(1 + |v|)`` of the deletion's solved flux, per gene, from a solve store."""

    #: model name -> the store file under ``$DATA_ROOT/data/torchcell/reference_flux``.
    MODEL_TO_STORE: dict[str, str] = {
        "solve_log1p": "ess_solve_encodings_YPD.npz",
        "solve_log1p_ipm": "ess_solve_encodings_YPD_ipm.npz",
        "solve_log1p_pdlp": "ess_solve_encodings_YPD_pdlp.npz",
        "solve_log1p_sc": "ess_solve_encodings_SC.npz",
    }
    #: The base class validates ``model_name`` against this name; the stores are the
    #: "windows" here.
    MODEL_TO_WINDOW = MODEL_TO_STORE

    def __init__(
        self,
        root: str,
        genome: SCerevisiaeGenome,
        model_name: str,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
    ) -> None:
        """Store the genome (its gene set orders the rows) and process."""
        assert model_name in self.MODEL_TO_STORE, model_name
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
    def store_path(self) -> str:
        """The solve store this embedding is built from."""
        data_root = os.environ["DATA_ROOT"]
        return osp.join(
            data_root,
            "data/torchcell/reference_flux",
            self.MODEL_TO_STORE[cast(str, self.model_name)],
        )

    @property
    def provenance_path(self) -> str:
        """The JSON sidecar naming the store, its digest and the coverage."""
        return osp.join(self.processed_dir, f"{self.model_name}.provenance.json")

    def process(self) -> None:
        """Build the rows from the store, write the tensor and the sidecar."""
        if osp.isfile(self.processed_paths[0]) and osp.isfile(self.provenance_path):
            return
        payload = np.load(self.store_path, allow_pickle=False)
        store_genes = [str(g) for g in payload["genes"]]
        v = payload["v_deletion"].astype(np.float32)
        enc = torch.from_numpy(np.log1p(np.abs(v)))
        row_of = {g: i for i, g in enumerate(store_genes)}
        data_list = []
        covered = 0
        for gene_id in sorted(self.genome.gene_set):
            if gene_id in row_of:
                row = enc[row_of[gene_id]]
                covered += 1
            else:
                row = torch.zeros(enc.shape[1], dtype=torch.float)
            data = Data(id=gene_id)
            data.embeddings = {self.model_name: row.unsqueeze(0)}
            data_list.append(data)
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
        with open(self.provenance_path, "w") as fh:
            json.dump(
                {
                    "model_name": self.model_name,
                    "store": self.store_path,
                    "store_sha256": hashlib.sha256(
                        np.ascontiguousarray(v).tobytes()
                    ).hexdigest(),
                    "store_provenance": json.loads(str(payload["provenance"])),
                    "encoding": "log(1 + |v|), v in mmol/gDW/h as stored",
                    "n_columns": int(enc.shape[1]),
                    "n_genes": len(data_list),
                    "n_genes_covered": covered,
                    "rxn_ids": [str(r) for r in payload["rxn_ids"]],
                },
                fh,
                indent=1,
            )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    for name in SolveVectorEmbeddingDataset.MODEL_TO_STORE:
        ds = SolveVectorEmbeddingDataset(
            root=osp.join(data_root, "data/scerevisiae/solve_vector"),
            genome=genome,
            model_name=name,
        )
        with open(ds.provenance_path) as fh:
            meta = json.load(fh)
        print(name, meta["n_columns"], "columns,", meta["n_genes_covered"], "covered")

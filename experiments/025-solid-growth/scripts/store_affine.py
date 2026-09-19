# experiments/025-solid-growth/scripts/store_affine.py
# [[experiments.025-solid-growth.scripts.store_affine]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/store_affine

"""The metabolic module's store read on the 025 cell's prediction.

The 028 essentiality line settled the module's scalable form (rounds 27 to 57): the
growth program solved once per distinct (genotype, medium) pair offline, stored, and read
as $\\log(1 + |v|)$ over the 4,131 reactions through an affine added to the phenotype
head's output, with no solver in training. It reads the in-loop module's number on
essentiality, fits on every seed, and trains to one number over stores from any solver
that stops inside the optimal face. This is that read on the 025 transformer.

The store (``experiments/028-gene-essentiality/scripts/solid_growth_store.py``) holds one
row per distinct GEM gene subset the build's genotypes delete, keyed by the sorted
systematic names joined with ``+``; the empty key is the wild type. A genotype's key is
the GEM genes among its perturbed genes, so a genotype deleting no GEM gene reads the
wild-type row, and a key the store does not hold (none, when the store was built from
the same build's keys; counted otherwise) reads it too. The table stays on the host and
each batch gathers its rows, so a store of a million rows costs nothing on the card.

.. math::

    \\hat{y} = f(z) + w^\\top \\log(1 + |v_{\\text{key}}|) + b

``f(z)`` is the transformer's own prediction. The affine's weight norm is exposed for the
epoch log, as the 028 cells log it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch import nn


class StoreAffineReadout(nn.Module):
    """Wrap a cell model: its prediction plus an affine of the genotype's stored solve."""

    model: nn.Module
    table: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        store_path: str,
        node_ids: list[str],
        n_outputs: int = 1,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        payload = np.load(store_path, allow_pickle=False)
        keys = [str(k) for k in payload["keys"]]
        v = payload["v_deletion"].astype(np.float32)
        table = np.log1p(np.abs(v))
        if "" not in keys:
            keys.append("")
            table = np.concatenate(
                [table, np.log1p(np.abs(payload["v_wild_type"].astype(np.float32)))[None]]
            )
        self.row_of: dict[str, int] = {k: i for i, k in enumerate(keys)}
        self.wild_row = self.row_of[""]
        # Every gene that appears in any key is a GEM gene; the others never change the
        # program and are dropped from the key.
        gem_genes = {g for k in keys if k for g in k.split("+")}
        self.node_gene: list[str | None] = [g if g in gem_genes else None for g in node_ids]
        self.register_buffer("table", torch.from_numpy(table), persistent=False)
        self.linear = nn.Linear(table.shape[1], n_outputs)
        if zero_init:
            with torch.no_grad():
                self.linear.weight.zero_()
        self.store_path = store_path
        self.provenance = json.loads(str(payload["provenance"]))
        self.n_unknown_keys = 0
        self.n_lookups = 0

    def rows(self, batch: Any) -> torch.Tensor:
        """Row index per genotype of the batch, from its perturbed gene indices."""
        gene = batch["gene"]
        indices = gene.perturbation_indices.tolist()
        assignment = gene.perturbation_indices_batch.tolist()
        n = int(batch.num_graphs)
        genes_per: list[list[str]] = [[] for _ in range(n)]
        for idx, b in zip(indices, assignment, strict=True):
            g = self.node_gene[idx]
            if g is not None:
                genes_per[b].append(g)
        rows = []
        for genes in genes_per:
            key = "+".join(sorted(genes))
            row = self.row_of.get(key)
            if row is None:
                row = self.wild_row
                self.n_unknown_keys += 1
            rows.append(row)
        self.n_lookups += n
        return torch.tensor(rows, dtype=torch.long)

    def forward(
        self, cell_graph: Any, batch: Any, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        predictions, reps = self.model(cell_graph, batch, *args, **kwargs)
        rows = self.rows(batch)
        x = self.table[rows].to(predictions.device, non_blocking=True)
        affine = self.linear(x)
        reps["store_affine"] = affine
        return predictions + affine.reshape(predictions.shape), reps

    def affine_weight_norm(self) -> float:
        return float(self.linear.weight.detach().norm())

    @property
    def num_parameters(self) -> dict[str, int]:
        counts: dict[str, int] = dict(self.model.num_parameters)
        extra = sum(int(p.numel()) for p in self.linear.parameters())
        counts["store_affine"] = extra
        counts["total"] = counts.get("total", 0) + extra
        return counts

    def __getattr__(self, name: str) -> Any:
        # The regression task reads the model's regularizer configuration and adjacency
        # matrices; everything the wrapper does not define is the wrapped model's.
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)

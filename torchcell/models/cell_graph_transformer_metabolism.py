# torchcell/models/cell_graph_transformer_metabolism
# [[torchcell.models.cell_graph_transformer_metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_graph_transformer_metabolism
# Test file: tests/torchcell/models/test_cell_graph_transformer_metabolism.py

"""CGT-Metabolism: the Cell Graph Transformer with production / metabolome readouts.

This is the model class the metabolism work (pigment production, metabolome transfer,
and later the enzyme-constrained flux layer) is built inside. Track A -- the SIMB
demonstration -- activates only the readout heads; no flux layer is involved.

WHAT IS REUSED, AND WHY IT IS REUSED RATHER THAN COPIED
-------------------------------------------------------
The encoder (learnable gene embeddings + CLS token + the graph-regularized transformer
layers), the graph-regularization KL term, and the EQUIVARIANT perturbation operator
(``EquivariantPerturbationTransform``) plus the Type-II ``PerturbationHead`` are taken
from :mod:`torchcell.models.equivariant_cell_graph_transformer` **unchanged**, by
subclassing :class:`~torchcell.models.equivariant_cell_graph_transformer.CellGraphTransformer`
and delegating the whole forward to it.

That is a stronger guarantee than transcribing those ~900 lines into this file: a
transcription can silently drift from the original, whereas an inherited implementation
IS the original. What subclassing does NOT make free is initialization parity -- new
parameters created in a subclass ``__init__`` consume the global RNG stream, so a head
constructed in the wrong place would change every encoder weight for a given seed and
quietly invalidate any comparison against a Fig-3 run. Every metabolism head is therefore
constructed strictly AFTER ``super().__init__()`` returns, and
``tests/torchcell/models/test_cell_graph_transformer_metabolism.py`` asserts allclose
outputs against the parent at a fixed seed with the metabolism heads disabled.

THE THREE HEADS (Track A)
-------------------------
Declared through ``heads_config`` alongside the inherited ``global`` / ``per_gene`` /
``per_metabolite`` entries. Each extra entry needs a ``kind``:

* ``betaxanthin``   -- ``kind: scalar``. Cachera 2023 CRI-SPA corrected fluorescence,
  population-centred (so it can be negative), one value per strain.
* ``beta_carotene`` -- ``kind: scalar``. Ozaydin 2013 colony-colour score, a SUBJECTIVE
  ORDINAL on -5..+5. Report Spearman for this target, not Pearson.
* ``mulleder19``    -- ``kind: vector``, ``output_dim: 19``. Mulleder 2016 intracellular
  amino-acid concentrations in mM over 19 FIXED key-sorted columns
  (``alanine ... valine``). Deliberately NOT aligned to Yeast9 metabolite ids: the 19
  columns are dense in every record, so Track A needs no ``s_NNNN`` mapping at all.

**They must not share a head.** The three targets are a centred fluorescence, an ordinal
in -5..+5, and an absolute mM concentration -- mutually incomparable units. One shared
scalar head would be asked to regress three different scales onto one output, and the
largest-magnitude target would dominate the loss. Separate heads, separate losses, one
shared encoder: that separation IS the multitask claim, and it costs three small MLPs.

TRACK B (not implemented here)
------------------------------
The metabolite/enzyme entity layers, the flux readout with the box satisfied by
reparameterization, the soft ``Sv = 0`` / capacity / budget / thermodynamic terms, and
the GPR-derived availability chain all attach at the same place these heads do -- the
``(h_CLS, H_genes_pert)`` pair returned by the inherited forward. Nothing in Track A
forecloses them; nothing in Track A pays for them either.
"""

from typing import Any, cast

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

#: heads_config keys handled by the parent class; everything else is a metabolism head.
INHERITED_HEADS = frozenset({"global", "per_gene", "per_metabolite"})


class ProductScalarHead(nn.Module):
    """Whole-cell readout for a SCALAR production phenotype (one value per strain).

    Reads the CLS token, optionally concatenated with a mean pool over the equivariant
    perturbed gene embeddings, and emits exactly one value. Structurally this is the S1
    pooled readout; it is a distinct class from the vector head so that "this head is
    scalar" is a property of the module rather than of a config string that some other
    code has to agree with. ``output_dim`` is fixed at 1 by construction -- a scalar
    target assigned into a wider head broadcasts silently, and the cheapest place to
    make that impossible is here.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_gene_pool: bool = True,
        dropout: float = 0.1,
        param_dim: int = 1,
    ) -> None:
        """Build the MLP mapping ``[h_CLS (|| mean_i h_i_pert)]`` to one value.

        Args:
            hidden_dim: Model hidden dimension.
            use_gene_pool: Concatenate a mean pool over the perturbed gene tokens.
            dropout: Dropout probability.
            param_dim: Distributional params for the single feature (1 point, 2
                gaussian, K quantile). ``output_dim`` stays 1 either way.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.use_gene_pool = use_gene_pool
        self.param_dim = param_dim
        in_dim = hidden_dim * 2 if use_gene_pool else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, param_dim),
        )

    def forward(self, h_CLS: torch.Tensor, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, 1]`` (point) or ``[batch, 1, param_dim]`` (distributional)."""
        batch_size = H_genes_pert.shape[0]
        h = h_CLS.unsqueeze(0).expand(batch_size, -1)
        if self.use_gene_pool:
            h = torch.cat([h, H_genes_pert.mean(dim=1)], dim=-1)
        out = self.mlp(h)
        if self.param_dim > 1:
            out = out.view(batch_size, 1, self.param_dim)
        return cast(torch.Tensor, out)


class MetabolomeVectorHead(nn.Module):
    """Whole-cell readout for a FIXED-COLUMN metabolome vector (Mulleder's 19 AAs).

    The columns are the measured metabolite keys in ``sorted()`` order -- the same order
    the ``Perturbation`` graph processor uses when it flattens the ``metabolite_level``
    dict -- so column ``k`` of the output is always the same amino acid. This head does
    NOT read the Yeast9 metabolite node set and needs no ``s_NNNN`` mapping: the
    measured set is small, dense, and fixed, which is precisely why Track A can skip the
    metabolite-id alignment that a general per-metabolite readout would require.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        use_gene_pool: bool = True,
        dropout: float = 0.1,
        param_dim: int = 1,
    ) -> None:
        """Build the MLP mapping the pooled representation to ``output_dim`` columns.

        Args:
            hidden_dim: Model hidden dimension.
            output_dim: Number of measured metabolite columns F (19 for Mulleder).
            use_gene_pool: Concatenate a mean pool over the perturbed gene tokens.
            dropout: Dropout probability.
            param_dim: Distributional params PER COLUMN; ``output_dim`` stays F.
        """
        super().__init__()
        if output_dim < 2:
            raise ValueError(
                f"MetabolomeVectorHead output_dim must be >= 2, got {output_dim}; "
                "use ProductScalarHead for a single-value phenotype."
            )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gene_pool = use_gene_pool
        self.param_dim = param_dim
        in_dim = hidden_dim * 2 if use_gene_pool else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * param_dim),
        )

    def forward(self, h_CLS: torch.Tensor, H_genes_pert: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, F]`` (point) or ``[batch, F, param_dim]``."""
        batch_size = H_genes_pert.shape[0]
        h = h_CLS.unsqueeze(0).expand(batch_size, -1)
        if self.use_gene_pool:
            h = torch.cat([h, H_genes_pert.mean(dim=1)], dim=-1)
        out = self.mlp(h)
        if self.param_dim > 1:
            out = out.view(batch_size, self.output_dim, self.param_dim)
        return cast(torch.Tensor, out)


class CellGraphTransformerMetabolism(CellGraphTransformer):
    """CellGraphTransformer + named production / metabolome readout heads.

    Every extra ``heads_config`` entry (any key outside :data:`INHERITED_HEADS`) becomes
    one head, registered as the submodule ``f"{name}_head"`` and emitted into
    ``reps["head_outputs"][name]``. The spec keys are:

    * ``kind`` -- ``"scalar"`` (:class:`ProductScalarHead`) or ``"vector"``
      (:class:`MetabolomeVectorHead`). Required; there is no default, because guessing
      it is exactly the mistake that lets a scalar target broadcast across a vector head.
    * ``output_dim`` -- feature count F (must be 1 for ``scalar``).
    * ``use_gene_pool`` / ``dropout`` / ``param_dim`` -- as for the inherited heads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the inherited model, then the metabolism heads.

        The metabolism heads are constructed strictly after ``super().__init__()`` so the
        encoder + PERT parameters consume the RNG stream in exactly the parent's order
        and are bit-identical to the parent's at the same seed (parity-tested).
        """
        super().__init__(*args, **kwargs)
        self.metabolism_head_names: list[str] = []
        for name, raw_spec in self.heads_config.items():
            if name in INHERITED_HEADS:
                continue
            spec = dict(raw_spec or {})
            kind = spec.get("kind")
            output_dim = int(spec.get("output_dim", 1))
            head: nn.Module
            if kind == "scalar":
                if output_dim != 1:
                    raise ValueError(
                        f"head '{name}' is kind='scalar' but output_dim={output_dim}; "
                        "a scalar head emits exactly one value."
                    )
                head = ProductScalarHead(
                    hidden_dim=self.hidden_channels,
                    use_gene_pool=bool(spec.get("use_gene_pool", True)),
                    dropout=float(spec.get("dropout", 0.1)),
                    param_dim=int(spec.get("param_dim", 1)),
                )
            elif kind == "vector":
                head = MetabolomeVectorHead(
                    hidden_dim=self.hidden_channels,
                    output_dim=output_dim,
                    use_gene_pool=bool(spec.get("use_gene_pool", True)),
                    dropout=float(spec.get("dropout", 0.1)),
                    param_dim=int(spec.get("param_dim", 1)),
                )
            else:
                raise ValueError(
                    f"metabolism head '{name}' needs kind='scalar' or kind='vector' "
                    f"in its heads_config spec, got {kind!r}."
                )
            self.add_module(f"{name}_head", head)
            self.metabolism_head_names.append(name)

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData, return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the inherited forward, then append the metabolism head outputs.

        The encoder, graph-regularization loss and equivariant perturbation operator are
        the parent's, invoked unchanged; the heads read the ``h_CLS`` /
        ``H_genes_pert`` pair the parent already returns.
        """
        predictions, reps = super().forward(cell_graph, batch, return_attention)
        h_CLS = reps["h_CLS"]
        H_genes_pert = reps["H_genes_pert"]
        head_outputs = cast(dict[str, torch.Tensor], reps["head_outputs"])
        for name in self.metabolism_head_names:
            head = cast(nn.Module, getattr(self, f"{name}_head"))
            head_outputs[name] = head(h_CLS, H_genes_pert)
        return predictions, reps

    @property
    def num_parameters(self) -> dict[str, int]:
        """Parameter counts, with one entry per metabolism head."""
        counts = super().num_parameters
        total = counts.pop("total")
        for name in self.metabolism_head_names:
            head = cast(nn.Module, getattr(self, f"{name}_head"))
            n = sum(p.numel() for p in head.parameters() if p.requires_grad)
            counts[f"{name}_head"] = n
            total += n
        counts["total"] = total
        return counts

# torchcell/trainers/coo_targets
# [[torchcell.trainers.coo_targets]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/coo_targets
# Test file: tests/torchcell/trainers/test_coo_targets.py
"""Decode one head's dense targets and row mask from a batch's COO phenotype fields.

A collated ``Perturbation`` batch carries every phenotype value of every genotype as one
flat COO list on ``batch['gene']``:

* ``phenotype_values`` ``[V]``, the values;
* ``phenotype_type_indices`` ``[V]``, each value's phenotype type, LOCAL to its graph
  (index into that graph's ``phenotype_types`` name list);
* ``phenotype_values_batch`` ``[V]``, the batch row (graph) of each value;
* ``phenotype_sample_indices`` ``[V]``, which experiment WITHIN the genotype produced the
  value, also local to the graph.

A head selects the values whose type name is one of its phenotype names, splits them by
experiment, and keeps the experiment group(s) whose size equals the head's ``raw_dim``.
A scalar head averages its groups; a vector head must resolve to exactly one group, whose
values (restricted by ``keep`` to the measured features) are the target row.

Two implementations, held equal by ``tests/torchcell/trainers/test_coo_targets.py``:

* :func:`decode_head_targets_loop`, the original one-graph-at-a-time decode;
* :func:`decode_head_targets`, the same decode in a fixed number of tensor operations.

WHY THE SECOND EXISTS (2026-10-06, GilaHyper stack samples of live 019 runs). The loop
does about ten small operations per graph on device tensors, each a point where the CPU
waits for the GPU (``.tolist()``, ``bool(x.any())``, ``unique``), and a Python list
comprehension over every value of the graph. At batch 32 with 6,000 expression values per
graph that was 59 to 65 percent of a training process's wall time, more when several
processes share a card, because every wait queues behind the other processes' kernels.
"""

import torch


def decode_head_targets_loop(
    values: torch.Tensor,
    type_idx: torch.Tensor,
    val_batch: torch.Tensor,
    samp_idx: torch.Tensor,
    per_graph_types: list[list[str]],
    names: set[str],
    raw_dim: int,
    is_scalar: bool,
    keep: torch.Tensor | None,
    target: torch.Tensor,
    head: str = "head",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference decode, one graph at a time. Fills ``target`` in place and returns it.

    Args:
        values: ``[V]`` phenotype values.
        type_idx: ``[V]`` graph-local phenotype type index of each value.
        val_batch: ``[V]`` batch row of each value.
        samp_idx: ``[V]`` graph-local experiment index of each value.
        per_graph_types: one phenotype-type name list per graph.
        names: the head's phenotype names.
        raw_dim: number of values one measurement of this head carries.
        is_scalar: scalar head (replicate groups average) or vector head.
        keep: boolean ``[raw_dim]`` mask of the measured features, or ``None``.
        target: zero buffer ``[B]`` or ``[B, F]`` to fill.
        head: head name, for error messages.

    Returns:
        ``(target, row_mask)`` with ``row_mask`` ``[B]`` True where the head is supervised.
    """
    bsz = int(target.shape[0])
    device = target.device
    row_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    expected = int(target.shape[1]) if target.ndim > 1 else 1
    for b in range(bsz):
        sel_b = val_batch == b
        if not bool(sel_b.any()):
            continue
        gtypes = per_graph_types[b]
        tb = type_idx[sel_b].tolist()
        vb = values[sel_b]
        sb = samp_idx[sel_b]
        name_sel = torch.tensor([gtypes[t] in names for t in tb], dtype=torch.bool)
        if not bool(name_sel.any()):
            continue
        cand_vals = vb[name_sel]
        cand_samp = sb[name_sel]
        groups = [cand_vals[cand_samp == s] for s in sorted(set(cand_samp.tolist()))]
        groups = [g for g in groups if int(g.numel()) == raw_dim]
        if not groups:
            continue
        if is_scalar:
            head_vals = torch.stack([g.reshape(()) for g in groups]).mean()
            target[b] = head_vals.float().to(device)
        else:
            if len(groups) > 1:
                raise ValueError(
                    f"head '{head}' matched {len(groups)} value groups of width "
                    f"{raw_dim} in batch row {b}; a vector head must resolve to "
                    "exactly one measurement per genotype."
                )
            head_vals = groups[0]
            if keep is not None:
                head_vals = head_vals[keep]
            if int(head_vals.numel()) != expected:
                raise ValueError(
                    f"head '{head}' decoded {int(head_vals.numel())} target "
                    f"values but the head emits {expected}. Assigning these "
                    "would BROADCAST rather than align; fix the head's "
                    "output_dim / drop_features / head_phenotype_keys."
                )
            target[b] = head_vals.to(device)
        row_mask[b] = True
    return target, row_mask


def decode_head_targets(
    values: torch.Tensor,
    type_idx: torch.Tensor,
    val_batch: torch.Tensor,
    samp_idx: torch.Tensor,
    per_graph_types: list[list[str]],
    names: set[str],
    raw_dim: int,
    is_scalar: bool,
    keep: torch.Tensor | None,
    target: torch.Tensor,
    head: str = "head",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode of :func:`decode_head_targets_loop` without a loop over graphs.

    Same arguments, same return, same errors. The only Python loop left is over each
    graph's phenotype-type NAMES (one to a few per graph) to build the name table.
    """
    bsz = int(target.shape[0])
    device = target.device
    expected = int(target.shape[1]) if target.ndim > 1 else 1
    row_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    if values.numel() == 0:
        return target, row_mask

    # name_ok[b, t]: is graph b's local type t one of this head's phenotypes.
    max_types = max(len(t) for t in per_graph_types)
    name_ok_cpu = torch.zeros(len(per_graph_types), max(max_types, 1), dtype=torch.bool)
    for b, gtypes in enumerate(per_graph_types):
        for t, name in enumerate(gtypes):
            if name in names:
                name_ok_cpu[b, t] = True
    name_ok = name_ok_cpu.to(device)
    name_sel = name_ok[val_batch, type_idx]  # [V]

    # One id per (graph, experiment); keep the groups of exactly raw_dim selected values.
    n_samp = int(samp_idx.max().item()) + 1
    group_id = val_batch * n_samp + samp_idx  # [V]
    counts = torch.bincount(group_id[name_sel], minlength=bsz * n_samp)
    keep_group = counts == raw_dim  # [bsz * n_samp]
    sel = name_sel & keep_group[group_id]  # [V]
    groups_per_graph = keep_group.view(bsz, n_samp).sum(dim=1)  # [bsz]

    if is_scalar:
        b_of = val_batch[sel]
        sums = torch.zeros(bsz, dtype=torch.float32, device=device)
        sums = sums.index_add(0, b_of, values[sel].float())
        rows = groups_per_graph > 0
        mean = sums / groups_per_graph.clamp(min=1).to(sums.dtype)
        mean = mean.view(-1, *([1] * (target.ndim - 1))).to(target.dtype)
        target = torch.where(rows.view(-1, *([1] * (target.ndim - 1))), mean, target)
        return target, rows

    if bool((groups_per_graph > 1).any()):
        b_bad = int((groups_per_graph > 1).nonzero()[0].item())
        raise ValueError(
            f"head '{head}' matched {int(groups_per_graph[b_bad].item())} value groups "
            f"of width {raw_dim} in batch row {b_bad}; a vector head must resolve to "
            "exactly one measurement per genotype."
        )
    if not bool(sel.any()):
        return target, row_mask
    # Rows in (graph, experiment) order with each group's values in their COO order.
    order = torch.argsort(group_id[sel], stable=True)
    vals = values[sel][order].view(-1, raw_dim)
    b_rows = val_batch[sel][order].view(-1, raw_dim)[:, 0]
    if keep is not None:
        vals = vals[:, keep.to(device)]
    if int(vals.shape[1]) != expected:
        raise ValueError(
            f"head '{head}' decoded {int(vals.shape[1])} target "
            f"values but the head emits {expected}. Assigning these "
            "would BROADCAST rather than align; fix the head's "
            "output_dim / drop_features / head_phenotype_keys."
        )
    target[b_rows] = vals.to(target.dtype)
    row_mask[b_rows] = True
    return target, row_mask

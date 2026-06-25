---
id: opxiayi087gjm3wggrngekq
title: Ruff up Breaks Pyg Messagepassing
desc: ''
updated: 1782411212336
created: 1782411212336
---

## 2026.06.25 - ruff UP (PEP 604 unions) breaks PyG MessagePassing

**Symptom.** A blanket `ruff check --fix` over `torchcell/` regressed all 13 tests in
`tests/torchcell/nn/test_stoichiometric_hypergraph_conv.py` with:

```
AttributeError: 'types.UnionType' object has no attribute '__qualname__'
  torch_geometric/inspector.py:type_repr -> _get_name(obj.__qualname__, ...)
  via MessagePassing.__init__ -> Inspector.inspect_signature(self.message)
```

**Cause.** ruff's `UP` rules (`UP045` `Optional[X]->X | None`, `UP007`
`Union[X,Y]->X | Y`) rewrote type hints in a `MessagePassing.message()` signature.
`torch_geometric.nn.MessagePassing` **inspects that signature at runtime** to wire up
message passing; its inspector calls `obj.__qualname__`, which `types.UnionType`
(the `X | Y` object) does not have, so layer construction raises.

**Not a version bug.** PyG issue
[#10138](https://github.com/pyg-team/pytorch_geometric/issues/10138) is **closed as
"not planned"** -- maintainers deliberately do not support PEP 604 unions in inspected
methods. torch_geometric 2.8 / torch 2.12 do **not** fix it. Upgrading is unrelated.

**Fix.** Keep `Optional[...]`/`Union[...]` (typing module) in MessagePassing
`message()`/`forward()` signatures. In `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`,
ignore `UP007`, `UP045` for the MessagePassing modules. torchcell has exactly two:

- `torchcell/nn/stoichiometric_hypergraph_conv.py`
- `torchcell/nn/masked_gin_conv.py`

If a new MessagePassing subclass is added, add it to that ignore list (and grep
`MessagePassing` over `torchcell/` to re-scope). Context: [[plan.ci-foundation-ruff-mypy-pytest.2026.06.18]].

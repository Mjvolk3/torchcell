# experiments/019-simb-multimodal/scripts/verify_head_decode.py
# [[experiments.019-simb-multimodal.scripts.verify_head_decode]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/verify_head_decode
"""Verify the three harness decode defects are actually fixed, on synthetic COO batches.

The defects all produced a PLAUSIBLE NUMBER rather than an error, which is why they went
unnoticed; a verification that only checks "it runs" would miss them again. Each check
below asserts on the decoded VALUES, or asserts that a raise happens.

1. LABEL COLLISION. Betaxanthin and the Mulleder metabolome are both
   ``MetabolitePhenotype`` -> both label ``metabolite_level``, and the ``Perturbation``
   processor drops the dict keys. Selecting by label name alone concatenated 1 + 19 values
   into a 20-value blob. Check: on a genotype carrying BOTH, the betaxanthin head must
   decode exactly the betaxanthin value and the 19-D head exactly the 19 amino acids.
2. BROADCAST. ``target[b] = head_vals`` with a 1-element ``head_vals`` and an ``[F]`` row
   is a legal torch broadcast, so one scalar was silently copied across all F columns.
   Check: a width mismatch must RAISE.
3. SCALAR/VECTOR EXPLICITNESS. ``is_scalar`` was hardcoded True only for
   ``gene_interaction``. Check: the ordinal ``visual_score`` head decodes as a scalar, and
   two heads sharing one label with EQUAL widths is rejected at alignment time (they would
   be indistinguishable once the keys are dropped).

Writes ``results/head_decode_verification.json``.

Run:  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/verify_head_decode.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from train_cgt_multitask import (  # type: ignore[import-not-found]  # noqa: E402
    MultitaskCGTTask,
    build_head_alignments,
)

AA = [
    "alanine", "arginine", "asparagine", "aspartate", "glutamate", "glutamine",
    "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine",
    "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine",
    "valine",
]
BX_VALUE = 0.7
AA_VALUES = [round(0.1 * (i + 1), 3) for i in range(19)]
SCORE_VALUE = 3.0


class _FakeDataset:
    """Minimal stand-in exposing only what ``build_head_alignments`` reads."""

    def __init__(self, records: dict[str, list[dict[str, Any]]]) -> None:
        """Store label -> list of raw ``{"experiment": {"phenotype": {...}}}`` items."""
        self._records = records
        self.env = object()
        self.phenotype_label_index = {k: list(range(len(v))) for k, v in records.items()}

    def _read_from_lmdb(self, idx: int) -> int:
        return idx

    def _deserialize_json(self, raw: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for items in self._records.values():
            if raw < len(items):
                out.append(items[raw])
        return out


def _fake_dataset() -> _FakeDataset:
    """One record carrying betaxanthin, the 19-AA metabolome, and a visual score."""
    return _FakeDataset(
        {
            "metabolite_level": [
                {"experiment": {"phenotype": {"metabolite_level": {"betaxanthin": BX_VALUE}}}},
                {
                    "experiment": {
                        "phenotype": {
                            "metabolite_level": dict(zip(AA, AA_VALUES, strict=True))
                        }
                    }
                },
            ],
            "visual_score": [
                {"experiment": {"phenotype": {"visual_score": SCORE_VALUE}}}
            ],
        }
    )


def _batch(groups: list[list[tuple[str, list[float]]]], types: list[str]) -> HeteroData:
    """Build a COO batch: ``groups[b]`` is that row's (label, values) experiments."""
    values: list[float] = []
    type_idx: list[int] = []
    samp_idx: list[int] = []
    val_batch: list[int] = []
    for b, row in enumerate(groups):
        for s, (label, vals) in enumerate(row):
            values.extend(vals)
            type_idx.extend([types.index(label)] * len(vals))
            samp_idx.extend([s] * len(vals))
            val_batch.extend([b] * len(vals))
    batch = HeteroData()
    gene = batch["gene"]
    gene.phenotype_values = torch.tensor(values, dtype=torch.float)
    gene.phenotype_type_indices = torch.tensor(type_idx, dtype=torch.long)
    gene.phenotype_sample_indices = torch.tensor(samp_idx, dtype=torch.long)
    gene.phenotype_values_batch = torch.tensor(val_batch, dtype=torch.long)
    gene["phenotype_types"] = [list(types) for _ in groups]
    return batch


def _task(head_align: dict[str, dict[str, Any]], heads: list[str]) -> MultitaskCGTTask:
    """A MultitaskCGTTask with no model, used only for its decode method."""
    task = MultitaskCGTTask.__new__(MultitaskCGTTask)
    task.head_phenotypes = {
        "betaxanthin": ["metabolite_level"],
        "mulleder19": ["metabolite_level"],
        "beta_carotene": ["visual_score"],
    }
    task.head_align = head_align
    task.dist_heads = {}
    task.norm_heads = []
    task.active_heads = heads
    return task


def main() -> None:
    """Run the three checks and write the verification report."""
    report: dict[str, Any] = {}
    ds = _fake_dataset()
    align = build_head_alignments(
        dataset=ds,
        active_heads=["betaxanthin", "mulleder19", "beta_carotene"],
        head_phenotypes={
            "betaxanthin": ["metabolite_level"],
            "mulleder19": ["metabolite_level"],
            "beta_carotene": ["visual_score"],
        },
        node_ids=[],
        scalar_heads=["betaxanthin", "beta_carotene"],
        head_phenotype_keys={"betaxanthin": ["betaxanthin"], "mulleder19": AA},
    )
    report["alignment"] = {
        h: {
            "is_scalar": a["is_scalar"],
            "feat_dim": a["feat_dim"],
            "raw_dim": a["raw_dim"],
            "n_keys": None if a["keys"] is None else len(a["keys"]),
        }
        for h, a in align.items()
    }
    assert align["betaxanthin"]["is_scalar"] is True
    assert align["betaxanthin"]["raw_dim"] == 1
    assert align["beta_carotene"]["is_scalar"] is True
    assert align["mulleder19"]["is_scalar"] is False
    assert align["mulleder19"]["raw_dim"] == 19
    assert align["mulleder19"]["keys"] == AA

    # ---- CHECK 1: label collision. One genotype carries all three phenotypes. ----
    task = _task(align, ["betaxanthin", "mulleder19", "beta_carotene"])
    batch = _batch(
        [
            [
                ("metabolite_level", [BX_VALUE]),
                ("visual_score", [SCORE_VALUE]),
                ("metabolite_level", AA_VALUES),
            ]
        ],
        ["metabolite_level", "visual_score"],
    )
    head_outputs = {
        "betaxanthin": torch.zeros(1, 1),
        "mulleder19": torch.zeros(1, 19),
        "beta_carotene": torch.zeros(1, 1),
    }
    targets, masks = task._extract_targets_and_masks(batch, head_outputs, 1)
    got_bx = float(targets["betaxanthin"][0, 0])
    got_aa = [round(float(v), 3) for v in targets["mulleder19"][0]]
    got_bc = float(targets["beta_carotene"][0, 0])
    report["check_1_label_collision"] = {
        "betaxanthin_decoded": got_bx,
        "betaxanthin_expected": BX_VALUE,
        "mulleder19_decoded": got_aa,
        "mulleder19_expected": AA_VALUES,
        "beta_carotene_decoded": got_bc,
        "beta_carotene_expected": SCORE_VALUE,
        "all_masks_true": all(bool(m[0]) for m in masks.values()),
    }
    assert abs(got_bx - BX_VALUE) < 1e-6, got_bx
    assert got_aa == AA_VALUES, got_aa
    assert abs(got_bc - SCORE_VALUE) < 1e-6, got_bc
    assert all(bool(m[0]) for m in masks.values())

    # ---- CHECK 2: a width mismatch RAISES instead of broadcasting. ----
    # The head declares 19 columns; the record supplies 19 values but the head's model
    # output is 18 wide -- the exact shape the old code broadcast into silently.
    narrow = {"mulleder19": torch.zeros(1, 18)}
    narrow_batch = _batch(
        [[("metabolite_level", AA_VALUES)]], ["metabolite_level", "visual_score"]
    )
    narrow_task = _task({"mulleder19": align["mulleder19"]}, ["mulleder19"])
    raised = ""
    try:
        narrow_task._extract_targets_and_masks(narrow_batch, narrow, 1)
    except ValueError as e:
        raised = str(e)
    report["check_2_broadcast_guard"] = {"raised": raised}
    assert "BROADCAST" in raised, f"expected a broadcast guard, got: {raised!r}"

    # The pre-fix behaviour, shown explicitly: assigning 1 value into a 19-wide row is a
    # legal torch broadcast that silently fills every column.
    demo = torch.zeros(1, 19)
    demo[0] = torch.tensor([BX_VALUE])
    report["check_2_broadcast_demo"] = {
        "assigning_1_value_into_19_columns_gives": [
            round(float(v), 3) for v in demo[0][:4]
        ],
        "note": "this is what the old code did with a scalar target in a vector head",
    }
    assert abs(float(demo[0, 18]) - BX_VALUE) < 1e-6

    # ---- CHECK 3: two heads sharing a label with EQUAL widths is rejected. ----
    rejected = ""
    try:
        build_head_alignments(
            dataset=ds,
            active_heads=["betaxanthin", "betaxanthin_twin"],
            head_phenotypes={
                "betaxanthin": ["metabolite_level"],
                "betaxanthin_twin": ["metabolite_level"],
            },
            node_ids=[],
            scalar_heads=["betaxanthin", "betaxanthin_twin"],
            head_phenotype_keys={
                "betaxanthin": ["betaxanthin"],
                "betaxanthin_twin": ["betaxanthin"],
            },
        )
    except ValueError as e:
        rejected = str(e)
    report["check_3_equal_width_collision_rejected"] = {"raised": rejected}
    assert "DISTINCT value counts" in rejected, rejected

    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, "head_decode_verification.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nALL CHECKS PASSED -- wrote {out}")


if __name__ == "__main__":
    main()

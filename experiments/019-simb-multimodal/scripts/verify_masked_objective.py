# experiments/019-simb-multimodal/scripts/verify_masked_objective.py
# [[experiments.019-simb-multimodal.scripts.verify_masked_objective]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/verify_masked_objective
"""Contracts for the teacher-forced masked-label objective.

Each check targets a way this objective FAILS SILENTLY -- producing a plausible number that
means nothing. None of these is a style test.

C1  masked_mean scores only the selected entries. The whole objective rests on this one
    reduction; if it leaks, revealed genes get scored.
C2  feature_mask=None is EXACTLY the old behaviour. Every arm already running must be
    bit-identical, or this change silently re-baselines the project.
C3  A revealed gene contributes ZERO GRADIENT. This is the copying failure in its most
    direct form: if a revealed entry can push gradient, the model is being trained to
    reproduce its own input and train metrics inflate for free.
C4  Nested observed sets. M_1 subset M_2 subset ... -- the validation sweep is only an
    unmasking TRAJECTORY if the sets grow by inclusion rather than being K unrelated draws.
C5  Reveal counts are exact, and -1 means all.
C6  The token-space scatter is the inverse of the gather. Observed values must land on the
    graph nodes that `col_idx` maps the measured genes to; an off-by-one here would feed the
    model the right values attached to the wrong genes and still train.
C7  Unsupervised rows reveal nothing. A row with no ground truth has nothing to teacher-force,
    and revealing zeros would teach "unmeasured == 0".

Usage
-----
    python experiments/019-simb-multimodal/scripts/verify_masked_objective.py

Writes ``results/verify_masked_objective.json``; exits non-zero if any contract fails.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import torch
from dotenv import load_dotenv

from torchcell.losses.distributional import DistHead, masked_mean

load_dotenv()

RESULTS: list[dict[str, object]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"contract": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    torch.manual_seed(0)
    B, F, K = 6, 40, 19

    # ---- C1: masked_mean selects exactly the marked entries -------------------------
    elem = torch.arange(B * F, dtype=torch.float32).reshape(B, F)
    fm = torch.zeros(B, F, dtype=torch.bool)
    fm[0, :3] = True
    expected = elem[0, :3].mean()
    got = masked_mean(elem, fm)
    check(
        "C1_masked_mean_selects",
        torch.allclose(got, expected),
        f"got {got.item():.6f} vs expected {expected.item():.6f}",
    )

    # ---- C2: feature_mask=None reproduces the unmasked reduction exactly --------------
    # K knots come from the quantile grid, not a param_dim kwarg.
    taus = torch.linspace(0.05, 0.95, K)
    head = DistHead(mode="quantile", quantiles=taus)
    params = torch.randn(B, F, K)
    target = torch.randn(B, F)
    a = head.loss(params, target)
    b = head.loss(params, target, feature_mask=None)
    c = head.loss(params, target, feature_mask=torch.ones(B, F, dtype=torch.bool))
    check(
        "C2_none_is_identity",
        torch.allclose(a, b) and torch.allclose(a, c),
        f"none={a.item():.8f} explicit_none={b.item():.8f} all_true={c.item():.8f}",
    )

    # ---- C3: a revealed gene contributes zero gradient --------------------------------
    p = torch.randn(B, F, K, requires_grad=True)
    hidden = torch.zeros(B, F, dtype=torch.bool)
    hidden[:, : F // 2] = True  # score only the first half
    head.loss(p, target, feature_mask=hidden).backward()
    assert p.grad is not None
    g_hidden = p.grad[:, : F // 2].abs().sum().item()
    g_revealed = p.grad[:, F // 2 :].abs().sum().item()
    check(
        "C3_revealed_zero_grad",
        g_revealed == 0.0 and g_hidden > 0.0,
        f"|grad| hidden={g_hidden:.4f} revealed={g_revealed:.3e}",
    )

    # ---- C4/C5/C7: sampler semantics --------------------------------------------------
    sys.path.insert(
        0, osp.join(os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "scripts")
    )
    from train_cgt_multitask import MultitaskCGTTask as T

    scores = torch.rand(B, F)
    sets = [T._observed_feature_mask(None, B, F, n, scores) for n in (0, 5, 10, 20, -1)]
    nested = all(
        bool((sets[i] & sets[i + 1]).equal(sets[i])) for i in range(len(sets) - 1)
    )
    check(
        "C4_nested_sets",
        nested,
        "M_k subset M_{k+1} for every consecutive pair"
        if nested
        else "NOT nested -- the sweep is not a trajectory",
    )
    counts = [int(s[0].sum()) for s in sets]
    check(
        "C5_reveal_counts",
        counts == [0, 5, 10, 20, F],
        f"per-row revealed = {counts} (expected [0, 5, 10, 20, {F}])",
    )
    row_mask = torch.tensor([True, True, False, True, False, True])
    gated = sets[3] & row_mask.unsqueeze(1)
    check(
        "C7_unsupervised_rows_reveal_nothing",
        int(gated[~row_mask].sum()) == 0,
        f"revealed entries on unsupervised rows = {int(gated[~row_mask].sum())}",
    )

    # ---- C6: scatter is the inverse of the gather -------------------------------------
    n_nodes = 100
    col = torch.randperm(n_nodes)[:F]  # measured-feature -> node
    tgt = torch.randn(B, F)
    obs = torch.zeros(B, F, dtype=torch.bool)
    obs[:, ::3] = True

    class _Stub:
        cell_graph = {"gene": type("g", (), {"num_nodes": n_nodes})()}

    vals, msk = T._to_token_space(_Stub(), obs, tgt, col)
    # gather back exactly as _gather_predictions would
    back_v = vals.index_select(1, col)
    back_m = msk.index_select(1, col).bool()
    round_trip = torch.allclose(back_v, tgt * obs) and bool(back_m.equal(obs))
    off_target = float(msk.sum()) == float(obs.sum())
    check(
        "C6_scatter_is_gather_inverse",
        round_trip and off_target,
        f"round-trip exact={round_trip}, no mass off the mapped nodes={off_target}",
    )

    out_dir = osp.join(os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, "verify_masked_objective.json")
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nwrote {path}")

    failed = [r for r in RESULTS if not r["pass"]]
    if failed:
        print(f"\n{len(failed)} CONTRACT(S) FAILED -- do not launch")
        raise SystemExit(1)
    print("\nall contracts hold -- safe to launch")


if __name__ == "__main__":
    main()

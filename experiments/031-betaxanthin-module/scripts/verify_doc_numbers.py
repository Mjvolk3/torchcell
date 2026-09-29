# experiments/031-betaxanthin-module/scripts/verify_doc_numbers.py
# [[experiments.031-betaxanthin-module.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/verify_doc_numbers

"""Every number typed into notes-tex/031-betaxanthin-module's prose, checked against its source file.

The essentiality verifier's rule and helpers, applied to this document: each check names
the claim, the quantity as the document states it, and how to recompute it from the
study's results tree. Add one whenever a number is written into prose. Exits non-zero
on any mismatch. Run from the repo root::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/verify_doc_numbers.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "scripts")
)
from verify_doc_numbers import (  # noqa: E402
    Check,
    degeneracy,
    ensemble,
    physics_store,
    store_baseline,
    transfer,
    transfer_comparator,
)

RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "031-betaxanthin-module", "results")


def driving_force(variant: str, *keys: str):
    """A value of the driving-force store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(
                    RESULTS, f"target_driving_force_store_betaxanthin_{variant}.json"
                )
            )
        )
        for k in keys:
            d = d[k]
        return float(d)

    return run


def polytope(*keys: str):
    """A value of the concentration-polytope store's summary, by key path."""

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(
                    RESULTS,
                    "target_concentration_polytope_store_betaxanthin_mdfcap100poly.json",
                )
            )
        )
        for k in keys:
            d = d[k]
        return float(d)

    return run


def go_model(arm: str, key: str):
    """A value of one arm of the GO-models read (target_go_models.py)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "go_models_betaxanthin.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "n_columns_with_train_support":
            return float(d["n_columns_with_train_support"])
        if key == "select_epoch_min":
            return float(min(r["select_epoch"] for r in a["seeds"]))
        if key == "select_epoch_max":
            return float(max(r["select_epoch"] for r in a["seeds"]))
        return float(a[key])

    return run


def raw_mlp(name: str, arm: str, key: str):
    """A value of one arm of a raw-encoding (log1p) CPU perceptron run over a store
    (target_go_models.py --store-raw); ``name`` is the result file's stem.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"go_models_betaxanthin_{name}.json")))
        assert d.get("store_raw") is True or "raw" in name, name
        a = next(x for x in d["arms"] if x["arm"] == arm)
        return float(a[key])

    return run


def cooper_mlp(name: str, arm: str, key: str):
    """A value of one arm of a CPU perceptron run over Cooper's store or the pool with
    Cooper's peaks appended (target_go_models.py --store YPD_bx_cooperpool or
    YPD_bx_aapoolcooper); ``name`` is the result file's stem after ``go_models_betaxanthin_``.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"go_models_betaxanthin_{name}.json")))
        assert d["store"] in ("YPD_bx_cooperpool", "YPD_bx_aapoolcooper"), name
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "select_epoch_min":
            return float(min(r["select_epoch"] for r in a["seeds"]))
        if key == "select_epoch_max":
            return float(max(r["select_epoch"] for r in a["seeds"]))
        return float(a[key])

    return run


def cooper_mlp_paired_wins(name: str, arm: str, ref_name: str, ref_arm: str):
    """Seeds on which the Cooper run's arm reads above the reference pool run's arm."""

    def run() -> float:
        import json

        a = json.load(open(osp.join(RESULTS, f"go_models_betaxanthin_{name}.json")))
        b = json.load(open(osp.join(RESULTS, f"go_models_betaxanthin_{ref_name}.json")))
        x = {
            r["seed"]: r["test_pinned_spearman"]
            for r in next(t for t in a["arms"] if t["arm"] == arm)["seeds"]
        }
        y = {
            r["seed"]: r["test_pinned_spearman"]
            for r in next(t for t in b["arms"] if t["arm"] == ref_arm)["seeds"]
        }
        assert set(x) == set(y)
        return float(sum(x[k] > y[k] for k in x))

    return run


def decomposition(name: str, key: str):
    """A summary value of a target_head_decomposition run (``name`` its stem)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"target_head_decomposition_{name}.json")))
        return float(d["summary"][key])

    return run


def decomposition_cells(name: str, what: str):
    """Over a decomposition's cells: ``n_constant`` (head spread under a hundredth of
    the slot's), ``max_full_vs_reported`` (the rebuild against the reported Spearman),
    ``max_sd_ratio_constant``, or the min or max over LIVE cells of ``backbone_adds``,
    ``head``, ``head_vs_slot``, or of ``select_epoch`` over all cells.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"target_head_decomposition_{name}.json")))
        cells = d["cells"]
        live = [c for c in cells if c["sd_ratio_head_over_slot"] >= 1e-2]
        if what == "n_constant":
            return float(len(cells) - len(live))
        if what == "max_full_vs_reported":
            return max(
                abs(c["full"]["spearman_vs_target"] - c["reported_spearman"])
                for c in cells
            )
        if what == "max_sd_ratio_constant":
            return max(c["sd_ratio_head_over_slot"] for c in cells if c not in live)
        stat, key = what.split("_", 1)
        f = {"min": min, "max": max}[stat]
        if key == "select_epoch":
            return float(f(c["select_epoch"] for c in cells))
        if key == "backbone_adds":
            return float(f(c["backbone_adds"] for c in live))
        if key == "head":
            return float(f(c["backbone_head"]["spearman_vs_target"] for c in live))
        if key == "head_vs_slot":
            return float(f(c["spearman_head_vs_slot"] for c in live))
        raise KeyError(what)

    return run


def agreement(name: str, key: str):
    """A summary value of a target_prediction_agreement run."""

    def run() -> float:
        import json

        d = json.load(
            open(osp.join(RESULTS, f"target_prediction_agreement_{name}.json"))
        )
        return float(d["summary"][key])

    return run


def agreement_max_below(name: str, ceiling: float):
    """The largest same-seed agreement under ``ceiling`` (the live seeds' band)."""

    def run() -> float:
        import json

        d = json.load(
            open(osp.join(RESULTS, f"target_prediction_agreement_{name}.json"))
        )
        return max(r["spearman"] for r in d["seeds"] if r["spearman"] < ceiling)

    return run


def graph_reg_loss_over_cells(layers: int, what: str, max_wave: int = 219):
    """Over every result file of the study with ``layers`` transformer blocks up to wave
    ``max_wave`` at seeds 2700 to 2705 (the cells landed when the decomposition was
    read): the count of cells,
    or the maximum graph regularization loss logged at any epoch.
    """

    def run() -> float:
        import glob
        import json
        import re

        mx, n = 0.0, 0
        for path in glob.glob(osp.join(RESULTS, "w*_s27*.json")):
            wave = re.match(r"w(\d+)_.*_s(\d+)\.json$", osp.basename(path))
            if (
                wave is None
                or int(wave.group(1)) > max_wave
                or int(wave.group(2)) > 2705
            ):
                continue
            d = json.load(open(path))
            if "history" not in d or d["cell"]["layers"] != layers:
                continue
            n += 1
            mx = max(mx, max(x.get("graph_reg_loss", 0.0) for x in d["history"]))
        return float(n if what == "n" else mx)

    return run


def coverage(name: str, stem: str, key: str):
    """An arm's mean over seeds from a target_coverage_split run (``key`` an ArmSplit
    mean), or a first-seed count/target statistic (``n_pinned``, ``n_covered``,
    ``mean_target_covered``, ``mean_target_uncovered``, ``coverage_indicator_spearman``).
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"target_coverage_split_{name}.json")))
        a = next(x for x in d["arms"] if x["stem"] == stem)
        if key in a:
            return float(a[key])
        return float(a["seeds"][0][key])

    return run


def fit_variant(name: str, key: str):
    """A value of the pool perceptron's fit variant ``name`` (store_mlp arm); ``covered``
    is the mean over seeds of the read on the pinned genes the pool covers.
    """

    def run() -> float:
        import json

        d = json.load(
            open(osp.join(RESULTS, f"go_models_betaxanthin_pool_fit_{name}.json"))
        )
        a = next(x for x in d["arms"] if x["arm"] == "store_mlp")
        if key == "covered":
            v = [r["test_pinned_spearman_covered"] for r in a["seeds"]]
            assert all(x is not None for x in v)
            return float(sum(v) / len(v))
        return float(a[key])

    return run


def pool_mlp_covered(name: str):
    """The mean covered-genes read of a pool sweep file that recorded it."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, name)))
        a = next(x for x in d["arms"] if x["arm"] == "store_mlp")
        v = [r["test_pinned_spearman_covered"] for r in a["seeds"]]
        assert all(x is not None for x in v), name
        return float(sum(v) / len(v))

    return run


def store_baseline_covered(
    target: str, rep: str, model: str, key: str = "mean_spearman"
):
    """The ladder's read on the pinned genes the store covers (target_store_baselines.py,
    2026-09-27: ``test_pinned_covered``).
    """

    def run() -> float:
        import json

        study = {
            "betaxanthin": "031-betaxanthin-module",
            "amino_acids": "032-amino-acid-module",
        }[target]
        d = json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    study,
                    "results",
                    f"store_baselines_{target}.json",
                )
            )
        )
        block = d["representations"][rep][model]["selected_on_val"][
            "test_pinned_covered"
        ]
        assert block is not None, (rep, model)
        return float(block[key])

    return run


def covered_read(name: str, arm: str, key: str):
    """A value of an arm of a go_models file, or ``covered`` for the mean over seeds of
    the read on the pinned genes the pool covers.
    """

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"go_models_betaxanthin_{name}.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "covered":
            v = [r["test_pinned_spearman_covered"] for r in a["seeds"]]
            assert all(x is not None for x in v), (name, arm)
            return float(sum(v) / len(v))
        return float(a[key])

    return run


def pair_seed_subset(pair: str, side: str, seeds: tuple[int, ...], key: str):
    """The mean (``key`` ``mean``) or sample sd (``sd``) of the spearman over the named
    seeds of one side of a transfer-readout pair, or the standard error (``se``).
    """

    def run() -> float:
        import json

        import numpy as np

        d = json.load(open(osp.join(RESULTS, "transfer_readout.json")))
        p = next(q for q in d["pairs"] if q["name"] == pair)
        v = np.array([c["spearman"] for c in p[side]["cells"] if c["seed"] in seeds])
        assert len(v) == len(seeds), (len(v), seeds)
        if key == "mean":
            return float(v.mean())
        if key == "sd":
            return float(v.std(ddof=1))
        if key == "se":
            return float(v.std(ddof=1) / np.sqrt(len(v)))
        raise KeyError(key)

    return run


def release(stem: str, what: str):
    """Over an arm's result files: how many cells released their slot (``n_released``),
    the release epoch they share (``epoch``), the mean or sd of the pinned Spearman over
    the released cells (``mean``, ``sd``), the unreleased cell's read (``unreleased``),
    or the released cells' smallest selected epoch (``select_epoch_min``).
    """

    def run() -> float:
        import glob
        import json

        import numpy as np

        rows = []
        for path in sorted(glob.glob(osp.join(RESULTS, f"{stem}_s27*.json"))):
            d = json.load(open(path))
            rows.append(
                (
                    d.get("affine_start_epoch_actual"),
                    float(d["test"]["pinned"]["spearman"]),
                    int(d["select_epoch"]),
                )
            )
        rel = [r for r in rows if r[0] is not None]
        if what == "n_released":
            return float(len(rel))
        if what == "epoch":
            epochs = {r[0] for r in rel}
            assert len(epochs) == 1, epochs
            return float(next(iter(epochs)))
        if what == "mean":
            return float(np.mean([r[1] for r in rel]))
        if what == "sd":
            return float(np.std([r[1] for r in rel], ddof=1))
        if what == "select_epoch_min":
            return float(min(r[2] for r in rel))
        if what == "unreleased":
            out = [r[1] for r in rows if r[0] is None]
            assert len(out) == 1, len(out)
            return float(out[0])
        raise KeyError(what)

    return run


def pool_mlp(hidden: int, arm: str, key: str, name: str | None = None):
    """A value of one arm of the CPU perceptron sweep over the measured pool
    (target_go_models.py --store YPD_bx_aapool --mlp-hidden <hidden>); ``name`` picks a
    result file past the width series (the two-slot and depth runs).
    """

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(RESULTS, name or f"go_models_betaxanthin_pool_h{hidden}.json")
            )
        )
        assert d["store"] == "YPD_bx_aapool" and d["mlp_hidden"] == hidden
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "select_epoch_min":
            return float(min(r["select_epoch"] for r in a["seeds"]))
        if key == "select_epoch_max":
            return float(max(r["select_epoch"] for r in a["seeds"]))
        if key == "seconds_max":
            return float(max(r["seconds"] for r in a["seeds"]))
        return float(a[key])

    return run


def go_model_w20(arm: str, key: str):
    """A value of one arm of the width-20 GO-models read (results/go_models_betaxanthin_w20.json)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "go_models_betaxanthin_w20.json")))
        a = next(x for x in d["arms"] if x["arm"] == arm)
        if key == "n_select_at_200":
            return float(sum(1 for s in a["seeds"] if s["select_epoch"] == 200))
        if key == "seconds_min":
            return float(min(s["seconds"] for s in a["seeds"]) / 60)
        if key == "seconds_max":
            return float(max(s["seconds"] for s in a["seeds"]) / 60)
        return float(a[key])

    return run


def cross_target(key: str):
    """A value of the betaxanthin-against-amino-acid read, from the 032 results tree."""

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    "032-amino-acid-module",
                    "results",
                    "betaxanthin_amino_acid_correlation.json",
                )
            )
        )
        if key == "n_shared_mulleder":
            return float(d["n_shared_mulleder"])
        inputs, _, field = key.rpartition(":")
        return float(
            next(
                p
                for p in d["pool"]
                if p["screen"] == "mulleder2016" and p["inputs"] == (inputs or "pool")
            )[field]
        )

    return run


def seed_level(stem: str, which: str):
    """The single-seed level of an arm over a seed subset of the ensemble readout:
    ``first6``, ``last6`` or ``all12`` mean, ``sd``/``sem`` over all seeds.
    """

    def run() -> float:
        import json

        import numpy as np

        d = json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    "031-betaxanthin-module",
                    "results",
                    "ensemble_readout.json",
                )
            )
        )
        a = next(q for q in d["arms"] if q["stem"] == stem)
        v = np.array([x["spearman"] for x in a["single"]], dtype=float)
        seeds = a["seeds"]
        assert seeds == sorted(seeds), stem
        if which == "first6":
            return float(v[:6].mean())
        if which == "last6":
            return float(v[6:].mean())
        if which == "last6_sd":
            return float(v[6:].std(ddof=1))
        if which == "all12":
            return float(v.mean())
        if which == "sd":
            return float(v.std(ddof=1))
        if which == "sem":
            return float(v.std(ddof=1) / np.sqrt(v.size))
        if which == "min":
            return float(v.min())
        if which == "max":
            return float(v.max())
        if which == "first6_min":
            return float(v[:6].min())
        if which == "first6_max":
            return float(v[:6].max())
        if which == "mean_top50":
            return float(np.mean([x["top50_enrichment"] for x in a["single"]]))
        raise KeyError(which)

    return run


def pair_cells(pair: str, side: str, key: str, which: str):
    """The per-seed cells of one side of a transfer-readout pair: ``min`` or ``max`` of
    ``select_epoch``, ``select_val_pearson`` or ``spearman``.
    """

    def run() -> float:
        import json

        d = json.load(
            open(
                osp.join(
                    os.environ["EXPERIMENT_ROOT"],
                    "031-betaxanthin-module",
                    "results",
                    "transfer_readout.json",
                )
            )
        )
        p = next(q for q in d["pairs"] if q["name"] == pair)
        v = [c[key] for c in p[side]["cells"]]
        if which == "min":
            return float(min(v))
        if which == "max":
            return float(max(v))
        raise KeyError(which)

    return run



def homophily(target: str, graph: str, hops: int, key: str):
    """A value of one graph at one reach from target_graph_homophily.py, or a run-level
    count (``n_labeled``, ``n_pinned``)."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, f"target_graph_homophily_{target}.json")))
        if key in ("n_labeled", "n_pinned"):
            return float(d[key])
        r = next(x for x in d["reads"] if x["graph"] == graph and x["hops"] == hops)
        return float(r[key])

    return run


def pack_log(stem: str, seed: int, epoch: int, what: str):
    """``loss``, ``data`` or their ``ratio`` (penalty over data) at one epoch of a
    pack cell's slurm log, the trainer's ``[pooled sSEED] ep N loss L data D`` line."""

    def run() -> float:
        import glob

        (path,) = glob.glob(
            osp.join(osp.dirname(RESULTS), "slurm", "output", f"*_{stem}_s{seed}.out")
        )
        for line in open(path):
            parts = line.split()
            if len(parts) > 8 and parts[2] == "ep" and int(parts[3]) == epoch and parts[4] == "loss":
                loss, data = float(parts[5]), float(parts[7])
                return {"loss": loss, "data": data, "ratio": (loss - data) / data}[what]
        raise AssertionError(f"epoch {epoch} not in {path}")

    return run



def ladder_read(rep: str, model: str, key: str = "mean_spearman", covered: bool = False):
    """A betaxanthin ladder read (store_baselines_betaxanthin.json), on all pinned genes
    or on the genes the store covers; ``key`` is ``mean_spearman`` or ``top50_enrichment``."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "store_baselines_betaxanthin.json")))
        sel = d["representations"][rep][model]["selected_on_val"]
        return float(sel["test_pinned_covered" if covered else "test_pinned"][key])

    return run


def ladder_rank(model: str, key: str, rep: str):
    """1.0 when ``rep`` has the highest ``key`` of every representation under ``model``."""

    def run() -> float:
        import json

        d = json.load(open(osp.join(RESULTS, "store_baselines_betaxanthin.json")))
        vals = {
            r: (o[model]["selected_on_val"]["test_pinned"].get(key) or 0.0)
            for r, o in d["representations"].items()
        }
        return 1.0 if max(vals, key=vals.get) == rep else 0.0

    return run


def reg_ratio(stem: str):
    """Mean over seeds 2700 to 2705 of the KL penalty over the data term at the selected
    epoch, from the cells' histories."""

    def run() -> float:
        vals = []
        for seed in range(2700, 2706):
            d = json.load(open(osp.join(RESULTS, f"{stem}_s{seed}.json")))
            h = d["history"][d["select_epoch"]]
            vals.append(h["graph_reg_loss"] / h["train_data_loss"])
        return float(np.mean(vals))

    return run


def uncovered_percentile_range(which: str):
    """Min or max over the pool arms (w206, w231, w239) of the uncovered strains' mean
    percentile among the covered, from the coverage-split runs."""

    def run() -> float:
        vals = [
            coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_uncovered_prediction_percentile")(),
            coverage("w238w239", "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "mean_uncovered_prediction_percentile")(),
            coverage("w238w239", "w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200", "mean_uncovered_prediction_percentile")(),
        ]
        return float(min(vals) if which == "min" else max(vals))

    return run


def cells_stat(stem: str, key: str, what: str, seeds: tuple[int, ...] = tuple(range(2700, 2706))):
    """``min`` or ``max`` over the seeds of one per-cell value: ``select_epoch``,
    ``select_val_pearson`` or the pinned ``spearman``."""

    def run() -> float:
        vals = []
        for seed in seeds:
            d = json.load(open(osp.join(RESULTS, f"{stem}_s{seed}.json")))
            vals.append(
                float(d["test"]["pinned"]["spearman"]) if key == "spearman" else float(d[key])
            )
        return float(min(vals) if what == "min" else max(vals))

    return run


def ladder_file(fname: str, rep: str, model: str, key: str = "mean_spearman"):
    """A pinned read of one representation in a ladder subset file
    (store_baselines_betaxanthin_<suffix>.json, 2026-09-28)."""

    def run() -> float:
        d = json.load(open(osp.join(RESULTS, fname)))
        return float(d["representations"][rep][model]["selected_on_val"]["test_pinned"][key])

    return run


def gcn(arm: str, what: str, budget: str = "e600"):
    """One arm of the message-passing readout: ``mean``, ``sd``, ``n_edges``,
    ``n_nodes``, ``n_parameters``, ``epoch_min`` or ``epoch_max``."""

    def run() -> float:
        fname = (
            "target_gcn_readout_betaxanthin.json"
            if budget == "e200"
            else f"target_gcn_readout_betaxanthin_{budget}.json"
        )
        d = json.load(open(osp.join(RESULTS, fname)))
        a = next(x for x in d["arms"] if x["name"] == arm)
        if what == "epoch_min":
            return float(min(a["select_epoch"]))
        if what == "epoch_max":
            return float(max(a["select_epoch"]))
        return float(a[what])

    return run


def job_hours(job: int):
    """Elapsed hours of a slurm job from sacct, for the round headers."""

    def run() -> float:
        import subprocess

        out = subprocess.run(
            ["sacct", "-j", str(job), "-X", "-n", "-o", "Elapsed"], capture_output=True, text=True
        ).stdout.strip()
        h, m, sec = out.split(":")
        return int(h) + int(m) / 60 + int(sec) / 3600

    return run


def feature_store_columns(store: str):
    """Column count of a store from its build JSON (feature store or concatenation)."""

    def run() -> float:
        import json

        for name in (f"target_graph_feature_store_{store}.json", f"target_store_concat_{store}.json"):
            path = osp.join(RESULTS, name)
            if osp.exists(path):
                d = json.load(open(path))
                return float(d["n_columns"])
        raise FileNotFoundError(store)

    return run


def n_params(stem: str, part: str, seed: int = 2700):
    """A parameter count of one cell's result file (``n_parameters[part]``)."""

    def run() -> float:
        import json

        with open(osp.join(RESULTS, f"{stem}_s{seed}.json")) as fh:
            return float(json.load(fh)["n_parameters"][part])

    return run


def lift_ratio(pair: str, fname: str, rep_hi: str, rep_lo: str):
    """A transfer pair's paired difference over a ladder file's forest lift (rep_hi minus rep_lo)."""

    def run() -> float:
        return transfer(pair, "mean_difference")() / (
            ladder_file(fname, rep_hi, "forest")() - ladder_file(fname, rep_lo, "forest")()
        )

    return run


CHECKS: list[Check] = [
    # sec:standing -- the comparators on the same split and the cross-target read.
    Check(
        claim="ladder: GO ridge",
        stated=0.076,
        recompute=store_baseline("betaxanthin", "go", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO kNN",
        stated=0.179,
        recompute=store_baseline("betaxanthin", "go", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO forest",
        stated=0.187,
        recompute=store_baseline("betaxanthin", "go", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO forest top-50",
        stated=5.39,
        recompute=store_baseline("betaxanthin", "go", "forest", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: GO with HiGHS ridge",
        stated=0.123,
        recompute=store_baseline("betaxanthin", "gopfba", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with HiGHS kNN",
        stated=0.198,
        recompute=store_baseline("betaxanthin", "gopfba", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with HiGHS forest",
        stated=0.204,
        recompute=store_baseline("betaxanthin", "gopfba", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with HiGHS forest top-50",
        stated=4.79,
        recompute=store_baseline("betaxanthin", "gopfba", "forest", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: GO with HiGHS forest under round 151",
        stated=0.008,
        recompute=lambda: (
            transfer(
                "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
                "mean_spearman",
                "arm",
            )()
            - store_baseline("betaxanthin", "gopfba", "forest", "mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: round 151 ensemble minus single",
        stated=0.001,
        recompute=lambda: (
            ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman")()
            - transfer(
                "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
                "mean_spearman",
                "arm",
            )()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: kNN over the HiGHS store",
        stated=0.195,
        recompute=store_baseline("betaxanthin", "pfba_highs", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: best ridge (driving-force cap 100)",
        stated=0.152,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_conc", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: kNN over ProtT5",
        stated=0.145,
        recompute=transfer_comparator("betaxanthin", "cosine kNN"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: FCL random forest",
        stated=0.039,
        recompute=transfer_comparator("betaxanthin", "FCL RandomForest"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: amino-acid pool ridge reads betaxanthin out of fold",
        stated=0.226,
        recompute=cross_target("spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: strains shared with Mulleder",
        stated=4477,
        recompute=cross_target("n_shared_mulleder"),
        tolerance=0.5,
    ),
    Check(
        claim="standing: GO ridge on the Mulleder strains",
        stated=0.165,
        recompute=cross_target("go:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: GO with the pool on the Mulleder strains",
        stated=0.237,
        recompute=cross_target("go+pool:spearman_oof"),
        tolerance=0.0006,
    ),
    Check(
        claim="standing: the pool's addition beyond GO",
        stated=0.072,
        recompute=lambda: (
            cross_target("go+pool:spearman_oof")() - cross_target("go:spearman_oof")()
        ),
        tolerance=0.0006,
    ),
    # sec:physics -- the annotation-only family.
    Check(
        claim="go-models: GO columns with train support",
        stated=2341,
        recompute=go_model("go_mlp", "n_columns_with_train_support"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models: GO perceptron",
        stated=0.200,
        recompute=go_model("go_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: GO perceptron sd",
        stated=0.002,
        recompute=go_model("go_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: GO perceptron top-50",
        stated=4.79,
        recompute=go_model("go_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: GO perceptron earliest selected epoch",
        stated=21,
        recompute=go_model("go_mlp", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models: GO perceptron latest selected epoch",
        stated=22,
        recompute=go_model("go_mlp", "select_epoch_max"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models: store perceptron",
        stated=0.119,
        recompute=go_model("store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: store perceptron top-50",
        stated=2.00,
        recompute=go_model("store_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: ProtT5 perceptron",
        stated=0.103,
        recompute=go_model("prot_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: ProtT5 perceptron top-50",
        stated=4.13,
        recompute=go_model("prot_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: GO with store",
        stated=0.198,
        recompute=go_model("go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: GO with store sd",
        stated=0.041,
        recompute=go_model("go_store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: GO with ProtT5",
        stated=0.173,
        recompute=go_model("go_prot_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: GO with ProtT5 top-50",
        stated=4.99,
        recompute=go_model("go_prot_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: GO with ProtT5 ensemble top-50",
        stated=5.39,
        recompute=go_model("go_prot_mlp", "ensemble_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: all three",
        stated=0.175,
        recompute=go_model("go_prot_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: all three top-50",
        stated=5.03,
        recompute=go_model("go_prot_store_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: DCell hierarchy",
        stated=0.049,
        recompute=go_model("dcell_style_shared", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: DCell hierarchy sd",
        stated=0.043,
        recompute=go_model("dcell_style_shared", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    # sec:physics -- DCell's published input (2026-09-25).
    Check(
        claim="go-models: DCell published input",
        stated=0.028,
        recompute=go_model("dcell_published_input", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: DCell published input sd",
        stated=0.058,
        recompute=go_model("dcell_published_input", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: DCell published input top-50",
        stated=1.26,
        recompute=go_model("dcell_published_input", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models: DCell published input ensemble",
        stated=0.073,
        recompute=go_model("dcell_published_input", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models: DCell published input ensemble top-50",
        stated=1.40,
        recompute=go_model("dcell_published_input", "ensemble_top50"),
        tolerance=0.006,
    ),
    # sec:physics -- both hierarchy arms at width 20 (2026-09-25).
    Check(
        claim="go-models w20: shared hierarchy",
        stated=0.069,
        recompute=go_model_w20("dcell_style_shared", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models w20: shared hierarchy sd",
        stated=0.036,
        recompute=go_model_w20("dcell_style_shared", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models w20: shared hierarchy top-50",
        stated=3.33,
        recompute=go_model_w20("dcell_style_shared", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models w20: shared hierarchy ensemble",
        stated=0.066,
        recompute=go_model_w20("dcell_style_shared", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models w20: shared hierarchy ensemble top-50",
        stated=3.79,
        recompute=go_model_w20("dcell_style_shared", "ensemble_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="go-models w20: shared hierarchy minutes a seed, min",
        stated=13,
        recompute=go_model_w20("dcell_style_shared", "seconds_min"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models w20: shared hierarchy minutes a seed, max",
        stated=24,
        recompute=go_model_w20("dcell_style_shared", "seconds_max"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models w8: shared hierarchy minutes a seed, min",
        stated=7,
        recompute=lambda: (
            min(
                s["seconds"]
                for s in next(
                    a
                    for a in json.load(
                        open(osp.join(RESULTS, "go_models_betaxanthin.json"))
                    )["arms"]
                    if a["arm"] == "dcell_style_shared"
                )["seeds"]
            )
            / 60
        ),
        tolerance=0.5,
    ),
    Check(
        claim="go-models w8: shared hierarchy minutes a seed, max",
        stated=8,
        recompute=lambda: (
            max(
                s["seconds"]
                for s in next(
                    a
                    for a in json.load(
                        open(osp.join(RESULTS, "go_models_betaxanthin.json"))
                    )["arms"]
                    if a["arm"] == "dcell_style_shared"
                )["seeds"]
            )
            / 60
        ),
        tolerance=0.5,
    ),
    Check(
        claim="go-models w20: published input",
        stated=0.041,
        recompute=go_model_w20("dcell_published_input", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models w20: published input sd",
        stated=0.046,
        recompute=go_model_w20("dcell_published_input", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="go-models w20: published input seeds selecting at epoch 200",
        stated=3,
        recompute=go_model_w20("dcell_published_input", "n_select_at_200"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models: DCell earliest selected epoch",
        stated=8,
        recompute=go_model("dcell_style_shared", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="go-models: DCell latest selected epoch",
        stated=13,
        recompute=go_model("dcell_style_shared", "select_epoch_max"),
        tolerance=0.5,
    ),
    # sec:rounds -- round 132, the wide control on the deletion-only graph.
    Check(
        claim="target-transfer: wide deletion-only control",
        stated=0.092,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control sd",
        stated=0.043,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control vs old-graph wide control",
        stated=-0.017,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control vs old-graph wide control sd",
        stated=0.08,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control vs old-graph wide control seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: wide deletion-only control top-50",
        stated=4.49,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: old-graph wide control top-50",
        stated=4.73,
        recompute=transfer(
            "betaxanthin: width 90 on deletion-only graphs against the width-90 control",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: wide store vs wide control on the deletion-only graph",
        stated=0.028,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide store vs wide control on the deletion-only graph sd",
        stated=0.037,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide store vs wide control on the deletion-only graph seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: wide store on the deletion-only graph top-50",
        stated=4.43,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against width 90 on deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control six-seed ensemble",
        stated=0.122,
        recompute=ensemble("w149_bxwidedel_h90l2p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide deletion-only control six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble("w149_bxwidedel_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 134, the solve head on betaxanthin.
    Check(
        claim="target-transfer: solve head arm",
        stated=0.07,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head arm sd",
        stated=0.049,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head vs control",
        stated=0.012,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head vs control sd",
        stated=0.074,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head vs control seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: solve head vs store affine",
        stated=-0.051,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head vs store affine sd",
        stated=0.061,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head vs store affine seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against the store affine there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: solve head top-50",
        stated=3.89,
        recompute=transfer(
            "betaxanthin: solve head (hidden 256) on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: solve head six-seed ensemble",
        stated=0.095,
        recompute=ensemble("w153_bxdelhead256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: solve head six-seed ensemble top-50",
        stated=5.39,
        recompute=ensemble("w153_bxdelhead256_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 135, the linear MOMA store on betaxanthin.
    Check(
        claim="target-transfer: MOMA store arm",
        stated=0.107,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store arm sd",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store vs control",
        stated=0.049,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store vs control sd",
        stated=0.064,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: MOMA store vs parsimonious store",
        stated=-0.014,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store vs parsimonious store sd",
        stated=0.029,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store vs parsimonious store seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against the parsimonious store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: MOMA store top-50",
        stated=4.99,
        recompute=transfer(
            "betaxanthin: linear MOMA store on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: MOMA store six-seed ensemble",
        stated=0.115,
        recompute=ensemble("w154_bxdelmoma_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA store six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w154_bxdelmoma_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 133, the rule counts on betaxanthin.
    Check(
        claim="target-transfer: rule-count store arm",
        stated=0.114,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store arm sd",
        stated=0.019,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store vs control",
        stated=0.056,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store vs control sd",
        stated=0.056,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rule-count store vs plain store",
        stated=-0.007,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against the store affine there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store vs plain store sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against the store affine there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store vs plain store seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against the store affine there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rule-count store top-50",
        stated=3.96,
        recompute=transfer(
            "betaxanthin: vector with rule counts on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rule-count store six-seed ensemble",
        stated=0.137,
        recompute=ensemble("w152_bxdelgpr_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: rule-count store six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w152_bxdelgpr_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 139, the HiGHS parsimonious store on betaxanthin.
    Check(
        claim="target-transfer: HiGHS store arm",
        stated=0.13,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store arm sd",
        stated=0.034,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store vs control",
        stated=0.072,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store vs control sd",
        stated=0.067,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: HiGHS store vs GLPK store",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store vs GLPK store sd",
        stated=0.041,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store vs GLPK store seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against the GLPK store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: HiGHS store top-50",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS parsimonious store on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS store six-seed ensemble",
        stated=0.148,
        recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 140, the driving-force concentration store on betaxanthin.
    Check(
        claim="target-transfer: driving-force store arm",
        stated=0.118,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store arm sd",
        stated=0.055,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store vs control",
        stated=0.059,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store vs control sd",
        stated=0.055,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: driving-force store vs GLPK store",
        stated=-0.003,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store vs GLPK store sd",
        stated=0.037,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store vs GLPK store seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against the GLPK store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: driving-force store top-50",
        stated=4.06,
        recompute=transfer(
            "betaxanthin: driving-force concentration store on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: driving-force store six-seed ensemble",
        stated=0.133,
        recompute=ensemble("w157_bxdelmdf_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: driving-force store six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w157_bxdelmdf_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 142, the sampled cone on betaxanthin.
    Check(
        claim="target-transfer: sampled cone arm",
        stated=0.088,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone arm sd",
        stated=0.022,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone vs control",
        stated=0.03,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone vs control sd",
        stated=0.07,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone vs control seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: sampled cone vs GLPK store",
        stated=-0.033,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone vs GLPK store sd",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone vs GLPK store seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against the GLPK store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: sampled cone top-50",
        stated=2.9,
        recompute=transfer(
            "betaxanthin: sampled-cone mean store on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: sampled cone six-seed ensemble",
        stated=0.083,
        recompute=ensemble("w159_bxdelsampled_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: sampled cone six-seed ensemble top-50",
        stated=3.00,
        recompute=ensemble("w159_bxdelsampled_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 144, the store arm at width 6.
    Check(
        claim="target-transfer: width-6 store arm",
        stated=0.113,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: width-6 store arm sd",
        stated=0.038,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: width-6 store vs width-12 store",
        stated=-0.008,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: width-6 store vs width-12 store sd",
        stated=0.055,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: width-6 store vs width-12 store seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: width-6 store top-50",
        stated=4.19,
        recompute=transfer(
            "betaxanthin: deletion-only store at width 6 against width 12",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: width-6 store six-seed ensemble",
        stated=0.129,
        recompute=ensemble("w161_bxdelstoreh6_h6l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: width-6 store six-seed ensemble top-50",
        stated=5.39,
        recompute=ensemble("w161_bxdelstoreh6_h6l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 145, the store arm at dropout 0.3.
    Check(
        claim="target-transfer: dropout-0.3 store arm",
        stated=0.145,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store arm sd",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store vs dropout-0.1 store",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store vs dropout-0.1 store sd",
        stated=0.04,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store vs dropout-0.1 store seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store top-50",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.3 against 0.1",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store six-seed ensemble",
        stated=0.149,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout-0.3 store six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 147, the polytope mean on betaxanthin.
    Check(
        claim="target-transfer: polytope mean arm",
        stated=0.125,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean arm sd",
        stated=0.041,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs control",
        stated=0.067,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs control sd",
        stated=0.047,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: polytope mean vs GLPK store",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs GLPK store sd",
        stated=0.047,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs GLPK store seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the GLPK store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: polytope mean vs driving-force point",
        stated=0.008,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs driving-force point sd",
        stated=0.054,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean vs driving-force point seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against the driving-force point",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: polytope mean top-50",
        stated=3.69,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope mean six-seed ensemble",
        stated=0.151,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean six-seed ensemble top-50",
        stated=5.59,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 149, the release rule on the MOMA store.
    Check(
        claim="target-transfer: MOMA release arm",
        stated=0.106,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release arm sd",
        stated=0.034,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release vs MOMA store",
        stated=-0.001,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release vs MOMA store sd",
        stated=0.038,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release vs MOMA store seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: MOMA release vs GLPK release",
        stated=-0.004,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release vs GLPK release sd",
        stated=0.063,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release vs GLPK release seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: MOMA release top-50",
        stated=4.53,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the MOMA store",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK release level",
        stated=0.11,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GLPK release top-50",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: MOMA store with the affine released on escape against the GLPK store under the rule",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: MOMA release six-seed ensemble",
        stated=0.105,
        recompute=ensemble("w166_bxdelmomarel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: MOMA release six-seed ensemble top-50",
        stated=5.59,
        recompute=ensemble("w166_bxdelmomarel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 150, GO in the affine slot on betaxanthin.
    Check(
        claim="target-transfer: GO affine arm",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine arm sd",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine vs control",
        stated=0.144,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine vs control sd",
        stated=0.075,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO affine vs GLPK store",
        stated=0.082,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine vs GLPK store sd",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine vs GLPK store seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against the GLPK store there",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO affine top-50",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot on deletion-only graphs against deletion-only graphs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO affine six-seed ensemble",
        stated=0.202,
        recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 151, the HiGHS store with GO appended.
    Check(
        claim="target-transfer: GO with HiGHS arm",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS arm sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs HiGHS alone",
        stated=0.081,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs HiGHS alone sd",
        stated=0.032,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs HiGHS alone seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs GO alone",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs GO alone sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS vs GO alone seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO with HiGHS top-50",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS alone top-50 (round 151 pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against the HiGHS store alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS six-seed ensemble",
        stated=0.212,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 152, the release rule on the HiGHS store.
    Check(
        claim="target-transfer: HiGHS release arm",
        stated=0.108,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release arm sd",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs HiGHS store",
        stated=-0.022,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs HiGHS store sd",
        stated=0.042,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs HiGHS store seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: HiGHS store level (round 152 pair)",
        stated=0.13,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs GLPK release",
        stated=-0.002,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs GLPK release sd",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release vs GLPK release seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: HiGHS release top-50",
        stated=3.99,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS store top-50 (round 152 pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the HiGHS store",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK release top-50 (round 152 pair)",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: HiGHS store with the affine released on escape against the GLPK store under the rule",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS release six-seed ensemble",
        stated=0.139,
        recompute=ensemble("w169_bxdelpfbarel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS release six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w169_bxdelpfbarel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS store six-seed ensemble (round 152 pair)",
        stated=0.148,
        recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS store six-seed ensemble top-50 (round 152 pair)",
        stated=5.79,
        recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK release six-seed ensemble (round 152 pair)",
        stated=0.12,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GLPK release six-seed ensemble top-50 (round 152 pair)",
        stated=6.19,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 154, the three levers on one cell.
    Check(
        claim="target-transfer: three levers arm",
        stated=0.125,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers arm sd",
        stated=0.036,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs no dropout",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs no dropout sd",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs no dropout seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: HiGHS release level (round 154 pair)",
        stated=0.108,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs GLPK dropout",
        stated=-0.020,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs GLPK dropout sd",
        stated=0.052,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers vs GLPK dropout seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GLPK dropout level (round 154 pair)",
        stated=0.145,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers top-50",
        stated=4.36,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: HiGHS release top-50 (round 154 pair)",
        stated=3.99,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the same without the dropout",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK dropout top-50 (round 154 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: HiGHS store, release rule and dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: three levers six-seed ensemble",
        stated=0.148,
        recompute=ensemble("w171_bxdelpfbareldrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: three levers six-seed ensemble top-50",
        stated=4.39,
        recompute=ensemble("w171_bxdelpfbareldrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK dropout six-seed ensemble (round 154 pair)",
        stated=0.149,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GLPK dropout six-seed ensemble top-50 (round 154 pair)",
        stated=5.79,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 155, dropout 0.5 on the store arm.
    Check(
        claim="target-transfer: dropout 0.5 arm",
        stated=0.125,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 arm sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.3",
        stated=-0.019,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.3 sd",
        stated=0.019,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.3 seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: dropout 0.3 level (round 155 pair)",
        stated=0.145,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.1",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.1 sd",
        stated=0.039,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 vs 0.1 seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.1", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: dropout 0.1 level (round 155 pair)",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 top-50",
        stated=4.16,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout 0.3 top-50 (round 155 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.3",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout 0.1 top-50 (round 155 pair)",
        stated=4.93,
        recompute=transfer(
            "betaxanthin: deletion-only store at dropout 0.5 against 0.1",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 six-seed ensemble",
        stated=0.127,
        recompute=ensemble("w172_bxdelstoredrop5_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.5 six-seed ensemble top-50",
        stated=5.59,
        recompute=ensemble("w172_bxdelstoredrop5_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout 0.1 six-seed ensemble (round 155 pair)",
        stated=0.131,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: dropout 0.1 six-seed ensemble top-50 (round 155 pair)",
        stated=5.59,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 156, the polytope spread beside its mean.
    Check(
        claim="target-transfer: polytope spread arm",
        stated=0.116,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread arm sd",
        stated=0.041,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs mean",
        stated=-0.010,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs mean sd",
        stated=0.062,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs mean seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: polytope mean level (round 156 pair)",
        stated=0.125,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs GLPK",
        stated=-0.005,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs GLPK sd",
        stated=0.048,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread vs GLPK seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GLPK level (round 156 pair)",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread top-50",
        stated=3.73,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope mean top-50 (round 156 pair)",
        stated=3.69,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the mean alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK top-50 (round 156 pair)",
        stated=4.93,
        recompute=transfer(
            "betaxanthin: driving-force polytope mean with the spread against the GLPK store",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope spread six-seed ensemble",
        stated=0.155,
        recompute=ensemble("w174_bxdelmdfpolyspread_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope spread six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble("w174_bxdelmdfpolyspread_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope mean six-seed ensemble (round 156 pair)",
        stated=0.151,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope mean six-seed ensemble top-50 (round 156 pair)",
        stated=5.59,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: dropout 0.3 six-seed ensemble (round 156 cited)",
        stated=0.149,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: HiGHS six-seed ensemble (round 156 cited)",
        stated=0.148,
        recompute=ensemble("w156_bxdelpfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: polytope spread kNN (round 156 cited)",
        stated=0.138,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_spread", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: polytope mean kNN (round 156 cited)",
        stated=0.127,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_mean", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 158, the GO affine under the release rule.
    Check(
        claim="target-transfer: GO release arm",
        stated=0.199,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release arm sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GO affine",
        stated=-0.003,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GO affine sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GO affine seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO affine level (round 158 pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine sd (round 158 pair)",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GLPK release",
        stated=0.089,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GLPK store under the rule",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GLPK release sd",
        stated=0.047,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GLPK store under the rule",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release vs GLPK release seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GLPK store under the rule",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GLPK release level (round 158 pair)",
        stated=0.110,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GLPK store under the rule",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release top-50",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO affine top-50 (round 158 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GO affine",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK release top-50 (round 158 pair)",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: GO affine released on escape against the GLPK store under the rule",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO release six-seed ensemble",
        stated=0.206,
        recompute=ensemble("w175_bxdelgorel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO release six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble("w175_bxdelgorel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO affine six-seed ensemble (round 158 pair)",
        stated=0.202,
        recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO affine six-seed ensemble top-50 (round 158 pair)",
        stated=4.99,
        recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK release six-seed ensemble (round 158 pair)",
        stated=0.120,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GLPK release six-seed ensemble top-50 (round 158 pair)",
        stated=6.19,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO release ensemble gain over GO affine ensemble",
        stated=0.003,
        recompute=lambda: (
            ensemble("w175_bxdelgorel_h12l1p1_e200", "spearman")()
            - ensemble("w167_bxdelgo_h12l1p1_e200", "spearman")()
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 159, the GO affine at dropout 0.3.
    Check(
        claim="target-transfer: GO dropout arm",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout arm sd",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GO affine",
        stated=-0.001,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GO affine sd",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GO affine seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO affine level (round 159 pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GLPK dropout",
        stated=0.057,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GLPK dropout sd",
        stated=0.019,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout vs GLPK dropout seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GLPK dropout level (round 159 pair)",
        stated=0.145,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout top-50",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO affine top-50 (round 159 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GO affine",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK dropout top-50 (round 159 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: GO affine at dropout 0.3 against the GLPK store at dropout 0.3",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO dropout six-seed ensemble",
        stated=0.202,
        recompute=ensemble("w176_bxdelgodrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w176_bxdelgodrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GLPK dropout six-seed ensemble (round 159 pair)",
        stated=0.149,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GLPK dropout six-seed ensemble top-50 (round 159 pair)",
        stated=5.79,
        recompute=ensemble("w162_bxdelstoredrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 161, GO with the HiGHS vector at dropout 0.3.
    Check(
        claim="target-transfer: GO with HiGHS dropout arm",
        stated=0.212,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout arm sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs 0.1",
        stated=0.000,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs 0.1 sd",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs 0.1 seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO with HiGHS 0.1 level (round 161 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs GO dropout",
        stated=0.010,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs GO dropout sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout vs GO dropout seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO dropout level (round 161 pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout top-50",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS 0.1 top-50 (round 161 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against 0.1",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO dropout top-50 (round 161 pair)",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended at dropout 0.3 against GO alone at 0.3",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout six-seed ensemble",
        stated=0.213,
        recompute=ensemble("w178_bxdelgopfbadrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS dropout six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w178_bxdelgopfbadrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS six-seed ensemble (round 161 pair)",
        stated=0.212,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS six-seed ensemble top-50 (round 161 pair)",
        stated=4.99,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO dropout six-seed ensemble (round 161 pair)",
        stated=0.202,
        recompute=ensemble("w176_bxdelgodrop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO dropout six-seed ensemble top-50 (round 161 pair)",
        stated=4.79,
        recompute=ensemble("w176_bxdelgodrop_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 163, GO, the HiGHS vector and the polytope mean in one affine.
    Check(
        claim="target-transfer: GO HiGHS polytope arm",
        stated=0.198,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope arm sd",
        stated=0.011,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs GO HiGHS",
        stated=-0.013,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs GO HiGHS sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs GO HiGHS seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS level (round 163 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs polytope",
        stated=0.073,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs polytope sd",
        stated=0.045,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope vs polytope seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: polytope level (round 163 pair)",
        stated=0.125,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope top-50",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS top-50 (round 163 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against HiGHS store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope top-50 (round 163 pair)",
        stated=3.69,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean against the polytope mean alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope six-seed ensemble",
        stated=0.201,
        recompute=ensemble("w180_bxdelgopfbapoly_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble("w180_bxdelgopfbapoly_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: polytope six-seed ensemble (round 163 pair)",
        stated=0.151,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: polytope six-seed ensemble top-50 (round 163 pair)",
        stated=5.59,
        recompute=ensemble("w164_bxdelmdfpoly_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 164, twelve seeds of the best arm.
    Check(
        claim="round 164: second six level",
        stated=0.218,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: second six sd",
        stated=0.009,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "last6_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: first six level",
        stated=0.211,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve-seed level",
        stated=0.214,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve-seed sd",
        stated=0.008,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve-seed standard error",
        stated=0.002,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "sem"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: lowest seed",
        stated=0.202,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: highest seed",
        stated=0.232,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve-seed ensemble",
        stated=0.216,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble(
            "w168_bxdelgopfba_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 164: six-seed ensemble",
        stated=0.212,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: twelve single seeds' mean call",
        stated=4.98,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="round 164: level above the forest",
        stated=0.011,
        recompute=lambda: (
            seed_level("w168_bxdelgopfba_h12l1p1_e200", "all12")()
            - store_baseline("betaxanthin", "gopfba", "forest", "mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: level above the HiGHS kNN",
        stated=0.019,
        recompute=lambda: (
            seed_level("w168_bxdelgopfba_h12l1p1_e200", "all12")()
            - store_baseline("betaxanthin", "pfba_highs", "knn", "mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 164: ensemble gain from the second six",
        stated=0.004,
        recompute=lambda: (
            ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman", "all_seeds")()
            - ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman")()
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 166, the stack with the region's width.
    Check(
        claim="target-transfer: GO HiGHS polytope spread arm",
        stated=0.193,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread arm sd",
        stated=0.015,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs GO HiGHS",
        stated=-0.018,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs GO HiGHS sd",
        stated=0.016,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs GO HiGHS seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS level (round 166 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs mean alone",
        stated=-0.005,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs mean alone sd",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread vs mean alone seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope mean level (round 166 pair)",
        stated=0.198,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread top-50",
        stated=4.23,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS top-50 (round 166 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against HiGHS store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope mean top-50 (round 166 pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO and the polytope mean and spread against the same without the spread",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread six-seed ensemble",
        stated=0.201,
        recompute=ensemble("w183_bxdelgopfbapolyspread_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS polytope spread six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble(
            "w183_bxdelgopfbapolyspread_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 166: the stack's columns",
        stated=12205,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gopfbapolyspread.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 166: spread columns beyond the mean",
        stated=2831,
        recompute=lambda: (
            float(
                json.load(
                    open(
                        osp.join(
                            os.environ["EXPERIMENT_ROOT"],
                            "031-betaxanthin-module",
                            "results",
                            "target_store_concat_YPD_bx_gopfbapolyspread.json",
                        )
                    )
                )["n_columns"]
            )
            - float(
                json.load(
                    open(
                        osp.join(
                            os.environ["EXPERIMENT_ROOT"],
                            "031-betaxanthin-module",
                            "results",
                            "target_store_concat_YPD_bx_gopfbapoly.json",
                        )
                    )
                )["n_columns"]
            )
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 167, the adjustment store beside the annotation.
    Check(
        claim="target-transfer: GO with MOMA arm",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA arm sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO alone",
        stated=0.006,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO alone sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone", "sd_difference"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO alone seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO alone level (round 167 pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO with HiGHS",
        stated=-0.004,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO with HiGHS sd",
        stated=0.006,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA vs GO with HiGHS seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO with HiGHS level (round 167 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA top-50",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO alone top-50 (round 167 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against GO alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with HiGHS top-50 (round 167 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: MOMA store with GO appended against HiGHS store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO with MOMA six-seed ensemble",
        stated=0.209,
        recompute=ensemble("w184_bxdelgomoma_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO with MOMA six-seed ensemble top-50",
        stated=4.39,
        recompute=ensemble("w184_bxdelgomoma_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 167: the store's columns",
        stated=6539,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gomoma.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 169, the affine's weight decay at 0.1.
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 arm",
        stated=0.210,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 arm sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 vs no decay",
        stated=-0.002,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 vs no decay sd",
        stated=0.008,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 vs no decay seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS level (round 169 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 top-50",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS top-50 (round 169 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 0.1 against no decay",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 six-seed ensemble",
        stated=0.211,
        recompute=ensemble("w188_bxdelgopfbawd_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w188_bxdelgopfbawd_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS six-seed ensemble (round 169 pair)",
        stated=0.212,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS six-seed ensemble top-50 (round 169 pair)",
        stated=4.99,
        recompute=ensemble("w168_bxdelgopfba_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 170, the affine's weight decay at 1.0.
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 arm",
        stated=0.204,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 arm sd",
        stated=0.012,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs no decay",
        stated=-0.007,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs no decay sd",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs no decay seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs decay 0.1",
        stated=-0.005,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against decay 0.1",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs decay 0.1 sd",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against decay 0.1",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 vs decay 0.1 seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against decay 0.1",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 0.1 level (round 170 pair)",
        stated=0.210,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against decay 0.1",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 top-50",
        stated=4.76,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO at affine decay 1.0 against no decay",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 six-seed ensemble",
        stated=0.208,
        recompute=ensemble("w189_bxdelgopfbawd1_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS affine decay 1.0 six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w189_bxdelgopfbawd1_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 171, both flux programs beside GO.
    Check(
        claim="target-transfer: GO HiGHS MOMA arm",
        stated=0.216,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA arm sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO HiGHS",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO HiGHS sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO HiGHS seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS level (round 171 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO MOMA",
        stated=0.008,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO MOMA sd",
        stated=0.006,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA vs GO MOMA seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO MOMA level (round 171 pair)",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA top-50",
        stated=4.73,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS top-50 (round 171 pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO MOMA top-50 (round 171 pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against MOMA store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA six-seed ensemble",
        stated=0.217,
        recompute=ensemble("w190_bxdelgopfbamoma_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: GO HiGHS MOMA six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble("w190_bxdelgopfbamoma_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO MOMA six-seed ensemble (round 171 pair)",
        stated=0.209,
        recompute=ensemble("w184_bxdelgomoma_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 171: the store's columns",
        stated=10716,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gopfbamoma.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 172, twelve seeds of the annotation-only arm.
    Check(
        claim="round 172: GO affine second six",
        stated=0.204,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: GO affine first six",
        stated=0.202,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: GO affine twelve seeds",
        stated=0.203,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: GO affine twelve-seed sd",
        stated=0.002,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: GO affine twelve-seed sem",
        stated=0.0005,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "sem"),
        tolerance=0.00006,
    ),
    Check(
        claim="round 172: GO affine twelve-seed ensemble",
        stated=0.203,
        recompute=ensemble("w167_bxdelgo_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: GO affine twelve-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble(
            "w167_bxdelgo_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 172: flux gain beside GO on twelve seeds",
        stated=0.012,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: flux gain beside GO on twelve seeds sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 172: flux gain beside GO seeds above",
        stated=11,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 172: GO HiGHS twelve-seed level (pair)",
        stated=0.214,
        recompute=transfer(
            "betaxanthin: HiGHS store with GO appended against GO alone, twelve seeds",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 173, the control on twelve seeds.
    Check(
        claim="round 173: control second six",
        stated=0.094,
        recompute=seed_level("w137_bxdel_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: control first six",
        stated=0.058,
        recompute=seed_level("w137_bxdel_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: control twelve seeds",
        stated=0.076,
        recompute=seed_level("w137_bxdel_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: control twelve-seed sd",
        stated=0.055,
        recompute=seed_level("w137_bxdel_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: GO affine vs control, twelve seeds",
        stated=0.127,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot against the control, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: GO affine vs control, twelve seeds sd",
        stated=0.054,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot against the control, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: GO affine vs control, seeds above",
        stated=12,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot against the control, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 173: control top-50 (twelve seeds)",
        stated=4.06,
        recompute=transfer(
            "betaxanthin: GO multi-hot in the affine slot against the control, twelve seeds",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 173: control twelve-seed ensemble",
        stated=0.102,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: control twelve-seed ensemble top-50",
        stated=5.59,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "top50_enrichment", "all_seeds"),
        tolerance=0.006,
    ),
    Check(
        claim="round 173: control six-seed ensemble",
        stated=0.082,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 173: control six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- rounds 175 and 176, the measured pool in the affine.
    Check(
        claim="target-transfer: pool with GO arm",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO arm sd",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO vs GO alone",
        stated=0.006,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO vs GO alone sd",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO vs GO alone seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO alone level (round 175 pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO top-50",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: GO alone top-50 (round 175 pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO against GO alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: pool with GO six-seed ensemble",
        stated=0.209,
        recompute=ensemble("w194_bxdelgoaapool_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w194_bxdelgoaapool_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS arm",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS arm sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS vs GO HiGHS",
        stated=-0.003,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS vs GO HiGHS sd",
        stated=0.014,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS vs GO HiGHS seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: GO HiGHS level (round 176 pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against GO with the HiGHS vector",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS vs pool with GO",
        stated=0.000,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against the pool with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS vs pool with GO seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool with GO and the HiGHS vector against the pool with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS six-seed ensemble",
        stated=0.209,
        recompute=ensemble("w195_bxdelgopfbaaapool_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: pool with GO HiGHS six-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble("w195_bxdelgopfbaaapool_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:rounds -- round 177, twelve seeds of both flux programs beside GO.
    Check(
        claim="round 177: second six",
        stated=0.219,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: first six",
        stated=0.216,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: twelve seeds",
        stated=0.217,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: twelve-seed sd",
        stated=0.005,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: twelve-seed sem",
        stated=0.0015,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "sem"),
        tolerance=0.00006,
    ),
    Check(
        claim="round 177: twelve-seed ensemble",
        stated=0.218,
        recompute=ensemble(
            "w190_bxdelgopfbamoma_h12l1p1_e200", "spearman", "all_seeds"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: twelve-seed ensemble top-50",
        stated=4.79,
        recompute=ensemble(
            "w190_bxdelgopfbamoma_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 177: gain vs GO HiGHS on twelve seeds",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: gain vs GO HiGHS on twelve seeds sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO, twelve seeds",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: gain vs GO HiGHS seeds above",
        stated=8,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 177: GO HiGHS twelve-seed level",
        stated=0.214,
        recompute=transfer(
            "betaxanthin: HiGHS and MOMA stores with GO against HiGHS store with GO, twelve seeds",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 177: GO HiGHS second six",
        stated=0.218,
        recompute=seed_level("w168_bxdelgopfba_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 178, the three families in one affine.
    Check(
        claim="round 178: three families arm",
        stated=0.218,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: three families arm sd",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs both flux programs with GO",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs both flux programs with GO sd",
        stated=0.008,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs both flux programs with GO seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 178: both flux programs level (pair)",
        stated=0.216,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs pool with GO HiGHS",
        stated=0.010,
        recompute=transfer(
            "betaxanthin: three families against the pool with GO and the HiGHS vector",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs pool with GO HiGHS sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: three families against the pool with GO and the HiGHS vector",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: vs pool with GO HiGHS seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: three families against the pool with GO and the HiGHS vector",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 178: pool with GO HiGHS level (pair)",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: three families against the pool with GO and the HiGHS vector",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: three families top-50",
        stated=4.63,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 178: both flux programs top-50 (pair)",
        stated=4.73,
        recompute=transfer(
            "betaxanthin: three families, GO with both flux programs and the measured pool, against GO with both flux programs",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 178: pool with GO HiGHS top-50 (pair)",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: three families against the pool with GO and the HiGHS vector",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 178: three families six-seed ensemble",
        stated=0.218,
        recompute=ensemble("w197_bxdelgopfbamomaaapool_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 178: three families six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble(
            "w197_bxdelgopfbamomaaapool_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 178: the store's columns",
        stated=10735,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gopfbamomaaapool.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 179, the pool alone in the affine.
    Check(
        claim="round 179: pool alone arm",
        stated=0.195,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: pool alone arm sd",
        stated=0.029,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: control level (pair)",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs control",
        stated=0.137,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs control sd",
        stated=0.088,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 179: GO alone level (pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs GO alone",
        stated=-0.008,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs GO alone sd",
        stated=0.030,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: vs GO alone seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 179: pool alone top-50",
        stated=2.53,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: control top-50 (pair)",
        stated=3.39,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: GO alone top-50 (pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: pool alone six-seed ensemble",
        stated=0.211,
        recompute=ensemble("w198_bxdelaapool_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: pool alone six-seed ensemble top-50",
        stated=3.00,
        recompute=ensemble("w198_bxdelaapool_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: select epoch min",
        stated=13,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 179: select epoch max",
        stated=32,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 179: validation Pearson min",
        stated=0.33,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: validation Pearson max",
        stated=0.41,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool alone in the affine against the control",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: GO alone validation Pearson max",
        stated=0.26,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
            "control",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 179: seed min",
        stated=0.150,
        recompute=seed_level("w198_bxdelaapool_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: seed max",
        stated=0.220,
        recompute=seed_level("w198_bxdelaapool_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: seed spread ratio to GO alone (twenty times)",
        stated=20,
        recompute=lambda: (
            transfer(
                "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
                "sd_spearman",
                "arm",
            )()
            / transfer(
                "betaxanthin: measured amino-acid pool alone in the affine against GO alone",
                "sd_spearman",
                "control",
            )()
        ),
        tolerance=1.5,
    ),
    Check(
        claim="round 179: pool ridge on the ladder",
        stated=0.149,
        recompute=store_baseline("betaxanthin", "aa_pool", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: pool kNN on the ladder",
        stated=0.285,
        recompute=store_baseline("betaxanthin", "aa_pool", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: pool forest on the ladder",
        stated=0.280,
        recompute=store_baseline("betaxanthin", "aa_pool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 179: gap under the pool kNN",
        stated=0.09,
        recompute=lambda: (
            store_baseline("betaxanthin", "aa_pool", "knn", "mean_spearman")()
            - transfer(
                "betaxanthin: measured amino-acid pool alone in the affine against the control",
                "mean_spearman",
                "arm",
            )()
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 180, the pool through a hidden layer of 64.
    Check(
        claim="round 180: pool mlp arm",
        stated=0.240,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: pool mlp arm sd",
        stated=0.048,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs linear pool",
        stated=0.045,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs linear pool sd",
        stated=0.047,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs linear pool seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 180: linear pool level (pair)",
        stated=0.195,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the pool in the linear affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs control",
        stated=0.182,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs control sd",
        stated=0.099,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 180: pool mlp top-50",
        stated=4.03,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 180: pool mlp six-seed ensemble",
        stated=0.292,
        recompute=ensemble("w199_bxdelaapoolmlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: pool mlp six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w199_bxdelaapoolmlp_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 180: ensemble over the best linear ensemble",
        stated=0.074,
        recompute=lambda: (
            ensemble("w199_bxdelaapoolmlp_h12l1p1_e200", "spearman")()
            - ensemble("w197_bxdelgopfbamomaaapool_h12l1p1_e200", "spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: seed min",
        stated=0.159,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: seed max",
        stated=0.300,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 180: select epoch min",
        stated=13,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 180: select epoch max",
        stated=197,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 180: validation Pearson min",
        stated=0.35,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 180: validation Pearson max",
        stated=0.38,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 181, the minimum-norm store alone.
    Check(
        claim="round 181: l2 store arm",
        stated=0.093,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: l2 store arm sd",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: HiGHS store level (pair)",
        stated=0.130,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: vs HiGHS store",
        stated=-0.037,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: vs HiGHS store sd",
        stated=0.043,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: vs HiGHS store seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 181: vs control",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: vs control sd",
        stated=0.076,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the control", "sd_difference"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: vs control seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 181: l2 store top-50",
        stated=4.59,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 181: HiGHS store top-50 (pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store against the HiGHS parsimonious store",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 181: l2 store six-seed ensemble",
        stated=0.094,
        recompute=ensemble("w200_bxdell2_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: l2 store six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble("w200_bxdell2_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 181: seed min",
        stated=0.070,
        recompute=seed_level("w200_bxdell2_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: seed max",
        stated=0.107,
        recompute=seed_level("w200_bxdell2_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: select epoch min",
        stated=3,
        recompute=pair_cells(
            "betaxanthin: minimum-norm flux store against the control",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 181: select epoch max",
        stated=64,
        recompute=pair_cells(
            "betaxanthin: minimum-norm flux store against the control",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 181: l2 ridge on the ladder",
        stated=0.068,
        recompute=store_baseline("betaxanthin", "pfba_l2", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: l2 kNN on the ladder",
        stated=0.126,
        recompute=store_baseline("betaxanthin", "pfba_l2", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: l2 forest on the ladder",
        stated=0.133,
        recompute=store_baseline("betaxanthin", "pfba_l2", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: HiGHS ridge on the ladder",
        stated=0.098,
        recompute=store_baseline("betaxanthin", "pfba_highs", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: HiGHS kNN on the ladder",
        stated=0.195,
        recompute=store_baseline("betaxanthin", "pfba_highs", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 181: HiGHS forest on the ladder",
        stated=0.117,
        recompute=store_baseline(
            "betaxanthin", "pfba_highs", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 182, the minimum-norm store with GO.
    Check(
        claim="round 182: GO l2 arm",
        stated=0.204,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: GO l2 arm sd",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: GO HiGHS level (pair)",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO HiGHS",
        stated=-0.007,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO HiGHS sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO HiGHS seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 182: GO alone level (pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO alone",
        stated=0.002,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO alone sd",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: vs GO alone seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 182: GO l2 top-50",
        stated=4.36,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 182: GO HiGHS top-50 (pair)",
        stated=4.83,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against HiGHS store with GO",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 182: GO alone top-50 (pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 182: GO l2 six-seed ensemble",
        stated=0.206,
        recompute=ensemble("w201_bxdelgol2_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: GO l2 six-seed ensemble top-50",
        stated=4.39,
        recompute=ensemble("w201_bxdelgol2_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 182: seed min",
        stated=0.198,
        recompute=seed_level("w201_bxdelgol2_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: seed max",
        stated=0.212,
        recompute=seed_level("w201_bxdelgol2_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: select epoch min",
        stated=8,
        recompute=pair_cells(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 182: select epoch max",
        stated=16,
        recompute=pair_cells(
            "betaxanthin: minimum-norm flux store with GO against GO alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 182: GO l2 kNN on the ladder",
        stated=0.208,
        recompute=store_baseline("betaxanthin", "gol2", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 182: the store's columns",
        stated=6539,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gol2.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 185, GO with the pool through the hidden layer.
    Check(
        claim="round 185: GO pool mlp arm",
        stated=0.206,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: GO pool mlp arm sd",
        stated=0.006,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: pool mlp level (pair)",
        stated=0.240,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs pool mlp",
        stated=-0.034,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs pool mlp sd",
        stated=0.052,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs pool mlp seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 185: linear GO pool level (pair)",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs linear GO pool",
        stated=-0.002,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs linear GO pool sd",
        stated=0.008,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs linear GO pool seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 185: GO alone level (pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs GO alone",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs GO alone sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: vs GO alone seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 185: GO pool mlp top-50",
        stated=4.59,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: pool mlp top-50 (pair)",
        stated=4.03,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the pool alone through the same layer",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: linear GO pool top-50 (pair)",
        stated=4.89,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against the same columns in the linear affine",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: GO alone top-50 (pair)",
        stated=5.03,
        recompute=transfer(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: six-seed ensemble",
        stated=0.208,
        recompute=ensemble("w204_bxdelgoaapoolmlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 185: six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w204_bxdelgoaapoolmlp_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: select epoch min",
        stated=2,
        recompute=pair_cells(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 185: select epoch max",
        stated=4,
        recompute=pair_cells(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 185: validation Pearson min",
        stated=0.22,
        recompute=pair_cells(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: validation Pearson max",
        stated=0.29,
        recompute=pair_cells(
            "betaxanthin: GO with the measured pool through a hidden layer of 64 against GO alone",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 185: the store's columns",
        stated=2381,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_goaapool.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 186, twelve seeds of the pool through the hidden layer.
    Check(
        claim="round 186: pool mlp second six",
        stated=0.268,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: pool mlp second six sd",
        stated=0.032,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "last6_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: pool mlp first six",
        stated=0.240,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: pool mlp twelve seeds",
        stated=0.254,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: pool mlp twelve-seed sd",
        stated=0.041,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: pool mlp twelve-seed sem",
        stated=0.012,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "sem"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: twelve-seed min",
        stated=0.159,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: twelve-seed max",
        stated=0.300,
        recompute=seed_level("w199_bxdelaapoolmlp_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: twelve-seed ensemble",
        stated=0.297,
        recompute=ensemble("w199_bxdelaapoolmlp_h12l1p1_e200", "spearman", "all_seeds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: twelve-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble(
            "w199_bxdelaapoolmlp_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 186: vs control twelve seeds",
        stated=0.178,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: vs control twelve seeds above",
        stated=12,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against the control, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 186: vs GO alone twelve seeds",
        stated=0.051,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 186: vs GO alone twelve seeds above",
        stated=11,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 186: twelve-seed top-50",
        stated=4.24,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 186: GO alone twelve-seed top-50 (pair)",
        stated=4.99,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 186: select epoch min",
        stated=4,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 186: select epoch max",
        stated=197,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 186: spread ratio to GO alone (twenty times)",
        stated=20,
        recompute=lambda: (
            transfer(
                "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
                "sd_spearman",
                "arm",
            )()
            / transfer(
                "betaxanthin: measured amino-acid pool through a hidden layer of 64 against GO alone, twelve seeds",
                "sd_spearman",
                "control",
            )()
        ),
        tolerance=2.5,
    ),
    # sec:rounds -- round 187, the pool through a hidden layer of 256.
    Check(
        claim="round 187: pool mlp256 arm",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: pool mlp256 arm sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: vs layer of 64",
        stated=0.039,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: vs layer of 64 sd",
        stated=0.051,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: vs layer of 64 seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 187: vs linear slot",
        stated=0.084,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the pool in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: vs linear slot sd",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the pool in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: vs linear slot seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the pool in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 187: pool mlp256 top-50",
        stated=4.69,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 187: six-seed ensemble",
        stated=0.288,
        recompute=ensemble("w206_bxdelaapoolmlp256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble("w206_bxdelaapoolmlp256_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 187: seed min (first six)",
        stated=0.242,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: seed max (first six)",
        stated=0.303,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 187: select epoch min",
        stated=29,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 187: select epoch max",
        stated=138,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 187: validation Pearson min",
        stated=0.36,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 187: validation Pearson max",
        stated=0.40,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 188, the three families through the hidden layer.
    Check(
        claim="round 188: three families mlp arm",
        stated=0.217,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: three families mlp arm sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: linear three families level (pair)",
        stated=0.218,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs linear",
        stated=-0.002,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs linear sd",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs linear seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 188: GO pool mlp level (pair)",
        stated=0.206,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against GO with the pool through the same layer",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs GO pool mlp",
        stated=0.011,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against GO with the pool through the same layer",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs GO pool mlp sd",
        stated=0.014,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against GO with the pool through the same layer",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs GO pool mlp seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against GO with the pool through the same layer",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 188: vs pool mlp",
        stated=-0.023,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the pool alone through the same layer",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs pool mlp sd",
        stated=0.046,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the pool alone through the same layer",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: vs pool mlp seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the pool alone through the same layer",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 188: three families mlp top-50",
        stated=4.53,
        recompute=transfer(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 188: six-seed ensemble",
        stated=0.227,
        recompute=ensemble("w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: six-seed ensemble top-50",
        stated=4.19,
        recompute=ensemble(
            "w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 188: seed min",
        stated=0.200,
        recompute=seed_level("w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: seed max",
        stated=0.226,
        recompute=seed_level("w207_bxdelgopfbamomaaapoolmlp_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 188: select epoch min",
        stated=5,
        recompute=pair_cells(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 188: select epoch max",
        stated=20,
        recompute=pair_cells(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 188: validation Pearson min",
        stated=0.25,
        recompute=pair_cells(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 188: validation Pearson max",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: three families through a hidden layer of 64 against the same columns in the linear affine",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 191, the pool with the HiGHS vector through the hidden layer.
    Check(
        claim="round 191: pool pfba mlp arm",
        stated=0.186,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: pool pfba mlp arm sd",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs pool mlp",
        stated=-0.054,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs pool mlp sd",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs pool mlp seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the pool alone through the same layer",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: HiGHS store level (pair)",
        stated=0.130,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs HiGHS store",
        stated=0.056,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs HiGHS store sd",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: vs HiGHS store seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: pool pfba mlp top-50",
        stated=2.90,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 191: HiGHS store top-50 (pair)",
        stated=4.46,
        recompute=transfer(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 191: six-seed ensemble",
        stated=0.207,
        recompute=ensemble("w210_bxdelaapoolpfbamlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: six-seed ensemble top-50",
        stated=4.59,
        recompute=ensemble("w210_bxdelaapoolpfbamlp_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 191: seed min",
        stated=0.138,
        recompute=seed_level("w210_bxdelaapoolpfbamlp_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: seed max",
        stated=0.234,
        recompute=seed_level("w210_bxdelaapoolpfbamlp_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 191: select epoch min",
        stated=29,
        recompute=pair_cells(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: select epoch max",
        stated=66,
        recompute=pair_cells(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: validation Pearson min",
        stated=0.34,
        recompute=pair_cells(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 191: validation Pearson max",
        stated=0.37,
        recompute=pair_cells(
            "betaxanthin: pool with the HiGHS vector through a hidden layer of 64 against the HiGHS store alone",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 191: the store's columns",
        stated=4196,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_aapoolpfba.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: genes with a flux row",
        stated=815,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_aapoolpfba.json",
                    )
                )
            )["n_genes_with_appended_row"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 191: the pool's genes",
        stated=4678,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_aapoolpfba.json",
                    )
                )
            )["n_genes"]
        ),
        tolerance=0.5,
    ),
    # sec:physics -- the CPU perceptron sweep over the measured pool (2026-09-26).
    Check(
        claim="pool mlp: pool alone 16",
        stated=0.178,
        recompute=pool_mlp(16, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 64",
        stated=0.217,
        recompute=pool_mlp(64, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 256",
        stated=0.219,
        recompute=pool_mlp(256, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 256 sd",
        stated=0.042,
        recompute=pool_mlp(256, "store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 1024",
        stated=0.236,
        recompute=pool_mlp(1024, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 1024 sd",
        stated=0.007,
        recompute=pool_mlp(1024, "store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 1024 ensemble",
        stated=0.240,
        recompute=pool_mlp(1024, "store_mlp", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 16",
        stated=0.237,
        recompute=pool_mlp(16, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 64",
        stated=0.232,
        recompute=pool_mlp(64, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 256",
        stated=0.238,
        recompute=pool_mlp(256, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 1024",
        stated=0.240,
        recompute=pool_mlp(1024, "go_store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 1024 sd",
        stated=0.003,
        recompute=pool_mlp(1024, "go_store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 1024 ensemble",
        stated=0.241,
        recompute=pool_mlp(1024, "go_store_mlp", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: GO with pool 1024 top-50",
        stated=5.16,
        recompute=pool_mlp(1024, "go_store_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="pool mlp: GO with pool 1024 select epoch min",
        stated=15,
        recompute=pool_mlp(1024, "go_store_mlp", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="pool mlp: GO with pool 1024 select epoch max",
        stated=16,
        recompute=pool_mlp(1024, "go_store_mlp", "select_epoch_max"),
        tolerance=0.5,
    ),
    Check(
        claim="pool mlp: seconds a seed at 1024 (GO with pool)",
        stated=14,
        recompute=pool_mlp(1024, "go_store_mlp", "seconds_max"),
        tolerance=1.5,
    ),
    Check(
        claim="pool mlp: GO perceptron alone (existing)",
        stated=0.200,
        recompute=go_model("go_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:physics -- the pool with the HiGHS vector on the ladder (2026-09-26).
    Check(
        claim="ladder: pool with HiGHS ridge",
        stated=0.095,
        recompute=store_baseline(
            "betaxanthin", "aapool_pfba", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with HiGHS kNN",
        stated=0.153,
        recompute=store_baseline("betaxanthin", "aapool_pfba", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with HiGHS forest",
        stated=0.220,
        recompute=store_baseline(
            "betaxanthin", "aapool_pfba", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with HiGHS kNN gap under the pool alone",
        stated=0.13,
        recompute=lambda: (
            store_baseline("betaxanthin", "aa_pool", "knn", "mean_spearman")()
            - store_baseline("betaxanthin", "aapool_pfba", "knn", "mean_spearman")()
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 196, the pool through 256 at 400 epochs.
    Check(
        claim="round 196: pool mlp256 e400 arm",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with a budget of 400 epochs against the budget of 200",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 196: pool mlp256 e400 arm sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with a budget of 400 epochs against the budget of 200",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 196: paired difference is zero",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with a budget of 400 epochs against the budget of 200",
            "mean_difference",
        ),
        tolerance=0.00005,
    ),
    Check(
        claim="round 196: paired sd is zero",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with a budget of 400 epochs against the budget of 200",
            "sd_difference",
        ),
        tolerance=0.00005,
    ),
    # sec:physics -- the raw-encoding perceptrons and their 1,000-epoch rerun (2026-09-27).
    Check(
        claim="raw mlp: pool alone 256",
        stated=0.158,
        recompute=raw_mlp("pool_raw_h256", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: pool alone 1024",
        stated=0.173,
        recompute=raw_mlp("pool_raw_h1024", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: pool alone 1024 at 1000 epochs",
        stated=0.177,
        recompute=raw_mlp("pool_raw_h1024_e1000", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: two slots 1024",
        stated=0.230,
        recompute=raw_mlp(
            "pool_raw_h1024", "go_linear_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: two slots 1024 at 1000 epochs",
        stated=0.230,
        recompute=raw_mlp(
            "pool_raw_h1024_e1000", "go_linear_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: three families split 1024",
        stated=0.243,
        recompute=raw_mlp(
            "threefam_split_raw_h1024", "store_split_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: three families split at 1000 epochs",
        stated=0.242,
        recompute=raw_mlp(
            "threefam_split_raw_h1024_e1000", "store_split_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: GO pool split 1024",
        stated=0.237,
        recompute=raw_mlp(
            "gopool_split_raw_h1024", "store_split_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: GO pool split at 1000 epochs",
        stated=0.237,
        recompute=raw_mlp(
            "gopool_split_raw_h1024_e1000", "store_split_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: standardized best (pool alone 1024)",
        stated=0.236,
        recompute=pool_mlp(1024, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: encoding cost on the pool alone",
        stated=0.06,
        recompute=lambda: (
            pool_mlp(1024, "store_mlp", "single_mean_spearman")()
            - raw_mlp("pool_raw_h1024", "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.006,
    ),
    Check(
        claim="raw mlp: gap to the transformer",
        stated=0.043,
        recompute=lambda: (
            transfer(
                "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64",
                "mean_spearman",
                "arm",
            )()
            - pool_mlp(1024, "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.0015,
    ),
    Check(
        claim="raw mlp: standardized GO pool split 1024",
        stated=0.179,
        recompute=lambda: float(
            next(
                a
                for a in json.load(
                    open(
                        osp.join(
                            RESULTS, "go_models_betaxanthin_gopool_split_h1024.json"
                        )
                    )
                )["arms"]
                if a["arm"] == "store_split_mlp"
            )["single_mean_spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: standardized three families split 1024",
        stated=0.106,
        recompute=lambda: float(
            next(
                a
                for a in json.load(
                    open(
                        osp.join(
                            RESULTS, "go_models_betaxanthin_threefam_split_h1024.json"
                        )
                    )
                )["arms"]
                if a["arm"] == "store_split_mlp"
            )["single_mean_spearman"]
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="raw mlp: transformer three families two slots within a thousandth",
        stated=0.003,
        recompute=lambda: abs(
            transfer(
                "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
                "mean_spearman",
                "arm",
            )()
            - raw_mlp(
                "threefam_split_raw_h1024_e1000",
                "store_split_mlp",
                "single_mean_spearman",
            )()
        ),
        tolerance=0.0025,
    ),
    # sec:physics -- the Cooper pool on the ladder (2026-09-26).
    Check(
        claim="ladder: Cooper pool ridge",
        stated=-0.001,
        recompute=store_baseline(
            "betaxanthin", "cooper_pool", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: Cooper pool kNN",
        stated=0.094,
        recompute=store_baseline("betaxanthin", "cooper_pool", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: Cooper pool forest",
        stated=0.113,
        recompute=store_baseline(
            "betaxanthin", "cooper_pool", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: Cooper pool top-50 min (ridge)",
        stated=0.6,
        recompute=store_baseline(
            "betaxanthin", "cooper_pool", "ridge", "top50_enrichment"
        ),
        tolerance=0.06,
    ),
    Check(
        claim="ladder: Cooper pool top-50 max (kNN)",
        stated=1.8,
        recompute=store_baseline(
            "betaxanthin", "cooper_pool", "knn", "top50_enrichment"
        ),
        tolerance=0.06,
    ),
    Check(
        claim="ladder: Cooper pool rows filled with the reference",
        stated=466,
        recompute=lambda: float(
            sum(
                json.load(open(osp.join(RESULTS, "store_baselines_betaxanthin.json")))[
                    "representations"
                ]["cooper_pool"]["filled_wild_type"].values()
            )
        ),
        tolerance=0.5,
    ),
    Check(
        claim="Cooper pool store: strains",
        stated=4313,
        recompute=lambda: float(
            json.load(
                open(osp.join(RESULTS, "target_pool_store_YPD_bx_cooperpool.json"))
            )["n_genes"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="Cooper pool store: percent of cells missing",
        stated=8.5,
        recompute=lambda: (
            100.0
            * json.load(
                open(osp.join(RESULTS, "target_pool_store_YPD_bx_cooperpool.json"))
            )["n_missing_cells_filled_with_reference"]
            / (4313 * 17)
        ),
        tolerance=0.06,
    ),
    # sec:physics -- the two-slot perceptron and depth over the pool (2026-09-26).
    Check(
        claim="pool mlp: two slots 256",
        stated=0.260,
        recompute=pool_mlp(
            256,
            "go_linear_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_twoslot_h256.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024",
        stated=0.265,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_twoslot_h1024.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024 sd",
        stated=0.006,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "single_sd_spearman",
            "go_models_betaxanthin_pool_twoslot_h1024.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024 ensemble",
        stated=0.266,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "ensemble_spearman",
            "go_models_betaxanthin_pool_twoslot_h1024.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024 top-50",
        stated=5.36,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "single_mean_top50",
            "go_models_betaxanthin_pool_twoslot_h1024.json",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="pool mlp: two slots 1024 parameters",
        stated=23872,
        recompute=lambda: float(
            next(
                a
                for a in json.load(
                    open(
                        osp.join(
                            RESULTS, "go_models_betaxanthin_pool_twoslot_h1024.json"
                        )
                    )
                )["arms"]
                if a["arm"] == "go_linear_store_mlp"
            )["seeds"][0]["n_parameters"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="pool mlp: pool alone 1024 depth 2",
        stated=0.185,
        recompute=pool_mlp(
            1024,
            "store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d2.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: pool alone 1024 depth 3",
        stated=0.195,
        recompute=pool_mlp(
            1024,
            "store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d3.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: shared layer 1024 depth 2",
        stated=0.247,
        recompute=pool_mlp(
            1024,
            "go_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d2.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: shared layer 1024 depth 3",
        stated=0.243,
        recompute=pool_mlp(
            1024,
            "go_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d3.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024 depth 2",
        stated=0.262,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d2.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots 1024 depth 3",
        stated=0.258,
        recompute=pool_mlp(
            1024,
            "go_linear_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_h1024_d3.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="pool mlp: two slots over the pool alone at 1024",
        stated=0.03,
        recompute=lambda: (
            pool_mlp(
                1024,
                "go_linear_store_mlp",
                "single_mean_spearman",
                "go_models_betaxanthin_pool_twoslot_h1024.json",
            )()
            - pool_mlp(1024, "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.006,
    ),
    # sec:rounds -- round 190, the pool through 64 at 400 epochs.
    Check(
        claim="round 190: pool mlp e400 arm",
        stated=0.240,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 190: pool mlp e400 arm sd",
        stated=0.048,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 190: paired difference is zero",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
            "mean_difference",
        ),
        tolerance=0.00005,
    ),
    Check(
        claim="round 190: paired sd is zero",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
            "sd_difference",
        ),
        tolerance=0.00005,
    ),
    Check(
        claim="round 190: select epoch max is the same 197",
        stated=197,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 64 with a budget of 400 epochs against the budget of 200",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- rounds 192 to 194, the width curve of the pool's hidden read.
    Check(
        claim="round 192: pool mlp512 arm",
        stated=0.243,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: pool mlp512 arm sd",
        stated=0.048,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: vs 256",
        stated=-0.036,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: vs 256 sd",
        stated=0.059,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: vs 256 seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 192: vs 64",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: vs 64 sd",
        stated=0.072,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: vs 64 seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 192: top-50",
        stated=4.36,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 192: six-seed ensemble",
        stated=0.278,
        recompute=ensemble("w211_bxdelaapoolmlp512_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: six-seed ensemble top-50",
        stated=5.59,
        recompute=ensemble("w211_bxdelaapoolmlp512_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 192: seed min",
        stated=0.181,
        recompute=seed_level("w211_bxdelaapoolmlp512_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: seed max",
        stated=0.289,
        recompute=seed_level("w211_bxdelaapoolmlp512_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 192: select epoch min",
        stated=2,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 192: select epoch max",
        stated=91,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 192: validation Pearson min",
        stated=0.36,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 192: validation Pearson max",
        stated=0.39,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 512 against the layer of 64",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 193: pool mlp256 second six",
        stated=0.277,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "last6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: pool mlp256 second six sd",
        stated=0.029,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "last6_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: pool mlp256 first six",
        stated=0.279,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "first6"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: pool mlp256 twelve seeds",
        stated=0.278,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: pool mlp256 twelve-seed sd",
        stated=0.025,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: pool mlp256 twelve-seed sem",
        stated=0.007,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "sem"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: twelve-seed min",
        stated=0.241,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: twelve-seed max",
        stated=0.313,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: select epoch min",
        stated=29,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 193: select epoch max",
        stated=141,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 193: twelve-seed ensemble",
        stated=0.296,
        recompute=ensemble(
            "w206_bxdelaapoolmlp256_h12l1p1_e200", "spearman", "all_seeds"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: twelve-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble(
            "w206_bxdelaapoolmlp256_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 193: vs 64 twelve seeds",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: vs 64 twelve seeds above",
        stated=9,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the layer of 64, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 193: vs GO alone twelve seeds",
        stated=0.075,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: vs GO alone twelve seeds above",
        stated=12,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 193: vs control twelve seeds",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the control, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 193: vs control twelve seeds above",
        stated=12,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against the control, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 193: twelve-seed top-50",
        stated=4.61,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 193: GO alone twelve-seed top-50 (pair)",
        stated=4.99,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 against GO alone, twelve seeds",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 194: pool mlp128 arm",
        stated=0.262,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: pool mlp128 arm sd",
        stated=0.028,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: vs 64",
        stated=0.022,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: vs 64 sd",
        stated=0.069,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: vs 64 seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 194: vs 256",
        stated=-0.017,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: vs 256 sd",
        stated=0.042,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: vs 256 seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 194: top-50",
        stated=4.53,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 194: six-seed ensemble",
        stated=0.279,
        recompute=ensemble("w213_bxdelaapoolmlp128_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble("w213_bxdelaapoolmlp128_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 194: seed min",
        stated=0.214,
        recompute=seed_level("w213_bxdelaapoolmlp128_h12l1p1_e200", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: seed max",
        stated=0.290,
        recompute=seed_level("w213_bxdelaapoolmlp128_h12l1p1_e200", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: select epoch min",
        stated=21,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 194: select epoch max",
        stated=51,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 194: validation Pearson min",
        stated=0.36,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 194: validation Pearson max",
        stated=0.41,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 128 against the layer of 64",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 194: width curve ensemble 64",
        stated=0.292,
        recompute=ensemble("w199_bxdelaapoolmlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 194: width curve ensemble 256 (first six)",
        stated=0.288,
        recompute=ensemble("w206_bxdelaapoolmlp256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- rounds 197 and 198, the two-slot head in the transformer.
    Check(
        claim="round 197: two slots 64 arm",
        stated=0.224,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: two slots 64 arm sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: shared layer level (pair)",
        stated=0.206,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns through one shared layer of 64",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs shared layer",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns through one shared layer of 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs shared layer sd",
        stated=0.007,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns through one shared layer of 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs shared layer seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns through one shared layer of 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 197: linear level (pair)",
        stated=0.208,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns in the linear affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs linear",
        stated=0.016,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs linear sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs linear seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the same columns in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 197: GO alone level (pair)",
        stated=0.202,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs GO alone",
        stated=0.022,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs GO alone sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs GO alone seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 197: pool mlp64 level (pair)",
        stated=0.240,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the pool alone through 64",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs pool mlp64",
        stated=-0.016,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the pool alone through 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs pool mlp64 sd",
        stated=0.049,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the pool alone through 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: vs pool mlp64 seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against the pool alone through 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 197: top-50",
        stated=5.66,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 197: six-seed ensemble",
        stated=0.225,
        recompute=ensemble("w216_bxdelgolinaapoolmlp_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 197: six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w216_bxdelgolinaapoolmlp_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 197: select epoch min",
        stated=5,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 198: select epoch max",
        stated=15,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against GO alone",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 197: validation Pearson min",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 64, against GO alone",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 198: validation Pearson max",
        stated=0.29,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against GO alone",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 198: two slots 256 arm",
        stated=0.225,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against GO alone",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: two slots 256 arm sd",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against GO alone",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: vs two slots 64",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against the two slots at 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: vs pool mlp256",
        stated=-0.054,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against the pool alone through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: vs pool mlp256 sd",
        stated=0.020,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against the pool alone through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: vs pool mlp256 seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against the pool alone through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 198: pool mlp256 level (pair)",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 256, against the pool alone through 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 198: six-seed ensemble",
        stated=0.226,
        recompute=ensemble("w217_bxdelgolinaapoolmlp256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 199, the three families with two slots.
    Check(
        claim="round 199: arm",
        stated=0.239,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: arm sd",
        stated=0.003,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: linear level (pair)",
        stated=0.218,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs linear",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs linear sd",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs linear seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: shared level (pair)",
        stated=0.217,
        recompute=transfer(
            "betaxanthin: three families with two slots against the same columns through one shared layer of 64",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs shared",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: three families with two slots against the same columns through one shared layer of 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs shared sd",
        stated=0.010,
        recompute=transfer(
            "betaxanthin: three families with two slots against the same columns through one shared layer of 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs shared seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: three families with two slots against the same columns through one shared layer of 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: GO two-slot level (pair)",
        stated=0.225,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO linear beside the pool through 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs GO two-slot",
        stated=0.014,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO linear beside the pool through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs GO two-slot sd",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO linear beside the pool through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs GO two-slot seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO linear beside the pool through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: pool mlp256 level (pair)",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs pool mlp256",
        stated=-0.040,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs pool mlp256 sd",
        stated=0.022,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: vs pool mlp256 seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: top-50",
        stated=5.32,
        recompute=transfer(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 199: six-seed ensemble",
        stated=0.241,
        recompute=ensemble(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200", "spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: six-seed ensemble top-50",
        stated=5.19,
        recompute=ensemble(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 199: seed min",
        stated=0.235,
        recompute=seed_level(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200", "first6_min"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: seed max",
        stated=0.243,
        recompute=seed_level(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200", "first6_max"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 199: select epoch min",
        stated=7,
        recompute=pair_cells(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: select epoch max",
        stated=12,
        recompute=pair_cells(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 199: validation Pearson min",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 199: validation Pearson max",
        stated=0.28,
        recompute=pair_cells(
            "betaxanthin: three families with two slots, GO and both flux programs linear beside the pool through 256, against the same columns in the linear affine",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 199: the store's columns",
        stated=10735,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_store_concat_YPD_bx_gopfbamomaaapool.json",
                    )
                )
            )["n_columns"]
        ),
        tolerance=0.5,
    ),
    # sec:physics -- the minimum-norm (l2) store (2026-09-25).
    Check(
        claim="l2 store: genes",
        stated=1155,
        recompute=physics_store("betaxanthin", "l2", "n_genes"),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: seconds",
        stated=49,
        recompute=physics_store("betaxanthin", "l2", "seconds"),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: not converged",
        stated=0,
        recompute=physics_store("betaxanthin", "l2", "n_not_converged"),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: lethal",
        stated=93,
        recompute=physics_store("betaxanthin", "l2", "n_lethal"),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: growth ratios moved",
        stated=17,
        recompute=physics_store("betaxanthin", "l2", "n_genes_ratio_moved_gt_0.01"),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: wild-type reactions above 1e-6",
        stated=996,
        recompute=lambda: float(
            (
                np.abs(
                    np.load(
                        osp.join(
                            os.environ["DATA_ROOT"],
                            "data/torchcell/reference_flux/ess_solve_encodings_YPD_bx_l2.npz",
                        )
                    )["v_wild_type"]
                )
                > 1e-6
            ).sum()
        ),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: HiGHS wild-type reactions above 1e-6",
        stated=426,
        recompute=lambda: float(
            (
                np.abs(
                    np.load(
                        osp.join(
                            os.environ["DATA_ROOT"],
                            "data/torchcell/reference_flux/ess_solve_encodings_YPD_bx_pfba.npz",
                        )
                    )["v_wild_type"][:4177]
                )
                > 1e-6
            ).sum()
        ),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: wild-type l1 norm",
        stated=452,
        recompute=lambda: float(
            np.abs(
                np.load(
                    osp.join(
                        os.environ["DATA_ROOT"],
                        "data/torchcell/reference_flux/ess_solve_encodings_YPD_bx_l2.npz",
                    )
                )["v_wild_type"].astype(np.float64)
            ).sum()
        ),
        tolerance=0.5,
    ),
    Check(
        claim="l2 store: HiGHS wild-type l1 norm",
        stated=426,
        recompute=lambda: float(
            np.abs(
                np.load(
                    osp.join(
                        os.environ["DATA_ROOT"],
                        "data/torchcell/reference_flux/ess_solve_encodings_YPD_bx_pfba.npz",
                    )
                )["v_wild_type"][:4177].astype(np.float64)
            ).sum()
        ),
        tolerance=0.5,
    ),
    Check(
        claim="degeneracy: l2 no-effect vectors differing",
        stated=427,
        recompute=degeneracy("YPD_bx_l2", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    Check(
        claim="degeneracy: l2 median distance",
        stated=0.0,
        recompute=degeneracy("YPD_bx_l2", "median_rel_l2_log1p"),
        tolerance=1e-7,
    ),
    Check(
        claim="degeneracy: l2 p90 distance",
        stated=3e-5,
        recompute=degeneracy("YPD_bx_l2", "p90_rel_l2_log1p"),
        tolerance=6e-6,
    ),
    Check(
        claim="degeneracy: HiGHS p90 distance",
        stated=0.39,
        recompute=degeneracy("YPD_bx_pfba", "p90_rel_l2_log1p"),
        tolerance=0.005,
    ),
    Check(
        claim="degeneracy: GLPK p90 distance",
        stated=0.38,
        recompute=degeneracy("YPD_bx", "p90_rel_l2_log1p"),
        tolerance=0.005,
    ),
    Check(
        claim="ladder: l2 store ridge",
        stated=0.068,
        recompute=store_baseline("betaxanthin", "pfba_l2", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: l2 store kNN",
        stated=0.126,
        recompute=store_baseline("betaxanthin", "pfba_l2", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: l2 store forest",
        stated=0.133,
        recompute=store_baseline("betaxanthin", "pfba_l2", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: l2 store ridge top-50",
        stated=0.40,
        recompute=store_baseline("betaxanthin", "pfba_l2", "ridge", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: l2 store forest top-50",
        stated=2.00,
        recompute=store_baseline(
            "betaxanthin", "pfba_l2", "forest", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: HiGHS store ridge (l2 comparison)",
        stated=0.098,
        recompute=store_baseline("betaxanthin", "pfba_highs", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: HiGHS store forest (l2 comparison)",
        stated=0.117,
        recompute=store_baseline(
            "betaxanthin", "pfba_highs", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with l2 ridge",
        stated=0.068,
        recompute=store_baseline("betaxanthin", "gol2", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with l2 kNN",
        stated=0.208,
        recompute=store_baseline("betaxanthin", "gol2", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with l2 forest",
        stated=0.173,
        recompute=store_baseline("betaxanthin", "gol2", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:physics -- the measured amino-acid pool in the ladder (2026-09-25).
    Check(
        claim="ladder: pool ridge",
        stated=0.149,
        recompute=store_baseline("betaxanthin", "aa_pool", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool kNN",
        stated=0.285,
        recompute=store_baseline("betaxanthin", "aa_pool", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool forest",
        stated=0.280,
        recompute=store_baseline("betaxanthin", "aa_pool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool ridge top-50",
        stated=3.99,
        recompute=store_baseline("betaxanthin", "aa_pool", "ridge", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: pool kNN top-50",
        stated=4.19,
        recompute=store_baseline("betaxanthin", "aa_pool", "knn", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: pool forest top-50",
        stated=4.79,
        recompute=store_baseline(
            "betaxanthin", "aa_pool", "forest", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: GO with pool ridge",
        stated=0.102,
        recompute=store_baseline("betaxanthin", "go_aapool", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with pool kNN",
        stated=0.174,
        recompute=store_baseline("betaxanthin", "go_aapool", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with pool forest",
        stated=0.264,
        recompute=store_baseline("betaxanthin", "go_aapool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO with pool forest top-50",
        stated=5.99,
        recompute=store_baseline(
            "betaxanthin", "go_aapool", "forest", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: GO HiGHS with pool ridge",
        stated=0.156,
        recompute=store_baseline(
            "betaxanthin", "gopfba_aapool", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO HiGHS with pool kNN",
        stated=0.195,
        recompute=store_baseline(
            "betaxanthin", "gopfba_aapool", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO HiGHS with pool forest",
        stated=0.247,
        recompute=store_baseline(
            "betaxanthin", "gopfba_aapool", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: GO HiGHS with pool forest top-50",
        stated=6.19,
        recompute=store_baseline(
            "betaxanthin", "gopfba_aapool", "forest", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: pool store strains",
        stated=4678,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        os.environ["EXPERIMENT_ROOT"],
                        "031-betaxanthin-module",
                        "results",
                        "target_pool_store_YPD_bx_aapool.json",
                    )
                )
            )["n_genes"]
        ),
        tolerance=0.5,
    ),
    # sec:physics -- the driving-force polytope sampled.
    Check(
        claim="physics: polytope wild-type constrained metabolites",
        stated=234,
        recompute=polytope("wild_type", "n_constrained_metabolites"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope wild-type spread over constrained",
        stated=2.18,
        recompute=polytope("wild_type", "median_sd_ln_c_constrained"),
        tolerance=0.006,
    ),
    Check(
        claim="physics: polytope wild-type spread over free",
        stated=2.63,
        recompute=polytope("wild_type", "median_sd_ln_c_free"),
        tolerance=0.006,
    ),
    Check(
        claim="physics: polytope rows differing from wild type",
        stated=455,
        recompute=polytope("n_mean_differs_from_wild_type"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope viable rows solved",
        stated=1062,
        recompute=polytope("n_solved"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: polytope mean distance to point",
        stated=0.12,
        recompute=polytope("median_rel_distance_mean_to_point_viable"),
        tolerance=0.006,
    ),
    Check(
        claim="store-baselines: polytope mean ridge",
        stated=0.136,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_mean", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: polytope mean kNN",
        stated=0.127,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_mean", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: polytope mean forest",
        stated=0.126,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_mean", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: polytope spread ridge",
        stated=0.096,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_spread", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: polytope spread kNN",
        stated=0.138,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_spread", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: polytope spread forest",
        stated=0.128,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_poly_spread", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: capped concentrations kNN",
        stated=0.072,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_conc", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: capped concentrations forest",
        stated=0.098,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_conc", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: HiGHS store forest",
        stated=0.117,
        recompute=store_baseline(
            "betaxanthin", "pfba_highs", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:physics -- the max-min driving-force store.
    Check(
        claim="physics: driving-force wild-type optimum",
        stated=-1.16,
        recompute=driving_force("mdfcap100", "wild_type", "mdf_kj_per_mol"),
        tolerance=0.006,
    ),
    Check(
        claim="physics: driving-force wild-type constrained reactions",
        stated=172,
        recompute=driving_force("mdfcap100", "wild_type", "n_active"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: driving-force wild-type binding reactions",
        stated=32,
        recompute=driving_force("mdfcap100", "wild_type", "n_binding"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: driving-force viable deletions solved",
        stated=1062,
        recompute=driving_force("mdfcap100", "n_solved"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: driving-force deletions with a negative optimum",
        stated=1062,
        recompute=driving_force("mdfcap100", "n_mdf_negative"),
        tolerance=0.5,
    ),
    Check(
        claim="physics: driving-force deletions moving the concentrations",
        stated=451,
        recompute=driving_force("mdfcap100", "n_conc_moved"),
        tolerance=0.5,
    ),
    # sec:target-transfer -- rounds 117 and 118.
    Check(
        claim="target-transfer: betaxanthin store arm mean",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin store arm sd",
        stated=0.011,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin control mean",
        stated=0.079,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin control sd",
        stated=0.056,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_spearman",
            "control",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin paired difference",
        stated=0.043,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin paired sd",
        stated=0.049,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: betaxanthin seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: betaxanthin store top-50 enrichment",
        stated=3.49,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin control top-50 enrichment",
        stated=4.06,
        recompute=transfer(
            "betaxanthin: whole-vector store affine against the control",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin store six-seed ensemble",
        stated=0.135,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: betaxanthin store triple 0",
        stated=0.146,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman", "triple0"),
    ),
    Check(
        claim="target-transfer: betaxanthin store triple 1",
        stated=0.133,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "spearman", "triple1"),
    ),
    Check(
        claim="target-transfer: betaxanthin control six-seed ensemble",
        stated=0.076,
        recompute=ensemble("w134_bxctx_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: betaxanthin store ensemble top-50",
        stated=4.39,
        recompute=ensemble("w134_bxstore_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: betaxanthin control ensemble top-50",
        stated=4.59,
        recompute=ensemble("w134_bxctx_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: 027 precursor comparator",
        stated=0.131,
        recompute=transfer_comparator("betaxanthin", "027 flux precursor"),
    ),
    Check(
        claim="target-transfer: 027 context comparator",
        stated=0.115,
        recompute=transfer_comparator("betaxanthin", "027 context-only"),
    ),
    Check(
        claim="target-transfer: kNN comparator",
        stated=0.145,
        recompute=transfer_comparator("betaxanthin", "cosine kNN"),
    ),
    # sec:target-transfer -- rounds 119 and 120.
    Check(
        claim="target-transfer: two summary columns arm mean",
        stated=0.101,
        recompute=transfer(
            "betaxanthin: two summary columns against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: two summary columns arm sd",
        stated=0.061,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "sd_spearman", "arm"
        ),
    ),
    Check(
        claim="target-transfer: two summary columns paired difference",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: two summary columns paired sd",
        stated=0.072,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: two summary columns seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: two summary columns against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: two summary columns single-seed top-50",
        stated=3.03,
        recompute=transfer(
            "betaxanthin: two summary columns against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: two summary columns six-seed ensemble",
        stated=0.071,
        recompute=ensemble("w136_bxratio_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: decayed affine arm mean",
        stated=0.073,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine arm sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine paired difference",
        stated=-0.005,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine paired sd",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: decayed affine seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only graph arm mean",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph arm sd",
        stated=0.075,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph paired difference",
        stated=-0.020,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph paired sd",
        stated=0.076,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only graph seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: deletion-only cell graphs against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rule-count columns arm mean",
        stated=0.074,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns arm sd",
        stated=0.059,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns paired difference",
        stated=-0.004,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns paired sd",
        stated=0.092,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: rule-count columns seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: rounds 119/120 ensemble top-50 floor (deletion-only)",
        stated=4.79,
        recompute=ensemble("w137_bxdel_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 ensemble top-50 ceiling (decayed affine)",
        stated=5.19,
        recompute=ensemble("w136_bxstorewd_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 single-seed top-50 floor (rule counts)",
        stated=2.96,
        recompute=transfer(
            "betaxanthin: six rule-count columns against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: rounds 119/120 single-seed top-50 ceiling (decayed affine)",
        stated=3.59,
        recompute=transfer(
            "betaxanthin: whole vector, affine decayed and slowed, against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 123, the deletion-only graph with the store affine.
    Check(
        claim="target-transfer: deletion-only store arm mean",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store arm sd",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, paired",
        stated=0.063,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, paired sd",
        stated=0.065,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs deletion-only graph, seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, paired",
        stated=0.000,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, paired sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store vs old-graph store, seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store single-seed top-50",
        stated=4.93,
        recompute=transfer(
            "betaxanthin: deletion-only graphs with the store affine against the store affine",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: deletion-only store ensemble top-50",
        stated=5.59,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: deletion-only store six-seed ensemble",
        stated=0.131,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 122, the width-90 betaxanthin control.
    Check(
        claim="target-transfer: width-90 control arm mean",
        stated=0.109,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control arm sd",
        stated=0.053,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control paired difference",
        stated=0.030,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control paired sd",
        stated=0.079,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: width-90 control seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: width-90 control single-seed top-50",
        stated=4.73,
        recompute=transfer(
            "betaxanthin: width 90, two blocks, against the width-12 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: width-90 control six-seed ensemble",
        stated=0.134,
        recompute=ensemble("w139_bxwide_h90l2p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: width-90 control ensemble top-50",
        stated=5.59,
        recompute=ensemble("w139_bxwide_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 123, the store affine at width 90.
    Check(
        claim="target-transfer: wide store arm mean",
        stated=0.134,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: wide store arm sd",
        stated=0.020,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: wide store paired difference",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: wide store paired sd",
        stated=0.051,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: wide store seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: wide store single-seed top-50",
        stated=4.09,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine against the width-90 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: wide store six-seed ensemble",
        stated=0.153,
        recompute=ensemble("w140_bxwidestore_h90l2p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: wide store ensemble top-50",
        stated=5.19,
        recompute=ensemble("w140_bxwidestore_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 125, the growth and production blocks alone.
    Check(
        claim="target-transfer: growth block arm mean",
        stated=0.033,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: growth block arm sd",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector",
        stated=-0.088,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector sd",
        stated=0.029,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: growth block vs whole vector seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: the growth block alone against the whole vector", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: growth block vs control",
        stated=-0.046,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "mean_difference"
        ),
    ),
    Check(
        claim="target-transfer: growth block vs control sd",
        stated=0.064,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "sd_difference"
        ),
    ),
    Check(
        claim="target-transfer: growth block vs control seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: growth block single-seed top-50",
        stated=3.06,
        recompute=transfer(
            "betaxanthin: the growth block alone against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: growth block six-seed ensemble",
        stated=0.059,
        recompute=ensemble("w142_bxgrowth_h12l1p1_e200", "spearman"),
    ),
    Check(
        claim="target-transfer: production block arm mean",
        stated=0.091,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "mean_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: production block arm sd",
        stated=0.040,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "sd_spearman",
            "arm",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector",
        stated=-0.030,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector sd",
        stated=0.034,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs whole vector seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: the production block alone against the whole vector",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: production block vs control",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: the production block alone against the control",
            "mean_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs control sd",
        stated=0.021,
        recompute=transfer(
            "betaxanthin: the production block alone against the control",
            "sd_difference",
        ),
    ),
    Check(
        claim="target-transfer: production block vs control seeds above",
        stated=5,
        recompute=transfer(
            "betaxanthin: the production block alone against the control", "n_positive"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: production block single-seed top-50",
        stated=3.63,
        recompute=transfer(
            "betaxanthin: the production block alone against the control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: production block six-seed ensemble",
        stated=0.112,
        recompute=ensemble("w142_bxprod_h12l1p1_e200", "spearman"),
    ),
    # sec:target-transfer -- round 127, twelve seeds of the deletion-only store.
    Check(
        claim="target-transfer: deletion-only store twelve-seed single mean",
        stated=0.116,
        recompute=ensemble(
            "w140_bxdelstore_h12l1p1_e200", "mean_single_spearman", "arm"
        ),
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed count",
        stated=12,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "n_seeds", "arm"),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed ensemble",
        stated=0.129,
        recompute=ensemble("w140_bxdelstore_h12l1p1_e200", "spearman", "all_seeds"),
    ),
    Check(
        claim="target-transfer: deletion-only store twelve-seed ensemble top-50",
        stated=5.39,
        recompute=ensemble(
            "w140_bxdelstore_h12l1p1_e200", "top50_enrichment", "all_seeds"
        ),
        tolerance=0.006,
    ),
    # sec:target-transfer -- round 128, the wide store on the deletion-only graph.
    Check(
        claim="target-transfer: wide corrected-graph store arm",
        stated=0.120,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store arm sd",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store vs wide control",
        stated=0.011,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store vs wide control sd",
        stated=0.067,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store vs wide control seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store vs width-12 store",
        stated=-0.001,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-12 one",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store vs width-12 store sd",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-12 one",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store top-50",
        stated=4.43,
        recompute=transfer(
            "betaxanthin: width 90 with the store affine on deletion-only graphs against the width-90 control",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store six-seed ensemble",
        stated=0.147,
        recompute=ensemble("w145_bxwidedelstore_h90l2p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: wide corrected-graph store six-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble("w145_bxwidedelstore_h90l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-transfer -- rounds 130 and 131, the two affine schedules.
    Check(
        claim="target-transfer: affine at 40 arm",
        stated=0.074,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 arm sd",
        stated=0.046,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 store level",
        stated=0.121,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs store",
        stated=-0.047,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs store sd",
        stated=0.053,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs store seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: affine at 40 control level",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs control",
        stated=0.016,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs control sd",
        stated=0.068,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 vs control seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: affine at 40 top-50",
        stated=3.86,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: affine at 40 store top-50",
        stated=4.93,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against the deletion-only store",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: affine at 40 control top-50",
        stated=3.39,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine frozen to epoch 40, against deletion-only graphs",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: affine at 40 six-seed ensemble",
        stated=0.129,
        recompute=ensemble("w147_bxdelstore40_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine at 40 six-seed ensemble top-50",
        stated=5.79,
        recompute=ensemble("w147_bxdelstore40_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: affine from escape arm",
        stated=0.11,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape arm sd",
        stated=0.05,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape vs store",
        stated=-0.01,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape vs store sd",
        stated=0.052,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape vs store seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: affine from escape vs control",
        stated=0.052,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against deletion-only graphs",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape vs control sd",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against deletion-only graphs",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against deletion-only graphs",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-transfer: affine from escape top-50",
        stated=4.66,
        recompute=transfer(
            "betaxanthin: deletion-only store, affine from escape, against the deletion-only store",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="target-transfer: affine from escape six-seed ensemble",
        stated=0.120,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="target-transfer: affine from escape six-seed ensemble top-50",
        stated=6.19,
        recompute=ensemble("w148_bxdelstorer15_h12l1p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    # sec:target-physics -- the direction rule, the MOMA and HiGHS stores, degeneracy.
    Check(
        claim="target-physics: bx sign10 fixed forward",
        stated=1754,
        recompute=physics_store("betaxanthin", "sign10", "thermo", "n_fixed_forward"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign10 fixed reverse",
        stated=1068,
        recompute=physics_store("betaxanthin", "sign10", "thermo", "n_fixed_reverse"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign10 conflicts",
        stated=97,
        recompute=physics_store(
            "betaxanthin", "sign10", "thermo", "n_conflicts_with_plain_wild_type_flux"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign10 wild type infeasible",
        stated=1,
        recompute=physics_store("betaxanthin", "sign10", "wild_type_infeasible"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign30 fixed forward",
        stated=1071,
        recompute=physics_store("betaxanthin", "sign30", "thermo", "n_fixed_forward"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign30 fixed reverse",
        stated=905,
        recompute=physics_store("betaxanthin", "sign30", "thermo", "n_fixed_reverse"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx sign30 conflicts",
        stated=69,
        recompute=physics_store(
            "betaxanthin", "sign30", "thermo", "n_conflicts_with_plain_wild_type_flux"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx MOMA lethal",
        stated=107,
        recompute=physics_store("betaxanthin", "moma", "n_lethal"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx MOMA lethal in the source",
        stated=93,
        recompute=physics_store("betaxanthin", "moma", "n_lethal_source"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx MOMA ratios moved",
        stated=59,
        recompute=physics_store("betaxanthin", "moma", "n_genes_ratio_moved_gt_0.01"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx MOMA vectors moved",
        stated=1056,
        recompute=physics_store("betaxanthin", "moma", "n_genes_log1p_vector_moved"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx MOMA median distance",
        stated=0.39,
        recompute=physics_store(
            "betaxanthin", "moma", "median_relative_l2_log1p_to_source"
        ),
        tolerance=0.005,
    ),
    Check(
        claim="target-physics: bx HiGHS lethal",
        stated=93,
        recompute=physics_store("betaxanthin", "pfba", "n_lethal"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx HiGHS ratios moved",
        stated=17,
        recompute=physics_store("betaxanthin", "pfba", "n_genes_ratio_moved_gt_0.01"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: bx HiGHS median distance to GLPK",
        stated=0.015,
        recompute=physics_store(
            "betaxanthin", "pfba", "median_relative_l2_log1p_to_source"
        ),
        tolerance=0.0005,
    ),
    Check(
        claim="target-physics: GLPK store no-effect deletions",
        stated=1031,
        recompute=degeneracy("YPD_bx", "n_no_effect"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: GLPK store no-effect vectors differing",
        stated=1025,
        recompute=degeneracy("YPD_bx", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: GLPK store no-effect median distance",
        stated=0.38,
        recompute=degeneracy("YPD_bx", "median_rel_l2_log1p"),
        tolerance=0.005,
    ),
    Check(
        claim="target-physics: HiGHS store no-effect vectors differing",
        stated=428,
        recompute=degeneracy("YPD_bx_pfba", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    Check(
        claim="target-physics: HiGHS store no-effect p90 distance",
        stated=0.39,
        recompute=degeneracy("YPD_bx_pfba", "p90_rel_l2_log1p"),
        tolerance=0.005,
    ),
    Check(
        claim="target-physics: bx MOMA store no-effect vectors differing",
        stated=18,
        recompute=degeneracy("YPD_bx_moma", "n_no_effect_vector_differs"),
        tolerance=0.5,
    ),
    # sec:target-physics -- the store read without the backbone.
    Check(
        claim="store-baselines: GLPK growth block ridge",
        stated=-0.006,
        recompute=store_baseline(
            "betaxanthin", "pfba_glpk_growth", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: GLPK growth block kNN",
        stated=-0.004,
        recompute=store_baseline(
            "betaxanthin", "pfba_glpk_growth", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: GLPK growth block forest",
        stated=0.009,
        recompute=store_baseline(
            "betaxanthin", "pfba_glpk_growth", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: GLPK whole vector kNN",
        stated=0.097,
        recompute=store_baseline(
            "betaxanthin", "pfba_glpk_full", "knn", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: HiGHS store kNN",
        stated=0.195,
        recompute=store_baseline("betaxanthin", "pfba_highs", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: HiGHS store kNN top-50",
        stated=3.39,
        recompute=store_baseline(
            "betaxanthin", "pfba_highs", "knn", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="store-baselines: MOMA store ridge",
        stated=0.135,
        recompute=store_baseline("betaxanthin", "moma", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: flux statistics kNN",
        stated=0.145,
        recompute=store_baseline("betaxanthin", "flux_stats", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: ProtT5 kNN",
        stated=0.144,
        recompute=store_baseline("betaxanthin", "prot_T5_all", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: ProtT5 kNN top-50",
        stated=4.59,
        recompute=store_baseline(
            "betaxanthin", "prot_T5_all", "knn", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="store-baselines: GLPK store with ProtT5 forest top-50",
        stated=4.99,
        recompute=store_baseline(
            "betaxanthin", "pfba_glpk_full+prot_T5_all", "forest", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="store-baselines: capped driving-force concentrations ridge",
        stated=0.152,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_conc", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: capped driving-force concentrations ridge top-50",
        stated=3.39,
        recompute=store_baseline(
            "betaxanthin", "mdfcap100_conc", "ridge", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="store-baselines: sampled-cone mean ridge",
        stated=0.007,
        recompute=store_baseline(
            "betaxanthin", "sampled_mean", "ridge", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: sampled-cone mean kNN",
        stated=0.064,
        recompute=store_baseline("betaxanthin", "sampled_mean", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="store-baselines: sampled-cone mean forest",
        stated=0.102,
        recompute=store_baseline(
            "betaxanthin", "sampled_mean", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    # sec:physics -- Cooper's peaks through the perceptron, alone and beside the pool.
    Check(
        claim="cooper mlp: peaks alone 256",
        stated=0.131,
        recompute=cooper_mlp("cooper_h256", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: peaks alone top-50",
        stated=1.03,
        recompute=cooper_mlp("cooper_h256", "store_mlp", "single_mean_top50"),
        tolerance=0.006,
    ),
    Check(
        claim="cooper mlp: GO linear beside the peaks",
        stated=0.233,
        recompute=cooper_mlp(
            "cooper_h256", "go_linear_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: GO linear beside the pool at 256 (reference)",
        stated=0.260,
        recompute=pool_mlp(
            256,
            "go_linear_store_mlp",
            "single_mean_spearman",
            "go_models_betaxanthin_pool_twoslot_h256.json",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: two-slot pool minus two-slot peaks",
        stated=0.027,
        recompute=lambda: (
            pool_mlp(
                256,
                "go_linear_store_mlp",
                "single_mean_spearman",
                "go_models_betaxanthin_pool_twoslot_h256.json",
            )()
            - cooper_mlp("cooper_h256", "go_linear_store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool with peaks alone 256",
        stated=0.245,
        recompute=cooper_mlp("poolcooper_h256", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool alone 256 (reference)",
        stated=0.219,
        recompute=pool_mlp(256, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool with peaks minus pool alone",
        stated=0.026,
        recompute=lambda: (
            cooper_mlp("poolcooper_h256", "store_mlp", "single_mean_spearman")()
            - pool_mlp(256, "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool with peaks above pool alone on seeds",
        stated=3,
        recompute=cooper_mlp_paired_wins(
            "poolcooper_h256", "store_mlp", "pool_h256", "store_mlp"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="cooper mlp: pool alone sd 256",
        stated=0.042,
        recompute=pool_mlp(256, "store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool with peaks sd 256",
        stated=0.010,
        recompute=cooper_mlp("poolcooper_h256", "store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool alone at 1024 (reference)",
        stated=0.236,
        recompute=pool_mlp(1024, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: pool with peaks minus pool alone at 1024",
        stated=0.009,
        recompute=lambda: (
            cooper_mlp("poolcooper_h256", "store_mlp", "single_mean_spearman")()
            - pool_mlp(1024, "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: two slots over pool with peaks",
        stated=0.264,
        recompute=cooper_mlp(
            "poolcooper_h256", "go_linear_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: two slots gain from the peaks",
        stated=0.004,
        recompute=lambda: (
            cooper_mlp(
                "poolcooper_h256", "go_linear_store_mlp", "single_mean_spearman"
            )()
            - pool_mlp(
                256,
                "go_linear_store_mlp",
                "single_mean_spearman",
                "go_models_betaxanthin_pool_twoslot_h256.json",
            )()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: two slots with peaks above without on seeds",
        stated=5,
        recompute=cooper_mlp_paired_wins(
            "poolcooper_h256",
            "go_linear_store_mlp",
            "pool_twoslot_h256",
            "go_linear_store_mlp",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="cooper mlp: two slots with peaks ensemble",
        stated=0.265,
        recompute=cooper_mlp(
            "poolcooper_h256", "go_linear_store_mlp", "ensemble_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="cooper mlp: two slots without peaks ensemble",
        stated=0.266,
        recompute=pool_mlp(
            256,
            "go_linear_store_mlp",
            "ensemble_spearman",
            "go_models_betaxanthin_pool_twoslot_h256.json",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 204, the pool through 256 read standardized.
    Check(
        claim="round 204: pool mlp256 standardized arm",
        stated=0.182,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: arm sd",
        stated=0.017,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: log1p control",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: paired difference",
        stated=-0.097,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: paired sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: seeds above control",
        stated=0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 204: arm top-50",
        stated=3.99,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 204: control top-50",
        stated=4.69,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 204: ensemble",
        stated=0.187,
        recompute=ensemble("w221_bxdelaapoolmlp256std_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: seed min",
        stated=0.160,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: seed max",
        stated=0.207,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 204: select epoch min",
        stated=2,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 204: select epoch max",
        stated=8,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 204: select val pearson min",
        stated=0.22,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 204: select val pearson max",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 204: control select epoch min",
        stated=29,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "control",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 204: control select epoch max",
        stated=138,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 read standardized against log1p",
            "control",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    # sec:rounds -- round 203, dropout 0.3 on the pool through 256; the agreement of predictions.
    Check(
        claim="round 203: arm",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: arm sd",
        stated=0.018,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: control",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: control sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "sd_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: paired difference",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: paired sd",
        stated=0.028,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 203: arm top-50",
        stated=4.56,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 203: control top-50",
        stated=4.69,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 203: ensemble",
        stated=0.293,
        recompute=ensemble("w220_bxdelaapoolmlp256drop_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 203: ensemble top-50",
        stated=5.59,
        recompute=ensemble(
            "w220_bxdelaapoolmlp256drop_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 203: select epoch min",
        stated=32,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 203: select epoch max",
        stated=138,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 with transformer dropout 0.3 against without",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 203: seeds agreeing at 1.0000",
        stated=2,
        recompute=agreement("w220_vs_w206", "n_seeds_at_or_above_0.9999"),
        tolerance=0.5,
    ),
    Check(
        claim="round 203: min same-seed agreement",
        stated=0.91,
        recompute=agreement("w220_vs_w206", "min_same_seed_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim="round 203: max same-seed agreement among the live seeds",
        stated=0.98,
        recompute=agreement_max_below("w220_vs_w206", 0.9999),
        tolerance=0.006,
    ),
    Check(
        claim="round 203: seed-pair agreement",
        stated=0.91,
        recompute=agreement("w220_vs_w206", "mean_control_seed_pair_spearman"),
        tolerance=0.006,
    ),
    # sec:decomposition -- the backbone's head and the slot, separated.
    Check(
        claim="decomposition: rebuild matches reported (max over arms)",
        stated=0.0,
        recompute=lambda: max(
            decomposition_cells(n, "max_full_vs_reported")()
            for n in (
                "w206_aapoolmlp256",
                "w167_go",
                "w156_pfba",
                "w198_aapool_linear",
                "w199_aapoolmlp64",
                "w217_twoslot256",
                "w218_threefam_twoslot",
            )
        ),
        tolerance=0.001,
    ),
    Check(
        claim="decomposition: pool 256 constant heads",
        stated=2,
        recompute=decomposition_cells("w206_aapoolmlp256", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: pool 256 constant head sd ratio",
        stated=0.0,
        recompute=decomposition_cells("w206_aapoolmlp256", "max_sd_ratio_constant"),
        tolerance=1e-06,
    ),
    Check(
        claim="decomposition: pool 256 live head min",
        stated=0.05,
        recompute=decomposition_cells("w206_aapoolmlp256", "min_head"),
        tolerance=0.006,
    ),
    Check(
        claim="decomposition: pool 256 live head max",
        stated=0.15,
        recompute=decomposition_cells("w206_aapoolmlp256", "max_head"),
        tolerance=0.006,
    ),
    Check(
        claim="decomposition: pool 256 head vs slot min",
        stated=0.04,
        recompute=decomposition_cells("w206_aapoolmlp256", "min_head_vs_slot"),
        tolerance=0.006,
    ),
    Check(
        claim="decomposition: pool 256 head vs slot max",
        stated=0.14,
        recompute=decomposition_cells("w206_aapoolmlp256", "max_head_vs_slot"),
        tolerance=0.006,
    ),
    Check(
        claim="decomposition: pool 256 backbone adds min (live)",
        stated=-0.02,
        recompute=decomposition_cells("w206_aapoolmlp256", "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool 256 backbone adds max (live)",
        stated=0.055,
        recompute=decomposition_cells("w206_aapoolmlp256", "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool 256 slot alone",
        stated=0.257,
        recompute=decomposition("w206_aapoolmlp256", "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool 256 backbone adds mean",
        stated=0.021,
        recompute=decomposition("w206_aapoolmlp256", "mean_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool 256 full",
        stated=0.279,
        recompute=decomposition("w206_aapoolmlp256", "mean_full_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: GO affine constant heads",
        stated=6,
        recompute=decomposition_cells("w167_go", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: two-slot 256 constant heads",
        stated=6,
        recompute=decomposition_cells("w217_twoslot256", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: three families constant heads",
        stated=6,
        recompute=decomposition_cells("w218_threefam_twoslot", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: annotation arms select epoch min",
        stated=7,
        recompute=lambda: min(
            decomposition_cells(n, "min_select_epoch")()
            for n in ("w167_go", "w217_twoslot256", "w218_threefam_twoslot")
        ),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: annotation arms select epoch max",
        stated=15,
        recompute=lambda: max(
            decomposition_cells(n, "max_select_epoch")()
            for n in ("w167_go", "w217_twoslot256", "w218_threefam_twoslot")
        ),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: annotation arms read the slot within two ten-thousandths",
        stated=0.0,
        recompute=lambda: max(
            abs(decomposition(n, "mean_backbone_adds")())
            for n in ("w167_go", "w217_twoslot256", "w218_threefam_twoslot")
        ),
        tolerance=0.0002,
    ),
    Check(
        claim="decomposition: flux affine constant heads",
        stated=1,
        recompute=decomposition_cells("w156_pfba", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="decomposition: flux affine sd ratio",
        stated=1.0,
        recompute=decomposition("w156_pfba", "mean_sd_ratio_head_over_slot"),
        tolerance=0.15,
    ),
    Check(
        claim="decomposition: flux affine head vs slot",
        stated=0.13,
        recompute=decomposition("w156_pfba", "mean_spearman_head_vs_slot"),
        tolerance=0.006,
    ),
    Check(
        claim="decomposition: flux affine backbone adds",
        stated=0.023,
        recompute=decomposition("w156_pfba", "mean_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool linear backbone adds",
        stated=0.004,
        recompute=decomposition("w198_aapool_linear", "mean_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: pool 64 backbone adds",
        stated=-0.004,
        recompute=decomposition("w199_aapoolmlp64", "mean_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="decomposition: perceptron over the pool at 1024 (reference)",
        stated=0.236,
        recompute=pool_mlp(1024, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="graphs: one-block cells",
        stated=330,
        recompute=graph_reg_loss_over_cells(1, "n"),
        tolerance=0.5,
    ),
    Check(
        claim="graphs: one-block max graph_reg_loss",
        stated=0.0,
        recompute=graph_reg_loss_over_cells(1, "max"),
        tolerance=1e-09,
    ),
    Check(
        claim="graphs: two-block cells",
        stated=24,
        recompute=graph_reg_loss_over_cells(2, "n"),
        tolerance=0.5,
    ),
    # sec:rounds -- round 200, the two-slot head at 128.
    Check(
        claim="round 200: arm",
        stated=0.225,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: arm sd",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: pool alone 128",
        stated=0.262,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs pool 128 paired",
        stated=-0.037,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs pool 128 paired sd",
        stated=0.026,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs pool 128 seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 200: two slots at 64",
        stated=0.224,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 64 paired",
        stated=0.001,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 64 paired sd",
        stated=0.005,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 64 seeds above",
        stated=3,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 200: two slots at 256",
        stated=0.225,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 256 paired",
        stated=0.0,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 256 paired sd",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: vs 256 seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 200: arm top-50",
        stated=5.49,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: pool 128 top-50",
        stated=4.53,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: 64 top-50",
        stated=5.66,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 64",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: 256 top-50",
        stated=4.99,
        recompute=transfer(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the two slots at 256",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: ensemble",
        stated=0.225,
        recompute=ensemble("w219_bxdelgolinaapoolmlp128_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: ensemble top-50",
        stated=5.79,
        recompute=ensemble(
            "w219_bxdelgolinaapoolmlp128_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: seed min",
        stated=0.216,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: seed max",
        stated=0.231,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 200: select epoch min",
        stated=7,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 200: select epoch max",
        stated=10,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 200: select val pearson min",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 200: select val pearson max",
        stated=0.29,
        recompute=pair_cells(
            "betaxanthin: two-slot head, GO linear beside the pool through 128, against the pool alone through 128",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    # sec:coverage -- the pool's coverage indicator and the slot rebuilt on CPU.
    Check(
        claim="coverage: pinned genes with a target",
        stated=629,
        recompute=coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "n_pinned"),
        tolerance=0.5,
    ),
    Check(
        claim="coverage: covered genes",
        stated=556,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "n_covered"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="coverage: uncovered genes",
        stated=73,
        recompute=lambda: (
            coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "n_pinned")()
            - coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "n_covered")()
        ),
        tolerance=0.5,
    ),
    Check(
        claim="coverage: mean target covered",
        stated=0.1,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_target_covered"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="coverage: mean target uncovered",
        stated=-0.26,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_target_uncovered"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="coverage: indicator alone",
        stated=0.196,
        recompute=coverage(
            "aapool",
            "w206_bxdelaapoolmlp256_h12l1p1_e200",
            "coverage_indicator_spearman",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 256 all",
        stated=0.279,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_all"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 256 covered",
        stated=0.212,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 256 uncovered percentile",
        stated=0.05,
        recompute=coverage(
            "aapool",
            "w206_bxdelaapoolmlp256_h12l1p1_e200",
            "mean_uncovered_prediction_percentile",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="coverage: pool linear covered",
        stated=0.111,
        recompute=coverage(
            "aapool", "w198_bxdelaapool_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool linear all",
        stated=0.195,
        recompute=coverage(
            "aapool", "w198_bxdelaapool_h12l1p1_e200", "mean_spearman_all"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 64 covered",
        stated=0.156,
        recompute=coverage(
            "aapool", "w199_bxdelaapoolmlp_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 64 all",
        stated=0.24,
        recompute=coverage(
            "aapool", "w199_bxdelaapoolmlp_h12l1p1_e200", "mean_spearman_all"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: control all",
        stated=0.058,
        recompute=coverage("aapool", "w137_bxdel_h12l1p1_e200", "mean_spearman_all"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: control covered",
        stated=-0.013,
        recompute=coverage(
            "aapool", "w137_bxdel_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: GO affine all",
        stated=0.202,
        recompute=coverage("aapool", "w167_bxdelgo_h12l1p1_e200", "mean_spearman_all"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: GO affine covered",
        stated=0.14,
        recompute=coverage(
            "aapool", "w167_bxdelgo_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: pool 256 over GO on covered genes",
        stated=0.07,
        recompute=lambda: (
            coverage(
                "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_covered"
            )()
            - coverage("aapool", "w167_bxdelgo_h12l1p1_e200", "mean_spearman_covered")()
        ),
        tolerance=0.006,
    ),
    Check(
        claim="coverage: standardized covered",
        stated=0.189,
        recompute=coverage(
            "aapool", "w221_bxdelaapoolmlp256std_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: standardized all",
        stated=0.182,
        recompute=coverage(
            "aapool", "w221_bxdelaapoolmlp256std_h12l1p1_e200", "mean_spearman_all"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage: standardized uncovered percentile",
        stated=0.44,
        recompute=coverage(
            "aapool",
            "w221_bxdelaapoolmlp256std_h12l1p1_e200",
            "mean_uncovered_prediction_percentile",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="fit: trainer fit on standardized",
        stated=0.164,
        recompute=fit_variant("trainer", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: harness own on standardized (reference)",
        stated=0.219,
        recompute=pool_mlp(256, "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: minibatches alone",
        stated=0.216,
        recompute=fit_variant("batch32", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: CRPS alone",
        stated=0.214,
        recompute=fit_variant("crps", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: GELU alone",
        stated=0.169,
        recompute=fit_variant("gelu", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: weight decay alone",
        stated=0.219,
        recompute=fit_variant("wd", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: trainer fit on log1p, wild-type row",
        stated=0.179,
        recompute=fit_variant("trainer_raw", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: trainer fit on log1p, zero row",
        stated=0.268,
        recompute=fit_variant("zero_raw_trainer", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: replica all",
        stated=0.274,
        recompute=fit_variant("replica", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: replica sd",
        stated=0.006,
        recompute=fit_variant("replica", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: replica ensemble",
        stated=0.275,
        recompute=fit_variant("replica", "ensemble_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: replica covered",
        stated=0.216,
        recompute=fit_variant("replica", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="fit: replica parameters",
        stated=5890,
        recompute=lambda: float(
            __import__("json").load(
                open(osp.join(RESULTS, "go_models_betaxanthin_pool_fit_replica.json"))
            )["arms"][0]["seeds"][0]["n_parameters"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="fit: ladder forest over the pool (reference)",
        stated=0.28,
        recompute=store_baseline("betaxanthin", "aa_pool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:coverage -- the ladder's covered-only reads.
    Check(
        claim="coverage ladder: forest covered",
        stated=0.226,
        recompute=store_baseline_covered("betaxanthin", "aa_pool", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: forest all (reference)",
        stated=0.28,
        recompute=store_baseline("betaxanthin", "aa_pool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: knn covered",
        stated=0.239,
        recompute=store_baseline_covered("betaxanthin", "aa_pool", "knn"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: knn all (reference)",
        stated=0.285,
        recompute=store_baseline("betaxanthin", "aa_pool", "knn", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: ridge covered",
        stated=0.165,
        recompute=store_baseline_covered("betaxanthin", "aa_pool", "ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: ridge all (reference)",
        stated=0.149,
        recompute=store_baseline("betaxanthin", "aa_pool", "ridge", "mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="coverage ladder: covered genes",
        stated=556,
        recompute=lambda: float(
            __import__("json").load(
                open(osp.join(RESULTS, "store_baselines_betaxanthin.json"))
            )["representations"]["aa_pool"]["n_pinned_covered"]
        ),
        tolerance=0.5,
    ),
    # sec:coverage -- the perceptron arms on the covered genes.
    Check(
        claim="covered perceptrons: GO alone covered",
        stated=0.148,
        recompute=covered_read("gopool_cov_h1024", "go_mlp", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: GO alone all",
        stated=0.2,
        recompute=covered_read("gopool_cov_h1024", "go_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: pool alone covered",
        stated=0.207,
        recompute=covered_read("gopool_cov_h1024", "store_mlp", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: pool alone all",
        stated=0.236,
        recompute=covered_read("gopool_cov_h1024", "store_mlp", "single_mean_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: GO with pool one layer covered",
        stated=0.195,
        recompute=covered_read("gopool_cov_h1024", "go_store_mlp", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: GO with pool one layer all",
        stated=0.24,
        recompute=covered_read(
            "gopool_cov_h1024", "go_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: GO linear beside pool covered",
        stated=0.214,
        recompute=covered_read("gopool_cov_h1024", "go_linear_store_mlp", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: GO linear beside pool all",
        stated=0.265,
        recompute=covered_read(
            "gopool_cov_h1024", "go_linear_store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: two-slot gain on covered genes",
        stated=0.007,
        recompute=lambda: (
            covered_read("gopool_cov_h1024", "go_linear_store_mlp", "covered")()
            - covered_read("gopool_cov_h1024", "store_mlp", "covered")()
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: two-slot gain on all genes",
        stated=0.03,
        recompute=lambda: (
            covered_read(
                "gopool_cov_h1024", "go_linear_store_mlp", "single_mean_spearman"
            )()
            - covered_read("gopool_cov_h1024", "store_mlp", "single_mean_spearman")()
        ),
        tolerance=0.006,
    ),
    Check(
        claim="covered perceptrons: GO affine uncovered percentile",
        stated=0.32,
        recompute=coverage(
            "aapool",
            "w167_bxdelgo_h12l1p1_e200",
            "mean_uncovered_prediction_percentile",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="covered perceptrons: GO linear beside rebuilt slot covered",
        stated=0.185,
        recompute=covered_read(
            "pool_fit_replica_twoslot", "go_linear_store_mlp", "covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: Cooper substitute covered",
        stated=0.103,
        recompute=covered_read("cooper_fit_replica", "store_mlp", "covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: Cooper substitute all",
        stated=0.093,
        recompute=covered_read(
            "cooper_fit_replica", "store_mlp", "single_mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="covered perceptrons: Cooper substitute sd",
        stated=0.016,
        recompute=covered_read("cooper_fit_replica", "store_mlp", "single_sd_spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 201, the linear slot at a tenth of the rate.
    Check(
        claim="round 201: arm",
        stated=0.234,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: arm sd",
        stated=0.015,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: full rate",
        stated=0.225,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs full rate paired",
        stated=0.009,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs full rate paired sd",
        stated=0.013,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs full rate seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: pool alone",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs pool paired",
        stated=-0.045,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs pool paired sd",
        stated=0.024,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: vs pool seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: arm top-50",
        stated=5.56,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: full rate top-50",
        stated=4.99,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: pool top-50",
        stated=4.69,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against the pool alone through 256",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: ensemble",
        stated=0.239,
        recompute=ensemble("w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: ensemble top-50",
        stated=5.79,
        recompute=ensemble(
            "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: seed min",
        stated=0.215,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: seed max",
        stated=0.247,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: select epoch min",
        stated=55,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: select epoch max",
        stated=104,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: select val pearson min",
        stated=0.29,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: select val pearson max",
        stated=0.33,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: full rate select epoch min",
        stated=9,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "control",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: full rate select epoch max",
        stated=15,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a tenth of the rate against both slots at the full rate",
            "control",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 201: covered",
        stated=0.184,
        recompute=coverage(
            "w222",
            "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200",
            "mean_spearman_covered",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 201: uncovered percentile",
        stated=0.34,
        recompute=coverage(
            "w222",
            "w222_bxdelgolinaapoolmlp256lr01_h12l1p1_e200",
            "mean_uncovered_prediction_percentile",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 201: full rate covered (reference)",
        stated=0.171,
        recompute=coverage(
            "aapool",
            "w217_bxdelgolinaapoolmlp256_h12l1p1_e200",
            "mean_spearman_covered",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 202, the linear slot at a hundredth of the rate.
    Check(
        claim="round 202: arm",
        stated=0.211,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: arm sd",
        stated=0.027,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs full rate paired",
        stated=-0.014,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs full rate paired sd",
        stated=0.028,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs full rate seeds above",
        stated=2,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 202: tenth (reference)",
        stated=0.234,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against a tenth",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs tenth paired",
        stated=-0.023,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against a tenth",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs tenth paired sd",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against a tenth",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs tenth seeds above",
        stated=1,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against a tenth",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 202: vs pool paired",
        stated=-0.068,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against the pool alone through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs pool paired sd",
        stated=0.044,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against the pool alone through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: vs pool seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against the pool alone through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 202: arm top-50",
        stated=5.26,
        recompute=transfer(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 202: ensemble",
        stated=0.241,
        recompute=ensemble("w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: ensemble top-50",
        stated=5.39,
        recompute=ensemble(
            "w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 202: seed min",
        stated=0.18,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: seed max",
        stated=0.242,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 202: select epoch min",
        stated=34,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 202: select epoch max",
        stated=197,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 202: select val pearson min",
        stated=0.3,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 202: select val pearson max",
        stated=0.34,
        recompute=pair_cells(
            "betaxanthin: two-slot head at 256 with the linear slot at a hundredth of the rate against both slots at the full rate",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 202: covered",
        stated=0.166,
        recompute=coverage(
            "w223",
            "w223_bxdelgolinaapoolmlp256lr001_h12l1p1_e200",
            "mean_spearman_covered",
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 208, twelve seeds of the three families with two slots.
    Check(
        claim="round 208: second six",
        stated=0.236,
        recompute=pair_seed_subset(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            (2706, 2707, 2708, 2709, 2710, 2711),
            "mean",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: second six sd",
        stated=0.005,
        recompute=pair_seed_subset(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            (2706, 2707, 2708, 2709, 2710, 2711),
            "sd",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: first six",
        stated=0.239,
        recompute=pair_seed_subset(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            (2700, 2701, 2702, 2703, 2704, 2705),
            "mean",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: twelve",
        stated=0.237,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: twelve sd",
        stated=0.004,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: twelve se",
        stated=0.001,
        recompute=pair_seed_subset(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            tuple(range(2700, 2712)),
            "se",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: n",
        stated=12,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "n",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: seed min",
        stated=0.23,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: seed max",
        stated=0.243,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: select epoch min",
        stated=6,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: select epoch max",
        stated=14,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: select val pearson min",
        stated=0.27,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 208: select val pearson max",
        stated=0.29,
        recompute=pair_cells(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 208: twelve-seed ensemble",
        stated=0.24,
        recompute=ensemble(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200", "spearman", "all_seeds"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: twelve-seed ensemble top-50",
        stated=4.99,
        recompute=ensemble(
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
            "top50_enrichment",
            "all_seeds",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 208: over GO",
        stated=0.035,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO alone, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: over GO seeds",
        stated=12,
        recompute=transfer(
            "betaxanthin: three families with two slots against GO alone, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: over GO with HiGHS",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: three families with two slots against the HiGHS store with GO, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: over GO with HiGHS seeds",
        stated=12,
        recompute=transfer(
            "betaxanthin: three families with two slots against the HiGHS store with GO, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: under the pool",
        stated=-0.041,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: under the pool seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: three families with two slots against the pool alone through 256, twelve seeds",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 208: covered twelve",
        stated=0.186,
        recompute=coverage(
            "w218_twelve",
            "w218_bxdelgopfbamomalinaapoolmlp256_h12l1p1_e200",
            "mean_spearman_covered",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: pool covered twelve",
        stated=0.206,
        recompute=coverage(
            "w218_twelve",
            "w206_bxdelaapoolmlp256_h12l1p1_e200",
            "mean_spearman_covered",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 208: GO covered twelve",
        stated=0.142,
        recompute=coverage(
            "w218_twelve", "w167_bxdelgo_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 210, the pool slot under the release rule.
    Check(
        claim="round 210: released seeds",
        stated=5,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "n_released"),
        tolerance=0.5,
    ),
    Check(
        claim="round 210: release epoch",
        stated=5,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "epoch"),
        tolerance=0.5,
    ),
    Check(
        claim="round 210: released mean",
        stated=0.255,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: released sd",
        stated=0.027,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: unreleased seed",
        stated=0.005,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "unreleased"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: six-seed arm",
        stated=0.213,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: six-seed sd",
        stated=0.105,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: control",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: ensemble",
        stated=0.251,
        recompute=ensemble("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 210: released select epoch min",
        stated=30,
        recompute=release("w229_bxdelaapoolmlp256rel_h12l1p1_e200", "select_epoch_min"),
        tolerance=0.5,
    ),
    Check(
        claim="round 210: released select epoch max",
        stated=138,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 210: released val pearson max",
        stated=0.42,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 210: control val pearson min",
        stated=0.36,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "control",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 210: control val pearson max",
        stated=0.4,
        recompute=pair_cells(
            "betaxanthin: measured amino-acid pool through a hidden layer of 256 under the release rule (the slot frozen until epoch 5 and validation Pearson 0.15) against the slot from epoch 0",
            "control",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 210: covered",
        stated=0.154,
        recompute=coverage(
            "w229", "w229_bxdelaapoolmlp256rel_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 211, Cooper's peaks substituted for the pool.
    Check(
        claim="round 211: arm",
        stated=0.111,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: arm sd",
        stated=0.025,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: pool",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs pool paired",
        stated=-0.168,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs pool paired sd",
        stated=0.022,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs pool seeds above",
        stated=0,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 211: control",
        stated=0.058,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs control paired",
        stated=0.052,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs control paired sd",
        stated=0.061,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: vs control seeds above",
        stated=6,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 211: arm top-50",
        stated=2.73,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 211: control top-50",
        stated=3.39,
        recompute=transfer(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the control",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 211: ensemble",
        stated=0.139,
        recompute=ensemble("w230_bxdelcooperpoolmlp256_h12l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: seed min",
        stated=0.088,
        recompute=pair_cells(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: seed max",
        stated=0.155,
        recompute=pair_cells(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: select epoch min",
        stated=44,
        recompute=pair_cells(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 211: select epoch max",
        stated=137,
        recompute=pair_cells(
            "betaxanthin: Cooper's seventeen peaks through a hidden layer of 256 against the Mulleder pool through 256",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 211: Cooper covers the pin",
        stated=554,
        recompute=coverage(
            "w230", "w230_bxdelcooperpoolmlp256_h12l1p1_e200", "n_covered"
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 211: Cooper indicator here",
        stated=-0.051,
        recompute=coverage(
            "w230",
            "w230_bxdelcooperpoolmlp256_h12l1p1_e200",
            "coverage_indicator_spearman",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: covered",
        stated=0.111,
        recompute=coverage(
            "w230", "w230_bxdelcooperpoolmlp256_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: pool on the same genes",
        stated=0.287,
        recompute=coverage(
            "w230", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: forest over Cooper (ladder)",
        stated=0.113,
        recompute=store_baseline(
            "betaxanthin", "cooper_pool", "forest", "mean_spearman"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 211: forest over the pool (ladder)",
        stated=0.28,
        recompute=store_baseline("betaxanthin", "aa_pool", "forest", "mean_spearman"),
        tolerance=0.0006,
    ),
    # sec:rounds -- round 212, the width control for the layer-pattern ladder.
    Check(
        claim="round 212: arm",
        stated=0.291,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "mean_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: arm sd",
        stated=0.023,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "sd_spearman",
            "arm",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: width 12",
        stated=0.279,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "mean_spearman",
            "control",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: paired",
        stated=0.012,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "mean_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: paired sd",
        stated=0.032,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "sd_difference",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: seeds above",
        stated=4,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "n_positive",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 212: arm top-50",
        stated=5.26,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "mean_top50_enrichment",
            "arm",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 212: width 12 top-50",
        stated=4.69,
        recompute=transfer(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "mean_top50_enrichment",
            "control",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 212: ensemble",
        stated=0.309,
        recompute=ensemble("w231_bxdelaapoolmlp256w36_h36l1p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: ensemble top-50",
        stated=5.59,
        recompute=ensemble(
            "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "top50_enrichment"
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 212: seed min",
        stated=0.261,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "spearman",
            "min",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: seed max",
        stated=0.31,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "spearman",
            "max",
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: select epoch min",
        stated=24,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "select_epoch",
            "min",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 212: select epoch max",
        stated=95,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "select_epoch",
            "max",
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 212: select val pearson min",
        stated=0.37,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "select_val_pearson",
            "min",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 212: select val pearson max",
        stated=0.39,
        recompute=pair_cells(
            "betaxanthin: pool through 256 on a one-block backbone of width 36 with nine heads against width 12 with four",
            "arm",
            "select_val_pearson",
            "max",
        ),
        tolerance=0.006,
    ),
    Check(
        claim="round 212: covered",
        stated=0.214,
        recompute=coverage(
            "w231", "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "mean_spearman_covered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: uncovered",
        stated=0.142,
        recompute=coverage(
            "w231", "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "mean_spearman_uncovered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: width 12 uncovered (reference)",
        stated=0.095,
        recompute=coverage(
            "aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_uncovered"
        ),
        tolerance=0.0006,
    ),
    Check(
        claim="round 212: parameters",
        stated=84907,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(
                        RESULTS, "w231_bxdelaapoolmlp256w36_h36l1p1_e200_s2700.json"
                    )
                )
            )["n_parameters"]["total"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 212: width 12 parameters",
        stated=24283,
        recompute=lambda: float(
            json.load(
                open(
                    osp.join(RESULTS, "w206_bxdelaapoolmlp256_h12l1p1_e200_s2700.json")
                )
            )["n_parameters"]["total"]
        ),
        tolerance=0.5,
    ),
    Check(
        claim="round 212: graph_reg_loss is zero",
        stated=0.0,
        recompute=lambda: max(
            x["graph_reg_loss"]
            for x in json.load(
                open(
                    osp.join(
                        RESULTS, "w231_bxdelaapoolmlp256w36_h36l1p1_e200_s2700.json"
                    )
                )
            )["history"]
        ),
        tolerance=1e-09,
    ),
    Check(
        claim="homophily: labeled genes",
        stated=3764,
        recompute=homophily("betaxanthin", "physical", 1, "n_labeled"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: pinned genes",
        stated=629,
        recompute=homophily("betaxanthin", "physical", 1, "n_pinned"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: coexpression neighbors",
        stated=174,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "mean_labeled_neighbors"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: coexpression rho",
        stated=0.244,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: coexpression null mean",
        stated=-0.005,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "null_mean"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: coexpression null sd",
        stated=0.041,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "null_sd"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: coexpression z",
        stated=6.1,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "z"),
        tolerance=0.06,
    ),
    Check(
        claim="homophily: coexpression top-50",
        stated=4.59,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="homophily: coexpression on pool-covered",
        stated=0.192,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "spearman_store_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: coexpression LOO",
        stated=0.208,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "labeled_loo_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: coexpression LOO n",
        stated=3721,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 1, "n_labeled_covered"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: coexpression two hops",
        stated=0.007,
        recompute=homophily("betaxanthin", "string12_0_coexpression", 2, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: union rho",
        stated=0.221,
        recompute=homophily("betaxanthin", "union_of_9", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: union top-50",
        stated=5.19,
        recompute=homophily("betaxanthin", "union_of_9", 1, "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="homophily: union on pool-covered",
        stated=0.189,
        recompute=homophily("betaxanthin", "union_of_9", 1, "spearman_store_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: union LOO",
        stated=0.169,
        recompute=homophily("betaxanthin", "union_of_9", 1, "labeled_loo_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: experimental rho",
        stated=0.166,
        recompute=homophily("betaxanthin", "string12_0_experimental", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: experimental z",
        stated=3.1,
        recompute=homophily("betaxanthin", "string12_0_experimental", 1, "z"),
        tolerance=0.06,
    ),
    Check(
        claim="homophily: neighborhood rho",
        stated=0.195,
        recompute=homophily("betaxanthin", "string12_0_neighborhood", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: neighborhood covered",
        stated=458,
        recompute=homophily("betaxanthin", "string12_0_neighborhood", 1, "n_pinned_covered"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: neighborhood z",
        stated=4.3,
        recompute=homophily("betaxanthin", "string12_0_neighborhood", 1, "z"),
        tolerance=0.06,
    ),
    Check(
        claim="homophily: physical rho",
        stated=0.104,
        recompute=homophily("betaxanthin", "physical", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: physical neighbors",
        stated=14,
        recompute=homophily("betaxanthin", "physical", 1, "mean_labeled_neighbors"),
        tolerance=0.5,
    ),
    Check(
        claim="homophily: physical decile fold",
        stated=1.61,
        recompute=homophily("betaxanthin", "physical", 1, "top_decile_fold"),
        tolerance=0.006,
    ),
    Check(
        claim="homophily: physical decile p",
        stated=2.7e-05,
        recompute=homophily("betaxanthin", "physical", 1, "top_decile_fisher_p"),
        tolerance=6e-07,
    ),
    Check(
        claim="homophily: fusion rho",
        stated=-0.06,
        recompute=homophily("betaxanthin", "string12_0_fusion", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: tflink rho",
        stated=0.06,
        recompute=homophily("betaxanthin", "tflink", 1, "spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="homophily: pool-covered pinned genes",
        stated=556,
        recompute=homophily("betaxanthin", "physical", 1, "n_store_covered"),
        tolerance=0.5,
    ),
    Check(
        claim="graph reg: w233 penalty over data at epoch 0",
        stated=1.74,
        recompute=pack_log("w233_bxdelaapoolmlp256skl_h36l2p1_e200", 2700, 0, "ratio"),
        tolerance=0.006,
    ),
    Check(
        claim="graph reg: w233 penalty over data at epoch 40",
        stated=0.77,
        recompute=pack_log("w233_bxdelaapoolmlp256skl_h36l2p1_e200", 2700, 40, "ratio"),
        tolerance=0.006,
    ),
    Check(
        claim="graph reg: w233 loss at epoch 40",
        stated=0.849,
        recompute=pack_log("w233_bxdelaapoolmlp256skl_h36l2p1_e200", 2700, 40, "loss"),
        tolerance=0.0006,
    ),
    Check(
        claim="graph reg: w233 data term at epoch 40",
        stated=0.479,
        recompute=pack_log("w233_bxdelaapoolmlp256skl_h36l2p1_e200", 2700, 40, "data"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: graph label ridge",
        stated=0.219,
        recompute=ladder_read("graph_label", "ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: graph label knn",
        stated=0.224,
        recompute=ladder_read("graph_label", "knn"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: graph label forest",
        stated=0.254,
        recompute=ladder_read("graph_label", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: graph label ridge top-50",
        stated=4.99,
        recompute=ladder_read("graph_label", "ridge", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: graph label knn top-50",
        stated=5.39,
        recompute=ladder_read("graph_label", "knn", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: graph label forest top-50",
        stated=5.79,
        recompute=ladder_read("graph_label", "forest", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="ladder: graph label is the highest ridge read",
        stated=1,
        recompute=ladder_rank("ridge", "mean_spearman", "graph_label"),
        tolerance=0.5,
    ),
    Check(
        claim="ladder: graph label is the highest knn call",
        stated=1,
        recompute=ladder_rank("knn", "top50_enrichment", "graph_label"),
        tolerance=0.5,
    ),
    Check(
        claim="ladder: graph label ridge lead over the next",
        stated=0.06,
        recompute=lambda: ladder_read("graph_label", "ridge")() - sorted((o["ridge"]["selected_on_val"]["test_pinned"]["mean_spearman"] for o in json.load(open(osp.join(RESULTS, "store_baselines_betaxanthin.json")))["representations"].values()), reverse=True)[1],
        tolerance=0.006,
    ),
    Check(
        claim="ladder: pool with graph label forest",
        stated=0.284,
        recompute=ladder_read("aapool_graphlabel", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with graph label forest covered",
        stated=0.234,
        recompute=ladder_read("aapool_graphlabel", "forest", covered=True),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool forest",
        stated=0.28,
        recompute=ladder_read("aa_pool", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool forest covered",
        stated=0.226,
        recompute=ladder_read("aa_pool", "forest", covered=True),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with graph label is the highest forest read",
        stated=1,
        recompute=ladder_rank("forest", "mean_spearman", "aapool_graphlabel"),
        tolerance=0.5,
    ),
    Check(
        claim="ladder: pool with graph label knn",
        stated=0.258,
        recompute=ladder_read("aapool_graphlabel", "knn"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool knn",
        stated=0.285,
        recompute=ladder_read("aa_pool", "knn"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool with graph label ridge",
        stated=0.143,
        recompute=ladder_read("aapool_graphlabel", "ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="ladder: pool ridge",
        stated=0.149,
        recompute=ladder_read("aa_pool", "ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: arm",
        stated=0.27,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "mean_spearman", "arm"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: arm sd",
        stated=0.023,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "sd_spearman", "arm"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: one block",
        stated=0.291,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "mean_spearman", "control"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: paired",
        stated=-0.021,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "mean_difference"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: paired sd",
        stated=0.036,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "sd_difference"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: seeds above",
        stated=3,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round 213: arm top-50",
        stated=4.13,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "mean_top50_enrichment", "arm"),
        tolerance=0.006,
    ),
    Check(
        claim="round 213: one block top-50",
        stated=5.26,
        recompute=transfer("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "mean_top50_enrichment", "control"),
        tolerance=0.006,
    ),
    Check(
        claim="round 213: ensemble",
        stated=0.282,
        recompute=ensemble("w232_bxdelaapoolmlp256ss_h36l2p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: ensemble top-50",
        stated=4.19,
        recompute=ensemble("w232_bxdelaapoolmlp256ss_h36l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 213: seed min",
        stated=0.235,
        recompute=pair_cells("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "arm", "spearman", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: seed max",
        stated=0.306,
        recompute=pair_cells("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "arm", "spearman", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: select epoch min",
        stated=34,
        recompute=pair_cells("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "arm", "select_epoch", "min"),
        tolerance=0.5,
    ),
    Check(
        claim="round 213: select epoch max",
        stated=194,
        recompute=pair_cells("betaxanthin: pool through 256 on two unmasked blocks of width 36 against one block of width 36", "arm", "select_epoch", "max"),
        tolerance=0.5,
    ),
    Check(
        claim="round 213: covered",
        stated=0.207,
        recompute=coverage("aapool", "w232_bxdelaapoolmlp256ss_h36l2p1_e200", "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: uncovered",
        stated=-0.015,
        recompute=coverage("aapool", "w232_bxdelaapoolmlp256ss_h36l2p1_e200", "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: one block covered (reference)",
        stated=0.214,
        recompute=coverage("w231", "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 213: parameters",
        stated=100927,
        recompute=lambda: float(json.load(open(osp.join(RESULTS, "w232_bxdelaapoolmlp256ss_h36l2p1_e200_s2700.json")))["n_parameters"]["total"]),
        tolerance=0.5,
    ),
    Check(
        claim="round 213: graph_reg_loss is zero",
        stated=0.0,
        recompute=lambda: max(x["graph_reg_loss"] for x in json.load(open(osp.join(RESULTS, "w232_bxdelaapoolmlp256ss_h36l2p1_e200_s2700.json")))["history"]),
        tolerance=1e-09,
    ),
    Check(
        claim="round 214: arm",
        stated=0.228,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "mean_spearman", "arm"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: arm sd",
        stated=0.048,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "sd_spearman", "arm"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: unmasked",
        stated=0.27,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "mean_spearman", "control"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: paired",
        stated=-0.042,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "mean_difference"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: paired sd",
        stated=0.044,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "sd_difference"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: seeds above",
        stated=1,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "n_positive"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: arm top-50",
        stated=3.66,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "mean_top50_enrichment", "arm"),
        tolerance=0.006,
    ),
    Check(
        claim="round 214: unmasked top-50",
        stated=4.13,
        recompute=transfer("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "mean_top50_enrichment", "control"),
        tolerance=0.006,
    ),
    Check(
        claim="round 214: ensemble",
        stated=0.253,
        recompute=ensemble("w234_bxdelaapoolmlp256smask_h36l2p1_e200", "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: ensemble top-50",
        stated=5.79,
        recompute=ensemble("w234_bxdelaapoolmlp256smask_h36l2p1_e200", "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim="round 214: seed min",
        stated=0.159,
        recompute=pair_cells("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "arm", "spearman", "min"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: seed max",
        stated=0.277,
        recompute=pair_cells("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "arm", "spearman", "max"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: select epoch min",
        stated=21,
        recompute=pair_cells("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "arm", "select_epoch", "min"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: select epoch max",
        stated=194,
        recompute=pair_cells("betaxanthin: pool through 256 on two blocks of width 36 with the hard graph mask on the second block against two unmasked blocks", "arm", "select_epoch", "max"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: covered",
        stated=0.161,
        recompute=coverage("aapool", "w234_bxdelaapoolmlp256smask_h36l2p1_e200", "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: uncovered",
        stated=0.049,
        recompute=coverage("aapool", "w234_bxdelaapoolmlp256smask_h36l2p1_e200", "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: unmasked constant heads",
        stated=4,
        recompute=decomposition_cells("w232", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: unmasked live head adds min",
        stated=-0.016,
        recompute=decomposition_cells("w232", "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: unmasked live head adds max",
        stated=0.035,
        recompute=decomposition_cells("w232", "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked constant heads",
        stated=1,
        recompute=decomposition_cells("w234", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: masked head costs most",
        stated=-0.105,
        recompute=decomposition_cells("w234", "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked head second cost",
        stated=-0.081,
        recompute=lambda: sorted(c["backbone_adds"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w234.json")))["cells"])[1],
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked head own read max",
        stated=0.071,
        recompute=decomposition_cells("w234", "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked head own read min",
        stated=-0.058,
        recompute=decomposition_cells("w234", "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked head own read on seed 2701",
        stated=-0.004,
        recompute=lambda: float(next(c for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w234.json")))["cells"] if c["seed"] == 2701)["backbone_head"]["spearman_vs_target"]),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: masked slot alone",
        stated=0.261,
        recompute=decomposition("w234", "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: unmasked slot alone",
        stated=0.267,
        recompute=decomposition("w232", "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: seeds agreeing at 1.0000",
        stated=1,
        recompute=agreement("w234_vs_w232", "n_seeds_at_or_above_0.9999"),
        tolerance=0.5,
    ),
    Check(
        claim="round 214: the constant seed reads alike",
        stated=0.277,
        recompute=lambda: float(next(r for r in json.load(open(osp.join(RESULTS, "target_prediction_agreement_w234_vs_w232.json")))["seeds"] if r["seed"] == 2704)["arm_spearman_vs_target"]),
        tolerance=0.0006,
    ),
    Check(
        claim="round 214: rebuilt reads match reported (w234)",
        stated=0.0,
        recompute=decomposition_cells("w234", "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim="round 214: rebuilt reads match reported (w232)",
        stated=0.0,
        recompute=decomposition_cells("w232", "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim='round 215: arm',
        stated=0.226,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: arm sd',
        stated=0.023,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: n seeds',
        stated=6,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'n'),
        tolerance=0.5,
    ),
    Check(
        claim='round 215: control',
        stated=0.27,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: paired',
        stated=-0.044,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: paired sd',
        stated=0.033,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: seeds above',
        stated=0,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 215: arm top-50',
        stated=4.16,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: control top-50',
        stated=4.13,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: ensemble',
        stated=0.241,
        recompute=ensemble('w233_bxdelaapoolmlp256skl_h36l2p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: ensemble top-50',
        stated=4.79,
        recompute=ensemble('w233_bxdelaapoolmlp256skl_h36l2p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: seed min',
        stated=0.201,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: seed max',
        stated=0.263,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: select epoch min',
        stated=12,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 215: select epoch max',
        stated=102,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 215: val pearson min',
        stated=0.33,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: val pearson max',
        stated=0.4,
        recompute=pair_cells('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: reg loss at selection min',
        stated=0.34,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: reg loss at selection max',
        stated=0.44,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: data term at selection min',
        stated=0.46,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: data term at selection max',
        stated=0.5,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w233_bxdelaapoolmlp256skl_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: covered',
        stated=0.152,
        recompute=coverage("w238w239", 'w233_bxdelaapoolmlp256skl_h36l2p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: uncovered',
        stated=0.031,
        recompute=coverage("w238w239", 'w233_bxdelaapoolmlp256skl_h36l2p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: agreement min',
        stated=0.66,
        recompute=agreement("w233_vs_w232", "min_same_seed_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: agreement max',
        stated=0.81,
        recompute=lambda: max(r["spearman"] for r in json.load(open(osp.join(RESULTS, "target_prediction_agreement_w233_vs_w232.json")))["seeds"]),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: control seed-pair agreement min',
        stated=0.88,
        recompute=agreement("w233_vs_w232", "min_control_seed_pair_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: control seed-pair agreement max',
        stated=0.99,
        recompute=agreement("w233_vs_w232", "max_control_seed_pair_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 215: constant heads',
        stated=0,
        recompute=decomposition_cells("w233", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim='round 215: head spread ratio min',
        stated=1.0,
        recompute=lambda: min(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w233.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 215: head spread ratio max',
        stated=2.2,
        recompute=lambda: max(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w233.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 215: head own read min',
        stated=0.045,
        recompute=decomposition_cells("w233", "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: head own read max',
        stated=0.13,
        recompute=decomposition_cells("w233", "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: head subtracts least',
        stated=-0.002,
        recompute=decomposition_cells("w233", "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: head subtracts most',
        stated=-0.046,
        recompute=decomposition_cells("w233", "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: slot alone',
        stated=0.25,
        recompute=decomposition("w233", "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: rebuilt reads match reported',
        stated=0.0,
        recompute=decomposition_cells("w233", "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim='round 215: unmasked live head min (reference)',
        stated=-0.1025,
        recompute=decomposition_cells("w232", "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 215: unmasked live head max (reference)',
        stated=0.0875,
        recompute=decomposition_cells("w232", "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: arm',
        stated=0.182,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: arm sd',
        stated=0.041,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: pool control',
        stated=0.279,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: paired vs pool',
        stated=-0.097,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: paired sd vs pool',
        stated=0.041,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: seeds above pool',
        stated=0,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 219: affine-free control',
        stated=0.058,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the control", 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: paired vs control',
        stated=0.124,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the control", 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: paired sd vs control',
        stated=0.052,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the control", 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: seeds above control',
        stated=6,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the control", 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 219: arm top-50',
        stated=4.86,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: pool top-50',
        stated=4.69,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: control top-50',
        stated=3.39,
        recompute=transfer("betaxanthin: the graphs' one-hop label-propagation store through 256 against the control", 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: ensemble',
        stated=0.213,
        recompute=ensemble('w238_bxdelgraphlabelmlp256_h12l1p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: ensemble top-50',
        stated=5.79,
        recompute=ensemble('w238_bxdelgraphlabelmlp256_h12l1p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: seed min',
        stated=0.125,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: seed max',
        stated=0.227,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: select epoch min',
        stated=44,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 219: select epoch max',
        stated=197,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 219: val pearson min',
        stated=0.25,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: val pearson max',
        stated=0.28,
        recompute=pair_cells("betaxanthin: the graphs' one-hop label-propagation store through 256 against the pool through 256", 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 219: covered',
        stated=0.12,
        recompute=coverage("w238w239", 'w238_bxdelgraphlabelmlp256_h12l1p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: uncovered',
        stated=0.287,
        recompute=coverage("w238w239", 'w238_bxdelgraphlabelmlp256_h12l1p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: pool covered (reference)',
        stated=0.212,
        recompute=coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: pool uncovered (reference)',
        stated=0.095,
        recompute=coverage("aapool", "w206_bxdelaapoolmlp256_h12l1p1_e200", "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: n covered',
        stated=556,
        recompute=coverage("w238w239", 'w238_bxdelgraphlabelmlp256_h12l1p1_e200', "n_covered"),
        tolerance=0.5,
    ),
    Check(
        claim='round 219: ladder forest on the label store',
        stated=0.254,
        recompute=ladder_read("graph_label", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 219: ladder ridge on the label store',
        stated=0.219,
        recompute=ladder_read("graph_label", "ridge"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: arm',
        stated=0.288,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: arm sd',
        stated=0.013,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: paired vs pool',
        stated=0.009,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: paired sd vs pool',
        stated=0.017,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: seeds above pool',
        stated=4,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 220: paired vs labels',
        stated=0.106,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the label-propagation store alone', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: paired sd vs labels',
        stated=0.044,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the label-propagation store alone', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: seeds above labels',
        stated=6,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the label-propagation store alone', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 220: width-36 control',
        stated=0.291,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against one block of width 36', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: paired vs width 36',
        stated=-0.003,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against one block of width 36', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: paired sd vs width 36',
        stated=0.021,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against one block of width 36', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: seeds above width 36',
        stated=3,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against one block of width 36', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 220: arm top-50',
        stated=4.99,
        recompute=transfer('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 220: ensemble',
        stated=0.318,
        recompute=ensemble('w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: ensemble top-50',
        stated=5.59,
        recompute=ensemble('w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 220: seed min',
        stated=0.273,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: seed max',
        stated=0.31,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: select epoch min',
        stated=33,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 220: select epoch max',
        stated=192,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 220: val pearson min',
        stated=0.36,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 220: val pearson max',
        stated=0.42,
        recompute=pair_cells('betaxanthin: the pool with the label-propagation store appended, through 256, against the pool alone', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 220: covered',
        stated=0.219,
        recompute=coverage("w238w239", 'w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: uncovered',
        stated=0.087,
        recompute=coverage("w238w239", 'w239_bxdelaapoolgraphlabelmlp256_h12l1p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: ladder forest pool with labels',
        stated=0.284,
        recompute=ladder_read("aapool_graphlabel", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: ladder forest pool alone',
        stated=0.28,
        recompute=ladder_read("aa_pool", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: ladder forest pool with labels covered',
        stated=0.234,
        recompute=ladder_read("aapool_graphlabel", "forest", covered=True),
        tolerance=0.0006,
    ),
    Check(
        claim='round 220: ladder forest pool alone covered',
        stated=0.226,
        recompute=ladder_read("aa_pool", "forest", covered=True),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: arm',
        stated=0.16,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: arm sd',
        stated=0.022,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: YPD store control',
        stated=0.13,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: paired',
        stated=0.03,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: paired sd',
        stated=0.04,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: seeds above',
        stated=5,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 221: arm top-50',
        stated=4.23,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 221: control top-50',
        stated=4.46,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 221: ensemble',
        stated=0.188,
        recompute=ensemble('w240_bxdelpfba4m_h12l1p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: seed min',
        stated=0.12,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: seed max',
        stated=0.182,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 221: select epoch min',
        stated=40,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 221: select epoch max',
        stated=191,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 221: val pearson min',
        stated=0.03,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 221: val pearson max',
        stated=0.3,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store with four media growth ratios (control) against YPD alone', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 221: GEM genes scanned',
        stated=1161,
        recompute=lambda: float(json.load(open(osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "results", "fba_ratio_all_gem_genes.json")))["n_genes"]),
        tolerance=0.5,
    ),
    Check(
        claim='round 222: arm',
        stated=0.142,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: arm sd',
        stated=0.036,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: paired vs zero init',
        stated=0.012,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: paired sd vs zero init',
        stated=0.024,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: seeds above zero init',
        stated=5,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 222: paired vs four media',
        stated=-0.019,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against the four-media control', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: paired sd vs four media',
        stated=0.047,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against the four-media control', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: seeds above four media',
        stated=1,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against the four-media control', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 222: arm top-50',
        stated=4.49,
        recompute=transfer('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 222: ensemble',
        stated=0.163,
        recompute=ensemble('w241_bxdelpfbadef_h12l1p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: seed min',
        stated=0.08,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: seed max',
        stated=0.178,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 222: select epoch min',
        stated=2,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 222: select epoch max',
        stated=114,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 222: val pearson min',
        stated=0.02,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 222: val pearson max',
        stated=0.27,
        recompute=pair_cells('betaxanthin: the HiGHS parsimonious store under default affine initialization against zero init', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 222: GO alone (reference)',
        stated=0.203,
        recompute=seed_level("w167_bxdelgo_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: penalty over data at 019 weight',
        stated=0.8,
        recompute=reg_ratio("w233_bxdelaapoolmlp256skl_h36l2p1_e200"),
        tolerance=0.06,
    ),
    Check(
        claim='physics ladder: penalty over data at ten times',
        stated=6.5,
        recompute=reg_ratio("w236_bxdelaapoolmlp256skl10_h36l2p1_e200"),
        tolerance=0.6,
    ),
    Check(
        claim='physics ladder: penalty over data at a hundred times',
        stated=60,
        recompute=reg_ratio("w237_bxdelaapoolmlp256skl100_h36l2p1_e200"),
        tolerance=2.0,
    ),
    Check(
        claim='physics ladder: read at 019 weight (six seeds)',
        stated=0.226,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: read at ten times',
        stated=0.236,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: read at a hundred times',
        stated=0.243,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: unregularized two blocks',
        stated=0.27,
        recompute=transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: seeds above the control over the three weights',
        stated=1,
        recompute=lambda: transfer('betaxanthin: pool through 256 on two blocks of width 36 with the KL regularizer on the second block against two unmasked blocks', "n_positive")() + transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', "n_positive")() + transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', "n_positive")(),
        tolerance=0.5,
    ),
    Check(
        claim='physics ladder: hundred times vs 019 weight',
        stated=0.017,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='physics ladder: hundred times vs no prior',
        stated=-0.027,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: uncovered percentile min over pool arms',
        stated=0.05,
        recompute=uncovered_percentile_range("min"),
        tolerance=0.006,
    ),
    Check(
        claim='standing: uncovered percentile max over pool arms',
        stated=0.11,
        recompute=uncovered_percentile_range("max"),
        tolerance=0.006,
    ),
    Check(
        claim='standing: width-36 covered',
        stated=0.214,
        recompute=coverage("w238w239", "w231_bxdelaapoolmlp256w36_h36l1p1_e200", "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: three families twelve seeds',
        stated=0.217,
        recompute=seed_level("w190_bxdelgopfbamoma_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: ladder forest over GO with HiGHS',
        stated=0.204,
        recompute=ladder_read("gopfba", "forest"),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: ladder nearest neighbors over the pool',
        stated=0.285,
        recompute=ladder_read("aa_pool", "knn"),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: pool through 256 twelve seeds',
        stated=0.278,
        recompute=seed_level("w206_bxdelaapoolmlp256_h12l1p1_e200", "all12"),
        tolerance=0.0006,
    ),
    Check(
        claim='standing: three-block mask',
        stated=0.247,
        recompute=transfer("betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read", "mean_spearman", "arm"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: arm',
        stated=0.247,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: arm sd',
        stated=0.043,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: two-block mask control',
        stated=0.228,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: paired vs two-block mask',
        stated=0.019,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: paired sd vs two-block mask',
        stated=0.079,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: seeds above two-block mask',
        stated=3,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 216: paired vs width-12 incumbent',
        stated=-0.032,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: paired sd vs width-12 incumbent',
        stated=0.061,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: seeds above width-12 incumbent',
        stated=3,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 216: arm top-50',
        stated=3.39,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: two-block mask top-50',
        stated=3.66,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: incumbent top-50',
        stated=4.69,
        recompute=transfer('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against the width-12 one-block read', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: ensemble',
        stated=0.261,
        recompute=ensemble('w235_bxdelaapoolmlp256smm_h36l3p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: ensemble top-50',
        stated=3.59,
        recompute=ensemble('w235_bxdelaapoolmlp256smm_h36l3p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: seed min',
        stated=0.176,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: seed max',
        stated=0.283,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: select epoch min',
        stated=23,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 216: select epoch max',
        stated=170,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 216: val pearson min',
        stated=0.36,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: val pearson max',
        stated=0.41,
        recompute=pair_cells('betaxanthin: pool through 256 on three blocks of width 36 with the hard graph mask on the second and third against one unmasked block followed by one masked', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: covered',
        stated=0.183,
        recompute=coverage("w238w239", 'w235_bxdelaapoolmlp256smm_h36l3p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: uncovered',
        stated=0.075,
        recompute=coverage("w238w239", 'w235_bxdelaapoolmlp256smm_h36l3p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: constant heads',
        stated=3,
        recompute=decomposition_cells("w235", "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim='round 216: live head own read min',
        stated=0.025,
        recompute=decomposition_cells("w235", "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: live head own read max',
        stated=0.074,
        recompute=decomposition_cells("w235", "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: head costs least',
        stated=-0.006,
        recompute=decomposition_cells("w235", "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: head costs most',
        stated=-0.077,
        recompute=decomposition_cells("w235", "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: head costs middle',
        stated=-0.037,
        recompute=lambda: sorted(c["backbone_adds"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w235.json")))["cells"] if c["sd_ratio_head_over_slot"] >= 1e-2)[1],
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: slot alone',
        stated=0.267,
        recompute=decomposition("w235", "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 216: rebuilt reads match reported',
        stated=0.0,
        recompute=decomposition_cells("w235", "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim='round 216: agreement min',
        stated=0.74,
        recompute=agreement("w235_vs_w232", "min_same_seed_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 216: agreement max',
        stated=0.98,
        recompute=lambda: max(r["spearman"] for r in json.load(open(osp.join(RESULTS, "target_prediction_agreement_w235_vs_w232.json")))["seeds"]),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: arm',
        stated=0.236,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: arm sd',
        stated=0.018,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: 019 weight control',
        stated=0.226,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: paired vs 019 weight',
        stated=0.01,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: paired sd vs 019 weight',
        stated=0.016,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: seeds above 019 weight',
        stated=4,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 217: unmasked control',
        stated=0.27,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: paired vs unmasked',
        stated=-0.034,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: paired sd vs unmasked',
        stated=0.03,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: seeds above unmasked',
        stated=1,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 217: arm top-50',
        stated=3.93,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: 019 weight top-50',
        stated=4.16,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: unmasked top-50',
        stated=4.13,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight against two unmasked blocks', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: ensemble',
        stated=0.245,
        recompute=ensemble('w236_bxdelaapoolmlp256skl10_h36l2p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: ensemble top-50',
        stated=4.39,
        recompute=ensemble('w236_bxdelaapoolmlp256skl10_h36l2p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: seed min',
        stated=0.21,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: seed max',
        stated=0.258,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: select epoch min',
        stated=15,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 217: select epoch max',
        stated=194,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 217: val pearson min',
        stated=0.31,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: val pearson max',
        stated=0.4,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at ten times the 019 weight (6.5e-4) against the 019 weight', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: reg loss at selection min',
        stated=2.95,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.06,
    ),
    Check(
        claim='round 217: reg loss at selection max',
        stated=3.65,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.06,
    ),
    Check(
        claim='round 217: data term at selection min',
        stated=0.46,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: data term at selection max',
        stated=0.51,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w236_bxdelaapoolmlp256skl10_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: covered',
        stated=0.158,
        recompute=coverage("w238w239", 'w236_bxdelaapoolmlp256skl10_h36l2p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: uncovered',
        stated=0.061,
        recompute=coverage("w238w239", 'w236_bxdelaapoolmlp256skl10_h36l2p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: constant heads',
        stated=0,
        recompute=decomposition_cells('w236', "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim='round 217: head spread ratio min',
        stated=0.6,
        recompute=lambda: min(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w236.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 217: head spread ratio max',
        stated=1.2,
        recompute=lambda: max(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w236.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 217: head own read min',
        stated=0.024,
        recompute=decomposition_cells('w236', "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: head own read max',
        stated=0.076,
        recompute=decomposition_cells('w236', "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: head adds most',
        stated=-0.006,
        recompute=decomposition_cells('w236', "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: head subtracts most',
        stated=-0.042,
        recompute=decomposition_cells('w236', "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: slot alone',
        stated=0.26,
        recompute=decomposition('w236', "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 217: rebuilt reads match reported',
        stated=0.0,
        recompute=decomposition_cells('w236', "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim='round 217: agreement min',
        stated=0.73,
        recompute=agreement("w236_vs_w232", "min_same_seed_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 217: agreement max',
        stated=0.85,
        recompute=lambda: max(r["spearman"] for r in json.load(open(osp.join(RESULTS, "target_prediction_agreement_w236_vs_w232.json")))["seeds"]),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: arm',
        stated=0.243,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: arm sd',
        stated=0.031,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'sd_spearman', 'arm'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: 019 weight control',
        stated=0.226,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: paired vs 019 weight',
        stated=0.017,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: paired sd vs 019 weight',
        stated=0.037,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: seeds above 019 weight',
        stated=5,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 218: unmasked control',
        stated=0.27,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'mean_spearman', 'control'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: paired vs unmasked',
        stated=-0.027,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'mean_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: paired sd vs unmasked',
        stated=0.025,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'sd_difference'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: seeds above unmasked',
        stated=0,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'n_positive'),
        tolerance=0.5,
    ),
    Check(
        claim='round 218: arm top-50',
        stated=4.29,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_top50_enrichment', 'arm'),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: 019 weight top-50',
        stated=4.16,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: unmasked top-50',
        stated=4.13,
        recompute=transfer('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight against two unmasked blocks', 'mean_top50_enrichment', 'control'),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: ensemble',
        stated=0.255,
        recompute=ensemble('w237_bxdelaapoolmlp256skl100_h36l2p1_e200', "spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: ensemble top-50',
        stated=4.79,
        recompute=ensemble('w237_bxdelaapoolmlp256skl100_h36l2p1_e200', "top50_enrichment"),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: seed min',
        stated=0.186,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'spearman', 'min'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: seed max',
        stated=0.271,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'spearman', 'max'),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: select epoch min',
        stated=74,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'select_epoch', 'min'),
        tolerance=0.5,
    ),
    Check(
        claim='round 218: select epoch max',
        stated=189,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'select_epoch', 'max'),
        tolerance=0.5,
    ),
    Check(
        claim='round 218: val pearson min',
        stated=0.29,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'select_val_pearson', 'min'),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: val pearson max',
        stated=0.4,
        recompute=pair_cells('betaxanthin: pool through 256, two blocks of width 36, the KL regularizer on the second at a hundred times the 019 weight (6.5e-3) against the 019 weight', 'arm', 'select_val_pearson', 'max'),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: reg loss at selection min',
        stated=28.4,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.06,
    ),
    Check(
        claim='round 218: reg loss at selection max',
        stated=30.3,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["select_epoch"]]["graph_reg_loss"] for s in range(2700, 2706)),
        tolerance=0.06,
    ),
    Check(
        claim='round 218: data term at selection min',
        stated=0.48,
        recompute=lambda: min(json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: data term at selection max',
        stated=0.5,
        recompute=lambda: max(json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["history"][json.load(open(osp.join(RESULTS, f"w237_bxdelaapoolmlp256skl100_h36l2p1_e200_s{s}.json")))["select_epoch"]]["train_data_loss"] for s in range(2700, 2706)),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: covered',
        stated=0.172,
        recompute=coverage("w238w239", 'w237_bxdelaapoolmlp256skl100_h36l2p1_e200', "mean_spearman_covered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: uncovered',
        stated=0.127,
        recompute=coverage("w238w239", 'w237_bxdelaapoolmlp256skl100_h36l2p1_e200', "mean_spearman_uncovered"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: constant heads',
        stated=0,
        recompute=decomposition_cells('w237', "n_constant"),
        tolerance=0.5,
    ),
    Check(
        claim='round 218: head spread ratio min',
        stated=0.45,
        recompute=lambda: min(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w237.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 218: head spread ratio max',
        stated=0.94,
        recompute=lambda: max(c["sd_ratio_head_over_slot"] for c in json.load(open(osp.join(RESULTS, "target_head_decomposition_w237.json")))["cells"]),
        tolerance=0.06,
    ),
    Check(
        claim='round 218: head own read min',
        stated=-0.007,
        recompute=decomposition_cells('w237', "min_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: head own read max',
        stated=0.116,
        recompute=decomposition_cells('w237', "max_head"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: head adds most',
        stated=0.001,
        recompute=decomposition_cells('w237', "max_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: head subtracts most',
        stated=-0.078,
        recompute=decomposition_cells('w237', "min_backbone_adds"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: slot alone',
        stated=0.269,
        recompute=decomposition('w237', "mean_slot_spearman"),
        tolerance=0.0006,
    ),
    Check(
        claim='round 218: rebuilt reads match reported',
        stated=0.0,
        recompute=decomposition_cells('w237', "max_full_vs_reported"),
        tolerance=0.002,
    ),
    Check(
        claim='round 218: agreement min',
        stated=0.77,
        recompute=agreement("w237_vs_w232", "min_same_seed_spearman"),
        tolerance=0.006,
    ),
    Check(
        claim='round 218: agreement max',
        stated=0.89,
        recompute=lambda: max(r["spearman"] for r in json.load(open(osp.join(RESULTS, "target_prediction_agreement_w237_vs_w232.json")))["seeds"]),
        tolerance=0.006,
    ),
    Check(claim='round 223: arm mean', stated=0.219, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 223: arm sd', stated=0.006, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 223: control mean', stated=0.217, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 223: control sd', stated=0.005, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'sd_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 223: paired diff', stated=0.001, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 223: paired sd', stated=0.007, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 223: positive', stated=8, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'n_positive'), tolerance=0.5),
    Check(claim='round 223: n', stated=12, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'n'), tolerance=0.5),
    Check(claim='round 223: arm top-50', stated=4.91, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 223: control top-50', stated=4.94, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_top50_enrichment', 'control'), tolerance=0.006),
    Check(claim='round 223: ensemble first six', stated=0.218, recompute=ensemble('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 223: control ensemble first six', stated=0.217, recompute=ensemble('w190_bxdelgopfbamoma_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 223: ensemble twelve', stated=0.22, recompute=ensemble('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'spearman', 'all_seeds'), tolerance=0.0006),
    Check(claim='round 223: control ensemble twelve', stated=0.218, recompute=ensemble('w190_bxdelgopfbamoma_h12l1p1_e200', 'spearman', 'all_seeds'), tolerance=0.0006),
    Check(claim='round 223: seed min', stated=0.207, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'spearman', 'min', tuple(range(2700, 2712))), tolerance=0.0006),
    Check(claim='round 223: seed max', stated=0.226, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'spearman', 'max', tuple(range(2700, 2712))), tolerance=0.0006),
    Check(claim='round 223: epoch min', stated=6, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'select_epoch', 'min', tuple(range(2700, 2712))), tolerance=0.5),
    Check(claim='round 223: epoch max', stated=14, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'select_epoch', 'max', tuple(range(2700, 2712))), tolerance=0.5),
    Check(claim='round 223: val min', stated=0.24, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'select_val_pearson', 'min', tuple(range(2700, 2712))), tolerance=0.006),
    Check(claim='round 223: val max', stated=0.26, recompute=cells_stat('w242_bxdelgopfbamomaaux_h12l1p1_e200', 'select_val_pearson', 'max', tuple(range(2700, 2712))), tolerance=0.006),
    Check(claim='round 223: loss at epoch 0', stated=1.55, recompute=pack_log('w242_bxdelgopfbamomaaux_h12l1p1_e200', 2700, 0, 'loss'), tolerance=0.006),
    Check(claim='round 223: data at epoch 0', stated=0.53, recompute=pack_log('w242_bxdelgopfbamomaaux_h12l1p1_e200', 2700, 0, 'data'), tolerance=0.006),
    Check(claim='round 223: co-head over data at epoch 0', stated=1.9, recompute=pack_log('w242_bxdelgopfbamomaaux_h12l1p1_e200', 2700, 0, 'ratio'), tolerance=0.06),
    Check(claim='round 223: job 2945 hours', stated=6.65, recompute=job_hours(2945), tolerance=0.01),
    Check(claim='round 223: job 2946 hours', stated=6.86, recompute=job_hours(2946), tolerance=0.01),
    Check(claim='round 224: arm mean', stated=0.205, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 224: arm sd', stated=0.003, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 224: control mean', stated=0.202, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 224: control sd', stated=0.001, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'sd_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 224: paired diff', stated=0.003, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 224: paired sd', stated=0.004, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 224: positive', stated=4, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'n_positive'), tolerance=0.5),
    Check(claim='round 224: arm top-50', stated=5.06, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 224: control top-50', stated=5.03, recompute=transfer('betaxanthin: GO alone with the Mulleder co-head against without', 'mean_top50_enrichment', 'control'), tolerance=0.006),
    Check(claim='round 224: ensemble', stated=0.206, recompute=ensemble('w243_bxdelgoaux_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 224: control ensemble', stated=0.202, recompute=ensemble('w167_bxdelgo_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 224: seed min', stated=0.201, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'spearman', 'min'), tolerance=0.0006),
    Check(claim='round 224: seed max', stated=0.208, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'spearman', 'max'), tolerance=0.0006),
    Check(claim='round 224: epoch min', stated=5, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'select_epoch', 'min'), tolerance=0.5),
    Check(claim='round 224: epoch max', stated=12, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'select_epoch', 'max'), tolerance=0.5),
    Check(claim='round 224: val min', stated=0.25, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'select_val_pearson', 'min'), tolerance=0.006),
    Check(claim='round 224: val max', stated=0.26, recompute=cells_stat('w243_bxdelgoaux_h12l1p1_e200', 'select_val_pearson', 'max'), tolerance=0.006),
    Check(claim='round 224: job hours', stated=6.88, recompute=job_hours(2947), tolerance=0.01),
    Check(claim='round 225: arm mean', stated=0.075, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 225: arm sd', stated=0.035, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 225: control mean', stated=0.058, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 225: control sd', stated=0.075, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'sd_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 225: paired diff', stated=0.017, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 225: paired sd', stated=0.096, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 225: positive', stated=3, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'n_positive'), tolerance=0.5),
    Check(claim='round 225: arm top-50', stated=4.43, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 225: control top-50', stated=3.39, recompute=transfer('betaxanthin: the genotype-only control with the Mulleder co-head against without', 'mean_top50_enrichment', 'control'), tolerance=0.006),
    Check(claim='round 225: ensemble', stated=0.108, recompute=ensemble('w244_bxdelaux_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 225: control ensemble', stated=0.082, recompute=ensemble('w137_bxdel_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 225: seed min', stated=0.024, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'spearman', 'min'), tolerance=0.0006),
    Check(claim='round 225: seed max', stated=0.125, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'spearman', 'max'), tolerance=0.0006),
    Check(claim='round 225: epoch min', stated=32, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'select_epoch', 'min'), tolerance=0.5),
    Check(claim='round 225: epoch max', stated=147, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'select_epoch', 'max'), tolerance=0.5),
    Check(claim='round 225: val min', stated=0.23, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'select_val_pearson', 'min'), tolerance=0.006),
    Check(claim='round 225: val max', stated=0.31, recompute=cells_stat('w244_bxdelaux_h12l1p1_e200', 'select_val_pearson', 'max'), tolerance=0.006),
    Check(claim='round 225: job hours', stated=6.82, recompute=job_hours(2948), tolerance=0.01),
    Check(claim='round 226: arm mean', stated=0.145, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 226: arm sd', stated=0.072, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 226: control mean', stated=0.202, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 226: control sd', stated=0.001, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'sd_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 226: paired diff', stated=-0.058, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 226: paired sd', stated=0.071, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 226: positive', stated=1, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'n_positive'), tolerance=0.5),
    Check(claim='round 226: arm top-50', stated=4.26, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 226: control top-50', stated=5.03, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_top50_enrichment', 'control'), tolerance=0.006),
    Check(claim='round 226: ensemble', stated=0.149, recompute=ensemble('w245_bxdelgonbr_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 226: control ensemble', stated=0.202, recompute=ensemble('w167_bxdelgo_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 226: seed min', stated=0.027, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'spearman', 'min'), tolerance=0.0006),
    Check(claim='round 226: seed max', stated=0.210, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'spearman', 'max'), tolerance=0.0006),
    Check(claim='round 226: epoch min', stated=6, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'select_epoch', 'min'), tolerance=0.5),
    Check(claim='round 226: epoch max', stated=165, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'select_epoch', 'max'), tolerance=0.5),
    Check(claim='round 226: val min', stated=0.12, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'select_val_pearson', 'min'), tolerance=0.006),
    Check(claim='round 226: val max', stated=0.19, recompute=cells_stat('w245_bxdelgonbr_h12l1p1_e200', 'select_val_pearson', 'max'), tolerance=0.006),
    Check(claim='round 226: covered', stated=0.089, recompute=coverage('w245w247', 'w245_bxdelgonbr_h12l1p1_e200', 'mean_spearman_covered'), tolerance=0.0006),
    Check(claim='round 226: uncovered', stated=0.246, recompute=coverage('w245w247', 'w245_bxdelgonbr_h12l1p1_e200', 'mean_spearman_uncovered'), tolerance=0.0006),
    Check(claim='round 226: columns', stated=4726, recompute=feature_store_columns('YPD_bx_gonbr'), tolerance=0.5),
    Check(claim='round 226: ladder forest on the columns', stated=0.279, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'forest'), tolerance=0.0006),
    Check(claim='round 226: ladder ridge on the columns', stated=0.183, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'ridge'), tolerance=0.0006),
    Check(claim='round 226: job hours', stated=3.53, recompute=job_hours(2973), tolerance=0.01),
    Check(claim='round 227: arm mean', stated=0.221, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 227: arm sd', stated=0.004, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 227: paired diff', stated=0.019, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 227: paired sd', stated=0.005, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 227: positive', stated=6, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'n_positive'), tolerance=0.5),
    Check(claim='round 227: arm top-50', stated=5.86, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 227: ensemble', stated=0.221, recompute=ensemble('w246_bxdelgogonbr_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 227: seed min', stated=0.215, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'spearman', 'min'), tolerance=0.0006),
    Check(claim='round 227: seed max', stated=0.227, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'spearman', 'max'), tolerance=0.0006),
    Check(claim='round 227: epoch min', stated=3, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'select_epoch', 'min'), tolerance=0.5),
    Check(claim='round 227: epoch max', stated=8, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'select_epoch', 'max'), tolerance=0.5),
    Check(claim='round 227: val min', stated=0.22, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'select_val_pearson', 'min'), tolerance=0.006),
    Check(claim='round 227: val max', stated=0.23, recompute=cells_stat('w246_bxdelgogonbr_h12l1p1_e200', 'select_val_pearson', 'max'), tolerance=0.006),
    Check(claim='round 227: against propagation alone diff', stated=0.076, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against the propagation alone', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 227: against propagation alone sd', stated=0.073, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against the propagation alone', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 227: against propagation alone positive', stated=6, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against the propagation alone', 'n_positive'), tolerance=0.5),
    Check(claim='round 227: covered', stated=0.159, recompute=coverage('w245w247', 'w246_bxdelgogonbr_h12l1p1_e200', 'mean_spearman_covered'), tolerance=0.0006),
    Check(claim='round 227: GO covered', stated=0.140, recompute=coverage('aapool', 'w167_bxdelgo_h12l1p1_e200', 'mean_spearman_covered'), tolerance=0.0006),
    Check(claim='round 227: uncovered', stated=0.329, recompute=coverage('w245w247', 'w246_bxdelgogonbr_h12l1p1_e200', 'mean_spearman_uncovered'), tolerance=0.0006),
    Check(claim='round 227: columns', stated=7088, recompute=feature_store_columns('YPD_bx_gogonbr'), tolerance=0.5),
    Check(claim='round 227: round 190 twelve seeds', stated=0.217, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 227: forest GO on the ladder', stated=0.187, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go', 'forest'), tolerance=0.0006),
    Check(claim='round 227: a fifth of the forest lift', stated=0.20, recompute=lift_ratio('betaxanthin: GO with its neighbor propagation appended against GO alone', 'store_baselines_betaxanthin_nbr.json', 'go_nbr', 'go'), tolerance=0.03),
    Check(claim='round 227: job hours', stated=3.59, recompute=job_hours(2974), tolerance=0.01),
    Check(claim='round 228: arm mean', stated=0.217, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 228: arm sd', stated=0.013, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'sd_spearman', 'arm'), tolerance=0.0006),
    Check(claim='round 228: over linear diff', stated=0.073, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 228: over linear sd', stated=0.076, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 228: over linear positive', stated=6, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'n_positive'), tolerance=0.5),
    Check(claim='round 228: under pool diff', stated=-0.062, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the pool through 256', 'mean_difference'), tolerance=0.0006),
    Check(claim='round 228: under pool sd', stated=0.033, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the pool through 256', 'sd_difference'), tolerance=0.0006),
    Check(claim='round 228: under pool positive', stated=0, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the pool through 256', 'n_positive'), tolerance=0.5),
    Check(claim='round 228: pool through 256', stated=0.279, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the pool through 256', 'mean_spearman', 'control'), tolerance=0.0006),
    Check(claim='round 228: top-50', stated=5.03, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'mean_top50_enrichment', 'arm'), tolerance=0.006),
    Check(claim='round 228: ensemble', stated=0.232, recompute=ensemble('w247_bxdelgonbrmlp256_h12l1p1_e200', 'spearman', 'first_six'), tolerance=0.0006),
    Check(claim='round 228: seed min', stated=0.204, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'spearman', 'min'), tolerance=0.0006),
    Check(claim='round 228: seed max', stated=0.235, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'spearman', 'max'), tolerance=0.0006),
    Check(claim='round 228: epoch min', stated=14, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'select_epoch', 'min'), tolerance=0.5),
    Check(claim='round 228: epoch max', stated=128, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'select_epoch', 'max'), tolerance=0.5),
    Check(claim='round 228: val min', stated=0.13, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'select_val_pearson', 'min'), tolerance=0.006),
    Check(claim='round 228: val max', stated=0.19, recompute=cells_stat('w247_bxdelgonbrmlp256_h12l1p1_e200', 'select_val_pearson', 'max'), tolerance=0.006),
    Check(claim='round 228: head parameters', stated=1211095, recompute=n_params('w247_bxdelgonbrmlp256_h12l1p1_e200', 'betaxanthin_head'), tolerance=0.5),
    Check(claim='round 228: total parameters', stated=1229275, recompute=n_params('w247_bxdelgonbrmlp256_h12l1p1_e200', 'total'), tolerance=0.5),
    Check(claim='round 228: linear head parameters', stated=5198, recompute=n_params('w245_bxdelgonbr_h12l1p1_e200', 'betaxanthin_head'), tolerance=0.5),
    Check(claim='round 228: covered', stated=0.161, recompute=coverage('w245w247', 'w247_bxdelgonbrmlp256_h12l1p1_e200', 'mean_spearman_covered'), tolerance=0.0006),
    Check(claim='round 228: uncovered', stated=0.253, recompute=coverage('w245w247', 'w247_bxdelgonbrmlp256_h12l1p1_e200', 'mean_spearman_uncovered'), tolerance=0.0006),
    Check(claim='round 228: job hours', stated=3.61, recompute=job_hours(2975), tolerance=0.01),
    Check(claim='standing: round 227 arm', stated=0.221, recompute=transfer('betaxanthin: GO with its neighbor propagation appended against GO alone', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='standing: round 226 arm', stated=0.145, recompute=transfer('betaxanthin: the GO neighbor-propagation store (inductive) against GO alone', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='standing: round 228 arm', stated=0.217, recompute=transfer('betaxanthin: the GO neighbor-propagation store through 256 against the linear affine', 'mean_spearman', 'arm'), tolerance=0.0006),
    Check(claim='inductive: go_nbr forest', stated=0.279, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: go_nbr knn', stated=0.221, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'knn'), tolerance=0.0006),
    Check(claim='inductive: go_nbr ridge', stated=0.183, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'ridge'), tolerance=0.0006),
    Check(claim='inductive: go forest', stated=0.187, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go', 'forest'), tolerance=0.0006),
    Check(claim='inductive: pool forest', stated=0.28, recompute=ladder_file('store_baselines_betaxanthin.json', 'aa_pool', 'forest'), tolerance=0.0006),
    Check(claim='inductive: coexpression block forest', stated=0.251, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'go_nbr_coexpression', 'forest'), tolerance=0.0006),
    Check(claim='inductive: union block forest', stated=0.258, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'go_nbr_union', 'forest'), tolerance=0.0006),
    Check(claim='inductive: go+go_nbr forest', stated=0.271, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go+go_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: degree forest', stated=-0.003, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'go_nbr_degree', 'forest'), tolerance=0.0006),
    Check(claim='inductive: degree ridge', stated=-0.009, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'go_nbr_degree', 'ridge'), tolerance=0.0006),
    Check(claim='inductive: degree knn', stated=-0.011, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'go_nbr_degree', 'knn'), tolerance=0.0006),
    Check(claim='inductive: pfba_nbr forest', stated=0.182, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'pfba_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: pfba forest', stated=0.116, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'pfba_highs', 'forest'), tolerance=0.0006),
    Check(claim='inductive: pfba degree forest', stated=0.02, recompute=ladder_file('store_baselines_betaxanthin_nbr_decomp.json', 'pfba_nbr_degree', 'forest'), tolerance=0.0006),
    Check(claim='inductive: pfba+pfba_nbr forest', stated=0.189, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'pfba+pfba_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb aa_pool', stated=0.224, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'aa_pool', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest aa_pool beside xgb', stated=0.28, recompute=ladder_file('store_baselines_betaxanthin.json', 'aa_pool', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb aapool_graphlabel', stated=0.275, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'aapool_graphlabel', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest aapool_graphlabel beside xgb', stated=0.284, recompute=ladder_file('store_baselines_betaxanthin.json', 'aapool_graphlabel', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb go_nbr', stated=0.228, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'go_nbr', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest go_nbr beside xgb', stated=0.279, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb go+go_nbr', stated=0.249, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'go+go_nbr', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest go+go_nbr beside xgb', stated=0.271, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go+go_nbr', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb graph_label', stated=0.21, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'graph_label', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest graph_label beside xgb', stated=0.254, recompute=ladder_file('store_baselines_betaxanthin.json', 'graph_label', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb go', stated=0.18, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'go', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest go beside xgb', stated=0.187, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'go', 'forest'), tolerance=0.0006),
    Check(claim='inductive: xgb prot_T5_all', stated=0.13, recompute=ladder_file('store_baselines_betaxanthin_xgb.json', 'prot_T5_all', 'xgb'), tolerance=0.0006),
    Check(claim='inductive: forest prot_T5_all beside xgb', stated=0.143, recompute=ladder_file('store_baselines_betaxanthin_nbr.json', 'prot_T5_all', 'forest'), tolerance=0.0006),
    Check(claim='inductive: mlp_go', stated=0.199, recompute=gcn('mlp_go', 'mean'), tolerance=0.0006),
    Check(claim='inductive: gcn union', stated=0.265, recompute=gcn('gcn_union_go', 'mean'), tolerance=0.0006),
    Check(claim='inductive: gcn union edges', stated=1514071, recompute=gcn('gcn_union_go', 'n_edges'), tolerance=0.5),
    Check(claim='inductive: gcn union epoch min', stated=564, recompute=gcn('gcn_union_go', 'epoch_min'), tolerance=0.5),
    Check(claim='inductive: gcn union epoch max', stated=592, recompute=gcn('gcn_union_go', 'epoch_max'), tolerance=0.5),
    Check(claim='inductive: gcn coexpression', stated=0.259, recompute=gcn('gcn_coexpression_go', 'mean'), tolerance=0.0006),
    Check(claim='inductive: gcn coexpression edges', stated=996199, recompute=gcn('gcn_coexpression_go', 'n_edges'), tolerance=0.5),
    Check(claim='inductive: gcn coexpression epoch min', stated=419, recompute=gcn('gcn_coexpression_go', 'epoch_min'), tolerance=0.5),
    Check(claim='inductive: gcn coexpression epoch max', stated=552, recompute=gcn('gcn_coexpression_go', 'epoch_max'), tolerance=0.5),
    Check(claim='inductive: gcn union at 200', stated=0.237, recompute=gcn('gcn_union_go', 'mean', 'e200'), tolerance=0.0006),
    Check(claim='inductive: gcn union at 200 epoch min', stated=199, recompute=gcn('gcn_union_go', 'epoch_min', 'e200'), tolerance=0.5),
    Check(claim='inductive: gcn coexpression at 200', stated=0.251, recompute=gcn('gcn_coexpression_go', 'mean', 'e200'), tolerance=0.0006),
    Check(claim='inductive: incidence nodes', stated=8969, recompute=gcn('gcn_go_incidence', 'n_nodes'), tolerance=0.5),
    Check(claim='inductive: incidence edges', stated=271088, recompute=gcn('gcn_go_incidence', 'n_edges'), tolerance=0.5),
    Check(claim='inductive: incidence dag edges', stated=52830, recompute=gcn('gcn_go_incidence_dag', 'n_edges'), tolerance=0.5),
    Check(claim='inductive: incidence', stated=0.168, recompute=gcn('gcn_go_incidence', 'mean'), tolerance=0.0006),
    Check(claim='inductive: incidence epoch min', stated=188, recompute=gcn('gcn_go_incidence', 'epoch_min'), tolerance=0.5),
    Check(claim='inductive: incidence epoch max', stated=223, recompute=gcn('gcn_go_incidence', 'epoch_max'), tolerance=0.5),
    Check(claim='inductive: incidence dag', stated=0.141, recompute=gcn('gcn_go_incidence_dag', 'mean'), tolerance=0.0006),
    Check(claim='inductive: incidence dag epoch min', stated=62, recompute=gcn('gcn_go_incidence_dag', 'epoch_min'), tolerance=0.5),
    Check(claim='inductive: incidence dag epoch max', stated=73, recompute=gcn('gcn_go_incidence_dag', 'epoch_max'), tolerance=0.5),
    Check(claim='inductive: incidence ones', stated=0.014, recompute=gcn('gcn_go_incidence_ones', 'mean'), tolerance=0.0006),
    Check(claim='inductive: incidence dag ones', stated=-0.031, recompute=gcn('gcn_go_incidence_dag_ones', 'mean'), tolerance=0.0006),
    Check(claim='standing: transformer inductive incumbent', stated=0.217, recompute=transfer('betaxanthin: the inductive incumbent (GO, HiGHS pFBA, MOMA) with the Mulleder co-head against without, twelve seeds', 'mean_spearman', 'control'), tolerance=0.0006),
]


def main() -> None:
    failed = 0
    for check in CHECKS:
        got = check.recompute()
        agrees = abs(got - check.stated) < check.tolerance
        failed += 0 if agrees else 1
        mark = "ok  " if agrees else "WRONG"
        print(f"{mark} {check.claim:58s} document {check.stated:,g}  files {got:,.4f}")
    print(f"\n{len(CHECKS)} checks, {failed} disagreeing with the document")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

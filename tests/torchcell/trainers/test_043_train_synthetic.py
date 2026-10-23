# tests/torchcell/trainers/test_043_train_synthetic.py
# [[tests.torchcell.trainers.test_043_train_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_043_train_synthetic.py
"""The 043 training stack end to end on synthetic data, on CPU, without the store.

``experiments/043-expression-proteome-morphology/scripts/train_043.py`` is loaded by path.
The fixture writes four tiny ``<encoder>.h5`` tables and their vocabularies, a fake
``GlobalSplit043`` fold file, and a :class:`PreparedStore043` over 12 genes: Kemmeren
deletions, Messner deletions, Caudal isolates on their own reference contexts (route A,
with accessory ORF marks), Vaishnav reporter records (record level) and two Sameith
doubles. Roles come from the fake fold file through ``GlobalSplit043.role``. Each loss arm
then trains two optimizer steps and runs the evaluation, the doubles read and the
checkpoint write.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch

SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "043-expression-proteome-morphology"
    / "scripts"
)
G = 12
DIMS = (8, 8, 12, 8)
ENCODERS = (
    "fudt_species_upstream",
    "fudt_species_downstream",
    "prot_t5_xl_uniref50",
    "calm",
)
GENES = [f"Y{i:02d}" for i in range(G)]


@pytest.fixture(scope="module")
def tr() -> ModuleType:
    """The 043 training script, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "train_043_synth", SCRIPTS / "train_043.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_043_synth"] = module
    spec.loader.exec_module(module)
    return module


def _key(e: int, i: int) -> str:
    return f"{e:02x}{i:062x}"


def _write_vectors(root: Path, n_vocab: int) -> None:
    """Vocabularies of ``n_vocab`` keys per encoder and h5 tables holding all of them."""
    rng = np.random.default_rng(0)
    (root / "rep").mkdir()
    (root / "alleles").mkdir()
    for e, (enc, dim) in enumerate(zip(ENCODERS, DIMS, strict=True)):
        keys = np.asarray(sorted(_key(e, i) for i in range(n_vocab)), dtype="S64")
        np.save(root / "rep" / f"vocab_{enc}.npy", keys)
        order = rng.permutation(n_vocab)  # table rows in another order than the vocab
        with h5py.File(root / "alleles" / f"{enc}.h5", "w") as f:
            f["sha256"] = keys[order]
            f["embedding"] = rng.normal(size=(n_vocab, dim)).astype(np.float16)


def _fake_split(tr: ModuleType, path: Path) -> Any:
    split = tr.GlobalSplit043(
        store_sha256="0" * 64,
        store_index_path="/fake/raw_dataset_name_index.json",
        gene_set_sha256="1" * 64,
        pinned_training_genes=["Y00", "Y01"],
        pinned_reason={"Y00": "Sameith single", "Y01": "Sameith single"},
        gene_folds=dict(zip(GENES, [2, 3, 4, 5, 6, 7, 1, 8, 9, 1, 0, 0], strict=True)),
        gene_strata={g: "x" for g in GENES},
        isolate_folds={"ISO_A": 2, "ISO_B": 3, "ISO_C": 1, "ISO_D": 4},
        isolate_strata={},
        segregant_folds={},
        promoter_folds={f"MER{i}": (i % 5) for i in range(8)},
        counts={},
        generated_by="test",
    )
    path.write_text(split.model_dump_json())
    return tr.load_split(str(path))


def _store(tr: ModuleType, split: Any) -> Any:
    """Records with roles from the fake fold file; test-fold records are left out."""
    rng = np.random.default_rng(1)
    shorts = list(tr.SPECS)
    rows: list[dict[str, Any]] = []
    # Kemmeren / Messner single deletions of genes 1..10; Y06 and Y09 validation, Y10 test
    for name, short in (
        ("MicroarrayKemmeren2014Dataset", "kemmeren"),
        ("ProteomeMessner2023Dataset", "messner"),
    ):
        for gi in range(1, 11):
            rows.append({"short": short, "name": name, "key": [GENES[gi]], "gene": gi})
    rows[0]["key"] = ["Y00"]  # pinned single
    rows[0]["gene"] = 0
    for iso, ctx in (("ISO_A", 1), ("ISO_B", 2), ("ISO_C", 1), ("ISO_D", 2)):
        rows.append(
            {
                "short": "caudal",
                "name": "CaudalPanTranscriptome2024Dataset",
                "key": iso,
                "ctx": ctx,
            }
        )
    for i in range(8):
        rows.append(
            {
                "short": "vaishnav",
                "name": "PooledPromoterExpressionVaishnav2022Dataset",
                "key": f"MER{i}",
            }
        )
    for a, b in ((0, 2), (3, 4)):
        rows.append(
            {
                "short": "sameith_dm",
                "name": "DmMicroarraySameith2015Dataset",
                "key": [GENES[a], GENES[b]],
                "genes2": (a, b),
            }
        )
    kept = []
    for r in rows:
        role = split.role(r["name"], r["key"])
        if role in tr.ROLE_CODE:
            r["role"] = tr.ROLE_CODE[role]
            kept.append(r)
    # contexts: 0 = S288C, 1 and 2 = isolates with a gap
    C = 3
    ctx_keys = rng.integers(0, 20, size=(C, G, 4)).astype(np.int32)
    ctx_gaps = np.zeros((C, G, 4), np.int8)
    ctx_keys[1, 3, 2] = -1
    ctx_gaps[1, 3, 2] = 2
    ctx_gene_copy = np.ones((C, G), np.float32)
    ctx_gene_copy[2, 5] = np.nan
    ctx_pgap = np.zeros((C, G), np.int8)
    ctx_pgap[2, 5] = 1
    Y: list[npt.NDArray[np.float32]] = []
    rec: dict[str, list[Any]] = {
        k: []
        for k in ("dataset", "raw", "role", "env", "ctx", "yrow", "ys", "g2", "unit")
    }
    marks: dict[str, list[Any]] = {
        k: []
        for k in (
            "gene",
            "on",
            "change",
            "parent",
            "keys",
            "gaps",
            "rep",
            "rel",
            "cc",
            "gc",
            "pg",
        )
    }
    ptr = [0]
    deletion, present, promoter = 1, 5, 6
    for i, r in enumerate(kept):
        short = r["short"]
        rec["dataset"].append(shorts.index(short))
        rec["raw"].append(1000 + i)
        rec["role"].append(r["role"])
        rec["env"].append(0 if short != "messner" else 1)
        rec["ctx"].append(r.get("ctx", 0))
        rec["unit"].append(
            r["key"] if isinstance(r["key"], str) else "+".join(r["key"])
        )
        g2 = r.get("genes2", (r.get("gene", -1), -1))
        rec["g2"].append(list(g2))
        if short == "vaishnav":
            rec["yrow"].append(-1)
            rec["ys"].append(float(rng.normal()))
        else:
            y = rng.normal(size=G).astype(np.float32)
            y[rng.random(G) < 0.2] = np.nan
            rec["yrow"].append(len(Y))
            rec["ys"].append(float("nan"))
            Y.append(y)
        # marks
        if short in ("kemmeren", "messner"):
            ms = [(r["gene"], True, deletion)]
        elif short == "sameith_dm":
            ms = [(g2[0], True, deletion), (g2[1], True, deletion)]
        elif short == "caudal":
            ms = [(0, False, present)] * 3  # accessory ORFs, exceeds the cap of 2
        else:
            ms = [(0, False, promoter)]
        for gene, on, ch in ms:
            marks["gene"].append(gene)
            marks["on"].append(on)
            marks["change"].append(ch)
            marks["parent"].append(0)
            marks["keys"].append(rng.integers(0, 20, size=4))
            marks["gaps"].append(np.zeros(4))
            marks["rep"].append(2 if on else (17 if ch == present else 18))
            marks["rel"].append(0.3 if on else 0.0)
            marks["cc"].append(1.0)
            marks["gc"].append(1.0)
            marks["pg"].append(0)
        ptr.append(ptr[-1] + len(ms))
    meta = tr.PreparedMeta(
        prep_version=tr.PREP_VERSION,
        route="A",
        split_version=split.version,
        split_store_sha256=split.store_sha256,
        datasets=shorts,
        genes=GENES,
        env_keys=["env0", "env1"],
        context_ids=["S288C", "ISO_A", "ISO_B"],
        graph_names=list(tr.GRAPH_NAMES),
        counts={},
        created_at="now",
        host="test",
        slurm_job_id=None,
        torchcell_commit="test",
        wall_seconds=0.0,
    )
    edges = {
        name: np.asarray([[0, 1, 2, 5], [1, 2, 3, 6]], np.int64)
        for name in tr.GRAPH_NAMES
    }
    return tr.PreparedStore043(
        meta=meta,
        rec_dataset=np.asarray(rec["dataset"], np.int16),
        rec_raw_index=np.asarray(rec["raw"], np.int64),
        rec_role=np.asarray(rec["role"], np.int8),
        rec_env=np.asarray(rec["env"], np.int32),
        rec_context=np.asarray(rec["ctx"], np.int32),
        rec_yrow=np.asarray(rec["yrow"], np.int32),
        rec_yscalar=np.asarray(rec["ys"], np.float32),
        rec_genes2=np.asarray(rec["g2"], np.int32),
        rec_unit=np.asarray(rec["unit"], dtype=np.str_),
        mark_ptr=np.asarray(ptr, np.int64),
        mark_gene=np.asarray(marks["gene"], np.int32),
        mark_on_graph=np.asarray(marks["on"], bool),
        mark_change=np.asarray(marks["change"], np.int8),
        mark_parent=np.asarray(marks["parent"], np.int8),
        mark_keys=np.asarray(marks["keys"], np.int32),
        mark_gaps=np.asarray(marks["gaps"], np.int8),
        mark_replicon=np.asarray(marks["rep"], np.int8),
        mark_relmid=np.asarray(marks["rel"], np.float32),
        mark_chrom_copy=np.asarray(marks["cc"], np.float32),
        mark_gene_copy=np.asarray(marks["gc"], np.float32),
        mark_pgap=np.asarray(marks["pg"], np.int8),
        ctx_keys=ctx_keys,
        ctx_gaps=ctx_gaps,
        ctx_chrom_copy=np.ones((C, G), np.float32),
        ctx_gene_copy=ctx_gene_copy,
        ctx_pgap=ctx_pgap,
        chrom_index=np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 0], np.int8),
        rel_midpoint=np.linspace(0.05, 0.95, G).astype(np.float32),
        Y=np.stack(Y),
        edges=edges,
    )


@pytest.fixture(scope="module")
def world(tr: ModuleType, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Vectors, split, store and the base config, shared by the tests."""
    root = tmp_path_factory.mktemp("w043")
    _write_vectors(root, n_vocab=20)
    split = _fake_split(tr, root / "split.json")
    store = _store(tr, split)
    store.save(str(root / "prep"))
    return {"root": root, "split": split, "store": store}


def _cfg(tr: ModuleType, root: Path, loss: str, weighting: str = "equal") -> Any:
    return tr.TrainConfig043.model_validate(
        {
            "seed": 0,
            "device": "cpu",
            "arm": {
                "name": "joint_test",
                "datasets": ["kemmeren", "messner", "caudal", "vaishnav"],
                "doubles": True,
            },
            "loss": {"kind": loss},
            "weighting": {"kind": weighting},
            "data": {
                "batch_size": 6,
                "num_workers": 0,
                "mark_cap": 2,
                "val_max_records_per_dataset": 50,
                "traineval_max_records_per_dataset": 4,
                "min_strains_per_gene": 2,
            },
            "model": {
                "hidden_dim": 16,
                "num_heads": 2,
                "num_layers": 2,
                "dropout": 0.0,
                "mask_layers": [1],
                "head_graphs": {0: "physical", 1: "regulatory"},
                "head_hidden": 8,
            },
            "optim": {"lr": 1e-3, "warmup_steps": 1},
            "budget": {"max_epochs": 1, "max_steps_per_epoch": 2, "probe_steps": 1},
            "wandb": {"mode": "disabled"},
            "paths": {
                "data_root": str(root),
                "split_file": str(root / "split.json"),
                "prep_dir": str(root / "prep"),
                "representation_root": str(root / "rep"),
                "alleles_root": str(root / "alleles"),
                "checkpoint_root": str(root / "ckpt"),
            },
        }
    )


def test_store_roundtrip_and_roles(tr: ModuleType, world: dict[str, Any]) -> None:
    store = world["store"]
    back = tr.PreparedStore043.load(str(world["root"] / "prep"))
    for k, v in store.array_fields().items():
        np.testing.assert_array_equal(back.array_fields()[k], v)
    assert back.meta == store.meta
    # fold 0 is test: gene Y10 (fold 0) is never loaded; ISO_C (fold 1) is validation
    units = set(store.rec_unit.tolist())
    assert "Y10" not in units
    roles = dict(zip(store.rec_unit.tolist(), store.rec_role.tolist(), strict=False))
    assert roles["ISO_C"] == tr.ROLE_CODE[tr.Role.VAL]
    assert roles["Y00+Y02"] == tr.ROLE_CODE[tr.Role.SPLIT_B]


def test_view_fits_standardization_on_training_rows_only(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    store = world["store"]
    arm = tr.ArmConfig(
        name="a", datasets=["kemmeren", "caudal", "vaishnav"], doubles=True
    )
    v1 = tr.build_view(store, arm, mark_cap=2)
    # perturbing a validation row's target leaves the statistics unchanged
    val_rows = v1.rec[v1.role == tr.ROLE_CODE[tr.Role.VAL]]
    gene_val = [r for r in val_rows if store.rec_yrow[r] >= 0]
    Y2 = store.Y.copy()
    Y2[store.rec_yrow[gene_val]] += 100.0
    store2 = tr.PreparedStore043(**{**store.__dict__, "Y": Y2})
    v2 = tr.build_view(store2, arm, mark_cap=2)
    assert v1.stats.sha256() == v2.stats.sha256()
    assert v1.stats.get("caudal").center_per_gene
    # doubles: both genes have a Kemmeren single in the arm
    assert len(v1.doubles) >= 1
    assert v1.doubles_singles.shape == (len(v1.doubles), 2)
    # compact rows index the used ids
    for e in range(4):
        assert v1.ctx_rows[..., e].max() < len(v1.used_ids[e])


def test_make_batch_caps_marks_and_dedups_contexts(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    store = world["store"]
    arm = tr.ArmConfig(
        name="a", datasets=["kemmeren", "caudal", "vaishnav"], doubles=False
    )
    view = tr.build_view(store, arm, mark_cap=2)
    pos = np.arange(len(view.rec))
    batch, tg = tr.make_batch(view, pos.tolist())
    assert batch.marks.valid.sum(1).max() <= 2  # the smoke cap
    n_ctx = len(np.unique(store.rec_context[view.rec]))
    assert batch.contexts.key_rows.shape[0] == n_ctx
    assert tg.gene_z.shape == (int(view.level_gene.sum()), G)
    assert len(tg.record_rows) == int((~view.level_gene).sum())


@pytest.mark.parametrize("loss", ["hlgauss", "pinball", "mse"])
def test_two_training_steps_per_loss_arm(
    tr: ModuleType, world: dict[str, Any], loss: str
) -> None:
    root = world["root"]
    cfg = _cfg(tr, root, loss)
    logger = tr.MemoryLogger()
    result = tr.train(cfg, world["store"], tr.h5_vector_source(cfg.paths), logger)
    assert result.steps == 2
    assert result.n_epochs == 1
    last = result.last_metrics
    assert np.isfinite(last["train/loss"])
    assert "train/kemmeren/loss" in last
    assert "val/expression/pearson_per_gene_median" in last
    assert "val/vaishnav/pearson" in last or "traineval/vaishnav/pearson" in last
    assert "val/doubles/pearson_per_strain_centered_median" in last
    if loss == "hlgauss":
        assert "val/acc_within_one_bin" in last
    ckpt = Path(result.checkpoint_dir)
    assert (ckpt / "last.pt").exists()
    stats = json.loads((ckpt / "standardization.json").read_text())
    assert stats["split_version"] == world["split"].version
    state = torch.load(ckpt / "last.pt", weights_only=False)
    assert state["standardization_sha256"] == result.stats_sha256


def test_records_weighting_and_missing_vector_raises(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    root = world["root"]
    cfg = _cfg(tr, root, "mse", "records")
    result = tr.train(
        cfg, world["store"], tr.h5_vector_source(cfg.paths), tr.MemoryLogger()
    )
    assert result.steps == 2

    def missing(e: int, ids: npt.NDArray[np.int64]) -> torch.Tensor:
        enc = ENCODERS[e]
        vocab = np.load(root / "rep" / f"vocab_{enc}.npy")
        keys = vocab[ids].copy()
        keys[0] = b"f" * 64  # a key with no vector
        out: torch.Tensor = tr.load_vector_table(
            str(root / "alleles" / f"{enc}.h5"), keys
        )
        return out

    with pytest.raises(KeyError, match="have no vector"):
        tr.train(cfg, world["store"], missing, tr.MemoryLogger())


def test_transforms_and_window(tr: ModuleType) -> None:
    y = np.array([4.0, 0.0, np.nan])
    ref = np.array([2.0, 1.0, 1.0])
    assert tr.transform_target("log2_ratio_to_reference", y, ref)[0] == pytest.approx(
        1.0
    )
    assert np.isnan(tr.transform_target("log2_ratio_to_reference", y, ref)[1])
    assert tr.transform_target("difference_to_reference", y, ref)[0] == 2.0
    assert np.isnan(tr.transform_target("log2_positive", y)[1])
    cpm = tr.transform_target(
        "log2_cpm_plus_one", np.array([1.0, np.nan]), total_umis=1e6
    )
    assert cpm.tolist() == [1.0, 0.0]  # omitted keys are zero counts
    with pytest.raises(ValueError, match="total_umis"):
        tr.transform_target("log2_cpm_plus_one", y)
    w = tr.WindowConfig(start_frac=0.8, end_frac=1.0)
    assert w.epochs(50) == (41, 50)
    assert w.epochs(1) == (1, 1)
    with pytest.raises(ValueError, match="start_frac"):
        tr.WindowConfig(start_frac=0.9, end_frac=0.5)


def test_gene_level_metrics_on_known_values(tr: ModuleType) -> None:
    Y = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [3.0, 6.0, 2.0]])
    m = tr.gene_level_metrics(Y.copy(), Y, np.zeros(3), min_strains=3)
    assert m["pearson_per_gene_median"] == pytest.approx(1.0)
    assert m["spearman_median"] == pytest.approx(1.0)
    assert m["pearson_per_strain_centered_median"] == pytest.approx(1.0)
    r = tr.column_pearson(-Y, Y, 3)
    assert np.allclose(r, -1.0)
    assert np.isnan(tr.column_pearson(Y[:2], Y[:2], 3)).all()  # fewer than min_n


def test_arm_config_validation(tr: ModuleType) -> None:
    with pytest.raises(ValueError, match="not a trainable"):
        tr.ArmConfig(name="x", datasets=["sameith_dm"])
    with pytest.raises(ValueError, match="kemmeren"):
        tr.ArmConfig(name="x", datasets=["messner"], doubles=True)


def test_balanced_sampler_draws_datasets_uniformly(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    store = world["store"]
    arm = tr.ArmConfig(
        name="a", datasets=["kemmeren", "messner", "caudal", "vaishnav"], doubles=False
    )
    view = tr.build_view(store, arm, mark_cap=2)
    train = view.positions(tr.ROLE_CODE[tr.Role.TRAIN])
    batches = tr.balanced_batches(view, train, batch_size=50, seed=7)
    assert len(batches) == int(np.ceil(len(train) / 50))
    drawn = np.concatenate(batches)
    assert set(drawn.tolist()) <= set(train.tolist())
    # every draw is a training position; datasets are near-uniform, not record-weighted
    counts = np.bincount(view.local_dataset[drawn], minlength=4)
    assert counts.sum() == len(batches) * 50
    assert counts.min() > 0.15 * counts.sum()
    # the same seed gives the same epoch; another seed another
    again = tr.balanced_batches(view, train, batch_size=50, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    other = tr.balanced_batches(view, train, batch_size=50, seed=8)
    assert not all(np.array_equal(a, b) for a, b in zip(batches, other, strict=True))


def test_two_training_steps_with_balanced_sampler(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    base = _cfg(tr, world["root"], "hlgauss")
    cfg = base.model_copy(
        update={"data": base.data.model_copy(update={"sampler": "balanced"})}
    )
    assert cfg.run_name == "joint_test_balanced_hlgauss_equal_s0"
    result = tr.train(
        cfg, world["store"], tr.h5_vector_source(cfg.paths), tr.MemoryLogger()
    )
    assert result.steps == 2
    assert np.isfinite(result.last_metrics["train/loss"])
    assert Path(result.checkpoint_dir).name == cfg.run_name


def test_max_steps_matches_steps_across_epochs(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    """A step-matched run takes ceil(max_steps / steps_per_epoch) epochs and stops at
    exactly max_steps, whatever max_epochs says.
    """
    root = world["root"]
    base = _cfg(tr, root, "mse")
    cfg = base.model_copy(
        update={
            "budget": base.budget.model_copy(
                update={"max_epochs": 1, "max_steps_per_epoch": 2, "max_steps": 5}
            )
        }
    )
    logger = tr.MemoryLogger()
    result = tr.train(cfg, world["store"], tr.h5_vector_source(cfg.paths), logger)
    assert result.steps == 5
    assert result.n_epochs == 3  # 2 + 2 + 1
    setup = [v for _, v in logger.rows if "setup/max_steps" in v]
    assert setup[0]["setup/max_steps"] == 5.0
    assert [v["epoch"] for _, v in logger.rows if "epoch" in v] == [1.0, 2.0, 3.0]


def test_max_steps_cut_by_time_budget_raises(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    root = world["root"]
    base = _cfg(tr, root, "mse")
    cfg = base.model_copy(
        update={
            "budget": base.budget.model_copy(
                update={
                    "max_steps_per_epoch": 1,
                    "max_steps": 10**6,
                    "time_budget_hours": 1e-6,
                    "time_margin_minutes": 0.0,
                }
            )
        }
    )
    with pytest.raises(RuntimeError, match="not step-matched"):
        tr.train(cfg, world["store"], tr.h5_vector_source(cfg.paths), tr.MemoryLogger())


@pytest.mark.parametrize(
    ("loss", "weighting"),
    [("hlgauss", "equal"), ("pinball", "equal"), ("mse", "equal"), ("mse", "records")],
)
def test_validation_loss_is_the_training_objective(
    tr: ModuleType, world: dict[str, Any], loss: str, weighting: str
) -> None:
    """``val/loss`` is the objective over the validation pass: the per-dataset losses
    (``val/<dataset>/loss``) combined by the run's weighting.
    """
    root = world["root"]
    cfg = _cfg(tr, root, loss, weighting)
    result = tr.train(
        cfg, world["store"], tr.h5_vector_source(cfg.paths), tr.MemoryLogger()
    )
    last = result.last_metrics
    per_ds = {
        k.split("/")[1]: v
        for k, v in last.items()
        if k.startswith("val/") and k.endswith("/loss") and k.count("/") == 2
    }
    assert per_ds, "no per-dataset validation loss logged"
    assert np.isfinite(last["val/loss"])
    assert np.isfinite(last["traineval/loss"])
    if weighting == "equal":
        assert last["val/loss"] == pytest.approx(
            np.mean(list(per_ds.values())), rel=1e-6
        )
    else:
        lo, hi = min(per_ds.values()), max(per_ds.values())
        assert lo - 1e-9 <= last["val/loss"] <= hi + 1e-9


@pytest.mark.parametrize("residual", ["plain", "rezero"])
def test_operator_residual_and_relative_norm_logged(
    tr: ModuleType, world: dict[str, Any], residual: str
) -> None:
    """``model.operator_residual`` picks the residual; ``operator/relative_norm`` is
    logged at every evaluation; the beta gates exist (and are logged) only under rezero.
    """
    base = _cfg(tr, world["root"], "hlgauss")
    cfg = base.model_copy(
        update={"model": base.model.model_copy(update={"operator_residual": residual})}
    )
    assert (
        tr.TrainConfig043.model_validate(
            {**base.model_dump(), "model": {**base.model.model_dump()}}
        ).model.operator_residual
        == "plain"
    )  # the default
    suffix = "_rezero" if residual == "rezero" else ""
    assert cfg.run_name == f"joint_test_hlgauss_equal{suffix}_s0"
    logger = tr.MemoryLogger()
    result = tr.train(cfg, world["store"], tr.h5_vector_source(cfg.paths), logger)
    last = result.last_metrics
    for k in (
        "operator/relative_norm",
        "operator/ffn_relative_norm",
        "operator/update_relative_norm",
    ):
        assert np.isfinite(last[k])
    if residual == "plain":
        assert last["operator/relative_norm"] > 0.0
        assert "train/beta_attn" not in last
    else:
        assert "train/beta_attn" in last and "train/beta_ffn" in last
        # two steps from zero-initialized gates: the branch is still nearly shut
        assert last["operator/relative_norm"] < 0.05


def test_nadal_sentinels_and_jackson_ratio(tr: ModuleType) -> None:
    y = np.array([1.5, -23.0, 20.0, -19.99, np.nan], np.float32)
    out, n = tr.mask_sentinels(y, tr.SPECS["nadal"].sentinel_abs)
    assert n == 2
    assert (
        np.isnan(out[[1, 2, 4]]).all()
        and out[0] == 1.5
        and out[3] == np.float32(-19.99)
    )
    same, zero = tr.mask_sentinels(y, None)
    assert zero == 0 and same is y
    # Jackson: log2((CPM + 1) / (CPM_ref + 1)); omitted keys are zero counts; zero on both
    # sides is missing
    r = tr.transform_target(
        "log2_cpm_ratio_to_reference",
        np.array([3.0, np.nan, np.nan, 1.0]),
        np.array([1.0, 1.0, np.nan, 1.0]),
        total_umis=1e6,
        total_umis_ref=1e6,
    )
    assert r[0] == pytest.approx(1.0)  # (3 + 1) / (1 + 1)
    assert r[1] == pytest.approx(-1.0)
    assert np.isnan(r[2])
    assert r[3] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="total_umis_ref"):
        tr.transform_target("log2_cpm_ratio_to_reference", y, y, total_umis=1.0)


def test_arm_all_lists_every_headed_dataset(tr: ModuleType) -> None:
    import yaml

    raw = yaml.safe_load((SCRIPTS.parent / "conf" / "arm" / "all.yaml").read_text())
    arm = tr.ArmConfig.model_validate(raw)
    assert tuple(arm.datasets) == tr.ALL_DATASETS
    assert {"hu", "jackson", "nadal"} <= set(arm.datasets)
    assert "sameith_dm" not in arm.datasets  # split B, read through Kemmeren


def test_setup_line_prints_records_per_dataset_and_sentinels(
    tr: ModuleType, world: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Messner's rows relabeled as Nadal-Ribelles: the new spec trains end to end and the
    setup line names its records and its masked sentinel cells.
    """
    store = world["store"]
    shorts = store.meta.datasets
    rec_dataset = store.rec_dataset.copy()
    rec_dataset[rec_dataset == shorts.index("messner")] = shorts.index("nadal")
    meta = store.meta.model_copy(
        update={"counts": {"nadal": {"sentinel_masked_cells": 17}}}
    )
    store2 = tr.PreparedStore043(
        **{**store.__dict__, "rec_dataset": rec_dataset, "meta": meta}
    )
    base = _cfg(tr, world["root"], "hlgauss")
    cfg = base.model_copy(
        update={
            "arm": tr.ArmConfig(
                name="all_test",
                datasets=["kemmeren", "nadal", "vaishnav"],
                doubles=True,
            )
        }
    )
    result = tr.train(cfg, store2, tr.h5_vector_source(cfg.paths), tr.MemoryLogger())
    out = capsys.readouterr().out
    n_train = int(
        (
            (store2.rec_dataset == shorts.index("nadal"))
            & (store2.rec_role == tr.ROLE_CODE[tr.Role.TRAIN])
        ).sum()
    )
    assert f"nadal {n_train}/" in out
    assert "(17 sentinel cells |v| >= 20 masked)" in out
    assert "val/nadal/loss" in result.last_metrics

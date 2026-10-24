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
        prep_version=tr.store_paths("v2").prep_version,
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
    result = tr.train(
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        logger,
    )
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
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        tr.MemoryLogger(),
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
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        tr.MemoryLogger(),
    )
    assert result.steps == 2
    assert np.isfinite(result.last_metrics["train/loss"])
    # one directory per job under the run name
    assert Path(result.checkpoint_dir).parent.name == cfg.run_name


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
    result = tr.train(
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        logger,
    )
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
        tr.train(
            cfg,
            world["store"],
            tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
            tr.MemoryLogger(),
        )


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
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        tr.MemoryLogger(),
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


def test_relative_norm_logged_and_no_operator_gates(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    """The operator is the post-norm residual only: ``operator/*`` norms are logged at
    every evaluation, there is no residual option and no beta gate.
    """
    cfg = _cfg(tr, world["root"], "hlgauss")
    assert "operator_residual" not in tr.ModelConfig.model_fields
    with pytest.raises(ValueError, match="operator_residual"):
        tr.ModelConfig.model_validate({"operator_residual": "plain"})
    assert cfg.run_name == "joint_test_hlgauss_equal_s0"
    logger = tr.MemoryLogger()
    result = tr.train(
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        logger,
    )
    last = result.last_metrics
    for k in (
        "operator/relative_norm",
        "operator/ffn_relative_norm",
        "operator/update_relative_norm",
    ):
        assert np.isfinite(last[k])
    assert last["operator/relative_norm"] > 0.0
    assert not any("beta" in k for _, row in logger.rows for k in row)


def test_checkpoint_config_drops_legacy_plain_key_and_refuses_rezero(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    cfg = _cfg(tr, world["root"], "hlgauss")
    dumped = cfg.model_dump(mode="json")
    legacy = {**dumped, "model": {**dumped["model"], "operator_residual": "plain"}}
    assert tr.checkpoint_config({"config": legacy}) == cfg
    rezero = {**dumped, "model": {**dumped["model"], "operator_residual": "rezero"}}
    with pytest.raises(ValueError, match="only plain exists"):
        tr.checkpoint_config({"config": rezero})


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
    result = tr.train(
        cfg,
        store2,
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        tr.MemoryLogger(),
    )
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


def test_finetune_from_stage1_checkpoint(
    tr: ModuleType, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 2: a one-dataset arm starts from the stage-1 model weights (its dataset's
    rows taken by name), with a fresh optimizer at its own lr, and is named after the
    stage-1 group.
    """
    root = world["root"]
    stage1 = _cfg(
        tr, root, "hlgauss"
    )  # joint_test: kemmeren, messner, caudal, vaishnav
    monkeypatch.setenv("SLURM_JOB_ID", "1111")
    r1 = tr.train(
        stage1,
        world["store"],
        tr.h5_vector_source(stage1.paths, tr.store_paths(stage1.data.store_version)),
        tr.MemoryLogger(),
    )
    ckpt = Path(r1.checkpoint_dir) / "last.pt"
    assert Path(r1.checkpoint_dir).name == "1111"
    s1 = torch.load(ckpt, weights_only=False)["model"]
    label = tr.stage1_label(str(ckpt))
    assert label == "joint_test_hlgauss_equal"
    base = _cfg(tr, root, "hlgauss")
    ft = base.model_copy(
        update={
            "arm": tr.ArmConfig(
                name="alone_kemmeren", datasets=["kemmeren"], doubles=True
            ),
            "optim": base.optim.model_copy(update={"lr": 3e-4}),
            "budget": base.budget.model_copy(update={"max_steps_per_epoch": 1}),
            "init_from": str(ckpt),
            "init_from_label": label,
        }
    )
    assert ft.run_name == "finetune_kemmeren_from_joint_test_hlgauss_equal_s0"
    view = tr.build_view(world["store"], ft.arm, ft.data.mark_cap)
    tables = [
        tr.h5_vector_source(ft.paths, tr.store_paths(ft.data.store_version))(
            e, view.used_ids[e]
        )
        for e in range(4)
    ]
    model = tr.build_model(ft, view, tables)
    logs = tr.init_from_checkpoint(model, ft, view)
    assert logs["setup/init_from_step"] == 2.0
    assert logs["setup/init_from_lr"] == stage1.optim.lr
    sd = model.state_dict()
    k = stage1.arm.datasets.index("kemmeren")
    for name in tr.DATASET_INDEXED:
        assert torch.equal(sd[name], s1[name][[k]])
    for name in (
        "operator.q_proj.weight",
        "trunk.layers.0.norm1.weight",
        "gene_head.mlp.0.weight",
    ):
        if name in s1:
            assert torch.equal(sd[name], s1[name])
    assert torch.equal(sd["operator.q_proj.weight"], s1["operator.q_proj.weight"])
    # end to end: the doubles are read each epoch, as in the alone run
    monkeypatch.setenv("SLURM_JOB_ID", "2222")
    logger = tr.MemoryLogger()
    r2 = tr.train(
        ft,
        world["store"],
        tr.h5_vector_source(ft.paths, tr.store_paths(ft.data.store_version)),
        logger,
    )
    assert r2.steps == 1
    assert "val/doubles/pearson_per_strain_centered_median" in r2.last_metrics
    setup = next(v for _, v in logger.rows if "setup/init_from_step" in v)
    assert setup["setup/init_from_epoch"] == 1.0
    # a dataset stage 1 never trained, or another architecture, refuses
    bad = ft.model_copy(
        update={
            "arm": tr.ArmConfig(
                name="alone_zelezniak", datasets=["zelezniak"], doubles=False
            )
        }
    )
    vbad = tr.build_view(world["store"], ft.arm, None)
    with pytest.raises(KeyError, match="not trained in stage 1"):
        tr.init_from_checkpoint(
            model, bad, tr.TrainingView(**{**vbad.__dict__, "datasets": ["zelezniak"]})
        )
    other = ft.model_copy(
        update={"model": ft.model.model_copy(update={"num_layers": 3})}
    )
    with pytest.raises(ValueError, match="model section differs"):
        tr.init_from_checkpoint(model, other, view)
    with pytest.raises(ValueError, match="init_from_label without init_from"):
        tr.TrainConfig043.model_validate({**base.model_dump(), "init_from_label": "x"})


def test_mark_cap_report_counts_truncation(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    store = world["store"]
    arm = tr.ArmConfig(
        name="a", datasets=["kemmeren", "caudal", "vaishnav"], doubles=False
    )
    n_caudal = int((store.rec_dataset == list(tr.SPECS).index("caudal")).sum())
    capped = tr.build_view(store, arm, mark_cap=2)
    line, logs = tr.mark_cap_report(capped)
    # every Caudal record carries 3 accessory marks: cap 2 drops one mark each
    assert logs["setup/marks_max"] == 3.0
    assert logs["setup/records_truncated"] == float(n_caudal)
    assert logs["setup/marks_dropped"] == float(n_caudal)
    assert (
        f"{n_caudal} of {len(capped.rec)} records truncated {{'caudal': {n_caudal}}}"
        in line
    )
    free = tr.build_view(store, arm, mark_cap=None)
    line, logs = tr.mark_cap_report(free)
    assert logs["setup/records_truncated"] == 0.0 and "cap none" in line
    batch, _ = tr.make_batch(free, np.arange(len(free.rec)).tolist())
    assert int(batch.marks.valid.sum(1).max()) == 3


# --------------------------------------------------------------------------- #
# Ontology conditioning (prep_v4, 2026-10-23)
# --------------------------------------------------------------------------- #
def _onto_store(tr: ModuleType, world: dict[str, Any], directory: Path) -> Any:
    """The synthetic store plus a per-dataset annotation over a tiny vocabulary."""
    from torchcell.data.experiment_ontology import (
        ExperimentOntologyVocabulary,
        OntologyNode,
    )

    store = world["store"]
    vocab = ExperimentOntologyVocabulary(
        nodes=[
            OntologyNode(name="Experiment", kind="class"),
            OntologyNode(name="Expr", kind="class", parents=["Experiment"]),
            OntologyNode(name="Prot", kind="class", parents=["Experiment"]),
            OntologyNode(name="Temperature.value", kind="numeric", parents=["Expr"]),
        ]
    )
    profiles = np.asarray([[1, 1, 0, 0.3], [1, 0, 1, 0], [1, 1, 0, 0.25]], np.float32)
    shorts = store.meta.datasets
    by_short = {"messner": 1, "caudal": 2}
    rec_onto = np.asarray(
        [by_short.get(shorts[d], 0) for d in store.rec_dataset], np.int32
    )
    vocab.to_json(str(directory / "v.json"))
    import hashlib

    sha = hashlib.sha256((directory / "v.json").read_bytes()).hexdigest()
    meta = store.meta.model_copy(
        update={
            "prep_version": "prep_v4",
            "ontology_vocab_sha256": sha,
            "ontology_n_nodes": len(vocab),
            "ontology_n_profiles": 3,
        }
    )
    out = tr.PreparedStore043(
        **{
            **store.__dict__,
            "meta": meta,
            "rec_onto": rec_onto,
            "onto_profiles": profiles,
            "ontology_vocabulary": vocab,
        }
    )
    out.save(str(directory / "prep"))
    return tr.PreparedStore043.load(str(directory / "prep"))


def test_ontology_arm_trains_leaves_a_dataset_out_resumes_and_finetunes(
    tr: ModuleType, world: dict[str, Any], tmp_path: Path
) -> None:
    store = _onto_store(tr, world, tmp_path)
    assert store.ontology_vocabulary is not None and store.rec_onto.size
    root = world["root"]
    base = _cfg(tr, root, "hlgauss").model_dump()
    base["model"]["conditioning"] = {"source": "ontology", "state_dim": 4}
    base["paths"]["prep_dir"] = str(tmp_path / "prep")
    base["paths"]["checkpoint_root"] = str(tmp_path / "ckpt")
    cfg = tr.TrainConfig043.model_validate(base)
    assert cfg.arm_label == "joint_test_onto_hlgauss_equal"
    src = tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version))
    res = tr.train(cfg, store, src, tr.MemoryLogger())
    assert res.steps == 2 and np.isfinite(res.last_metrics["train/loss"])
    sd = torch.load(Path(res.checkpoint_dir) / "last.pt", weights_only=False)["model"]
    assert not any(
        k.startswith(("dataset_token", "env_token", "type_token")) for k in sd
    )
    assert "gene_bias" not in sd and "gene_bias_map.weight" in sd
    # leave-dataset-out: messner never in a training batch, still scored
    lodo = {**base, "arm": {**base["arm"], "exclude_from_training": ["messner"]}}
    cfg_l = tr.TrainConfig043.model_validate(lodo)
    assert cfg_l.arm_label == "joint_test_lodo_messner_onto_hlgauss_equal"
    view = tr.build_view(store, cfg_l.arm, 2)
    train_pos = view.positions(tr.ROLE_CODE[tr.Role.TRAIN])
    kept = train_pos[view.train_mask[train_pos]]
    assert (view.local_dataset[kept] != view.datasets.index("messner")).all()
    assert len(kept) < len(train_pos)
    logger = tr.MemoryLogger()
    res_l = tr.train(cfg_l, store, src, logger)
    trained = {k for _, row in logger.rows for k in row if k.startswith("train/")}
    assert "train/messner/loss" not in trained
    assert any(
        k.startswith(("val/messner/", "traineval/messner/")) for k in res_l.last_metrics
    )
    # name conditioning cannot leave a dataset out
    bad = {**lodo, "model": {**base["model"], "conditioning": {"source": "name"}}}
    with pytest.raises(ValueError, match="ontology"):
        tr.TrainConfig043.model_validate(bad)
    # resume: a second run with the same id continues from epoch 1, step 2
    rid = {**base, "resume_id": "r1"}
    rid["budget"] = {**base["budget"], "max_epochs": 1}
    r1 = tr.train(tr.TrainConfig043.model_validate(rid), store, src, tr.MemoryLogger())
    assert r1.steps == 2 and Path(r1.checkpoint_dir).name == "r1"
    rid2 = {**rid, "budget": {**rid["budget"], "max_epochs": 2}}
    with pytest.raises(ValueError, match="another config"):
        tr.train(tr.TrainConfig043.model_validate(rid2), store, src, tr.MemoryLogger())
    r1b = tr.train(tr.TrainConfig043.model_validate(rid), store, src, tr.MemoryLogger())
    assert r1b.steps == 2 and r1b.epoch_seconds == []  # nothing left to run
    # stage 2 from the ontology checkpoint: no dataset-indexed rows to remap
    ft = {
        **base,
        "init_from": str(Path(res.checkpoint_dir) / "last.pt"),
        "arm": {"name": "alone_kemmeren", "datasets": ["kemmeren"], "doubles": True},
    }
    cfg_ft = tr.TrainConfig043.model_validate(
        {**ft, "init_from_label": "joint_test_onto_hlgauss_equal"}
    )
    view_ft = tr.build_view(store, cfg_ft.arm, 2)
    model = tr.build_model(
        cfg_ft, view_ft, [src(e, view_ft.used_ids[e]) for e in range(4)]
    )
    tr.init_from_checkpoint(model, cfg_ft, view_ft)


# --------------------------------------------------------------------------- #
# GilaHyper sweep knobs (2026-10-24)
# --------------------------------------------------------------------------- #
def test_layer_pattern_sets_layers_and_widths_refuse_loudly(tr: ModuleType) -> None:
    m = tr.ModelConfig(layer_pattern=["S", "M", "M", "S", "S"])
    assert m.num_layers == 5 and m.mask_layers == [1, 2]
    assert m.pattern_label == "SMMSS"
    assert tr.ModelConfig().pattern_label == "SMSS"  # the default trunk, 0-based [1]
    assert tr.ModelConfig(layer_pattern=["S"] * 4).mask_layers == []
    with pytest.raises(ValueError, match="not divisible by num_heads 9"):
        tr.ModelConfig(hidden_dim=160)
    for hd in (144, 288):
        assert tr.ModelConfig(hidden_dim=hd).hidden_dim // 9 in (16, 32)


def test_sweep_arm_yamls_validate(tr: ModuleType) -> None:
    import yaml

    conf = SCRIPTS.parent / "conf" / "model"
    want = {
        "pattern_smss.yaml": "SMSS",
        "pattern_smmss.yaml": "SMMSS",
        "pattern_mmmm.yaml": "MMMM",
        "pattern_ssss.yaml": "SSSS",
        "pattern_msmsms.yaml": "MSMSMS",
    }
    for name, label in want.items():
        over = yaml.safe_load((conf / name).read_text())
        m = tr.ModelConfig.model_validate(over)
        assert m.pattern_label == label, name


@pytest.mark.parametrize(
    "over",
    [
        {"loss": {"kind": "mle"}},
        {"loss": {"kind": "binned_quantile"}},
        {"data": {"target_transform": "quantile_normal"}},
        {"data": {"target_transform": "none"}, "loss": {"kind": "mse"}},
        {"model": {"layer_pattern": ["M", "S", "M"], "trunk_ffn_mult": 2}},
        {"model": {"hidden_dim": 32, "head_hidden": 16}},
        {"optim": {"schedule": "cosine", "warmup_steps": 1}},
    ],
    ids=[
        "mle",
        "binned_quantile",
        "quantile_normal",
        "none",
        "pattern_ffn2",
        "dims",
        "cos",
    ],
)
def test_two_training_steps_per_sweep_knob(
    tr: ModuleType, world: dict[str, Any], tmp_path: Path, over: dict[str, Any]
) -> None:
    base = _cfg(tr, world["root"], "hlgauss").model_dump()
    for sec, vals in over.items():
        base[sec] = {**base[sec], **vals}
    base["paths"]["checkpoint_root"] = str(tmp_path / "ckpt")
    base["wandb"]["label_knobs"] = True
    cfg = tr.TrainConfig043.model_validate(base)
    res = tr.train(
        cfg,
        world["store"],
        tr.h5_vector_source(cfg.paths, tr.store_paths(cfg.data.store_version)),
        tr.MemoryLogger(),
    )
    last = res.last_metrics
    assert res.steps == 2 and np.isfinite(last["train/loss"])
    assert np.isfinite(last["val/loss"]) and "val/kemmeren/loss" in last
    assert "val/expression/pearson_per_gene_median" in last
    ckpt = Path(res.checkpoint_dir)
    if cfg.data.target_transform != "zscore":
        assert (ckpt / "target_transform.json").exists()
    if cfg.loss.kind == "binned_quantile":
        assert (ckpt / "quantile_bins.json").exists()
    assert cfg.arm_label.startswith("joint_test_")


def test_quantile_normal_targets_invert_to_the_legacy_z(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    arm = tr.ArmConfig(name="a", datasets=["kemmeren", "caudal", "vaishnav"])
    legacy = tr.build_view(world["store"], arm, 2)
    assert legacy.T is None and legacy.target_map is None  # zscore by dataset untouched
    v = tr.build_view(world["store"], arm, 2, "quantile_normal", "dataset")
    assert v.T is not None and v.target_map is not None
    assert v.group_names == ["kemmeren", "caudal", "vaishnav"]
    g = np.nonzero(v.level_gene)[0]
    yr = world["store"].rec_yrow[v.rec[g]]
    back = v.target_map.t_to_z(v.T[yr], v.group[g], v.row_mu[g], v.row_sd[g])
    fin = np.isfinite(v.Z[yr])
    inside = fin & (np.abs(v.T[yr]) < 3.4)  # tails clip to the grid ends by design
    np.testing.assert_allclose(back[inside], v.Z[yr][inside], atol=1e-4)


def _profile_store(tr: ModuleType, world: dict[str, Any], directory: Path) -> Any:
    """Kemmeren and Caudal share an expression profile; Messner and Vaishnav do not."""
    from torchcell.data.experiment_ontology import (
        ExperimentOntologyVocabulary,
        OntologyNode,
    )

    store = world["store"]
    vocab = ExperimentOntologyVocabulary(
        nodes=[
            OntologyNode(name="Experiment", kind="class"),
            OntologyNode(name="Phenotype", kind="class", parents=["Experiment"]),
            OntologyNode(name="ExprPhenotype", kind="class", parents=["Phenotype"]),
            OntologyNode(name="ProtPhenotype", kind="class", parents=["Phenotype"]),
            OntologyNode(name="SortSeqPhenotype", kind="class", parents=["Phenotype"]),
            OntologyNode(name="Media", kind="class", parents=["Experiment"]),
        ]
    )
    profiles = np.asarray(
        [
            [1, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        np.float32,
    )
    shorts = store.meta.datasets
    by_short = {"messner": 1, "vaishnav": 2, "caudal": 3}  # caudal: other medium
    rec_onto = np.asarray(
        [by_short.get(shorts[d], 0) for d in store.rec_dataset], np.int32
    )
    vocab.to_json(str(directory / "v.json"))
    import hashlib

    sha = hashlib.sha256((directory / "v.json").read_bytes()).hexdigest()
    meta = store.meta.model_copy(
        update={
            "prep_version": "prep_v4",
            "ontology_vocab_sha256": sha,
            "ontology_n_nodes": len(vocab),
            "ontology_n_profiles": 4,
        }
    )
    return tr.PreparedStore043(
        **{
            **store.__dict__,
            "meta": meta,
            "rec_onto": rec_onto,
            "onto_profiles": profiles,
            "ontology_vocabulary": vocab,
        }
    )


def test_profile_grouping_shares_statistics_and_weighting(
    tr: ModuleType, world: dict[str, Any], tmp_path: Path
) -> None:
    store = _profile_store(tr, world, tmp_path)
    arm = tr.ArmConfig(
        name="a", datasets=["kemmeren", "messner", "caudal", "vaishnav"], doubles=True
    )
    v = tr.build_view(store, arm, 2, "zscore", "profile", "profile")
    assert v.group_names == ["ExprPhenotype", "ProtPhenotype", "SortSeqPhenotype"]
    ds = {
        d: int(v.group[np.nonzero(v.local_dataset == i)[0][0]])
        for i, d in enumerate(arm.datasets)
    }
    assert ds["kemmeren"] == ds["caudal"] != ds["messner"]
    assert v.weight_group is not None and v.weight_group.tolist() == [0, 1, 0, 2]
    # profile weighting: kemmeren and caudal pool, then three profiles weigh equally
    sums = torch.tensor([2.0, 3.0, 6.0, 4.0])
    counts = torch.tensor([1.0, 1.0, 2.0, 1.0])
    total, means = tr.combine_loss(
        sums, counts, torch.ones(4), "equal", torch.from_numpy(v.weight_group)
    )
    assert float(total) == pytest.approx(((2 + 6) / 3 + 3.0 + 4.0) / 3)
    assert means.tolist() == [2.0, 3.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="ontology annotation"):
        tr.build_view(world["store"], arm, 2, "zscore", "profile")


def test_cosine_schedule_and_resume_reads_missing_sweep_keys_as_defaults(
    tr: ModuleType, world: dict[str, Any]
) -> None:
    o = tr.OptimConfig(lr=1.0, warmup_steps=10, schedule="cosine", min_lr_frac=0.1)
    assert tr._lr_at(4, o, 110) == pytest.approx(0.5)
    assert tr._lr_at(10, o, 110) == pytest.approx(1.0)
    assert tr._lr_at(60, o, 110) == pytest.approx(0.55)
    assert tr._lr_at(500, o, 110) == pytest.approx(0.1)
    assert tr._lr_at(500, tr.OptimConfig(lr=1.0), None) == 1.0
    cfg = _cfg(tr, world["root"], "hlgauss")
    old = cfg.model_dump(mode="json")
    for (sec, key), _ in tr.SWEEP_DEFAULTS.items():
        old[sec].pop(key)
    assert tr.config_matches(old, cfg)
    changed = tr.TrainConfig043.model_validate(
        {**cfg.model_dump(), "optim": {**cfg.optim.model_dump(), "schedule": "cosine"}}
    )
    assert not tr.config_matches(old, changed)

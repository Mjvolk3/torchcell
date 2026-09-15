# experiments/026-metabolism-flux/scripts/probe_genotype_reference_init.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_genotype_reference_init
r"""The init check on REAL dataset genotypes, with and without a per-genotype reference.

TWO THINGS THE EARLIER INIT CHECKS COULD NOT SEE, and both are why this script exists.

**They ran on four WILD TYPES.** `probe_warmstart_init.py` forwards random tokens with no
perturbation, so every genotype it scored was the unperturbed cell. The coordinator's read of
job 1946 shows the training-batch active-row balance max was already about 1.0 at epoch 0
while the init probe read 0.03, which can only mean the DELETION genotypes fail the rule from
the start. A check that never deletes anything cannot see that.

**They could not see the across-genotype flux spread.** With one wild-type reference the
proposal is the same vector for every row, so the flux is near-constant across genotypes and
the head reading it is close to a constant predictor. That spread is the diagnostic for
whether the reference now carries the genotype, and it is only defined over real genotypes.

So this loads 32 real genotypes from the 027 dataset through the trainer's own loader, builds
the cell's layer through the real build path, and reports the five statistics, the
across-genotype flux standard deviation, and the table's fallback rate, with the
single-reference head as the paired control at the same seed on the same batch.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_genotype_reference_init.py

Writes ``experiments/026-metabolism-flux/results/probe_genotype_reference_init.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import torch
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
BX_SCRIPTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")
BX_CONF = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "conf")
SCRIPTS_026 = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
#: The YPD table. The shipped-bounds table it replaced is kept on disk and is a measurement
#: of the medium rather than of the genotypes: on the GEM's own boundary every one of this
#: dataset's strains has growth exactly 0, because almost all of them delete ARO4 and ARO7
#: and nothing supplies aromatic amino acids.
TABLE_PATH = osp.join(
    DATA_ROOT, "data/torchcell/reference_flux/reference_flux_table_ypd.npz"
)
sys.path.insert(0, BX_SCRIPTS)
sys.path.insert(0, SCRIPTS_026)

from train_bx import build_dataset, load_config, resolve_pinned_test  # noqa: E402
from train_flux import ARMS, build_flux_layer  # noqa: E402

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    gene_removal_indices,
)

TURNOVER_THRESHOLD = 1.0e-3
FLUX_ACTIVE_RELATIVE_THRESHOLD = 1.0e-6


def statistics(layer, flux: dict[str, torch.Tensor]) -> dict[str, float]:
    """The five statistics, plus the across-genotype spread the table is judged by."""
    v = flux["v"]
    scale = float(layer.config.flux_scale)
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = residual / (omega + layer.config.balance_epsilon)
    active = omega > TURNOVER_THRESHOLD * scale
    selected = ratio[active]
    pre = flux["v_pre_projection"]
    out = {
        "balance_active_max": float(selected.max()) if selected.numel() else float("nan"),
        "balance_active_p95": (
            float(torch.quantile(selected, 0.95)) if selected.numel() else float("nan")
        ),
        "balance_active_rows": float(active.float().sum(dim=1).mean()),
        "flux_active_frac": float(
            (v.abs() > FLUX_ACTIVE_RELATIVE_THRESHOLD * scale).float().mean()
        ),
        "proposal_distance_rel": float(
            ((v - pre).norm(dim=1) / pre.norm(dim=1).clamp(min=1e-12)).mean()
        ),
        # THE DIAGNOSTIC. Per reaction, the standard deviation across genotypes, then its
        # mean and its ratio to the mean flux magnitude. A head whose flux does not move
        # with the genotype cannot predict a genotype-dependent phenotype, whatever its
        # feasibility says.
        "across_genotype_flux_sd_mean": float(v.std(dim=0).mean()),
        "across_genotype_flux_sd_max": float(v.std(dim=0).max()),
        "across_genotype_flux_sd_over_mean_abs": float(
            v.std(dim=0).mean() / v.abs().mean().clamp(min=1e-12)
        ),
        "across_genotype_reference_sd_mean": float(
            flux["warm_reference"].std(dim=0).mean()
        )
        if "warm_reference" in flux
        else None,
    }
    for key in (
        "feas_warm_alpha",
        "feas_warm_proposal_rel",
        "feas_warm_reference_clipped_frac",
        "feas_warm_reference_active_frac",
        "feas_box_violation_frac",
        "feas_thermo_violation_frac",
    ):
        if key in flux:
            out[key] = float(flux[key].detach())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotypes", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table", default=TABLE_PATH)
    parser.add_argument("--medium", default="YPD")
    parser.add_argument("--arm-file", default="arm_flux_warmstart_boxforce_nominal.yaml")
    parser.add_argument("--out", default="probe_genotype_reference_init.json")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0; the k_cat table depends on it.")

    started = time.time()
    torch.set_num_threads(8)
    import yaml

    with open(osp.join(BX_CONF, args.arm_file)) as handle:
        arm_file = yaml.safe_load(handle)

    cfg = load_config()
    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=args.genotypes,
        random_seed=0,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        # `deletion_indices` needs its own batch vector, exactly as the trainer collates it;
        # without it the helper finds the field but not its row assignment.
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    batch = next(iter(dm.all_dataloader()))
    # THE SAME HELPER THE MODEL USES, so the availability gate, the reference key and this
    # probe read one gene set. It returns which index it took, and the answer is recorded:
    # a batch built before `deletion_indices` existed falls back to the perturbation index,
    # and every number below would then be measured on a layer that reads the betaxanthin
    # cassette as a knockout.
    gate_indices, gate_batch, gate_source = gene_removal_indices(batch)
    print(f"gene removal source: {gate_source}; {int(gate_indices.numel())} entries")
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    print(
        f"batch: {int(batch.num_graphs)} genotypes, "
        f"{int(batch['gene'].perturbation_indices.numel())} perturbation entries"
    )

    rows: list[dict[str, object]] = []
    for name, table in (("single_wild_type_reference", None), ("per_genotype", args.table)):
        arm = dict(ARMS[arm_file["base_arm"]])
        arm["parameterization"] = arm_file["flux_parameterization"]
        arm["kcat_tier"] = "predicted"
        arm["thermo_source"] = "union_external"
        # THE SAME MEDIUM THE REFERENCE WAS SOLVED ON. A reference solved on YPD and
        # clipped into a box built on the shipped boundary is two different cells, and the
        # shipped boundary is the one on which every strain in this dataset is dead.
        arm["media_name"] = args.medium if args.medium != "none" else None
        torch.manual_seed(args.seed)
        layer, meta = build_flux_layer(arm, gene_ids, args.hidden)
        assert layer is not None
        layer.config.availability_fold = "normalized_softmin"
        layer.apply_thermo_exempt_list(
            osp.join(RESULTS_DIR, "thermo_exempt_candidates.json")
        )
        layer.apply_log_c_init(osp.join(RESULTS_DIR, "thermo_consistent_log_c.json"))
        if table is not None:
            # ONE CALL, because setting the record without the background gene set misses
            # every row and falls back to the wild type on the whole batch, silently.
            n_rows = layer.attach_reference_table(table)
            assert layer.reference_background_genes, (
                f"the table at {table} carries no background_genes in its provenance, so "
                "the key would name the engineered cassette and miss every row."
            )
            print(f"   attached {n_rows} rows, background {sorted(layer.reference_background_genes)}")
        layer.n_reference_lookups = 0
        layer.n_reference_fallbacks = 0

        torch.manual_seed(args.seed)
        h = torch.randn(int(batch.num_graphs), len(gene_ids), args.hidden)
        context = torch.randn(int(batch.num_graphs), args.hidden)
        with torch.no_grad():
            reference_rows = layer.genotype_reference(
                int(batch.num_graphs), gate_indices, gate_batch
            )
            flux = layer(h, context, gate_indices, gate_batch)
            flux["warm_reference"] = reference_rows
        stats = statistics(layer, flux)
        wild = layer.v_reference
        moved = (
            (reference_rows - wild).norm(dim=1) / wild.norm().clamp(min=1e-12)
        )
        row = {
            "name": name,
            "reference_table": table,
            "coverage_warm_start_reference_table": layer.coverage_report()[
                "warm_start_reference_table"
            ],
            "media": meta.get("media"),
            "n_genotypes": int(batch.num_graphs),
            "deletion_effect": {
                "fraction_reference_differs_from_wild_type_by_1pct": float(
                    (moved > 0.01).float().mean()
                ),
                "median_relative_difference": float(moved.median()),
                "max_relative_difference": float(moved.max()),
            },
            "statistics": stats,
        }
        rows.append(row)
        print(f"-- {name}")
        print(json.dumps(stats, indent=2))
        print(json.dumps(row["deletion_effect"], indent=2))

    payload = {
        "definitions": {
            "across_genotype_flux_sd_mean": (
                "per reaction, the standard deviation of the flux across the batch's "
                "genotypes, averaged over reactions. Near zero means the head emits one "
                "vector whatever the genotype, which is what made job 1946 a constant "
                "predictor."
            ),
            "fraction_reference_differs_from_wild_type_by_1pct": (
                "how many of the batch's genotypes get a reference that is actually "
                "different from the wild type. A table that changed nothing would train "
                "exactly as the single-reference head did."
            ),
        },
        "genotypes": int(batch.num_graphs),
        "seed": args.seed,
        "arm_file": args.arm_file,
        "medium": args.medium,
        "gene_removal_source": gate_source,
        "n_removal_entries": int(gate_indices.numel()),
        "n_perturbation_entries": int(batch["gene"].perturbation_indices.numel()),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "rows": rows,
    }
    out_path = osp.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

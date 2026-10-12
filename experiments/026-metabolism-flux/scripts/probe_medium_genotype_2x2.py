# experiments/026-metabolism-flux/scripts/probe_medium_genotype_2x2.py
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/probe_medium_genotype_2x2
r"""Which axis carries the 50x: the MEDIUM or the GENOTYPES.

THE CONFOUND THIS SCRIPT EXISTS TO BREAK. The warm-start init check reads
``balance_active_max`` **0.0306** on four wild types on the SHIPPED yeast-GEM bounds
(job 1937) and **1.5898** on 32 real deletion genotypes on YPD (job 1973). Two things
changed at once, so neither number attributes anything:

* the MEDIUM. YPD opens 20 amino-acid exchanges, wild-type growth 0.0858 to 0.5352, so the
  box is wider and the pFBA reference the layer solves inside its own build step is a
  different vector entirely;
* the GENOTYPES. A deletion closes its reactions through the availability gate, so the
  per-genotype box differs from the wild-type box only through that gate.

The 2x2 crosses them at one seed with ONE wild-type reference in every cell, so the
reference never varies within the table and the only moving parts are the medium and the
gate. ``wild_type_32`` is the row that isolates the genotype axis exactly: it is the same
``h`` and the same ``context`` as ``deletion_32``, with the removal index emptied, so the
two differ in nothing but the gate. ``wild_type_4`` is carried only to tie the table back
to job 1937, whose four-row batch draws a different ``h``.

Each cell also carries the CLIP-COST DECOMPOSITION, which is what says WHERE a wider or
narrower box bites: ``n_sign_forbidden`` is the directional tightening excluding the
reference's direction outright, ``n_over_ceiling`` is the enzyme-capacity ceiling cutting
its magnitude, and ``clip_cost_rel`` is the two together as a fraction of the reference's
norm.

The exemption list and the concentration witness are ARGUMENTS rather than constants,
because both were solved on the shipped bounds against the shipped FBA flux; if the medium
is what carries the result, the honest rerun is with a YPD-solved pair.

Run from the repo root, on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/probe_medium_genotype_2x2.py

Writes ``experiments/026-metabolism-flux/results/probe_medium_genotype_2x2.json``.
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


def flux_statistics(layer, flux: dict[str, torch.Tensor]) -> dict[str, float]:
    """The five statistics of the pre-registered read, plus the proposal distance."""
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
        "across_genotype_flux_sd_mean": float(v.std(dim=0).mean())
        if v.shape[0] > 1
        else 0.0,
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


def clip_decomposition(
    layer,
    h: torch.Tensor,
    context: torch.Tensor,
    removal_indices: torch.Tensor,
    removal_batch: torch.Tensor,
) -> dict[str, object]:
    """Does the genotype's own box contain the wild-type reference, and what does it cost?

    Built exactly as ``FluxLayer.forward`` builds it, with the removal index the model
    passes, so the wild-type row and the deletion row differ in the gate and in nothing
    else. Three quantities, because they have three different remedies: a sign-forbidden
    reaction is the directional tightening, an over-ceiling reaction is the enzyme-capacity
    bound, and the clip cost is the two together.
    """
    with torch.no_grad():
        gamma = layer.gene_availability(h, removal_indices, removal_batch)
        c_j = layer.reaction_availability(gamma)
        e_u = (
            layer.unit_abundance(layer.gene_abundance(h, gamma))
            if layer.config.use_abundance_head
            else None
        )
        lb, ub, capacity = layer.dynamic_box(c_j, e_u)
        delta_g, _, _ = layer.reaction_potential(context)
        tight_lb, tight_ub, emptied = layer._thermo_tighten(lb, ub, delta_g)
        reference = layer.v_reference
        carrying = reference.abs() > 1e-9
        out: dict[str, object] = {
            "reference_n_carrying_flux": int(carrying.sum()),
            "reference_max_abs_flux": float(reference.abs().max()),
            "reference_norm": float(reference.norm()),
            "thermo_emptied_box_frac": float(emptied.float().mean()),
            "mean_reaction_availability": float(c_j.mean()),
            "n_reactions_gated_to_zero": float((c_j < 1e-9).float().sum(dim=1).mean()),
        }
        if capacity is not None:
            gpr = layer.has_gpr
            out["capacity_median_over_gpr_reactions"] = float(capacity[:, gpr].median())
        for tag, (low, high) in {
            "box": (lb, ub),
            "boxforce": (tight_lb, tight_ub),
        }.items():
            clipped = torch.minimum(
                torch.maximum(reference.unsqueeze(0).expand_as(low), low), high
            )
            forbidden = ((reference.unsqueeze(0) > 1e-9) & (high < 1e-9)) | (
                (reference.unsqueeze(0) < -1e-9) & (low > -1e-9)
            )
            ceiling = torch.where(reference.unsqueeze(0) > 0, high, -low)
            over = (
                carrying.unsqueeze(0)
                & (~forbidden)
                & (reference.abs().unsqueeze(0) > ceiling)
            )
            excess = (reference.abs().unsqueeze(0) / ceiling.clamp(min=1e-12))[over]
            out[tag] = {
                "clip_cost_rel": float(
                    ((clipped - reference).norm(dim=1) / reference.norm()).mean()
                ),
                "n_sign_forbidden": float(forbidden.float().sum(dim=1).mean()),
                "n_over_ceiling": float(over.float().sum(dim=1).mean()),
                "median_excess_over_ceiling": (
                    float(excess.median()) if excess.numel() else None
                ),
                "over_ceiling_reactions": _named(layer, reference, ceiling, over),
            }
    return out


def _named(layer, reference, ceiling, over, limit: int = 10) -> list[dict[str, object]]:
    columns = torch.nonzero(over.any(dim=0), as_tuple=False).flatten()
    entries = [
        {
            "reaction": layer.rxn_ids[j],
            "reference_flux": float(reference[j]),
            "ceiling": float(ceiling[:, j].median()),
            "excess": float(
                reference[j].abs() / ceiling[:, j].median().clamp(min=1e-12)
            ),
            "norm_share": float(reference[j] ** 2 / reference.pow(2).sum()),
        }
        for j in columns.tolist()
    ]
    entries.sort(key=lambda row: -float(row["norm_share"]))
    return entries[:limit]


def _gem_gene_ids() -> list[str]:
    """The GEM's own gene list, which is the universe every init check before job 1974 used."""
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    return list(gem.catalytic_units.gene_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genotypes", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm-file", default="arm_flux_warmstart_boxforce_nominal.yaml")
    parser.add_argument(
        "--media",
        nargs="+",
        default=["none", "YPD"],
        help="The medium axis. 'none' is the shipped yeast-GEM boundary.",
    )
    parser.add_argument(
        "--exempt",
        default="thermo_exempt_candidates.json",
        help=(
            "The thermodynamic exemption list, under results/. The shipped one was solved "
            "on the SHIPPED bounds against the shipped FBA flux."
        ),
    )
    parser.add_argument(
        "--log-c",
        default="thermo_consistent_log_c.json",
        help="The concentration witness, under results/. Same caveat as --exempt.",
    )
    parser.add_argument(
        "--exempt-by-medium",
        default=None,
        help=(
            "JSON mapping medium name to an exemption file under results/, for the rerun "
            "with medium-consistent files. Overrides --exempt for the media it names."
        ),
    )
    parser.add_argument(
        "--log-c-by-medium",
        default=None,
        help="The same mapping for the concentration witness.",
    )
    parser.add_argument(
        "--gem-universe-control",
        action="store_true",
        help=(
            "Also build a second layer per medium on the GEM's OWN gene list and score four "
            "wild types on it. Every init check before job 1974 used that universe "
            "(probe_warmstart_init._gene_ids), while a real cell passes the transformer's "
            "6,607-token universe, and the random heads are initialized over whichever one "
            "they are given. This row is what says how much of the gap to job 1937 is the "
            "universe rather than the medium or the genotypes."
        ),
    )
    parser.add_argument("--out", default="probe_medium_genotype_2x2.json")
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0; the k_cat table depends on it.")

    started = time.time()
    torch.set_num_threads(8)
    import yaml

    with open(osp.join(BX_CONF, args.arm_file)) as handle:
        arm_file = yaml.safe_load(handle)

    exempt_map = json.loads(args.exempt_by_medium) if args.exempt_by_medium else {}
    log_c_map = json.loads(args.log_c_by_medium) if args.log_c_by_medium else {}

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
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    batch = next(iter(dm.all_dataloader()))
    gate_indices, gate_batch, gate_source = gene_removal_indices(batch)
    if gate_source != "deletion_indices":
        raise SystemExit(
            f"the batch resolved {gate_source!r}; every number in this table would then be "
            "measured on a layer that reads the betaxanthin cassette as a knockout."
        )
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    n = int(batch.num_graphs)
    empty = torch.zeros(0, dtype=torch.long)
    print(
        f"gene removal source: {gate_source}; {int(gate_indices.numel())} removal entries "
        f"against {int(batch['gene'].perturbation_indices.numel())} perturbation entries "
        f"over {n} genotypes"
    )

    sources = {
        "wild_type_4": (4, empty, empty),
        "wild_type_32": (n, empty, empty),
        "deletion_32": (n, gate_indices, gate_batch),
    }

    def build(universe: list[str], medium: str, exempt_file: str, log_c_file: str):
        arm = dict(ARMS[arm_file["base_arm"]])
        arm["parameterization"] = arm_file["flux_parameterization"]
        arm["kcat_tier"] = "predicted"
        arm["thermo_source"] = "union_external"
        arm["media_name"] = None if medium == "none" else medium
        torch.manual_seed(args.seed)
        built, built_meta = build_flux_layer(arm, universe, args.hidden)
        assert built is not None
        built.config.availability_fold = "normalized_softmin"
        built.apply_thermo_exempt_list(osp.join(RESULTS_DIR, exempt_file))
        built.apply_log_c_init(osp.join(RESULTS_DIR, log_c_file))
        return built, built_meta

    rows: list[dict[str, object]] = []
    for medium in args.media:
        exempt_file = exempt_map.get(medium, args.exempt)
        log_c_file = log_c_map.get(medium, args.log_c)
        layer, meta = build(gene_ids, medium, exempt_file, log_c_file)
        media_meta = meta.get("media", {})
        print(
            f"== medium {medium}: growth "
            f"{media_meta.get('wild_type_growth_after', media_meta.get('note'))}; "
            f"reference norm {float(layer.v_reference.norm()):.4f}, "
            f"max {float(layer.v_reference.abs().max()):.4f}, "
            f"carrying {int((layer.v_reference.abs() > 1e-9).sum())}; "
            f"exempt {exempt_file}; log_c {log_c_file}"
        )
        cells = [(name, layer, gene_ids, spec) for name, spec in sources.items()]
        if args.gem_universe_control:
            gem_layer, _ = build(_gem_gene_ids(), medium, exempt_file, log_c_file)
            cells.append(
                (
                    "wild_type_4_gem_universe",
                    gem_layer,
                    _gem_gene_ids(),
                    (4, empty, empty),
                )
            )
        for name, cell_layer, universe, (rows_n, removal, removal_batch) in cells:
            torch.manual_seed(args.seed)
            h = torch.randn(rows_n, len(universe), args.hidden)
            context = torch.randn(rows_n, args.hidden)
            with torch.no_grad():
                flux = cell_layer(h, context, removal, removal_batch)
            stats = flux_statistics(cell_layer, flux)
            clip = clip_decomposition(cell_layer, h, context, removal, removal_batch)
            rows.append(
                {
                    "medium": medium,
                    "genotype_source": name,
                    "n_rows": rows_n,
                    "n_gene_tokens": len(universe),
                    "reference": "single_wild_type",
                    "exempt_file": exempt_file,
                    "log_c_file": log_c_file,
                    "media_meta": media_meta,
                    "statistics": stats,
                    "clip": clip,
                }
            )
            print(f"-- {medium} / {name}")
            print(json.dumps(stats, indent=2))
            print(
                json.dumps(
                    {
                        k: clip[k]
                        for k in (
                            "mean_reaction_availability",
                            "n_reactions_gated_to_zero",
                            "thermo_emptied_box_frac",
                            "capacity_median_over_gpr_reactions",
                        )
                        if k in clip
                    },
                    indent=2,
                )
            )
            for tag in ("box", "boxforce"):
                block = clip[tag]
                print(
                    f"   {tag}: clip {block['clip_cost_rel']:.4f}, "
                    f"forbidden {block['n_sign_forbidden']:.1f}, "
                    f"over {block['n_over_ceiling']:.1f}, "
                    f"median excess {block['median_excess_over_ceiling']}"
                )

    payload = {
        "design": (
            "2x2 over {shipped bounds, YPD} x {wild type, real deletion genotypes}, one "
            "wild-type reference per medium (solved inside the layer's own build step on "
            "that medium's bounds), the same seed and the same h in every cell of a "
            "genotype column. wild_type_32 and deletion_32 differ in the removal index "
            "alone; wild_type_4 ties the table back to job 1937."
        ),
        "seed": args.seed,
        "arm_file": args.arm_file,
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

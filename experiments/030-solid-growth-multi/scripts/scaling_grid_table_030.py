# experiments/030-solid-growth-multi/scripts/scaling_grid_table_030.py
# [[experiments.030-solid-growth-multi.scripts.scaling_grid_table_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/scaling_grid_table_030
"""The tables of the 030 scaling-study document, generated from the grid's own files.

Every number a reader of ``notes-tex/030-scaling-study/`` meets in a table comes
from here, reading the committed artifacts the grid was built from rather than a
hand-typed copy of them:

- **t1-sizes**: the N ladder. Each size is instantiated as the trainer instantiates it
  (``CellGraphTransformer`` at the width, depth and 9 heads of the grid, the per-entry
  dataset token of the 030 arm) and its non-embedding parameter count is read from the
  model's own ``num_parameters``, the count ``ComputeAccounting`` logs. The 180x8 count
  is asserted against the value W&B holds for the instrumented IGB run (53qqijnu:
  3,587,350), so a construction that drifted from the trainer's would fail here rather
  than print a wrong N.
- **t2-subsets**: the D ladder from ``subset_s3q_fractions_030_summary.json``: query
  pairs kept, training triples, closure doubles, training records (the 025 unit) and
  gene_interaction entry rows (the 030 unit), the epoch budget the submit script gives
  each subset, the records seen under it, and the wall time at the assumed 180x8 pace.
- **t3-floors**: the replicate floor per screen in label units, E = (1 - r) sd^2, from
  ``triple_noise_ceiling_030.json``: the sd of released scores per screen and the
  replicate reliabilities of the pair classes the readout uses (2020 same-strain
  pairs; 2018 bounded by the Dango published 0.59 and the cross-year 0.89), beside the
  mean-prediction loss sd^2 those floors are read against.
- **t4-jobs**: the job list of both submit scripts, mirrored from their loops: which
  (size, subset, seed, lr) each job trains and under what epoch budget.

The epoch budgets and the 68 records/s pace are the submit script's and the weekly
note's (2026.10.11); the pace is an ASSUMPTION until the calibration jobs report
``compute/records_seen`` against ``compute/gpu_hours_cumulative``, and the table says so.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_grid_table_030.py

Writes ``notes-tex/030-scaling-study/tables/t{1-sizes,2-subsets,3-floors,4-jobs}.tex``
and ``experiments/030-solid-growth-multi/results/scaling_grid_table_030.json``.
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from torch_geometric.data import HeteroData

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import load_json_artifact, results_dir  # noqa: E402

from torchcell.models.equivariant_cell_graph_transformer import (  # noqa: E402
    CellGraphTransformer,
)
from torchcell.trainers.compute_accounting import count_parameters  # noqa: E402

SCRIPT = "experiments/030-solid-growth-multi/scripts/scaling_grid_table_030.py"
SIZES = [(45, 2), (63, 3), (90, 4), (126, 6), (180, 8)]
HEADS = 9
VOCAB_SIZE = 15  # the build's source datasets, subset_definitions_030_summary.json
TOKEN_DIM = 8
N_180x8_WANDB = 3_587_350  # compute/params_non_embedding of run 53qqijnu
EPOCHS = {64: 240, 16: 120, 4: 40, 1: 10}  # delta_submit_scaling_030.sh
PACE_RECORDS_PER_S = 68.0  # one A40 at 180x8, weekly note 2026.10.11; ASSUMED
STAGE_IN_MIN = 2.0  # the pool store copy, delta_cgt_030.slurm
RELIABILITY_2018 = {"Dango 0.59": 0.59}  # the published 2018 replicate Pearson


class SizeRow(BaseModel):
    """One rung of the N ladder with its measured parameter count."""

    width: int
    depth: int
    head_dim: int
    params_non_embedding: int
    params_total_counted: int


class SubsetRow(BaseModel):
    """One rung of the D ladder with its budget."""

    denominator: int
    pairs_kept: int
    train_triples: int
    doubles: int
    train_records: int
    gi_entry_rows: int
    rows_per_record: float
    epochs: int
    records_seen: int
    hours_at_pace: float


class FloorRow(BaseModel):
    """The replicate floor of one screen under one reliability."""

    screen: str
    reliability_source: str
    reliability: float
    sd_of_scores: float
    variance: float
    floor_E: float


class JobRow(BaseModel):
    """One job of a submit script."""

    sweep: str
    name: str
    width: int
    depth: int
    denominator: int
    seed: int
    lr: str
    epochs: int
    hours: int


class GridTables(BaseModel):
    """Everything the four tables print."""

    sizes: list[SizeRow]
    subsets: list[SubsetRow]
    floors: list[FloorRow]
    jobs: list[JobRow]
    pace_records_per_s: float
    n_180x8_wandb: int


def _tiny_cell_graph() -> HeteroData:
    """The smallest cell graph the model constructs on; the gene count only sizes the
    embedding table, which the non-embedding count excludes.
    """
    cg = HeteroData()
    cg["gene"].num_nodes = 8
    cg["gene", "physical", "gene"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
    )
    return cg


def count_sizes() -> list[SizeRow]:
    rows = []
    for width, depth in SIZES:
        torch.manual_seed(0)
        model = CellGraphTransformer(
            gene_num=8,
            hidden_channels=width,
            num_transformer_layers=depth,
            num_attention_heads=HEADS,
            cell_graph=_tiny_cell_graph(),
            heads_config={
                "global": {
                    "decoder": "s1_pool",
                    "output_dim": 1,
                    "use_gene_pool": False,
                    "linear": True,
                }
            },
            perturbation_head_config={"num_heads": HEADS, "dropout": 0.1},
            perturb_cls=True,
            perturbation_head_cls="wildtype",
            dataset_token={"enabled": True, "vocab_size": VOCAB_SIZE, "dim": TOKEN_DIM},
        )
        counts = count_parameters(model, ("gene_embedding", "embedding_preprocessor"))
        rows.append(
            SizeRow(
                width=width,
                depth=depth,
                head_dim=width // HEADS,
                params_non_embedding=counts["non_embedding"],
                params_total_counted=counts["total"],
            )
        )
    n180 = next(r for r in rows if r.width == 180).params_non_embedding
    assert n180 == N_180x8_WANDB, (
        f"180x8 counts {n180:,} non-embedding parameters here but W&B run 53qqijnu "
        f"logged {N_180x8_WANDB:,}; the construction no longer matches the trainer's"
    )
    return rows


def subset_rows() -> list[SubsetRow]:
    s = load_json_artifact("subset_s3q_fractions_030_summary.json")
    full = {
        "denominator": 1,
        "n_train_pairs_kept": s["n_train_pairs"],
        "n_train_triples": None,
        "n_doubles": None,
        "n_train_records": s["n_s3q_train_records"],
        "n_train_gi_entry_rows": s["n_s3q_train_gi_entry_rows"],
    }
    q = load_json_artifact("query_pair_disjoint_splits_030.json.gz")
    full["n_train_triples"] = len(q["pinned"]["train"])
    # closure doubles of the full pool: every S3Q record that is neither a single nor a
    # pinned triple
    full["n_doubles"] = (
        s["n_s3q"]
        - s["fractions"][0]["n_singles"]
        - sum(len(q["pinned"][k]) for k in ("train", "val", "test"))
    )
    rows = []
    for fs in sorted(s["fractions"], key=lambda f: -f["denominator"]) + [full]:
        d = int(fs["denominator"])
        epochs = EPOCHS[d]
        seen = int(fs["n_train_records"]) * epochs
        rows.append(
            SubsetRow(
                denominator=d,
                pairs_kept=int(fs["n_train_pairs_kept"]),
                train_triples=int(fs["n_train_triples"]),
                doubles=int(fs["n_doubles"]),
                train_records=int(fs["n_train_records"]),
                gi_entry_rows=int(fs["n_train_gi_entry_rows"]),
                rows_per_record=fs["n_train_gi_entry_rows"] / fs["n_train_records"],
                epochs=epochs,
                records_seen=seen,
                hours_at_pace=seen / PACE_RECORDS_PER_S / 3600 + STAGE_IN_MIN / 60,
            )
        )
    return rows


def floor_rows() -> list[FloorRow]:
    nc = load_json_artifact("triple_noise_ceiling_030.json")
    nm, pc = nc["noise_model"], nc["pair_classes"]
    sd18 = float(nm["TmiKuzmin2018Dataset"]["sd_of_scores"])
    sd20 = float(nm["TmiKuzmin2020Dataset"]["sd_of_scores"])
    rows = []
    for src, r in {
        **RELIABILITY_2018,
        "cross-year 2018 vs 2020 pairs": float(
            pc["cross_dataset_2018_vs_2020"]["pearson"]
        ),
    }.items():
        rows.append(
            FloorRow(
                screen="Kuzmin 2018",
                reliability_source=src,
                reliability=r,
                sd_of_scores=sd18,
                variance=sd18**2,
                floor_E=(1 - r) * sd18**2,
            )
        )
    r20 = float(pc["same_dataset_same_strains"]["pearson"])
    rows.append(
        FloorRow(
            screen="Kuzmin 2020",
            reliability_source="same-strain re-measured pairs",
            reliability=r20,
            sd_of_scores=sd20,
            variance=sd20**2,
            floor_E=(1 - r20) * sd20**2,
        )
    )
    return rows


def job_rows() -> list[JobRow]:
    """The loops of delta_submit_scaling_030.sh and delta_submit_scaling.sh."""
    jobs: list[JobRow] = []
    denoms = [64, 16, 4, 1]

    def name(prefix: str, w: int, d: int, den: int, s: int, extra: str = "") -> str:
        sub = "full" if den == 1 else str(den)
        return f"{prefix}-N{w}L{d}-D{sub}{extra}-s{s}"

    for w, d in [(180, 8), (45, 2)]:
        jobs.append(
            JobRow(
                sweep="030_sclw1",
                name=f"030-sclw1-calib-N{w}L{d}",
                width=w,
                depth=d,
                denominator=1,
                seed=1,
                lr="2.5e-4",
                epochs=0,
                hours=3,
            )
        )
    for den in denoms:
        for w, d in SIZES:
            jobs.append(
                JobRow(
                    sweep="030_sclw1",
                    name=name("030-sclw1", w, d, den, 1),
                    width=w,
                    depth=d,
                    denominator=den,
                    seed=1,
                    lr="2.5e-4",
                    epochs=EPOCHS[den],
                    hours=48,
                )
            )
    for den in (64, 16):
        for w, d in [(45, 2), (180, 8)]:
            for s in (2, 3):
                jobs.append(
                    JobRow(
                        sweep="030_sclw1",
                        name=name("030-sclw1", w, d, den, s),
                        width=w,
                        depth=d,
                        denominator=den,
                        seed=s,
                        lr="2.5e-4",
                        epochs=EPOCHS[den],
                        hours=48,
                    )
                )
    for lr in ("1e-4", "1e-3"):
        jobs.append(
            JobRow(
                sweep="030_sclw1",
                name=name("030-sclw1", 45, 2, 4, 1, f"-lr{lr}"),
                width=45,
                depth=2,
                denominator=4,
                seed=1,
                lr=lr,
                epochs=EPOCHS[4],
                hours=48,
            )
        )
    for den in denoms:
        jobs.append(
            JobRow(
                sweep="025_sclw1",
                name=name("025-sclw1", 180, 8, den, 1),
                width=180,
                depth=8,
                denominator=den,
                seed=1,
                lr="2.5e-4",
                epochs=EPOCHS[den],
                hours=48,
            )
        )
    for den in (64, 16):
        for s in (2, 3):
            jobs.append(
                JobRow(
                    sweep="025_sclw1",
                    name=name("025-sclw1", 180, 8, den, s),
                    width=180,
                    depth=8,
                    denominator=den,
                    seed=s,
                    lr="2.5e-4",
                    epochs=EPOCHS[den],
                    hours=48,
                )
            )
    return jobs


def _header(reads: str) -> str:
    return f"%% GENERATED FILE -- do not hand-edit.\n%% SOURCE: {SCRIPT}, reading {reads}\n"


def _den(d: int) -> str:
    return "full" if d == 1 else f"1/{d}"


def write_tables(t: GridTables, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n180 = t.n_180x8_wandb
    lines = [
        _header(
            "the model's num_parameters at each size (checked against W&B 53qqijnu)"
        ),
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{The $N$ ladder. Width and depth scale together at 9 attention heads; "
        "$N$ is the non-embedding count the compute callback logs, read from the "
        "model's own tally at each size, with the gene table and the composite "
        "preprocessor excluded. The 180x8 count equals the one W\\&B holds for the "
        f"instrumented IGB run (53qqijnu, {n180:,}), asserted by the generating script.}}",
        "\\label{tab:sizes}",
        "\\begin{tabular}{rrrrr}",
        "\\toprule",
        "width $\\times$ depth & head dim & $N$ (non-embedding) & $N / N_{180\\times 8}$ & $\\log_{10} N$ \\\\",
        "\\midrule",
    ]
    import math

    for r in t.sizes:
        lines.append(
            f"{r.width} $\\times$ {r.depth} & {r.head_dim} & {r.params_non_embedding:,} & "
            f"{r.params_non_embedding / n180:.3f} & {math.log10(r.params_non_embedding):.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(osp.join(out_dir, "t1-sizes.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [
        _header(
            "subset_s3q_fractions_030_summary.json, query_pair_disjoint_splits_030.json.gz and the epoch table of delta_submit_scaling_030.sh"
        ),
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{The $D$ ladder and its budgets. Subsets are nested by query pair "
        "(seed 42, the same kept pairs on the 025 build); singles are in every subset "
        "and the pinned validation and test triples are untouched. Records are the "
        "025 unit of $D$; gene\\_interaction entry rows are the 030 unit, what the "
        "per-entry loss averages over, and rows per record is the ratio between them. "
        "Epochs are the submit script's budget, records seen is records $\\times$ "
        "epochs, and hours is that count at the ASSUMED pace of "
        f"{t.pace_records_per_s:.0f} records/s on one A40 at 180x8 plus stage-in, "
        "pending the calibration jobs.}",
        "\\label{tab:subsets}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "subset & pairs & triples & doubles & records & gi rows & rows/rec & epochs & records seen & h at pace \\\\",
        "\\midrule",
    ]
    for r in t.subsets:
        lines.append(
            f"{_den(r.denominator)} & {r.pairs_kept} & {r.train_triples:,} & {r.doubles:,} & "
            f"{r.train_records:,} & {r.gi_entry_rows:,} & {r.rows_per_record:.2f} & {r.epochs} & "
            f"{r.records_seen / 1e6:.2f}M & {r.hours_at_pace:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(osp.join(out_dir, "t2-subsets.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [
        _header(
            "triple_noise_ceiling_030.json (noise_model sd_of_scores, pair_classes pearson)"
        ),
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        "\\caption{The replicate floor per screen in label units. Under a squared loss "
        "the best predictor of a single noisy measurement has expected error "
        "$E = (1 - r)\\,\\mathrm{var}(\\tau)$ where $r$ is the replicate reliability, "
        "the correlation of two independent measurements of the same triple. The "
        "variance column is the loss of predicting the screen mean, the top of the "
        "range a curve can move through, so the floor's share of it is the fraction "
        "of the loss no model can remove. The 2020 reliability is measured on this "
        "build's same-strain re-measured pairs; 2018 has 61 such pairs, so it is "
        "bounded by the published Dango replicate correlation below and the "
        "cross-year 2018 vs 2020 pairs above.}",
        "\\label{tab:floors}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "screen & reliability from & $r$ & sd($\\tau$) & var($\\tau$) & $E$ & $E$/var \\\\",
        "\\midrule",
    ]
    for r in t.floors:
        lines.append(
            f"{r.screen} & {r.reliability_source} & {r.reliability:.2f} & {r.sd_of_scores:.4f} & "
            f"{r.variance * 1e3:.2f}e-3 & {r.floor_E * 1e3:.2f}e-3 & {r.floor_E / r.variance:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(osp.join(out_dir, "t3-floors.tex"), "w") as f:
        f.write("\n".join(lines))

    n030 = sum(1 for j in t.jobs if j.sweep == "030_sclw1")
    n025 = sum(1 for j in t.jobs if j.sweep == "025_sclw1")
    lines = [
        _header("the loops of delta_submit_scaling_030.sh and delta_submit_scaling.sh"),
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        f"\\caption{{The job list, {n030} jobs of the 030 grid and {n025} of the 025 pair, "
        "one GPU each on the Delta A40 partition, 48 h apiece except the two 3 h "
        "calibration jobs (400 training batches at 180x8 and 45x2, which report the "
        "pace and the measured FLOPs per step). Every 030 job is the composite "
        "closure arm on the query-pair-disjoint split (cgt\\_030\\_s3\\_q\\_tok\\_embfit\\_004) "
        "at batch 256; every 025 job is its merged-label twin (cgt\\_s3\\_q\\_kl\\_embfit\\_037) "
        "at 180x8.}",
        "\\label{tab:jobs}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "sweep & what varies & jobs & sizes & subsets & seeds \\\\",
        "\\midrule",
    ]

    def _summ(sweep: str, pred, label: str) -> str:
        js = [j for j in t.jobs if j.sweep == sweep and pred(j)]
        sizes = sorted({(j.width, j.depth) for j in js})
        dens = sorted({j.denominator for j in js}, reverse=True)
        seeds = sorted({j.seed for j in js})
        return (
            f"{sweep.replace('_', '\\_')} & {label} & {len(js)} & "
            f"{', '.join(f'{w}x{d}' for w, d in sizes)} & "
            f"{', '.join(_den(d) for d in dens)} & {', '.join(str(s) for s in seeds)} \\\\"
        )

    lines.append(_summ("030_sclw1", lambda j: j.epochs == 0, "calibration"))
    lines.append(
        _summ(
            "030_sclw1",
            lambda j: j.epochs > 0 and j.seed == 1 and j.lr == "2.5e-4",
            "size $\\times$ subset",
        )
    )
    lines.append(_summ("030_sclw1", lambda j: j.seed > 1, "seed replicates"))
    lines.append(
        _summ(
            "030_sclw1", lambda j: j.lr != "2.5e-4", "learning-rate check (1e-4, 1e-3)"
        )
    )
    lines.append(_summ("025_sclw1", lambda j: j.seed == 1, "subset"))
    lines.append(_summ("025_sclw1", lambda j: j.seed > 1, "seed replicates"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(osp.join(out_dir, "t4-jobs.tex"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    load_dotenv()
    repo = osp.dirname(os.environ["EXPERIMENT_ROOT"].rstrip("/"))
    out_dir = osp.join(repo, "notes-tex/030-scaling-study/tables")
    tables = GridTables(
        sizes=count_sizes(),
        subsets=subset_rows(),
        floors=floor_rows(),
        jobs=job_rows(),
        pace_records_per_s=PACE_RECORDS_PER_S,
        n_180x8_wandb=N_180x8_WANDB,
    )
    write_tables(tables, out_dir)
    with open(osp.join(results_dir(), "scaling_grid_table_030.json"), "w") as f:
        f.write(tables.model_dump_json(indent=2))
    for r in tables.sizes:
        print(f"  {r.width}x{r.depth}: N = {r.params_non_embedding:,}")
    for r in tables.subsets:
        print(
            f"  {_den(r.denominator)}: {r.train_records:,} records, {r.gi_entry_rows:,} rows, "
            f"{r.epochs} epochs, {r.records_seen / 1e6:.2f}M seen, {r.hours_at_pace:.1f} h"
        )
    for r in tables.floors:
        print(f"  {r.screen} ({r.reliability_source}): E = {r.floor_E:.3e}")
    print(
        f"  {len(tables.jobs)} jobs; wrote {out_dir}/t{{1-sizes,2-subsets,3-floors,4-jobs}}.tex"
    )


if __name__ == "__main__":
    main()

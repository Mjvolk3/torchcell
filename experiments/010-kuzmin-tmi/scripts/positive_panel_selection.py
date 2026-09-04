# experiments/010-kuzmin-tmi/scripts/positive_panel_selection.py
# [[experiments.010-kuzmin-tmi.scripts.positive_panel_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/positive_panel_selection

"""Select a small positive-trigenic-interaction panel from inference_1.

The previous panel was chosen by greedy top-k coverage on ``inference_3``, whose
predictions were invalid, and its objective had no diversity term. Three things
about it are worth not repeating.

    tau closure, not construction cost.  tau on a triple consumes seven
        fitnesses: the triple, its three doubles and its three singles. A design
        that set-covers 20 triples with 8 doubles yields 20 fitnesses and zero
        tau. The previous 20 triples spanned 11 genes and pulled in 33 distinct
        doubles, 65 strains in total.
    A hub gene took the panel over.  Nothing penalized concentration and the
        ``top_count`` tiebreak actively rewarded hubs; participation ran 46 down
        to 16 across the 12 genes. One round earlier the same pathology
        collapsed inference_2 onto a single gene.
    Confidence tracked absence of evidence.  Predicted interaction rose with the
        number of panel genes having no trigenic training record (Spearman
        0.634), and zero-data genes took ranks 1 to 11 of 39 targets.

The design here answers all three structurally rather than by penalty terms.

**Two gene-disjoint complete blocks.** A complete design on ``BLOCK_SIZE`` genes
contributes C(k,3) triples using only C(k,2) doubles, and every gene appears in
exactly C(k-1,2) of them. Two disjoint 5-gene blocks give 20 triples over 10
genes on 20 doubles, with every gene in exactly 6 triples. Balance is a property
of the shape, so no diversity term is needed and no hub can form. Two blocks
rather than one also means two independent bets: a block that fails at the bench
does not take the round with it.

**Ranked by the worst checkpoint, not the mean.** Two training runs share only
0.04 to 0.14 of their top 100 on this space, so a mean can be carried by one
optimistic run. Scoring a block by the minimum across the three checkpoints
makes disagreement cost score directly.

**Every gate the previous panel needed is applied before scoring**, not after:
name resolution through the shared reconciler, membership in the model's gene
set, no two genes resolving to one locus, a floor on trigenic training records,
and single-mutant fitness aggregated by mean across strain backgrounds rather
than max. The max aggregation is what let CBF1 onto the last panel at a true
fitness of 0.59.

Outputs, under ``experiments/010-kuzmin-tmi/results/``:

    positive_panel_candidates.csv   the best 5,000 blocks, scored
    positive_panel_candidate_quantiles.csv  the full distribution they sit in
    positive_panel_selected.csv     the chosen blocks' 20 triples
    positive_panel_strains.csv      the full tau closure to put on a plate
    positive_panel_summary.json     gates, counts, diversity and cost
"""

import glob
import itertools
import json
import os
import os.path as osp
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
INFERENCE_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
)

CHECKPOINTS = {"M01_lzs9pcj3": "0.4520", "M02_yv4r30bi": "0.4472", "M03_c7671wgj": "0.4619"}
TAGS = list(CHECKPOINTS)

BLOCK_SIZE = 5          # C(5,3) = 10 triples on C(5,2) = 10 doubles
N_BLOCKS = 2            # 10 genes, 20 triples, 20 doubles
MIN_TRAIN_RECORDS = 200  # the previous panel's cliff was 519 -> 10 -> 0
# A record count is not support. Kuzmin groups records by the query double
# they were measured under, so a gene with 1,097 records from ONE screen has
# been measured once. 88 genes are in that position and their median record
# count is 1,046, so the record floor cannot see them. The distinct-screen
# distribution has a gap: 2,339 genes sit at 10 to 49 and 1,171 at 200 or
# more, with only 11 in between, so any floor in that gap gives the same
# answer. See screen_diversity_audit.py.
MIN_DISTINCT_SCREENS = 50
SEED_POOL = 60000        # rank cut on the positive tail used to seed block search
MAX_SEED_TRIPLES = 4000  # how many seeds to expand into blocks
N_CANDIDATES_KEPT = 5000  # millions qualify; only the head is committed


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def prediction_path(pearson: str) -> str:
    matches = [
        f
        for f in glob.glob(osp.join(INFERENCE_DIR, f"*Pearson={pearson}*.parquet"))
        if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))
    ]
    assert len(matches) == 1, f"expected one file for Pearson={pearson}, got {matches}"
    return matches[0]


def load_space() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sorted gene-code triples, per-checkpoint predictions, and the vocabulary."""
    preds: dict[str, np.ndarray] = {}
    triples: np.ndarray | None = None
    vocab: list[str] = []
    index_ref: np.ndarray | None = None

    for tag, pearson in CHECKPOINTS.items():
        table = pq.read_table(
            prediction_path(pearson),
            columns=["index", "gene1", "gene2", "gene3", "prediction"],
        )
        idx = table["index"].to_numpy()
        if index_ref is None:
            index_ref = idx
        else:
            assert np.array_equal(idx, index_ref), f"{tag} row order differs"
        if triples is None:
            col_of: dict[str, int] = {}
            cols = []
            for c in ("gene1", "gene2", "gene3"):
                s = table[c].to_pandas()
                for g in s.unique():
                    if g not in col_of:
                        col_of[g] = len(vocab)
                        vocab.append(g)
                cols.append(s.map(col_of).to_numpy(dtype=np.int32))
            triples = np.sort(np.stack(cols, axis=1), axis=1)
        preds[tag] = table["prediction"].to_numpy().astype(np.float64)
        del table

    assert triples is not None
    return triples, np.stack([preds[t] for t in TAGS], axis=1), vocab


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def distinct_screens(gene_index: dict[str, list]) -> dict[str, int]:
    """How many distinct Kuzmin query doubles each gene was measured under.

    Every record is a query double crossed against one array gene, so grouping a
    gene's records by their query double gives the number of independent screens
    its behavior was observed in. A gene seen under one screen has one
    measurement of context, however many array genes that screen crossed.
    """
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    label_df = label_df.sort_values("index")
    row_of = {int(r): i for i, r in enumerate(label_df["index"].to_numpy())}
    names = sorted(gene_index)
    col = {g: j for j, g in enumerate(names)}
    rows = np.full((len(row_of), 3), -1, dtype=np.int32)
    fill = np.zeros(len(row_of), dtype=np.int8)
    for g, ids in gene_index.items():
        c = col[g]
        for r in ids:
            i = row_of[int(r)]
            rows[i, fill[i]] = c
            fill[i] += 1
    rows = np.sort(rows, axis=1)
    base = len(names) + 1
    a, b, c = (rows[:, j].astype(np.int64) for j in range(3))
    pairs = np.stack([a * base + b, a * base + c, b * base + c], axis=1)
    keys, counts = np.unique(pairs.reshape(-1), return_counts=True)
    recurring = set(keys[counts >= 5].tolist())
    query = np.array(
        [next(int(p) for p in r if int(p) in recurring) for r in pairs], dtype=np.int64
    )
    out: dict[str, int] = {}
    for c_, g in enumerate(names):
        m = (rows == c_).any(axis=1)
        out[g] = len(set(query[m].tolist())) if m.any() else 0
    return out


def gene_gates(vocab: list[str]) -> tuple[dict[str, str], pd.DataFrame]:
    """Resolve every gene and record why each one passes or fails.

    Applied before any scoring, so a gene that cannot be built or cannot be
    scored never reaches the objective. Each gate corresponds to a specific way
    the previous panel broke.
    """
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)

    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    support = {g: len(v) for g, v in gene_index.items()}
    screens = distinct_screens(gene_index)

    rows: list[dict[str, object]] = []
    resolved: dict[str, str] = {}
    seen_systematic: dict[str, str] = {}
    for g in vocab:
        r = genome.resolve_gene_name(g)
        sysname = r.systematic_name
        n_train = support.get(sysname, 0)
        # A second gene resolving to a locus already claimed is the SPH1 /
        # YLR312C-B failure: two wells, one gene.
        collision = seen_systematic.get(sysname)
        reasons = []
        if sysname not in gene_set:
            reasons.append("outside the model gene set")
        if sysname != g:
            reasons.append(f"stale name, resolves to {sysname}")
        if collision is not None:
            reasons.append(f"same locus as {collision}")
        if n_train < MIN_TRAIN_RECORDS:
            reasons.append(f"only {n_train} trigenic training records")
        n_screens = screens.get(sysname, 0)
        if n_screens < MIN_DISTINCT_SCREENS:
            reasons.append(f"measured under only {n_screens} distinct screens")
        ok = not reasons
        if ok:
            resolved[g] = sysname
            seen_systematic[sysname] = g
        rows.append(
            {
                "gene": g,
                "resolves_to": sysname,
                "status": r.status.value,
                "in_gene_set": sysname in gene_set,
                "trigenic_train_records": n_train,
                "distinct_screens": n_screens,
                "passes": ok,
                "reasons": "; ".join(reasons),
            }
        )
    return resolved, pd.DataFrame(rows)


def fitness_table(genes: list[str]) -> pd.DataFrame:
    """Single-mutant fitness per gene, aggregated by MEAN across strains.

    The previous selection used max() across strain backgrounds, which let
    YJR060W through at 0.923 (KanMX) while its NatMX strain measured 0.590. The
    per-strain spread is reported so a disagreement is visible rather than
    averaged away.
    """
    from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset

    ds = SmfCostanzo2016Dataset(root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016"))
    df = ds.df
    df = df[df["perturbation_type"].isin(["KanMX_deletion", "NatMX_deletion"])]
    want = set(genes)
    df = df[df["Systematic gene name"].isin(want)]
    out = (
        df.groupby("Systematic gene name")["Single mutant fitness"]
        .agg(smf_mean="mean", smf_min="min", smf_max="max", n_strains="size")
        .reset_index()
        .rename(columns={"Systematic gene name": "gene"})
    )
    out["smf_strain_spread"] = out["smf_max"] - out["smf_min"]
    return out


# ---------------------------------------------------------------------------
# Block search
# ---------------------------------------------------------------------------


def build_block_candidates(
    triples: np.ndarray, score: np.ndarray, allowed: np.ndarray
) -> tuple[dict[frozenset, float], dict[int, set[int]]]:
    """Score lookup for allowed triples, and the pair-adjacency they induce.

    A pair is an edge exactly when some triple in the space contains it, which
    is what the upstream generator guaranteed: every triple is a viable
    3-clique with measured single and double fitness for all of its members.
    """
    keep = allowed[triples].all(axis=1)
    tri = triples[keep]
    sc = score[keep]
    lookup = {frozenset(map(int, r)): float(s) for r, s in zip(tri, sc)}
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b, c in tri:
        adj[int(a)].update((int(b), int(c)))
        adj[int(b)].update((int(a), int(c)))
        adj[int(c)].update((int(a), int(b)))
    return lookup, adj


def complete_blocks(
    lookup: dict[frozenset, float], adj: dict[int, set[int]], seeds: list[frozenset]
) -> pd.DataFrame:
    """Every BLOCK_SIZE gene set whose C(k,3) triples are ALL in the space.

    Grown from high-scoring seed triples by intersecting neighborhoods, so the
    search never enumerates the full C(526,5). A block is kept only if all
    C(k,3) of its triples are present, which is what makes the design complete
    and its doubles count exactly C(k,2).
    """
    found: dict[frozenset, dict[str, object]] = {}
    for seed in seeds:
        members = list(seed)
        common = set.intersection(*(adj[m] for m in members)) - set(members)
        for extra in itertools.combinations(sorted(common), BLOCK_SIZE - 3):
            block = frozenset(members) | frozenset(extra)
            if len(block) != BLOCK_SIZE or block in found:
                continue
            tri = list(itertools.combinations(sorted(block), 3))
            vals = [lookup.get(frozenset(t)) for t in tri]
            if any(v is None for v in vals):
                continue
            arr = np.array(vals, dtype=float)
            found[block] = {
                "genes": tuple(sorted(block)),
                "n_triples": len(arr),
                "worst_triple": float(arr.min()),
                "mean_triple": float(arr.mean()),
                "median_triple": float(np.median(arr)),
                "n_above_zero": int((arr > 0).sum()),
            }
    return pd.DataFrame(list(found.values()))


def pick_disjoint(cands: pd.DataFrame, n_blocks: int) -> list[tuple]:
    """Best block by median, then the best block sharing no gene with it.

    Ranked on the median of a block's ten triples rather than its worst. The
    worst is dominated by whichever single triple the checkpoints disagree on,
    which for the leading block is the one triple lacking its hub gene, so
    ranking on it would understate exactly the block whose claim is strongest.

    Disjointness holds every gene at exactly C(k-1,2) of the triples. It does
    NOT buy two equally strong bets: the landscape below shows the model's
    positive confidence is concentrated in one clique, so the second block is a
    generalization control rather than a second wager, and is labeled as one.
    """
    chosen: list[tuple] = []
    used: set[int] = set()
    for _, row in cands.sort_values("median_triple", ascending=False).iterrows():
        genes = row["genes"]
        if used.intersection(genes):
            continue
        chosen.append(row)
        used.update(genes)
        if len(chosen) == n_blocks:
            break
    return chosen


def training_enrichment(genes: list[str]) -> pd.DataFrame:
    """Per-gene share of training triples that are published-scale positives.

    A model that liked a gene the labels do not would be a warning. This is the
    check that separates a learned effect from an extrapolation.
    """
    lab = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    lab = lab.sort_values("index")
    tau = lab["gene_interaction"].to_numpy()
    row_of = {int(r): i for i, r in enumerate(lab["index"].to_numpy())}
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    base_pos = float((tau > 0.08).mean())
    rows = []
    for g in genes:
        idx = np.array([row_of[int(r)] for r in gene_index.get(g, [])], dtype=int)
        t = tau[idx]
        rows.append(
            {
                "gene": g,
                "n_train": len(t),
                "mean_tau": float(t.mean()),
                "frac_pos": float((t > 0.08).mean()),
                "frac_neg": float((t < -0.08).mean()),
                "pos_enrichment": float((t > 0.08).mean() / base_pos),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("loading inference_1 ...")
    triples, preds, vocab = load_space()
    worst = preds.min(axis=1)   # the objective: the least optimistic checkpoint
    mean = preds.mean(axis=1)
    print(f"  {len(triples):,} triples over {len(vocab)} genes")

    print("\napplying gene gates ...")
    resolved, gates = gene_gates(vocab)
    gates.to_csv(osp.join(RESULTS_DIR, "positive_panel_gene_gates.csv"), index=False)
    print(f"  {int(gates['passes'].sum())} of {len(gates)} genes pass")
    for reason, n in gates.loc[~gates["passes"], "reasons"].value_counts().head(6).items():
        print(f"    {n:>4}  {reason}")

    allowed = np.zeros(len(vocab), dtype=bool)
    name_to_code = {g: i for i, g in enumerate(vocab)}
    for g in resolved:
        allowed[name_to_code[g]] = True

    print("\nsearching complete blocks ...")
    order = np.argsort(worst)[::-1][:SEED_POOL]
    seeds = [
        frozenset(map(int, triples[i]))
        for i in order
        if allowed[triples[i]].all()
    ][:MAX_SEED_TRIPLES]
    lookup, adj = build_block_candidates(triples, worst, allowed)
    cands = complete_blocks(lookup, adj, seeds)
    print(f"  {len(seeds)} seed triples -> {len(cands)} complete {BLOCK_SIZE}-gene blocks")

    names = np.array(vocab, dtype=object)
    cands["gene_names"] = cands["genes"].map(lambda gs: ", ".join(names[list(gs)]))
    # Millions of blocks qualify, so only the head is committed. The full
    # distribution is what the selection is read against, so its quantiles are
    # persisted alongside rather than left in memory.
    cands.nlargest(N_CANDIDATES_KEPT, "median_triple").to_csv(
        osp.join(RESULTS_DIR, "positive_panel_candidates.csv"), index=False
    )
    q = [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]
    pd.DataFrame(
        {
            "quantile": q,
            "median_triple": [float(cands["median_triple"].quantile(x)) for x in q],
            "worst_triple": [float(cands["worst_triple"].quantile(x)) for x in q],
        }
    ).to_csv(
        osp.join(RESULTS_DIR, "positive_panel_candidate_quantiles.csv"), index=False
    )
    print("\ntop blocks by worst triple:")
    print(
        cands.nlargest(8, "worst_triple")[
            ["gene_names", "worst_triple", "median_triple", "mean_triple", "n_above_zero"]
        ].to_string(index=False)
    )

    chosen = pick_disjoint(cands, N_BLOCKS)
    assert len(chosen) == N_BLOCKS, "not enough gene-disjoint blocks"

    rows: list[dict[str, object]] = []
    for bi, row in enumerate(chosen, start=1):
        for t in itertools.combinations(sorted(row["genes"]), 3):
            key = frozenset(t)
            hit = np.where((triples == np.array(sorted(t))).all(axis=1))[0][0]
            rows.append(
                {
                    "block": bi,
                    "triple": " + ".join(names[list(t)]),
                    "gene1": names[t[0]],
                    "gene2": names[t[1]],
                    "gene3": names[t[2]],
                    **{tag: float(preds[hit, j]) for j, tag in enumerate(TAGS)},
                    "worst_checkpoint": float(worst[hit]),
                    "mean_checkpoint": float(mean[hit]),
                    "spread": float(preds[hit].max() - preds[hit].min()),
                    "all_positive": bool((preds[hit] > 0).all()),
                }
            )
    sel = pd.DataFrame(rows).sort_values(
        ["block", "worst_checkpoint"], ascending=[True, False]
    )

    panel_genes = sorted({g for row in chosen for g in row["genes"]})
    panel_names = [str(names[g]) for g in panel_genes]

    # The hub is the gene carrying block 1's claim. A complete block splits
    # itself into the C(k-1,2) triples containing it and the C(k-1,3) that do
    # not, which is the contrast this design exists to measure.
    b1 = sel[sel["block"] == 1]
    b1_genes = sorted({g for t in b1["triple"] for g in t.split(" + ")})
    hub = max(
        b1_genes,
        key=lambda g: b1[b1["triple"].str.contains(g, regex=False)][
            "worst_checkpoint"
        ].mean(),
    )
    sel["tier"] = np.where(
        sel["block"] == 2,
        "C generalization control",
        np.where(
            sel["triple"].str.contains(hub, regex=False),
            f"A claim, contains {hub}",
            f"B same block without {hub}",
        ),
    )
    sel.to_csv(osp.join(RESULTS_DIR, "positive_panel_selected.csv"), index=False)
    print(f"\nhub gene of block 1: {hub}")
    print(sel.groupby("tier")["worst_checkpoint"].agg(["size", "min", "median", "max"]).to_string())

    # tau closure: every single, every double of a chosen triple, every triple.
    doubles = sorted({frozenset(p) for t in sel["triple"] for p in itertools.combinations(sorted(t.split(" + ")), 2)})
    strains = (
        [{"kind": "wild type", "strain": "BY4741"}]
        + [{"kind": "single", "strain": g} for g in panel_names]
        + [{"kind": "double", "strain": " + ".join(sorted(d))} for d in doubles]
        + [{"kind": "triple", "strain": t} for t in sel["triple"]]
    )
    pd.DataFrame(strains).to_csv(
        osp.join(RESULTS_DIR, "positive_panel_strains.csv"), index=False
    )

    fit = fitness_table(panel_names)
    enr = training_enrichment(panel_names)
    fit = fit.merge(enr, on="gene", how="left")
    fit.to_csv(osp.join(RESULTS_DIR, "positive_panel_fitness.csv"), index=False)

    per_gene = pd.Series(
        [g for t in sel["triple"] for g in t.split(" + ")]
    ).value_counts()
    summary = {
        "block_size": BLOCK_SIZE,
        "n_blocks": N_BLOCKS,
        "panel_genes": panel_names,
        "n_genes": len(panel_names),
        "n_triples": int(len(sel)),
        "n_doubles": len(doubles),
        "n_strains_total": len(strains),
        "triples_per_gene_min": int(per_gene.min()),
        "triples_per_gene_max": int(per_gene.max()),
        "worst_checkpoint_range": [float(sel["worst_checkpoint"].min()), float(sel["worst_checkpoint"].max())],
        "median_spread": float(sel["spread"].median()),
        "all_three_checkpoints_positive": int(sel["all_positive"].sum()),
        "min_trigenic_train_records": int(
            gates.set_index("gene").loc[panel_names, "trigenic_train_records"].min()
        ),
        "min_smf_mean": float(fit["smf_mean"].min()),
        "max_smf_strain_spread": float(fit["smf_strain_spread"].max()),
        "gate_min_train_records": MIN_TRAIN_RECORDS,
        "gate_min_distinct_screens": MIN_DISTINCT_SCREENS,
        "min_distinct_screens_in_panel": int(
            gates.set_index("gene").loc[panel_names, "distinct_screens"].min()
        ),
    }
    with open(osp.join(RESULTS_DIR, "positive_panel_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== selected panel ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nper-gene triple count (a complete block gives every gene the same):")
    print(per_gene.to_string())
    print("\nsingle-mutant fitness, mean across strain backgrounds:")
    print(fit.to_string(index=False))

    plot(sel, cands, fit, per_gene, summary)


def plot(
    sel: pd.DataFrame,
    cands: pd.DataFrame,
    fit: pd.DataFrame,
    per_gene: pd.Series,
    summary: dict,
) -> None:
    apply_paper_style()
    fig, axgrid = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(118.0))
    )
    axes = axgrid.ravel()

    # a. the 20 triples by tier, which is the contrast the panel is built on
    ax = axes[0]
    tiers = list(dict.fromkeys(sel["tier"]))
    show = sel.iloc[::-1]
    y = np.arange(len(show))
    tier_color = {t: PLOT_PALETTE[i] for i, t in enumerate(tiers)}
    ax.barh(
        y,
        show["worst_checkpoint"],
        color=[tier_color[t] for t in show["tier"]],
        edgecolor="black",
        linewidth=0.4,
        height=0.74,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(show["triple"], fontsize=3.6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Predicted $\tau$, worst of three checkpoints")
    ax.set_title("The 20 triples, by tier", fontsize=6)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=tier_color[t], edgecolor="black", linewidth=0.4)
        for t in tiers
    ]
    ax.legend(handles, tiers, frameon=False, fontsize=4.2, loc="lower right")

    # b. balance: this design vs the previous panel's participation spread
    ax = axes[1]
    x = np.arange(len(per_gene))
    ax.bar(x, per_gene.to_numpy(), color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, width=0.66)
    ax.axhline(
        per_gene.mean(), color="black", linewidth=0.7, linestyle="--",
        label=f"every gene in {int(per_gene.iloc[0])} of {summary['n_triples']}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(per_gene.index, rotation=90, fontsize=4.5)
    ax.set_ylabel("Triples containing the gene")
    ax.set_ylim(0, max(per_gene.max() * 1.5, 8))
    ax.set_title("Balance is structural, not a penalty term", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="upper right")

    # c. where the chosen blocks sit among all complete blocks
    ax = axes[2]
    ax.hist(cands["worst_triple"], bins=60, color=PLOT_PALETTE[5], edgecolor="black", linewidth=0.2)
    for bi, val in enumerate(sorted(sel.groupby("block")["worst_checkpoint"].min(), reverse=True)):
        ax.axvline(val, color=PLOT_PALETTE[bi], linewidth=1.0, label=f"block {bi + 1}")
    ax.set_xlabel(r"Block's worst triple, predicted $\tau$")
    ax.set_ylabel("Complete blocks")
    ax.set_yscale("log")
    ax.set_title(f"{len(cands):,} complete {summary['block_size']}-gene blocks", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="upper right")

    # d. viability: the reason to believe these strains will grow
    ax = axes[3]
    f = fit.sort_values("smf_mean")
    x = np.arange(len(f))
    ax.bar(x, f["smf_mean"], color=PLOT_PALETTE[3], edgecolor="black", linewidth=0.4, width=0.66)
    ax.errorbar(
        x, f["smf_mean"],
        yerr=[f["smf_mean"] - f["smf_min"], f["smf_max"] - f["smf_mean"]],
        fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2,
    )
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", label="wild type")
    ax.axhline(0.9, color=PLOT_PALETTE[1], linewidth=0.7, linestyle=":", label="viability floor 0.9")
    ax.set_xticks(x)
    ax.set_xticklabels(f["gene"], rotation=90, fontsize=4.5)
    ax.set_ylabel("Costanzo single-mutant fitness")
    ax.set_ylim(0, 1.25)
    ax.set_title("Mean across strain backgrounds, bars span strains", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="lower right")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)

    fig.suptitle(
        "A positive trigenic panel from inference_1: two complete gene-disjoint blocks",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "positive_panel_selection")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nwrote {stem}.svg")


if __name__ == "__main__":
    main()

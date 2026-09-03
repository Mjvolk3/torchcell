# experiments/010-kuzmin-tmi/scripts/review_inference_1_top_triples.py
# [[experiments.010-kuzmin-tmi.scripts.review_inference_1_top_triples]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/review_inference_1_top_triples

"""Review the top trigenic interactions inference_1 nominates, across three checkpoints.

``inference_2`` and ``inference_3`` are invalid: the genome gene set shrank from
6,607 to 6,579 entries between training and those runs, shifting every gene index
by 28, and their candidate genes were never intersected with the model's gene
space so unresolvable names silently turned triples into doubles.
``inference_1`` is reported as unaffected by both. This script does not take that
on trust. It re-derives both properties for this run and only then reads the
predictions:

    defect one   a fresh sample of stored triples is re-scored through the
                 validated direct path under both the 6,607-gene and the
                 6,579-gene index space. Reproduction under exactly one of them
                 identifies which space the run used.

    defect two   every distinct gene name in the space is resolved through the
                 shared reconciler and checked against the model's gene set. A
                 name the index cannot resolve is what shortens a triple to a
                 double, so zero unresolvable names is the direct test that no
                 triple collapsed.

The ranking is an ensemble of all three checkpoints rather than one. Two training
runs of this architecture share only 0.39 to 0.47 of their top 100, and on
``inference_1`` specifically the checkpoint-to-checkpoint correlation in the tail
runs 0.524 to 0.560, so a single checkpoint's extreme tail is not the model's
opinion. Ranking by the mean of three and reporting the spread is what makes a
nomination readable.

Direction is negative. The published Kuzmin call for a trigenic interaction is
one-sided, and retrieval of strong negatives is the only use of this model that
has been measured to beat an additive null. Positive nominations are reported for
completeness but carry no such support.

Outputs, all under ``experiments/010-kuzmin-tmi/results/``:

    inference_1_validity.json            the two defect checks, with numbers
    inference_1_checkpoint_agreement.csv pairwise correlation and top-K overlap
    inference_1_top_triples.csv          top 200 negative and top 50 positive
    inference_1_gene_frequency.csv       genes carrying the negative tail
"""

import glob
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from dotenv import load_dotenv
from scipy.stats import pearsonr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import score_010_checkpoints_directly as S  # noqa: E402

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

# The three training runs, keyed by the Pearson in each prediction filename.
CHECKPOINTS = {
    "M01_lzs9pcj3": "0.4520",
    "M02_yv4r30bi": "0.4472",
    "M03_c7671wgj": "0.4619",
}
TAGS = list(CHECKPOINTS)

VALIDITY_SAMPLE = 3000
TOP_K_GRID = [10, 100, 1000, 10000]
N_NEGATIVE = 200
N_POSITIVE = 50


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def prediction_path(pearson: str) -> str:
    matches = [
        f
        for f in glob.glob(osp.join(INFERENCE_DIR, f"*Pearson={pearson}*.parquet"))
        if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))
    ]
    assert len(matches) == 1, f"expected one file for Pearson={pearson}, got {matches}"
    return matches[0]


def load_predictions() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Gene-code triples plus one prediction column per checkpoint.

    Gene names are interned to integer codes once. The three files must describe
    the same triples in the same order for a row-wise ensemble to mean anything,
    so that is checked rather than assumed.
    """
    frames: dict[str, np.ndarray] = {}
    codes: np.ndarray | None = None
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

        if codes is None:
            g1 = table["gene1"].to_pandas().astype("category")
            vocab = list(g1.cat.categories)
            code_of = {g: i for i, g in enumerate(vocab)}
            stack = []
            for col in ("gene1", "gene2", "gene3"):
                s = table[col].to_pandas()
                new = [g for g in s.unique() if g not in code_of]
                for g in new:
                    code_of[g] = len(vocab)
                    vocab.append(g)
                stack.append(s.map(code_of).to_numpy(dtype=np.int32))
            codes = np.stack(stack, axis=1)
        else:
            # Row alignment already holds on `index`; confirm the genes agree on a
            # sample so a silently reordered file cannot pass.
            take = np.linspace(0, len(idx) - 1, 20000, dtype=np.int64)
            for j, col in enumerate(("gene1", "gene2", "gene3")):
                s = table[col].to_pandas().to_numpy()[take]
                assert np.array_equal(
                    s, np.array(vocab, dtype=object)[codes[take, j]]
                ), f"{tag} gene column {col} differs from the reference file"

        frames[tag] = table["prediction"].to_numpy().astype(np.float64)
        del table

    df = pd.DataFrame(frames)
    assert codes is not None
    return df, codes, vocab


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def check_gene_space(vocab: list[str]) -> dict[str, object]:
    """Defect two: does every gene in the space resolve inside the model's index?"""
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)

    unresolvable: list[str] = []
    renamed: list[dict[str, str]] = []
    resolved: dict[str, str] = {}
    for g in vocab:
        r = genome.resolve_gene_name(g)
        if r.systematic_name not in gene_set:
            unresolvable.append(g)
            continue
        resolved[g] = r.systematic_name
        if r.systematic_name != g:
            renamed.append(
                {"as_scored": g, "resolves_to": r.systematic_name, "status": r.status.value}
            )

    return {
        "genes_in_space": len(vocab),
        "genome_gene_set": len(gene_set),
        "unresolvable": unresolvable,
        "renamed": renamed,
        "resolved": resolved,
    }


@torch.no_grad()
def check_index_space(
    df: pd.DataFrame, codes: np.ndarray, vocab: list[str], resolved: dict[str, str]
) -> dict[str, float]:
    """Defect one: re-score a sample under each candidate index space.

    Reproduction under exactly one map identifies the space the run used. The
    6,607-gene map is the genome gene set the checkpoints were trained against;
    the 6,579-gene map is the build's own smaller set, which is what inference_2
    and inference_3 were scored under.

    The 6,579 arm is now expected to REFUSE rather than return a number. The
    guard added after the defect raises when the cell graph's gene node count
    differs from the checkpoint's ``gene_num``, which is precisely this case.
    A refusal is recorded as the outcome, since the counterfactual it used to
    produce is exactly what the guard exists to prevent.
    """
    rng = np.random.default_rng(0)
    sample = rng.choice(len(df), size=VALIDITY_SAMPLE, replace=False)
    stored = df["M03_c7671wgj"].to_numpy()[sample]
    names = np.array(vocab, dtype=object)[codes[sample]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, float] = {}
    for space in ("genome", "build"):
        os.environ["BASE_GENE_SET"] = space
        S.BASE_GENE_SET = space
        cell_graph, embeddings = S.build_cell_graph()
        node_to_idx = {g: i for i, g in enumerate(cell_graph["gene"].node_ids)}
        missing = {
            resolved[g] for row in names for g in row if resolved[g] not in node_to_idx
        }
        if missing:
            out[f"{space}_n_genes"] = float(len(node_to_idx))
            out[f"{space}_pearson"] = float("nan")
            out[f"{space}_missing_genes"] = float(len(missing))
            continue
        idx = np.array(
            [[node_to_idx[resolved[g]] for g in row] for row in names], dtype=np.int64
        )
        model = S.CellGraphTransformer(
            cell_graph=cell_graph,
            graph_regularization_config=S.GRAPH_REG_CONFIG,
            perturbation_head_config=S.PERT_HEAD_CONFIG,
            graph_reg_lambda=1.0,
            node_embeddings=embeddings,
            learnable_embedding_config=S.LEARNABLE_EMBEDDING_CONFIG,
            **S.MODEL_KWARGS,
        ).to(device)
        ckpt = torch.load(
            osp.join(S.CKPT_ROOT, S.CHECKPOINTS["M03_c7671wgj"]),
            map_location="cpu",
            weights_only=False,
        )
        state = {
            k[len("model.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        inc = model.load_state_dict(state, strict=False)
        assert not inc.unexpected_keys, inc.unexpected_keys
        out[f"{space}_n_genes"] = float(len(node_to_idx))
        out[f"{space}_missing_genes"] = 0.0
        try:
            rescored = S.score(model, cell_graph.to(device), idx, device)
        except ValueError as exc:
            out[f"{space}_refused_by_guard"] = str(exc).split(".")[0]
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        out[f"{space}_pearson"] = float(pearsonr(stored, rescored)[0])
        out[f"{space}_max_abs_diff"] = float(np.max(np.abs(stored - rescored)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    os.environ["BASE_GENE_SET"] = "genome"
    S.BASE_GENE_SET = "genome"
    return out


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation over the whole space, and overlap in the negative tail."""
    order = {tag: np.argsort(df[tag].to_numpy(), kind="stable") for tag in TAGS}
    rows: list[dict[str, object]] = []
    for i, a in enumerate(TAGS):
        for b in TAGS[i + 1 :]:
            row: dict[str, object] = {
                "pair": f"{a} vs {b}",
                "pearson_all": float(pearsonr(df[a], df[b])[0]),
            }
            for k in TOP_K_GRID:
                sa, sb = set(order[a][:k].tolist()), set(order[b][:k].tolist())
                row[f"overlap_top{k}"] = len(sa & sb) / k
            rows.append(row)
    return pd.DataFrame(rows)


def training_support(resolved: dict[str, str]) -> dict[str, int]:
    """Trigenic training records per gene, counted under the resolved name.

    Counting under an outdated alias returns zero for a gene that has records,
    which is the same miss that dropped indices in inference_3.
    """
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    return {g: len(gene_index.get(sysname, [])) for g, sysname in resolved.items()}


def label_spread() -> float:
    label = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    return float(label["gene_interaction"].std(ddof=0))


def calibration(df: pd.DataFrame) -> dict[str, float]:
    """How far the nominations sit outside the range the model produces on labels.

    A squared-error fit on a noisy target shrinks hard toward the mean, so on the
    held-out test split these checkpoints emit a narrow band. A prediction on an
    unmeasured triple that lands many multiples of that band away is not a
    confident call, it is the model extrapolating off the end of its own output
    distribution. The comparison is stored so a nomination can be read against it
    rather than against the label scale alone.
    """
    test = np.stack(
        [
            np.load(osp.join(RESULTS_DIR, f"cgt_predictions_{tag}_test.npy"))
            for tag in TAGS
        ]
    ).mean(axis=0)
    ens = df[TAGS].to_numpy().mean(axis=1)
    test_sd = float(test.std(ddof=0))
    return {
        "test_pred_sd": test_sd,
        "test_pred_min": float(test.min()),
        "test_pred_max": float(test.max()),
        "inference_1_pred_sd": float(ens.std(ddof=0)),
        "inference_1_pred_min": float(ens.min()),
        "inference_1_pred_max": float(ens.max()),
        "most_negative_in_test_sd": float(ens.min() / test_sd),
        "most_positive_in_test_sd": float(ens.max() / test_sd),
        "frac_beyond_test_range": float(
            ((ens < test.min()) | (ens > test.max())).mean()
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_table(
    df: pd.DataFrame,
    codes: np.ndarray,
    vocab: list[str],
    support: dict[str, int],
    rows: np.ndarray,
) -> pd.DataFrame:
    names = np.array(vocab, dtype=object)[codes[rows]]
    vals = df.iloc[rows][TAGS].to_numpy()
    out = pd.DataFrame(
        {
            "triple": [" + ".join(sorted(r)) for r in names],
            "gene1": names[:, 0],
            "gene2": names[:, 1],
            "gene3": names[:, 2],
        }
    )
    for j, tag in enumerate(TAGS):
        out[tag] = vals[:, j]
    out["ensemble_mean"] = vals.mean(axis=1)
    out["ensemble_sd"] = vals.std(axis=1, ddof=0)
    out["ensemble_min"] = vals.min(axis=1)
    out["ensemble_max"] = vals.max(axis=1)
    out["sign_agree"] = np.isin((vals > 0).sum(axis=1), [0, 3])
    out["min_train_records"] = [min(support[g] for g in r) for r in names]
    out["no_trigenic_labels"] = out["min_train_records"] == 0
    return out


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("loading three checkpoint prediction files ...")
    df, codes, vocab = load_predictions()
    print(f"  {len(df):,} triples over {len(vocab)} distinct genes")

    print("\nvalidity, defect two: resolving every gene in the space ...")
    space = check_gene_space(vocab)
    print(
        f"  {space['genes_in_space']} genes, "
        f"{len(space['unresolvable'])} unresolvable, "
        f"{len(space['renamed'])} needing alias resolution"
    )
    for r in space["renamed"]:
        print(f"    {r['as_scored']} -> {r['resolves_to']} ({r['status']})")

    print("\nvalidity, defect one: re-scoring a sample under each index space ...")
    index_check = check_index_space(df, codes, vocab, space["resolved"])
    for k, v in index_check.items():
        print(f"  {k}: {v}")

    calib = calibration(df)
    print("\ncalibration against the labeled test split ...")
    for k, v in calib.items():
        print(f"  {k}: {v:.4f}")

    validity = {
        "n_triples": int(len(df)),
        "n_genes": int(len(vocab)),
        "unresolvable_genes": space["unresolvable"],
        "renamed_genes": space["renamed"],
        "index_space_check": index_check,
        "label_sd": label_spread(),
        "calibration": calib,
    }
    with open(osp.join(RESULTS_DIR, "inference_1_validity.json"), "w") as f:
        json.dump(validity, f, indent=2)

    print("\ncheckpoint agreement ...")
    agree = agreement(df)
    agree.to_csv(osp.join(RESULTS_DIR, "inference_1_checkpoint_agreement.csv"), index=False)
    print(agree.to_string(index=False))

    support = training_support(space["resolved"])

    ens = df[TAGS].to_numpy().mean(axis=1)
    order = np.argsort(ens, kind="stable")
    neg_rows = order[:N_NEGATIVE]
    pos_rows = order[::-1][:N_POSITIVE]

    neg = build_table(df, codes, vocab, support, neg_rows)
    neg.insert(0, "direction", "negative")
    neg.insert(0, "rank", np.arange(1, len(neg) + 1))
    pos = build_table(df, codes, vocab, support, pos_rows)
    pos.insert(0, "direction", "positive")
    pos.insert(0, "rank", np.arange(1, len(pos) + 1))
    top = pd.concat([neg, pos], ignore_index=True)
    top.to_csv(osp.join(RESULTS_DIR, "inference_1_top_triples.csv"), index=False)

    sd = validity["label_sd"]
    n_agree = int(neg["sign_agree"].sum())
    print(
        f"\ntop {N_NEGATIVE} negative by ensemble mean: "
        f"{neg['ensemble_mean'].min():.4f} to {neg['ensemble_mean'].max():.4f}, "
        f"training-label sd {sd:.4f}"
    )
    print(f"  all three checkpoints agree on sign for {n_agree} of {N_NEGATIVE}")
    print(f"  median across-checkpoint spread {neg['ensemble_sd'].median():.4f}")
    print(f"  triples containing a gene with zero trigenic labels: "
          f"{int(neg['no_trigenic_labels'].sum())}")

    # Which genes carry the negative tail.
    tail = np.array(vocab, dtype=object)[codes[order[:10000]]]
    counts = pd.Series(tail.reshape(-1)).value_counts()
    freq = pd.DataFrame(
        {
            "gene": counts.index,
            "appearances_in_top_10000": counts.to_numpy(),
            "trigenic_train_records": [support[g] for g in counts.index],
        }
    )
    freq.to_csv(osp.join(RESULTS_DIR, "inference_1_gene_frequency.csv"), index=False)
    print("\ngenes most often in the 10,000 most negative triples:")
    print(freq.head(12).to_string(index=False))

    plot(df, agree, neg, freq, sd, calib)


def plot(
    df: pd.DataFrame,
    agree: pd.DataFrame,
    neg: pd.DataFrame,
    freq: pd.DataFrame,
    sd: float,
    calib: dict[str, float],
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )

    # a. Where the whole space sits against the training-label scale.
    ax = axes[0]
    ens = df[TAGS].to_numpy().mean(axis=1)
    ax.hist(ens, bins=160, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.2)
    ax.axvspan(
        calib["test_pred_min"],
        calib["test_pred_max"],
        color=PLOT_PALETTE[2],
        alpha=0.18,
        linewidth=0,
    )
    ax.axvline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"Ensemble predicted $\tau$")
    ax.set_ylabel("Triples")
    ax.set_yscale("log")
    ax.set_title("All 4,370,595 triples", fontsize=6)
    ax.text(
        0.02,
        0.93,
        "shaded: range the model\nemits on labeled test data",
        transform=ax.transAxes,
        fontsize=4.5,
        va="top",
    )

    # b. How far two training runs agree about the negative tail.
    ax = axes[1]
    for i, (_, row) in enumerate(agree.iterrows()):
        ax.plot(
            TOP_K_GRID,
            [row[f"overlap_top{k}"] for k in TOP_K_GRID],
            marker="o",
            ms=2.5,
            linewidth=0.9,
            color=PLOT_PALETTE[i % 6],
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=str(row["pair"]).replace("_", " "),
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("K most negative predictions")
    ax.set_ylabel("Fraction of top K shared")
    ax.set_title("Checkpoint agreement in the tail", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="upper left")

    # c. The nominations themselves, with the range over three checkpoints.
    ax = axes[2]
    show = neg.head(20).iloc[::-1]
    y = np.arange(len(show))
    ax.barh(
        y,
        show["ensemble_mean"],
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        height=0.72,
    )
    ax.errorbar(
        show["ensemble_mean"],
        y,
        xerr=[
            show["ensemble_mean"] - show["ensemble_min"],
            show["ensemble_max"] - show["ensemble_mean"],
        ],
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=1.2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(show["triple"], fontsize=4)
    ax.set_xlabel(r"Ensemble predicted $\tau$")
    ax.set_title("Top 20 negative, bars span 3 checkpoints", fontsize=6)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)

    fig.suptitle(
        "inference_1 trigenic nominations, ensembled over three training runs",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "inference_1_top_triples")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nwrote {stem}.svg")


if __name__ == "__main__":
    main()

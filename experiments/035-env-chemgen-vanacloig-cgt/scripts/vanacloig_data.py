# experiments/035-env-chemgen-vanacloig-cgt/scripts/vanacloig_data.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.vanacloig_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/vanacloig_data
"""The Vanacloig 2022 cells as tensors, their compound-cold folds, and the scoring rule.

Reads the cell table that experiment 033 flattened from its pooled store
(``flatten_cells.py``, slurm 2988) and keeps the 143,218 Vanacloig cells. A cell is one
strain in one environment: the queried deletion on the three-deletion sensitized host,
under one of 41 compounds. Every cell holds exactly one measurement, so there is no label
policy to choose here.

THE SPLIT holds out whole compounds. The 41 compounds are shuffled once with a fixed
seed and cut into ``n_folds`` groups. Fold ``k`` tests on group ``k``, validates on
``n_val`` compounds drawn from the rest, and trains on what remains. Validation picks the
epoch; the test group is never used for a choice.

THE SCORE is per held-out compound, over that compound's genes: Spearman and Pearson
between prediction and measurement, on two targets. The raw target is the served
response. The centered target subtracts each gene's mean over the TRAINING compounds from
both sides, so what remains is the compound-specific response and a model that knows only
which genes are generally sick scores zero. Each compound's ceiling, the square root of
its reliability index, is carried beside its score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import pearsonr, spearmanr

VANACLOIG = "EnvChemgenVanacloig2022Dataset"
#: The three efflux-regulator deletions of the sensitized host (PDR1, PDR3, SNQ2).
HOST_GENES: tuple[str, str, str] = ("YGL013C", "YBL005W", "YDR011W")
#: A compound is scored only over at least this many genes.
MIN_GENES = 10


class VanacloigCells(BaseModel):
    """Every Vanacloig cell, indexed against one gene order and one compound order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    genes: list[str]  # queried genes, the row order of the matrix view
    compounds: list[str]  # compound names, in a fixed order
    inchikeys: list[str]
    gene_of_cell: NDArray[np.int64]  # [n] index into ``genes``
    compound_of_cell: NDArray[np.int64]  # [n] index into ``compounds``
    response: NDArray[np.float64]  # [n] served log2 ratio, negative is a defect
    response_se: NDArray[np.float64]  # [n] served standard error
    compound_features: NDArray[np.float64]  # [n_compounds, n_features]

    def matrix(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``values`` as genes by compounds, NaN where a cell was not measured."""
        out = np.full((len(self.genes), len(self.compounds)), np.nan)
        out[self.gene_of_cell, self.compound_of_cell] = values
        return out


class Fold(BaseModel):
    """The compounds of one fold, as indices into ``VanacloigCells.compounds``."""

    fold: int
    train: list[int]
    val: list[int]
    test: list[int]


def load_cells(cell_table: str, embedding_npz: str) -> VanacloigCells:
    table = pd.read_parquet(
        cell_table,
        columns=[
            "dataset",
            "query_gene",
            "genes",
            "n_measurements",
            "n_compounds",
            "compound_names",
            "inchikeys",
            "responses",
            "response_ses",
        ],
        filters=[("dataset", "==", VANACLOIG)],
    ).reset_index(drop=True)
    assert (table["n_measurements"] == 1).all(), "a Vanacloig cell folds measurements"
    assert (table["n_compounds"] == 1).all(), "a Vanacloig cell doses several compounds"
    host = ";".join(sorted(HOST_GENES))
    expected = table["query_gene"].map(lambda q: ";".join(sorted((q, *HOST_GENES))))
    assert (table["genes"] == expected).all(), f"a strain is not the query plus {host}"

    genes = sorted(table["query_gene"].unique())
    pairs = table[["compound_names", "inchikeys"]].drop_duplicates()
    pairs = pairs.sort_values("compound_names", ignore_index=True)
    assert pairs["compound_names"].is_unique and pairs["inchikeys"].is_unique
    gene_index = {g: i for i, g in enumerate(genes)}
    compound_index = {c: i for i, c in enumerate(pairs["compound_names"])}

    embedded = np.load(embedding_npz, allow_pickle=True)
    row = {key: i for i, key in enumerate(embedded["inchikey"])}
    features = np.stack([embedded["X"][row[key]] for key in pairs["inchikeys"]])

    return VanacloigCells(
        genes=genes,
        compounds=list(pairs["compound_names"]),
        inchikeys=list(pairs["inchikeys"]),
        gene_of_cell=table["query_gene"].map(gene_index).to_numpy(dtype=np.int64),
        compound_of_cell=table["compound_names"]
        .map(compound_index)
        .to_numpy(dtype=np.int64),
        response=table["responses"].map(lambda r: r[0]).to_numpy(dtype=np.float64),
        response_se=table["response_ses"]
        .map(lambda s: s[0])
        .to_numpy(dtype=np.float64),
        compound_features=features.astype(np.float64),
    )


def make_folds(n_compounds: int, n_folds: int, n_val: int, seed: int) -> list[Fold]:
    """Compound-cold folds: every compound is tested exactly once."""
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(n_compounds), n_folds)
    folds = []
    for k, test in enumerate(groups):
        rest = np.concatenate([g for j, g in enumerate(groups) if j != k])
        # the validation compounds are drawn per fold from its own generator, so adding
        # or reordering folds does not move another fold's validation set
        val = np.random.default_rng([seed, k]).choice(rest, size=n_val, replace=False)
        train = np.setdiff1d(rest, val)
        folds.append(
            Fold(
                fold=k,
                train=sorted(int(i) for i in train),
                val=sorted(int(i) for i in val),
                test=sorted(int(i) for i in test),
            )
        )
    tested = sorted(i for f in folds for i in f.test)
    assert tested == list(range(n_compounds)), "a compound is tested twice or never"
    return folds


def ceiling(response: NDArray[np.float64], se: NDArray[np.float64]) -> float:
    """Square root of 1 - mean(SE^2) / Var(response); zero where that is negative."""
    ok = np.isfinite(response) & np.isfinite(se)
    reliability = 1.0 - float(np.mean(se[ok] ** 2)) / float(
        np.var(response[ok], ddof=1)
    )
    return float(np.sqrt(reliability)) if reliability > 0 else 0.0


def score_compounds(
    cells: VanacloigCells,
    prediction: NDArray[np.float64],
    train: list[int],
    held_out: list[int],
) -> pd.DataFrame:
    """One row per held-out compound and target.

    ``prediction`` is genes by compounds in the units of the served response; only the
    ``held_out`` columns are read. The gene mean that centers both sides is taken over
    the ``train`` columns of the MEASURED matrix, never the prediction.
    """
    measured = cells.matrix(cells.response)
    se = cells.matrix(cells.response_se)
    gene_mean = np.nanmean(measured[:, train], axis=1)
    rows = []
    for j in held_out:
        for target in ("raw", "centered"):
            shift = gene_mean if target == "centered" else np.zeros_like(gene_mean)
            obs = measured[:, j] - shift
            pred = prediction[:, j] - shift
            ok = np.isfinite(obs) & np.isfinite(pred)
            constant = ok.sum() < MIN_GENES or np.std(pred[ok]) == 0
            rows.append(
                {
                    "compound": cells.compounds[j],
                    "target": target,
                    "n_genes": int(ok.sum()),
                    "spearman": (
                        float("nan")
                        if constant
                        else float(spearmanr(pred[ok], obs[ok])[0])
                    ),
                    "pearson": (
                        float("nan")
                        if constant
                        else float(pearsonr(pred[ok], obs[ok])[0])
                    ),
                    "ceiling": ceiling(obs, se[:, j]),
                }
            )
    return pd.DataFrame(rows)

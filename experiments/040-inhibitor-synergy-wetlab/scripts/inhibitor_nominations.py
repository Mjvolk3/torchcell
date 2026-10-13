# experiments/040-inhibitor-synergy-wetlab/scripts/inhibitor_nominations.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.inhibitor_nominations]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/inhibitor_nominations
"""Composed mixture profiles, nominated genes, and GO enrichment (claim 4, post-analysis).

Reads ``results/profile_matrix.csv``, ``profile_all.csv`` and ``profile_matrix.json``
written by ``inhibitor_profiles.py``. SIGN: NEGATIVE = sick, as there.

NOMINATIONS ARE HYPOTHESES for a future screen under each combination; no screen tests
them.

COMPOSITION. Each single profile (one column of a matrix, target ``raw`` by default: what
a screen would recover, general stress genes included) is turned into a z score within
itself, ``z = (x - mean) / sd`` over its measured genes, so measured and predicted
columns, whose scales differ (a ridge prediction is shrunk), compose on one scale. A
combination's composed profile is a rule over its members' z, for a gene every member
carries:

- ``sum`` and ``mean``: the same order within a combination; they differ only across
  combinations of different size.
- ``min``: the most sick member decides (a gene is sick when ANY member makes it sick).
- ``max``: the least sick member decides (sick only when EVERY member makes it sick).

The HET analysis (claim 3) picks the rule; every rule is written here.

CATEGORIES per combination (15 pairs, 20 triples), sensitive = ``z < -Z``, resistant =
``z > Z``:

- ``core``: sensitive in at least ``CORE_MIN`` of the six singles (the same genes for
  every combination; ranked by that combination's composed score).
- ``specific``: sensitive in exactly one of the six singles, and that single is a member.
- ``conflict``: sensitive in one member and resistant in another.
- ``composed_top``: the ``TOP_N`` lowest composed scores.

Each row carries every member's z and the evidence of each member's column: its source
(measured or predicted) and that source's reliability (measured: replicate reliability or,
for Hoepfner, the agreement of its two doses; predicted: the compound-cold centered
Spearman against a measured profile where one exists, else the leave-one-compound-out
median over the 32 Vanacloig compounds).

GO ENRICHMENT. Annotations are the S288C GFF ``Ontology_term`` GO terms the torchcell
genome serves (``SCerevisiaeGenome.go_genes``, ``DATA_ROOT/data/sgd/genome``), propagated
to every ``is_a`` ancestor in the genome's ``go.obo``. Background: the build-002
Vanacloig genes with at least one annotation; terms with 5 to 500 background genes are
tested. One-sided hypergeometric, Benjamini-Hochberg over the tested terms of each set,
significant at FDR < ``FDR``. Sets: the ``TOP_N`` most sensitive genes of (a) each
measured single profile, (b) each predicted single profile, (c) the core set and each
pair's composed top ``TOP_N`` under each rule; plus a CONTROL, the gene mean over the 32
build-002 compounds, the profile a model with no compound-specific signal would output.
Overlap of significant terms, measured against predicted for the same compound, is
reported beside the same overlap against the control.

Writes ``results/nominations_<rule>.csv``, ``nominations_summary.csv``, ``go_*.csv``,
``go_summary.json`` and a figure under ``ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import os.path as osp
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.stats import hypergeom

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "040-inhibitor-synergy-wetlab")
DATA_ROOT = os.environ["DATA_ROOT"]

Z = 2.0
CORE_MIN = 4
TOP_N = 100
FDR = 0.05
TERM_MIN, TERM_MAX = 5, 500
TARGET = "raw"
RULES = ("sum", "mean", "max", "min")
VERSIONS = ("best_available", "all_predicted")
Rule = Literal["sum", "mean", "max", "min"]


class GeneSetResult(BaseModel):
    """One enrichment run: the set, its size, and the significant term ids."""

    set_id: str
    kind: str
    compound: str | None
    source: str | None
    target: str | None
    n_study: int  # study genes with at least one annotation
    n_background: int
    n_tested_terms: int
    significant: list[str]


# --------------------------------------------------------------------------- composition
def zscore(v: NDArray[np.float64]) -> NDArray[np.float64]:
    ok = np.isfinite(v)
    return (v - v[ok].mean()) / v[ok].std(ddof=1)


def compose(z: NDArray[np.float64], rule: Rule) -> NDArray[np.float64]:
    """``z`` is genes by members; NaN where any member lacks the gene."""
    ops = {"sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min}
    return ops[rule](z, axis=1)


def nominate(
    matrix: pd.DataFrame,
    version: str,
    six: list[str],
    abbr: dict[str, str],
    evidence: dict[str, str],
    rule: Rule,
) -> pd.DataFrame:
    z = np.stack(
        [zscore(matrix[f"{version}|{TARGET}|{c}"].to_numpy()) for c in six], axis=1
    )
    sensitive = z < -Z
    resistant = z > Z
    n_sensitive = sensitive.sum(axis=1)
    core = n_sensitive >= CORE_MIN
    only = np.where(n_sensitive == 1, sensitive.argmax(axis=1), -1)
    rows = []
    for size in (2, 3):
        for members in itertools.combinations(range(6), size):
            m = list(members)
            composed = compose(z[:, m], rule)
            finite = np.isfinite(composed)
            order = np.argsort(np.where(finite, composed, np.inf), kind="stable")
            rank = np.empty(len(order), dtype=int)
            rank[order] = np.arange(1, len(order) + 1)
            categories = {
                "core": core & finite,
                "specific": np.isin(only, m) & finite,
                "conflict": sensitive[:, m].any(axis=1)
                & resistant[:, m].any(axis=1)
                & finite,
                "composed_top": finite & (rank <= TOP_N),
            }
            name = " + ".join(six[i] for i in m)
            for category, mask in categories.items():
                for g in np.flatnonzero(mask):
                    row = {
                        "version": version,
                        "rule": rule,
                        "combination": name,
                        "combination_abbr": "+".join(abbr[six[i]] for i in m),
                        "n_members": size,
                        "category": category,
                        "gene": matrix["gene"].iat[g],
                        "gene_name": matrix["gene_name"].iat[g],
                        "composed_z": float(composed[g]),
                        "composed_rank": int(rank[g]),
                        "n_singles_sensitive": int(n_sensitive[g]),
                        "specific_to": six[only[g]] if only[g] >= 0 else None,
                        "evidence": " | ".join(
                            f"{six[i]}: z={z[g, i]:.2f}, {evidence[six[i]]}" for i in m
                        ),
                    }
                    for i, c in enumerate(six):
                        row[f"z_{abbr[c]}"] = float(z[g, i])
                    rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["version", "combination", "category", "composed_rank"], ignore_index=True
    )


# --------------------------------------------------------------------------- GO
class GoAnnotation(BaseModel):
    """Propagated GO annotations restricted to the background, with term metadata."""

    model_config = {"arbitrary_types_allowed": True}

    background: list[str]
    term_genes: dict[str, set[str]]
    names: dict[str, str]
    namespaces: dict[str, str]
    obo_path: str
    obo_sha256: str


def go_annotation(universe: list[str]) -> GoAnnotation:
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    dag = genome.go_dag
    universe_set = set(universe)
    gene_terms: dict[str, set[str]] = {}
    for term, genes in genome.go_genes.items():
        if term not in dag:  # a GFF term the OBO release no longer carries
            continue
        expanded = {dag[term].id} | dag[term].get_all_parents()
        for g in genes:
            if g in universe_set:
                gene_terms.setdefault(g, set()).update(expanded)
    term_genes: dict[str, set[str]] = {}
    for g, terms in gene_terms.items():
        for t in terms:
            term_genes.setdefault(t, set()).add(g)
    term_genes = {
        t: gs for t, gs in term_genes.items() if TERM_MIN <= len(gs) <= TERM_MAX
    }
    obo = str(genome._obo_path)
    with open(obo, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return GoAnnotation(
        background=sorted(gene_terms),
        term_genes=term_genes,
        names={t: dag[t].name for t in term_genes},
        namespaces={t: dag[t].namespace for t in term_genes},
        obo_path=obo,
        obo_sha256=sha,
    )


def bh(p: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def enrich(study: set[str], go: GoAnnotation) -> pd.DataFrame:
    bg = set(go.background)
    s = study & bg
    big_n, n = len(bg), len(s)
    terms = list(go.term_genes)
    k = np.array([len(go.term_genes[t] & s) for t in terms])
    big_k = np.array([len(go.term_genes[t]) for t in terms])
    p = hypergeom.sf(k - 1, big_n, big_k, n)
    return pd.DataFrame(
        {
            "term": terms,
            "name": [go.names[t] for t in terms],
            "namespace": [go.namespaces[t] for t in terms],
            "k_study_in_term": k,
            "n_study": n,
            "K_term": big_k,
            "N_background": big_n,
            "fold_enrichment": (k / max(n, 1)) / (big_k / big_n),
            "p": p,
            "fdr": bh(p),
            "genes": [";".join(sorted(go.term_genes[t] & s)) for t in terms],
        }
    )


def top_genes(values: NDArray[np.float64], genes: list[str]) -> set[str]:
    ok = np.flatnonzero(np.isfinite(values))
    return {genes[i] for i in ok[np.argsort(values[ok], kind="stable")[:TOP_N]]}


def run_set(
    set_id: str,
    kind: str,
    study: set[str],
    go: GoAnnotation,
    compound: str | None,
    source: str | None,
    target: str | None,
) -> tuple[GeneSetResult, pd.DataFrame]:
    table = enrich(study, go).sort_values("p", ignore_index=True)
    sig = table[table["fdr"] < FDR]
    result = GeneSetResult(
        set_id=set_id,
        kind=kind,
        compound=compound,
        source=source,
        target=target,
        n_study=int(table["n_study"].iat[0]),
        n_background=len(go.background),
        n_tested_terms=len(table),
        significant=sig["term"].tolist(),
    )
    # every significant term, and the ten best by p whether significant or not
    kept = table[(table["fdr"] < FDR) | (table.index < 10)].copy()
    kept.insert(0, "set_id", set_id)
    kept.insert(1, "kind", kind)
    kept.insert(2, "compound", compound)
    kept.insert(3, "source", source)
    kept.insert(4, "target", target)
    kept["significant"] = kept["fdr"] < FDR
    return result, kept


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


# --------------------------------------------------------------------------- figure
def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "svg.fonttype": "none",
        }
    )


def go_figure(
    singles: pd.DataFrame, panels: list[tuple[str, str, str, str]], target: str
) -> str:
    """Top terms of each measured profile: -log10 FDR measured, predicted, control."""
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(36 * len(panels))),
        constrained_layout=True,
    )
    for k, (ax, (title, measured_id, predicted_id, control_id)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        m = singles[singles["set_id"] == measured_id].sort_values("p").head(8)
        terms = m["term"].tolist()

        def lookup(set_id: str) -> list[float]:
            t = singles[singles["set_id"] == set_id].set_index("term")["fdr"]
            # a term outside a set's kept rows was neither significant nor in its top 10
            return [float(-np.log10(t[x])) if x in t.index else 0.0 for x in terms]

        y = np.arange(len(terms))
        h = 0.27
        ax.barh(
            y - h,
            -np.log10(m["fdr"].to_numpy()),
            h,
            color=PLOT_PALETTE[k],
            edgecolor="black",
            lw=0.3,
            label="measured",
        )
        ax.barh(
            y,
            lookup(predicted_id),
            h,
            color=PLOT_PALETTE_FILL[k],
            edgecolor="black",
            lw=0.3,
            label="predicted",
        )
        ax.barh(
            y + h,
            lookup(control_id),
            h,
            color=PLOT_PALETTE[5],
            edgecolor="black",
            lw=0.3,
            label="32-compound gene mean",
        )
        ax.axvline(-np.log10(FDR), color="black", lw=0.4, ls="--")
        labels = [
            f"{n[:55]} ({ns[0].upper()})"
            for n, ns in zip(m["name"], m["namespace"], strict=True)
        ]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("-log10 FDR (BH); dashed line FDR 0.05")
        ax.set_title(title, loc="left")
        if k == 0:
            ax.legend(loc="center right", frameon=False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, f"go_measured_vs_predicted_{target}_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.svg")
    return stem + ".svg"


# --------------------------------------------------------------------------- main
def main() -> None:
    style()
    matrix = pd.read_csv(osp.join(RESULTS_DIR, "profile_matrix.csv"))
    allp = pd.read_csv(osp.join(RESULTS_DIR, "profile_all.csv"))
    with open(osp.join(RESULTS_DIR, "profile_matrix.json")) as f:
        meta = json.load(f)
    six = [i["name"] for i in meta["inhibitors"]]
    abbr = {i["name"]: i["abbreviation"] for i in meta["inhibitors"]}
    genes = matrix["gene"].tolist()
    assert genes == allp["gene"].tolist()
    loco_median = meta["loco_median_centered_spearman"]

    def evidence_of(choice: dict) -> str:
        if choice["source"] == "measured_vanacloig_b002":
            return f"measured Vanacloig b002 (replicate reliability {choice['measured_reliability']:.2f})"
        if choice["source"] == "measured_hoepfner_hop":
            d = meta["hoepfner_hop_acetate"]["dose_spearman"]
            return (
                f"measured Hoepfner HOP sodium acetate (125 vs 150 mM Spearman {d:.2f})"
            )
        check = choice["prediction_check_spearman_centered"]
        if check is None:
            return f"predicted ridge, never measured (LOCO median centered Spearman {loco_median:.2f}, n=32)"
        return f"predicted ridge (centered Spearman vs measured {check:.2f})"

    evidence = {
        v: {
            c["inhibitor"]: evidence_of(c) for c in meta["choices"] if c["version"] == v
        }
        for v in VERSIONS
    }

    # ---- nominations
    summaries = []
    for rule in RULES:
        frames = [nominate(matrix, v, six, abbr, evidence[v], rule) for v in VERSIONS]
        nom = pd.concat(frames, ignore_index=True)
        nom.insert(
            0,
            "sign",
            "z of log2(inhibitor/control) within the single profile; NEGATIVE = sick",
        )
        nom.to_csv(osp.join(RESULTS_DIR, f"nominations_{rule}.csv"), index=False)
        summaries.append(
            nom.groupby(["version", "rule", "combination", "n_members", "category"])
            .size()
            .rename("n_genes")
            .reset_index()
        )
        print(f"nominations_{rule}.csv: {len(nom)} rows")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(osp.join(RESULTS_DIR, "nominations_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(
        summary[summary["rule"] == "sum"]
        .pivot_table(
            index=["version", "n_members"],
            columns="category",
            values="n_genes",
            aggfunc=["median", "min", "max"],
        )
        .to_string()
    )

    # ---- GO
    go = go_annotation(genes)
    print(
        f"GO: {len(go.background)} of {len(genes)} matrix genes annotated, {len(go.term_genes)} "
        f"terms with {TERM_MIN}-{TERM_MAX} genes; obo {go.obo_path} sha256 {go.obo_sha256[:12]}"
    )
    results: list[GeneSetResult] = []
    tables: list[pd.DataFrame] = []

    def add(set_id, kind, study, compound=None, source=None, target=None):
        r, t = run_set(set_id, kind, study, go, compound, source, target)
        results.append(r)
        tables.append(t)
        return r

    singles_spec = [
        # (compound, measured profile label, predicted profile label)
        ("furfural", "furfural|measured_b002", "furfural|predicted_loco"),
        (
            "5-(hydroxymethyl)furfural",
            "5-(hydroxymethyl)furfural|measured_b002",
            "5-(hydroxymethyl)furfural|predicted_loco",
        ),
        (
            "acetic acid",
            "sodium acetate|measured_hoepfner_hop",
            "acetic acid|predicted_all32",
        ),
        (
            "levulinic acid",
            "levulinic acid|measured_b001",
            "levulinic acid|predicted_all32",
        ),
    ]
    predicted_only = [
        "formic acid|predicted_all32",
        "lactic acid|predicted_all32",
        "sodium acetate|predicted_all32",
    ]
    overlap_rows = []
    for target in ("raw", "centered"):
        control = (
            add(
                f"control|gene_mean_32|{target}",
                "control",
                top_genes(allp["vanacloig_b002_gene_mean_32|raw"].to_numpy(), genes),
                "gene mean over 32",
                "vanacloig_b002_gene_mean_32",
                "raw",
            )
            if target == "raw"
            else None
        )
        for compound, measured, predicted in singles_spec:
            rm = add(
                f"single|{measured}|{target}",
                "single_measured",
                top_genes(allp[f"{measured}|{target}"].to_numpy(), genes),
                compound,
                measured,
                target,
            )
            rp = add(
                f"single|{predicted}|{target}",
                "single_predicted",
                top_genes(allp[f"{predicted}|{target}"].to_numpy(), genes),
                compound,
                predicted,
                target,
            )
            sm, sp = set(rm.significant), set(rp.significant)
            gm = top_genes(allp[f"{measured}|{target}"].to_numpy(), genes)
            gp = top_genes(allp[f"{predicted}|{target}"].to_numpy(), genes)
            row = {
                "compound": compound,
                "target": target,
                "measured": measured,
                "predicted": predicted,
                "top_genes_shared": len(gm & gp),
                "top_n": TOP_N,
                "n_sig_measured": len(sm),
                "n_sig_predicted": len(sp),
                "n_sig_shared": len(sm & sp),
                "jaccard_sig_terms": jaccard(sm, sp),
                "fraction_measured_terms_recovered": len(sm & sp) / len(sm)
                if sm
                else float("nan"),
            }
            if target == "raw":
                assert control is not None
                sc = set(control.significant)
                gc = top_genes(
                    allp["vanacloig_b002_gene_mean_32|raw"].to_numpy(), genes
                )
                row.update(
                    {
                        "control_top_genes_shared_with_measured": len(gm & gc),
                        "n_sig_control": len(sc),
                        "n_sig_shared_measured_control": len(sm & sc),
                        "jaccard_sig_terms_measured_control": jaccard(sm, sc),
                        "fraction_measured_terms_recovered_by_control": len(sm & sc)
                        / len(sm)
                        if sm
                        else float("nan"),
                        "n_sig_shared_measured_predicted_not_control": len(
                            (sm & sp) - sc
                        ),
                        "control_top_genes_shared_with_predicted": len(gp & gc),
                        "n_sig_shared_predicted_control": len(sp & sc),
                    }
                )
            overlap_rows.append(row)
        for predicted in predicted_only:
            add(
                f"single|{predicted}|{target}",
                "single_predicted",
                top_genes(allp[f"{predicted}|{target}"].to_numpy(), genes),
                predicted.split("|")[0],
                predicted,
                target,
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        osp.join(RESULTS_DIR, "go_overlap_measured_vs_predicted.csv"), index=False
    )
    print(overlap.to_string(index=False))
    singles = pd.concat(tables, ignore_index=True)
    singles.to_csv(osp.join(RESULTS_DIR, "go_singles.csv"), index=False)

    # (c) the core set and every pair's composed top N, per version and rule
    combo_tables: list[pd.DataFrame] = []
    combo_rows = []
    for version in VERSIONS:
        nom = pd.read_csv(osp.join(RESULTS_DIR, "nominations_sum.csv"))
        core = set(
            nom[(nom["version"] == version) & (nom["category"] == "core")]["gene"]
        )
        r, t = run_set(f"core|{version}", "core", core, go, None, version, TARGET)
        results.append(r)
        combo_tables.append(t)
        combo_rows.append(
            {
                "version": version,
                "rule": None,
                "combination": "core",
                "n_genes": len(core),
                "n_sig": len(r.significant),
            }
        )
        for rule in RULES:
            nom = pd.read_csv(osp.join(RESULTS_DIR, f"nominations_{rule}.csv"))
            nom = nom[
                (nom["version"] == version)
                & (nom["n_members"] == 2)
                & (nom["category"] == "composed_top")
            ]
            for combination, part in nom.groupby("combination"):
                set_id = f"pair|{version}|{rule}|{combination}"
                r, t = run_set(
                    set_id,
                    "pair_composed_top",
                    set(part["gene"]),
                    go,
                    combination,
                    version,
                    TARGET,
                )
                results.append(r)
                t.insert(5, "rule", rule)
                combo_tables.append(t)
                top_terms = t[t["significant"]].head(3)["name"].tolist()
                combo_rows.append(
                    {
                        "version": version,
                        "rule": rule,
                        "combination": combination,
                        "n_genes": len(part),
                        "n_sig": len(r.significant),
                        "top_terms": "; ".join(top_terms),
                    }
                )
    pd.concat(combo_tables, ignore_index=True).to_csv(
        osp.join(RESULTS_DIR, "go_combinations.csv"), index=False
    )
    combos = pd.DataFrame(combo_rows)
    combos.to_csv(osp.join(RESULTS_DIR, "go_combinations_summary.csv"), index=False)
    print(
        combos[
            combos["rule"].isin([None, "sum", "min"]) | combos["rule"].isna()
        ].to_string(index=False)
    )

    with open(osp.join(RESULTS_DIR, "go_summary.json"), "w") as f:
        json.dump(
            {
                "obo_path": go.obo_path,
                "obo_sha256": go.obo_sha256,
                "n_background": len(go.background),
                "n_tested_terms": len(go.term_genes),
                "term_size": [TERM_MIN, TERM_MAX],
                "fdr": FDR,
                "top_n": TOP_N,
                "propagation": "is_a ancestors (goatools GODag.get_all_parents)",
                "sets": [r.model_dump() for r in results],
            },
            f,
            indent=2,
        )

    # ---- figure
    panels = [
        (
            "furfural: measured b002 vs LOCO prediction",
            "single|furfural|measured_b002|{t}",
            "single|furfural|predicted_loco|{t}",
        ),
        (
            "5-HMF: measured b002 (reliability -0.05) vs LOCO prediction",
            "single|5-(hydroxymethyl)furfural|measured_b002|{t}",
            "single|5-(hydroxymethyl)furfural|predicted_loco|{t}",
        ),
        (
            "acetic acid: Hoepfner HOP sodium acetate vs acid prediction",
            "single|sodium acetate|measured_hoepfner_hop|{t}",
            "single|acetic acid|predicted_all32|{t}",
        ),
    ]
    for target in ("raw", "centered"):
        go_figure(
            singles,
            [
                (
                    title,
                    m.format(t=target),
                    p.format(t=target),
                    "control|gene_mean_32|raw",
                )
                for title, m, p in panels
            ],
            target,
        )


if __name__ == "__main__":
    main()

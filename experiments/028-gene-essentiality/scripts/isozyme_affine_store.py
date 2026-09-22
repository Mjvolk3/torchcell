# experiments/028-gene-essentiality/scripts/isozyme_affine_store.py
# [[experiments.028-gene-essentiality.scripts.isozyme_affine_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/isozyme_affine_store

r"""The gene-protein-reaction redundancy of each gene, as affine columns beside the solve.

``misclass_enrichment.py`` measured which essential genes the transformer with the
layer-free solve ranks low on the released 195: the 19 of 31 whose FBA knockout growth
ratio is 1.0 (mean misrank 0.134 against 0.025, effect 0.91, $q$ 0.003). For those genes
the solve reroutes, through an isozyme or around the reaction, so the deletion's flux
vector is close to the wild type's and cannot mark them. What the vector does not carry is
HOW the gene is dispensable to the program: whether another gene set catalyzes its
reactions, and how many. This script reads that from yeast-GEM's gene-protein-reaction
rules, expanded to disjunctive normal form (one term per alternative gene set).

Per gene, over the reactions whose rule names it:

* ``gpr_n_reactions``: how many reactions;
* ``gpr_n_sole``: how many of them the gene's deletion alone disables (the rule evaluates
  false with the gene knocked out);
* ``gpr_n_covered``: how many stay enabled, so an alternative gene set covers them;
* ``gpr_min_alternatives`` and ``gpr_mean_alternatives``: the number of alternative gene
  sets (terms not containing the gene), minimum and mean over its reactions;
* ``gpr_max_cosubunits``: the largest number of other genes in a term with it, its complex
  membership.

All are non-negative counts, so the trainer's $\log(1 + |v|)$ keeps their order. The store
``ess_solve_encodings_YPD_pfbagpr.npz`` is the pFBA deletion vector with the six columns
appended, in the solve store's shape, read through ``ESS_SOLVE_TABLE``. Writes
``results/isozyme_affine_store.json`` with the counts and, for the released held-out
essential genes, the columns beside the growth ratio.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/isozyme_affine_store.py
"""

from __future__ import annotations

import ast
import csv
import gzip
import itertools
import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv

from torchcell.metabolism.yeast_GEM import YeastGEM

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ess_reference_flux_table import (  # noqa: E402
    DerivedStoreProvenance,
    StoreProvenance,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PFBA_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD.npz")
#: ``--store-tag nuc`` reads ``ess_solve_encodings_YPD_nuc.npz`` (the store rebuilt on the
#: corrected YPD) and writes ``..._YPD_nuc_<suffix>.npz`` beside the untagged stores.
STORE_TAG = ""


def stem(name: str) -> str:
    """``ess_solve_encodings_YPD<_tag><_name>.npz`` under the table directory."""
    tag = f"_{STORE_TAG}" if STORE_TAG else ""
    return osp.join(TABLE_DIR, f"ess_solve_encodings_YPD{tag}{name}.npz")


COLUMNS = [
    "gpr_n_reactions",
    "gpr_n_sole",
    "gpr_n_covered",
    "gpr_min_alternatives",
    "gpr_mean_alternatives",
    "gpr_max_cosubunits",
]
#: A rule whose normal form passes this many terms is a modeling artifact worth seeing.
MAX_TERMS = 20000


def normal_form(node: ast.AST) -> list[frozenset[str]]:
    """A rule's alternative gene sets: OR concatenates, AND takes the product."""
    if isinstance(node, ast.Name):
        return [frozenset([node.id])]
    assert isinstance(node, ast.BoolOp), ast.dump(node)
    parts = [normal_form(v) for v in node.values]
    if isinstance(node.op, ast.Or):
        return list(dict.fromkeys(t for p in parts for t in p))
    n_terms = int(np.prod([len(p) for p in parts]))
    assert n_terms <= MAX_TERMS, n_terms
    return list(
        dict.fromkeys(frozenset().union(*combo) for combo in itertools.product(*parts))
    )


def main() -> None:
    global STORE_TAG
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--store-tag", default="")
    parser.add_argument(
        "--genes",
        choices=["store", "gem"],
        default="store",
        help=(
            "store: the source store's genes, and every derived store beside it; gem: "
            "every yeast-GEM gene from the model alone, writing only the rules-only "
            "store as ess_solve_encodings_GEM_gpronly.npz (round 120, the betaxanthin "
            "and amino-acid targets, whose stores cover every GEM gene)."
        ),
    )
    args = parser.parse_args()
    STORE_TAG = args.store_tag
    model = YeastGEM().model
    per_gene: dict[str, list[tuple[int, int, int]]] = {}
    largest = 0
    for rxn in model.reactions:
        if rxn.gpr.body is None:
            continue
        terms = normal_form(rxn.gpr.body)
        # A term that contains another is redundant: the smaller set already suffices.
        terms = [t for t in terms if not any(o < t for o in terms)]
        largest = max(largest, len(terms))
        for gene in set().union(*terms):
            with_gene = [t for t in terms if gene in t]
            without = [t for t in terms if gene not in t]
            per_gene.setdefault(gene, []).append(
                (len(without), max(len(t) - 1 for t in with_gene), int(not without))
            )

    if args.genes == "gem":
        genes = sorted(g.id for g in model.genes)
    else:
        source = stem("")
        pfba = np.load(source, allow_pickle=False)
        source_provenance = StoreProvenance.model_validate_json(str(pfba["provenance"]))
        genes = [str(g) for g in pfba["genes"]]
    block = np.zeros((len(genes), len(COLUMNS)), dtype=np.float32)
    for i, gene in enumerate(genes):
        rows = per_gene.get(gene, [])
        if not rows:
            continue
        alternatives = [r[0] for r in rows]
        n_sole = sum(r[2] for r in rows)
        block[i] = [
            len(rows),
            n_sole,
            len(rows) - n_sole,
            min(alternatives),
            float(np.mean(alternatives)),
            max(r[1] for r in rows),
        ]

    # The counts WITHOUT the flux vector, so the affine can be given the rules alone.
    # Round 78 measured the 4,131 flux columns at +0.010 over the four growth ratios and
    # round 77 measured these six counts at +0.019 over the vector; the question this
    # store answers is whether the solve is needed at all on this endpoint.
    alone = (
        osp.join(TABLE_DIR, "ess_solve_encodings_GEM_gpronly.npz")
        if args.genes == "gem"
        else stem("_gpronly")
    )
    np.savez_compressed(
        alone,
        genes=np.array(genes),
        v_deletion=block,
        v_wild_type=np.zeros(len(COLUMNS), dtype=np.float32),
        rxn_ids=np.array(COLUMNS),
        provenance=DerivedStoreProvenance(
            method=(
                "six columns of gene-protein-reaction redundancy from yeast-GEM's "
                "rules in disjunctive normal form, with NO flux vector; written by "
                "isozyme_affine_store.py"
                + (" over every yeast-GEM gene" if args.genes == "gem" else "")
            ),
            gem_version=YeastGEM().version,
            columns=COLUMNS,
        ).model_dump_json(),
    )
    print(f"{alone}: {block.shape}")
    if args.genes == "gem":
        print(
            f"{len(genes)} GEM genes: {int((block[:, 0] > 0).sum())} with a rule, "
            f"{int((block[:, 1] > 0).sum())} with a sole reaction"
        )
        return

    # Which of the six carries it: the sole-reaction count alone, and the other five
    # without it. Round 77 located the gain on the genes holding a reaction nothing else
    # catalyzes, which is exactly what ``gpr_n_sole`` counts, so this is the test of
    # whether one number per gene is the whole of it.
    sole = COLUMNS.index("gpr_n_sole")
    keep = [i for i in range(len(COLUMNS)) if i != sole]
    for suffix, idx in (("pfbagprsole", [sole]), ("pfbagprnosole", keep)):
        out_path = stem(f"_{suffix}")
        names = [COLUMNS[i] for i in idx]
        np.savez_compressed(
            out_path,
            genes=np.array(genes),
            v_deletion=np.concatenate(
                [pfba["v_deletion"].astype(np.float32), block[:, idx]], axis=1
            ),
            v_wild_type=np.concatenate(
                [
                    pfba["v_wild_type"].astype(np.float32),
                    np.zeros(len(idx), dtype=np.float32),
                ]
            ),
            rxn_ids=np.array([str(r) for r in pfba["rxn_ids"]] + names),
            provenance=DerivedStoreProvenance(
                method=(
                    "the pFBA deletion vector with a SUBSET of the "
                    "gene-protein-reaction redundancy columns appended; written by "
                    "isozyme_affine_store.py"
                ),
                gem_version=YeastGEM().version,
                columns=names,
                source_store=source,
                source_store_provenance=source_provenance,
            ).model_dump_json(),
        )
        print(f"{out_path}: {len(names)} extra column(s) {names}")

    path = stem("_pfbagpr")
    provenance = DerivedStoreProvenance(
        method=(
            "the pFBA deletion vector of the source store with six columns of "
            "gene-protein-reaction redundancy appended, from yeast-GEM's rules in "
            "disjunctive normal form; written by isozyme_affine_store.py"
        ),
        gem_version=YeastGEM().version,
        source_store=source,
        source_store_provenance=source_provenance,
        columns=COLUMNS,
    )
    np.savez_compressed(
        path,
        genes=np.array(genes),
        v_deletion=np.concatenate(
            [pfba["v_deletion"].astype(np.float32), block], axis=1
        ),
        v_wild_type=np.concatenate(
            [
                pfba["v_wild_type"].astype(np.float32),
                np.zeros(len(COLUMNS), dtype=np.float32),
            ]
        ),
        rxn_ids=np.array([str(r) for r in pfba["rxn_ids"]] + COLUMNS),
        provenance=provenance.model_dump_json(),
    )

    # What the columns say about the genes the enrichment named: the released held-out
    # essential genes, beside their YPD growth ratio.
    with gzip.open(
        osp.join(RESULTS_DIR, "splits", "ess_fcl_val0.2_s0.json.gz"), "rt"
    ) as fh:
        split = json.load(fh)
    with open(osp.join(RESULTS_DIR, "fba_essentiality_per_gene.csv")) as fh:
        ratio = {r["gene"]: float(r["YPD"]) for r in csv.DictReader(fh)}
    row_of = {g: i for i, g in enumerate(genes)}
    held_out_essential = [
        {
            "gene": g,
            "ratio_ypd": ratio[g],
            **dict(zip(COLUMNS, block[row_of[g]].tolist(), strict=True)),
        }
        for g in sorted(split["genes"]["test"])
        if int(split["labels"][g]) == 1
    ]
    viable = [r for r in held_out_essential if r["ratio_ypd"] >= 0.999]
    report = {
        "store": path,
        "shape": [len(genes), int(pfba["v_deletion"].shape[1]) + len(COLUMNS)],
        "columns": COLUMNS,
        "n_genes": len(genes),
        "n_genes_with_a_rule": int((block[:, 0] > 0).sum()),
        "n_genes_with_a_sole_reaction": int((block[:, 1] > 0).sum()),
        "n_genes_fully_covered": int(((block[:, 0] > 0) & (block[:, 1] == 0)).sum()),
        "largest_rule_terms": largest,
        "held_out_essential": held_out_essential,
        "held_out_essential_fba_viable": {
            "n": len(viable),
            "n_fully_covered_by_alternatives": sum(
                r["gpr_n_sole"] == 0 for r in viable
            ),
            "n_with_a_sole_reaction": sum(r["gpr_n_sole"] > 0 for r in viable),
        },
    }
    # A tagged store writes a tagged summary; the untagged name belongs to the plain store
    # and was overwritten by every tagged build before this.
    out = osp.join(
        RESULTS_DIR, f"isozyme_affine_store{'_' + STORE_TAG if STORE_TAG else ''}.json"
    )
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "held_out_essential"}, indent=2
        )
    )
    print(out)


if __name__ == "__main__":
    main()

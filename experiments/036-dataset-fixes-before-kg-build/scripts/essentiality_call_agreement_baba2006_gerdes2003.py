# experiments/036-dataset-fixes-before-kg-build/scripts/essentiality_call_agreement_baba2006_gerdes2003.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.essentiality_call_agreement_baba2006_gerdes2003]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/essentiality_call_agreement_baba2006_gerdes2003

r"""G4 by content: Baba 2006 and Gerdes 2003 calls against the served essentiality stores.

The candidate gate's own value comparison (``candidate-gate overlap``) refuses a store
built from a dirty tree (``DirtyStoreError``), and every dev store built from a worktree
with an untracked guard file is dirty by its manifest's ``git status --porcelain``. So the
comparison is this committed script, which reads the four dev-tree LMDBs read-only
(``runners.load_records``, never the dataset class) and states each store's build commit
and dirty flag beside the counts.

Stores read (``$DATA_ROOT/data/torchcell/<slug>``):

- ``gene_essentiality_baba2006``: Keio construction outcome per BW25113 locus.
- ``gene_essentiality_gerdes2003``: genetic-footprinting call per MG1655 locus, carried to
  BW25113 through the one-to-one ECK pairs of ``bacteria_common.eck_crosswalk`` (a locus
  with no pair is counted and left out of the BW25113 comparisons).
- ``gene_essentiality_goodall2018``: TraDIS calls per BW25113 locus in two conditions,
  ``TL`` (the selection plates, solid medium) and ``LB`` (after outgrowth, liquid).
- ``gene_essentiality_price2018_ecoli``: Price 2018 Table S1, essential-or-important
  calls only (every record ``is_essential=True``), so it is compared as a gene list.

For two boolean call sets the counts are: shared loci, both essential, both not, each
side's essential-only, the agreeing count, and Cohen's kappa. Agreement is stated as
counts; nothing here judges which method is right. The duplication question is answered
separately: a store would SUBSUME a candidate only if it held the same call on every one
of the candidate's loci, and the counts show whether any does.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/essentiality_call_agreement_baba2006_gerdes2003.py
"""

import json
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from torchcell.datasets.bacteria_common import (  # noqa: E402
    bacterial_genome,
    eck_crosswalk,
)
from torchcell.provenance.build_manifest import BuildManifest  # noqa: E402
from torchcell.sequence.genome.ecoli.k12 import (  # noqa: E402
    EcoliK12BW25113Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.runners import load_records  # noqa: E402

STORES: dict[str, str] = {
    "baba2006": "gene_essentiality_baba2006",
    "gerdes2003": "gene_essentiality_gerdes2003",
    "goodall2018": "gene_essentiality_goodall2018",
    "price2018": "gene_essentiality_price2018_ecoli",
}
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "essentiality_call_agreement_baba2006_gerdes2003.json",
)

Calls = dict[str, bool]


def store_root(slug: str) -> str:
    """``$DATA_ROOT/data/torchcell/<slug>``."""
    return osp.join(os.environ["DATA_ROOT"], "data", "torchcell", slug)


def manifest_of(slug: str) -> dict[str, Any]:
    """The build commit, dirty flag and build time of one store."""
    path = osp.join(store_root(slug), "preprocess", "build_manifest.json")
    with open(path) as handle:
        manifest = BuildManifest.model_validate_json(handle.read())
    return {
        "torchcell_commit": manifest.torchcell_commit,
        "torchcell_dirty": manifest.torchcell_dirty,
        "built_at": manifest.built_at,
    }


def calls_by_locus(
    records: list[dict[str, Any]], *, solid: bool | None = None
) -> Calls:
    """``{locus tag: is_essential}``; ``solid`` keeps one Goodall condition."""
    out: Calls = {}
    for record in records:
        experiment = record["experiment"]
        if (
            solid is not None
            and (experiment["environment"]["media"]["state"] == "solid") != solid
        ):
            continue
        (perturbation,) = experiment["genotype"]["perturbations"]
        tag = perturbation["systematic_gene_name"]
        if tag in out:
            raise ValueError(f"{tag} twice in one call set")
        out[tag] = bool(experiment["phenotype"]["is_essential"])
    return out


def kappa(a: Calls, b: Calls, shared: list[str]) -> float | None:
    """Cohen's kappa of two boolean call sets on their shared loci."""
    n = len(shared)
    if n == 0:
        return None
    observed = sum(1 for k in shared if a[k] == b[k]) / n
    pa = sum(1 for k in shared if a[k]) / n
    pb = sum(1 for k in shared if b[k]) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def compare(a: Calls, b: Calls) -> dict[str, Any]:
    """Counts of two boolean call sets on their shared loci."""
    shared = sorted(set(a) & set(b))
    both_e = sum(1 for k in shared if a[k] and b[k])
    both_n = sum(1 for k in shared if not a[k] and not b[k])
    return {
        "a_loci": len(a),
        "b_loci": len(b),
        "shared_loci": len(shared),
        "both_essential": both_e,
        "both_not_essential": both_n,
        "a_essential_only": sum(1 for k in shared if a[k] and not b[k]),
        "b_essential_only": sum(1 for k in shared if b[k] and not a[k]),
        "agreeing": both_e + both_n,
        "cohen_kappa": kappa(a, b, shared),
    }


def against_list(a: Calls, essential: Calls) -> dict[str, Any]:
    """A boolean call set against an essential-only list (Price 2018 Table S1)."""
    listed = sorted(set(a) & set(essential))
    return {
        "a_loci": len(a),
        "list_loci": len(essential),
        "shared_loci": len(listed),
        "a_calls_essential": sum(1 for k in listed if a[k]),
        "a_calls_not_essential": sum(1 for k in listed if not a[k]),
        "a_essential_not_in_list": sum(1 for k in a if a[k] and k not in essential),
    }


def gerdes_on_bw25113(gerdes: Calls) -> tuple[Calls, int]:
    """Gerdes's MG1655 calls carried to BW25113 by the one-to-one ECK pairs."""
    mg = bacterial_genome("ecoli", "MG1655")
    bw = bacterial_genome("ecoli", "BW25113")
    if not isinstance(mg, EcoliK12MG1655Genome) or not isinstance(
        bw, EcoliK12BW25113Genome
    ):
        raise RuntimeError("the K-12 genomes did not resolve to their classes")
    pair_of = {pair.mg1655: pair.bw25113 for pair in eck_crosswalk(mg, bw).pairs}
    out: Calls = {}
    unpaired = 0
    for tag, call in gerdes.items():
        target = pair_of.get(tag)
        if target is None:
            unpaired += 1
            continue
        out[target] = call
    return out, unpaired


def main() -> None:
    """Read the four stores, compare every pair, write the results JSON."""
    records = {name: load_records(store_root(slug)) for name, slug in STORES.items()}
    baba = calls_by_locus(records["baba2006"])
    gerdes_mg = calls_by_locus(records["gerdes2003"])
    goodall_tl = calls_by_locus(records["goodall2018"], solid=True)
    goodall_lb = calls_by_locus(records["goodall2018"], solid=False)
    price = calls_by_locus(records["price2018"])
    if not all(price.values()):
        raise ValueError("Price 2018 Table S1 is expected to be essential-only")
    gerdes, unpaired = gerdes_on_bw25113(gerdes_mg)
    result: dict[str, Any] = {
        "stores": {name: manifest_of(slug) for name, slug in STORES.items()},
        "records": {name: len(r) for name, r in records.items()},
        "call_sets": {
            "baba2006": {"loci": len(baba), "essential": sum(baba.values())},
            "gerdes2003_mg1655": {
                "loci": len(gerdes_mg),
                "essential": sum(gerdes_mg.values()),
            },
            "gerdes2003_on_bw25113": {
                "loci": len(gerdes),
                "essential": sum(gerdes.values()),
                "mg1655_loci_without_a_one_to_one_eck_pair": unpaired,
            },
            "goodall2018_tl": {
                "loci": len(goodall_tl),
                "essential": sum(goodall_tl.values()),
            },
            "goodall2018_lb": {
                "loci": len(goodall_lb),
                "essential": sum(goodall_lb.values()),
            },
            "price2018": {"loci": len(price), "essential": sum(price.values())},
        },
        "baba2006_vs_goodall2018_tl": compare(baba, goodall_tl),
        "baba2006_vs_goodall2018_lb": compare(baba, goodall_lb),
        "baba2006_vs_price2018_list": against_list(baba, price),
        "gerdes2003_vs_goodall2018_tl": compare(gerdes, goodall_tl),
        "gerdes2003_vs_goodall2018_lb": compare(gerdes, goodall_lb),
        "gerdes2003_vs_price2018_list": against_list(gerdes, price),
        "baba2006_vs_gerdes2003": compare(baba, gerdes),
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

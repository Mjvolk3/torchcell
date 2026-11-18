# experiments/048-pputida-isoprenol-cgt/scripts/provision_pputida_gems.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.provision_pputida_gems]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/provision_pputida_gems
"""Pin the two P. putida metabolic models into the GEM tier.

Two models, with different provenance standing, and the difference is the reason
this script exists rather than a path in a config:

* ``iJN1463`` is published and served by BiGG at a stable URL. BiGG calls it
  iJN1463 while Nogales et al. 2020 call it iJN1462 (1,462 genes), and BiGG's own
  record carries ``published_filename: iJN1462.json``. Asking BiGG for iJN1462
  returns 404, so the paper-to-artifact link is recorded in the manifest or it is
  lost.
* ``pputida_isoprenol`` is iJN1463 plus the IPP-bypass mevalonate route to
  isoprenol, and it is **not published anywhere**. It came from Juan Camilo at
  LBNL/JBEI in a shared folder, so our deposit is the only copy and its sha256 is
  the only identity it has. ``retrieval_method`` is ``manual_deposit`` and the
  command is a recipe, not a script, because there is no URL to re-run.

The diff between them is reported rather than assumed, so a later session can see
what the augmentation added without re-deriving it.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil

import cobra
from dotenv import load_dotenv

from torchcell.metabolism.bigg_gem import BiggGEM, bigg_model_url

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")
GEM_ROOT = osp.join(DATA_ROOT, "torchcell-gems")

SHARED = os.environ.get("CARRUTHERS_SHARED_DIR", "/tmp/screenshots/04_Data")

BASE_MODEL_ID = "iJN1463"
ISOPRENOL_MODEL_ID = "pputida_isoprenol"

ISOPRENOL_NOTES = (
    "iJN1463 plus the isoprenol route: the IPP-bypass mevalonate pathway "
    "(ACCOAAT/HMGCOAS/HMGCOAR by mvaE and mvaS, MEVKIN by MK, PMD by PMDHKQ) "
    "ending in APP (3mbut3e1yp_c -> ip_c + pi_c, GPR aphA, the pIY670 acid "
    "phosphatase), with transport IPtpp (PP_1384 and PP_1385 and PP_1386) and "
    "IPtex, and the exchange EX_ip_e. NOT PUBLISHED: received from Juan Camilo "
    "(LBNL/JBEI) in the Carruthers 2025 working-data folder, so this deposit is "
    "the only copy and its sha256 is its identity. Flag to the depositor if the "
    "upstream model is later released, so the published bytes can be pinned "
    "beside this one rather than replacing it."
)


def fetch_bigg_model(model_id: str, dest: str) -> str:
    """Download a BiGG model JSON to ``dest`` and return the URL used."""
    from torchcell.literature.retrieve import direct_url

    url = bigg_model_url(model_id)
    payload = direct_url(url)
    os.makedirs(osp.dirname(dest), exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(payload)
    return url


def model_diff(base: cobra.Model, augmented: cobra.Model) -> dict[str, object]:
    """What the augmented model adds to and removes from the base model."""
    base_rxn = {r.id for r in base.reactions}
    aug_rxn = {r.id for r in augmented.reactions}
    base_met = {m.id for m in base.metabolites}
    aug_met = {m.id for m in augmented.metabolites}
    base_gene = {g.id for g in base.genes}
    aug_gene = {g.id for g in augmented.genes}
    added = sorted(aug_rxn - base_rxn)
    return {
        "reactions": {"base": len(base_rxn), "augmented": len(aug_rxn)},
        "metabolites": {"base": len(base_met), "augmented": len(aug_met)},
        "genes": {"base": len(base_gene), "augmented": len(aug_gene)},
        "reactions_added": added,
        "reactions_removed": sorted(base_rxn - aug_rxn),
        "metabolites_added": sorted(aug_met - base_met),
        "genes_added": sorted(aug_gene - base_gene),
        "added_reaction_detail": [
            {
                "id": rxn.id,
                "name": rxn.name,
                "reaction": rxn.reaction,
                "gpr": rxn.gene_reaction_rule,
                "bounds": list(rxn.bounds),
                "subsystem": rxn.subsystem,
            }
            for rxn in (augmented.reactions.get_by_id(i) for i in added)
        ],
    }


def main() -> None:
    """Deposit both models, verify them, and write the diff report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem-root", default=GEM_ROOT)
    parser.add_argument("--shared", default=SHARED)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    staging = osp.join(args.gem_root, "_staging")
    os.makedirs(staging, exist_ok=True)

    base_download = osp.join(staging, f"{BASE_MODEL_ID}.json")
    url = fetch_bigg_model(BASE_MODEL_ID, base_download)
    base_gem = BiggGEM(BASE_MODEL_ID, gem_root=args.gem_root)
    base_manifest = base_gem.deposit(
        source_path=base_download,
        organism="Pseudomonas putida KT2440",
        source="BiGG Models",
        source_url=url,
        retrieval_method="direct_url",
        retrieval_command=(f'torchcell.literature.retrieve.direct_url("{url}")'),
        notes=(
            "BiGG serves this model as iJN1463; Nogales et al. 2020 publish it "
            "as iJN1462 (1,462 genes) and BiGG's API records "
            "published_filename iJN1462.json. A request for iJN1462 at BiGG "
            "returns 404, so iJN1463 is the retrievable id. Genome reference "
            "NC_002947.4, which is the RefSeq accession of the pinned "
            "pputida_KT2440_ASM756v2 replicon AE015451.2. GPR gene ids are "
            "bare PP_#### locus tags."
        ),
        extra={
            "published_filename": "iJN1462.json",
            "published_model_name": "iJN1462",
            "genome_ref": "ncbi_accession:NC_002947.4",
            "last_updated": "Oct 31, 2019",
            "gene_namespace": "pputida_kt2440_locus_tag",
        },
    )

    shared_isoprenol = osp.join(args.shared, "pputida_isoprenol.json")
    iso_gem = BiggGEM(ISOPRENOL_MODEL_ID, gem_root=args.gem_root)
    iso_manifest = iso_gem.deposit(
        source_path=shared_isoprenol,
        organism="Pseudomonas putida KT2440",
        source="Carruthers 2025 working data, shared by the authors",
        source_url=shared_isoprenol,
        retrieval_method="manual_deposit",
        retrieval_command=(
            "Received as pputida_isoprenol.json in the Carruthers 2025 "
            "working-data folder shared by Juan Camilo (LBNL/JBEI); copied "
            "verbatim into the GEM tier. There is no URL to re-run: if this "
            "deposit is lost, the file must be requested from the depositor."
        ),
        notes=ISOPRENOL_NOTES,
        extra={
            "genome_ref": "ncbi_accession:NC_002947.4",
            "gene_namespace": "pputida_kt2440_locus_tag",
        },
    )

    diff = model_diff(base_gem.model, iso_gem.model)
    report = {
        "gem_root": args.gem_root,
        "base": base_gem.summary(),
        "isoprenol": iso_gem.summary(),
        "base_manifest_notes": base_manifest.notes,
        "isoprenol_manifest_notes": iso_manifest.notes,
        "diff": diff,
    }
    out = osp.join(RESULTS, "pputida_gem_provenance.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    shutil.rmtree(staging, ignore_errors=True)

    print(f"base      {base_gem.summary()}")
    print(f"isoprenol {iso_gem.summary()}")
    print(
        f"diff: +{len(diff['reactions_added'])} reactions, "
        f"-{len(diff['reactions_removed'])} reactions, "
        f"+{len(diff['metabolites_added'])} metabolites, "
        f"+{len(diff['genes_added'])} genes"
    )
    print(f"genes added: {', '.join(diff['genes_added'])}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()

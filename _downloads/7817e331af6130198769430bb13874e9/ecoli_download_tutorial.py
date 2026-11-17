# experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
# [[experiments.034-showcase-datasets.scripts.ecoli_download_tutorial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/ecoli_download_tutorial
"""Download four built E. coli datasets from ``tc-data``, load them and print their data.

This is the code of the documentation page ``datasets/ecoli/download-tutorial``; the page
shows each block between a ``# [name]`` and a ``# [/name]`` marker and, under it, what
that block printed. It needs an installed ``torchcell`` and a network route to the public
endpoint, nothing else: no repository checkout, no ``DATA_ROOT``, no key.

Run it in an empty directory::

    python ecoli_download_tutorial.py

It downloads into ``./data/torchcell/<slug>/``. Standard output carries the data only and
is the same on every machine; the seconds each loader took to download, verify, unpack
and open its store go to ``./data/torchcell/ecoli_download_tutorial_timings.json`` and to
standard error, because they are not.

The four datasets:

- ``GrowthRateCampos2018Dataset``: maximal growth rate of 3,664 Keio single-gene deletions.
- ``MorphologyCampos2018Dataset``: cell-shape and cell-cycle features of the same strains.
- ``GeneInteractionBabu2014Dataset``: eSGA S-scores of 42,705 double mutants.
- ``MetabolomeFuhrer2017Dataset``: 7,534 ion z-scores for each of 3,806 Keio strains.
"""

from __future__ import annotations

# [setup]
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from torchcell.datasets.artifact import content_sha256
from torchcell.datasets.client import DatasetClient
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.campos2018 import (
    GrowthRateCampos2018Dataset,
    MorphologyCampos2018Dataset,
)
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.endpoint_http import DEFAULT_URL

# A loader downloads its built store only when TC_DATA_URL is set. Without it the loader
# builds the store from the publisher's files, which is the developer path.
os.environ.setdefault("TC_DATA_URL", DEFAULT_URL)

DATA = Path("data/torchcell")  # each dataset unpacks into DATA/<slug>/
SECONDS: dict[str, float] = {}  # slug -> seconds to download, verify, unpack and open
# [/setup]

SECTION = "=" * 12


def section(title: str) -> None:
    """Print the banner the page generator splits standard output on."""
    print(f"{SECTION} {title} {SECTION}")


section("index")
# [index]
client = DatasetClient.from_env()
print("endpoint", client.url)
SLUGS = (
    "ecoli_growth_rate_campos2018",
    "ecoli_morphology_campos2018",
    "gene_interaction_babu2014",
    "metabolome_fuhrer2017",
)
for slug in SLUGS:
    artifact = client.select(slug)  # newest archive the installed major.minor reads
    if artifact is None:
        raise SystemExit(f"{client.url} serves no {slug} archive this torchcell reads")
    print(slug)
    print(f"  loader   {artifact.dataset_class}")
    print(f"  archive  {artifact.archive}  ({artifact.archive_bytes:,} bytes)")
    print(f"  records  {artifact.n_experiments:,}")
    print(f"  release  knowledge graph {artifact.kg_version} ({artifact.kg_release})")
# [/index]

section("growth rate")
# [growth]
started = time.perf_counter()
growth = GrowthRateCampos2018Dataset(root=str(DATA / "ecoli_growth_rate_campos2018"))
SECONDS["ecoli_growth_rate_campos2018"] = time.perf_counter() - started
print(f"{len(growth):,} records")

record = growth[0]  # a dict with three keys: experiment, reference, publication
deletion = record["experiment"]["genotype"]["perturbations"][0]
environment = record["experiment"]["environment"]
phenotype = record["experiment"]["phenotype"]
print("gene       ", deletion["perturbed_gene_name"], deletion["systematic_gene_name"])
print("strain     ", deletion["collection"], "plate", deletion["construction"]["plate"])
print("medium     ", environment["media"]["name"])
print(
    "temperature",
    environment["temperature"]["value"],
    environment["temperature"]["unit"],
)
print(
    "fitness    ", phenotype["fitness"], "from", phenotype["n_samples"], "growth curve"
)
print("wild type  ", record["reference"]["phenotype_reference"]["fitness"])
print("genome     ", record["reference"]["genome_reference"]["assembly_accession"])
print("source     ", record["publication"]["doi_url"])

growth_table = pd.DataFrame(
    {
        "gene": r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"],
        "locus_tag": r["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ],
        "fitness": r["experiment"]["phenotype"]["fitness"],
    }
    for r in (growth[i] for i in range(len(growth)))
)
print(growth_table.head(6).to_string(index=False, float_format="{:.4f}".format))
print(
    "fitness min {:.4f}  median {:.4f}  max {:.4f}".format(
        growth_table["fitness"].min(),
        growth_table["fitness"].median(),
        growth_table["fitness"].max(),
    )
)
# [/growth]

section("typed record")
# [typed]
typed = growth.transform_item(record)  # the same record as validated pydantic objects
experiment = typed["experiment"]
print(type(experiment).__name__, type(experiment.phenotype).__name__)
print(
    experiment.genotype.perturbations[0].perturbed_gene_name,
    experiment.phenotype.fitness,
)
# [/typed]

section("morphology")
# [morphology]
started = time.perf_counter()
morphology = MorphologyCampos2018Dataset(root=str(DATA / "ecoli_morphology_campos2018"))
SECONDS["ecoli_morphology_campos2018"] = time.perf_counter() - started
print(f"{len(morphology):,} records")

phenotype = morphology[0]["experiment"]["phenotype"]
print("cells measured", phenotype["n_samples"])
print("features      ", list(phenotype["morphology"]))

morphology_table = pd.DataFrame(
    {
        "gene": r["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"],
        **r["experiment"]["phenotype"]["morphology"],
    }
    for r in (morphology[i] for i in range(len(morphology)))
).set_index("gene")
print("table shape   ", morphology_table.shape)
print(
    "features per strain:",
    morphology_table.notna().sum(axis=1).value_counts().to_dict(),
)
shown = ["<L>", "<W>", "<A>", "<SA/V>", "rho_CD", "%2N"]
print(morphology_table[shown].head(6).to_string(float_format="{:.4f}".format))
# [/morphology]

section("genetic interaction")
# [interaction]
started = time.perf_counter()
interaction = GeneInteractionBabu2014Dataset(
    root=str(DATA / "gene_interaction_babu2014")
)
SECONDS["gene_interaction_babu2014"] = time.perf_counter() - started
print(f"{len(interaction):,} records")

record = interaction[0]
phenotype = record["experiment"]["phenotype"]
for perturbation in record["experiment"]["genotype"]["perturbations"]:
    print(
        "gene   ",
        perturbation["perturbed_gene_name"],
        perturbation["systematic_gene_name"],
        "|",
        perturbation["collection"],
    )
print(
    "S-score", phenotype["gene_interaction"], "from", phenotype["n_samples"], "colonies"
)
print("medium ", record["experiment"]["environment"]["media"]["name"])

rows = []
for r in (interaction[i] for i in range(len(interaction))):
    first, second = r["experiment"]["genotype"]["perturbations"]
    rows.append(
        {
            "gene_a": first["perturbed_gene_name"],
            "gene_b": second["perturbed_gene_name"],
            "id_a": first["systematic_gene_name"],
            "id_b": second["systematic_gene_name"],
            "s_score": r["experiment"]["phenotype"]["gene_interaction"],
        }
    )
pairs = pd.DataFrame(rows)
print(pairs.head(6).to_string(index=False, float_format="{:.4f}".format))
print("genes in any pair:", len(interaction.gene_set))
print("negative scores:", int((pairs["s_score"] < 0).sum()))
print("positive scores:", int((pairs["s_score"] > 0).sum()))
print("strongest negative pairs:")
print(
    pairs.nsmallest(3, "s_score").to_string(index=False, float_format="{:.4f}".format)
)
# [/interaction]

section("metabolome")
# [metabolome]
started = time.perf_counter()
metabolome = MetabolomeFuhrer2017Dataset(root=str(DATA / "metabolome_fuhrer2017"))
SECONDS["metabolome_fuhrer2017"] = time.perf_counter() - started
print(f"{len(metabolome):,} records")

record = metabolome[0]
levels = record["experiment"]["phenotype"]["metabolite_level"]  # ion key -> z-score
print(
    "strain      ", record["experiment"]["genotype"]["perturbations"][0]["strain_label"]
)
print("ions        ", len(levels))
print("first ions  ", dict(list(levels.items())[:4]))

ions = pd.read_csv(Path(metabolome.preprocess_dir) / "ions.csv")  # ion key -> m/z
print(ions.head(4).to_string(index=False, float_format="{:.5f}".format))

ion_keys = list(levels)
strains, matrix_rows = [], []
for r in (metabolome[i] for i in range(len(metabolome))):
    perturbations = r["experiment"]["genotype"]["perturbations"]
    strains.append("+".join(p["perturbed_gene_name"] for p in perturbations))
    row_levels = r["experiment"]["phenotype"]["metabolite_level"]
    matrix_rows.append([row_levels[key] for key in ion_keys])
matrix = np.asarray(matrix_rows, dtype=np.float32)  # strains x ions
print("matrix shape", matrix.shape, matrix.dtype)
corner = pd.DataFrame(matrix[:5, :5], index=strains[:5], columns=ion_keys[:5])
print(corner.to_string(float_format="{:.4f}".format))
print("z-scores beyond 5 in absolute value:", int((np.abs(matrix) > 5).sum()))
print("records that name more than one gene:", sum("+" in name for name in strains))
# [/metabolome]

section("verify")


# [verify]
def experiment_id(record: dict[str, Any]) -> str:
    """The id the knowledge graph gives this record: sha256 of its experiment."""
    return hashlib.sha256(json.dumps(record["experiment"]).encode("utf-8")).hexdigest()


loaded = {
    "ecoli_growth_rate_campos2018": growth,
    "ecoli_morphology_campos2018": morphology,
    "gene_interaction_babu2014": interaction,
    "metabolome_fuhrer2017": metabolome,
}
for slug, dataset in loaded.items():
    ids = (experiment_id(dataset[i]) for i in range(len(dataset)))
    digest = content_sha256(ids)  # one hash over every record of the dataset
    served = client.select(slug).content_sha256  # what the release recorded
    print(
        f"{slug}  {digest}  {'matches' if digest == served else 'DIFFERS FROM'} the release"
    )
# [/verify]

timings_path = DATA / "ecoli_download_tutorial_timings.json"
timings_path.write_text(json.dumps(SECONDS, indent=2) + "\n", encoding="utf-8")
for slug, seconds in SECONDS.items():
    print(
        f"[seconds to download, verify, unpack and open] {slug}: {seconds:.2f}",
        file=sys.stderr,
    )

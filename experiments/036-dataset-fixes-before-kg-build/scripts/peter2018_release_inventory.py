# experiments/036-dataset-fixes-before-kg-build/scripts/peter2018_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.peter2018_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/peter2018_release_inventory.py
"""Measure the Peter 2018 phenome release, its genotypes and its loss-of-function matrices.

Every number in the 2026.10.11 section of ``[[torchcell.datasets.scerevisiae.peter2018]]``
that is not a verifier row comes from this script. It reads only sha256-pinned bytes:
the release and ``md5.txt`` from the raw mirror, Supplementary Table 1 from the library
mirror, Peter's matrices from the genomes tier, and the built Caudal 2024 dev store's
``data.csv`` and ``sequence_variants.parquet`` (read only).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/peter2018_release_inventory.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/peter2018_release_inventory.py
"""

from __future__ import annotations

import gzip
import io
import os
import os.path as osp
import tarfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.scerevisiae import peter2018 as p
from torchcell.datasets.scerevisiae.caudal2024 import (
    COPYNUMBER_NAME,
    PRESENCE_NAME,
    REFGENE_TAR_NAME,
    SGD_FSA_NAME,
)
from torchcell.literature.manifest import sha256_file
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    load_genome_manifest,
    resolve,
)

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "peter2018_release_inventory.json",
)
LOF_META = ["CODE", "NAME", "Gene", "SGDorder", "CH", "START", "STOP"]
FRAMESHIFT_META = ["CODE", "Gene", "SGDorder", "CH", "START", "STOP"]
HO_COLUMN = "X2672.YDL227C"


class Release(BaseModel):
    """The growth-ratio release as read by the loader."""

    model_config = ConfigDict(extra="forbid")

    sha256: str
    md5: str
    md5_published: str
    n_isolates: int
    n_conditions: int
    n_values: int
    n_zero: int
    minimum: float
    maximum: float


class Panel(BaseModel):
    """The phenotyped isolates against the genome panel, Caudal and Table S1."""

    model_config = ConfigDict(extra="forbid")

    n_genome_panel: int
    n_phenotyped_in_genome_panel: int
    n_caudal_store: int
    n_phenotyped_in_caudal_store: int
    n_phenotyped_not_in_caudal_store: int
    n_caudal_not_phenotyped: int
    caudal_sequence_variant_rows: int
    caudal_sequence_variants_identical: bool
    ho_deletion: dict[str, int]
    ho_deleted_presence_matrix: dict[str, int]
    ploidy: dict[str, int]
    zygosity: dict[str, int]
    aneuploid: int


class Genotypes(BaseModel):
    """Per-isolate perturbation counts over the 971 phenotyped isolates."""

    model_config = ConfigDict(extra="forbid")

    sequence_variants_total: int
    sequence_variants_mean: float
    sequence_variants_min: int
    sequence_variants_max: int
    accessory_present_total: int
    reference_absent_total: int


class LossOfFunction(BaseModel):
    """The two loss-of-function matrices over the 971 phenotyped isolates."""

    model_config = ConfigDict(extra="forbid")

    lof_rows: int
    lof_calls: int
    lof_genes_called: int
    lof_on_sequence_variant: int
    lof_on_gene_absent_from_tarball: int
    lof_genes_absent_from_tarball: int
    lof_on_reference_allele: int
    frameshift_rows: int
    frameshift_calls: int
    frameshift_rows_without_code: int
    frameshift_calls_without_code: int


class Inventory(BaseModel):
    """Everything the dendron note cites from this script."""

    model_config = ConfigDict(extra="forbid")

    release: Release
    panel: Panel
    genotypes: Genotypes
    loss_of_function: LossOfFunction


def _release(data_root: str) -> tuple[Release, pd.DataFrame]:
    mirror = p.raw_mirror_dir(data_root)
    path = mirror / p.PHENO_REL
    frame = p.read_release(path, p.build_conditions())
    values = frame.to_numpy(dtype=float)
    release = Release(
        sha256=sha256_file(path),
        md5=p.md5_file(path),
        md5_published=p.read_published_md5(mirror / p.MD5_REL)[p.PHENO_NAME],
        n_isolates=frame.shape[0],
        n_conditions=frame.shape[1],
        n_values=int(frame.size),
        n_zero=int((values == 0).sum()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )
    return release, frame


def _caudal(data_root: str) -> tuple[set[str], pd.DataFrame]:
    root = osp.join(data_root, "data/torchcell/caudal_pantranscriptome2024/preprocess")
    strains = set(pd.read_csv(osp.join(root, "data.csv"))["strain_id"].astype(str))
    return strains, pd.read_parquet(osp.join(root, "sequence_variants.parquet"))


def main() -> None:
    """Measure and write the inventory."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    release, frame = _release(data_root)
    isolates = set(frame.index)
    pangenome = p.read_pangenome(
        resolve(PETER2018_1011, PRESENCE_NAME), resolve(PETER2018_1011, COPYNUMBER_NAME)
    )
    tar_path = resolve(PETER2018_1011, REFGENE_TAR_NAME)
    variants = p.diff_reference_genes(
        tar_path, resolve(SGD_S288C_R64, SGD_FSA_NAME), set(pangenome.row_of)
    )
    caudal_strains, caudal_variants = _caudal(data_root)
    key = ["strain_id", "systematic_gene_name", "symbol", "header_token"]
    mine = variants[variants["strain_id"].isin(caudal_strains)]
    identical = set(map(tuple, mine[key].to_numpy())) == set(
        map(tuple, caudal_variants[key].to_numpy())
    )

    library = osp.join(data_root, "torchcell-library", p.CITATION_KEY)
    table_s1 = pd.read_excel(
        osp.join(library, p.SI_TABLES_XLS), sheet_name="Table S1", header=3
    )
    table_s1["Standardized name"] = table_s1["Standardized name"].astype(str)
    s1 = table_s1.set_index("Standardized name").loc[sorted(isolates)]
    ho_yes = s1.index[s1["HO deletion"] == "yes"]
    presence = pd.read_csv(
        resolve(PETER2018_1011, PRESENCE_NAME), sep="\t", index_col=0
    )
    ho_cells = presence.loc[ho_yes, HO_COLUMN].value_counts()
    panel = Panel(
        n_genome_panel=len(pangenome.row_of),
        n_phenotyped_in_genome_panel=len(isolates & set(pangenome.row_of)),
        n_caudal_store=len(caudal_strains),
        n_phenotyped_in_caudal_store=len(isolates & caudal_strains),
        n_phenotyped_not_in_caudal_store=len(isolates - caudal_strains),
        n_caudal_not_phenotyped=len(caudal_strains - isolates),
        caudal_sequence_variant_rows=len(caudal_variants),
        caudal_sequence_variants_identical=identical,
        ho_deletion={
            str(k): int(v) for k, v in s1["HO deletion"].value_counts().items()
        },
        ho_deleted_presence_matrix={
            "absent": int(ho_cells.get(0, 0)),
            "present": int(ho_cells.get(1, 0)),
        },
        ploidy={str(k): int(v) for k, v in s1["Ploidy"].value_counts().items()},
        zygosity={str(k): int(v) for k, v in s1["Zygosity"].value_counts().items()},
        aneuploid=int(s1["Aneuploidies"].astype(str).str.startswith("aneu").sum()),
    )

    phenotyped = variants[variants["strain_id"].isin(isolates)]
    per_isolate = phenotyped.groupby("strain_id").size()
    present = absent = 0
    for strain in isolates:
        row = pangenome.row_of[strain]
        present += int(((pangenome.presence[row] == 1) & ~pangenome.s288c_mask).sum())
        absent += int(((pangenome.presence[row] == 0) & pangenome.s288c_mask).sum())
    genotypes = Genotypes(
        sequence_variants_total=len(phenotyped),
        sequence_variants_mean=round(float(per_isolate.mean()), 3),
        sequence_variants_min=int(per_isolate.min()),
        sequence_variants_max=int(per_isolate.max()),
        accessory_present_total=present,
        reference_absent_total=absent,
    )

    with tarfile.open(tar_path, "r:gz") as tf:
        tar_genes = {m.name.replace(".fasta", "") for m in tf if m.isfile()}
    raw = gzip.open(resolve(PETER2018_1011, p.LOF_NAME)).read().decode("latin-1")
    lof = pd.read_csv(io.StringIO(raw.replace("\r", "\n")), sep="\t", low_memory=False)
    lof_matrix = lof.set_index("NAME")[sorted(isolates)]
    calls = {
        (strain, gene)
        for gene, row in lof_matrix.iterrows()
        for strain in row.index[row.to_numpy() == 1]
    }
    variant_pairs = set(
        zip(phenotyped["strain_id"], phenotyped["systematic_gene_name"])
    )
    off_tar = {c for c in calls if c[1] not in tar_genes}
    frameshift = pd.read_csv(
        resolve(PETER2018_1011, p.FRAMESHIFT_NAME), sep="\t", low_memory=False
    )
    fs_values = frameshift[sorted(isolates)].to_numpy()
    no_code = frameshift["CODE"].isna().to_numpy()
    loss = LossOfFunction(
        lof_rows=len(lof),
        lof_calls=len(calls),
        lof_genes_called=len({g for _, g in calls}),
        lof_on_sequence_variant=len(calls & variant_pairs),
        lof_on_gene_absent_from_tarball=len(off_tar),
        lof_genes_absent_from_tarball=len({g for _, g in off_tar}),
        lof_on_reference_allele=len(calls - variant_pairs - off_tar),
        frameshift_rows=len(frameshift),
        frameshift_calls=int(np.sum(fs_values)),
        frameshift_rows_without_code=int(no_code.sum()),
        frameshift_calls_without_code=int(np.sum(fs_values[no_code])),
    )
    inventory = Inventory(
        release=release, panel=panel, genotypes=genotypes, loss_of_function=loss
    )
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        handle.write(inventory.model_dump_json(indent=2) + "\n")
    print(inventory.model_dump_json(indent=2))
    print("peter tier manifest files:", len(load_genome_manifest(PETER2018_1011).files))


if __name__ == "__main__":
    main()

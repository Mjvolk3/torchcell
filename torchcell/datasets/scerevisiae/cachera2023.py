# torchcell/datasets/scerevisiae/cachera2023
# [[torchcell.datasets.scerevisiae.cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/cachera2023
# Test file: tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 CRI-SPA betaxanthin dataset (genome-wide product-proxy screen).

Cachera et al. 2023 (Nucleic Acids Research, doi:10.1093/nar/gkad656) used CRI-SPA to
transfer four betaxanthin-biosynthesis genes into each strain of the ~4800-strain yeast
knockout (YKO) collection, then read out per-colony **betaxanthin** (a yellow, naturally
fluorescent plant metabolite) by image analysis. The "CRI-SPA score" is a
corrected/normalized colony fluorescence intensity -- a quantitative proxy for
betaxanthin level (it can be negative because it is population-centered).

Source: the paper's Data Availability points to the CRI-SPA GitHub repo
(github.com/pc2912/CRI-SPA_repo). We ingest `GA1_2_4_6.csv` -- the gene-level
corrected+filtered dataset combining screen replicates 1/2/4/6 -- using the 24 h
`corrected_mean_intensity` (mean/std/count) as the betaxanthin level + SE + n.

Maps to `MetabolitePhenotype` (WS4): `metabolite_level = {"betaxanthin": score}` with
`measurement_type = "cri_spa_corrected_fluorescence_intensity_24h"`. Gene names in the
source are COMMON names, so a genome is required to resolve them to systematic ORF ids
(same pattern as Sameith); unresolved names + control/NaN rows are excluded and logged.
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from typing import Any, cast

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "cri_spa_corrected_fluorescence_intensity_24h"
TARGET_METABOLITE = "betaxanthin"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


# The constant engineered background transferred by CRI-SPA into every YKO strain: the
# Btx-cassette, chromosomally integrated at site XII-5 (paper Methods; pBTX1/pBTX2). Two
# heterologous plant genes (CYP76AD1, DOD) + two feedback-resistant mutant alleles of
# NATIVE yeast genes (ARO4^K229L / YBR249C, ARO7^G141S / YPR060C) integrated ectopically
# (hence GeneAddition with a variant, NOT AllelePerturbation at the native locus). The
# natMX marker is not a betaxanthin gene and is omitted. source_organism for DOD needs
# confirmation against DeLoache 2015 (ref 25) at review. plasmid_contig_id stays None
# until the plasmid-sequence store lands (Cachera GenBank maps in the OUP SI zip). See
# [[torchcell.datamodels.gene-addition-perturbation-design]].
def _betaxanthin_cassette() -> list[GeneAdditionPerturbation]:
    """Fresh Btx-cassette perturbations (new objects per record)."""
    return [
        GeneAdditionPerturbation(
            systematic_gene_name="CYP76AD1",
            perturbed_gene_name="CYP76AD1",
            source_organism="Beta vulgaris",
            is_heterologous=True,
            localization="chromosomal_integration",
            integration_locus="XII-5",
            construct_name="Btx-cassette",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="DOD",
            perturbed_gene_name="DOD",
            source_organism="Mirabilis jalapa",
            is_heterologous=True,
            localization="chromosomal_integration",
            integration_locus="XII-5",
            construct_name="Btx-cassette",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YBR249C",
            perturbed_gene_name="ARO4",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="chromosomal_integration",
            integration_locus="XII-5",
            construct_name="Btx-cassette",
            variant="K229L",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YPR060C",
            perturbed_gene_name="ARO7",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="chromosomal_integration",
            integration_locus="XII-5",
            construct_name="Btx-cassette",
            variant="G141S",
        ),
    ]


# Gene-level corrected+filtered dataset (replicates 1/2/4/6) from the CRI-SPA repo.
DATA_URL = "https://raw.githubusercontent.com/pc2912/CRI-SPA_repo/main/GA1_2_4_6.csv"
DATA_FILENAME = "GA1_2_4_6.csv"
_LEVEL = "corrected_mean_intensity.24_mean"
_STD = "corrected_mean_intensity.24_std"
_COUNT = "corrected_mean_intensity.24_count"


@register_dataset
class BetaxanthinCachera2023Dataset(ExperimentDataset):
    """Genome-wide betaxanthin CRI-SPA product-proxy screen of the YKO collection."""

    def __init__(
        self,
        root: str = "data/torchcell/betaxanthin_cachera2023",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is required to resolve common gene names."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MetaboliteExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The gene-level CRI-SPA dataset required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the gene-level CRI-SPA dataset from the authors' GitHub repo."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading CRI-SPA data from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 10000:
            raise RuntimeError(f"CRI-SPA download too small: {len(data)} bytes")
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes)", dest, len(data))

    def _resolve_systematic(self, gene: str) -> str | None:
        """Resolve a source gene name (common or systematic) to a systematic ORF id."""
        gene = gene.strip().upper()
        if _SYSTEMATIC_RE.match(gene):
            return gene
        genome = cast(SCerevisiaeGenome, self.genome)
        candidates = genome.alias_to_systematic.get(gene, [])
        if candidates:
            return candidates[0]
        return None

    @post_process
    def process(self) -> None:
        """Parse the CRI-SPA dataset into per-ORF Metabolite experiments and write LMDB."""
        if self.genome is None:
            raise RuntimeError(
                "Cachera2023 requires an injected SCerevisiaeGenome to resolve common "
                "gene names to systematic ORF ids (source uses common names)."
            )
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        n_control_or_nan = 0
        unresolved: set[str] = set()
        seen: set[str] = set()
        collisions: set[str] = set()
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            gene = str(row["gene"]).strip()
            level = row[_LEVEL]
            if gene in ("0", "nan", "") or pd.isna(level):
                n_control_or_nan += 1
                continue
            systematic = self._resolve_systematic(gene)
            if systematic is None:
                unresolved.add(gene)
                continue
            # Distinct source names can alias to the same ORF; keep the first, log rest.
            if systematic in seen:
                collisions.add(systematic)
                continue
            seen.add(systematic)
            count = int(row[_COUNT]) if pd.notna(row[_COUNT]) else 1
            std = row[_STD]
            se = (
                float(std) / math.sqrt(count)
                if pd.notna(std) and count > 1
                else float("nan")
            )
            rows.append(
                {"orf": systematic, "level": float(level), "se": se, "n": max(count, 1)}
            )
        log.info(
            "Cachera: %d usable ORFs, %d control/NaN rows, %d unresolved names (e.g. %s), "
            "%d ORF collisions deduped",
            len(rows),
            n_control_or_nan,
            len(unresolved),
            sorted(unresolved)[:5],
            len(collisions),
        )
        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for record_row in tqdm(rows):
                experiment, reference, publication = self.create_experiment(record_row)
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": publication.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        log.info("Wrote %d Cachera betaxanthin experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one ORF."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                ),
                *_betaxanthin_cassette(),
            ]
        )
        environment = Environment(
            media=Media(name="SC", state="solid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        phenotype = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: row["level"]},
            metabolite_level_se={TARGET_METABOLITE: row["se"]},
            n_replicates={TARGET_METABOLITE: int(row["n"])},
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=None,
        )
        # Reference = the CRI-SPA Donor background (4 betaxanthin genes, no extra
        # deletion). Scores are population-centered, so the control level is 0.
        phenotype_reference = MetabolitePhenotype(
            metabolite_level={TARGET_METABOLITE: 0.0},
            metabolite_level_se=None,
            n_replicates={TARGET_METABOLITE: 1},
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = MetaboliteExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = MetaboliteExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="37572348",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37572348/",
            doi="10.1093/nar/gkad656",
            doi_url="https://doi.org/10.1093/nar/gkad656",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    A genome is REQUIRED (the source uses common gene names). Loads the existing LMDB
    if already built; to step through ``process()``/``create_experiment`` under a
    debugger, delete ``<root>/processed`` first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/betaxanthin_cachera2023")
    dataset = BetaxanthinCachera2023Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

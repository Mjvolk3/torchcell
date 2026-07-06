# torchcell/datasets/scerevisiae/zelezniak2018
# [[torchcell.datasets.scerevisiae.zelezniak2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/zelezniak2018
# Test file: tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py
"""Zelezniak 2018 kinase-knockout proteome dataset (protein-abundance profiles).

Zelezniak et al. 2018 (Cell Systems, doi:10.1016/j.cels.2018.08.001) measured the
quantitative PROTEOME of 97 S. cerevisiae kinase-deletion strains (plus WT) by SWATH-MS
(data-independent acquisition). We ingest the authors' processed per-strain matrix from
Zenodo (``proteins_dataset.data_prep.tsv``): a long-format table of batch-corrected
(SVA-adjusted) label-free protein signal per (protein, sample, KO strain, replicate),
covering 726 proteins across the strains.

We aggregate the replicate samples of each knockout strain to a per-protein mean +
standard error, mapping to ``ProteinAbundancePhenotype`` (WS9):
``protein_abundance = {protein_ORF -> mean log signal}`` with
``measurement_type = "swath_ms_label_free_log_signal_sva"``. The parent **WT** strain
(``KO_ORF == "WT"``) is measured in the same table and supplies the reference profile
(the background is BY4741 made prototrophic by the pHLUM minichromosome). Every protein
has >=2 replicate samples per strain, so the standard error is always defined.

Source (scriptable, hash-pinned): Zenodo record 1320289 (concept DOI
10.5281/zenodo.1320288), file ``proteins_dataset.data_prep.tsv``.
"""

import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "swath_ms_label_free_log_signal_sva"
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")
_WT = "WT"

# Zenodo record 1320289 -> proteins_dataset.data_prep.tsv (direct, pinned).
DATA_URL = (
    "https://zenodo.org/records/1320289/files/proteins_dataset.data_prep.tsv?download=1"
)
DATA_FILENAME = "proteins_dataset.data_prep.tsv"
DATA_SHA256 = "9ff81ecb1e2dd44d2f6e072ce5b628f0be1abdf57cdbd90d645db4d1fb64bfeb"


@register_dataset
class ProteomeZelezniak2018Dataset(ExperimentDataset):
    """SWATH-MS proteome of the yeast kinase-knockout collection (97 strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/proteome_zelezniak2018",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (KO_ORF is systematic; no genome injection needed)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Zenodo proteome matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the proteome matrix from Zenodo and verify its sha256."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Zelezniak proteome from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != DATA_SHA256:
            raise RuntimeError(
                f"Zelezniak proteome sha256 mismatch: got {got}, expected {DATA_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    @staticmethod
    def _aggregate(sub: pd.DataFrame) -> dict[str, Any]:
        """Aggregate one strain's replicate rows to per-protein mean/se/n dicts."""
        grp = sub.groupby("ORF")["value"].agg(["mean", "std", "count"])
        abundance: dict[str, float] = {}
        se: dict[str, float] = {}
        n_reps: dict[str, int] = {}
        for orf, r in grp.iterrows():
            n = int(r["count"])
            abundance[str(orf)] = float(r["mean"])
            n_reps[str(orf)] = n
            se[str(orf)] = float(r["std"]) / math.sqrt(n) if n > 1 else float("nan")
        return {"abundance": abundance, "se": se, "n": n_reps}

    @post_process
    def process(self) -> None:
        """Aggregate the proteome matrix into per-strain experiments and write LMDB."""
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME), sep="\t")
        bad = df[~df["ORF"].astype(str).str.match(_SYSTEMATIC_RE)]
        if len(bad):
            raise RuntimeError(f"non-systematic protein ORF ids present: {len(bad)}")

        wt_rows = df[df["KO_ORF"] == _WT]
        if wt_rows.empty:
            raise RuntimeError("Zelezniak matrix missing the WT reference strain")
        self._reference = self._aggregate(wt_rows)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        n_bad_orf = 0
        rows: list[dict[str, Any]] = []
        for ko_orf, sub in df[df["KO_ORF"] != _WT].groupby("KO_ORF"):
            if not _SYSTEMATIC_RE.match(str(ko_orf)):
                n_bad_orf += 1
                continue
            gene = str(sub["KO_gene_name"].iloc[0])
            rows.append({"orf": str(ko_orf), "gene": gene, "agg": self._aggregate(sub)})
        log.info(
            "Zelezniak: %d knockout strains, WT reference with %d proteins, "
            "%d non-systematic KO_ORF skipped",
            len(rows),
            len(self._reference["abundance"]),
            n_bad_orf,
        )
        pd.DataFrame([{"orf": r["orf"], "gene": r["gene"]} for r in rows]).to_csv(
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
        log.info("Wrote %d Zelezniak proteome experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        ProteinAbundanceExperiment, ProteinAbundanceExperimentReference, Publication
    ]:
        """Build the ProteinAbundance experiment/reference/publication for one strain."""
        # Background = BY4741 kinase-deletion collection made prototrophic by the pHLUM
        # minichromosome (restores HIS3/LEU2/URA3/MET17); pHLUM not yet modeled here.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["gene"]
                )
            ]
        )
        # SWATH-MS on cells in synthetic minimal (SM) liquid medium, 30 C.
        environment = Environment(
            media=Media(name="SM", state="liquid"), temperature=Temperature(value=30)
        )
        agg = row["agg"]
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=agg["abundance"],
            protein_abundance_se=agg["se"],
            n_replicates=agg["n"],
            measurement_type=MEASUREMENT_TYPE,
        )
        ref = self._reference
        phenotype_reference = ProteinAbundancePhenotype(
            protein_abundance=dict(ref["abundance"]),
            protein_abundance_se=dict(ref["se"]),
            n_replicates=dict(ref["n"]),
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = ProteinAbundanceExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="30195436",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
            doi="10.1016/j.cels.2018.08.001",
            doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built. To step through
    ``process()``/``create_experiment`` under a debugger, delete ``<root>/processed``
    first so the build re-runs.
    """
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/proteome_zelezniak2018")
    dataset = ProteomeZelezniak2018Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

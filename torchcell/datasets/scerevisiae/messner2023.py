# torchcell/datasets/scerevisiae/messner2023
# [[torchcell.datasets.scerevisiae.messner2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/messner2023
# Test file: tests/torchcell/datasets/scerevisiae/test_messner2023.py
"""Messner 2023 genome-wide gene-KO proteome dataset (Cell).

Messner et al. 2023 (Cell, doi:10.1016/j.cell.2023.03.026, PMID 37080200) measured
the proteomes of the whole *S. cerevisiae* (S288C) haploid (MATa) gene-deletion
collection with restored prototrophy, by an adapted microflow-SWATH-MS (DIA)
approach. ``ProteomeMessner2023Dataset`` ingests the authors' processed protein
matrix and maps each knockout strain to ``ProteinAbundancePhenotype`` (WS9).

Data (mirror-once + hash-pin). The curated processed matrix is deposited ONLY on
Mendeley Data (doi:10.17632/w8jtmnszd9.1); PRIDE/MassIVE holds only the 74.8 GB raw
DIA-NN report, the Cell SI defers the matrix to Mendeley, and the y5k web app has no
scriptable download. Per the "artifact + sha256 is canonical, not the URL" rule we
fetched the matrix from Mendeley ONCE into the library mirror
(``$DATA_ROOT/torchcell-library/messnerProteomicLandscapeGenomewide2023/data/``) and
pin it by sha256; the loader reads from that mirror -- there is NO live Mendeley
dependency. Two files are used:

- ``yeast5k_noimpute_wide.csv`` -- proteins (rows) x samples (columns) batch-corrected
  MaxLFQ quantities. We use the NO-IMPUTATION variant so every stored value is an
  actually-measured quantity (missing values are dropped per strain, never imputed).
  Row ids are UniProt accessions (``Protein.Group``); 1,850 proteins x 5,476 samples.
- ``yeast5k_metadata.csv`` -- per-sample annotation: ``Filename`` (the matrix column
  id), ``sampletype`` (``ko`` | ``HIS3`` | ``qc``), ``ORF`` (deleted gene), plate nr.

Structure sourced from the paper (STAR Methods, OCR'd mirror):

- **Single-replicate KOs.** "Strains were not measured in replicates." Each of the
  4,699 KO strains is ONE measurement, so per-strain ``n_replicates = 1`` and the
  per-strain standard error is undefined (``protein_abundance_se = None``).
- **388-replicate WT reference.** The control is the his3D::kanMX strain complemented
  by heterologous HIS3 expression (``sampletype == "HIS3"``, ``ORF == YOR202W``),
  measured 388 times across 57 batches. We aggregate those to a per-protein
  mean + standard error (n = number of WT samples in which the protein was measured),
  giving ONE shared reference profile whose SE IS defined.
- **Duplicate strains kept per-instance.** 145 ORFs have >1 deletion strain "of
  different origins" (141 duplicated, 4 triplicated); the paper analyses each strain
  independently, so we emit one experiment per KO SAMPLE (4,699 records), not per ORF.
- **UniProt -> systematic ORF.** Protein ids are UniProt accessions; we map them to
  systematic ORFs (authoritatively, via the SGD S288C GFF ``protein_id=UniProtKB:``
  cross-refs) so this proteome is joinable with the ORF-keyed Zelezniak2018 proteome.
  All 1,850 map (incl. the mito-encoded ``P00410`` -> ``Q0250``/COX2); an unmapped id
  raises rather than being silently dropped.

The value is a LINEAR (not log2) batch-corrected MaxLFQ quantity
(``measurement_type = "swath_ms_maxlfq_batch_corrected_quantity"``); log2 is applied
only downstream in the paper's differential analysis. Background = S288C BY4741 MATa
deletion collection with restored prototrophy (Mulleder); the prototrophy-restoring
marker is not yet modeled as a GeneAddition (as in the Mulleder loader). Cells were
grown in synthetic minimal (SM) liquid medium at 30 C.
"""

import glob
import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
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

MEASUREMENT_TYPE = "swath_ms_maxlfq_batch_corrected_quantity"

# Library-mirror data files (fetched once from Mendeley 10.17632/w8jtmnszd9.1;
# canonical = these bytes + sha256, NOT the Mendeley URL).
_CITATION_KEY = "messnerProteomicLandscapeGenomewide2023"
MATRIX_FILENAME = "yeast5k_noimpute_wide.csv"
MATRIX_SHA256 = "69a9df05b6db011f595a4e0b3ce25c1cc247f22cbdd066c79e6da9a706aa1df9"
METADATA_FILENAME = "yeast5k_metadata.csv"
METADATA_SHA256 = "48864282c82d516ae929dc87aff7fae9e05e9b922e316c001f3d29dce0ff878b"
# Historical retrieval metadata (Mendeley public-file ids), for provenance only.
MENDELEY_DOI = "10.17632/w8jtmnszd9.1"

# Sample classes in yeast5k_metadata.csv "sampletype".
_KO = "ko"
_WT = "HIS3"

# Systematic ORF: nuclear (Y..) for KO deletions; protein ids also include mito
# (Q\d{4}, e.g. COX2 = Q0250) after UniProt->ORF mapping.
_NUCLEAR_ORF_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")
_ORF_RE = re.compile(r"(?:Y[A-P][LR]\d{3}[WC](?:-[A-Z])?|Q\d{4})")
_UNIPROT_RE = re.compile(r"UniProtKB:([A-Z0-9]+)")


def build_uniprot_to_orf_map(data_root: str | None = None) -> dict[str, str]:
    """Map UniProt accession -> systematic ORF from the SGD S288C GFF.

    Authoritative source: the ``protein_id=UniProtKB:<acc>`` cross-references in the
    SGD reference GFF (``$DATA_ROOT/data/sgd/genome/*/saccharomyces_cerevisiae_*.gff``,
    the same file :class:`SCerevisiaeGenome` uses). The first ORF token on a line
    carrying a UniProt id is taken as that accession's ORF. Never invents ids.
    """
    root = data_root if data_root is not None else os.environ["DATA_ROOT"]
    pattern = osp.join(
        root, "data", "sgd", "genome", "*", "saccharomyces_cerevisiae_*.gff"
    )
    gffs = glob.glob(pattern)
    if not gffs:
        raise FileNotFoundError(f"SGD GFF not found under {pattern}")
    up2orf: dict[str, str] = {}
    with open(gffs[0]) as handle:
        for line in handle:
            if "UniProtKB:" not in line:
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                continue
            orf_match = _ORF_RE.search(cols[8])
            if orf_match is None:
                continue
            for acc in _UNIPROT_RE.findall(cols[8]):
                up2orf.setdefault(acc, orf_match.group(0))
    return up2orf


@register_dataset
class ProteomeMessner2023Dataset(ExperimentDataset):
    """Microflow-SWATH-MS proteome of the yeast gene-KO collection (4,699 strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/proteome_messner2023",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (KO ORFs are systematic; no genome injection)."""
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
        """The processed matrix + sample metadata required before processing."""
        return [MATRIX_FILENAME, METADATA_FILENAME]

    def download(self) -> None:
        """Copy the hash-pinned matrix + metadata from the library mirror to raw_dir.

        The mirror is the canonical source (Mendeley fetched once); we verify each
        file's sha256 after copy and never touch a live Mendeley URL.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        mirror = osp.join(
            os.environ["DATA_ROOT"], "torchcell-library", _CITATION_KEY, "data"
        )
        for filename, expected in (
            (MATRIX_FILENAME, MATRIX_SHA256),
            (METADATA_FILENAME, METADATA_SHA256),
        ):
            dest = osp.join(self.raw_dir, filename)
            if osp.exists(dest):
                continue
            src = osp.join(mirror, filename)
            if not osp.exists(src):
                raise FileNotFoundError(
                    f"Messner mirror file missing: {src}. The mirror is canonical; "
                    f"restore it from backup (fetched once from Mendeley {MENDELEY_DOI})."
                )
            h = hashlib.sha256()
            with open(src, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise RuntimeError(
                    f"Messner {filename} sha256 mismatch: got {h.hexdigest()}, "
                    f"expected {expected}"
                )
            shutil.copy2(src, dest)
            log.info(
                "Copied %s from mirror (%d bytes, sha256 verified)",
                filename,
                os.path.getsize(dest),
            )

    @post_process
    def process(self) -> None:
        """Aggregate the proteome matrix into per-strain experiments and write LMDB."""
        matrix = pd.read_csv(osp.join(self.raw_dir, MATRIX_FILENAME))
        id_col = matrix.columns[0]  # "Protein.Group" (UniProt accession)
        matrix = matrix.set_index(id_col)

        # UniProt -> systematic ORF; every measured protein must map (fail loud).
        up2orf = build_uniprot_to_orf_map()
        unmapped = [p for p in matrix.index.astype(str) if p not in up2orf]
        if unmapped:
            raise RuntimeError(
                f"{len(unmapped)} Messner proteins have no UniProt->ORF mapping "
                f"(e.g. {unmapped[:5]}); refusing to silently drop."
            )
        matrix.index = matrix.index.astype(str).map(up2orf)

        meta = pd.read_csv(osp.join(self.raw_dir, METADATA_FILENAME))
        by_filename = meta.set_index("Filename")
        cols = set(matrix.columns)
        wt_cols = [f for f in meta[meta["sampletype"] == _WT]["Filename"] if f in cols]
        ko_cols = [f for f in meta[meta["sampletype"] == _KO]["Filename"] if f in cols]
        if not wt_cols:
            raise RuntimeError("Messner matrix missing the HIS3 (WT) control columns")

        # WT reference: per-protein mean / SE / n across the 388 WT replicates
        # (non-NaN only). This one profile is shared by every KO record.
        wt = matrix[wt_cols]
        wt_mean = wt.mean(axis=1)
        wt_std = wt.std(axis=1, ddof=1)
        wt_n = wt.count(axis=1)
        keep = wt_n[wt_n >= 1].index
        self._reference: dict[str, Any] = {
            "abundance": {str(o): float(wt_mean[o]) for o in keep},
            "se": {
                str(o): (
                    float(wt_std[o]) / math.sqrt(int(wt_n[o]))
                    if int(wt_n[o]) > 1
                    else float("nan")
                )
                for o in keep
            },
            "n": {str(o): int(wt_n[o]) for o in keep},
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        rows: list[dict[str, Any]] = []
        n_bad_orf = 0
        for filename in ko_cols:
            # Metadata ORF casing is inconsistent (e.g. "YML009c", "YAL043C-a");
            # systematic names are uppercase -- normalize, never drop on case alone.
            deletion_orf = str(by_filename.loc[filename, "ORF"]).upper()
            if not _NUCLEAR_ORF_RE.match(deletion_orf):
                n_bad_orf += 1
                continue
            series = matrix[filename].dropna()  # measured proteins only (noimpute)
            abundance = {str(o): float(v) for o, v in series.items()}
            rows.append(
                {
                    "filename": filename,
                    "orf": deletion_orf,
                    "gene": _gene_from_filename(filename, deletion_orf),
                    "abundance": abundance,
                }
            )
        log.info(
            "Messner: %d KO strains, WT reference with %d proteins, %d non-systematic "
            "KO ORF skipped",
            len(rows),
            len(self._reference["abundance"]),
            n_bad_orf,
        )
        pd.DataFrame(
            [
                {
                    "filename": r["filename"],
                    "orf": r["orf"],
                    "gene": r["gene"],
                    "n_proteins": len(r["abundance"]),
                }
                for r in rows
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

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
        log.info("Wrote %d Messner proteome experiments to LMDB", idx)

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
        # Background = S288C BY4741 MATa deletion collection with restored prototrophy
        # (Mulleder); the prototrophy-restoring marker is not yet modeled as a
        # GeneAddition here (as in the Mulleder loader).
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
        # Grown in synthetic minimal (SM) liquid medium at 30 C.
        environment = Environment(
            media=Media(name="SM", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30),
        )
        # Single-replicate KO: n=1 per protein, per-strain SE undefined.
        abundance = row["abundance"]
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=abundance,
            protein_abundance_se=None,
            n_replicates={orf: 1 for orf in abundance},
            measurement_type=MEASUREMENT_TYPE,
        )
        # Reference = the 388-replicate WT baseline RESTRICTED to the proteins this
        # strain measured (as in the Zelezniak metabolite loader). Keeps reference keys
        # a subset-equal of the experiment's and every reference value a real WT
        # measurement; all 1,850 proteins have WT coverage, so every exp key resolves.
        ref = self._reference
        phenotype_reference = ProteinAbundancePhenotype(
            protein_abundance={k: ref["abundance"][k] for k in abundance},
            protein_abundance_se={k: ref["se"][k] for k in abundance},
            n_replicates={k: ref["n"][k] for k in abundance},
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
            pubmed_id="37080200",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37080200/",
            doi="10.1016/j.cell.2023.03.026",
            doi_url="https://doi.org/10.1016/j.cell.2023.03.026",
        )
        return experiment, reference, publication


def _gene_from_filename(filename: str, orf: str) -> str:
    """Standard gene name from the sample Filename, falling back to the ORF.

    Filenames look like ``10_9_hpr1_ko_YAL059W_ECM1_0.47`` -- the token immediately
    after the deletion ORF is the standard gene name (or a lowercased ORF when no
    standard name exists). The ORF token is matched case-insensitively (metadata
    casing is inconsistent). Falls back to the systematic ORF if not parseable.
    """
    parts = filename.split("_")
    upper = [p.upper() for p in parts]
    if orf.upper() in upper:
        i = upper.index(orf.upper())
        if i + 1 < len(parts) and parts[i + 1]:
            return parts[i + 1]
    return orf


def main() -> None:
    """Build/load the dataset for interactive debugging.

    Loads the existing LMDB if already built; delete ``<root>/processed`` to re-run
    ``process()`` under a debugger.
    """
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/proteome_messner2023")
    dataset = ProteomeMessner2023Dataset(root=root)
    print(f"proteome len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

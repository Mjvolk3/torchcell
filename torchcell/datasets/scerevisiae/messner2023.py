# torchcell/datasets/scerevisiae/messner2023
# [[torchcell.datasets.scerevisiae.messner2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/messner2023
# Test file: tests/torchcell/datasets/scerevisiae/test_messner2023.py
"""Messner 2023 genome-wide gene-KO proteome dataset (Cell).

Messner et al. 2023 (Cell, doi:10.1016/j.cell.2023.03.026, PMID 37080200) measured
the proteomes of the whole *S. cerevisiae* (S288C) haploid (MATa) gene-deletion
collection with restored prototrophy, by an adapted microflow-SWATH-MS (DIA)
approach. ``ProteomeMessner2023Dataset`` ingests the authors' processed protein
matrix and maps each knockout strain and each HIS3 control culture to
``ProteinAbundancePhenotype`` (WS9).

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

- **One record per released sample (per-replicate policy, #1178 / #1171).** Every
  ``ko`` and every ``HIS3`` column of the matrix is ONE culture in its own well (0 of
  the 5,087 share a plate and well), and each is its own record: 4,699 KO records then
  388 control records, 5,087 in all. ``replicate_id`` is the column's ``Filename``
  verbatim (the release's own sample label), ``n_replicates = 1`` per protein,
  ``protein_abundance_se = None`` with a ``ProvenanceGap`` of reason
  ``not_applicable_single_measurement``: one measurement has no dispersion.
- **Single-replicate KOs.** "Strains were not measured in replicates." (paper.md L579.)
- **The 388 HIS3 controls are biological replicates and are records.** "The control
  strain (388 replicates) is the complemented his3D deletion strain, haploid from a
  BY4741 prototrophic deletion collection. This control strain was introduced in 7
  positions on each plate: A11, B8, C5, D2, F11, G8, H5." (paper.md L507); Figure 1D
  calls them "the biological replicates of the wild-type strain" (L69). Their genotype
  is the library's own ``his3D::kanMX4`` strain (metadata ``ORF == YOR202W``): Mulleder
  2016 (this paper's ref 15) names "the complemented deletion strains his3D::kanMX4 and
  met17D::kanMX4" (its paper.md L379), complemented by the library's "minichromosome
  carrying the genes HIS3, LEU2, URA3 and MET17" (its L487), which every strain of the
  collection carries. So a control record is ``KanMxDeletionPerturbation(YOR202W)``
  under the same convention as every KO record (the minichromosome is modeled on no
  record; see below); no ``ko`` sample deletes YOR202W.
- **The reference is a summary, and it says so.** ``phenotype_reference`` is the
  per-protein mean + sample-SD standard error over the 388 HIS3 columns (n = number of
  HIS3 samples in which the protein was measured), restricted to the proteins the
  record measured, with ``replicate_id = None``. The 388 cultures it summarizes are
  themselves records.
- **Duplicate strains kept per-instance.** The paper states "for 145 ORFs, more than one
  strain exists in the library (these strains have different origins). 141 gene
  deletions are duplicated and 4 triplicated" (L579), which is exactly the pinned
  metadata read case-sensitively. Upper-cased, ``YML009C`` and ``YML009c`` (both
  labeled MRPL39 in their Filenames) join, so 146 ORFs carry 2 or 3 KO samples
  (142 x 2, 4 x 3) over 4,549 ORFs. Each strain is one record, told apart by its
  ``replicate_id`` (the ``Filename``).
- **Not records: the 389 ``qc`` columns.** They are repeat injections of a pooled digest
  ("QC samples for repeat injections were prepared by pooling digested and cleaned-up
  samples from 4 different 96-well plates", L525), a technical control of no strain.
- **Protein keys are the released UniProt accessions.** ``protein_abundance`` is keyed
  by the matrix's ``Protein.Group`` accession verbatim (``P00410``, the mito-encoded
  COX2, included). No annotation is consulted (#1045, 2026.10.12): the previous build
  lifted every accession to its ORF through the SGD GFF ``protein_id=UniProtKB:``
  cross-references; joining an accession to a gene is a modeling-time step, kept out of
  the stored record.
- **Gene names are the released ORF.** ``systematic_gene_name`` and
  ``perturbed_gene_name`` are both the metadata ``ORF`` (uppercased); no standard name
  is looked up in the GFF (the previous build read one for 3,780 of 4,699 records). The
  token after the ORF in the sample Filename is never read: on real filenames it is
  sometimes a number (``10_9_hpr57_ko_YBL007C_2824_0.49``) or a lowercase ORF. A
  metadata ORF with no ORF form is counted under ``deferred_to_released_gene_name_leaf``
  (0 of 4,699).

The value is a LINEAR (not log2) batch-corrected MaxLFQ quantity
(``measurement_type = "swath_ms_maxlfq_batch_corrected_quantity"``); log2 is applied
only downstream in the paper's differential analysis. Background = S288C BY4741 MATa
deletion collection with restored prototrophy (Mulleder); the prototrophy-restoring
marker is not yet modeled as a GeneAddition (as in the Mulleder loader). Cells were
grown in synthetic minimal (SM) liquid medium at 30 C.
"""

import glob
import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.parse
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    copy_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import SM
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

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

_SINGLE_SAMPLE_SE_NOTE = (
    "a record is ONE released sample (one culture, one acquisition): 'Strains were not "
    "measured in replicates.' (paper.md L579), and each of the 388 HIS3 controls is one "
    "of 'the biological replicates of the wild-type strain' (L69). One measurement has "
    "no dispersion of its own; the per-protein SE over the 388 controls is what "
    "phenotype_reference carries"
)

# Systematic ORF: nuclear (Y..) for KO deletions; protein ids also include mito
# (Q\d{4}, e.g. COX2 = Q0250) after UniProt->ORF mapping.
_NUCLEAR_ORF_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")


def _released_deletion(
    orf: str,
) -> KanMxDeletionPerturbation | ReleasedIdentifierPerturbation:
    """The deletion under the released identifier (#1045): an ORF-shaped one on the
    placed KanMX leaf with both names the ORF, any other on the released-identifier leaf
    verbatim (kind deletion, marker KanMX).
    """
    if _NUCLEAR_ORF_RE.match(orf):
        return KanMxDeletionPerturbation(
            systematic_gene_name=orf, perturbed_gene_name=orf
        )
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=orf,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=orf,
        identifier_kind="ORF column entry of the sample metadata, as released",
        released_kind=ReleasedKind.deletion,
        cassette="KanMX",
    )


_ORF_RE = re.compile(r"(?:Y[A-P][LR]\d{3}[WC](?:-[A-Z])?|Q\d{4})")
_UNIPROT_RE = re.compile(r"UniProtKB:([A-Z0-9]+)")


class MissingWildTypeReferenceError(ValueError):
    """A KO sample measured a protein that no HIS3 (WT) sample measured.

    The reference is the WT profile restricted to the proteins a strain measured, so
    such a protein has no reference value. The pinned matrix has none (0 of 1,850
    proteins are measured in a KO sample and in no WT sample).
    """


def _sgd_gff_path(data_root: str | None = None) -> str:
    """Path of the SGD S288C reference GFF under ``$DATA_ROOT`` (or ``data_root``).

    ``$DATA_ROOT/data/sgd/genome/*/saccharomyces_cerevisiae_*.gff``, the same file
    :class:`SCerevisiaeGenome` uses; the first glob match is read.
    """
    root = data_root if data_root is not None else os.environ["DATA_ROOT"]
    pattern = osp.join(
        root, "data", "sgd", "genome", "*", "saccharomyces_cerevisiae_*.gff"
    )
    gffs = glob.glob(pattern)
    if not gffs:
        raise FileNotFoundError(f"SGD GFF not found under {pattern}")
    return gffs[0]


def build_uniprot_to_orf_map(data_root: str | None = None) -> dict[str, str]:
    """Map UniProt accession -> systematic ORF from the SGD S288C GFF.

    A downstream utility, not loader logic: this loader stores the released accessions
    (#1045). Kept for ``messner2021``, which still calls it.

    Authoritative source: the ``protein_id=UniProtKB:<acc>`` cross-references in the
    SGD reference GFF (see :func:`_sgd_gff_path`). The first ORF token on a line
    carrying a UniProt id is taken as that accession's ORF. Never invents ids.
    """
    up2orf: dict[str, str] = {}
    with open(_sgd_gff_path(data_root)) as handle:
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


def build_orf_to_gene_name_map(data_root: str | None = None) -> dict[str, str]:
    """Map nuclear systematic ORF -> SGD standard gene name, from the SGD S288C GFF.

    Every GFF feature whose ``ID`` attribute IS a nuclear systematic ORF (``gene``,
    ``pseudogene``, ``transposable_element_gene``, ``blocked_reading_frame`` in
    R64-4-1) contributes one entry: its ``gene=`` attribute, percent-decoded (GFF3
    escapes ``(`` and ``)``, so ``MF%28ALPHA%291`` is ``MF(ALPHA)1``), or the ORF itself
    when the feature has no ``gene=`` (no standard name). The first feature per ORF
    wins. An ORF that is no feature's ``ID`` (retired or renamed in this GFF release)
    is absent from the map; the caller decides what to store for it.
    """
    orf2name: dict[str, str] = {}
    with open(_sgd_gff_path(data_root)) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
            feature_id = attrs.get("ID", "")
            if not _NUCLEAR_ORF_RE.match(feature_id):
                continue
            name = attrs.get("gene")
            orf2name.setdefault(
                feature_id, urllib.parse.unquote(name) if name else feature_id
            )
    return orf2name


@register_dataset
class ProteomeMessner2023Dataset(ExperimentDataset):
    """Microflow-SWATH-MS proteome of the yeast gene-KO collection: one record per
    released sample (4,699 KO strains and 388 HIS3 control cultures, 5,087 records).
    """

    organism = Organism.scerevisiae

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

        The mirror is the canonical source (Mendeley fetched once); each file's sha256
        is verified before the copy (and again at build time by ``process``), and no
        live Mendeley URL is touched.
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
            copy_verified(src, dest, expected)
            log.info(
                "Copied %s from mirror (%d bytes, sha256 verified)",
                filename,
                os.path.getsize(dest),
            )

    @post_process
    def process(self) -> None:
        """Write one experiment per released KO and HIS3 sample column to LMDB."""
        verify_raw_files(
            self.raw_dir,
            {MATRIX_FILENAME: MATRIX_SHA256, METADATA_FILENAME: METADATA_SHA256},
        )
        matrix = pd.read_csv(osp.join(self.raw_dir, MATRIX_FILENAME))
        id_col = matrix.columns[0]  # "Protein.Group" (UniProt accession)
        matrix = matrix.set_index(id_col)

        # Protein keys are the released UniProt accessions, verbatim (#1045).
        matrix.index = matrix.index.astype(str)
        repeated = matrix.index[matrix.index.duplicated()].tolist()
        if repeated:
            raise RuntimeError(
                f"Messner matrix lists {len(repeated)} accession(s) twice: {repeated[:5]}"
            )

        meta = pd.read_csv(osp.join(self.raw_dir, METADATA_FILENAME))
        by_filename = meta.set_index("Filename")
        cols = set(matrix.columns)
        wt_cols = [f for f in meta[meta["sampletype"] == _WT]["Filename"] if f in cols]
        ko_cols = [f for f in meta[meta["sampletype"] == _KO]["Filename"] if f in cols]
        if not wt_cols:
            raise RuntimeError("Messner matrix missing the HIS3 (WT) control columns")

        # WT reference: per-protein mean / SE / n across the 388 WT replicates
        # (non-NaN only). This one summary is shared by every record; the 388 control
        # cultures it summarizes are records too (written after the KOs, below).
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
        released_names: list[str] = []
        # One record per released sample: the KO strains first (their order and
        # indices are unchanged from the per-strain build), then the HIS3 controls.
        for sampletype, filenames in ((_KO, ko_cols), (_WT, wt_cols)):
            for filename in filenames:
                # Metadata ORF casing is inconsistent (e.g. "YML009c", "YAL043C-a");
                # systematic names are uppercase -- normalize, never drop on case alone.
                deletion_orf = str(by_filename.loc[filename, "ORF"]).upper()
                if not _NUCLEAR_ORF_RE.match(deletion_orf):
                    # no ORF form: the released-identifier leaf, verbatim (0 of 5,087)
                    released_names.append(deletion_orf)
                series = matrix[filename].dropna()  # measured proteins only (noimpute)
                abundance = {str(o): float(v) for o, v in series.items()}
                no_wt = sorted(set(abundance) - set(self._reference["abundance"]))
                if no_wt:
                    raise MissingWildTypeReferenceError(
                        f"Messner: KO sample {filename} ({deletion_orf}) measured "
                        f"{len(no_wt)} protein(s) no WT sample measured (e.g. "
                        f"{no_wt[:5]}); the reference has no value for them"
                    )
                rows.append(
                    {
                        "filename": filename,
                        "sampletype": sampletype,
                        "orf": deletion_orf,
                        "gene": deletion_orf,
                        "abundance": abundance,
                    }
                )
        log.info(
            "Messner: %d KO strains and %d HIS3 control samples, WT reference with %d "
            "proteins, %d ORFs on the released-identifier leaf",
            len(ko_cols),
            len(wt_cols),
            len(self._reference["abundance"]),
            len(released_names),
        )
        pd.DataFrame(
            [
                {
                    "filename": r["filename"],
                    "sampletype": r["sampletype"],
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
        log.info("Wrote %d Messner proteome sample records to LMDB", idx)

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
        """Build the ProteinAbundance experiment/reference/publication for one sample.

        ``row`` is one released ``ko`` or ``HIS3`` column; a control's ``orf`` is the
        metadata ``YOR202W``, so its genotype is the library's his3D::kanMX4 deletion.
        """
        # Background = S288C BY4741 MATa deletion collection with restored prototrophy
        # (Mulleder); the prototrophy-restoring marker is not yet modeled as a
        # GeneAddition here (as in the Mulleder loader).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        genotype = Genotype(perturbations=[_released_deletion(row["orf"])])
        # Grown in synthetic minimal (SM) liquid medium at 30 C: 6.7 g/L YNB without
        # amino acids + 2% glucose, restated by this paper and deferred by it to
        # Mulleder 2016 (its ref 15), both quoted on ``SM``.
        environment = Environment(media=SM, temperature=Temperature(value=30))
        # One released sample: its own values, n = 1 per protein, no dispersion.
        abundance = row["abundance"]
        phenotype = ProteinAbundancePhenotype(
            protein_abundance=abundance,
            protein_abundance_se=None,
            n_replicates=dict.fromkeys(abundance, 1),
            measurement_type=MEASUREMENT_TYPE,
            replicate_id=row["filename"],
            provenance_gaps=[
                ProvenanceGap(
                    field="protein_abundance_se",
                    reason=ProvenanceGapReason.not_applicable_single_measurement,
                    note=_SINGLE_SAMPLE_SE_NOTE,
                )
            ],
        )
        # Reference = the 388-replicate WT summary RESTRICTED to the proteins this
        # sample measured (as in the Zelezniak metabolite loader). Keeps reference keys
        # a subset-equal of the experiment's and every reference value a real WT
        # measurement; all 1,850 proteins have WT coverage, so every exp key resolves.
        ref = self._reference
        phenotype_reference = ProteinAbundancePhenotype(
            protein_abundance={k: ref["abundance"][k] for k in abundance},
            protein_abundance_se={k: ref["se"][k] for k in abundance},
            n_replicates={k: ref["n"][k] for k in abundance},
            measurement_type=MEASUREMENT_TYPE,
            replicate_id=None,
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

# torchcell/datasets/scerevisiae/nadal_ribelles2025
# [[torchcell.datasets.scerevisiae.nadal_ribelles2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/nadal_ribelles2025
# Test file: tests/torchcell/datasets/scerevisiae/test_nadal_ribelles2025.py
"""Nadal-Ribelles 2025 genome-scale single-cell Perturb-seq (pseudobulk, WS15).

Nadal-Ribelles et al. 2025 (Nat. Commun. 16, doi:10.1038/s41467-025-57600-4) reengineered
the yeast knockout collection (YKOC) into an RNA-traceable clone+genotype barcoded deletion
library (KANMX4 marker replaced by URA3 so the genotype barcode in the URA3 3'UTR is
polyA-readable), then profiled >3500 non-essential deletion mutants by microwell single-cell
RNA-seq under CONTROL and osmostress (0.4 M NaCl, 15 min) conditions.

REPRESENTATION (locked decision -- PSEUDOBULK per genotype, NOT per-cell; the 43 GB
single-cell object is NOT used):
  - Phenotype ``PseudobulkExpressionPhenotype``: per-gene pseudobulk log2 fold-change vs the
    WILD TYPE profiled in the SAME condition (scanpy ``logfoldchanges``; Wilcoxon rank-sum
    DE), PLUS the two per-genotype single-cell scalars that a bulk assay cannot give and are
    the POINT of the pseudobulk+dispersion decision:
      * ``dispersion`` = SD of the scaled SVD leverage score across the genotype's cells
        (source ``sd_lvscore_scaledFU2``; z-scored vs WT so WT ~= 1) -- transcriptional
        HETEROGENEITY.
      * ``n_cells`` = number of assigned single cells (source ``cell_number``).
  - Genotype: a single ``MarkerDeletionPerturbation`` (marker ``URA3``; provenance
    engineered). The source genotype barcode label (``bc_YAL012W``; replacement strains
    ``bc_YBR020W-1``/``-2``) is carried on ``strain_id`` so replicate strains that delete the
    same ORF in the same condition stay distinct records.
  - Environment: two records per genotype -- CONTROL (base YPD) and OSMOSTRESS (YPD + 0.4 M
    NaCl, 15 min, a ``SmallMoleculePerturbation``).
  - Reference = the WT profiled in that condition: log2 fold-change 0 for every gene, carrying
    the WT ``dispersion`` (~1) and ``n_cells`` for that condition. Two references (control,
    NaCl); ``reference_centered = True``.

RAGGED gene sets: each mutant-vs-WT DE first drops genes with 0 counts, so the tested gene
set differs per genotype (min 4223, max 6313, median ~5837 genes; union 6796, all 6188
tables distinct). A gene not tested for a genotype is KEY-ABSENT (no key), NEVER 0 -- the same
honest convention the RNA-seq expression family uses for genome-absent genes. Gene common
names (``MET14``/``MUP1``) are mapped to current-R64 systematic ORFs via the genome alias
table; unresolvable names (~6.5%: ncRNAs/retired symbols such as ``15S-RRNA``/``SNR*``) are
dropped and logged.

SOURCED VALUES (paper Methods; OCR ``paper.md`` in the library mirror):
  - Environment: profiled cells grown in YPD to mid-exponential (OD660 0.6-0.8); osmostress =
    0.4 M NaCl for 15 min ("subjected or not to stress (0.4 M NaCl for 15 min)"). Media = YPD
    liquid. TEMPERATURE of the 6-h YPD growth + 15-min treatment is NOT explicitly stated
    (only the 48-h URA- recovery is stated, at 25 C); 30 C (standard S. cerevisiae growth,
    and the temperature used for this paper's own growth-curve assays) is used as the
    documented representative and FLAGGED for review. Temperature is a shared constant across
    both conditions and every genotype, so it does not affect any mutant-vs-WT logFC.
  - Perturbation: single non-essential gene deletion; URA3 marker swap ("replace the KAN
    resistance marker with URA3"). RNA-barcoded.
  - WT reference: "A total of 7 different WT strains were generated with distinct clone and
    genotype barcodes as controls"; each mutant "compared to wild type in the corresponding
    condition" -> reference logFC 0.
  - Dispersion: "standard deviation of the scaled leverage score" (SVD leverage score, Replogle
    2022 method; z-scored vs WT). n_cells: ``cell_number``.

DATA SOURCE + PROVENANCE (Zenodo 10.5281/zenodo.14062629; raw mirror
``$DATA_ROOT/torchcell-raw/nadalRibelles2025/``, read with the pure-Python ``rdata`` reader):
  - ``FC_genotype.Rdata`` (R object ``fcs``; sha256 c210fe54...): 6188 DEG tables keyed
    ``DEG_{Control,NaCl}_bc_<ORF>.csv``, columns ``names`` (gene) + ``logfoldchanges``.
  - ``ptb_summary.Rdata`` (R object ``ptbs``; sha256 01c2d54a...): per-condition table with
    ``cell_number`` + ``sd_lvscore_scaledFU2`` per genotype.
  - ``README.txt`` (sha256 268533b1...).
"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    MarkerDeletionPerturbation,
    Media,
    PseudobulkExpressionExperiment,
    PseudobulkExpressionExperimentReference,
    PseudobulkExpressionPhenotype,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MEASUREMENT_TYPE = "pseudobulk_scrnaseq_log2fc"
DELETION_MARKER = "URA3"
NACL_MOLAR = 0.4  # osmostress: 0.4 M NaCl
TREATMENT_HOURS = 0.25  # 15 min osmostress
# Assay-phase temperature is NOT explicitly stated in the paper (only the 48-h URA- recovery,
# at 25 C). 30 C is the documented representative (standard growth + this paper's own
# growth-curve assays); FLAGGED for review. Shared across both conditions -> no logFC effect.
GROWTH_TEMP_C = 30.0

RAW_DIR_REL = "torchcell-raw/nadalRibelles2025"
FC_NAME = "FC_genotype.Rdata"
PTB_NAME = "ptb_summary.Rdata"
README_NAME = "README.txt"
FC_SHA256 = "c210fe541b0b91bc6eead28aa2265065afceec763ade1abd682c58896299a240"
PTB_SHA256 = "01c2d54ac838179be29694ed300cb17edac47dd4db23a4018407546e0651b165"
README_SHA256 = "268533b10c59d3f4ca941ff31ac8b9c108b61f55f00d85792d44b3a90b3b9da8"
RAW_FILES = [FC_NAME, PTB_NAME, README_NAME]
SHA256_EXPECTED = {FC_NAME: FC_SHA256, PTB_NAME: PTB_SHA256, README_NAME: README_SHA256}

ZENODO_DOI = "10.5281/zenodo.14062629"
PAPER_DOI = "10.1038/s41467-025-57600-4"

_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
_REPLACEMENT_RE = re.compile(r"^(.+?)-\d+$")  # base-<digits> replacement-strain label


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_fc_key(key: str) -> tuple[str, str]:
    """``DEG_Control_bc_YAL012W.csv`` -> ``("Control", "YAL012W")`` (raw ORF label)."""
    ks = key[:-4] if key.endswith(".csv") else key
    if not ks.startswith("DEG_"):
        raise ValueError(f"unexpected fcs key (no DEG_ prefix): {key!r}")
    cond, raw_orf = ks[4:].split("_bc_", 1)
    return cond, raw_orf


def _deletion_systematic(raw_orf: str) -> str | None:
    """Base systematic ORF a genotype label deletes, else None.

    A valid ORF name (incl. a ``-A`` suffix, e.g. ``YAL037C-A``) is kept as-is; a
    replacement-strain label (``YBR020W-1``) is stripped of its ``-<digits>`` suffix to the
    base ORF (the strain identity is preserved separately on ``strain_id``). Case is
    normalized (one source label, ``YMR175w-1``, is lower-case).
    """
    up = raw_orf.upper()
    if _SYSTEMATIC_RE.match(up):
        return up
    match = _REPLACEMENT_RE.match(up)
    if match and _SYSTEMATIC_RE.match(match.group(1)):
        return match.group(1)
    return None


@register_dataset
class NadalRibellesPerturbSeq2025Dataset(ExperimentDataset):
    """Nadal-Ribelles 2025 pseudobulk Perturb-seq: one record per (genotype, condition)."""

    def __init__(
        self,
        root: str = "data/torchcell/nadal_ribelles_perturbseq2025",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for common-name -> ORF mapping."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return PseudobulkExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return PseudobulkExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Zenodo R objects + README (symlinked from the raw mirror)."""
        return RAW_FILES

    def _data_root(self) -> str:
        """Return $DATA_ROOT (the raw-mirror parent)."""
        from dotenv import load_dotenv

        load_dotenv()
        return os.environ["DATA_ROOT"]

    def download(self) -> None:
        """Symlink the hash-pinned mirror files into ``raw_dir`` and verify each sha256.

        The 426 MB ``FC_genotype.Rdata`` is referenced in place (never copied). Every file's
        sha256 is verified against the pinned value.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in RAW_FILES:
            src = osp.join(data_root, RAW_DIR_REL, name)
            if not osp.exists(src):
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            got = _sha256(src)
            if got != SHA256_EXPECTED[name]:
                raise RuntimeError(
                    f"{name} sha256 mismatch: got {got}, expected {SHA256_EXPECTED[name]}"
                )
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Nadal-Ribelles raw files linked into %s (sha256 verified)", self.raw_dir
        )

    def _resolvers(self) -> tuple[Any, dict[str, str]]:
        """Build (common-name -> R64 ORF) resolver + (ORF -> common name) map from genome."""
        if self.genome is None:
            raise RuntimeError(
                "NadalRibellesPerturbSeq2025Dataset requires a genome for gene-name "
                "resolution; inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        df = genome.gene_attribute_table
        ids = set(df["ID"].astype(str))
        alias_map = genome.alias_to_systematic
        gene_col = dict(zip(df["gene"].astype(str), df["ID"].astype(str)))
        alias_col = dict(zip(df["Alias"].astype(str), df["ID"].astype(str)))
        sys_to_common = dict(zip(df["ID"].astype(str), df["gene"].astype(str)))

        def resolve(token: str) -> str | None:
            gene = str(token).upper()
            if gene in ids:
                return gene
            if _SYSTEMATIC_RE.match(gene):
                cand = alias_map.get(gene, [])
                if cand and cand[0] in ids:
                    return cand[0]
            if gene in gene_col:
                return gene_col[gene]
            if gene in alias_col:
                return alias_col[gene]
            cand = alias_map.get(gene, [])
            if cand and cand[0] in ids:
                return cand[0]
            return None

        return resolve, sys_to_common

    def _load_ptbs(self, data_root: str) -> dict[str, pd.DataFrame]:
        """Load ptb_summary -> {condition: DataFrame indexed by normalized genotype label}.

        The genotype label ``assignment_consensus2`` uses a hyphen (``bc-YAL012W``); it is
        normalized to the fcs underscore form (``bc_YAL012W``). ``WT`` is left as-is.
        """
        import rdata

        ptbs = rdata.read_rda(osp.join(self.raw_dir, PTB_NAME))["ptbs"]
        out: dict[str, pd.DataFrame] = {}
        for cond in ptbs:
            df = ptbs[cond].copy()
            df.columns = [str(c) for c in df.columns]
            labels = [str(x) for x in df["assignment_consensus2"]]
            df["_geno"] = [
                label.replace("bc-", "bc_", 1) if label.startswith("bc-") else label
                for label in labels
            ]
            out[str(cond)] = df.set_index("_geno")
        return out

    @post_process
    def process(self) -> None:
        """Build the per-(genotype, condition) pseudobulk records and write LMDB."""
        import rdata

        data_root = self._data_root()
        os.makedirs(self.preprocess_dir, exist_ok=True)

        resolve, sys_to_common = self._resolvers()
        ptbs = self._load_ptbs(data_root)
        # ptbs condition keys are lower-case ('control'/'nacl'); fcs uses 'Control'/'NaCl'.
        ptb_by_cond = {k.lower(): v for k, v in ptbs.items()}

        log.info("Loading %s (426 MB; ~6 min) ...", FC_NAME)
        fcs = rdata.read_rda(osp.join(self.raw_dir, FC_NAME))["fcs"]

        # ---- Pass 1: resolve every record's logFC vector; accumulate per-condition union.
        records: list[dict[str, Any]] = []
        cond_union: dict[str, dict[str, float]] = {}
        n_gene_dropped = 0
        n_gene_kept = 0
        n_collision = 0
        n_orf_unresolved = 0
        for key in tqdm(fcs.keys(), desc="resolving pseudobulk records"):
            cond, raw_orf = _parse_fc_key(str(key))
            sys_name = _deletion_systematic(raw_orf)
            if sys_name is None:
                n_orf_unresolved += 1
                log.warning("Dropping record with unparseable ORF label: %s", raw_orf)
                continue
            table = fcs[key]
            names = [str(x) for x in table["names"]]
            lfcs = [float(x) for x in table["logfoldchanges"]]
            expr: dict[str, float] = {}
            for name, lfc in zip(names, lfcs):
                orf = resolve(name)
                if orf is None:
                    n_gene_dropped += 1
                    continue
                if orf in expr:
                    n_collision += 1
                    continue  # keep first common-name mapping to this ORF
                expr[orf] = lfc
                n_gene_kept += 1
            if not expr:
                log.warning(
                    "Record %s/%s has no resolvable genes; skipping", cond, raw_orf
                )
                continue
            cond_union.setdefault(cond.lower(), {}).update({g: 0.0 for g in expr})
            strain_label = f"bc_{raw_orf}"
            dispersion, n_cells = self._ptb_scalars(
                ptb_by_cond.get(cond.lower()), strain_label
            )
            records.append(
                {
                    "condition": cond.lower(),
                    "systematic_gene_name": sys_name,
                    "perturbed_gene_name": sys_to_common.get(sys_name) or sys_name,
                    "strain_id": strain_label,
                    "expression_log2_ratio": expr,
                    "dispersion": dispersion,
                    "n_cells": n_cells,
                }
            )
        log.info(
            "Resolved %d records; genes kept=%d dropped=%d (unresolvable), "
            "collisions=%d, ORF-labels unparsed=%d",
            len(records),
            n_gene_kept,
            n_gene_dropped,
            n_collision,
            n_orf_unresolved,
        )

        # ---- Per-condition WT reference (logFC 0 over the condition union; WT dispersion/n).
        self._references = {
            cond: self._build_reference(cond, union, ptb_by_cond.get(cond))
            for cond, union in cond_union.items()
        }

        # ---- Pass 2: assemble typed records and write LMDB.
        pd.DataFrame(
            {
                "strain_id": [r["strain_id"] for r in records],
                "condition": [r["condition"] for r in records],
            }
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        with env.begin(write=True) as txn:
            for idx, row in enumerate(tqdm(records, desc="writing LMDB")):
                experiment, reference, publication = self.create_experiment(row)
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
        env.close()
        log.info("Wrote %d pseudobulk records to LMDB", len(records))

    @staticmethod
    def _ptb_scalars(
        ptb: pd.DataFrame | None, strain_label: str
    ) -> tuple[float | None, int | None]:
        """Return (dispersion=sd_lvscore_scaledFU2, n_cells=cell_number) for a strain."""
        if ptb is None or strain_label not in ptb.index:
            return None, None
        rowobj = ptb.loc[strain_label]
        row = rowobj.iloc[0] if isinstance(rowobj, pd.DataFrame) else rowobj
        dispersion = float(row["sd_lvscore_scaledFU2"])
        n_cells = int(round(float(row["cell_number"])))
        return dispersion, n_cells

    def _build_reference(
        self, cond: str, union: dict[str, float], ptb: pd.DataFrame | None
    ) -> PseudobulkExpressionExperimentReference:
        """WT reference for one condition: logFC 0 over the union; WT dispersion + n_cells."""
        wt_dispersion, wt_n_cells = self._ptb_scalars(ptb, "WT")
        phenotype_reference = PseudobulkExpressionPhenotype(
            expression_log2_ratio=union,
            dispersion=wt_dispersion,
            n_cells=wt_n_cells,
            measurement_type=MEASUREMENT_TYPE,
        )
        return PseudobulkExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C"
            ),
            environment_reference=self._environment(cond),
            phenotype_reference=phenotype_reference,
        )

    @staticmethod
    def _environment(cond: str) -> Environment:
        """Base YPD (control) or YPD + 0.4 M NaCl for 15 min (osmostress)."""
        media = Media(name="YPD", state="liquid", is_synthetic=False)
        temperature = Temperature(value=GROWTH_TEMP_C)
        if cond == "nacl":
            return Environment(
                media=media,
                temperature=temperature,
                perturbations=[
                    SmallMoleculePerturbation(
                        compound=resolved_compound("sodium chloride"),
                        concentration=Concentration(
                            value=NACL_MOLAR, unit=ConcentrationUnit.molar
                        ),
                    )
                ],
                aerobicity="aerobic",
                duration_hours=TREATMENT_HOURS,
            )
        return Environment(media=media, temperature=temperature, aerobicity="aerobic")

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[
        PseudobulkExpressionExperiment,
        PseudobulkExpressionExperimentReference,
        Publication,
    ]:
        """Build the pseudobulk experiment/reference/publication for one record."""
        genotype = Genotype(
            perturbations=[
                MarkerDeletionPerturbation(
                    systematic_gene_name=row["systematic_gene_name"],
                    perturbed_gene_name=row["perturbed_gene_name"],
                    marker=DELETION_MARKER,
                    strain_id=row["strain_id"],
                )
            ]
        )
        environment = self._environment(row["condition"])
        phenotype = PseudobulkExpressionPhenotype(
            expression_log2_ratio=row["expression_log2_ratio"],
            dispersion=row["dispersion"],
            n_cells=row["n_cells"],
            measurement_type=MEASUREMENT_TYPE,
        )
        experiment = PseudobulkExpressionExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = self._references[row["condition"]]
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/nadal_ribelles_perturbseq2025")
    dataset = NadalRibellesPerturbSeq2025Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    record = dataset[0]
    exp = record["experiment"]
    print("record[0] perturbations:", exp["genotype"]["perturbations"])
    print(
        "record[0] n phenotype genes:", len(exp["phenotype"]["expression_log2_ratio"])
    )
    print(
        "record[0] dispersion:",
        exp["phenotype"]["dispersion"],
        "n_cells:",
        exp["phenotype"]["n_cells"],
    )
    print("record[0] environment:", exp["environment"])


if __name__ == "__main__":
    main()

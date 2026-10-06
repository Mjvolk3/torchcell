# torchcell/datasets/scerevisiae/yoshida2012
# [[torchcell.datasets.scerevisiae.yoshida2012]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/yoshida2012
# Test file: tests/torchcell/datasets/scerevisiae/test_yoshida2012.py
"""Yoshida & Yokoyama 2012 organic-acid titers of yeast deletion mutants (Table 3).

Yoshida S & Yokoyama A 2012 (J Biosci Bioeng 113(5):556-561,
doi:10.1016/j.jbiosc.2011.12.017, PMID 22277779) screened the S. cerevisiae haploid
deletion collection on YPD-BCP plates for organic-acid overproducers, then quantified the
organic-acid titers of the 17 "multifunctional" gene deletions by HPLC. This loader
ingests **Table 3** ("Organic acid productivity of the deletion mutants"): one row per
strain (WT reference + 17 gene deletions), each giving OD plus six analyte titers in mM
(mean +/- SD of n=3 independent experiments; ``*`` in the source marks p<0.05 vs WT by
paired Student's t-test).

Maps to ``MetabolitePhenotype`` (WS15): ``metabolite_level = {analyte -> mM}`` for the
five ORGANIC ACIDS (acetate, citrate, malate, pyruvate, succinate) PLUS the inorganic
analyte phosphate (a measured analyte kept verbatim, but left OUT of
``target_metabolite_ids``). The ``OD`` column is BIOMASS (optical density), not a
metabolite, so it is dropped (retained only in the embedded literal for review).
``measurement_type = "hplc_organic_acid_titer_mM"``; ``metabolite_level_se = SD/sqrt(3)``;
``n_replicates = 3``. The schema stores uncertainty as a per-analyte SE only (no
uncertainty-TYPE field), so the source **sample SD** is converted to a standard error
(SD/sqrt(n)); the SD-origin is documented here. The five organic acids are mapped to
Yeast9 ``s_NNNN`` species ids via ``build_metabolite_s_id_map`` (KEGG-anchored, sourced
from ``YeastGEM``, never invented); phosphate is left unmapped.

Reference = the MEASURED **WT** (BY4742) row of Table 3 (mM +/- SD, n=3), restricted per
record to the analytes the strain measured (every strain measures all six here). This is
a measured WT baseline, not a population mean.

Background: the BY4741/BY4742 haploid deletion collection (KanMX4 deletions). The genotype
is one ``KanMxDeletionPerturbation`` per strain; the source uses common gene names, so an
injected ``SCerevisiaeGenome`` resolves them to systematic R64 ORF ids (Cachera pattern).

Provenance / extraction: the data IS Table 3 in the paper -- there is no external SI file.
The canonical source is the sha256-pinned library-mirror ``paper.pdf`` (verified by
``download()``). The MinerU OCR (``paper.md``) scrambled Table 3's numeric cells, so the
values were recovered from the BORN-DIGITAL text layer via
``pdftotext -layout paper.pdf`` (poppler); the recovered cells are transcribed into the
module-level ``TABLE_3`` literal so the build is deterministic and reviewable (the PDF is
never re-parsed at ``process()`` time).
"""

import logging
import math
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from typing import Any, cast

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    copy_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import YPD_LIQUID, restated
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
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
from torchcell.datasets.scerevisiae.zelezniak2018 import build_metabolite_s_id_map
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Growth medium (issue #622). Quotes are verbatim substrings of the mirror OCR
# ``paper.md`` (MinerU output, so LaTeX markup is kept as written); the line in each
# note is that file's.
# --------------------------------------------------------------------------- #
MEDIUM_CITATION_KEY = "yoshidaIdentificationCharacterizationGenes2012"
MEDIUM_PAPER_SHA256 = "e2c0b0b43dc8825504797efd52ff4bb2b13d96aa3fe3045eeae094888d06f657"
YPD_RECIPE_QUOTE = (
    "Strains were grown in YPD ( $1 \\%$ yeast extract, $2 \\%$ peptone, and $2 \\%$ "
    "glucose)"
)
YPD_FERMENTATION_QUOTE = (
    "in $3 0 0 \\mathrm { m l }$ of fresh YPD, and grown at $2 5 ^ { \\circ } "
    "\\mathbb { C }$ for 4 days without shaking."
)
TABLE_3_MEDIUM_QUOTE = (
    "Data show the results from 72-h incubation in YPD medium at "
    "$2 5 ^ { \\circ } \\mathbb { C } .$"
)


def _medium_sv(value: object, quote: str, note: str) -> SourcedValue:
    """A SourcedValue pinned to the Yoshida mirror OCR ``paper.md``."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=MEDIUM_CITATION_KEY,
            sha256=MEDIUM_PAPER_SHA256,
        ),
        quote=quote,
        note=note,
    )


MEDIUM_SOURCED_VALUES: dict[str, SourcedValue] = {
    "recipe": _medium_sv(
        "1% yeast extract, 2% peptone, 2% glucose",
        YPD_RECIPE_QUOTE,
        "line 25; the paper's own recipe, identical to the library YPD_LIQUID "
        "percentages, so the record carries YPD_LIQUID's components",
    ),
    "state": _medium_sv(
        "liquid, static",
        YPD_FERMENTATION_QUOTE,
        "line 39; the fermentation culture the HPLC supernatants come from",
    ),
    "table_3_medium": _medium_sv("YPD", TABLE_3_MEDIUM_QUOTE, "line 82; Table 3 note"),
}

YOSHIDA_YPD: Media = restated(YPD_LIQUID, *MEDIUM_SOURCED_VALUES.values())
"""Static liquid YPD at the paper's own 1/2/2% recipe: library ``YPD_LIQUID`` plus the
three Methods/Table 3 sentences. Same ``media_identity`` as ``YPD_LIQUID``."""

MEASUREMENT_TYPE = "hplc_organic_acid_titer_mM"
N_REPLICATES = 3
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

# Library mirror (canonical, sha256-pinned) holding the paper PDF the data is read from.
LIBRARY_CITATION_KEY = "yoshidaIdentificationCharacterizationGenes2012"
PDF_FILENAME = "paper.pdf"
PDF_SHA256 = "19df2d21ec52f333969468b916bfe468920e9d3974cd1f996feda21ea0d3d0dd"

# The five ORGANIC ACIDS mapped to Yeast9 s_NNNN (KEGG-anchored via YeastGEM). Phosphate
# is inorganic and intentionally omitted from the model-id mapping (kept as an analyte).
ACID_KEGG_IDS = {
    "acetate": "C00033",
    "citrate": "C00158",
    "malate": "C00149",
    "pyruvate": "C00022",
    "succinate": "C00042",
}
# Analytes stored as metabolite_level (five organic acids + inorganic phosphate). OD is
# excluded (it is biomass, not a metabolite).
ANALYTES = ["acetate", "citrate", "malate", "pyruvate", "succinate", "phosphate"]

# Column order of the embedded Table 3 literal: OD is retained for review but NOT stored.
_TABLE_3_COLUMNS = [
    "OD",
    "acetate",
    "citrate",
    "malate",
    "phosphate",
    "pyruvate",
    "succinate",
]

# Table 3 "Organic acid productivity of the deletion mutants" -- recovered from the
# born-digital text layer (`pdftotext -layout paper.pdf`) of the sha256-pinned mirror PDF.
# Each value is (mean_mM, sd_mM) of n=3 independent experiments. WT = BY4742 reference row.
# Column order = _TABLE_3_COLUMNS (OD, acetate, citrate, malate, phosphate, pyruvate,
# succinate). Source `*` significance marks (p<0.05 vs WT) are not stored.
TABLE_3: dict[str, list[tuple[float, float]]] = {
    "WT": [
        (4.01, 0.11),
        (4.21, 0.30),
        (0.10, 0.04),
        (0.15, 0.01),
        (0.69, 0.11),
        (0.18, 0.05),
        (1.29, 0.05),
    ],
    "ASM4": [
        (3.95, 0.02),
        (4.02, 0.30),
        (0.10, 0.03),
        (0.15, 0.02),
        (0.78, 0.07),
        (0.13, 0.04),
        (1.31, 0.06),
    ],
    "EMI5": [
        (4.17, 0.18),
        (6.11, 0.43),
        (0.14, 0.03),
        (0.15, 0.02),
        (0.66, 0.07),
        (0.28, 0.05),
        (1.34, 0.03),
    ],
    "GTR1": [
        (3.87, 0.05),
        (4.18, 0.24),
        (0.10, 0.04),
        (0.15, 0.02),
        (0.86, 0.03),
        (0.48, 0.09),
        (1.06, 0.05),
    ],
    "GTR2": [
        (3.85, 0.04),
        (4.29, 0.26),
        (0.10, 0.04),
        (0.15, 0.02),
        (0.83, 0.06),
        (0.43, 0.07),
        (1.07, 0.04),
    ],
    "LIP5": [
        (3.97, 0.08),
        (4.35, 0.24),
        (0.13, 0.04),
        (0.25, 0.02),
        (0.90, 0.06),
        (0.22, 0.09),
        (0.82, 0.03),
    ],
    "LSM1": [
        (3.66, 0.03),
        (5.45, 0.46),
        (0.12, 0.04),
        (0.18, 0.02),
        (1.26, 0.06),
        (0.94, 0.07),
        (0.83, 0.03),
    ],
    "MKS1": [
        (3.13, 0.03),
        (1.12, 0.13),
        (0.83, 0.05),
        (0.18, 0.01),
        (1.22, 0.05),
        (0.55, 0.08),
        (3.11, 0.11),
    ],
    "NFU1": [
        (3.91, 0.12),
        (3.99, 0.23),
        (0.13, 0.03),
        (0.15, 0.02),
        (0.75, 0.04),
        (0.15, 0.05),
        (1.28, 0.04),
    ],
    "PCK1": [
        (4.07, 0.19),
        (3.89, 0.24),
        (0.10, 0.03),
        (0.19, 0.01),
        (0.93, 0.03),
        (0.12, 0.04),
        (1.40, 0.07),
    ],
    "PHO85": [
        (2.89, 0.29),
        (7.20, 0.51),
        (0.17, 0.03),
        (0.17, 0.02),
        (0.61, 0.06),
        (0.26, 0.06),
        (1.09, 0.05),
    ],
    "PLM2": [
        (4.07, 0.13),
        (4.47, 0.39),
        (0.10, 0.04),
        (0.18, 0.02),
        (1.05, 0.04),
        (0.73, 0.07),
        (0.83, 0.03),
    ],
    "RTG1": [
        (4.25, 0.10),
        (5.41, 0.38),
        (0.07, 0.04),
        (0.14, 0.02),
        (0.53, 0.08),
        (0.35, 0.05),
        (1.06, 0.04),
    ],
    "RTG2": [
        (3.87, 0.05),
        (5.12, 0.36),
        (0.07, 0.04),
        (0.14, 0.02),
        (0.69, 0.07),
        (0.16, 0.05),
        (1.02, 0.05),
    ],
    "TIF3": [
        (2.66, 0.11),
        (6.98, 0.49),
        (0.12, 0.04),
        (0.14, 0.02),
        (0.71, 0.05),
        (0.23, 0.02),
        (1.30, 0.07),
    ],
    "UBA3": [
        (3.97, 0.18),
        (3.89, 0.19),
        (0.16, 0.03),
        (0.17, 0.02),
        (0.74, 0.05),
        (0.23, 0.05),
        (1.49, 0.07),
    ],
    "UBP3": [
        (3.36, 0.08),
        (4.65, 0.31),
        (0.09, 0.03),
        (0.15, 0.02),
        (1.20, 0.05),
        (0.21, 0.03),
        (1.29, 0.07),
    ],
    "YDR379C-A": [
        (3.81, 0.09),
        (4.12, 0.22),
        (0.11, 0.03),
        (0.15, 0.02),
        (0.74, 0.06),
        (0.14, 0.04),
        (1.40, 0.06),
    ],
}

_WT_KEY = "WT"


def _row_analytes(row: list[tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Map a Table 3 row to {analyte -> (mean, sd)} for the stored analytes (drop OD)."""
    by_col = dict(zip(_TABLE_3_COLUMNS, row, strict=True))
    return {a: by_col[a] for a in ANALYTES}


@register_dataset
class OrganicAcidYoshida2012Dataset(ExperimentDataset):
    """HPLC organic-acid titers of 17 yeast deletion mutants (Yoshida 2012, Table 3)."""

    def __init__(
        self,
        root: str = "data/torchcell/organic_acid_yoshida2012",
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
        """The sha256-pinned mirror PDF the Table 3 literal was transcribed from."""
        return [PDF_FILENAME]

    def download(self) -> None:
        """Stage the mirror ``paper.pdf`` into raw_dir and verify its sha256.

        The dataset values are the embedded ``TABLE_3`` literal, but the canonical source
        artifact (the paper PDF) is staged + hash-verified here so every built LMDB traces
        to an exact, hash-pinned raw file.
        """
        dest = osp.join(self.raw_dir, PDF_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        mirror = osp.join(
            os.environ["DATA_ROOT"],
            "torchcell-library",
            LIBRARY_CITATION_KEY,
            PDF_FILENAME,
        )
        if not osp.exists(mirror):
            raise RuntimeError(f"Yoshida2012 mirror PDF not found: {mirror}")
        copy_verified(mirror, dest, PDF_SHA256)
        log.info("Staged %s (%d bytes, sha256 verified)", dest, osp.getsize(dest))

    def _resolve_systematic(self, gene: str) -> str:
        """Resolve a common/systematic gene name to a systematic ORF id (must succeed).

        A systematic-shaped name must be a gene of the injected genome, and a common name
        must name exactly one gene; anything else is refused by name rather than stored.
        """
        gene = gene.strip().upper()
        genome = cast(SCerevisiaeGenome, self.genome)
        if _SYSTEMATIC_RE.match(gene):
            if gene not in genome.gene_set:
                raise RuntimeError(
                    f"Yoshida2012: systematic name '{gene}' is not a gene of the genome"
                )
            return gene
        candidates = genome.alias_to_systematic.get(gene, [])
        if not candidates:
            raise RuntimeError(f"Yoshida2012: could not resolve gene name '{gene}'")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Yoshida2012: gene name '{gene}' is an alias of {len(candidates)} "
                f"ORFs {candidates}"
            )
        return candidates[0]

    @post_process
    def process(self) -> None:
        """Build per-strain Metabolite experiments from the Table 3 literal, write LMDB.

        Every record is built and every refusal raised BEFORE ``data.csv`` or the store is
        written, so a refused table leaves nothing for a retry to serve. Refusals (each
        0 of the 17 strains of the released Table 3, 2026.10.01): an unresolvable or
        ambiguous name, two strains resolving to one ORF, and a NaN mean or SD.
        """
        verify_raw_files(self.raw_dir, {PDF_FILENAME: PDF_SHA256})
        if self.genome is None:
            raise RuntimeError(
                "OrganicAcidYoshida2012Dataset requires an injected SCerevisiaeGenome to "
                "resolve common gene names to systematic ORF ids (Table 3 uses names)."
            )
        # Organic-acid -> Yeast9 s_NNNN, sourced from YeastGEM (never invented). Phosphate
        # stays unmapped (inorganic; not a modelled organic-acid target).
        self._s_id_map = build_metabolite_s_id_map(ACID_KEGG_IDS)

        for gene, values in TABLE_3.items():
            for column, (mean, sd) in zip(_TABLE_3_COLUMNS, values, strict=True):
                if math.isnan(mean) or math.isnan(sd):
                    raise RuntimeError(
                        f"Yoshida2012: Table 3 row '{gene}' has a NaN {column} cell "
                        f"({mean}, {sd})"
                    )
        self._reference = _row_analytes(TABLE_3[_WT_KEY])

        rows: list[dict[str, Any]] = []
        gene_by_orf: dict[str, str] = {}
        for gene, values in TABLE_3.items():
            if gene == _WT_KEY:
                continue
            orf = self._resolve_systematic(gene)
            if orf in gene_by_orf:
                raise RuntimeError(
                    f"Yoshida2012: Table 3 rows '{gene_by_orf[orf]}' and '{gene}' both "
                    f"resolve to {orf}"
                )
            gene_by_orf[orf] = gene
            rows.append({"orf": orf, "gene": gene, "analytes": _row_analytes(values)})
        records = [self.create_experiment(record_row) for record_row in rows]
        log.info(
            "Yoshida2012: %d deletion strains, WT reference over %d analytes, "
            "%d organic acids mapped to Yeast9 s_NNNN",
            len(rows),
            len(self._reference),
            len(self._s_id_map),
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        pd.DataFrame([{"orf": r["orf"], "gene": r["gene"]} for r in rows]).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        with env.begin(write=True) as txn:
            for idx, (experiment, reference, publication) in enumerate(tqdm(records)):
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
        log.info("Wrote %d Yoshida2012 organic-acid experiments to LMDB", len(records))

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def _phenotype(
        self, analytes: dict[str, tuple[float, float]]
    ) -> MetabolitePhenotype:
        """Build a MetabolitePhenotype from {analyte -> (mean_mM, sd_mM)}.

        Source uncertainty is a **sample SD** (n=3); the schema stores only a per-analyte
        SE, so SD is converted to SE = SD / sqrt(n) (n=3). Only the five organic acids
        carry a Yeast9 s_NNNN target id (phosphate stays unmapped).
        """
        level = {a: mean for a, (mean, _sd) in analytes.items()}
        se = {a: sd / math.sqrt(N_REPLICATES) for a, (_mean, sd) in analytes.items()}
        n_reps = {a: N_REPLICATES for a in analytes}
        targets = {a: self._s_id_map[a] for a in analytes if a in self._s_id_map}
        return MetabolitePhenotype(
            metabolite_level=level,
            metabolite_level_se=se,
            n_replicates=n_reps,
            measurement_type=MEASUREMENT_TYPE,
            target_metabolite_ids=targets or None,
        )

    def create_experiment(  # type: ignore[override]
        self, row: dict[str, Any]
    ) -> tuple[MetaboliteExperiment, MetaboliteExperimentReference, Publication]:
        """Build the Metabolite experiment/reference/publication for one deletion strain."""
        # Background = BY4742/BY4741 haploid KanMX4 deletion collection.
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4742"
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["gene"]
                )
            ]
        )
        # Static (no shaking) YPD liquid, 25 C, 72 h (Methods; Table 3 caption).
        environment = Environment(media=YOSHIDA_YPD, temperature=Temperature(value=25))
        phenotype = self._phenotype(row["analytes"])
        # Reference = measured WT (BY4742) row, restricted to the analytes this strain
        # measured (all six here); a measured baseline, never a population mean.
        exp_keys = set(row["analytes"])
        ref_analytes = {k: v for k, v in self._reference.items() if k in exp_keys}
        phenotype_reference = self._phenotype(ref_analytes)
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
            pubmed_id="22277779",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/22277779/",
            doi="10.1016/j.jbiosc.2011.12.017",
            doi_url="https://doi.org/10.1016/j.jbiosc.2011.12.017",
        )
        return experiment, reference, publication


def main() -> None:
    """Build/load the dataset for interactive debugging.

    A genome is REQUIRED (Table 3 uses common gene names). Loads the existing LMDB if
    already built; delete ``<root>/processed`` first to re-run ``process()``.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/organic_acid_yoshida2012")
    dataset = OrganicAcidYoshida2012Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

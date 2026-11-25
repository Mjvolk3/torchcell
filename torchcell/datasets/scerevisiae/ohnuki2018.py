"""CalMorph morphology dataset for essential-gene het diploids (Ohnuki & Ohya 2018).

Ohnuki S & Ohya Y 2018, "High-dimensional single-cell phenotyping reveals extensive
haploinsufficiency", PLoS Biol 16(5):e2005130 (doi:10.1371/journal.pbio.2005130;
PMID 29768403). CalMorph high-dimensional morphology of a EUROSCARF collection of
HETEROZYGOUS diploid deletion strains, each dropping ONE essential gene from 2 -> 1
copies in the ``BY4743`` diploid wild type (a HIP-style heterozygous / 50%-dosage
deletion, NOT a full knockout). This is the diploid, essential-gene, dosage counterpart
of :class:`~torchcell.datasets.scerevisiae.ohya2005.ScmdOhya2005Dataset` (which is a
haploid ``BY4741`` full-deletion screen).

Scope: this loader builds ONLY the OPTIMAL (nutrient-rich) arm of the paper -- the 1,112
essential-gene heterozygotes phenotyped in liquid YPD at 25 C. The severe arm (50
heterozygotes in nutrient-poor SD medium at 37 C) has no raw CalMorph matrix released and
is ignored.

Growth conditions sourced verbatim from the Methods ("Yeast strains, media, and growth
conditions"): "The yeast diploid strain BY4743 was used as the wild type. Strains
heterozygous for 1,112 essential genes ... were cultured under optimal growth conditions
at 25 C in nutrient-rich yeast extract peptone dextrose (YPD) medium containing 1% (w/v)
Bacto yeast extract, 2% (w/v) Bacto peptone, and 2% (w/v) glucose". Phenotyped cells were
harvested from early-log-phase LIQUID YPD culture (media state ``"liquid"``). Heterozygous
gene-deletion mutants were purchased from EUROSCARF (kanMX4 marker); essential genes were
defined per the paper's ref [34].

Values are RAW per-strain CalMorph population averages (same semantics as Ohya 2005), one
501-length vector per strain: 281 base parameters (``CALMORPH_LABELS``: 220 mean + 61
ratio) + 220 coefficient-of-variation / noise parameters (``CALMORPH_STATISTICS``,
prefixed CCV/ACV/DCV).

ONE RECORD PER RELEASED SAMPLE (per-replicate policy, #1178 / #1171; 2026.11.24). The
store holds the 1,112 heterozygote rows (indices 0..1111, unchanged) and then the 114
wild-type rows of ``wt114data.tsv`` (indices 1112..1225), 1,226 records:

- A heterozygote row is ONE culture's CalMorph population average: "Three colonies from
  each strain were inoculated into 2 mL of YPD liquid medium" (paper.md line 181, one
  culture per strain) and "analyzed more than 200 cells for each strain" (line 57).
  Heterozygote records carry ``replicate_id = None`` and are byte-identical to the
  per-strain build.
- Each wild-type row is ONE wild-type replicate: "The yeast diploid strain BY4743 was
  used as the wild type" (line 179); the null distribution is "estimated from 114
  replicated wild-type strains" (line 191); Fig. 1B's "114 wild-type replicates" (line
  68); the SCMD2 datasheet page lists the file as "114 replicated wild-type (BY4743)"
  (``data/summary_ess1112het.html`` line 42). Each is its own record with an EMPTY
  genotype against the ``BY4743`` reference genome and ``replicate_id`` = the row's
  released ``NAME`` verbatim (``WBD001N`` ... ``WBD138S``, 0 duplicates). The paper does
  not use the word "biological" for them; each replicate is one pass of the per-strain
  culture protocol of line 181.
- ``phenotype_reference`` stays the per-feature mean over the 114 wild-type rows, a
  summary with ``replicate_id = None``; the 114 samples it summarizes are records too.

PROVENANCE / SOURCING (sha256-pinned; loader reads the library-mirror ``data/`` files and
verifies both hashes, never the live URL):
- ``ess1112data.tsv`` (mutant matrix, 1,112 strains x 501 features; ID column ``ORF``)
  sha256 ``2d168bd1c436c7edae0ab3eb07e99e4c41f3b12f69607a7f3a00092eed7c4b03``.
- ``wt114data.tsv`` (114 WT replicate averages, one record each; ID column ``NAME``;
  feature columns byte-identical to the mutant file) sha256
  ``f48d42da2c727854b83b70e8768cfb42ada0e5118f6074765698d3045862804e``.
- Historical retrieval URLs (SCMD2 portal, per the paper's Data Availability statement):
  ``http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=ess1112data.tsv`` and
  ``...=wt114data.tsv``.

ORFs are already SGD systematic names and are stored AS RELEASED (stripped and
upper-cased); no genome is consulted (#1045). A blank CalMorph cell is key-absent in the
record, never written as 0.0 (the previous build wrote 0.0; the pinned matrix has 0 blank
cells, so no stored value changes).
"""

# torchcell/datasets/scerevisiae/ohnuki2018
# [[torchcell.datasets.scerevisiae.ohnuki2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/ohnuki2018
# Test file: tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py

import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable
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
from torchcell.datamodels.media import YPD_LIQUID, restated
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    EngineeredCopyNumberPerturbation,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Growth medium (issue #622). Quotes are verbatim substrings of the mirror OCR
# ``paper.md`` (MinerU output, so LaTeX markup is kept as written); the line in each
# note is that file's.
# --------------------------------------------------------------------------- #
MEDIUM_CITATION_KEY = "ohnukiHighdimensionalSinglecellPhenotyping2018"
MEDIUM_PAPER_SHA256 = "8c9d991fa03278b130d8824b755b9e1b14921a1e431eb8ed918caabf6a11ab1c"
YPD_RECIPE_QUOTE = (
    "in nutrient-rich yeast extract peptone dextrose (YPD) medium containing "
    "$1 \\%$ (w/v) Bacto yeast extract (BD Biosciences, San Jose, CA), $2 \\%$ (w/v) "
    "Bacto peptone (BD Biosciences), and $2 \\%$ (w/v) glucose, which was prepared as "
    "described previously [15]."
)
YPD_LIQUID_QUOTE = (
    "Three colonies from each strain were inoculated into $2 \\mathrm { m L }$ of YPD "
    "liquid medium"
)

MEDIUM_SOURCED_VALUES: dict[str, SourcedValue] = {
    "recipe": SourcedValue(
        value="1% (w/v) yeast extract, 2% (w/v) peptone, 2% (w/v) glucose",
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=MEDIUM_CITATION_KEY,
            sha256=MEDIUM_PAPER_SHA256,
        ),
        quote=YPD_RECIPE_QUOTE,
        note="line 179; the paper's own recipe, identical to the library YPD_LIQUID "
        "percentages, so the record carries YPD_LIQUID's components; ref 15 is Ohya "
        "2005 (line 339)",
    ),
    "state": SourcedValue(
        value="liquid",
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=MEDIUM_CITATION_KEY,
            sha256=MEDIUM_PAPER_SHA256,
        ),
        quote=YPD_LIQUID_QUOTE,
        note="line 181",
    ),
}

OHNUKI_YPD: Media = restated(YPD_LIQUID, *MEDIUM_SOURCED_VALUES.values())
"""Liquid YPD at the paper's own 1/2/2% recipe: library ``YPD_LIQUID`` plus the two
Methods sentences. Same ``media_identity`` as ``YPD_LIQUID``."""

# CV parameters are prefixed CCV/ACV/DCV (60 + 33 + 127 = 220 in the pinned matrices; no
# TCV parameter exists, #494); everything else is a base parameter.
_CV_PREFIXES = ("CCV", "ACV", "DCV")

# sha256-pinned raw matrices in the library mirror ``data/`` directory.
_RAW_FILES: dict[str, dict[str, str]] = {
    "ess1112data.tsv": {
        "sha256": "2d168bd1c436c7edae0ab3eb07e99e4c41f3b12f69607a7f3a00092eed7c4b03",
        "id_column": "ORF",
    },
    "wt114data.tsv": {
        "sha256": "f48d42da2c727854b83b70e8768cfb42ada0e5118f6074765698d3045862804e",
        "id_column": "NAME",
    },
}

# Library mirror holding the sha256-pinned raw matrices (relative to DATA_ROOT).
_MIRROR_DIR = "torchcell-library/ohnukiHighdimensionalSinglecellPhenotyping2018/data"


@register_dataset
class ScmdOhnuki2018Dataset(ExperimentDataset):
    """CalMorph morphology of essential-gene heterozygous diploid deletions (Ohnuki 2018).

    Diploid ``BY4743`` HIP-style heterozygous (50%-dosage) deletions of 1,112 essential
    genes, phenotyped in liquid YPD at 25 C; the diploid essential-gene counterpart of the
    haploid Ohya 2005 full-deletion CalMorph screen.
    """

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/scmd_ohnuki2018",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Return the experiment schema class for this dataset."""
        return CalMorphExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Return the experiment-reference schema class for this dataset."""
        return CalMorphExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw mutant and wildtype TSV filenames."""
        return list(_RAW_FILES)

    def download(self) -> None:
        """Copy both matrices from the sha256-pinned library mirror and verify hashes."""
        data_root = os.environ["DATA_ROOT"]
        mirror = osp.join(data_root, _MIRROR_DIR)
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, spec in _RAW_FILES.items():
            dest = osp.join(self.raw_dir, filename)
            if not osp.exists(dest):
                src = osp.join(mirror, filename)
                if not osp.exists(src):
                    raise RuntimeError(
                        f"{filename} not found in the library mirror {mirror}. The SCMD2 "
                        "portal is the historical source; recover the file and deposit it, "
                        "then rebuild (sha256 verified)."
                    )
                copy_verified(src, dest, spec["sha256"])

    @post_process
    def process(self) -> None:
        """Load raw TSVs, build CalMorph experiments, and write the LMDB store."""
        verify_raw_files(
            self.raw_dir, {name: spec["sha256"] for name, spec in _RAW_FILES.items()}
        )
        df_mutant = pd.read_csv(osp.join(self.raw_dir, "ess1112data.tsv"), sep="\t")
        df_wt = pd.read_csv(osp.join(self.raw_dir, "wt114data.tsv"), sep="\t")

        df = self.preprocess_calmorph_data(df_mutant)
        df_wt = self.preprocess_wt_data(df_wt)

        # The reference is a summary: the per-feature mean over the 114 WT replicates,
        # each of which is also a record (written after the heterozygotes, below).
        self.wt_reference_phenotype = self._calculate_wt_reference(df_wt)

        log.info("Processing Ohnuki 2018 CalMorph morphology data...")

        # Every record is built (and so every refusal raised) before ``data.csv`` or the
        # store is written, so a refused matrix leaves neither behind for a retry.
        # Heterozygote records first (order and indices unchanged from the per-strain
        # build), then one record per released wild-type replicate.
        records = [
            self.create_calmorph_experiment(
                self.name, row, wt_reference_phenotype=self.wt_reference_phenotype
            )
            for _, row in tqdm(df.iterrows(), total=df.shape[0])
        ] + [
            self.create_wt_calmorph_experiment(
                self.name, row, wt_reference_phenotype=self.wt_reference_phenotype
            )
            for _, row in df_wt.iterrows()
        ]
        os.makedirs(self.preprocess_dir, exist_ok=True)
        df.to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        with env.begin(write=True) as txn:
            for index, (experiment, reference, publication) in enumerate(records):
                serialized_data = pickle.dumps(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
                txn.put(f"{index}".encode(), serialized_data)

        env.close()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocess raw data - for CalMorph this is handled in process()."""
        return df

    def preprocess_calmorph_data(self, df_mutant: pd.DataFrame) -> pd.DataFrame:
        """Clean the mutant matrix and name every strain as released (no genome)."""
        df_mutant = df_mutant.copy()
        # The ORF column carries the systematic gene name; there is no common-name column.
        df_mutant["systematic_gene_name"] = df_mutant["ORF"].str.strip().str.upper()
        # A row with a blank ORF names no strain; it is dropped and counted (0 of the
        # 1112 rows of the pinned matrix, 2026.10.01).
        blank = df_mutant["systematic_gene_name"].isna() | (
            df_mutant["systematic_gene_name"] == ""
        )
        n_blank = int(blank.sum())
        if n_blank:
            log.warning(
                "Ohnuki 2018: dropping %d mutant row(s) with a blank ORF", n_blank
            )
        df_mutant = df_mutant[~blank].reset_index(drop=True)
        duplicated = df_mutant["systematic_gene_name"].duplicated(keep=False)
        if duplicated.any():
            raise RuntimeError(
                "Ohnuki 2018: the same strain is listed more than once (ORF stripped "
                "and uppercased): "
                f"{df_mutant.loc[duplicated, 'ORF'].tolist()}"
            )

        df_mutant["perturbed_gene_name"] = df_mutant["systematic_gene_name"]

        log.info(
            "Ohnuki 2018: %d essential-gene heterozygote strains (0 dropped for naming)",
            len(df_mutant),
        )
        return df_mutant

    def preprocess_wt_data(self, df_wt: pd.DataFrame) -> pd.DataFrame:
        """Check the WT matrix names every replicate once; each row becomes one record.

        ``NAME`` is the replicate's released label and its ``replicate_id``, so a blank
        or repeated ``NAME`` is refused (0 blank, 0 repeated among the 114 pinned rows).
        """
        names = df_wt["NAME"].astype("string").str.strip()
        blank = names.isna() | (names == "")
        if blank.any():
            raise RuntimeError(
                f"Ohnuki 2018: {int(blank.sum())} wild-type row(s) with a blank NAME; "
                "the NAME is the replicate's replicate_id"
            )
        repeated = names[names.duplicated(keep=False)].tolist()
        if repeated:
            raise RuntimeError(
                f"Ohnuki 2018: wild-type replicate NAME(s) listed more than once: "
                f"{repeated}"
            )
        log.info("Ohnuki 2018: %d wild-type replicates, one record each", len(df_wt))
        return df_wt.reset_index(drop=True)

    def _calculate_wt_reference(self, df_wt: pd.DataFrame) -> dict[str, Any]:
        """Summarize the WT replicates as a per-feature mean (blank cells skipped)."""
        info_columns = [c for c in ("NAME", "ORF") if c in df_wt.columns]
        morphology_columns = [c for c in df_wt.columns if c not in info_columns]
        wt_means: dict[str, Any] = {}
        for col in morphology_columns:
            # A non-numeric cell refuses here, as it does in the replicate's own record.
            numeric_values = pd.to_numeric(df_wt[col], errors="raise")
            wt_means[col] = numeric_values.mean()
        return wt_means

    def create_experiment(self) -> None:
        """Not used: records are built by ``create_calmorph_experiment``."""
        raise NotImplementedError(
            "ScmdOhnuki2018Dataset builds records with create_calmorph_experiment"
        )

    @staticmethod
    def _split_traits(
        row: pd.Series, info_columns: set[str], replicate_id: str | None
    ) -> CalMorphPhenotype:
        """Split the 501-length vector into 281 base + 220 CV parameters.

        A blank cell is key-absent, never 0.0. ``replicate_id`` is the released label
        of a wild-type row and None on a heterozygote row (one culture per strain).
        """
        base_measurements: dict[str, float] = {}
        cv_measurements: dict[str, float] = {}
        for col in row.index:
            if col in info_columns:
                continue
            value = row[col]
            if pd.isna(value):
                continue  # a blank cell is key-absent, never 0.0
            float_value = float(value)
            if col.startswith(_CV_PREFIXES):
                cv_measurements[col] = float_value
            else:
                base_measurements[col] = float_value
        return CalMorphPhenotype(
            calmorph=base_measurements,
            calmorph_coefficient_of_variation=(
                cv_measurements if cv_measurements else None
            ),
            replicate_id=replicate_id,
        )

    @staticmethod
    def _build_record(
        dataset_name: str,
        genotype: Genotype,
        phenotype: CalMorphPhenotype,
        wt_reference_phenotype: dict[str, Any],
    ) -> tuple[CalMorphExperiment, CalMorphExperimentReference, Publication]:
        """Pair one experiment with the shared WT-summary reference and publication."""
        # Reference genome - diploid BY4743 wild type (Methods).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4743",
            ploidy="diploid",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )

        # Optimal growth conditions: nutrient-rich liquid YPD at 25 C (Methods).
        environment = Environment(media=OHNUKI_YPD, temperature=Temperature(value=25))
        environment_reference = environment.model_copy()

        # Reference phenotype from the WT summary, split the same way; it states an
        # aggregate over the WT replicates, so it carries no replicate_id.
        wt_base = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if not k.startswith(_CV_PREFIXES)
        }
        wt_cv = {
            k: v
            for k, v in wt_reference_phenotype.items()
            if k.startswith(_CV_PREFIXES)
        }
        phenotype_reference = CalMorphPhenotype(
            calmorph=wt_base, calmorph_coefficient_of_variation=wt_cv if wt_cv else None
        )

        reference = CalMorphExperimentReference(
            dataset_name=dataset_name,
            genome_reference=genome_reference,
            environment_reference=environment_reference,
            phenotype_reference=phenotype_reference,
        )
        experiment = CalMorphExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        publication = Publication(
            pubmed_id="29768403",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29768403/",
            doi="10.1371/journal.pbio.2005130",
            doi_url="https://doi.org/10.1371/journal.pbio.2005130",
        )
        return experiment, reference, publication

    @staticmethod
    def create_calmorph_experiment(
        dataset_name: str, row: pd.Series, wt_reference_phenotype: dict[str, Any]
    ) -> tuple[CalMorphExperiment, CalMorphExperimentReference, Publication]:
        """Build the record of one heterozygote row (one culture; ``replicate_id`` None)."""
        # HIP-style heterozygous deletion: one essential-gene copy dropped 2 -> 1 in the
        # diploid (50% dosage), kanMX4 marker (EUROSCARF). NOT a full knockout.
        genotype = Genotype(
            perturbations=[
                EngineeredCopyNumberPerturbation(
                    systematic_gene_name=row["systematic_gene_name"],
                    perturbed_gene_name=row["perturbed_gene_name"],
                    copy_number=1,
                    reference_copy_number=2,
                    marker="KanMX",
                )
            ]
        )
        phenotype = ScmdOhnuki2018Dataset._split_traits(
            row, {"ORF", "systematic_gene_name", "perturbed_gene_name"}, None
        )
        return ScmdOhnuki2018Dataset._build_record(
            dataset_name, genotype, phenotype, wt_reference_phenotype
        )

    @staticmethod
    def create_wt_calmorph_experiment(
        dataset_name: str, row: pd.Series, wt_reference_phenotype: dict[str, Any]
    ) -> tuple[CalMorphExperiment, CalMorphExperimentReference, Publication]:
        """Build the record of one released BY4743 wild-type replicate.

        The genotype is empty (the record's strain is the ``BY4743`` reference genome)
        and ``replicate_id`` is the row's released ``NAME`` verbatim.
        """
        phenotype = ScmdOhnuki2018Dataset._split_traits(row, {"NAME"}, str(row["NAME"]))
        return ScmdOhnuki2018Dataset._build_record(
            dataset_name, Genotype(perturbations=[]), phenotype, wt_reference_phenotype
        )


if __name__ == "__main__":
    pass

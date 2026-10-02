# torchcell/datasets/scerevisiae/cachera2023
# [[torchcell.datasets.scerevisiae.cachera2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/cachera2023
# Test file: tests/torchcell/datasets/scerevisiae/test_cachera2023.py
"""Cachera 2023 CRI-SPA betaxanthin dataset (genome-wide product-proxy screen).

Cachera et al. 2023 (Nucleic Acids Research, doi:10.1093/nar/gkad656) used CRI-SPA to
transfer four betaxanthin-biosynthesis genes into each strain of the ~4800-strain yeast
knockout (YKO) collection, then read out per-colony **betaxanthin** (a yellow plant
metabolite) by image analysis. The "CRI-SPA score" is a corrected/normalized colony
YELLOWNESS score, the geometric mean of Value and Saturation of the colony's pixels in
HSV color space (paper line 58), not a fluorescence measurement -- a quantitative proxy
for betaxanthin level (it can be negative because it is population-centered).

Environment (issue #509): the final screen plate is solid YPD with 200 mg/l G418
(``CACHERA_YPD_G418``, every value quoted from the mirror OCR), and the paper states no
growth temperature, so ``temperature=None`` with a typed ``ProvenanceGap``.

Source: the paper's Data Availability points to the CRI-SPA GitHub repo
(github.com/pc2912/CRI-SPA_repo). We ingest `GA1_2_4_6.csv` -- the gene-level
corrected+filtered dataset combining screen replicates 1/2/4/6 -- using the 24 h
`corrected_mean_intensity` (mean/std/count) as the betaxanthin level + SE + n.

Maps to `MetabolitePhenotype` (WS4): `metabolite_level = {"betaxanthin": score}` with
`measurement_type = "cri_spa_corrected_hsv_yellowness_24h"`. Gene names in the
source are COMMON names, so a genome is required to resolve them to systematic ORF ids
(same pattern as Sameith); unresolved names + control/NaN rows are excluded and logged.
The varying deletion stores the genome's own standard name as `perturbed_gene_name`
(`canonical_common_names`, shared with the Smith/Mormino/Lian loaders), so one gene keeps
one spelling across datasets; an ORF with no round-tripping standard name stores the id.
"""

import logging
import math
import os
import os.path as osp
import pickle
import urllib.request
from collections.abc import Callable
from typing import Any, cast

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD
from torchcell.datamodels.schema import (
    Concentration,
    ConcentrationUnit,
    Environment,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MediaComponent,
    MediaComponentRole,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.smith2006 import canonical_common_names
from torchcell.sequence.genome.scerevisiae import GeneNameStatus, SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TARGET_METABOLITE = "betaxanthin"

# --------------------------------------------------------------------------- #
# Paper anchors (issue #509). Every quote below is a verbatim substring of the mirror
# OCR (MinerU output, so LaTeX markup is kept as written); the line is that file's.
# --------------------------------------------------------------------------- #
CITATION_KEY = "cacheraCRISPAHighthroughputMethod2023"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "5fb7310dacb44bc04fc121a990e4ccc70020c9598877c4b074656e91fc4f32d5"
_PAPER = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)

# line 58: the readout is a color score from image pixels, not a fluorescence intensity.
READOUT_QUOTE = (
    "filter taking the geometric mean of Value and Saturation of image pixels in the "
    "HSV color space"
)
# line 58: the color score was calibrated on YPD.
READOUT_MEDIUM_QUOTE = "which was our observation on YPD media"
# line 117: the plate the screen colonies grew on.
SCREEN_PLATE_QUOTE = (
    "we picked cells from each of the four colonies obtained for each strain on the "
    "final CRI-SPA screen plate (YPD-G418)"
)
# line 31: YPD recipe deferral, agar and G418 doses on solid medium.
YPD_RECIPE_QUOTE = (
    "YPD and synthetic complete (SC) media, and SC drop out media were prepared as "
    "described by Sherman et al. (28)"
)
AGAR_QUOTE = "For growth on solid medium, $2 0 \\ \\mathrm { g / l }$ agar was added."
G418_QUOTE = (
    "For selection on solid medium, plates were supplemented with "
    "$2 0 0 ~ \\mathrm { { m g / l } }$ geneticin (G418)"
)

#: What the stored level IS: the 24 h ``corrected_mean_intensity`` column of
#: ``GA1_2_4_6.csv``, an HSV yellowness score (``READOUT_QUOTE``), corrected for plate
#: position. The word "intensity" in the column name is the image Value/Saturation
#: intensity, not fluorescence.
MEASUREMENT_TYPE = "cri_spa_corrected_hsv_yellowness_24h"


def _paper_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the Cachera mirror OCR (quote + sha256)."""
    return SourcedValue(value=value, provenance=_PAPER, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "measurement_type": _paper_sv(
        MEASUREMENT_TYPE,
        READOUT_QUOTE,
        note="line 58; a colony-color score, so the name says HSV yellowness",
    ),
    "screen_medium": _paper_sv("solid YPD + G418", SCREEN_PLATE_QUOTE, note="line 117"),
    "readout_medium": _paper_sv("YPD", READOUT_MEDIUM_QUOTE, note="line 58"),
    "ypd_recipe": _paper_sv(
        "YPD per Sherman et al. (ref 28)",
        YPD_RECIPE_QUOTE,
        note="line 31; Sherman et al. is not mirrored, so the YPD ingredient amounts "
        "are the library YPD root's (Tong and Boone 2006 YEPD), which this sentence "
        "does not restate",
    ),
    "agar": _paper_sv("20 g/l", AGAR_QUOTE, note="line 31"),
    "g418": _paper_sv("200 mg/l", G418_QUOTE, note="line 31"),
}

CACHERA_YPD_G418: Media = Media(
    name="YPD + G418 (solid, 2% agar; Cachera 2023 final CRI-SPA screen plate)",
    state="solid",
    is_synthetic=False,
    base_medium="YPD",
    components=[
        *YPD.components,
        MediaComponent(
            compound=resolved_compound("agar"),
            role=MediaComponentRole.gelling_agent,
            concentration=Concentration(value=2.0, unit=ConcentrationUnit.percent_w_v),
            provenance=[SOURCED_VALUES["agar"]],
        ),
        MediaComponent(
            compound=resolved_compound("G418 (geneticin)"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(value=200.0, unit=ConcentrationUnit.ug_per_ml),
            provenance=[SOURCED_VALUES["g418"]],
            note="selects the kanMX marker of the YKO deletion (CRI-SPA step 5)",
        ),
    ],
    provenance=[
        SOURCED_VALUES["screen_medium"],
        SOURCED_VALUES["ypd_recipe"],
        SOURCED_VALUES["readout_medium"],
    ],
)
"""The final CRI-SPA screen plate: library ``YPD`` root components plus the paper's own
agar and G418 doses. Loader-local on purpose (as ``COOPER_SC`` is): ``media.py`` is a
value surface of the served graph, so editing it would touch every YPD dataset."""

#: No growth temperature in the paper. Measured by
#: ``experiments/036-dataset-fixes-before-kg-build/scripts/cachera2023_environment_readout.py``
#: (2026.10.02, issue #509): the mirror ``paper.md`` has 0 matches for a degree sign,
#: ``\circ``, "temperature" or a two-digit number followed by C, and the SI archive's
#: four documents (Supplementary Material.docx, Supp Methods S1/S2, Figure S8) have 3,
#: all SGD gene descriptions ("required for growth at low temperature") in a hit table.
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER,
    note="no growth or incubation temperature is stated in the paper or its SI "
    "(gkad656_supplemental_files.zip); the loader formerly stored an unsourced 30 C",
)


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
# Pinned sha256 of GA1_2_4_6.csv (role si_data in the library manifest:
# torchcell-library/cacheraCRISPAHighthroughputMethod2023/manifest.json). The stored
# artifact + this hash is canonical, NOT the live GitHub URL; verified on download so
# upstream drift is detected rather than silently followed.
DATA_SHA256 = "71f55609067301e1430a1ab3226618c46428488753c6412603e8f0c21bbcac8a"
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
        """Download the gene-level CRI-SPA dataset from the authors' GitHub repo.

        The stored artifact's sha256 is canonical: a freshly downloaded payload is
        verified before it is written, and the file in ``raw/`` (present or fresh) is
        verified against ``DATA_SHA256`` at the start of ``process``. A mismatch means
        upstream drift or corruption and raises (never silently followed).
        """
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
        write_verified(data, dest, DATA_SHA256, DATA_URL)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    def _resolve_systematic(self, gene: str) -> str | None:
        """Resolve a source gene name (common or systematic) to an R64 identifier.

        Uses the genome's layered resolver. Returns the current R64 identifier for a live
        gene, an SGD rename, or a valid non-"gene" feature (e.g. a ``blocked_reading_frame``
        pseudogene such as ``YER109C``/FLO8 or ``YFL056C``/AAD6 -- REAL loci that are
        retained, not dropped). Returns None only when the name is not a recognisable R64
        feature (a control like ``WT`` or a malformed id), which the caller drops.
        """
        genome = cast(SCerevisiaeGenome, self.genome)
        res = genome.resolve_gene_name(gene)
        if (
            res.status
            in (
                GeneNameStatus.CURRENT,
                GeneNameStatus.RENAMED,
                GeneNameStatus.NON_GENE_FEATURE,
            )
            and res.systematic_name is not None
        ):
            return res.systematic_name
        return None

    @post_process
    def process(self) -> None:
        """Parse the CRI-SPA dataset into per-ORF Metabolite experiments and write LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        if self.genome is None:
            raise RuntimeError(
                "Cachera2023 requires an injected SCerevisiaeGenome to resolve common "
                "gene names to systematic ORF ids (source uses common names)."
            )
        df = pd.read_csv(osp.join(self.raw_dir, DATA_FILENAME))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        canonical = canonical_common_names(self.genome)
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
            # `common` is the genome's own standard name for the resolved ORF, so
            # `perturbed_gene_name` carries the same spelling this gene gets in every
            # other loader (the L1 `canonical_gene_names` rule), with the systematic id
            # kept for the 789 ORFs that have no round-tripping standard name. Taking it
            # from the genome rather than from the source column is what normalizes the
            # 408 rows whose 2023-era spelling has since been superseded (ACN9 -> SDH7,
            # AIM1 -> BOL3). Storing `systematic` twice, as this loader did before issue
            # #195, threw a name the source supplied for 3,930 of 4,719 records.
            rows.append(
                {
                    "orf": systematic,
                    "common": canonical.get(systematic, systematic),
                    "level": float(level),
                    "se": se,
                    "n": max(count, 1),
                }
            )
        # Unresolved names are dropped by design (never guessed into an ORF): 'WT' is a
        # control and 'YLR287-A' is a malformed id. Names that resolve to a valid non-"gene"
        # R64 feature (AAD6/CRS5/FLO8 -> YFL056C/YOR031W/YER109C, blocked_reading_frame
        # pseudogenes) are now RETAINED via the genome resolver rather than dropped -- they
        # are real loci with a real perturbation. Log the full unresolved list (only a
        # handful) so the drop stays auditable.
        log.info(
            "Cachera: %d usable ORFs, %d control/NaN rows, %d unresolved names dropped %s, "
            "%d ORF collisions deduped",
            len(rows),
            n_control_or_nan,
            len(unresolved),
            sorted(unresolved),
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
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["common"]
                ),
                *_betaxanthin_cassette(),
            ]
        )
        # Issue #509: YPD + G418 screen plate (line 117), no stated temperature.
        environment = Environment(
            media=CACHERA_YPD_G418, temperature=None, provenance_gaps=[TEMPERATURE_GAP]
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

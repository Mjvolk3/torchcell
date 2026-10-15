# torchcell/datasets/scerevisiae/ozaydin2013
# [[torchcell.datasets.scerevisiae.ozaydin2013]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/ozaydin2013
# Test file: tests/torchcell/datasets/scerevisiae/test_ozaydin2013.py
"""Ozaydin 2013 carotenoid visual-screen dataset (beta-carotene colony color).

Ozaydin, Bao et al. 2013 (Metabolic Engineering, doi:10.1016/j.ymben.2012.07.010)
transformed the haploid yeast deletion collection (BY4741 background) with the
carotenogenic plasmid **YB/I/BTS1** (`YEplac195 TDH3p-crtYB-CYC1t; TDH3p-crtI-CYC1t;
TDH3p-BTS1-CYC1t`, from Verwaal et al. 2007 -- crtYB + crtI from *Xanthophyllomyces
dendrorhous* plus an extra copy of the NATIVE GGPP synthase BTS1, NOT crtE). Every
deletion strain then makes beta-carotene, and **colony color is a visual proxy for
carotenoid flux** on a -5..+5 scale (WT carrying YB/I/BTS1 = 0; more orange = more
carotenoid). This is the heterologous background a downstream constraint-based model
would add on top of Yeast9; the base strain + deletion + score are captured here.

Source: paper SI `1-s2.0-S109671761200081X-mmc1.xlsx` (Elsevier ESM). Sheet 1
"Color scores of all deletions" is the whole-collection screen; Sheet 2 "Names and
Functions of TOP200" is the curated high-hit subset (gene name/function/category).

Maps to `VisualScorePhenotype` (WS4). One record per RELEASED ROW with a numeric color
(2026.10.12, #1045): an ORF scored on two plates is two records of one deletion, told
apart by their values, with ``n_replicates = 1`` each (the previous build collapsed them
to the maximum: 132 rows into 130 ORFs). The ORF is stored as released (upper-cased,
no genome consulted); the strain is the row's own ``Strain`` cell. A row whose Color cell
is text (``pet``, ``tiny``, ``_``; 546 rows) carries no number the ``visual_score`` field
can hold and is listed, with its text, in ``preprocess/dropped_records.json``; a row whose
ORF cell is not ORF-shaped (``YLR287-A``, 1 row) is counted there under
``deferred_to_released_gene_name_leaf`` until that leaf lands.
"""

import json
import logging
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Localization,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    ReferenceGenome,
    ReleasedGeneKind,
    ReleasedGeneNamePerturbation,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The visual color scale (Sheet 1). WT carrying YB/I/BTS1 = 0.
SCORE_SCALE_MIN = -5
SCORE_SCALE_MAX = 5
SCORE_SEMANTICS = (
    "colony color -5..+5 vs WT (harboring carotenogenic plasmid YB/I/BTS1): "
    "higher = more orange/red colony = more carotenoid (beta-carotene) accumulation"
)
TARGET_PRODUCT = "beta-carotene"

# Pinned sha256 of the SI spreadsheet (role si_data in the library manifest:
# torchcell-library/ozaydinCarotenoidbasedPhenotypicScreen2013a/manifest.json).
# The stored artifact + this hash is canonical, NOT the live URL; we verify on
# download and refuse to silently follow upstream drift.
_SI_SHA256 = "4818726e352ead3cb739fd9becf08a0c04d14b8a8761732184214344447507f0"

# --------------------------------------------------------------------------- #
# Screen medium (issue #622). Quotes are verbatim substrings of the mirror OCR
# (MinerU, LaTeX markup kept); line numbers are that file's.
# --------------------------------------------------------------------------- #
CITATION_KEY = "ozaydinCarotenoidbasedPhenotypicScreen2013a"
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "df2f2b639331f3da4137ab61e2df2ba0899b164a85da48acafc30c55dcc699b8"
_PAPER = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)

# line 42: the transformants were selected on SC-URA agar ...
SELECTION_PLATE_QUOTE = "spotted onto selective medium agar plates (SC-URA)"
# line 42: ... and the carotenoid color was scored on SC-URA plates after 2 days at 30 C.
SCORING_PLATE_QUOTE = (
    "Next day, about $5 \\mu \\mathrm { l }$ of this culture was spotted onto SC-URA "
    "plates. After 2 day of growth at $3 0 ^ { \\circ } \\mathrm { C } ,$ , carotenoid "
    "levels were scored."
)


def _paper_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the Ozaydin mirror OCR (quote + sha256)."""
    return SourcedValue(value=value, provenance=_PAPER, quote=quote, note=note)


SOURCED_VALUES: dict[str, SourcedValue] = {
    "scoring_medium": _paper_sv(
        "SC-URA, solid (plates)",
        SCORING_PLATE_QUOTE,
        note="line 42; the paper prints no SC or SC-URA recipe, so every ingredient "
        "amount is the library SC_URA object's (SC with uracil dropped, Mormino 2022 "
        "/ Wildenhain 2015 glucose), a deferral, not an Ozaydin statement",
    ),
    "agar": _paper_sv(
        "agar present, amount not stated",
        SELECTION_PLATE_QUOTE,
        note="line 42; the paper says agar plates and prints no agar amount, so the "
        "agar row carries no concentration (an open gap in Media.open_gaps) rather "
        "than a borrowed 2%",
    ),
}

OZAYDIN_SC_URA_AGAR: Media = SC_URA.model_copy(
    update={
        "name": "SC-Ura agar (synthetic complete minus uracil, solid; Ozaydin 2013 "
        "screen plate)",
        "state": "solid",
        "components": [
            *SC_URA.components,
            MediaComponent(
                compound=resolved_compound("agar"),
                role=MediaComponentRole.gelling_agent,
                provenance=[SOURCED_VALUES["agar"]],
            ),
        ],
        "provenance": [*SC_URA.provenance, SOURCED_VALUES["scoring_medium"]],
    }
)
"""The scoring plate: library ``SC_URA`` (SC, uracil dropped, URA3-plasmid selection)
made solid with an agar row whose amount the paper does not print. Loader-local: no
other served dataset states this plate, and ``media.py``'s ``SC_URA`` is liquid."""


#: The backbone name Table 2 prints for YB/I/BTS1, verbatim.
REPLICON_STATEMENT = "YEplac195"

#: What was read before concluding that no construct sequence is available.
_PLACEMENT_GAP_NOTE = (
    "Table 2 'Plasmids used in this study' describes YB/I/BTS1 as 'YEplac195 "
    "TDH3p-crtYB-CYC1t; TDH3p-crtI-CYC1t; TDH3p-BTS1-CYC1t' and cites Verwaal et al. "
    "(2007); the paper deposits no GenBank map and states only 'Information about all "
    "the strains and plasmids have been deposited in the public instance of the JBEI "
    "Registry (https://public-registry.jbei.org/)', which answers 403 anonymously"
)


def _placement_gap() -> ProvenanceGap:
    """The typed absence of a construct sequence for the YB/I/BTS1 cassette."""
    return ProvenanceGap(
        field="placement",
        reason=ProvenanceGapReason.not_released_by_primary,
        looked_in=_PAPER,
        note=_PLACEMENT_GAP_NOTE,
    )


# The constant engineered background: every screened strain carries the carotenogenic
# plasmid YB/I/BTS1, whose backbone Table 2 names as ``YEplac195`` (Ozaydin Table 2 /
# Verwaal et al. 2007). Two heterologous genes from X. dendrorhous + one extra native
# copy of BTS1 (YPL069C). See
# [[torchcell.datamodels.gene-addition-perturbation-design]].
#
# ``replicon_statement`` carries the backbone name the paper prints and nothing more.
# The localization token this loader used before P4 asserted a 2-micron replicon, which
# this paper never writes: measured, ``paper.md`` carries no "2 micron", "2\mu" or
# "multicopy" string for YB/I/BTS1, and the 2-micron claim comes from the YEplac195
# backbone's own literature. ``Localization`` therefore reads ``episomal_replicon`` and
# the source's own word for the replicon is kept verbatim beside it.
#
# ``placement`` is a typed gap. The paper deposits no map and releases no sequence; what
# it states is a registry deposit, "Information about all the strains and plasmids have
# been deposited in the public instance of the JBEI Registry
# (https://public-registry.jbei.org/)", and that registry answers 403 to an anonymous
# REST request (measured 2026-10-15), so the artifact exists and is withheld rather than
# never reported.
def _carotenogenic_cassette() -> list[GeneAdditionPerturbation]:
    """Fresh YB/I/BTS1 cassette perturbations (new objects per record)."""
    return [
        GeneAdditionPerturbation(
            systematic_gene_name="crtYB",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="crtYB",
            source_organism="Xanthophyllomyces dendrorhous",
            is_heterologous=True,
            localization=Localization.episomal_replicon,
            replicon_statement=REPLICON_STATEMENT,
            construct_name="YB/I/BTS1",
            placement=None,
            provenance_gaps=[_placement_gap()],
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="crtI",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="crtI",
            source_organism="Xanthophyllomyces dendrorhous",
            is_heterologous=True,
            localization=Localization.episomal_replicon,
            replicon_statement=REPLICON_STATEMENT,
            construct_name="YB/I/BTS1",
            placement=None,
            provenance_gaps=[_placement_gap()],
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YPL069C",
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name="BTS1",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization=Localization.episomal_replicon,
            replicon_statement=REPLICON_STATEMENT,
            construct_name="YB/I/BTS1",
            placement=None,
            provenance_gaps=[_placement_gap()],
        ),
    ]


# Valid S. cerevisiae systematic ORF name (optional trailing -A/-B for sub-features).
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


def _released_deletion(
    orf: str,
) -> KanMxDeletionPerturbation | ReleasedGeneNamePerturbation:
    """The deletion under the released identifier (#1045): an ORF-shaped one on the
    placed KanMX leaf with both names the ORF, any other on the released-gene-name leaf
    verbatim (kind deletion, marker KanMX).
    """
    if _SYSTEMATIC_RE.match(orf):
        return KanMxDeletionPerturbation(
            systematic_gene_name=orf, perturbed_gene_name=orf
        )
    return ReleasedGeneNamePerturbation(
        systematic_gene_name=orf,
        gene_namespace="scerevisiae_s288c_sgd_systematic",
        perturbed_gene_name=orf,
        released_gene_name=orf,
        released_kind=ReleasedGeneKind.deletion,
        marker="KanMX",
    )


# QC / phenotype flags parsed from the free-text Comment column (case-insensitive).
_FLAG_PATTERNS: dict[str, str] = {
    "flag_petite": r"petite|\bpet\b",
    "flag_tiny": r"\btiny\b",
    "flag_slow_growth": r"slow grow|slow on ",
    "flag_qc_failure": r"incorrect|qc failure",
    "flag_het_diploid": r"het diploid",
    "flag_sterile": r"sterile|does not mate|bi-mater",
    "flag_unusual_color": r"red colony|\bpink\b|ade mutant|ade2",
}


def _parse_color(value: Any) -> tuple[float | None, str | None]:
    """Return (numeric_score, text) for a raw Color cell.

    Numeric scores in [-5, 5] become the score; non-numeric annotations (``pet``,
    ``tiny``, ``_``, ``?``) become the text and leave the numeric None.
    """
    if pd.isna(value):
        return None, None
    try:
        num = float(value)
        return num, None
    except (ValueError, TypeError):
        text = str(value).strip()
        return None, text or None


def _parse_comment(comment: Any) -> dict[str, bool]:
    """Parse the free-text Comment into boolean QC/phenotype flags."""
    text = "" if pd.isna(comment) else str(comment).lower()
    return {flag: bool(re.search(pat, text)) for flag, pat in _FLAG_PATTERNS.items()}


@register_dataset
class CarotenoidOzaydin2013Dataset(ExperimentDataset):
    """Beta-carotene colony-color visual screen of the yeast deletion collection."""

    organism = Organism.scerevisiae

    si_url = "https://ars.els-cdn.com/content/image/1-s2.0-S109671761200081X-mmc1.xlsx"
    si_filename = "1-s2.0-S109671761200081X-mmc1.xlsx"

    def __init__(
        self,
        root: str = "data/torchcell/carotenoid_ozaydin2013",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset rooted at ``root`` with optional transforms."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return VisualScoreExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return VisualScoreExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The paper SI spreadsheet required before processing."""
        return [self.si_filename]

    def download(self) -> None:
        """Download the SI spreadsheet from the Elsevier ESM (scriptable direct URL).

        The stored artifact's sha256 is canonical: a freshly downloaded payload is
        verified before it is written, and the file in ``raw/`` (present or fresh) is
        verified against ``_SI_SHA256`` at the start of ``process``. A mismatch means
        upstream drift or corruption and raises (never silently followed).
        """
        dest = osp.join(self.raw_dir, self.si_filename)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Ozaydin SI from %s", self.si_url)
        req = urllib.request.Request(self.si_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 10000:
            raise RuntimeError(f"Ozaydin SI download too small: {len(data)} bytes")
        write_verified(data, dest, _SI_SHA256, self.si_url)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    def _released_rows(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Read Sheet 1 as released: one row per scored plate, plus the accounting."""
        xlsx = osp.join(self.raw_dir, self.si_filename)
        df = pd.read_excel(xlsx, sheet_name="Color scores of all deletions")

        # TOP200 curated subset (Sheet 2): gene name / function / category + membership.
        top200 = pd.read_excel(xlsx, sheet_name="Names and Functions of TOP200")
        top200_orfs = set(
            top200["ORF name"].dropna().astype(str).str.strip().str.upper()
        )
        top200_meta = {
            str(r["ORF name"]).strip().upper(): {
                "gene_name": r.get("Gene name"),
                "gene_function": r.get("Gene Function"),
                "assigned_category": r.get("Assigned Category"),
            }
            for _, r in top200.iterrows()
            if pd.notna(r.get("ORF name"))
        }

        rows: list[dict[str, Any]] = []
        released_names: list[str] = []
        text_rows: list[dict[str, Any]] = []
        n_blank_orf = 0
        for _, row in df.iterrows():
            orf = row.get("ORF name")
            if pd.isna(orf):
                n_blank_orf += 1
                continue
            orf = str(orf).strip().upper()
            if not _SYSTEMATIC_RE.match(orf):
                released_names.append(orf)
            num, text = _parse_color(row.get("Color"))
            strain = (
                str(row["Strain"]).strip() if pd.notna(row.get("Strain")) else "BY4741"
            )
            if num is None:
                text_rows.append({"orf": orf, "strain": strain, "text": text})
                continue
            meta = top200_meta.get(orf, {})
            rows.append(
                {
                    "orf": orf,
                    "strain": strain,
                    "visual_score": num,
                    "visual_score_min": None,
                    "n_replicates": 1,
                    "score_text": text,
                    "comment_annotations": _parse_comment(row.get("Comment")),
                    "in_top200": orf in top200_orfs,
                    **meta,
                }
            )
        accounting = {
            "dataset": self.name,
            "source_rows": int(len(df)),
            "kept_records": len(rows),
            "rules": [
                {
                    "rule": "orf_cell_is_blank",
                    "description": "the released row names no ORF",
                    "n_records": n_blank_orf,
                },
                {
                    "rule": "stored_on_released_gene_name_leaf",
                    "description": (
                        "the released ORF cell is not ORF-shaped (YLR287-A: the release "
                        "dropped the strand letter), so its rows are stored on "
                        "ReleasedGeneNamePerturbation under that name verbatim, never "
                        "resolved against the genome and never dropped (#1045); listed "
                        "here, not removed"
                    ),
                    "n_records": 0,
                    "items": sorted(released_names),
                },
                {
                    "rule": "color_cell_is_text_not_a_score",
                    "description": (
                        "the released Color cell is a word (pet, tiny, _), not a score on "
                        "the -5..+5 scale; VisualScorePhenotype.visual_score holds a "
                        "number, so the row's text is kept here, verbatim, with its ORF "
                        "and strain"
                    ),
                    "n_records": len(text_rows),
                    "items": text_rows,
                },
            ],
        }
        return pd.DataFrame(rows), accounting

    @post_process
    def process(self) -> None:
        """Parse the SI into per-row VisualScore experiments and write LMDB."""
        verify_raw_files(self.raw_dir, {self.si_filename: _SI_SHA256})
        usable, accounting = self._released_rows()
        log.info(
            "Ozaydin: %d released rows, %d records (one per scored row); %d text-only "
            "rows listed in the ledger, %d names on the released-gene-name leaf",
            accounting["source_rows"],
            len(usable),
            accounting["rules"][2]["n_records"],
            len(accounting["rules"][1]["items"]),
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            json.dump(accounting, fh, indent=2)
        usable.drop(columns=["comment_annotations"]).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        with env.begin(write=True) as txn:
            for _, row in tqdm(usable.iterrows(), total=len(usable)):
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
                idx += 1
        env.close()
        log.info("Wrote %d Ozaydin visual-score experiments to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(  # type: ignore[override]
        self, row: "pd.Series[Any]"
    ) -> tuple[VisualScoreExperiment, VisualScoreExperimentReference, Publication]:
        """Build the VisualScore experiment/reference/publication for one ORF."""
        # The row's own Strain cell, as released (BY4741 where the cell is blank).
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=str(row["strain"])
        )
        genotype = Genotype(
            perturbations=[_released_deletion(row["orf"]), *_carotenogenic_cassette()]
        )
        # Screen scored on SC-URA agar (URA-selective plasmid), 30 C.
        environment = Environment(
            media=OZAYDIN_SC_URA_AGAR, temperature=Temperature(value=30)
        )

        vmin = row["visual_score_min"]
        phenotype = VisualScorePhenotype(
            visual_score=float(row["visual_score"]),
            visual_score_min=None if pd.isna(vmin) else float(vmin),
            n_replicates=int(row["n_replicates"]),
            score_scale_min=SCORE_SCALE_MIN,
            score_scale_max=SCORE_SCALE_MAX,
            score_semantics=SCORE_SEMANTICS,
            target_product=TARGET_PRODUCT,
            target_metabolite_id=None,
            score_text=row["score_text"] if pd.notna(row["score_text"]) else None,
            comment_annotations=row["comment_annotations"],
        )
        # Reference = WT carrying YB/I/BTS1, scored 0 by construction.
        phenotype_reference = VisualScorePhenotype(
            visual_score=0.0,
            n_replicates=1,
            score_scale_min=SCORE_SCALE_MIN,
            score_scale_max=SCORE_SCALE_MAX,
            score_semantics=SCORE_SEMANTICS,
            target_product=TARGET_PRODUCT,
        )
        experiment = VisualScoreExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )
        reference = VisualScoreExperimentReference(
            dataset_name=self.name,
            genome_reference=genome_reference,
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )
        publication = Publication(
            pubmed_id="22918085",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/22918085/",
            doi="10.1016/j.ymben.2012.07.010",
            doi_url="https://doi.org/10.1016/j.ymben.2012.07.010",
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
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/carotenoid_ozaydin2013")
    dataset = CarotenoidOzaydin2013Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

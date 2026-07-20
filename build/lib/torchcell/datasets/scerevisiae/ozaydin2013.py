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

Maps to `VisualScorePhenotype` (WS4). One record per ORF that has a numeric color;
text-only rows (`pet`, `tiny`, `_`) carry no usable measurement and are excluded
(counted + logged, never silently dropped).
"""

import hashlib
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

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.datasets.dataset_registry import register_dataset

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


# The constant engineered background: every screened strain carries the carotenogenic
# 2-micron plasmid YB/I/BTS1 (YEplac195; Ozaydin Table 2 / Verwaal et al. 2007). Two
# heterologous genes from X. dendrorhous + one extra native copy of BTS1 (YPL069C).
# plasmid_contig_id/locus_tag stay None until the plasmid-sequence store lands; the
# raw sequence is Euroscarf P30796 (physical-only) -> reconstruct-from-parts. See
# [[torchcell.datamodels.gene-addition-perturbation-design]].
def _carotenogenic_cassette() -> list[GeneAdditionPerturbation]:
    """Fresh YB/I/BTS1 cassette perturbations (new objects per record)."""
    return [
        GeneAdditionPerturbation(
            systematic_gene_name="crtYB",
            perturbed_gene_name="crtYB",
            source_organism="Xanthophyllomyces dendrorhous",
            is_heterologous=True,
            localization="episomal_2micron",
            construct_name="YB/I/BTS1",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="crtI",
            perturbed_gene_name="crtI",
            source_organism="Xanthophyllomyces dendrorhous",
            is_heterologous=True,
            localization="episomal_2micron",
            construct_name="YB/I/BTS1",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="YPL069C",
            perturbed_gene_name="BTS1",
            source_organism="Saccharomyces cerevisiae",
            is_heterologous=False,
            localization="episomal_2micron",
            construct_name="YB/I/BTS1",
        ),
    ]


# Valid S. cerevisiae systematic ORF name (optional trailing -A/-B for sub-features).
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

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

        The stored artifact's sha256 is canonical: an already-present file is verified
        against ``_SI_SHA256`` rather than trusted, and a freshly downloaded file is
        verified before use. A mismatch means upstream drift or corruption and raises
        (never silently followed).
        """
        dest = osp.join(self.raw_dir, self.si_filename)
        if osp.exists(dest):
            digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
            if digest != _SI_SHA256:
                raise RuntimeError(
                    f"Ozaydin SI sha256 mismatch for {dest}: got {digest}, "
                    f"expected {_SI_SHA256}"
                )
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading Ozaydin SI from %s", self.si_url)
        req = urllib.request.Request(self.si_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if len(data) < 10000:
            raise RuntimeError(f"Ozaydin SI download too small: {len(data)} bytes")
        digest = hashlib.sha256(data).hexdigest()
        if digest != _SI_SHA256:
            raise RuntimeError(
                f"Ozaydin SI sha256 mismatch on download: got {digest}, "
                f"expected {_SI_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    def _aggregate_by_orf(self) -> pd.DataFrame:
        """Read Sheet 1, aggregate replicate rows to one row per ORF."""
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

        records: dict[str, dict[str, Any]] = {}
        malformed: set[str] = set()
        for _, row in df.iterrows():
            orf = row.get("ORF name")
            if pd.isna(orf):
                continue
            orf = str(orf).strip().upper()
            # Some SI ORF names are malformed (e.g. 'YLR287-A' missing the W/C). Do not
            # guess the missing letter -- exclude non-conforming names and log them.
            if not _SYSTEMATIC_RE.match(orf):
                malformed.add(orf)
                continue
            num, text = _parse_color(row.get("Color"))
            flags = _parse_comment(row.get("Comment"))
            strain = (
                str(row["Strain"]).strip() if pd.notna(row.get("Strain")) else "BY4741"
            )
            rec = records.setdefault(
                orf,
                {
                    "orf": orf,
                    "strain": strain,
                    "numeric_scores": [],
                    "texts": set(),
                    "flags": {k: False for k in _FLAG_PATTERNS},
                },
            )
            if num is not None:
                rec["numeric_scores"].append(num)
            if text is not None:
                rec["texts"].add(text)
            for k, v in flags.items():
                rec["flags"][k] = rec["flags"][k] or v

        rows: list[dict[str, Any]] = []
        for orf, rec in records.items():
            scores = rec["numeric_scores"]
            meta = top200_meta.get(orf, {})
            rows.append(
                {
                    "orf": orf,
                    "strain": rec["strain"],
                    "visual_score": max(scores) if scores else None,
                    "visual_score_min": min(scores) if len(scores) > 1 else None,
                    "n_replicates": len(scores),
                    "score_text": ";".join(sorted(rec["texts"])) or None,
                    "comment_annotations": rec["flags"],
                    "in_top200": orf in top200_orfs,
                    **meta,
                }
            )
        if malformed:
            log.info(
                "Ozaydin: excluded %d malformed ORF names (e.g. %s)",
                len(malformed),
                sorted(malformed)[:5],
            )
        return pd.DataFrame(rows)

    @post_process
    def process(self) -> None:
        """Parse the SI into per-ORF VisualScore experiments and write LMDB."""
        agg = self._aggregate_by_orf()
        n_total = len(agg)
        # Exclude ORFs with no numeric color (text-only / missing) -- not measurements.
        usable = agg[agg["visual_score"].notna()].reset_index(drop=True)
        n_excluded = n_total - len(usable)
        log.info(
            "Ozaydin: %d ORFs total, %d usable (numeric color), %d excluded (text-only)",
            n_total,
            len(usable),
            n_excluded,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
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
        strain = (
            str(row["strain"]) if row["strain"] in ("BY4741", "BY4730") else "BY4741"
        )
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=row["orf"], perturbed_gene_name=row["orf"]
                ),
                *_carotenogenic_cassette(),
            ]
        )
        # Screen scored on SC-URA agar (URA-selective plasmid), 30 C.
        environment = Environment(
            media=Media(name="SC-URA", state="solid", is_synthetic=True),
            temperature=Temperature(value=30),
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

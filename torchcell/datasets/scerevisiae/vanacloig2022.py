# torchcell/datasets/scerevisiae/vanacloig2022
# [[torchcell.datasets.scerevisiae.vanacloig2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/vanacloig2022
# Test file: tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 comparative chemical-genomic screen (env x geno -> response).

Vanacloig-Pedros et al. 2022 (FEMS Yeast Research, doi:10.1093/femsyr/foac036) profiled
the '3DeltaAlpha' drug-sensitized yeast deletion library (4309 barcoded non-essential
KanMX deletions in a MATalpha pdr1::natMX pdr3::KlURA3 snq2::KlLEU2 background) grown
ANAEROBICALLY in each of 34+ plant-hydrolysate inhibitor compounds at their IC30, in
independent BIOLOGICAL TRIPLICATE, alongside matched inhibitor-free controls. The readout
is log2(inhibitor/control) barcode abundance -- a per-deletion fitness response.

This maps cleanly onto the WS15 environment-perturbation ontology:
``EnvironmentResponseExperiment`` = deletion ``Genotype`` (library KanMX KO + the constant
3DeltaAlpha ``NatMxDeletion``/``MarkerDeletion`` background) x anaerobic ``Environment``
carrying a ``SmallMoleculePerturbation`` (compound at IC30) -> ``EnvironmentResponsePhenotype``
(``measurement_type=log2_ratio``, ``n_samples=3`` biological_replicate).

READOUT PROVENANCE / RIGOR. The paper's published log2 values are edgeR TMM+glmQLF paired
logFCs whose exact reproduction needs (a) R/edgeR (unavailable here) and (b) the OUP
Supplementary Table S1 (per-compound control pairing + IC30 molar values), which is NOT
scriptable (academic.oup.com returns 403). The GEO deposit GSE186866 releases ONLY the raw
barcode-count matrix (scriptable, sha256-pinned). So this loader RECOMPUTES the readout
deterministically from that canonical raw artifact: per sample CPM (counts-per-million)
library-size normalization, then per gene ``log2((CPM_compound_rep + PRIOR) /
(CPM_pooled_control_mean + PRIOR))`` for each of the 3 replicates; the stored response is
the mean of the 3 rep log2 ratios and the reported uncertainty is their sample SD (SE =
SD/sqrt(3)). This is the paper's DEFINED quantity ("log2 of the normalized read counts for
inhibitor/control ratio") reproduced from raw counts -- it is NOT identical to the
published edgeR logFC and must not be treated as such (documented in ``units`` +
provenance). The pooled control is the mean of all ``ControlN`` columns (Table S1's
per-compound DMSO-vs-SynBase pairing is unavailable). Per-compound IC30 molar values are
likewise in Table S1, so ``concentration`` carries only ``basis="IC30"``.
"""

import gzip
import logging
import math
import os
import os.path as osp
import pickle
import re
import urllib.request
from collections.abc import Callable
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    Concentration,
    DoseBasis,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MarkerDeletionPerturbation,
    MeasurementType,
    Media,
    NatMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# GEO GSE186866 supplementary raw barcode-count matrix (the only scriptable source).
DATA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186866/suppl/"
    "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
)
DATA_FILENAME = "GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"
# sha256 of the retrieved artifact (pins the canonical raw version; verified on download).
DATA_SHA256 = "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"

MEASUREMENT_UNITS = "log2(mean CPM compound / mean CPM pooled control), 3 biol. reps"
CPM_PRIOR = 1.0  # CPM pseudocount avoiding log2(0) for dropout strains (documented)

_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)
# Compound sample columns: '<compound>_CG<batch>_rep<n>'. Control columns: 'ControlN_CG..'.
_SAMPLE_RE = re.compile(r"^(?P<compound>.+)_CG\d+_rep\d+$")
_CONTROL_RE = re.compile(r"^Control\d+_CG\d+$")

# The constant drug-sensitized 3DeltaAlpha background, deleted in EVERY library strain
# (paper Methods: "MATalpha pdr1::natMX; pdr3::KI.URA3; snq2::KI.LEU2"). Systematic names
# verified against SGD R64: PDR1=YGL013C, PDR3=YBL005W, SNQ2=YDR011W. Immutable (frozen
# pydantic) so the same objects are reused across all records.
BACKGROUND_GENES = frozenset({"YGL013C", "YBL005W", "YDR011W"})


def _background_perturbations() -> list[Any]:
    """The constant 3DeltaAlpha background deletions (shared across every record)."""
    return [
        NatMxDeletionPerturbation(
            systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YBL005W", perturbed_gene_name="PDR3", marker="KlURA3"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YDR011W", perturbed_gene_name="SNQ2", marker="KlLEU2"
        ),
    ]


@register_dataset
class EnvChemgenVanacloig2022Dataset(ExperimentDataset):
    """Anaerobic chemical-genomic env x geno -> log2(inhibitor/control) response screen."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_vanacloig2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome needed: source rows carry systematic ORFs)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The GEO raw barcode-count matrix required before processing."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Download the GEO GSE186866 raw count matrix; verify its pinned sha256."""
        dest = osp.join(self.raw_dir, DATA_FILENAME)
        if osp.exists(dest):
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        log.info("Downloading GEO GSE186866 raw counts from %s", DATA_URL)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        if digest != DATA_SHA256:
            raise RuntimeError(
                f"GSE186866 sha256 mismatch: got {digest}, expected {DATA_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(data)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(data))

    def _load_matrix(self) -> pd.DataFrame:
        """Read the gzipped raw-count TSV into a DataFrame."""
        path = osp.join(self.raw_dir, DATA_FILENAME)
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle, sep="\t")

    @post_process
    def process(self) -> None:
        """Recompute per-(gene, compound) log2 responses from raw counts; write LMDB."""
        df = self._load_matrix()
        sample_cols = [c for c in df.columns if c not in ("gene", "std_name")]
        control_cols = [c for c in sample_cols if _CONTROL_RE.match(c)]
        if not control_cols:
            raise RuntimeError("no ControlN_CG* columns found in GSE186866 matrix")

        # Group compound columns by compound token (3 biological replicates each).
        compound_to_cols: dict[str, list[str]] = {}
        for c in sample_cols:
            if _CONTROL_RE.match(c):
                continue
            m = _SAMPLE_RE.match(c)
            if m is None:
                raise RuntimeError(f"unparseable sample column: {c!r}")
            compound_to_cols.setdefault(m.group("compound"), []).append(c)

        # Resolve systematic ORF per row (barcode prefix); drop non-ORF rows and rows
        # whose counts are entirely missing (QC-dropped barcodes: the source has NO
        # data for them, so they cannot yield a response -- dropped, never imputed).
        orfs = df["gene"].astype(str).str.split("_", n=1).str[0]
        is_orf = orfs.map(lambda g: bool(_SYSTEMATIC_RE.match(g)))
        has_counts = ~df[sample_cols].isna().any(axis=1)
        # Drop library barcodes whose ORF is a constant-background gene (PDR3/SNQ2 appear
        # in the pool): a gene already deleted in the 3DeltaAlpha background cannot be an
        # independent screened deletion, and keeping it would double-delete the same gene.
        not_background = ~orfs.isin(BACKGROUND_GENES)
        keep = is_orf & has_counts & not_background
        n_dropped_nonorf = int((~is_orf).sum())
        n_dropped_nan = int((is_orf & ~has_counts).sum())
        n_dropped_bg = int((is_orf & has_counts & ~not_background).sum())
        if n_dropped_bg:
            log.info(
                "Vanacloig: dropping %d library rows that are background genes: %s",
                n_dropped_bg,
                df.loc[is_orf & ~not_background, "std_name"].tolist(),
            )
        if n_dropped_nan:
            log.info(
                "Vanacloig: dropping %d ORF rows with all-missing counts: %s",
                n_dropped_nan,
                df.loc[is_orf & ~has_counts, "std_name"].tolist(),
            )
        df = df.loc[keep].reset_index(drop=True)
        orfs = orfs.loc[keep].reset_index(drop=True)
        std = df["std_name"].astype(str)

        counts = df[sample_cols].to_numpy(dtype=np.float64)  # genes x samples
        lib_sizes = counts.sum(axis=0)  # per-sample library size
        cpm = counts / lib_sizes * 1e6  # genes x samples (CPM)
        col_idx = {c: i for i, c in enumerate(sample_cols)}
        control_mean_cpm = cpm[:, [col_idx[c] for c in control_cols]].mean(axis=1)
        log_ctrl = np.log2(control_mean_cpm + CPM_PRIOR)  # per gene

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        log.info(
            "Vanacloig: %d ORF rows kept (%d non-ORF, %d all-NaN, %d background "
            "dropped), %d compounds, %d control columns",
            len(df),
            n_dropped_nonorf,
            n_dropped_nan,
            n_dropped_bg,
            len(compound_to_cols),
            len(control_cols),
        )

        publication = Publication(
            doi="10.1093/femsyr/foac036",
            doi_url="https://doi.org/10.1093/femsyr/foac036",
        )
        background = _background_perturbations()

        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        rows_written = 0
        with env.begin(write=True) as txn:
            for compound, cols in tqdm(sorted(compound_to_cols.items())):
                environment = self._environment(compound)
                reference = self._reference(compound, environment)
                ref_dump = reference.model_dump()
                pub_dump = publication.model_dump()
                rep_idx = [col_idx[c] for c in cols]
                # per-gene x per-rep log2 ratios (genes x n_reps)
                log_rep = np.log2(cpm[:, rep_idx] + CPM_PRIOR) - log_ctrl[:, None]
                resp = log_rep.mean(axis=1)
                sd = log_rep.std(axis=1, ddof=1) if log_rep.shape[1] > 1 else None
                n_rep = len(cols)
                for gi in range(len(df)):
                    experiment = self._experiment(
                        orf=orfs.iat[gi],
                        common=std.iat[gi],
                        environment=environment,
                        background=background,
                        response=float(resp[gi]),
                        sd=(float(sd[gi]) if sd is not None else None),
                        n_samples=n_rep,
                    )
                    txn.put(
                        f"{idx}".encode(),
                        pickle.dumps(
                            {
                                "experiment": experiment.model_dump(),
                                "reference": ref_dump,
                                "publication": pub_dump,
                            }
                        ),
                    )
                    idx += 1
                rows_written += len(df)
        env.close()
        log.info("Wrote %d Vanacloig environment-response experiments to LMDB", idx)

    def _environment(self, compound: str) -> Environment:
        """Anaerobic SynBase environment carrying the compound at its IC30."""
        return Environment(
            media=Media(name="SynBase", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30.0),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound(compound),
                    concentration=Concentration(basis=DoseBasis.IC30),
                )
            ],
            aerobicity="anaerobic",
            duration_hours=48.0,
        )

    def _reference(
        self, compound: str, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """The 3DeltaAlpha parent baseline: log2 ratio 0 in the same compound environment."""
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=0.0,
            units=MEASUREMENT_UNITS,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="3DeltaAlpha"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    def _experiment(
        self,
        *,
        orf: str,
        common: str,
        environment: Environment,
        background: list[Any],
        response: float,
        sd: float | None,
        n_samples: int,
    ) -> EnvironmentResponseExperiment:
        """Build one env x geno -> log2-response experiment for a (gene, compound) pair."""
        perturbed_name = common if common and common != "nan" else orf
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=perturbed_name
                ),
                *background,
            ]
        )
        unc = sd if (sd is not None and not math.isnan(sd)) else None
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=response,
            environment_response_uncertainty=unc,
            environment_response_uncertainty_type=(
                UncertaintyType.sample_sd if unc is not None else None
            ),
            n_samples=n_samples,
            sample_unit=SampleUnit.biological_replicate,
            units=MEASUREMENT_UNITS,
        )
        return EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/env_chemgen_vanacloig2022")
    dataset = EnvChemgenVanacloig2022Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

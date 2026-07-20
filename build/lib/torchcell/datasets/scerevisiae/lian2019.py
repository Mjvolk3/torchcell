# torchcell/datasets/scerevisiae/lian2019
# [[torchcell.datasets.scerevisiae.lian2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/lian2019
# Test file: tests/torchcell/datasets/scerevisiae/test_lian2019.py
"""Lian 2019 MAGIC genome-wide CRISPRa/i/d furfural-tolerance screen (per-guide enrichment).

Lian et al. 2019 (Nat Commun 10:5794, doi:10.1038/s41467-019-13621-4; PMID 31857575) built
MAGIC -- three genome-scale gRNA libraries driven by three ORTHOGONAL Cas effectors in one
pooled CRISPR-AID host, so a single cell carries exactly one guide of exactly one MODE:
- CRISPRa (activation)   -- ``dLbCas12a-VP``  (37,817 guides)
- CRISPRi (interference) -- ``dSpCas9-RD1152``(37,870 guides)
- CRISPRd (deletion)     -- ``SaCas9``        (24,806 guides)
The unique guide is a genetic barcode; furfural-tolerance is mapped by tracking each guide's
enrichment (furfural-selected vs untreated) by NGS. Screening is ITERATIVE in accumulating
integrated backgrounds: round 1 = bAID host (5 mM furfural); round 2 = +SIZ1i (10 mM);
round 3 = +SIZ1i +NAT1a (15 mM). So a round-2/3 record is a genuine 2-/3-perturbation
MIXED-MODALITY combo -- the strain actually in the tube.

RECORD = one (guide x round) EnvironmentResponseExperiment:
- GENOTYPE (per-guide): the library member as a ``CrisprActivation/Interference/Deletion``
  perturbation (target gene + this guide's spacer + its effector), PLUS the round's
  integrated background perturbations (SIZ1 interference for r2/r3; NAT1 activation for r3;
  guide unspecified -> ``guide_sequence=None``). This is the first dataset to exercise the
  AXIS-4 expression-modulation ontology (``[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]``).
- ENVIRONMENT: furfural (5/10/15 mM by round) as a ``SmallMoleculePerturbation`` on
  SED/G418 liquid medium, 30 C, aerobic.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` ``measurement_type=log2_ratio``,
  ``environment_response`` = mean log2(after/before) over 3 biological triplicates;
  ``environment_response_uncertainty`` = SD (``sample_sd``, n=3 -> SE = SD/sqrt(3)).
  Reference = no-enrichment baseline (log2FC 0) in the bAID host.

DATA SOURCE + PROVENANCE (fully reproducible from raw NGS, NOT a released table):
The furfural per-guide enrichment is NOT in any Nature supplement (only the design library
Supp Data 1-3 + reference Supp Data 4 were released; Fig 2a/c/e profiles are excluded from
Source Data). It is reprocessed from raw reads -- NCBI SRA PRJNA504483 (21 runs: 3 rounds x
before/after x triplicate + ecLibA/I/D baselines) mapped to the 100,493-guide reference
(Supp Data 4, sha256 4e3f225a...). The loader consumes the derived, sha256-pinned
``guide_enrichment_final.tsv`` in the library mirror; the reprocessing recipe + scripts are
archived beside it (``PROVENANCE.md``). Reprocessing was VALIDATED against the paper's hits:
PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d round-1 rank 2. Details:
``[[lian2019-magic-data-availability]]``.

GENE-RESOLUTION / DROP NOTES (documented, not guessed):
- Common gene names resolve to current R64 ORFs via the genome (5,060/5,226; the rest are
  ncRNA/rDNA features absent from the ORF genome). Guides whose target does not resolve are
  DROPPED and counted.
- 300 random negative-control guides (100/library; blank design score) are excluded.
- 16 guides carry a SOURCE-CORRUPTED gene name (an Excel date/serial artifact present
  IDENTICALLY in both the reference and the library, so unrecoverable from our inputs) --
  DROPPED, counted, and flagged for review.
- Per-modality guide barcode: activation 23 nt spacer / interference 20 nt spacer / deletion
  the 44 bp guide+donor window (stored verbatim as ``guide_sequence``).
"""

import hashlib
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
from collections.abc import Callable
from typing import Any

import lmdb
import pandas as pd
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    CrisprActivationPerturbation,
    CrisprConstruct,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/s41467-019-13621-4"
PMID = "31857575"

_LIBRARY_CITATION_KEY = "lianMultifunctionalGenomewideCRISPR2019"
_TSV_FILENAME = "guide_enrichment_final.tsv"
_TSV_SHA256 = "f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647"

# Biological triplicates (Methods: "biological triplicates for untreated and furfural
# stressed libraries").
_N_SAMPLES = 3

# Orthogonal Cas effector per modality (Table 1).
_EFFECTOR = {"a": "dLbCas12a-VP", "i": "dSpCas9-RD1152", "d": "SaCas9"}
_PERT_CLASS = {
    "a": CrisprActivationPerturbation,
    "i": CrisprInterferencePerturbation,
    "d": CrisprDeletionPerturbation,
}
# Iterative furfural concentration per round (Fig 2: 5, 10, 15 mM).
_ROUND_FURFURAL_MM = {1: 5.0, 2: 10.0, 3: 15.0}
# Integrated backgrounds accumulated per round (gene, modality). R1=bAID (none);
# R2=+SIZ1i; R3=+SIZ1i+NAT1a. SIZ1=YDR409W, NAT1=YDL040C.
_ROUND_BACKGROUND: dict[int, list[tuple[str, str, str]]] = {
    1: [],
    2: [("YDR409W", "SIZ1", "i")],
    3: [("YDR409W", "SIZ1", "i"), ("YDL040C", "NAT1", "a")],
}

_UNITS = (
    "log2(furfural-selected / untreated guide-barcode abundance); positive = the "
    "perturbation confers furfural tolerance"
)

_SYSTEMATIC_RE = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$"
)


def _crispr_pert(orf: str, common: str, mod: str, guide: str | None) -> Any:
    """Build the modality-appropriate CRISPR perturbation for a target gene."""
    construct = CrisprConstruct(
        effector=_EFFECTOR[mod], guide_sequence=guide, n_guides=1 if guide else None
    )
    cls = _PERT_CLASS[mod]
    return cls(systematic_gene_name=orf, perturbed_gene_name=common, crispr=construct)


@register_dataset
class CrisprMagicLian2019Dataset(ExperimentDataset):
    """Lian 2019 MAGIC per-guide CRISPRa/i/d furfural-enrichment env x geno dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/crispr_magic_lian2019",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for common-name -> current-R64-ORF resolution."""
        self.genome = genome
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
        """The sha256-pinned derived per-guide enrichment table."""
        return [_TSV_FILENAME]

    def download(self) -> None:
        """Copy the pinned derived enrichment table from the library mirror; verify sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _TSV_FILENAME)
        if not osp.exists(dest):
            src = osp.join(
                os.environ["DATA_ROOT"],
                "torchcell-library",
                _LIBRARY_CITATION_KEY,
                "data",
                _TSV_FILENAME,
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror data file not found: {src}. Source is the sha256-pinned "
                    f"guide_enrichment_final.tsv (SRA PRJNA504483 reprocessing; see PROVENANCE.md)."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _TSV_SHA256:
            raise RuntimeError(
                f"{_TSV_FILENAME} sha256 mismatch: got {digest}, expected {_TSV_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _TSV_SHA256)

    def _resolver(self) -> Callable[[str], str | None]:
        """Common/standard gene name -> current-R64 ORF (self if already an ID, else alias)."""
        if self.genome is None:
            raise RuntimeError(
                "CrisprMagicLian2019Dataset requires a genome; inject SCerevisiaeGenome(...)"
            )
        ids = set(self.genome.gene_attribute_table["ID"])
        alias = self.genome.alias_to_systematic

        def resolve(name: str) -> str | None:
            n = str(name).strip()
            if n in ids:
                return n
            u = n.upper()
            if u in ids:
                return u
            cand = alias.get(u) or alias.get(n)
            if cand and cand[0] in ids:
                return cand[0]
            return None

        return resolve

    def _environment(self, furfural_mm: float) -> Environment:
        """SED/G418 liquid culture carrying furfural (mM), 30 C, aerobic."""
        return Environment(
            media=Media(name="SED/G418", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30.0),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=Compound(name="furfural"),
                    concentration=Concentration(
                        value=furfural_mm, unit=ConcentrationUnit.millimolar
                    ),
                )
            ],
            aerobicity="aerobic",
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """No-enrichment baseline: a guide that neither enriches nor depletes -> log2FC 0."""
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=0.0,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.biological_replicate,
            units=_UNITS,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="bAID"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    @post_process
    def process(self) -> None:
        """Build one env x geno -> log2-enrichment record per (guide, round); write LMDB."""
        df = pd.read_csv(osp.join(self.raw_dir, _TSV_FILENAME), sep="\t")
        resolve = self._resolver()
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        pub_dump = publication.model_dump()

        # Constant per-round environment + reference + background perturbations.
        prepared: dict[int, dict[str, Any]] = {}
        for rnd, mm in _ROUND_FURFURAL_MM.items():
            environment = self._environment(mm)
            background = [
                _crispr_pert(orf, common, mod, None)
                for orf, common, mod in _ROUND_BACKGROUND[rnd]
            ]
            prepared[rnd] = {
                "environment": environment,
                "ref_dump": self._reference(environment).model_dump(),
                "background": background,
            }

        # ORFs integrated as this round's background; a foreground guide that TARGETS its
        # own background gene is redundant (and would collapse to an empty strain signature
        # once the background is subtracted), so it is skipped for that round.
        background_orfs = {
            rnd: {orf for orf, _, _ in _ROUND_BACKGROUND[rnd]} for rnd in (1, 2, 3)
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_ctrl = n_corrupt = n_unresolved = n_nan = n_bg_self = 0
        unresolved_genes: set[str] = set()
        with env.begin(write=True) as txn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="lian2019"):
                if bool(row["is_control"]):
                    n_ctrl += 1
                    continue
                if bool(row["corrupted_gene"]):
                    n_corrupt += 1
                    continue
                orf = resolve(row["gene"])
                if orf is None:
                    n_unresolved += 1
                    unresolved_genes.add(str(row["gene"]))
                    continue
                mod = str(row["mod"])
                common = str(row["gene"])
                guide = str(row["spacer"])
                for rnd in (1, 2, 3):
                    mean = row[f"r{rnd}_log2fc_mean"]
                    if pd.isna(mean):
                        n_nan += 1
                        continue
                    if orf in background_orfs[rnd]:
                        n_bg_self += 1
                        continue
                    sd = row[f"r{rnd}_log2fc_sd"]
                    has_sd = not pd.isna(sd) and not math.isinf(float(sd))
                    item = prepared[rnd]
                    genotype = Genotype(
                        perturbations=[
                            _crispr_pert(orf, common, mod, guide),
                            *item["background"],
                        ]
                    )
                    phenotype = EnvironmentResponsePhenotype(
                        measurement_type=MeasurementType.log2_ratio,
                        environment_response=float(mean),
                        environment_response_uncertainty=(
                            float(sd) if has_sd else None
                        ),
                        environment_response_uncertainty_type=(
                            UncertaintyType.sample_sd if has_sd else None
                        ),
                        n_samples=_N_SAMPLES,
                        sample_unit=SampleUnit.biological_replicate,
                        units=_UNITS,
                    )
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=item["environment"],
                        phenotype=phenotype,
                    )
                    txn.put(
                        f"{idx}".encode(),
                        pickle.dumps(
                            {
                                "experiment": experiment.model_dump(),
                                "reference": item["ref_dump"],
                                "publication": pub_dump,
                            }
                        ),
                    )
                    idx += 1
        env.close()
        log.info(
            "Lian2019: wrote %d (guide x round) records; dropped %d control + %d corrupted "
            "+ %d unresolved-gene guides (%d distinct genes); %d (guide,round) had no "
            "enrichment; %d skipped (guide targets its own round background)",
            idx,
            n_ctrl,
            n_corrupt,
            n_unresolved,
            len(unresolved_genes),
            n_nan,
            n_bg_self,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
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
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/crispr_magic_lian2019")
    dataset = CrisprMagicLian2019Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

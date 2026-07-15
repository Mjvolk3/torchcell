# torchcell/datasets/scerevisiae/mormino2022
# [[torchcell.datasets.scerevisiae.mormino2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/mormino2022
# Test file: tests/torchcell/datasets/scerevisiae/test_mormino2022.py
"""Mormino 2022 Haa1-biosensor CRISPRi acetic-acid-sensitivity screen (12 isolated strains).

Mormino, Lenitz, Siewers & Nygard 2022 (Microb Cell Fact 21:214,
doi:10.1186/s12934-022-01938-7; PMID 36284296; PMC9571444) screened a pooled S. cerevisiae
CRISPRi strain library (dCas9-Mxi1 repressing each essential / respiratory-growth-essential
gene; library from Smith JD et al. 2017, ref [17], derived from BY4742) with an integrated
Haa1-based acetic-acid biosensor (synthetic TF BM3R1-Haa1-mTurquoise2 -> mCherry RFP
reporter). FACS enriched cells with the HIGHEST RFP signal; a higher biosensor signal was
shown to mark cells MORE SENSITIVE to acetic acid. This is the FIRST torchcell dataset on
the CRISPR interference ontology (``[[plan.torchcell-crispr-expression-perturbation.2026.07.12]]``).

RECORD = one isolated strain (Table 1) as an EnvironmentResponseExperiment:
- GENOTYPE: a single ``CrisprInterferencePerturbation`` (target gene, effector ``dCas9-Mxi1``,
  ``guide_sequence=None`` -- Mormino does not release per-strain spacers; the guides live
  upstream in the Smith 2017 library) in the BY4742 background.
- ENVIRONMENT: acetic acid 50 mM at pH 3.5 (the screen/biosensor condition), 30 C, aerobic.
- PHENOTYPE: ``EnvironmentResponsePhenotype`` ``measurement_type=categorical``. The RFP
  biosensor call from Table 1 -- ``+`` (enhanced signal) -> ``category="sensitive"``; ``=``
  (same as control) -> ``category="no_effect"``. Reference = the CC23 control strain
  (``no_effect``).

SCOPE / WHAT IS DELIBERATELY NOT BUILT (documented, not guessed):
- Only Table 1's 12 individually-isolated strains carry a machine-readable, per-strain
  phenotype call. The genome-wide enrichment is figure-only (Figs 2-6, bar charts / growth
  curves, no released data matrix); digitizing bars would violate provenance, so the numeric
  FI / sfpHluorin / growth values are NOT ingested. The guide-level genome-wide screen data
  lives upstream in Smith JD et al. 2017 (the target if a CRISPRi x fitness matrix is wanted).
- Table 1's Growth column (``ns`` / ``-`` / ``ND`` / blank) is an ambiguous secondary readout
  and is NOT stored (retained in the literal for review only).
- ``n_samples`` is left None: Table 1 is a qualitative summary call, and the per-strain
  replicate count is not cleanly attributable to the RFP call (Figs report 3/5/6/7 replicates
  for different sample groups). Temperature 30 C is standard yeast cultivation (not otherwise
  specified for these isolates). Both are flagged for review.

PROVENANCE (sha256-pinned mirror PDF; the values are the embedded Table 1 literal):
- Verbatim (Results): "sorted by FACS to enrich the population for cells displaying the
  highest RFP signal"; "These cells with higher biosensor signal were demonstrated to be more
  sensitive to acetic acid"; biosensor characterized "at 0 and 50 mM acetic acid ... at
  pH 3.5". Library ref [17] = Smith 2017 (BY4742). Effector = "dCas9-Mxi".
- Table 1 "Properties of isolated strains" is embedded as the ``TABLE_1`` literal, transcribed
  from the sha256-pinned library-mirror ``paper.pdf`` (the only source; no SI data file was
  released -- manifest ``si_data_sources: []``).
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
import shutil
from collections.abc import Callable
from typing import Any

import lmdb
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
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
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1186/s12934-022-01938-7"
PMID = "36284296"

_LIBRARY_CITATION_KEY = "morminoIdentificationAceticAcid2022"
PDF_FILENAME = "paper.pdf"
PDF_SHA256 = "388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a"

_EFFECTOR = "dCas9-Mxi1"
_ACETIC_ACID_MM = 50.0  # FACS screen / biosensor condition (Results); pH 3.5.
_RFP_CATEGORY = {"+": "sensitive", "=": "no_effect"}
_REFERENCE_CATEGORY = "no_effect"  # CC23 control strain (no enhanced RFP).
_UNITS = (
    "Haa1 acetic-acid biosensor RFP (mCherry) call: 'sensitive' = enhanced reporter signal "
    "(higher acetic-acid retention / sensitivity), 'no_effect' = signal as the control strain"
)

# Table 1 "Properties of isolated strains" (embedded literal, transcribed from the sha256-
# pinned mirror paper.pdf). Columns: strain id, RFP expression call (+/=), growth (kept for
# review, NOT stored), target-gene common name. Systematic names resolved via the genome.
_TABLE_1: list[tuple[str, str, str, str]] = [
    ("#3", "+", "ns", "QCR8"),
    ("#8", "+", "-", "TIF34"),
    ("#13", "+", "", "MSN5"),
    ("#15", "=", "ND", "NDC1"),
    ("#17", "+", "", "PAP1"),
    ("#32", "=", "", "CBP2"),
    ("#33", "+", "", "COX10"),
    ("#35", "+", "", "TRA1"),
    ("#37", "=", "ns", "UBA2"),
    ("#43", "=", "ND", "RPS30B"),
    ("#46", "=", "ND", "HSH49"),
    ("#49", "=", "ND", "LCB1"),
]


@register_dataset
class CrispriMormino2022Dataset(ExperimentDataset):
    """Mormino 2022 CRISPRi acetic-acid-sensitivity biosensor screen (12 isolated strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/crispri_mormino2022",
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
        """The sha256-pinned mirror PDF the Table 1 literal was transcribed from."""
        return [PDF_FILENAME]

    def download(self) -> None:
        """Stage the mirror ``paper.pdf`` into raw_dir and verify its sha256.

        The dataset values are the embedded ``TABLE_1`` literal; the canonical source is the
        sha256-pinned library-mirror PDF (no SI data file was released).
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, PDF_FILENAME)
        if not osp.exists(dest):
            mirror = osp.join(
                os.environ["DATA_ROOT"],
                "torchcell-library",
                _LIBRARY_CITATION_KEY,
                PDF_FILENAME,
            )
            if not osp.exists(mirror):
                raise RuntimeError(f"library mirror PDF not found: {mirror}")
            data = open(mirror, "rb").read()
            got = hashlib.sha256(data).hexdigest()
            if got != PDF_SHA256:
                raise RuntimeError(
                    f"Mormino2022 paper.pdf sha256 mismatch: got {got}, expected {PDF_SHA256}"
                )
            shutil.copyfile(mirror, dest)
        log.info("Verified %s (sha256 %s)", dest, PDF_SHA256)

    def _resolve(self, common: str) -> str:
        """Common gene name -> current-R64 ORF (all 12 Table 1 targets resolve)."""
        if self.genome is None:
            raise RuntimeError(
                "CrispriMormino2022Dataset requires a genome; inject SCerevisiaeGenome(...)"
            )
        ids = set(self.genome.gene_attribute_table["ID"])
        cand = self.genome.alias_to_systematic.get(common)
        if cand and cand[0] in ids:
            return cand[0]
        raise RuntimeError(
            f"Mormino2022: gene {common!r} did not resolve to an R64 ORF"
        )

    def _environment(self) -> Environment:
        """Aerobic acetic-acid (50 mM, pH 3.5) liquid culture, 30 C."""
        return Environment(
            media=Media(name="SD, pH 3.5", state="liquid", is_synthetic=True),
            temperature=Temperature(value=30.0),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=Compound(name="acetic acid"),
                    concentration=Concentration(
                        value=_ACETIC_ACID_MM, unit=ConcentrationUnit.millimolar
                    ),
                )
            ],
            aerobicity="aerobic",
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """CC23 control strain: biosensor RFP as the control (no enhancement -> no_effect)."""
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            category=_REFERENCE_CATEGORY,
            units=_UNITS,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4742"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    @post_process
    def process(self) -> None:
        """Build one categorical env x geno record per Table 1 isolated strain; write LMDB."""
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        pub_dump = publication.model_dump()
        environment = self._environment()
        ref_dump = self._reference(environment).model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e9))
        idx = 0
        with env.begin(write=True) as txn:
            for _strain, rfp, _growth, common in tqdm(_TABLE_1, desc="mormino2022"):
                orf = self._resolve(common)
                category = _RFP_CATEGORY[rfp]
                genotype = Genotype(
                    perturbations=[
                        CrisprInterferencePerturbation(
                            systematic_gene_name=orf,
                            perturbed_gene_name=common,
                            crispr=CrisprConstruct(
                                effector=_EFFECTOR, guide_sequence=None
                            ),
                        )
                    ]
                )
                phenotype = EnvironmentResponsePhenotype(
                    measurement_type=MeasurementType.categorical,
                    category=category,
                    units=_UNITS,
                )
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment,
                    phenotype=phenotype,
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
        env.close()
        log.info("Wrote %d Mormino2022 CRISPRi env-response records to LMDB", idx)

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
    root = osp.join(data_root, "data/torchcell/crispri_mormino2022")
    dataset = CrispriMormino2022Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

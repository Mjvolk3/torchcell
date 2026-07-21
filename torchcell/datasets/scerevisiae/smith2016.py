# torchcell/datasets/scerevisiae/smith2016
# [[torchcell.datasets.scerevisiae.smith2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/smith2016
# Test file: tests/torchcell/datasets/scerevisiae/test_smith2016.py
"""Smith 2016 quantitative CRISPRi chemical-genetic screen (per-guide, per-drug fitness).

Smith JD et al. 2016 (Genome Biol 17:45, doi:10.1186/s13059-016-0900-9;
citation_key ``smithQuantitativeCRISPRInterference2016``) built a regulatable dCas9-Mxi1
CRISPRi platform in *S. cerevisiae* (strain BY4741, single pRS41X CEN/URA3 plasmid: TetR +
an ATc-inducible RPR1-TetO gRNA + constitutive dCas9-Mxi1). gRNA tiling libraries targeting
20 drug-partner genes were grown as POOLS in competitive-growth assays, +/- ATc and +/- a
small-molecule inhibitor, and each guide's abundance tracked by amplicon sequencing to
quantify how repression of its target gene changes fitness in each drug condition. This is a
GUIDE-AWARE chemogenomic fitness dataset (many gRNAs per gene).

RECORD = one (pool x guide x drug-condition) ``EnvironmentResponseExperiment``, one per row
of Additional file 10 (14,463 rows):
- GENOTYPE (per guide): a ``CrisprInterferencePerturbation`` -- target ORF (resolved to
  current R64) repressed by ``dCas9-Mxi1``, ``guide_sequence`` = the 18/20 nt
  ``Specificity_sequence`` spacer (joined from Additional file 4 by guide name), and
  ``library_pool`` = the screen pool (``#Pool``). Keeping the spacer keeps multiple guides
  per gene distinct; keeping the pool keeps the SAME spacer's two independent pool
  measurements distinct (see DUPLICATE-BY-POOL note).
- ENVIRONMENT: the row's ``Drug`` + ``Concentration`` as one ``SmallMoleculePerturbation``
  on SCM-Ura (synthetic complete - uracil) liquid medium, 30 C, aerobic. The 26 conditions
  span 18 inhibitors (10-166.7 uM / 2.2-30 nM) plus the 1% DMSO vehicle control. ATc
  induction is the mechanism that turns the CRISPRi genotype ON; its exact pooled-screen
  concentration is not released ("+/- ATc" only), so it is NOT asserted as an environment
  dose -- it is captured by the phenotype's induced-vs-uninduced definition + the uninduced
  reference (never guessed).
- PHENOTYPE: ``EnvironmentResponsePhenotype`` ``measurement_type=log2_ratio``,
  ``environment_response`` = ``A`` = the ATc-induced fold change (Methods: "ATc-induced fold
  change (A) ... A_ijk = f_ijk+ - f_ijk-", the difference of log2 median-centred guide
  read-count fitness between induced (+ATc) and uninduced (-ATc) cultures in that drug);
  negative = CRISPRi repression is a growth defect (drug-sensitivity). Uncertainty =
  ``var(A)`` (the released variance of the A estimate, a Gamma read-count-resampling
  posterior variance combined as s2_+ + s2_-) stored VERBATIM as
  ``UncertaintyType.variance`` with ``n_samples=1`` (each row is one released A estimate;
  for the 1% DMSO + 20 uM fluconazole conditions var(A) is already the inverse-variance
  combination of the 8/3 replicate experiments, i.e. the variance of the SINGLE released
  estimate), so ``derive_se`` -> SE = sqrt(var(A)/1) = sqrt(var(A)), the estimator's SE.
  Reference = the uninduced (-ATc) baseline: same drug environment, A = 0 (log2 ratio 0).

DATA SOURCE + PROVENANCE (sha256-pinned SI mirror, si/si_data/):
- Additional file 10 ``13059_2016_900_MOESM10_ESM.xlsx`` sheet 'Fitness and Effect Data'
  (14,463 rows x 24 cols; sha256 02962e51...): the per-(pool, guide, drug, concentration) A,
  var(A), and raw fitness columns consumed here.
- Additional file 4 ``13059_2016_900_MOESM4_ESM.xlsx`` sheet 'gRNAs' (1,060 rows;
  sha256 e5eb4e3c...): the guide -> ``Specificity_sequence`` (spacer) join. Every guide name
  maps to a UNIQUE spacer (0 conflicts; the 63 names appearing in two pools carry the same
  spacer), so a name->spacer dict is unambiguous, and all 977 screened guides resolve.

DUPLICATE-BY-POOL (the L1-uniqueness decision, documented not guessed):
272 (guide, drug, concentration) triples appear TWICE -- always the pool pair
(``broad_tiling``, ``gene_tiling_20bp``) sharing a 20 bp perfect-match spacer. These are two
independent pooled measurements (fitness is median-centred WITHIN a pool, so the two A
values differ by mean 0.95 / up to 7.8 log2 units): NOT replicates to average (averaging
would fabricate a value in neither pool), NOT dropped (that discards a released measurement).
So the record grain is one row = one (pool, guide, drug, concentration); ``library_pool``
joins the guide spacer in the strain identity, and the env-response verifier's genotype
signature keys on it (backward-compatible: None for single-pool studies).

GENE RESOLUTION: the 20 targeted genes are current R64 systematic ORFs and all resolve to
self via the genome (0 dropped). Unresolvable targets or guides missing a spacer would be
dropped and counted (0 here), never guessed.
"""

import hashlib
import logging
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
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
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
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1186/s13059-016-0900-9"

_LIBRARY_CITATION_KEY = "smithQuantitativeCRISPRInterference2016"

# Additional file 10: the per-(pool, guide, drug, concentration) fitness + effect table.
_EFFECT_FILENAME = "13059_2016_900_MOESM10_ESM.xlsx"
_EFFECT_SHEET = "Fitness and Effect Data"
_EFFECT_SHA256 = "02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0"
# Additional file 4: the guide -> Specificity_sequence (spacer) map.
_GUIDE_FILENAME = "13059_2016_900_MOESM4_ESM.xlsx"
_GUIDE_SHEET = "gRNAs"
_GUIDE_SHA256 = "e5eb4e3c7856782e36edff5ef55e680cb43fa8e9944bf8e40d7cb7b4f376c1e5"

# dCas9-Mxi1 CRISPRi repressor (Methods: "catalytically inactive ... Cas9 (dCas9) to which
# the Mxi1 transcriptional repressor was fused").
_EFFECTOR = "dCas9-Mxi1"

_UNITS = (
    "ATc-induced fold change A = f(+ATc) - f(-ATc): the difference in log2 median-centred "
    "guide read-count fitness between the induced (+ATc, dCas9-Mxi1 CRISPRi ON) and "
    "uninduced (-ATc) pooled cultures in this drug condition; negative = CRISPRi repression "
    "of the target causes a growth/fitness defect (drug-sensitivity)"
)

# Drug label of the vehicle-only control condition; the released Concentration cell is the
# v/v fraction 0.01, i.e. 1% (Methods: "the control condition (1% DMSO)").
_DMSO = "DMSO"
_DMSO_PERCENT = 1.0

_CONC_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(M|mM|uM|nM)\s*$")
_UNIT_MAP = {
    "M": ConcentrationUnit.molar,
    "mM": ConcentrationUnit.millimolar,
    "uM": ConcentrationUnit.micromolar,
    "nM": ConcentrationUnit.nanomolar,
}


def _concentration(drug: str, raw: Any) -> Concentration:
    """Parse the ``Concentration`` cell into a typed dose (fail loud, never guess)."""
    if drug == _DMSO:
        return Concentration(value=_DMSO_PERCENT, unit=ConcentrationUnit.percent_v_v)
    match = _CONC_RE.match(str(raw))
    if match is None:
        raise ValueError(f"unparseable concentration {raw!r} for drug {drug!r}")
    return Concentration(value=float(match.group(1)), unit=_UNIT_MAP[match.group(2)])


@register_dataset
class CrispriChemgenSmith2016Dataset(ExperimentDataset):
    """Smith 2016 per-guide dCas9-Mxi1 CRISPRi chemical-genetic env x geno dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/crispri_chemgen_smith2016",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for target-ORF -> current-R64 resolution."""
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
        """The two sha256-pinned SI workbooks (effect table + guide->spacer map)."""
        return [_EFFECT_FILENAME, _GUIDE_FILENAME]

    def download(self) -> None:
        """Copy the pinned SI workbooks from the library mirror; verify each sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for filename, digest_expected in (
            (_EFFECT_FILENAME, _EFFECT_SHA256),
            (_GUIDE_FILENAME, _GUIDE_SHA256),
        ):
            dest = osp.join(self.raw_dir, filename)
            if not osp.exists(dest):
                src = osp.join(
                    os.environ["DATA_ROOT"],
                    "torchcell-library",
                    _LIBRARY_CITATION_KEY,
                    "si",
                    "si_data",
                    filename,
                )
                if not osp.exists(src):
                    raise RuntimeError(f"library mirror file not found: {src}")
                shutil.copyfile(src, dest)
            digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
            if digest != digest_expected:
                raise RuntimeError(
                    f"{filename} sha256 mismatch: got {digest}, expected {digest_expected}"
                )
            log.info("Verified %s (sha256 %s)", dest, digest_expected)

    def _resolver(self) -> Callable[[str], str | None]:
        """Target ORF -> current-R64 ORF (self if already an ID, else via alias table)."""
        if self.genome is None:
            raise RuntimeError(
                "CrispriChemgenSmith2016Dataset requires a genome; "
                "inject SCerevisiaeGenome(...)"
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

    def _spacer_map(self) -> dict[str, str]:
        """Guide name -> Specificity_sequence spacer (unique per name; 0 conflicts)."""
        guides = pd.read_excel(
            osp.join(self.raw_dir, _GUIDE_FILENAME), sheet_name=_GUIDE_SHEET
        )
        mapping: dict[str, str] = {}
        for name, spacer in zip(
            guides["Guide_name"], guides["Specificity_sequence"], strict=True
        ):
            mapping[str(name)] = str(spacer)
        return mapping

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Uninduced (-ATc) baseline: no ATc-induced fold change -> A = 0 (log2 ratio 0)."""
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=0.0,
            units=_UNITS,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    @post_process
    def process(self) -> None:
        """Build one env x geno -> A record per (pool, guide, drug, condition); write LMDB."""
        df = pd.read_excel(
            osp.join(self.raw_dir, _EFFECT_FILENAME), sheet_name=_EFFECT_SHEET
        )
        df["A"] = pd.to_numeric(df["A"], errors="coerce")
        df["var(A)"] = pd.to_numeric(df["var(A)"], errors="coerce")
        resolve = self._resolver()
        spacer_map = self._spacer_map()
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        pub_dump = publication.model_dump()

        # Environment + reference are constant per (drug, concentration) condition; cache.
        env_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def prepared(drug: str, conc_raw: Any) -> dict[str, Any]:
            key = (drug, str(conc_raw))
            if key not in env_cache:
                environment = Environment(
                    media=Media(name="SCM-Ura", state="liquid", is_synthetic=True),
                    temperature=Temperature(value=30.0),
                    perturbations=[
                        SmallMoleculePerturbation(
                            compound=resolved_compound(
                                drug, known_proprietary=drug != _DMSO
                            ),
                            concentration=_concentration(drug, conc_raw),
                        )
                    ],
                    aerobicity="aerobic",
                )
                env_cache[key] = {
                    "environment": environment,
                    "ref_dump": self._reference(environment).model_dump(),
                }
            return env_cache[key]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_unresolved = n_no_spacer = n_nan = 0
        unresolved_genes: set[str] = set()
        with env.begin(write=True) as txn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="smith2016"):
                effect = row["A"]
                variance = row["var(A)"]
                if pd.isna(effect) or pd.isna(variance):
                    n_nan += 1
                    continue
                orf = resolve(row["ORF"])
                if orf is None:
                    n_unresolved += 1
                    unresolved_genes.add(str(row["ORF"]))
                    continue
                guide_name = str(row["Guide"])
                spacer = spacer_map.get(guide_name)
                if spacer is None:
                    n_no_spacer += 1
                    continue
                item = prepared(str(row["Drug"]), row["Concentration"])
                genotype = Genotype(
                    perturbations=[
                        CrisprInterferencePerturbation(
                            systematic_gene_name=orf,
                            perturbed_gene_name=str(row["Gene"]),
                            crispr=CrisprConstruct(
                                effector=_EFFECTOR,
                                guide_sequence=spacer,
                                n_guides=1,
                                library_pool=str(row["#Pool"]),
                            ),
                        )
                    ]
                )
                phenotype = EnvironmentResponsePhenotype(
                    measurement_type=MeasurementType.log2_ratio,
                    environment_response=float(effect),
                    environment_response_uncertainty=float(variance),
                    environment_response_uncertainty_type=UncertaintyType.variance,
                    n_samples=1,
                    sample_unit=SampleUnit.pooled,
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
            "Smith2016: wrote %d (pool x guide x drug-condition) records; dropped %d rows "
            "with no A/var(A), %d unresolved-target guides (%d distinct ORFs), %d guides "
            "with no spacer",
            idx,
            n_nan,
            n_unresolved,
            len(unresolved_genes),
            n_no_spacer,
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
    root = osp.join(data_root, "data/torchcell/crispri_chemgen_smith2016")
    dataset = CrispriChemgenSmith2016Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

# torchcell/datasets/scerevisiae/smith2006
# [[torchcell.datasets.scerevisiae.smith2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/smith2006
# Test file: tests/torchcell/datasets/scerevisiae/test_smith2006.py
"""Smith 2006 fatty-acid clear-zone screen (env x geno -> ordinal categorical response).

Smith et al. 2006 (Mol Syst Biol 2:2006.0009, doi:10.1038/msb4100051; PMID 16738555;
PMC1681483) screened the ENTIRE ``matalpha`` haploid viable S. cerevisiae gene-deletion
set (strain BY4742, Resgen/Invitrogen; KanMX deletions) for defects in peroxisomal
fatty-acid beta-oxidation. Two orthogonal readouts on solid agar omnitrays:

- CLEAR-ZONE assay: strains form a clear zone in turbid fatty-acid medium (oleate = YPBO,
  or myristate = YPBM) as they metabolize the fatty acid. Zone size is scored ordinally
  4/3/2/1 (larger / equal / smaller / small-or-absent vs wild type).
- ACETATE GROWTH control (YPBA): peroxisomal beta-oxidation demands a functional
  mitochondrial electron transport chain, so growth on the nonfermentable carbon source
  acetate is a control. Growth is scored 3/2/1 (wild-type / moderate / little-or-no).

This maps onto the WS15 environment-perturbation ontology. ``EnvironmentResponseExperiment``
= single-deletion ``Genotype`` (one ``KanMxDeletionPerturbation`` in the BY4742 reference
background) x aerobic solid-agar ``Environment`` carrying the EDIT: each of the three media
is the base + an added carbon/fatty-acid species, modeled as a ``SmallMoleculePerturbation``
(oleic acid 0.1% w/v, myristic acid 0.125% w/v, acetate 2% w/v). The readout is an ORDINAL
score stored on ``EnvironmentResponsePhenotype.environment_response`` (a float) with a
semantic ``category`` label; ``measurement_type=categorical`` (the schema has no dedicated
ordinal member -- the ordinal numeric lives in ``environment_response``, the qualitative
call in ``category``). One record per (strain x condition) for the three conditions; the
sparse 'Glucose (YEPD)' NG/LG growth-flag column is deliberately SKIPPED (not an ordinal
score). The parental BY4742 strain (wild type) is the per-environment reference.

PROVENANCE / SOURCING (born-digital STAR-Methods quotes, sha256-pinned data + PDF):
- Background: "The entire matalpha haploid viable yeast deletion set from S. cerevisiae
  strain BY 4742 (Resgen/Invitrogen) was assayed for the formation of clear zones ..."
  (Methods, "Myristate screen and generation of the 'fitness' data set"). Captured by
  ``ReferenceGenome(strain="BY4742")`` (haploid); genotype = one KanMxDeletionPerturbation.
- Media / concentrations (verbatim, Methods): YPBA "2% acetate", YPBO "0.1% oleic acid",
  YPBM "0.125% myristic acid" (all percent w/v). Solid agar omnitrays.
- Growth conditions: "Plates were incubated for 3-4 days at 30 C." -> duration_hours=84.0
  (3.5-day midpoint of 3-4 days) and Temperature(value=30). Aerobic.
- Scoring (verbatim, Methods): "Growth was scored as 3, 2 or 1 for patches with wild type,
  moderate or little/no growth, respectively. Clear zone sizes around cell patches were
  scored as 4 for larger than wild type, 3 for wild type, 2 for less than wild type and 1
  for small or not detectable." An undocumented acetate value 2.5 also appears in the
  released table and is KEPT verbatim (mapped to category "intermediate").
- Replication (verbatim, Methods): "The entire deletion set was pinned in quadruplicate on
  YEPD agar ... Colonies were replicated in triplicate onto acetate, oleate or myristate
  agar omnitrays." The TRIPLICATE replicate plates are the independent replicates ->
  ``n_samples=3``, ``sample_unit=biological_replicate``. The quadruplicate pinning is a
  WITHIN-plate technical replication of the source colony (not an independent measurement of
  the readout) and is NOT counted. The released value is a single visual consensus ordinal
  per (strain x condition); no SE/uncertainty is released, so uncertainty fields stay None.

DATA SOURCE (the per-strain score table):
- Supplementary Table 1 = ``msb4100051-s1.xls`` (legacy BIFF .xls; read via ``xlrd``,
  header on 0-based row 23, 4770 strain rows). sha256-pinned below. The file was fetched
  once from the Europe PMC supplementary bundle for PMC1681483 and deposited into the local
  library mirror (``smithExpressionFunctionalProfiling2006/data/``), which is the canonical,
  reproducible source (publisher SI downloads are not reliably scriptable).

GENE-RESOLUTION / COUNT NOTES (documented, not guessed):
- Systematic names are resolved to current SGD R64 ORFs: a name that is already a current
  R64 ID is taken directly; an old systematic name (e.g. YGR272c) is followed through the
  genome alias table to its current ID, but ONLY when the target is not already claimed by a
  direct-ID row. This drops the alias member of a "dubious-ORF-absorbed-into-verified-
  neighbor" collision (e.g. YOR240W aliases to YOR239W, which is present directly as
  ABP140) -- forcing the alias would store a distinct dubious-ORF strain mislabeled as the
  neighbor's deletion, so it is dropped to keep gene identity honest.
- Of 4770 screened strains: 4721 resolve to unique current R64 ORFs; 49 are dropped and
  counted -- 26 non-current/dubious systematic names with no confident mapping, and 23
  alias-resolutions that would collide with a directly-present ORF. Every row carries all
  three condition scores (no blank condition cells), so the record total is
  4721 x 3 = 14163 environment-response experiments.
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
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/msb4100051"
PMID = "16738555"

# Canonical source: sha256-pinned Supplementary Table 1 in the library mirror. Fetched once
# from the Europe PMC supplementary bundle for PMC1681483; the mirror is the reproducible
# source (publisher SI downloads are not reliably scriptable).
_LIBRARY_CITATION_KEY = "smithExpressionFunctionalProfiling2006"
_XLS_FILENAME = "msb4100051-s1.xls"
_XLS_SHA256 = "7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4"

# 0-based header row of the single sheet; data rows below = 4770 strains.
_HEADER_ROW = 23
_SYSTEMATIC_COL = "Systematic Name"
_STANDARD_COL = "Standard Name"

# Triplicate replicate plates (Methods) -> independent replicates; the quadruplicate pinning
# is within-plate technical replication of the source colony and is NOT counted.
_N_SAMPLES = 3

# WT-like baseline: the parental BY4742 strain scores 3 ("wild type") on every readout.
_REFERENCE_CATEGORY = "wild_type"

# Ordinal-score -> semantic category maps (verbatim scoring legend, Methods).
_CLEAR_ZONE_CATEGORY: dict[float, str] = {
    4.0: "enhanced",
    3.0: "wild_type",
    2.0: "reduced",
    1.0: "defective",
}
_GROWTH_CATEGORY: dict[float, str] = {
    3.0: "wild_type",
    2.5: "intermediate",
    2.0: "moderate",
    1.0: "poor",
}

_CLEAR_ZONE_UNITS = (
    "clear-zone size around the cell patch, scored by visual inspection: 4=larger than "
    "wild type, 3=wild type, 2=less than wild type, 1=small or not detectable"
)
_GROWTH_UNITS = (
    "growth of the cell patch on nonfermentable acetate, scored by visual inspection: "
    "3=wild-type, 2=moderate, 1=little/no growth (undocumented 2.5=intermediate kept "
    "verbatim from the released table)"
)

# Per-condition environment spec. Each medium is the base + an added carbon/fatty-acid
# species dosed as a percent-w/v ``SmallMoleculePerturbation``. The clear-zone (oleate,
# myristate) vs growth (acetate) readouts differ only in their ordinal->category legend.
_CONDITION_SPECS: list[dict[str, Any]] = [
    {
        "column": "Oleate (YPBO)",
        "media": "YPBO",
        "compound_name": "oleic acid",
        "concentration": (0.1, "percent_w/v"),
        "category_map": _CLEAR_ZONE_CATEGORY,
        "units": _CLEAR_ZONE_UNITS,
    },
    {
        "column": "Myristae (YPBM)",  # verbatim header typo for "Myristate"
        "media": "YPBM",
        "compound_name": "myristic acid",
        "concentration": (0.125, "percent_w/v"),
        "category_map": _CLEAR_ZONE_CATEGORY,
        "units": _CLEAR_ZONE_UNITS,
    },
    {
        "column": "Acetate (YPBA)",
        "media": "YPBA",
        "compound_name": "acetate",
        "concentration": (2.0, "percent_w/v"),
        "category_map": _GROWTH_CATEGORY,
        "units": _GROWTH_UNITS,
    },
]

_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


@register_dataset
class FattyAcidSmith2006Dataset(ExperimentDataset):
    """Smith 2006 fatty-acid clear-zone env x geno -> ordinal categorical response dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_smith2006",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for systematic-name -> ORF mapping."""
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
        """The mirrored Supplementary Table 1 .xls (per-strain ordinal score table)."""
        return [_XLS_FILENAME]

    def download(self) -> None:
        """Copy the pinned library .xls into raw_dir and verify its sha256.

        The canonical source is the sha256-pinned Supplementary Table 1 in the local library
        mirror (fetched once from the Europe PMC supplementary bundle for PMC1681483).
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _XLS_FILENAME)
        if not osp.exists(dest):
            data_root = os.environ["DATA_ROOT"]
            src = osp.join(
                data_root,
                "torchcell-library",
                _LIBRARY_CITATION_KEY,
                "data",
                _XLS_FILENAME,
            )
            if not osp.exists(src):
                raise RuntimeError(
                    f"library mirror data file not found: {src}. This dataset's source is "
                    f"the sha256-pinned Supplementary Table 1 in the torchcell-library mirror."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _XLS_SHA256:
            raise RuntimeError(
                f"{_XLS_FILENAME} sha256 mismatch: got {digest}, expected {_XLS_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _XLS_SHA256)

    def _read_table(self) -> pd.DataFrame:
        """Read the legacy BIFF .xls Supplementary Table 1 (header on 0-based row 23)."""
        xls_path = osp.join(self.raw_dir, _XLS_FILENAME)
        return pd.read_excel(xls_path, engine="xlrd", header=_HEADER_ROW)

    def _resolver(self, systematic_names: list[str]) -> Callable[[str], str | None]:
        """Build a collision-aware systematic-name -> current-R64-ORF resolver.

        A name already a current R64 ID resolves to itself; an old systematic name follows
        the genome alias table to its current ID ONLY when that target is not already a
        directly-present ORF (so a dubious ORF absorbed into a verified neighbor -- e.g.
        YOR240W -> YOR239W (present as ABP140) -- is dropped, not mislabeled).
        """
        if self.genome is None:
            raise RuntimeError(
                "FattyAcidSmith2006Dataset requires a genome for systematic-name "
                "resolution; inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        df = genome.gene_attribute_table
        ids = set(df["ID"])
        alias_map = genome.alias_to_systematic
        direct_orfs = {n for n in systematic_names if n in ids}

        def resolve(name: str) -> str | None:
            if name in ids:
                return name
            if _SYSTEMATIC_RE.match(name):
                candidates = alias_map.get(name, [])
                if (
                    candidates
                    and candidates[0] in ids
                    and candidates[0] not in direct_orfs
                ):
                    return candidates[0]
            return None

        return resolve

    def _resolve_strains(self, df: pd.DataFrame) -> dict[str, str]:
        """Resolve the strain rows to {current-R64-ORF: standard_name}; log + count drops."""
        systematic_names = [
            str(v).upper().strip() for v in df[_SYSTEMATIC_COL].tolist()
        ]
        standard_names = [str(v).upper().strip() for v in df[_STANDARD_COL].tolist()]
        resolve = self._resolver(systematic_names)
        assert self.genome is not None
        alias_map = self.genome.alias_to_systematic
        ids = set(self.genome.gene_attribute_table["ID"])
        direct_orfs = {n for n in systematic_names if n in ids}

        resolved: dict[str, str] = {}
        n_unresolved = 0
        n_collision = 0
        n_dup = 0
        for sysname, stdname in zip(systematic_names, standard_names):
            orf = resolve(sysname)
            if orf is None:
                candidates = alias_map.get(sysname, [])
                if (
                    _SYSTEMATIC_RE.match(sysname)
                    and candidates
                    and candidates[0] in direct_orfs
                ):
                    n_collision += 1
                else:
                    n_unresolved += 1
                continue
            if orf in resolved:
                n_dup += 1
                continue
            resolved[orf] = stdname
        log.info(
            "Smith2006: %d strains -> %d unique-ORF strains "
            "(dropped %d unresolved + %d alias-collision + %d duplicate)",
            len(systematic_names),
            len(resolved),
            n_unresolved,
            n_collision,
            n_dup,
        )
        return resolved

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """Aerobic solid-agar plate carrying the added carbon/fatty-acid species (percent w/v)."""
        value, unit = spec["concentration"]
        return Environment(
            media=Media(name=spec["media"], state="solid", is_synthetic=False),
            temperature=Temperature(value=30.0),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound(spec["compound_name"]),
                    concentration=Concentration(
                        value=value, unit=ConcentrationUnit(unit)
                    ),
                )
            ],
            aerobicity="aerobic",
            duration_hours=84.0,
        )

    def _reference(
        self, spec: dict[str, Any], environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Parental BY4742 baseline: the wild-type category (ordinal 3) for this readout.

        The numeric ``environment_response`` is left None (the categorical reference): the
        WT ordinal is 3, recorded in ``units``, but the response scale here is ordinal, not
        the 0-centered log-ratio scale, so no numeric baseline is asserted.
        """
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            category=_REFERENCE_CATEGORY,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.biological_replicate,
            units=spec["units"],
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4742"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    def _experiment(
        self,
        *,
        orf: str,
        standard_name: str,
        score: float,
        spec: dict[str, Any],
        environment: Environment,
    ) -> EnvironmentResponseExperiment:
        """Build one env x geno -> ordinal-categorical experiment for (strain, condition)."""
        category = spec["category_map"].get(score)
        if category is None:
            raise RuntimeError(
                f"unmapped score {score!r} in column {spec['column']!r} for {orf}"
            )
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=standard_name
                )
            ]
        )
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            environment_response=score,
            category=category,
            n_samples=_N_SAMPLES,
            sample_unit=SampleUnit.biological_replicate,
            units=spec["units"],
        )
        return EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    @post_process
    def process(self) -> None:
        """Parse the ordinal score table into per-(strain, condition) records; write LMDB."""
        df = self._read_table()
        resolved = self._resolve_strains(df)
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        pub_dump = publication.model_dump()

        # Pre-build per-condition environment + reference (constant across strains).
        prepared: list[dict[str, Any]] = []
        for spec in _CONDITION_SPECS:
            environment = self._environment(spec)
            reference = self._reference(spec, environment)
            prepared.append(
                {
                    "spec": spec,
                    "environment": environment,
                    "ref_dump": reference.model_dump(),
                }
            )

        systematic_names = [
            str(v).upper().strip() for v in df[_SYSTEMATIC_COL].tolist()
        ]
        # Resolve per-row so each row maps to its ORF (or is skipped if dropped).
        resolve = self._resolver(systematic_names)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        seen_orf: set[str] = set()
        with env.begin(write=True) as txn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="smith2006"):
                sysname = str(row[_SYSTEMATIC_COL]).upper().strip()
                orf = resolve(sysname)
                if orf is None or orf in seen_orf:
                    continue
                seen_orf.add(orf)
                standard_name = resolved[orf]
                for item in prepared:
                    spec = item["spec"]
                    raw = row[spec["column"]]
                    if pd.isna(raw):
                        continue
                    score = float(raw)
                    experiment = self._experiment(
                        orf=orf,
                        standard_name=standard_name,
                        score=score,
                        spec=spec,
                        environment=item["environment"],
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
        log.info("Wrote %d Smith2006 environment-response experiments to LMDB", idx)

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
    root = osp.join(data_root, "data/torchcell/env_chemgen_smith2006")
    dataset = FattyAcidSmith2006Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

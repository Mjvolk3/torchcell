# torchcell/datasets/scerevisiae/costanzo2021
# [[torchcell.datasets.scerevisiae.costanzo2021]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/costanzo2021
# Test file: tests/torchcell/datasets/scerevisiae/test_costanzo2021.py
"""Costanzo 2021 condition-SGA single-mutant fitness (env x geno -> differential fitness).

Costanzo et al. 2021 (Science 372, eabf8424; doi:10.1126/science.abf8424; PMC9132594),
"Environmental robustness of the global yeast genetic interaction network," measured
single- and double-mutant fitness across 14 diverse environmental conditions plus a
matched reference condition. This loader ingests the SINGLE-MUTANT FITNESS panel
(Data File S1, sheet "Diff. Mutant fitness_Conditions") -- the same table Nadal-Ribelles
2025 reuses as its stressor-fitness panel ("Costanzo Supplementary Table 1, 14 conditions
x 4429 genotypes").

READOUT -- the stored score is the DIFFERENTIAL mutant fitness
(``measurement_type=differential_fitness``): *"To obtain condition specific fitness
estimates, we computed the difference in colony size measured in a particular test
condition versus the matched reference condition for each mutant"* (Methods, PMC9132594).
It is a SIGNED value (negative = condition-hypersensitive, 0 = fitness unchanged vs the
reference condition), so the parent-strain reference is 0 (L3 reference_zero). The absolute
single-mutant fitness (sheet "Mutant Fitness_Conditions", ~1.0 = wild-type-like) is NOT
stored -- a ratio centred at 1 would violate the reference-zero invariant; the differential
is the condition-specific response this schema is built for.

REPLICATE STRUCTURE -> n_samples: *"Colony size measurements of SGA deletion and TS array
mutant strains were based on an average of 3 replicate control screens conducted per each
of 14 test conditions as well as the reference condition at 26 C"* (Methods). Hence
``n_samples=3``, ``sample_unit=screen`` (the SGA independent unit is the SCREEN, not the
colony -- colonies are pseudoreplicates, per the schema's ``SampleUnit`` docstring). No
per-strain SD is released in Data File S1, so no SE is stored (not overclaimed).

GENOTYPE -- the collection mixes two strain classes (``Strain ID`` prefix ``dma`` / ``tsa``).
Costanzo 2021 is condition-SGA, so both use the SGA perturbation leaves (which carry the SGA
``strain_id``), matching how Costanzo 2016 SGA data is modelled:
- Non-essential genes: KanMX deletion (``dma*``) -> ``SgaKanMxDeletionPerturbation``.
- Essential genes: temperature-sensitive allele (``tsa*``, allele in the "Allele (Essential
  genes only)" column, e.g. ``act1-101``) -> ``SgaTsAllelePerturbation``. Essential genes are
  screened as ALLELIC SERIES: one systematic ORF (e.g. ACT1/YFL039C) carries up to 18
  distinct ts alleles, each a separate strain. These are distinct genotypes (distinct
  ``perturbed_gene_name`` alleles), so the L1 uniqueness check -- which keys on the STRAIN
  (the genotype signature), not the bare gene -- treats them as distinct records, not
  duplicates. The SGA ``strain_id`` is retained on the perturbation as source provenance.

ENVIRONMENT -- 14 conditions from the "Conditions" sheet. 12 are small molecules
(``SmallMoleculePerturbation``); Galactose is a CARBON-SOURCE change
(``EnvironmentPhysicalPerturbation(factor=carbon_source)``); Sorbitol is a single added
osmoticum modelled as a ``SmallMoleculePerturbation`` (schema steer: a single named
compound that IS the edit -> small molecule, cf. NaCl). Temperature is 26 C throughout (the
stated reference-condition temperature; the 14 stressors are chemical/carbon/osmotic, run at
the TS-permissive 26 C) and carried on ``Environment.temperature`` (M2), never a perturbation.

CONCENTRATION provenance (verbatim from the Data File S1 "Conditions" sheet; mg/mL recorded
as the equal g/L unit): several deposited values are recorded verbatim and FLAGGED as
chemically implausible in the source (bortezomib "1300 mM"; actinomycin D "20 mM";
geldanamycin "10 mM" -- likely a unit mislabel in the SI). Two conditions are bare fractions
interpreted as percent (galactose "0.02" -> 2% w/v standard galactose carbon source; MMS
"0.0001" -> 0.01% v/v). The concentration is Environment METADATA and does not enter the
readout; the deposited SI is authoritative and the anomalies are flagged for review.

DATA SOURCE (manual-once -> mirror; Science SI is bot-blocked / 403 like Costanzo 2016):
Data File S1 ``Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx`` deposited to the
raw mirror and sha256-pinned. ORF names are validated against the SGD R64 universe (ORF +
RNA-coding FASTA headers); 23 old/merged names not in R64 are DROPPED (never guessed) and
logged. Final: 4406 strains x 14 conditions minus 366 empty cells = 61,318 records.
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
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
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1126/science.abf8424"

_S1_FILENAME = "Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx"
_S1_SHA256 = "f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad"
_FITNESS_SHEET = "Diff. Mutant fitness_Conditions"

# S288C R64 reference gene universe (systematic ORF + RNA-coding names) for R64 resolution.
_SGD_GENE_FASTAS = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
)

_UNIT = {
    "mM": ConcentrationUnit.millimolar,
    "nM": ConcentrationUnit.nanomolar,
    "uM": ConcentrationUnit.micromolar,
    "M": ConcentrationUnit.molar,
    "g/L": ConcentrationUnit.g_per_l,
    "percent_w_v": ConcentrationUnit.percent_w_v,
    "percent_v_v": ConcentrationUnit.percent_v_v,
}

# 14 conditions keyed by the (whitespace-stripped) sheet column name. ``kind``:
#   "sm"     -> SmallMoleculePerturbation(compound, concentration)
#   "carbon" -> EnvironmentPhysicalPerturbation(factor=carbon_source, agent, magnitude)
# ``conc`` = (value, unit-token). ``si_raw`` / ``flag`` = provenance notes recorded in units.
_CONDITIONS: list[dict[str, Any]] = [
    {
        "col": "Actinomycin D",
        "name": "actinomycin D",
        "kind": "sm",
        "conc": (20.0, "mM"),
        "flag": "SI '20 mM' implausibly high for actinomycin D",
    },
    {"col": "Benomyl", "name": "benomyl", "kind": "sm", "conc": (30.0, "g/L")},
    {
        "col": "Boretzeomib",
        "name": "bortezomib",
        "kind": "sm",
        "conc": (1300.0, "mM"),
        "flag": "SI '1300 mM' chemically implausible for bortezomib (normally nM-uM)",
    },
    {"col": "Caspofungin", "name": "caspofungin", "kind": "sm", "conc": (0.1, "g/L")},
    {
        "col": "Concanmycin A",
        "name": "concanamycin A",
        "kind": "sm",
        "conc": (100.0, "nM"),
    },
    {
        "col": "Cycloheximide",
        "name": "cycloheximide",
        "kind": "sm",
        "conc": (0.1, "g/L"),
    },
    {"col": "Fluconozole", "name": "fluconazole", "kind": "sm", "conc": (16.0, "g/L")},
    {
        "col": "Galactose",
        "name": "galactose",
        "kind": "carbon",
        "conc": (2.0, "percent_w_v"),
        "si_raw": "SI '0.02' fraction -> 2% w/v standard galactose carbon source",
    },
    {
        "col": "Geldenamycin",
        "name": "geldanamycin",
        "kind": "sm",
        "conc": (10.0, "mM"),
        "flag": "SI '10 mM' implausibly high for geldanamycin",
    },
    {
        "col": "MMS",
        "name": "methyl methanesulfonate",
        "kind": "sm",
        "conc": (0.01, "percent_v_v"),
        "si_raw": "SI '0.0001' fraction -> 0.01% v/v",
    },
    {"col": "Monensin", "name": "monensin", "kind": "sm", "conc": (50.0, "g/L")},
    {"col": "Rapamycin", "name": "rapamycin", "kind": "sm", "conc": (100.0, "nM")},
    {"col": "Sorbitol", "name": "sorbitol", "kind": "sm", "conc": (1.0, "M")},
    {"col": "Tunicamycin", "name": "tunicamycin", "kind": "sm", "conc": (1.0, "g/L")},
]

_REFERENCE_TEMPERATURE_C = 26.0
_BASE_MEDIUM = "SGA final selection medium (synthetic, agar)"

MEASUREMENT_UNITS = (
    "differential mutant fitness = (normalized colony-size fitness in the test condition) - "
    "(matched reference condition at 26 C), Costanzo 2021 condition-SGA; signed, negative = "
    "condition-hypersensitive, 0 = unchanged; mean of 3 replicate screens (no per-strain SE "
    "released)"
)


def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for rel in _SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


@register_dataset
class EnvChemgenCostanzo2021Dataset(ExperimentDataset):
    """Costanzo 2021 condition-SGA: env x (deletion | ts-allele) -> differential fitness."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_costanzo2021",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset. ORFs are already systematic, so no genome is required."""
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
        """The deposited Data File S1 spreadsheet required before processing."""
        return [_S1_FILENAME]

    def download(self) -> None:
        """Verify the manually deposited Data File S1 (Science SI is 403/bot-blocked).

        Science supplementary downloads are not scriptable (bot-blocked, like Costanzo
        2016); the file is deposited once to the raw mirror. Verify its pinned sha256.
        """
        dest = osp.join(self.raw_dir, _S1_FILENAME)
        if not osp.exists(dest):
            raise RuntimeError(
                f"{_S1_FILENAME} not found in {self.raw_dir}. Costanzo 2021 Science SI is "
                "not scriptable (403); deposit Data File S1 manually from "
                "https://www.science.org/doi/10.1126/science.abf8424 (Data File S1) then "
                "rebuild (sha256 verified)."
            )
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _S1_SHA256:
            raise RuntimeError(
                f"{_S1_FILENAME} sha256 mismatch: got {digest}, expected {_S1_SHA256}"
            )

    def _condition_units(self, spec: dict[str, Any]) -> str:
        """Per-condition units string = the shared definition plus any provenance note."""
        note = spec.get("flag") or spec.get("si_raw")
        return MEASUREMENT_UNITS if note is None else f"{MEASUREMENT_UNITS}; {note}"

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """Build the treated 26 C environment carrying this condition's edit."""
        value, unit_token = spec["conc"]
        concentration = Concentration(value=value, unit=_UNIT[unit_token])
        media = Media(name=_BASE_MEDIUM, state="solid", is_synthetic=True)
        temperature = Temperature(value=_REFERENCE_TEMPERATURE_C)
        if spec["kind"] == "carbon":
            perturbation: Any = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=concentration,
                agent=Compound(name=spec["name"]),
            )
        else:
            perturbation = SmallMoleculePerturbation(
                compound=Compound(name=spec["name"]), concentration=concentration
            )
        return Environment(
            media=media,
            temperature=temperature,
            perturbations=[perturbation],
            aerobicity="aerobic",
        )

    def _reference_dump(self) -> dict[str, Any]:
        """One shared reference: the matched reference condition (26 C, no edit),
        differential fitness 0. Compound-independent (the differential baseline is
        identical for every condition), so the whole dataset has ONE reference.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain="SGA reference (BY4741-derived deletion / TS array background)",
            ),
            environment_reference=Environment(
                media=Media(name=_BASE_MEDIUM, state="solid", is_synthetic=True),
                temperature=Temperature(value=_REFERENCE_TEMPERATURE_C),
                aerobicity="aerobic",
            ),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.differential_fitness,
                environment_response=0.0,
                units=MEASUREMENT_UNITS,
            ),
        ).model_dump()

    def _genotype(
        self, orf: str, gene_name: str | None, allele: str | None, strain_id: str
    ) -> Genotype:
        """SGA strains: essential -> ts allele; non-essential -> KanMX deletion.

        Both carry the SGA ``strain_id`` (Costanzo 2021 is condition-SGA, same assay family
        as Costanzo 2016), so the Sga* perturbation leaves are the correct types.
        """
        if allele is not None:
            return Genotype(
                perturbations=[
                    SgaTsAllelePerturbation(
                        systematic_gene_name=orf,
                        perturbed_gene_name=allele,
                        strain_id=strain_id,
                    )
                ]
            )
        return Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=orf,
                    perturbed_gene_name=gene_name if gene_name else orf,
                    strain_id=strain_id,
                )
            ]
        )

    @post_process
    def process(self) -> None:
        """Parse the differential-fitness sheet into records; write LMDB."""
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        sgd_genes = _load_sgd_genes(data_root)
        pub_dump = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}").model_dump()
        ref_dump = self._reference_dump()

        frame = pd.read_excel(
            osp.join(self.raw_dir, _S1_FILENAME), sheet_name=_FITNESS_SHEET
        )
        col_by_key = {c.strip().lower(): c for c in frame.columns}
        conditions = [
            {
                **spec,
                "sheet_col": col_by_key[spec["col"].strip().lower()],
                "environment": self._environment(spec).model_dump(),
                "units": self._condition_units(spec),
            }
            for spec in _CONDITIONS
        ]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        dropped: set[str] = set()
        with env.begin(write=True) as txn:
            for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Costanzo2021"):
                orf = str(row["Systematic Name"]).strip()
                if orf not in sgd_genes:
                    dropped.add(orf)
                    continue
                gene_name = row["Gene Name"]
                gene_name = None if pd.isna(gene_name) else str(gene_name).strip()
                allele = row["Allele (Essential genes only)"]
                allele = None if pd.isna(allele) else str(allele).strip()
                strain_id = str(row["Strain ID"]).strip()
                genotype = self._genotype(orf, gene_name, allele, strain_id)
                for spec in conditions:
                    value = row[spec["sheet_col"]]
                    if pd.isna(value):
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=spec["environment"],
                        phenotype=EnvironmentResponsePhenotype(
                            measurement_type=MeasurementType.differential_fitness,
                            environment_response=float(value),
                            n_samples=3,
                            sample_unit=SampleUnit.screen,
                            units=spec["units"],
                        ),
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
        log.info(
            "Wrote %d Costanzo2021 records; dropped %d non-R64 ORF names: %s",
            idx,
            len(dropped),
            sorted(dropped),
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_costanzo2021")
    dataset = EnvChemgenCostanzo2021Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()

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
(``measurement_type=differential_fitness``, ``assay_type=colony_size_array``), defined by
``_READOUT_DEFINITION``. It is a SIGNED value (negative = condition-hypersensitive, 0 =
fitness unchanged vs the reference condition), so the parent-strain reference is 0 (L3
reference_zero). The absolute single-mutant fitness (sheet "Mutant Fitness_Conditions",
~1.0 = wild-type-like) is NOT stored -- a ratio centred at 1 would violate the
reference-zero invariant; the differential is the condition-specific response this schema
is built for.

REPLICATE STRUCTURE -> ``n_samples=3``, ``sample_unit=screen`` (``_N_SAMPLES``). The SGA
independent unit is the SCREEN, not the colony -- colonies are pseudoreplicates, per the
schema's ``SampleUnit`` docstring. No per-strain SD is released in Data File S1, so no SE
is stored (not overclaimed); the paper's own variance statement (``_VARIANCE_NOTE``) says
the variance was bootstrapped but does not release it.

TEMPERATURE -- 26 C, and it is a DERIVATION, not a quote. The Methods sentence that names
26 C (``_TEMPERATURE``'s quote) attaches the clause "at 26 C" to a list that ends with the
reference condition; it does not separately state the temperature of the 14 test-condition
screens. The design sentence (``_MATCHED_DESIGN``) says the three copies of one array come
from a single screen, so the test copies share the reference copy's incubator, and the TS
array requires a permissive temperature. 26.0 is therefore recorded on every record with
that derivation stated here, in the verifier ``method`` string and in the dendron note --
and it matches served Kuzmin 2018's ``Temperature(value=26)``, so the temperature node
joins.

ENVIRONMENT -- 14 conditions from the "Conditions" sheet, on the SHARED SGA medium:
- 13 are added small molecules (``SmallMoleculePerturbation``) on
  ``media.SGA_DM_SELECTION``, the identical object served Costanzo 2016 uses.
- Galactose is NOT an added compound: the paper calls it "an alternative carbon source"
  (``_ALTERNATIVE_CARBON``), i.e. a REPLACEMENT of the medium's glucose, so it is the
  derived medium ``media.SGA_DM_SELECTION_GALACTOSE`` with NO perturbation. Modelling it
  as an additive perturbation asserted glucose AND galactose in one flask and hid the
  compound inside ``serialized_data`` (the adapter projects no compound for a physical
  perturbation).
Temperature is carried on ``Environment.temperature`` (M2), never a perturbation.
``Environment.duration_hours`` is a typed ``ProvenanceGap``: the source's own reference
sheet distinguishes "3 Day Incubation" from "5 Day Incubation" and the differential's
choice between them is in the un-mirrored Science SI.

CONCENTRATION provenance -- every dose is a ``SourcedValue`` quoting the verbatim cell of
the Data File S1 "Conditions" sheet (``_CONDITIONS[*]["dose"]``); mg/mL is recorded as the
numerically equal g/L unit. Three deposited values are chemically implausible (bortezomib
"1300 mM", actinomycin D "20 mM", geldanamycin "10 mM" -- likely SI unit mislabels) and two
are bare fractions read as percent (galactose "0.02" -> 2% w/v, MMS "0.0001" -> 0.01% v/v);
each carries that reading in its ``SourcedValue.note``. The anomaly text used to be
appended to the PHENOTYPE's ``units`` string, which made one measurement type carry six
different definitions and wrote a fact about the environment into the phenotype; ``units``
is now the single shared ``MEASUREMENT_UNITS``.

SOLVENT -- row 1 of the "Conditions" sheet reads "Reference condition + solvent", so a
vehicle was used, but the Science supplementary PDF that would name it is not in the
library mirror. ``SmallMoleculePerturbation.solvent`` is therefore ``None`` on every
record; it is not a ``ProvenanceGapMixin``, so the absence is recorded here and in the
dendron note rather than typed, and depositing the Science SI is a flagged follow-up.

GENOTYPE -- the collection mixes two strain classes (``Strain ID`` prefix ``dma`` / ``tsa``).
Costanzo 2021 is condition-SGA, so both use the SGA perturbation leaves (which carry the SGA
``strain_id``), matching how Costanzo 2016 SGA data is modelled:
- Non-essential genes: KanMX deletion (``dma*``) -> ``SgaKanMxDeletionPerturbation``.
- Essential genes: temperature-sensitive allele (``tsa*``, allele in the "Allele (Essential
  genes only)" column, e.g. ``act1-101``) -> ``SgaTsAllelePerturbation``. Essential genes are
  screened as ALLELIC SERIES: one systematic ORF (e.g. ACT1/YFL039C) carries up to 18
  distinct ts alleles, each a separate strain. These are distinct genotypes (distinct
  ``perturbed_gene_name`` alleles), so the L1 uniqueness check -- which keys on the STRAIN
  (the genotype signature), not the bare gene -- treats them as distinct records.
The scored strain is formally a double mutant (a neutral-locus natMX query crossed into the
array strain, ``_NEUTRAL_QUERY``); the neutral marker is NOT put in the genotype, which is
the same simplification served Costanzo 2016 makes, so the two datasets stay joinable.

GENE RESOLUTION -- every ``Systematic Name`` goes through the SHARED, layered
``SCerevisiaeGenome.resolve_gene_name`` instead of a raw-FASTA header membership test.
The FASTA test both dropped 18 strains it should have kept (old systematic names that
SGD RENAMED, e.g. YAR044W -> YAR042W) and kept 10 it should have dropped (blocked reading
frames and pseudogenes, which are valid R64 features but not genes). Retention: CURRENT and
RENAMED are kept (stored under the CURRENT systematic name, with the sheet's own name kept
as the strain's ``perturbed_gene_name``); NON_GENE_FEATURE and RETIRED are dropped,
counted, and written to ``dropped_records.json`` beside ``processed/``.

DATA SOURCE (manual-once -> mirror; Science SI is bot-blocked, verified 403 + Cloudflare
challenge, like Costanzo 2016): Data File S1
``Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx`` is deposited to
``$DATA_ROOT/torchcell-raw/costanzoEnvironmentalRobustnessGlobal2021/`` and sha256-pinned.
Final: 4414 strains (4396 CURRENT + 18 RENAMED) x 14 conditions minus empty cells =
61,430 records.
"""

import hashlib
import json
import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SGA_DM_SELECTION, SGA_DM_SELECTION_GALACTOSE
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1126/science.abf8424"
CITATION_KEY = "costanzoEnvironmentalRobustnessGlobal2021"

# Two provenance anchors, in two different mirrors.
#   ``paper.md``  -- the MinerU OCR of the publisher PDF, in the LIBRARY mirror
#                    ($DATA_ROOT/torchcell-library/<citation_key>/). Methods quotes.
#   the xlsx      -- Data File S1, in the RAW mirror
#                    ($DATA_ROOT/torchcell-raw/<citation_key>/data/). Dose cells.
# ``SourcedValue.source_path`` resolves against ONE root, so only the paper.md-anchored
# values are auditable with ``audit_sourced_value(sv, library_root)``; the xlsx-anchored
# ones name the raw mirror and are audited against that root.
_PAPER_MD = "paper.md"
_PAPER_MD_SHA256 = "ba22973ed0c53c00c37bcfb9f659d3b0373c451a3f7633158afae274035559fb"
_S1_FILENAME = "Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx"
_S1_SHA256 = "f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad"
_S1_RAW_RELPATH = f"data/{_S1_FILENAME}"
_FITNESS_SHEET = "Diff. Mutant fitness_Conditions"
_CONDITIONS_SHEET = "Conditions"

_DROPPED_FILENAME = "dropped_records.json"


def _paper_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced from the mirrored OCR of the paper's Methods."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=_PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (library mirror)",
            page="Science 372 eabf8424, Materials and methods",
        ),
    )


def _si_sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced from a verbatim cell of the Data File S1 'Conditions' sheet."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=_S1_RAW_RELPATH,
            citation_key=CITATION_KEY,
            sha256=_S1_SHA256,
            method="pandas.read_excel (raw mirror)",
            page=f"sheet {_CONDITIONS_SHEET!r}, column 'Concentration'",
        ),
    )


_N_SAMPLES = _paper_sv(
    3,
    "Colony size measurements of SGA deletion and TS array mutant strains were "
    "based on an average of three replicate control screens conducted per each of "
    "14 test conditions as well as the reference condition at $2 6 ^ { \\circ } "
    "\\mathrm { C } .$",
    note="the independent unit is the SCREEN (colonies within a screen are "
    "pseudoreplicates), hence sample_unit=screen",
)

_TEMPERATURE = _paper_sv(
    26.0,
    "Colony size measurements of SGA deletion and TS array mutant strains were "
    "based on an average of three replicate control screens conducted per each of "
    "14 test conditions as well as the reference condition at $2 6 ^ { \\circ } "
    "\\mathrm { C } .$",
    note="DERIVATION, not a quote: the clause 'at 26 C' closes a list that ends with "
    "the reference condition and the sentence does not separately state the 14 test "
    "screens' temperature. 26.0 is applied to all 15 conditions because the three "
    "copies of an array come from ONE screen (see _MATCHED_DESIGN) and so share an "
    "incubator, and because the TS array needs a permissive temperature; served "
    "Kuzmin 2018 carries the same Temperature(value=26)",
)

_MATCHED_DESIGN = _paper_sv(
    "one screen, three copies",
    "every double-mutant array generated from a singlequery SGA screen was copied "
    "three times. One copy was grown in the standard SGA reference condition, "
    "whereas the two other copies were each grown in different conditional media",
    note="the matched-copy design behind the 26 C derivation and behind the single "
    "shared reference (the standard SGA reference condition) used by every record",
)

_ASSAY = _paper_sv(
    AssayType.colony_size_array,
    "we applied our colony size scoring method (19) to a set of control SGA screens",
)

_READOUT_DEFINITION = _paper_sv(
    MeasurementType.differential_fitness,
    "To obtain condition-specific fitness estimates, we computed the difference in "
    "colony size measured in a particular test condition versus the matched reference "
    "condition for each mutant.",
)

_VARIANCE_NOTE = _paper_sv(
    None,
    "bootstrapped means, instead of medians, across replicates were used in variance "
    "estimation and final fitness values",
    note="the variance was estimated but is NOT released per strain in Data File S1, "
    "so no environment_response_uncertainty / SE is stored",
)

_NEUTRAL_QUERY = _paper_sv(
    "natMX at a neutral locus",
    "a query strain carrying a natMX marker inserted at a neutral genomic locus was "
    "crossed to the kanMX-marked DMA",
    note="the scored strain is formally a double mutant; the neutral natMX marker is "
    "deliberately left out of the Genotype, matching served Costanzo 2016",
)

_ALTERNATIVE_CARBON = _paper_sv(
    "galactose replaces glucose",
    "We examined 14 diverse conditions, including an alternative carbon source, "
    "osmotic stress, genotoxic stress, and 11 bioactive compounds",
    note="'an alternative carbon source' is a REPLACEMENT, which is why galactose is "
    "the derived medium SGA_DM_SELECTION_GALACTOSE rather than a compound added on "
    "top of a medium that still contains 2% glucose",
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

# The 14 conditions, keyed by the (whitespace-stripped, lower-cased) sheet column name.
#   ``kind="sm"``     -> a SmallMoleculePerturbation on media.SGA_DM_SELECTION.
#   ``kind="medium"`` -> no perturbation; the condition IS the derived medium.
# ``dose`` is a SourcedValue whose quote is the verbatim "Concentration" cell and whose
# value is ``(number, unit token)``; every unit conversion or reading is in its note.
_CONDITIONS: list[dict[str, Any]] = [
    {
        "col": "Actinomycin D",
        "name": "actinomycin D",
        "kind": "sm",
        "dose": _si_sv(
            (20.0, "mM"),
            "20 mM",
            note="recorded verbatim and FLAGGED: 20 mM is implausibly high for "
            "actinomycin D (normally nM-uM); the deposited SI is authoritative and "
            "the anomaly is carried forward rather than silently corrected",
        ),
    },
    {
        "col": "Benomyl",
        "name": "benomyl",
        "kind": "sm",
        "dose": _si_sv(
            (30.0, "g/L"), "30 mg/mL", note="mg/mL recorded as the equal g/L unit"
        ),
    },
    {
        "col": "Boretzeomib",
        "name": "bortezomib",
        "kind": "sm",
        "dose": _si_sv(
            (1300.0, "mM"),
            "1300 mM",
            note="recorded verbatim and FLAGGED: chemically implausible for "
            "bortezomib (normally nM-uM); the column header's 'Boretzeomib' spelling "
            "is the sheet's own",
        ),
    },
    {
        "col": "Caspofungin",
        "name": "caspofungin",
        "kind": "sm",
        "dose": _si_sv(
            (0.1, "g/L"), "0.1 mg/mL ", note="mg/mL recorded as the equal g/L unit"
        ),
    },
    {
        "col": "Concanmycin A",
        "name": "concanamycin A",
        "kind": "sm",
        "dose": _si_sv((100.0, "nM"), "100 nM "),
    },
    {
        "col": "Cycloheximide",
        "name": "cycloheximide",
        "kind": "sm",
        "dose": _si_sv(
            (0.1, "g/L"), "0.1 mg/mL ", note="mg/mL recorded as the equal g/L unit"
        ),
    },
    {
        "col": "Fluconozole",
        "name": "fluconazole",
        "kind": "sm",
        "dose": _si_sv(
            (16.0, "g/L"), "16 mg/ml ", note="mg/mL recorded as the equal g/L unit"
        ),
    },
    {
        "col": "Galactose",
        "name": "galactose",
        "kind": "medium",
        "media": SGA_DM_SELECTION_GALACTOSE,
        "dose": _si_sv(
            (2.0, "percent_w_v"),
            "0.02",
            note="a bare fraction with no unit in the SI, read as 2% w/v (the standard "
            "SGA galactose carbon source). The number is a property of the derived "
            "MEDIUM (media.SGA_DM_SELECTION_GALACTOSE carries the same quote on its "
            "galactose component), not of a perturbation, so it is not re-asserted on "
            "the record",
        ),
    },
    {
        "col": "Geldenamycin",
        "name": "geldanamycin",
        "kind": "sm",
        "dose": _si_sv(
            (10.0, "mM"),
            "10 mM",
            note="recorded verbatim and FLAGGED: implausibly high for geldanamycin",
        ),
    },
    {
        "col": "MMS",
        "name": "methyl methanesulfonate",
        "kind": "sm",
        "dose": _si_sv(
            (0.01, "percent_v_v"),
            "0.0001",
            note="a bare fraction with no unit in the SI, read as 0.01% v/v (the "
            "standard SGA MMS dose); the unit is a reading, not a quote",
        ),
    },
    {
        "col": "Monensin",
        "name": "monensin",
        "kind": "sm",
        "dose": _si_sv(
            (50.0, "g/L"), "50 mg/ml ", note="mg/mL recorded as the equal g/L unit"
        ),
    },
    {
        "col": "Rapamycin",
        "name": "rapamycin",
        "kind": "sm",
        "dose": _si_sv((100.0, "nM"), "100 nM"),
    },
    {
        "col": "Sorbitol",
        "name": "sorbitol",
        "kind": "sm",
        "dose": _si_sv(
            (1.0, "M"),
            "1M",
            note="a single named osmoticum IS the edit, so it is a "
            "SmallMoleculePerturbation rather than a PhysicalFactor.osmolarity (cf. NaCl)",
        ),
    },
    {
        "col": "Tunicamycin",
        "name": "tunicamycin",
        "kind": "sm",
        "dose": _si_sv(
            (1.0, "g/L"), "1 mg/ml", note="mg/mL recorded as the equal g/L unit"
        ),
    },
]

MEASUREMENT_UNITS = (
    "differential mutant fitness = (normalized colony-size fitness in the test condition) - "
    "(matched reference condition), Costanzo 2021 condition-SGA; signed, negative = "
    "condition-hypersensitive, 0 = unchanged; mean of 3 replicate screens (no per-strain SE "
    "released)"
)

#: Retention rule: a source ORF is kept only when it resolves to a LIVE R64 gene.
_KEPT_STATUSES = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})

DROP_RULE = (
    "Systematic Name resolves through SCerevisiaeGenome.resolve_gene_name to a status "
    "other than CURRENT or RENAMED (a NON_GENE_FEATURE such as a blocked reading frame "
    "or pseudogene, or a name RETIRED from R64-4-1); the strain and all of its condition "
    "cells are dropped"
)


class DroppedStrain(BaseModel):
    """One source ORF the retention rule removed, with what it resolved to."""

    source_name: str = Field(description="the 'Systematic Name' cell, verbatim")
    status: str = Field(description="GeneNameStatus the shared resolver returned")
    resolved_to: str | None = Field(
        default=None, description="what the resolver mapped it to, when anything"
    )
    feature_type: str | None = Field(
        default=None, description="GFF feature type for a NON_GENE_FEATURE"
    )
    n_records: int = Field(
        description="non-empty condition cells lost with this strain"
    )


class DropLog(BaseModel):
    """The build's retention accounting, written beside ``processed/``."""

    dataset: str
    rule: str
    n_source_strains: int
    n_kept_strains: int
    n_kept_records: int
    n_dropped_strains: int
    n_dropped_records: int
    dropped_by_status: dict[str, int] = Field(default_factory=dict)
    dropped: list[DroppedStrain] = Field(default_factory=list)


RAW_MIRROR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The manual recipe that produced the deposited bytes, recorded as the
#: ``retrieval_command`` of a typed ``RetrievalMethod.manual_browser`` record: the
#: retrieval is un-scriptable, but it is not unknown, so the manifest is
#: ``provenance_complete=True`` and the recipe is what a rebuild re-runs by hand.
SCIENCE_SI_URL = "https://www.science.org/doi/10.1126/science.abf8424"

MANUAL_RECIPE = (
    "manual browser download -- science.org returns HTTP 403 behind a Cloudflare "
    "challenge to any client (verified 2026-09-12), so: open "
    f"{SCIENCE_SI_URL} in a signed-in browser, "
    "follow 'Supplementary Materials', download 'Data file S1' "
    f"({_S1_FILENAME}), and verify sha256 {_S1_SHA256}"
)


def raw_mirror_dir(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-raw/costanzoEnvironmentalRobustnessGlobal2021``."""
    return osp.join(data_root or os.environ["DATA_ROOT"], RAW_MIRROR_REL)


def deposit_raw_mirror(
    *, source_xlsx: str, retrieved_at: str, data_root: str | None = None
) -> str:
    """Copy the ONE consumed file into the raw mirror and write its ``manifest.json``.

    Idempotent by sha256: an existing file with the recorded hash is left alone, a
    differing one raises. Only Data File S1 is deposited -- it is the single file this
    loader's first successful build consumed; data files S2-S5 sit beside it in the dev
    raw dir but no loader reads them.
    """
    root = Path(raw_mirror_dir(data_root))
    dest = root / _S1_RAW_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(Path(source_xlsx))
    if digest != _S1_SHA256:
        raise RuntimeError(
            f"{source_xlsx} sha256 {digest} != pinned {_S1_SHA256}; refusing to deposit"
        )
    if dest.exists():
        if sha256_file(dest) != _S1_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        dest.write_bytes(Path(source_xlsx).read_bytes())
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title="Environmental robustness of the global yeast genetic interaction network",
        library_id="6582362",
        zotero_item_key="CJ5NIJI9",
        files=[
            ArtifactRecord(
                path=_S1_RAW_RELPATH,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=_S1_SHA256,
                source=SCIENCE_SI_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.manual_browser,
                    source_url=SCIENCE_SI_URL,
                    retriever="manual",
                    params={"retrieval_command": MANUAL_RECIPE},
                    sha256=_S1_SHA256,
                    retrieved_at=retrieved_at,
                ),
            )
        ],
        si_data_sources=[SCIENCE_SI_URL],
        si_expected=[
            "Data file S1 (Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx)",
            "Supplementary Materials PDF (names the reference condition's solvent; NOT "
            "mirrored, and the reason SmallMoleculePerturbation.solvent is None here)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return str(root)


@register_dataset
class EnvChemgenCostanzo2021Dataset(ExperimentDataset):
    """Costanzo 2021 condition-SGA: env x (deletion | ts-allele) -> differential fitness."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_costanzo2021",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for R64 ORF resolution."""
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
        """The deposited Data File S1 spreadsheet required before processing."""
        return [_S1_FILENAME]

    def download(self) -> None:
        """Copy Data File S1 from the raw mirror into ``raw_dir``; verify its sha256.

        Science supplementary downloads are not scriptable (HTTP 403 behind a Cloudflare
        challenge, verified 2026-09-12, same as Costanzo 2016), so the file is deposited
        once into ``$DATA_ROOT/torchcell-raw/<citation_key>/`` and the mirror -- not the
        URL -- is the source of record.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _S1_FILENAME)
        if not osp.exists(dest):
            mirror = osp.join(
                os.environ["DATA_ROOT"], "torchcell-raw", CITATION_KEY, _S1_RAW_RELPATH
            )
            if not osp.exists(mirror):
                raise RuntimeError(
                    f"raw-mirror file not found: {mirror}. Costanzo 2021's Science SI is "
                    "not scriptable (403); deposit Data File S1 from "
                    "https://www.science.org/doi/10.1126/science.abf8424 into the raw "
                    "mirror with deposit_raw_mirror(), then rebuild (sha256 verified)."
                )
            with open(mirror, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _S1_SHA256:
            raise RuntimeError(
                f"{_S1_FILENAME} sha256 mismatch: got {digest}, expected {_S1_SHA256}"
            )

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """Build the 26 C environment carrying this condition's edit.

        A small-molecule condition is the shared SGA scoring medium plus one dosed
        compound; the galactose condition is the derived galactose medium with no
        perturbation (the carbon source REPLACES glucose, it is not added to it).
        ``duration_hours`` is a typed gap: the reference sheet distinguishes a 3-day
        from a 5-day incubation and the differential's choice is in the un-mirrored SI.
        """
        gaps = [
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.deferred_pending_source_review,
            )
        ]
        if spec["kind"] == "medium":
            return Environment(
                media=spec["media"],
                temperature=Temperature(value=_TEMPERATURE.value),
                perturbations=[],
                aerobicity="aerobic",
                provenance_gaps=gaps,
            )
        value, unit_token = spec["dose"].value
        return Environment(
            media=SGA_DM_SELECTION,
            temperature=Temperature(value=_TEMPERATURE.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound(spec["name"]),
                    concentration=Concentration(value=value, unit=_UNIT[unit_token]),
                )
            ],
            aerobicity="aerobic",
            provenance_gaps=gaps,
        )

    def _reference_dump(self) -> dict[str, Any]:
        """One shared reference: the standard SGA reference condition, differential 0.

        Every test copy is scored against the SAME matched reference copy of its array
        (``_MATCHED_DESIGN``), which is the standard SGA condition on glucose with no
        compound, so the whole dataset has ONE reference and its differential is 0.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="S288C"
            ),
            environment_reference=Environment(
                media=SGA_DM_SELECTION,
                temperature=Temperature(value=_TEMPERATURE.value),
                perturbations=[],
                aerobicity="aerobic",
                provenance_gaps=[
                    ProvenanceGap(
                        field="duration_hours",
                        reason=ProvenanceGapReason.deferred_pending_source_review,
                    )
                ],
            ),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=_READOUT_DEFINITION.value,
                assay_type=_ASSAY.value,
                environment_response=0.0,
                n_samples=_N_SAMPLES.value,
                sample_unit=SampleUnit.screen,
                units=MEASUREMENT_UNITS,
            ),
        ).model_dump()

    def _genotype(
        self,
        orf: str,
        source_orf: str,
        gene_name: str | None,
        allele: str | None,
        strain_id: str,
    ) -> Genotype:
        """SGA strains: essential -> ts allele; non-essential -> KanMX deletion.

        Both carry the SGA ``strain_id`` (Costanzo 2021 is condition-SGA, same assay family
        as Costanzo 2016), so the Sga* perturbation leaves are the correct types. ``orf`` is
        the CURRENT systematic name the shared resolver returned; ``source_orf`` is the
        sheet's own name for the strain, and it is what an unnamed strain stores in
        ``perturbed_gene_name``.

        Storing the SOURCE name matters where SGD MERGED two features: the array screened
        YPR089W (dma5081) and YPR090W (dma5080) as two strains with two measurements, and
        YPR090W now resolves to YPR089W. They are still two distinct strains, so the
        source name is what keeps them apart instead of collapsing two measurements into
        one L1 duplicate.
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
                    perturbed_gene_name=gene_name if gene_name else source_orf,
                    strain_id=strain_id,
                )
            ]
        )

    @post_process
    def process(self) -> None:
        """Parse the differential-fitness sheet into records; write LMDB + the drop log."""
        if self.genome is None:
            raise RuntimeError(
                "EnvChemgenCostanzo2021Dataset requires a genome for R64 ORF "
                "resolution; inject SCerevisiaeGenome(...)"
            )
        resolve = self.genome.resolve_gene_name
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
            }
            for spec in _CONDITIONS
        ]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        kept_strains = 0
        dropped: list[DroppedStrain] = []
        with env.begin(write=True) as txn:
            for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Costanzo2021"):
                source_orf = str(row["Systematic Name"]).strip()
                resolution = resolve(source_orf)
                n_cells = int(
                    sum(1 for spec in conditions if not pd.isna(row[spec["sheet_col"]]))
                )
                if resolution.status not in _KEPT_STATUSES:
                    dropped.append(
                        DroppedStrain(
                            source_name=source_orf,
                            status=resolution.status.value,
                            resolved_to=resolution.systematic_name,
                            feature_type=resolution.feature_type,
                            n_records=n_cells,
                        )
                    )
                    continue
                orf = str(resolution.systematic_name)
                kept_strains += 1
                gene_name = row["Gene Name"]
                gene_name = None if pd.isna(gene_name) else str(gene_name).strip()
                allele = row["Allele (Essential genes only)"]
                allele = None if pd.isna(allele) else str(allele).strip()
                strain_id = str(row["Strain ID"]).strip()
                genotype = self._genotype(orf, source_orf, gene_name, allele, strain_id)
                for spec in conditions:
                    value = row[spec["sheet_col"]]
                    if pd.isna(value):
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=spec["environment"],
                        phenotype=EnvironmentResponsePhenotype(
                            measurement_type=_READOUT_DEFINITION.value,
                            assay_type=_ASSAY.value,
                            environment_response=float(value),
                            n_samples=_N_SAMPLES.value,
                            sample_unit=SampleUnit.screen,
                            units=MEASUREMENT_UNITS,
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

        by_status: dict[str, int] = {}
        for entry in dropped:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
        drop_log = DropLog(
            dataset=self.name,
            rule=DROP_RULE,
            n_source_strains=len(frame),
            n_kept_strains=kept_strains,
            n_kept_records=idx,
            n_dropped_strains=len(dropped),
            n_dropped_records=sum(entry.n_records for entry in dropped),
            dropped_by_status=by_status,
            dropped=sorted(dropped, key=lambda entry: entry.source_name),
        )
        with open(osp.join(self.root, _DROPPED_FILENAME), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Wrote %d Costanzo2021 records from %d/%d strains; dropped %d strains "
            "(%d records) by rule: %s",
            idx,
            kept_strains,
            len(frame),
            len(dropped),
            drop_log.n_dropped_records,
            by_status,
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_costanzo2021")
    dataset = EnvChemgenCostanzo2021Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    with open(osp.join(root, _DROPPED_FILENAME)) as handle:
        print(json.load(handle)["n_dropped_records"])


if __name__ == "__main__":
    main()

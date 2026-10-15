# torchcell/datasets/pputida/fitness_browser_putida
# [[torchcell.datasets.pputida.fitness_browser_putida]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/fitness_browser_putida
# Test file: tests/torchcell/datasets/pputida/test_fitness_browser_putida.py
"""Fitness Browser, P. putida KT2440 (``Putida``): the browser's own experiments, keyed by
publication.

The Borchert 2024 decomposition (#931, plan decision 16; owner policy of 2026-10-10: "a
served compendium is a sanity check, not the source; load each original study from its own
release"). ONE loader for the organism, reading the Fitness Browser deposit
(:mod:`torchcell.datasets.fitness_browser`), with every stored record's ``Publication``
the browser's ``pubId`` for its sample (:data:`SOURCE_STUDIES`, the issues #932 to #940).
Borchert 2024 copied these samples' values to three decimals and dropped 46 loci per
sample; the browser is each study's own release, at full precision and with all 4,778
loci, so the studies are loaded from it (``source_kind = primary``) and
``RbTnseqBorchert2024Dataset`` stays served as their L4 cross-check.

WHAT IS STORED, of the deposit's 314 Putida experiments (the inventory of
``experiments/database/scripts/fitness_browser_inventory.py``, 2026-10-12): every sample of
the ten attributed studies whose medium and conditions type, 234 samples. NOT STORED, each
with its fate in ``preprocess/samples.json``:

* the 34 experiments with an empty ``pubId`` (sets 8 and 9, the bioreactor series, and 12
  carbon-source samples): an unattributed release, an open owner decision of #931;
* ``VillalobosEscobedo23`` (46): every sample is on ``R2A``, which has no MEDIA_LIBRARY
  entry; its groups include Trichoderma culture supernatants, whose agent has no compound
  identity.

RECORDS. One ``EnvironmentResponseExperiment`` per (locus, stored sample):

* the value is the browser's gene fitness (``FitByExp_Putida.fit``), its moderated t in
  ``environment_response_t_statistic`` and the standard error ``sqrt((f / t)^2 - 0.1^2)``
  in ``environment_response_se``;
* genotype: one ``TransposonInsertionPerturbation`` per locus in the sample's own
  ``mutantLibrary`` (``Putida_ML5`` or ``Putida_ML5_JBEI``), the transposon Borchert 2024
  sources (``borchert2024.TRANSPOSON``), under the locus tag the deposit gives it
  (``Gene.sysName``) and its ``Gene.gene`` symbol; no assembly is consulted (owner,
  2026-10-12: "the genome is separate.. it doesn't gate anything");
* environment: built exactly as ``borchert2024.build_environment`` builds it from the same
  columns (the medium through ``borchert2024.MEDIA_BY_LABEL``, ``condition_1`` as the
  carbon or nitrogen source, ``condition_2`` as DMSO or the 20-amino-acid mix and its
  dropout), plus ``nGenerations`` as ``duration_generations`` where the browser states it;
* reference: the typical gene of the same sample, response 0 by the normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SmallMoleculePerturbation,
    Temperature,
    TransposonInsertionPerturbation,
)
from torchcell.datasets import fitness_browser as fb
from torchcell.datasets.bacteria_common import assembly_reference
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.pputida import borchert2024 as b24
from torchcell.verification.report import Level, LevelResult, VerificationReport
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

log = logging.getLogger(__name__)

#: The raw-mirror key the deposit sits under; the records' Publication is the study's.
CITATION_KEY: Final = "priceMutantPhenotypesThousands2018"
ORG_ID: Final = "Putida"
REFERENCE_STRAIN: Final = "KT2440"
DATASET_ROOT_REL: Final = "data/torchcell/rbtnseq_fitness_browser_putida"
RAW_FILE: Final = "feba.db"

# --------------------------------------------------------------------------- #
# Studies: the deposit's Putida publications
# --------------------------------------------------------------------------- #
_B24 = b24.SOURCE_STUDIES

SOURCE_STUDIES: Final[dict[str, fb.SourceStudy]] = {
    "Schmidt22": fb.SourceStudy(
        pub_id="Schmidt22",
        browser_title="Nitrogen Metabolism in Pseudomonas putida: Functional Analysis "
        "Using Random Barcode Transposon Sequencing",
        browser_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9004399/",
        doi=_B24["schmidt2022"].doi,
        citation_key=_B24["schmidt2022"].citation_key,
        source_kind="primary",
        issues=(932,),
    ),
    "Thompson20": fb.SourceStudy(
        pub_id="Thompson20",
        browser_title="Functional analysis of the fatty acid and alcohol metabolism of "
        "Pseudomonas putida using RB-TnSeq",
        browser_url="https://www.biorxiv.org/content/10.1101/2020.07.04.188060v1",
        doi=_B24["thompson2020"].doi,
        citation_key=_B24["thompson2020"].citation_key,
        source_kind="primary",
        issues=(933,),
        note="the browser links the preprint; the journal version is the mirrored one",
    ),
    "Incha20": fb.SourceStudy(
        pub_id="Incha20",
        browser_title="Leveraging host metabolism for bisdemethoxycurcumin production in "
        "Pseudomonas putida",
        browser_url="https://doi.org/10.1016/j.mec.2019.e00119",
        doi="10.1016/j.mec.2019.e00119",
        citation_key=None,
        source_kind="primary",
        issues=(934,),
    ),
    "Rand17": fb.SourceStudy(
        pub_id="Rand17",
        browser_title="A metabolic pathway for catabolizing levulinic acid in bacteria",
        browser_url="https://doi.org/10.1038/s41564-017-0028-z",
        doi=_B24["rand2017"].doi,
        citation_key=_B24["rand2017"].citation_key,
        source_kind="primary",
        issues=(935,),
    ),
    "Thompson18": fb.SourceStudy(
        pub_id="Thompson18",
        browser_title="Massively parallel fitness profiling reveals multiple novel "
        "enzymes in Pseudomonas putida lysine metabolism",
        browser_url="https://doi.org/10.1101/450254",
        doi="10.1128/mBio.02577-18",
        citation_key="thompsonMassivelyParallelFitness2019",
        source_kind="primary",
        issues=(936,),
        note="the browser links the preprint; the DOI is the journal version's, from "
        "the raw mirror's manifest for the citation key",
    ),
    "Garber20": fb.SourceStudy(
        pub_id="Garber20",
        browser_title="Evolutionarily Driven Domain Swap Alters Sigma Factor Dependence "
        "in Bacterial Signaling System",
        browser_url="https://doi.org/10.1101/2020.09.30.321588",
        doi="10.1101/2020.09.30.321588",
        citation_key=None,
        source_kind="primary",
        issues=(937,),
    ),
    "Price18b": fb.SourceStudy(
        pub_id="Price18b",
        browser_title="Oxidative pathways of deoxyribose and deoxyribonate catabolism",
        browser_url="http://dx.doi.org/10.1128/msystems.00297-18",
        doi="10.1128/msystems.00297-18",
        citation_key=None,
        source_kind="primary",
        issues=(938,),
    ),
    "Price21b": fb.SourceStudy(
        pub_id="Price21b",
        browser_title="GapMind for Carbon Sources: Automated annotations of catabolic "
        "pathways",
        browser_url="https://doi.org/10.1101/2021.11.02.466981",
        doi="10.1101/2021.11.02.466981",
        citation_key=None,
        source_kind="primary",
        issues=(939,),
    ),
    "Thompson19": fb.SourceStudy(
        pub_id="Thompson19",
        browser_title="Omics-driven identification and elimination of valerolactam "
        "catabolism in Pseudomonas putida KT2440 for increased product titer",
        browser_url="https://pubmed.ncbi.nlm.nih.gov/31720214/",
        doi="10.1016/j.mec.2019.e00098",
        pubmed_id="31720214",
        citation_key="thompsonOmicsdrivenIdentificationElimination2019",
        source_kind="primary",
        issues=(940,),
        note="the browser links PubMed; the DOI is from the raw mirror's manifest",
    ),
    "Pearson24": fb.SourceStudy(
        pub_id="Pearson24",
        browser_title="Characterization and Diversification of AraC/XylS Family "
        "Regulators Guided by Transposon Sequencing",
        browser_url="https://pubmed.ncbi.nlm.nih.gov/38113125/",
        doi=None,
        pubmed_id="38113125",
        citation_key=None,
        source_kind="primary",
        note="not in Borchert 2024; the browser links PubMed and no DOI is in the bytes",
    ),
    "VillalobosEscobedo23": fb.SourceStudy(
        pub_id="VillalobosEscobedo23",
        browser_title="Genome-wide fitness profiling reveals molecular mechanisms that "
        "bacteria use to interact with Trichoderma atroviride exometabolites",
        browser_url="https://doi.org/10.1371/journal.pgen.1010909",
        doi="10.1371/journal.pgen.1010909",
        citation_key=None,
        source_kind="primary",
        note="not in Borchert 2024; every sample is on R2A (no MEDIA_LIBRARY entry)",
    ),
}

#: The publications whose samples this loader stores, measured on the 2026-10-12 deposit
#: (``decide`` over its 314 experiments; ``test_the_deposit_putida_fates`` re-measures it).
#: The decomposition audit reads it to mark these studies loaded.
STORED_PUB_IDS: Final = frozenset(
    {
        "Garber20",
        "Incha20",
        "Pearson24",
        "Price18b",
        "Price21b",
        "Rand17",
        "Schmidt22",
        "Thompson18",
        "Thompson19",
        "Thompson20",
    }
)
DATASET_CLASS_NAME: Final = "FitnessBrowserPutidaDataset"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: The browser's ``expGroup`` -> the factor its ``condition_1`` varies.
FACTOR_BY_GROUP: Final[dict[str, PhysicalFactor]] = {
    "carbon source": PhysicalFactor.carbon_source,
    "nitrogen source": PhysicalFactor.nitrogen_source,
}
#: The media a group may name (``borchert2024.MEDIA_BY_LABEL`` keys); else refused.
GROUP_MEDIA: Final[dict[str, frozenset[str]]] = {
    "carbon source": frozenset(
        {"MOPS minimal media_noCarbon", "RCH2_defined_noCarbon"}
    ),
    "nitrogen source": frozenset({"MOPS minimal media_Glucose_noNitrogen"}),
}
UNITS_BY_LABEL: Final[dict[str, ConcentrationUnit]] = {
    "mM": ConcentrationUnit.millimolar,
    "vol%": ConcentrationUnit.percent_v_v,
}
#: ``condition_2``'s unit for the 20-amino-acid mix (no ConcentrationUnit for it).
MIX_UNIT: Final = "X"


def unrepresentable(e: fb.FbExperiment) -> str | None:
    """Why the sample's columns have no typed representation, or None."""
    if e.aerobic != "Aerobic" or e.liquid != "Liquid":
        return f"aerobic={e.aerobic!r}, liquid={e.liquid!r}"
    if e.exp_group not in FACTOR_BY_GROUP:
        return f"group {e.exp_group!r}"
    if e.media not in GROUP_MEDIA[e.exp_group]:
        return f"group {e.exp_group!r} on medium {e.media!r}"
    if e.temperature_c is None:
        return "no temperature"
    if not 1 <= len(e.conditions) <= 2 or e.conditions[0].position != 1:
        return f"condition columns {[c.position for c in e.conditions]}"
    first = e.conditions[0]
    if first.condition is None or first.concentration is None:
        return f"condition_1 {first.condition!r} at {first.concentration_text!r}"
    if first.units != "mM":
        return f"condition_1 unit {first.units!r}"
    if len(e.conditions) == 2:
        second = e.conditions[1]
        if second.position != 2 or second.condition is None:
            return f"condition_{second.position} {second.condition!r}"
        if second.units == MIX_UNIT:
            if not second.condition.startswith(b24.AMINO_ACID_MIX):
                return f"unit 'X' on {second.condition!r}"
            dropped = second.condition.removeprefix(b24.AMINO_ACID_MIX)
            if dropped and dropped.removeprefix("_minus_") not in b24.AMINO_ACIDS:
                return f"amino-acid dropout {second.condition!r}"
        elif second.units not in UNITS_BY_LABEL or second.concentration is None:
            return f"condition_2 {second.condition!r} at {second.concentration_text!r} {second.units!r}"
    return None


def decide(e: fb.FbExperiment) -> fb.SampleDecision:
    """The sample's fate, by the first rule that applies.

    Order: no publication, a study loaded from its own release, a medium outside the
    library, an unrepresentable condition. A publication missing from
    :data:`SOURCE_STUDIES` is refused (the table is the attribution, never a default).
    """
    if e.pub_id is None:
        return fb.decision(e, "unattributed_release")
    if e.pub_id not in SOURCE_STUDIES:
        raise KeyError(f"{e.exp_name}: pubId {e.pub_id!r} is not in SOURCE_STUDIES")
    study = SOURCE_STUDIES[e.pub_id]
    if study.own_release_dataset is not None:
        return fb.decision(e, "own_release_loaded", study.own_release_dataset)
    if e.media not in b24.MEDIA_BY_LABEL:
        return fb.decision(e, "medium_not_in_media_library", e.media)
    reason = unrepresentable(e)
    if reason is not None:
        return fb.decision(e, "condition_not_representable", reason)
    return fb.decision(e, "stored")


DURATION_GAP_NOTE = (
    "the deposit's Experiment table has no growth-time column; nGenerations is stated "
    "for some samples and stored as duration_generations there"
)


def build_environment(e: fb.FbExperiment) -> Environment:
    """The sample's medium, condition variables, temperature and generations.

    The perturbations are ``borchert2024.build_environment``'s for the same columns:
    ``condition_1`` the varied carbon or nitrogen source, ``condition_2`` an added
    species (DMSO in vol%) or the 20-amino-acid mix (``X``) with its ``_minus_<X>``
    dropout as a ``nutrient_dropout``.
    """
    reason = unrepresentable(e)
    if reason is not None:
        raise ValueError(f"{e.exp_name}: {reason}")
    first = e.conditions[0]
    if first.condition is None or first.concentration is None or first.units is None:
        raise ValueError(f"{e.exp_name}: incomplete condition_1")
    if e.temperature_c is None:
        raise ValueError(f"{e.exp_name}: no temperature")
    perturbations: list[EnvironmentPerturbationType] = [
        EnvironmentPhysicalPerturbation(
            factor=FACTOR_BY_GROUP[e.exp_group],
            agent=b24.condition_compound(first.condition),
            magnitude=Concentration(
                value=first.concentration, unit=UNITS_BY_LABEL[first.units]
            ),
        )
    ]
    if len(e.conditions) == 2:
        second = e.conditions[1]
        if second.condition is None:
            raise ValueError(f"{e.exp_name}: condition_2 without a condition")
        if second.units == MIX_UNIT:
            perturbations.append(
                SmallMoleculePerturbation(
                    compound=b24.amino_acid_mix(),
                    concentration=Concentration(basis=DoseBasis.fixed),
                    description=f"the release's 20-amino-acid mix at "
                    f"{second.concentration:g} X (no ConcentrationUnit for 'X')",
                )
            )
            dropped = second.condition.removeprefix(b24.AMINO_ACID_MIX)
            if dropped:
                perturbations.append(
                    EnvironmentPhysicalPerturbation(
                        factor=PhysicalFactor.nutrient_dropout,
                        agent=resolved_compound(
                            b24.AMINO_ACIDS[dropped.removeprefix("_minus_")]
                        ),
                    )
                )
        else:
            if second.concentration is None or second.units is None:
                raise ValueError(f"{e.exp_name}: condition_2 without a dose")
            perturbations.append(
                SmallMoleculePerturbation(
                    compound=b24.condition_compound(second.condition),
                    concentration=Concentration(
                        value=second.concentration, unit=UNITS_BY_LABEL[second.units]
                    ),
                )
            )
    gaps = [
        ProvenanceGap(
            field="duration_hours",
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=fb.FEBA_PROVENANCE,
            note=DURATION_GAP_NOTE,
        )
    ]
    if e.n_generations is None:
        gaps.append(
            ProvenanceGap(
                field="duration_generations",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=fb.FEBA_PROVENANCE,
                note=DURATION_GAP_NOTE,
            )
        )
    return Environment(
        media=b24.MEDIA_BY_LABEL[e.media],
        temperature=Temperature(value=e.temperature_c),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_generations=e.n_generations,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Phenotype, reference, genotype
# --------------------------------------------------------------------------- #
UNITS = (
    "RB-TnSeq gene fitness (Fitness Browser FitByExp.fit): log2 ratio of the gene's "
    "insertion-strain barcode abundance at the end of growth over the time-zero sample, "
    "weighted over the gene's usable strains, typical gene = 0; t = the browser's "
    "moderated t (Wetmore 2015, sigma 0.1); SE = sqrt((fit / t)^2 - 0.1^2)"
)
UNITS_REFERENCE = "the typical gene of this sample (gene fitness is normalized to 0)"
_GAPS: Final = (
    ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the deposit releases the moderated t, not an uncertainty; the standard "
        "error is derived from it into environment_response_se",
    ),
    ProvenanceGap(
        field="n_samples", reason=ProvenanceGapReason.not_reported_by_primary
    ),
    ProvenanceGap(
        field="sample_unit", reason=ProvenanceGapReason.not_reported_by_primary
    ),
)


def build_phenotype(
    fitness: float, t: float, screen_id: str
) -> EnvironmentResponsePhenotype:
    """One gene's fitness in one sample, its t verbatim and its derived standard error."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_se=fb.derived_se(fitness, t),
        environment_response_t_statistic=t,
        screen_id=screen_id,
        units=UNITS,
        provenance_gaps=list(_GAPS),
    )


def build_reference(
    dataset_name: str,
    environment: Environment,
    genome_reference: ReferenceGenome,
    screen_id: str,
) -> EnvironmentResponseExperimentReference:
    """The typical gene of the same sample: fitness 0 by the normalization."""
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            screen_id=screen_id,
            units=UNITS_REFERENCE,
        ),
    )


def build_genotype(tag: str, name: str, library: str) -> Genotype:
    """A transposon insertion in one gene of the sample's library (gene-level call)."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=tag,
                perturbed_gene_name=name,
                gene_namespace="pputida_kt2440_locus_tag",
                transposon=b24.TRANSPOSON,
                library_pool=library,
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
#: Measured on the 2026-10-12 deposit: 234 stored samples x 4,778 loci.
EXPECTED_SAMPLES: Final = 234
EXPECTED_GENES: Final = 4778
EXPECTED_RECORDS: Final = EXPECTED_SAMPLES * EXPECTED_GENES


class LoaderInventory(BaseModel):
    """The loader's sample, gene and record accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_experiments: int
    fates: dict[str, int]
    fates_by_publication: dict[str, dict[str, int]]
    fate_descriptions: dict[fb.Fate, str]
    stored_samples: tuple[str, ...]
    stored_by_publication: dict[str, int]
    n_loci: int
    stored_records: int
    studies: dict[str, fb.SourceStudy]


def loader_inventory(
    decisions: Sequence[fb.SampleDecision], n_loci: int
) -> LoaderInventory:
    """The accounting of one build."""
    stored = [d for d in decisions if d.fate == "stored"]
    by_pub: dict[str, int] = {}
    for d in stored:
        by_pub[str(d.pub_id)] = by_pub.get(str(d.pub_id), 0) + 1
    return LoaderInventory(
        n_experiments=len(decisions),
        fates=fb.fate_counts(decisions),
        fates_by_publication=fb.fate_counts_by_publication(decisions),
        fate_descriptions=dict(fb.FATE_DESCRIPTIONS),
        stored_samples=tuple(d.exp_name for d in stored),
        stored_by_publication=dict(sorted(by_pub.items())),
        n_loci=n_loci,
        stored_records=len(stored) * n_loci,
        studies=dict(SOURCE_STUDIES),
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class FitnessBrowserPutidaDataset(ExperimentDataset):
    """Fitness Browser KT2440 experiments of the attributed studies, by publication."""

    organism = Organism.pputida

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; no genome is read, the deposit names every gene."""
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
        """The deposit, linked into ``raw/`` by ``download``."""
        return [RAW_FILE]

    def download(self) -> None:
        """Link the pinned deposit into ``raw/`` after verifying its size and sha256."""
        source = fb.feba_path()
        fb.verify_feba(source)
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(source, osp.join(self.raw_dir, RAW_FILE), fb.FEBA.sha256)

    def _pointer(self, obj: Any, hint: str, itxn: Any) -> Any:
        """``obj``'s dump, interned exactly as ``_intern_record`` interns it."""
        holder = {"value": obj.model_dump()}
        self._maybe_intern(holder, "value", obj, hint, itxn)
        return holder["value"]

    @post_process
    def process(self) -> None:
        """Build one record per (locus, stored sample) and the preprocess reports."""
        verify_raw_files(self.raw_dir, {RAW_FILE: fb.FEBA.sha256})
        feba = fb.Feba(osp.join(self.raw_dir, RAW_FILE), verify=False)
        fb.check_studies(SOURCE_STUDIES, feba.publications())
        experiments = feba.experiments(ORG_ID)
        decisions = [decide(e) for e in experiments]
        stored = [
            e for e, d in zip(experiments, decisions, strict=True) if d.fate == "stored"
        ]
        values = feba.values(ORG_ID, [e.exp_name for e in stored])
        genes = fb.released_genes(values.loci, feba.genes(ORG_ID))
        feba.close()
        genome_reference = assembly_reference(REFERENCE_STRAIN)
        counts = loader_inventory(decisions, len(genes))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        genotypes: dict[tuple[str, str], dict[str, Any]] = {}
        idx = 0
        for j, e in enumerate(tqdm(stored, desc=self.name)):
            if values.experiments[j] != e.exp_name:
                raise ValueError(f"column {j} is {values.experiments[j]}")
            environment = build_environment(e)
            reference = build_reference(
                self.name, environment, genome_reference, e.screen_id
            )
            publication = SOURCE_STUDIES[str(e.pub_id)].publication
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                env_ptr = self._pointer(
                    environment, environment.media.stated_name, itxn
                )
                ref_ptr = self._pointer(reference, reference.dataset_name, itxn)
                pub_ptr = self._pointer(publication, "publication", itxn)
                for i, gene in enumerate(genes):
                    key = (gene.tag, e.mutant_library)
                    if key not in genotypes:
                        genotypes[key] = build_genotype(
                            gene.tag, gene.name, e.mutant_library
                        ).model_dump()
                    phenotype = build_phenotype(
                        float(values.fitness[i, j]), float(values.t[i, j]), e.screen_id
                    )
                    record = {
                        "experiment": {
                            "experiment_type": "environment_response",
                            "dataset_name": self.name,
                            "genotype": genotypes[key],
                            "environment": env_ptr,
                            "phenotype": phenotype.model_dump(),
                        },
                        "reference": ref_ptr,
                        "publication": pub_ptr,
                    }
                    if i == 0:
                        self._check_record(
                            record,
                            genotypes[key],
                            environment,
                            phenotype,
                            reference,
                            publication,
                            itxn,
                        )
                    txn.put(f"{idx}".encode(), pickle.dumps(record))
                    idx += 1
        env.close()
        interned_env.close()
        if idx != counts.stored_records:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.stored_records}"
            )
        self._write_reports(experiments, decisions, counts, genes)
        log.info("Wrote %d %s records to LMDB", idx, self.name)

    def _check_record(
        self,
        record: dict[str, Any],
        genotype: dict[str, Any],
        environment: Environment,
        phenotype: EnvironmentResponsePhenotype,
        reference: EnvironmentResponseExperimentReference,
        publication: Publication,
        itxn: Any,
    ) -> None:
        """The fast-path record equals what ``_intern_record`` writes for the same objects."""
        experiment = EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=Genotype.model_validate(genotype),
            environment=environment,
            phenotype=phenotype,
        )
        expected = pickle.loads(
            self._intern_record(experiment, reference, publication, itxn)
        )
        if expected != record:
            raise AssertionError(
                f"{self.name}: assembled record differs from _intern_record's"
            )

    def _write_reports(
        self,
        experiments: Sequence[fb.FbExperiment],
        decisions: Sequence[fb.SampleDecision],
        counts: LoaderInventory,
        genes: Sequence[fb.ReleasedGene],
    ) -> None:
        """``dropped_records.json``, ``samples.json``, ``genes.json``."""
        payloads: dict[str, Any] = {
            "dropped_records.json": counts.model_dump(mode="json"),
            "samples.json": [
                {**d.model_dump(mode="json"), "experiment": e.model_dump(mode="json")}
                for e, d in zip(experiments, decisions, strict=True)
            ],
            "genes.json": [g.model_dump(mode="json") for g in genes],
        }
        for name, payload in payloads.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "FitnessBrowserPutidaDataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0 to L4)
# --------------------------------------------------------------------------- #
#: The served compendium's store the L4 row reads, read-only.
BORCHERT_STORE_REL: Final = "data/torchcell/rbtnseq_borchert2024"
#: Borchert 2024 copies the browser to three decimals, so a shared sample's two copies
#: differ by at most the half unit; the per-sample bar on Pearson r is identity up to
#: that rounding.
BORCHERT_MIN_PEARSON_R: Final = 0.9999


def stored_publications_match(
    records: Iterable[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> LevelResult:
    """SUPPLEMENTARY L1: each stored sample's records name its browser publication."""
    seen: dict[str, set[str]] = {}
    for record in records:
        sample = str(record["experiment"]["phenotype"]["screen_id"])
        publication = record["publication"]
        seen.setdefault(sample, set()).add(
            str(publication.get("doi") or publication.get("pubmed_id"))
        )
    expected: dict[str, str] = {}
    for row in samples:
        if row["fate"] != "stored":
            continue
        study = SOURCE_STUDIES[row["pub_id"]]
        expected[f"{ORG_ID}:{row['exp_name']}"] = str(study.doi or study.pubmed_id)
    wrong = sorted(s for s, ids in seen.items() if ids != {expected.get(s)})
    return LevelResult(
        level=Level.L1,
        name="records_carry_their_browser_publication",
        passed=not wrong and set(seen) == set(expected),
        message=(
            f"SUPPLEMENTARY: {len(seen)} stored samples, {len(expected)} expected; "
            f"{len(wrong)} carry a publication other than their pubId's"
        ),
        details={"wrong": wrong[:20]},
    )


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """The environment-response L0-L4 gate on a built tree, plus the L4 compendium row.

    The L4 row compares every stored sample Borchert 2024's served store also holds,
    over the loci both hold (Pearson r per sample and pooled); the stored samples
    Borchert 2024 lacks are named with the reason (not in its release, or dropped by its
    own rule).
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    root = fb.data_root(data_root)
    # The L4 row first: it reads the served compendium store, which a parallel rebuild
    # may retire, so it is read in the first minute rather than after the long pass.
    store = osp.join(root, BORCHERT_STORE_REL)
    if not osp.isdir(osp.join(store, "processed", "lmdb")):
        raise FileNotFoundError(f"{store}: the store the L4 row reads is absent")
    served = fb.stored_fitness(stream_records(store), ORG_ID)
    with open(osp.join(dataset_root, "preprocess", "samples.json")) as handle:
        samples = json.load(handle)
    own = fb.stored_fitness(stream_records(dataset_root), ORG_ID)
    pub_of = {r["exp_name"]: r["pub_id"] for r in samples}
    check = fb.compendium_check(
        own,
        served,
        {
            s: f"not in {BORCHERT_STORE_REL} (pubId {pub_of[s]})"
            for s in own
            if s not in served
        },
        against=BORCHERT_STORE_REL,
        min_pearson_r=BORCHERT_MIN_PEARSON_R,
    )
    preprocess = osp.join(dataset_root, "preprocess")
    with open(osp.join(preprocess, "compendium_cross_check.json"), "w") as handle:
        handle.write(check.model_dump_json(indent=2))
    feba = fb.Feba(osp.join(dataset_root, "raw", RAW_FILE), verify=False)
    released = {g.sys_name for g in feba.genes(ORG_ID).values() if g.sys_name}
    feba.close()
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=FitnessBrowserPutidaDataset.__name__,
        provenance=fb.FEBA_PROVENANCE,
        expected_count=expected_count,
        sgd_genes=released,
        gene_universe_label="the deposit's Putida Gene.sysName",
        min_containment=1.0,
    )
    report.add(stored_publications_match(stream_records(dataset_root), samples))
    report.add(fb.compendium_check_result(check, "compendium_cross_check_borchert2024"))
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    """``verify`` the built dev store, or print the ``inventory`` off the deposit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("verify", "inventory"))
    args = parser.parse_args(argv)
    root = fb.data_root()
    if args.command == "verify":
        print(verify_build(osp.join(root, DATASET_ROOT_REL), data_root=root).summary())
        return
    feba = fb.Feba(fb.feba_path(root))
    decisions = [decide(e) for e in feba.experiments(ORG_ID)]
    feba.close()
    print(
        json.dumps(
            {
                "fates": fb.fate_counts(decisions),
                "by_publication": fb.fate_counts_by_publication(decisions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

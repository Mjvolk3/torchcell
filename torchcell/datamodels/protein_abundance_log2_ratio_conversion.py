# torchcell/datamodels/protein_abundance_log2_ratio_conversion
# [[torchcell.datamodels.protein_abundance_log2_ratio_conversion]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/protein_abundance_log2_ratio_conversion
# Test file: tests/torchcell/datamodels/test_protein_abundance_log2_ratio_conversion.py
"""Build-time conversion of a protein-abundance experiment to the log2 ratio to its reference.

A ``ProteinAbundanceExperiment`` stores the per-strain quantity the assay reports (for
Messner 2023 a linear, batch-corrected MaxLFQ quantity) and its reference carries the
wild-type profile the same quantity was measured against (the 388-replicate HIS3 strain).
The expression datasets the multitask model trains on store ``log2(mutant / reference)``
per gene, so a proteome that is to be predicted the same way needs the same transform,
and it needs the reference to do it, which the per-experiment conversion hook of
:class:`~torchcell.datamodels.conversion.Converter` does not see. This converter overrides
``convert`` to read both halves of the record and rewrites

    protein_abundance[k]  ->  log2(protein_abundance[k] / reference[k])

for every protein ``k`` the experiment measured, leaving every other experiment type
untouched. The reference profile becomes exactly zero (a ratio to itself), its standard
error is carried to the log2 scale by the delta method, ``se / (value * ln 2)``, and the
phenotype's ``measurement_type`` is rewritten to name the transform and the original
quantity, so a record read back from the LMDB says what it holds. A protein missing from
the reference, or a non-positive quantity on either side, is an error: the loader
guarantees neither can happen for Messner, and a silent NaN would train as a zero.

**Fixed key set.** Strains do not all quantify the same proteins (Messner: 1,441 to 1,850
per strain over 1,850 measured in the reference), and the ``Perturbation`` graph processor
flattens a dict phenotype to a key-sorted vector WITHOUT its keys, so two strains with
different key sets would be misaligned column by column. ``process`` therefore scans the
raw store first for the union of protein keys and the full reference profile, and every
converted record is written over that union with ``NaN`` where the strain did not measure
the protein and ``n_replicates`` of ``0`` there. Downstream, the multitask loss and metrics
exclude non-finite target entries, and the mean deduplicator averages only finite values.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

import lmdb

from torchcell.datamodels.conversion import ConversionMap, Converter
from torchcell.datamodels.schema import (
    ExperimentReferenceType,
    ExperimentType,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
)

if TYPE_CHECKING:
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw

log = logging.getLogger(__name__)

LOG2_RATIO_PREFIX = "log2_ratio_to_reference"
_LN2 = math.log(2.0)
_NAN = float("nan")


def measurement_type_for_log2_ratio(original: str) -> str:
    """Name of the converted quantity, ``log2_ratio_to_reference(<original>)``."""
    return f"{LOG2_RATIO_PREFIX}({original})"


class ReferenceProfile:
    """The wild-type profile over the union of measured proteins, from the raw store."""

    def __init__(
        self,
        abundance: dict[str, float],
        se: dict[str, float] | None,
        n_replicates: dict[str, int],
    ) -> None:
        """Hold the reference abundance, SE and replicate count per protein."""
        self.abundance = abundance
        self.se = se
        self.n_replicates = n_replicates
        self.keys = sorted(abundance)


def scan_reference_union(input_path: str) -> ReferenceProfile:
    """Union of protein keys and the reference profile over every proteome record.

    Every proteome record carries the reference restricted to the proteins it measured,
    all drawn from one wild-type profile, so the union of the per-record references IS
    that profile. A key whose reference value differs between two records is an error.
    """
    env = lmdb.open(input_path, readonly=True, lock=False)
    abundance: dict[str, float] = {}
    se: dict[str, float] = {}
    n_rep: dict[str, int] = {}
    any_se = False
    n_records = 0
    with env.begin() as txn:
        for _, value in txn.cursor():
            data = json.loads(value.decode("utf-8"))
            if data["experiment"]["experiment_type"] != "protein_abundance":
                continue
            n_records += 1
            ref = data["experiment_reference"]["phenotype_reference"]
            for k, v in ref["protein_abundance"].items():
                if k in abundance and abundance[k] != v:
                    raise ValueError(
                        f"reference value for {k} differs between records "
                        f"({abundance[k]} vs {v}); the reference is not one profile"
                    )
                abundance[k] = v
            if ref.get("protein_abundance_se") is not None:
                any_se = True
                se.update(ref["protein_abundance_se"])
            n_rep.update(ref["n_replicates"])
    env.close()
    if n_records == 0:
        raise ValueError(f"no protein_abundance record in {input_path}")
    log.info(
        "reference profile: %d proteins over %d proteome records",
        len(abundance),
        n_records,
    )
    return ReferenceProfile(abundance, se if any_se else None, n_rep)


def convert_protein_abundance_pair(
    experiment: ProteinAbundanceExperiment,
    reference: ProteinAbundanceExperimentReference,
    profile: ReferenceProfile | None = None,
) -> tuple[ProteinAbundanceExperiment, ProteinAbundanceExperimentReference]:
    """Return the (experiment, reference) pair on the log2-ratio scale.

    With ``profile`` the output vectors span the profile's key union, ``NaN`` where the
    strain did not measure a protein; without it they span the record's own keys.
    """
    exp_ph = experiment.phenotype
    ref_ph = reference.phenotype_reference
    if exp_ph.measurement_type.startswith(LOG2_RATIO_PREFIX):
        raise ValueError(
            f"{experiment.dataset_name}: measurement_type {exp_ph.measurement_type!r} "
            "is already a log2 ratio; converting twice would be wrong"
        )
    ref_ab: dict[str, float] = (
        profile.abundance if profile is not None else ref_ph.protein_abundance
    )
    ref_se: dict[str, float] | None = (
        profile.se if profile is not None else ref_ph.protein_abundance_se
    )
    ref_n: dict[str, int] = (
        profile.n_replicates if profile is not None else ref_ph.n_replicates
    )
    keys = profile.keys if profile is not None else sorted(exp_ph.protein_abundance)
    missing = sorted(k for k in exp_ph.protein_abundance if k not in ref_ab)
    if missing:
        raise ValueError(
            f"{experiment.dataset_name}: {len(missing)} measured proteins have no "
            f"reference value (first: {missing[:5]})"
        )

    ratio: dict[str, float] = {}
    n_rep: dict[str, int] = {}
    for k in keys:
        if k in exp_ph.protein_abundance:
            v = exp_ph.protein_abundance[k]
            r = ref_ab[k]
            if not (v > 0.0 and r > 0.0):
                raise ValueError(
                    f"{experiment.dataset_name}: non-positive quantity for {k} "
                    f"(experiment {v}, reference {r}); log2 is undefined"
                )
            ratio[k] = math.log2(v / r)
            n_rep[k] = exp_ph.n_replicates[k]
        else:
            ratio[k] = _NAN
            n_rep[k] = 0

    exp_se = exp_ph.protein_abundance_se
    exp_se_log2: dict[str, float] | None = None
    if exp_se is not None:
        exp_se_log2 = {
            k: (
                exp_se[k] / (exp_ph.protein_abundance[k] * _LN2)
                if k in exp_se
                else _NAN
            )
            for k in keys
        }
    new_exp_ph = ProteinAbundancePhenotype(
        protein_abundance=ratio,
        protein_abundance_se=exp_se_log2,
        n_replicates=n_rep,
        measurement_type=measurement_type_for_log2_ratio(exp_ph.measurement_type),
    )
    ref_se_log2: dict[str, float] | None = None
    if ref_se is not None:
        ref_se_log2 = {k: ref_se[k] / (ref_ab[k] * _LN2) for k in keys}
    new_ref_ph = ProteinAbundancePhenotype(
        protein_abundance={k: 0.0 for k in keys},
        protein_abundance_se=ref_se_log2,
        n_replicates={k: ref_n[k] for k in keys},
        measurement_type=measurement_type_for_log2_ratio(ref_ph.measurement_type),
    )
    new_experiment = ProteinAbundanceExperiment(
        dataset_name=experiment.dataset_name,
        genotype=experiment.genotype,
        environment=experiment.environment,
        phenotype=new_exp_ph,
    )
    new_reference = ProteinAbundanceExperimentReference(
        dataset_name=reference.dataset_name,
        genome_reference=reference.genome_reference,
        environment_reference=reference.environment_reference,
        phenotype_reference=new_ref_ph,
    )
    return new_experiment, new_reference


class ProteinAbundanceLog2RatioConverter(Converter):
    """Rewrite every protein-abundance record to ``log2(experiment / reference)``.

    Other experiment types pass through unchanged, so the converter can sit in a build
    that unions the proteome with expression and fitness datasets. When run through
    ``process`` the output vectors span the union of proteins over the raw store (NaN for
    unmeasured entries); ``convert`` called directly, without a scan, keeps each record's
    own keys.
    """

    def __init__(self, root: str, query: Neo4jQueryRaw):
        """Store the root and query; the reference profile is filled in by ``process``."""
        super().__init__(root, query)
        self.profile: ReferenceProfile | None = None

    @property
    def conversion_map(self) -> ConversionMap:
        """Empty: ``convert`` is overridden because it needs the reference too."""
        return ConversionMap(entries=[])

    def process(self, input_path: str, output_path: str) -> None:
        """Scan the raw store for the key union and reference profile, then convert."""
        self.profile = scan_reference_union(input_path)
        super().process(input_path, output_path)

    def convert(
        self, data: dict[str, ExperimentType | ExperimentReferenceType]
    ) -> dict[str, ExperimentType | ExperimentReferenceType | None]:
        """Convert one record; non-proteome records are returned as they came."""
        if "experiment" not in data or "experiment_reference" not in data:
            raise ValueError(
                "Input data must contain both 'experiment' and 'experiment_reference' keys"
            )
        experiment = data["experiment"]
        reference = data["experiment_reference"]
        if not isinstance(experiment, ProteinAbundanceExperiment):
            return dict(data)
        if not isinstance(reference, ProteinAbundanceExperimentReference):
            raise TypeError(
                f"{experiment.dataset_name}: protein-abundance experiment paired with a "
                f"{type(reference).__name__} reference"
            )
        new_experiment, new_reference = convert_protein_abundance_pair(
            experiment, reference, self.profile
        )
        out: dict[str, Any] = dict(data)
        out["experiment"] = new_experiment
        out["experiment_reference"] = new_reference
        return out

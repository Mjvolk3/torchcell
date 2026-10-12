# torchcell/verification/phosphosite
# [[torchcell.verification.phosphosite]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/phosphosite
# Test file: tests/torchcell/verification/test_phosphosite.py
"""L0-L3 record-level gate for phosphosite-intensity datasets (#921).

For ``PhosphositeIntensityPhenotype`` stores: one record per released sample, every
record carrying the release's whole site list. The schema already refuses a site whose
key is not its own, disagreeing key sets, a stored zero and a silent sample field, so
L0 subsumes those. The family rules are the proteome family's shape rules
(``torchcell.verification.protein``: finite values, a key-matched reference, one
measurement_type) restated for sites, plus the two that only a site-level release has:

1. L1 ``one_record_per_sample``: no two records name the same released sample.
2. L1 ``site_list_is_shared``: every record and every reference carry the same site
   keys, because the identification is the release's, not a sample's.
3. L2 ``intensities_finite_positive``: every stored intensity is finite and > 0.
4. L2 ``site_evidence_is_per_site``: a site's identification record (proteins, residue,
   positions, localization, scores) is identical on every record.
5. L3 ``measurement_type_consistent``: one measurement_type across the store.

A release's own pins (its per-sample counts, a re-read of its cells, its sourced quotes
and any cross-table agreement) are its loader's ``verify_build``, which calls this gate.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from torchcell.verification.levels import l0_structural, l1_count
from torchcell.verification.protein import _l3_measurement_type_consistent
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _l1_one_record_per_sample(records: Sequence[Record]) -> LevelResult:
    """L1: each released sample label names exactly one record."""
    labels = Counter(
        str(rec["experiment"]["phenotype"]["sample_label"]) for rec in records
    )
    repeated = sorted(label for label, n in labels.items() if n > 1)
    return LevelResult(
        level=Level.L1,
        name="one_record_per_sample",
        passed=not repeated,
        message=f"{len(labels)} sample labels; {len(repeated)} repeated {repeated}",
        details={"n_samples": len(labels), "repeated": repeated},
    )


def _l1_site_list_is_shared(records: Sequence[Record]) -> LevelResult:
    """L1: every record and reference carries one and the same site key set."""
    key_sets = {
        frozenset(rec[side][field]["sites"])
        for rec in records
        for side, field in (
            ("experiment", "phenotype"),
            ("reference", "phenotype_reference"),
        )
    }
    sizes = sorted(len(keys) for keys in key_sets)
    return LevelResult(
        level=Level.L1,
        name="site_list_is_shared",
        passed=len(key_sets) == 1,
        message=f"{len(key_sets)} distinct site key set(s) of sizes {sizes}",
        details={"n_key_sets": len(key_sets), "sizes": sizes},
    )


def _l2_intensities_finite_positive(records: Sequence[Record]) -> LevelResult:
    """L2: every stored intensity is a finite, strictly positive number."""
    n = 0
    bad: list[str] = []
    for rec in records:
        for key, value in rec["experiment"]["phenotype"][
            "phosphosite_intensity"
        ].items():
            n += 1
            if not isinstance(value, int | float) or not math.isfinite(value):
                bad.append(key)
            elif value <= 0:
                bad.append(key)
    return LevelResult(
        level=Level.L2,
        name="intensities_finite_positive",
        passed=n > 0 and not bad,
        message=f"{n} intensities; {len(bad)} non-finite or non-positive",
        details={"n_values": n, "bad": bad[:20]},
    )


def _l2_site_evidence_is_per_site(records: Sequence[Record]) -> LevelResult:
    """L2: one identification record per site key across the whole store."""
    seen: dict[str, str] = {}
    differing: set[str] = set()
    for rec in records:
        for key, site in rec["experiment"]["phenotype"]["sites"].items():
            canonical = json.dumps(site, sort_keys=True)
            if seen.setdefault(key, canonical) != canonical:
                differing.add(key)
    return LevelResult(
        level=Level.L2,
        name="site_evidence_is_per_site",
        passed=bool(seen) and not differing,
        message=f"{len(seen)} sites; {len(differing)} differ across records",
        details={"n_sites": len(seen), "differing": sorted(differing)[:20]},
    )


def verify_phosphosite_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
) -> VerificationReport:
    """Run the L0-L3 family gate for a phosphosite-intensity dataset."""
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_one_record_per_sample(records))
    report.add(_l1_site_list_is_shared(records))
    report.add(_l2_intensities_finite_positive(records))
    report.add(_l2_site_evidence_is_per_site(records))
    report.add(_l3_measurement_type_consistent(records))
    return report

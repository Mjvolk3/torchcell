# torchcell/verification/natural_isolate_genotype
# [[torchcell.verification.natural_isolate_genotype]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/natural_isolate_genotype
# Test file: tests/torchcell/verification/test_natural_isolate_genotype.py
"""The genotype rules every natural-isolate store on the Peter 2018 panel shares.

A store built by Caudal 2024, Peter 2018, Muenzner 2024 or De Chiara 2022 holds one
genotype per isolate, built by ``torchcell.sequence.natural_isolate``. Each verifier of
those stores feeds every DISTINCT stored genotype to one ``IsolateGenotypeAudit`` and adds
its results:

- L1 ``isolate_genotype_identity``: every genotype carries ``natural_isolate``; every
  perturbation that names a strain names that isolate; an isolate has one genotype; the
  store's isolates are the ones the builder was asked for.
- L2 ``isolate_genotype_round_trip``: every stored genotype equals, dump for dump, the
  genotype ``NaturalIsolateGenotypes`` re-derives from the pinned releases.
- L4 ``isolate_descriptor_matches_table_s1``: each stored descriptor (manipulation,
  ploidy, zygosity, aneuploidy) and HO deletion against Supplementary Table 1 re-read
  from the LIBRARY mirror's copy of the workbook (``si/si3.xls``), not the tier copy the
  build read.
- L4 ``loss_of_function_calls_match_release``: per isolate and cause, the stored
  loss-of-function calls are exactly the rows with a 1 in that isolate's column of the
  release matrix, each named by the matrix's own row key (``NAME`` in
  ``1011LossOfFunction.xls.gz``, ``Gene`` in ``genesMatrix_Frameshift.tab.gz``),
  re-read here with pandas and not through the builder's row logic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torchcell.datamodels.schema import NaturalLossOfFunctionCause
from torchcell.sequence import natural_isolate as ni
from torchcell.verification.report import Level, LevelResult, sha256_file

DELETION = "deletion"
LOSS_OF_FUNCTION = "natural_loss_of_function"


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


class IsolateGenotypeAudit:
    """Fold in each distinct stored genotype; ``results()`` reports the four rules."""

    def __init__(
        self, genotypes: ni.NaturalIsolateGenotypes, library_root: str | Path
    ) -> None:
        """``genotypes`` re-derives each isolate; ``library_root`` holds Table S1."""
        self.genotypes = genotypes
        self.library_tables = (
            Path(library_root) / ni.CITATION_KEY / ni.LIBRARY_TABLES_REL
        )
        self.n_genotypes = 0
        self.isolates: Counter[str] = Counter()
        self.no_descriptor = 0
        self.foreign_strain: list[str] = []
        self.round_trip_failures: list[str] = []
        self.descriptors: dict[str, dict[str, Any]] = {}
        self.ho_deleted: set[str] = set()
        self.lof_keys: dict[tuple[str, str], Counter[str]] = {}

    def add(self, genotype: Mapping[str, Any]) -> str | None:
        """Check one stored genotype dump; return its isolate (None if it names none)."""
        self.n_genotypes += 1
        isolate = ni.genotype_isolate(genotype)
        if isolate is None:
            self.no_descriptor += 1
            return None
        self.isolates[isolate] += 1
        perturbations = genotype["perturbations"]
        strains = {
            str(p["strain_id"]) for p in perturbations if p.get("strain_id") is not None
        }
        if strains - {isolate}:
            self.foreign_strain.append(f"{isolate}: {sorted(strains - {isolate})}")
        for p in perturbations:
            kind = p["perturbation_type"]
            if kind == LOSS_OF_FUNCTION:
                cause = str(p["cause"])
                key = (
                    p["release_name"]
                    if cause == NaturalLossOfFunctionCause.nonsense_or_sift.value
                    else p["release_gene_name"]
                )
                self.lof_keys.setdefault((isolate, cause), Counter())[str(key)] += 1
            elif (
                kind == DELETION and p["systematic_gene_name"] == ni.HO_SYSTEMATIC_NAME
            ):
                self.ho_deleted.add(isolate)
        self.descriptors[isolate] = dict(genotype["natural_isolate"])
        if isolate not in self.genotypes.isolates:
            self.round_trip_failures.append(f"{isolate}: not an isolate of the build")
        elif dict(genotype) != self.genotypes.genotype(isolate).model_dump():
            self.round_trip_failures.append(f"{isolate}: differs from the rebuild")
        return isolate

    # ---- results --------------------------------------------------------------
    def results(self) -> list[LevelResult]:
        """The four rules over everything added."""
        return [
            self._identity(),
            self._round_trip(),
            self._descriptors_vs_table_s1(),
            self._loss_of_function_vs_release(),
        ]

    def _identity(self) -> LevelResult:
        repeated = sorted(i for i, n in self.isolates.items() if n > 1)
        missing = sorted(self.genotypes.isolates - set(self.isolates))
        passed = (
            self.no_descriptor == 0
            and not self.foreign_strain
            and not repeated
            and not missing
        )
        return _result(
            Level.L1,
            "isolate_genotype_identity",
            passed,
            f"{len(self.isolates)} isolates, one genotype each, every strain-carrying "
            "perturbation names its isolate"
            if passed
            else f"{self.no_descriptor} genotypes without natural_isolate; "
            f"{len(self.foreign_strain)} name another strain; {len(repeated)} isolates "
            f"with two genotypes; {len(missing)} builder isolates not stored",
            n_genotypes=self.n_genotypes,
            n_isolates=len(self.isolates),
            foreign_strain=self.foreign_strain[:10],
            repeated=repeated[:10],
            missing=missing[:10],
        )

    def _round_trip(self) -> LevelResult:
        passed = not self.round_trip_failures and self.n_genotypes > 0
        return _result(
            Level.L2,
            "isolate_genotype_round_trip",
            passed,
            f"{self.n_genotypes} stored genotypes equal their re-derivation from the "
            "pinned Peter 2018 releases"
            if passed
            else f"{len(self.round_trip_failures)} of {self.n_genotypes} genotypes "
            "differ from the re-derivation",
            failures=self.round_trip_failures[:20],
        )

    def _descriptors_vs_table_s1(self) -> LevelResult:
        integrity = sha256_file(self.library_tables) == ni.TABLES_SHA256
        problems: list[str] = []
        if integrity:
            table = ni.read_table_s1(self.library_tables)
            for isolate, stored in sorted(self.descriptors.items()):
                row = table[isolate]
                ho = row.ho_deletion == "yes"
                expected = {
                    "genetic_manipulation": "ho_deletion" if ho else None,
                    "ploidy": None
                    if row.ploidy is None or row.ploidy == "Xn"
                    else int(row.ploidy),
                    "zygosity": row.zygosity,
                    "chromosome_copy_changes": None
                    if row.aneuploidies is None
                    else [
                        c.model_dump() for c in ni.parse_aneuploidy(row.aneuploidies)
                    ],
                }
                observed = {
                    "genetic_manipulation": None
                    if stored["genetic_manipulation"] is None
                    else str(stored["genetic_manipulation"]),
                    "ploidy": stored["ploidy"],
                    "zygosity": str(stored["zygosity"]),
                    "chromosome_copy_changes": stored["chromosome_copy_changes"],
                }
                if observed != expected:
                    problems.append(f"{isolate}: {observed} != Table S1 {expected}")
                if ho != (isolate in self.ho_deleted):
                    problems.append(f"{isolate}: HO deletion {ho} in Table S1")
        passed = integrity and not problems
        return _result(
            Level.L4,
            "isolate_descriptor_matches_table_s1",
            passed,
            f"{len(self.descriptors)} descriptors and {len(self.ho_deleted)} HO deletions "
            f"equal Supplementary Table 1 re-read from {self.library_tables.name}"
            if passed
            else f"{len(problems)} descriptor problems; workbook sha256 ok={integrity}",
            n_descriptors=len(self.descriptors),
            n_ho_deleted=len(self.ho_deleted),
            problems=problems[:20],
            workbook_sha256_ok=integrity,
        )

    def _loss_of_function_vs_release(self) -> LevelResult:
        raw = self.genotypes.raw_dir
        isolates = sorted(self.descriptors)
        matrices = {
            NaturalLossOfFunctionCause.nonsense_or_sift.value: (
                ni.read_lof_matrix(raw / ni.LOF_NAME),
                "NAME",
            ),
            NaturalLossOfFunctionCause.homozygous_frameshift.value: (
                ni.read_frameshift_matrix(raw / ni.FRAMESHIFT_NAME),
                "Gene",
            ),
        }
        differ: list[str] = []
        totals: dict[str, int] = {}
        for cause, (frame, key_column) in matrices.items():
            keys = frame[key_column].astype(str)
            totals[cause] = 0
            for isolate in isolates:
                expected = Counter(keys[frame[isolate] == 1])
                stored = self.lof_keys.get((isolate, cause), Counter())
                totals[cause] += sum(stored.values())
                if stored != expected:
                    extra = sorted((stored - expected).elements())
                    lacking = sorted((expected - stored).elements())
                    differ.append(
                        f"{isolate}/{cause}: stored {sum(stored.values())}, release "
                        f"{sum(expected.values())}; extra {extra[:3]}, "
                        f"missing {lacking[:3]}"
                    )
        passed = not differ and bool(isolates)
        return _result(
            Level.L4,
            "loss_of_function_calls_match_release",
            passed,
            f"stored loss-of-function calls are the release's 1s, row for row by the "
            f"matrix's own key, over {len(isolates)} isolates: {totals}"
            if passed
            else f"{len(differ)} (isolate, cause) call sets differ from the release",
            totals=totals,
            differ=differ[:20],
        )

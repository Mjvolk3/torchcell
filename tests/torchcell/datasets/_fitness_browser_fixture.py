# tests/torchcell/datasets/_fitness_browser_fixture.py
# [[tests.torchcell.datasets._fitness_browser_fixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/_fitness_browser_fixture.py
"""A small SQLite database with the Fitness Browser's table layout, for hermetic tests.

The columns are the deposit's own (``feba.db``'s ``Experiment``, ``Publication``,
``Organism``, ``Gene`` and ``FitByExp_<orgId>`` definitions); the rows are written by the
test that needs them.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_COLUMNS: tuple[str, ...] = (
    "orgId",
    "expName",
    "expDesc",
    "timeZeroSet",
    "num",
    "nMapped",
    "nPastEnd",
    "nGenic",
    "nUsed",
    "gMed",
    "gMedt0",
    "gMean",
    "cor12",
    "mad12",
    "mad12c",
    "mad12c_t0",
    "opcor",
    "adjcor",
    "gccor",
    "maxFit",
    "expGroup",
    "expDescLong",
    "mutantLibrary",
    "person",
    "dateStarted",
    "setName",
    "seqindex",
    "media",
    "mediaStrength",
    "temperature",
    "pH",
    "vessel",
    "aerobic",
    "liquid",
    "shaking",
    "condition_1",
    "units_1",
    "concentration_1",
    "condition_2",
    "units_2",
    "concentration_2",
    "condition_3",
    "units_3",
    "concentration_3",
    "condition_4",
    "units_4",
    "concentration_4",
    "growthPlate",
    "growthWells",
    "nGenerations",
    "pubId",
)

#: Every column a test does not set, as the deposit stores an absent value.
DEFAULTS: dict[str, Any] = {
    "expDesc": "a sample",
    "timeZeroSet": "set1IT001",
    "nMapped": 1000,
    "nPastEnd": 0,
    "nGenic": 900,
    "nUsed": 800,
    "gMed": 120,
    "gMedt0": 200,
    "gMean": 150.0,
    "cor12": 0.4,
    "mad12": 0.3,
    "mad12c": 0.2,
    "mad12c_t0": 0.1,
    "opcor": 0.0,
    "adjcor": 0.0,
    "gccor": 0.0,
    "maxFit": 3.0,
    "expGroup": "carbon source",
    "expDescLong": "a sample, long",
    "mutantLibrary": "Lib_ML1",
    "person": "someone",
    "dateStarted": "1-Jan-20",
    "setName": "Lib_ML1_set1",
    "seqindex": "IT001",
    "media": "MOPS minimal media_noCarbon",
    "mediaStrength": 1.0,
    "temperature": "30",
    "pH": "",
    "vessel": "tube",
    "aerobic": "Aerobic",
    "liquid": "Liquid",
    "shaking": "200 rpm",
    "condition_1": "D-Glucose",
    "units_1": "mM",
    "concentration_1": "20",
    "condition_2": "",
    "units_2": "",
    "concentration_2": "",
    "condition_3": "",
    "units_3": "",
    "concentration_3": "",
    "condition_4": "",
    "units_4": "",
    "concentration_4": "",
    "growthPlate": "",
    "growthWells": "",
    "nGenerations": "",
    "pubId": "",
}


def experiment_row(
    org_id: str, exp_name: str, num: int, **cells: Any
) -> dict[str, Any]:
    """One ``Experiment`` row: the defaults, then ``cells``."""
    unknown = set(cells) - set(EXPERIMENT_COLUMNS)
    if unknown:
        raise KeyError(f"not Experiment columns: {sorted(unknown)}")
    return {**DEFAULTS, "orgId": org_id, "expName": exp_name, "num": num, **cells}


def write_feba(
    path: Path,
    *,
    experiments: Sequence[Mapping[str, Any]],
    publications: Sequence[tuple[str, str, str]],
    genes: Sequence[tuple[str, str, str | None, str | None, int]],
    fitness: Mapping[str, Sequence[tuple[str, str, float, float]]],
    organisms: Sequence[tuple[str, str, str, str]] = (),
) -> Path:
    """Write the fixture database.

    ``genes`` rows are ``(orgId, locusId, sysName, gene, type)``; ``fitness`` maps an
    orgId to its ``FitByExp`` rows ``(expName, locusId, fit, t)``; ``organisms`` rows are
    ``(orgId, genus, species, strain)``.
    """
    con = sqlite3.connect(path)
    columns = ", ".join(f'"{c}"' for c in EXPERIMENT_COLUMNS)
    con.execute(f"CREATE TABLE Experiment ({columns}, PRIMARY KEY (orgId, expName))")
    con.execute("CREATE TABLE Publication (pubId TEXT, title TEXT, URL TEXT)")
    con.execute(
        "CREATE TABLE Organism (orgId TEXT, domain TEXT, phylum TEXT, class TEXT, "
        '"order" TEXT, family TEXT, genus TEXT, species TEXT, strain TEXT)'
    )
    con.execute(
        "CREATE TABLE Gene (orgId TEXT, locusId TEXT, sysName TEXT, scaffoldId TEXT, "
        "begin INT, end INT, type INT, strand TEXT, gene TEXT, desc TEXT, GC REAL)"
    )
    placeholders = ", ".join("?" for _ in EXPERIMENT_COLUMNS)
    con.executemany(
        f"INSERT INTO Experiment VALUES ({placeholders})",
        [tuple(row[c] for c in EXPERIMENT_COLUMNS) for row in experiments],
    )
    con.executemany("INSERT INTO Publication VALUES (?, ?, ?)", publications)
    con.executemany(
        "INSERT INTO Organism VALUES (?, 'Bacteria', 'p', 'c', 'o', 'f', ?, ?, ?)",
        organisms,
    )
    con.executemany(
        "INSERT INTO Gene VALUES (?, ?, ?, 'chr', 1, 100, ?, '+', ?, '', 0.5)",
        [
            (org, locus, sys, gene_type, gene)
            for org, locus, sys, gene, gene_type in genes
        ],
    )
    for org, rows in fitness.items():
        con.execute(
            f"CREATE TABLE 'FitByExp_{org}' (expName TEXT NOT NULL, locusId TEXT NOT "
            "NULL, fit REAL NOT NULL, t REAL NOT NULL, PRIMARY KEY (expName, locusId))"
        )
        con.executemany(f"INSERT INTO 'FitByExp_{org}' VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path

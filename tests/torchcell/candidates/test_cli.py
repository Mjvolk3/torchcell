# tests/torchcell/candidates/test_cli.py
# [[tests.torchcell.candidates.test_cli]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/candidates/test_cli.py
"""``candidate-gate`` end to end on a fixture table, tier, mirror, store and verdict dir.

The table script, the module index, the commit and the date are swapped for fixtures; the
D2Cell row runs against its real settled-row finding.
"""

import json
import runpy
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.torchcell.candidates._builders import (
    COMMIT,
    admissible_verdict,
    evidence,
    fitness_record,
    sha,
    verdict,
    write_assembly_set,
    write_mirror,
    write_store,
)
from torchcell.candidates import cli, gates, store
from torchcell.candidates.verdict import GateResult
from torchcell.datasets.ecoli import li2024
from torchcell.sequence.genome.registry import ECOLI_K12_MG1655

ROW: dict[str, Any] = {
    "name": "Fixture 2020",
    "organism": "E. coli",
    "citation": "Fixture A. A screen. J 2020",
    "url": "https://doi.org/10.1000/fixture",
    "klass": "Transposon fitness",
    "genotypes": "MG1655 mutants",
    "env": "1 condition",
    "phenotype": "fitness",
    "seq_basis": "K-12+transposon",
    "modality": "transposon insertion",
    "why": "a screen",
    "accession": "none",
    "accession_confirmed": True,
    "status": "candidate",
    "schema_need": "",
}
ROWS = [
    ROW,
    {
        **ROW,
        "name": "D2Cell 2026",
        "url": f"https://doi.org/{li2024.PAPER_DOI}",
        "status": "aggregation",
    },
    {**ROW, "name": "No DOI 2021", "url": "https://zenodo.org/records/1"},
    {**ROW, "name": "Orphan 2022", "url": "https://doi.org/10.1000/orphan"},
]


class FixedDate(date):
    """date.today() pinned to the plan's date."""

    @classmethod
    def today(cls) -> "FixedDate":
        """2026-10-10."""
        return cls(2026, 10, 10)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Fixture table script, module index, tier, raw mirror, store dir; pinned commit/date."""
    script = tmp_path / "table.py"
    script.write_text(
        "from pydantic import BaseModel, ConfigDict\n\n\n"
        "class Candidate(BaseModel):\n"
        "    model_config = ConfigDict(extra='allow')\n\n\n"
        f"CANDIDATES = [Candidate(**row) for row in {ROWS!r}]\n"
    )
    monkeypatch.setitem(cli.TABLE_SCRIPTS, "bacteria", script)
    datasets = tmp_path / "repo" / "torchcell" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "fixture.py").write_text(
        'CITATION_KEY = "fixtureKey2020"\nDOI = "10.1000/fixture"\n'
    )
    index = gates.module_index(datasets)
    monkeypatch.setattr(gates, "module_index", lambda: index)
    monkeypatch.setattr(cli, "git_commit", lambda: COMMIT)
    monkeypatch.setattr(cli, "date", FixedDate)
    monkeypatch.setattr(cli, "DOI_TO_PMID_CACHE", tmp_path / "no-cache.json")
    root = tmp_path / "data"
    write_assembly_set(root, ECOLI_K12_MG1655)
    write_mirror(root, "torchcell-raw", "fixtureKey2020", {"data/t.csv": b"a,b\n1,2\n"})
    return {"root": root, "store": tmp_path / "verdicts", "tmp": tmp_path}


def run(env: dict[str, Path], *argv: str) -> int:
    """main() with the fixture store dir."""
    return cli.main(["--store-dir", str(env["store"]), *argv])


def test_schema(capsys: pytest.CaptureFixture[str]) -> None:
    """The schema subcommand prints the verdict's JSON schema."""
    assert cli.main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "CandidateVerdict"


def test_validate_refuses_with_the_field_and_stores_a_good_one(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A gap without an issue names gates.4; a valid file is written with --write."""
    good = admissible_verdict(gap_issue=854)
    broken = json.loads(good.model_dump_json())
    broken["gates"][4]["issue"] = None
    bad_path = env["tmp"] / "bad.json"
    bad_path.write_text(json.dumps(broken))
    assert run(env, "validate", str(bad_path)) == 1
    assert capsys.readouterr().out == (
        "refused: gates.4: Value error, G5: a 'gap' must name its issue\n"
    )
    good_path = env["tmp"] / "good.json"
    good_path.write_text(good.model_dump_json())
    assert run(env, "validate", str(good_path), "--write") == 0
    out = capsys.readouterr().out
    assert (
        "  G5 gap (#854): G5 gap\n  zotero item: present\n  outcome: admissible_with_gaps\n"
        in out
    )
    assert store.read_verdict("keyPaper2020", env["store"]) == good


def test_gate_a_primary_row_end_to_end(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """G1..G5 pass on the fixture row; --write stores the verdict."""
    code = run(
        env,
        "gate",
        "--row",
        "Fixture 2020",
        "--data-root",
        str(env["root"]),
        "--phenotype-class",
        "FitnessPhenotype",
        "--compound",
        "glucose",
        "--write",
    )
    assert code == 0
    assert capsys.readouterr().out == (
        "Fixture 2020  [bacteria]  fixtureKey2020\n"
        "  G1 pass: primary: status='candidate', klass='Transposon fitness', no "
        "aggregation marker\n"
        "  G2 pass: assembly_set: MG1655 -> ecoli_K12_MG1655_ASM584v2 resolves and reads\n"
        "  G3 pass: 1 file(s) in raw, 0 worksheet row(s) counted; all intact\n"
        "  G4 pass: no key held by another module (0 mention(s)); no PMID check; values "
        "not compared (no dev store named)\n"
        "  G5 pass: FitnessPhenotype exists; compounds resolve\n"
        "  zotero item: absent\n"
        "  outcome: admissible\n"
        f"wrote {env['store'] / 'fixtureKey2020.json'}\n"
    )
    stored = store.read_verdict("fixtureKey2020", env["store"])
    assert (stored.decided_at, stored.torchcell_commit, stored.passing) == (
        "2026-10-10",
        COMMIT,
        True,
    )


def test_gate_by_citation_key_and_json(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--citation-key alone finds the row by its DOI; --json prints the verdict."""
    assert (
        run(
            env,
            "gate",
            "--citation-key",
            "fixtureKey2020",
            "--data-root",
            str(env["root"]),
            "--json",
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert (printed["row_name"], printed["gates"][4]["outcome"]) == (
        "Fixture 2020",
        "unmeasured",
    )


def test_gate_d2cell_is_refused_at_g1(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """The real D2Cell finding fails G1 as a transcription; exit 1, nothing else runs."""
    code = run(
        env,
        "gate",
        "--row",
        "D2Cell 2026",
        "--citation-key",
        li2024.CITATION_KEY,
        "--data-root",
        str(env["root"]),
    )
    assert code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith(
        "  G1 fail: transcription: every database value is the output"
    )
    assert lines[2:] == [
        "  G2 unmeasured: not evaluated: G1 stopped the run",
        "  G3 unmeasured: not evaluated: G1 stopped the run",
        "  G4 unmeasured: not evaluated: G1 stopped the run",
        "  G5 unmeasured: not evaluated: G1 stopped the run",
        "  zotero item: absent",
        "  outcome: refused",
    ]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "gate needs --row or --citation-key"),
        (
            ("--row", "No DOI 2021"),
            "No DOI 2021: no DOI in its url; pass --citation-key",
        ),
        (
            ("--row", "Orphan 2022"),
            "Orphan 2022: no module or mirror files 10.1000/orphan",
        ),
        (("--row", "Nothing"), "row 'Nothing': 0 prefix matches (none)"),
        (
            ("--row", "D2Cell 2026", "--citation-key", "wrongKey"),
            "finding is filed under",
        ),
        (("--citation-key", "unknownKey"), "unknownKey: 0 rows carry its DOI(s) []"),
    ],
)
def test_gate_usage_errors(
    env: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    message: str,
) -> None:
    """Every refusal to act exits 2 with the reason on stderr."""
    assert run(env, "gate", "--data-root", str(env["root"]), *argv) == 2
    assert message in capsys.readouterr().err


def test_inventory(env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    """A deposit prints its record; a key with none exits 1."""
    assert (
        run(
            env,
            "inventory",
            "--citation-key",
            "fixtureKey2020",
            "--data-root",
            str(env["root"]),
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["files"][0]["sha256_disk"] == sha(b"a,b\n1,2\n")
    assert captured.err == "1 file(s) in raw, 0 worksheet row(s) counted; all intact\n"
    assert (
        run(env, "inventory", "--citation-key", "none", "--data-root", str(env["root"]))
        == 1
    )
    assert (
        capsys.readouterr().out
        == "none: no manifest in the literature or the raw mirror\n"
    )


def _released(env: dict[str, Path], text: str, name: str = "released.csv") -> Path:
    path = env["tmp"] / name
    path.write_text(text)
    return path


def _overlap_args(env: dict[str, Path], store_dir: Path, released: Path) -> list[str]:
    return [
        "overlap",
        "--citation-key",
        "fixtureKey2020",
        "--released",
        str(released),
        "--store",
        str(store_dir),
        "--sample-path",
        "experiment.phenotype.screen_id",
        "--key-path",
        "experiment.genotype.perturbations.0.systematic_gene_name",
        "--value-path",
        "experiment.phenotype.environment_response",
        "--tolerance",
        "0.05",
    ]


def test_overlap_folds_a_subsumption_into_the_verdict(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A released column inside a served sample fails G4 and drops G5."""
    assert (
        run(
            env,
            "gate",
            "--row",
            "Fixture 2020",
            "--data-root",
            str(env["root"]),
            "--phenotype-class",
            "FitnessPhenotype",
            "--write",
        )
        == 0
    )
    dev = env["tmp"] / "dev"
    write_store(
        dev,
        [
            fitness_record("s1", "b0001", 0.11),
            fitness_record("s1", "b0002", -1.0),
            fitness_record("s2", "b0001", 0.9),
            fitness_record("s2", "b0002", 0.2),
        ],
    )
    released = _released(env, "column,key,value\nlysine,b0001,0.1\nlysine,b0002,-1.0\n")
    capsys.readouterr()
    assert run(env, *_overlap_args(env, dev, released), "--write") == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "lysine: best s1 max|d| 0.01 over 2 keys; runner-up s2 margin 1.19"
    )
    updated = store.read_verdict("fixtureKey2020", env["store"])
    assert [g.outcome for g in updated.gates] == [
        "pass",
        "pass",
        "pass",
        "fail",
        "unmeasured",
    ]
    assert (updated.g5, updated.outcome) == (None, "refused")
    assert updated.g4_value is not None and updated.g4_value.records_read == 4


def test_overlap_refusals(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A dirty store exits 1; a bad header or a verdict without G4-key exits 2."""
    store.write_verdict(admissible_verdict(citation_key="fixtureKey2020"), env["store"])
    dirty = env["tmp"] / "dirty"
    write_store(dirty, [], dirty=True)
    good = _released(env, "column,key,value\nc,k,1\n")
    assert run(env, *_overlap_args(env, dirty, good)) == 1
    assert capsys.readouterr().out.startswith("refused: ")
    bad = _released(env, "col,key,value\nc,k,1\n", "bad.csv")
    assert run(env, *_overlap_args(env, dirty, bad)) == 2
    assert "header must be column,key,value" in capsys.readouterr().err
    store.write_verdict(verdict(citation_key="fixtureKey2020"), env["store"])
    clean = env["tmp"] / "clean"
    write_store(clean, [fitness_record("s", "k", 1.0)])
    assert run(env, *_overlap_args(env, clean, good)) == 2
    assert "G4-key was not run" in capsys.readouterr().err


def test_ledger_print_and_append(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """--print renders; otherwise the section is appended to the note."""
    store.write_verdict(verdict(), env["store"])
    assert run(env, "ledger", "--date", "2026-10-10", "--print") == 0
    assert capsys.readouterr().out.startswith("## 2026.10.10 - Candidate gate ledger\n")
    note = env["tmp"] / "ledger.md"
    note.write_text("intro\n")
    assert run(env, "ledger", "--date", "2026-10-10", "--note", str(note)) == 0
    assert note.read_text().count("| keyPaper2020 | bacteria | Key 2020 |") == 1


def test_audit_states(env: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    """ok, drift, quote-missing, binary and absent, read against the mirrors."""
    library = env["root"] / "torchcell-library" / "keyPaper2020"
    library.mkdir(parents=True)
    text = "The study measured\nfifty strains in tri­plicate.".encode()
    (library / "paper.md").write_bytes(text)
    (library / "si.bin").write_bytes(b"\xff\xfe\x00")
    items = (
        evidence("fifty strains in triplicate.", digest=sha(text)),
        evidence("not in the text", digest=sha(text)),
        evidence("x", digest="0" * 64),
        evidence("x", uri="si.bin", digest=sha(b"\xff\xfe\x00")),
        evidence("x", uri="missing.md"),
    )
    gates_ = list(verdict().gates)
    gates_[0] = GateResult(gate="G1", outcome="fail", reason="r", evidence=items)
    store.write_verdict(verdict(gates=tuple(gates_)), env["store"])
    assert run(env, "audit", "--data-root", str(env["root"])) == 1
    assert capsys.readouterr().out == (
        f"keyPaper2020: quote-missing: {library / 'paper.md'}\n"
        f"keyPaper2020: drift: {library / 'paper.md'}\n"
        "keyPaper2020: absent: keyPaper2020/missing.md\n"
        "absent 1, binary 1, drift 1, ok 1, quote-missing 1\n"
    )


def test_audit_clean_and_status(
    env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """No evidence audits clean; status counts outcomes."""
    store.write_verdict(admissible_verdict(), env["store"])
    store.write_verdict(verdict(citation_key="other"), env["store"])
    assert run(env, "audit", "--data-root", str(env["root"])) == 0
    assert capsys.readouterr().out == "no evidence\n"
    assert run(env, "status") == 0
    assert capsys.readouterr().out == (
        f"2 verdict(s) in {env['store']}\n  admissible: 1\n  refused: 1\n"
    )


def test_data_root_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """DATA_ROOT from the environment; unset (and no .env) is a usage error."""
    monkeypatch.setattr(cli, "load_dotenv", lambda path: False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert cli.data_root(None) == str(tmp_path)
    assert cli.data_root("/given") == "/given"
    monkeypatch.delenv("DATA_ROOT")
    assert cli.main(["inventory", "--citation-key", "k"]) == 2
    assert (
        capsys.readouterr().err
        == "candidate-gate: DATA_ROOT is not set; pass --data-root\n"
    )


def test_load_doi_to_pmid(tmp_path: Path) -> None:
    """The cache is lowercased by DOI; an absent cache is None."""
    path = tmp_path / "doi2pmid.json"
    assert cli.load_doi_to_pmid(path) is None
    path.write_text(json.dumps({"10.1000/ABC": 123}))
    assert cli.load_doi_to_pmid(path) == {"10.1000/abc": "123"}


def test_real_table_scripts_load() -> None:
    """Both committed table scripts import and project onto CandidateRow."""
    bacteria = cli.table_rows("bacteria", cli.load_table_module("bacteria"))
    yeast = cli.table_rows("yeast", cli.load_table_module("yeast"))
    assert gates.find_row(bacteria, "CeCaFDB").name == "CeCaFDB flux compendium"
    assert {r.table for r in bacteria} == {"bacteria"}
    assert {r.organism for r in yeast} == {"S. cerevisiae"}


def test_module_entry_point_exits_with_mains_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``python -m torchcell.candidates`` runs main and exits with its code."""
    monkeypatch.setattr(sys, "argv", ["candidate-gate", "schema"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("torchcell.candidates.__main__", run_name="__main__")
    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out)["title"] == "CandidateVerdict"

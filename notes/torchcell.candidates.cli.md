---
id: fx96w3j0rtlnbfn9gkvaz79
title: CLI
desc: ''
updated: 1791617801433
created: 1791617801433
---

## 2026.10.10 - candidate-gate

Entry points: `candidate-gate` (`pyproject.toml`) and `python -m torchcell.candidates`. `DATA_ROOT` comes from the environment or `.env` unless `--data-root` is passed. Exit codes: 0 success, 1 a refusal (a refused verdict, a refused overlap, a failed audit, invalid agent JSON), 2 a usage error.

| subcommand | what it does |
|---|---|
| `schema` | prints `CandidateVerdict.model_json_schema()` for an agent prompt |
| `validate <json> [--write]` | validates agent output; a refusal names the field (`refused: gates.4: Value error, G5: a 'gap' must name its issue`) |
| `gate --row NAME \| --citation-key K [--table yeast] [--phenotype-class C --compound X --issue N] [--write] [--json]` | G1, then G2, G3 and G4-key in order, stopping at the first fail or blocked; G5 when a class is named |
| `inventory --citation-key K` | G3's record of a deposit |
| `overlap --citation-key K --released CSV --store DIR --sample-path --key-path --value-path [--tolerance] [--write]` | G4-value against a dev store, folded into the stored verdict (G5 drops to unmeasured when G4 then fails) |
| `ledger [--date] [--print]` | appends the dated ledger section to [[torchcell.candidates.ledger]] |
| `audit` | re-reads every quoted evidence slice of every stored verdict (#758): sha256, then the quote with whitespace collapsed and soft hyphens dropped; a binary file is reported as sha256-only. `bash scripts/ops.sh candidates` wraps it; it is run by hand or under slurm, never from cron |
| `status` | counts stored verdicts by outcome |

`--row` takes an exact row name or a unique prefix (`--row CeCaFDB`). The citation key comes from the module whose own `DOI` or `PAPER_DOI` is the row's DOI, else a mirror manifest with that DOI, else `--citation-key`.

Smoke runs on GilaHyper at the commit of this section, with the mirror and genomes tier mounted (`PYTHONPATH=<worktree> python -m torchcell.candidates gate --row ...`):

- `--row "D2Cell 2026"`: key `liLeveragingLargeLanguage2024`; G1 fail (transcription, quoting the Qwen1.5-110B relation-extraction step); G2 to G5 unmeasured; outcome refused.
- `--row "Lim 2022 putidaPRECISE321"`: key `limMachinelearningPseudomonasPutida2022`; G1 pass as an aggregation (21 source projects, `re_measured`); G2 pass (KT2440 resolves and reads); G3 pass (4 files across library and raw, 38,838 worksheet rows counted); G4 pass (no PMID check, no dev store compared); G5 unmeasured; outcome pending.
- `--row "CeCaFDB"`: key `zhangCeCaFDBCuratedDatabase2015`; G1 pass as an aggregation (33 source studies, `derived_by_aggregator`); G2 unmeasured (an aggregation names no single host); G3 pass (34 files in the raw mirror); G4 pass; outcome pending. The plan expects CeCaFDB to be subsumed by its sources at G4; that needs the per-source comparison the re-audit runs, which this call did not.
